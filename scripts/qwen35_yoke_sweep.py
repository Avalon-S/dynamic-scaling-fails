"""What is the residual that survives scale-yoking made of?

Yoking cuts the scale-state feedback but still re-quantizes the state at every step, so
the excess NLL it leaves behind has two candidate explanations. The paper reads it as a
resolution limit; it could equally be accumulated rounding error, and a bit-width sweep
under yoking separates the two. Two probes, sharing the expensive lockstep machinery:

  bit-width      a resolution limit falls with the step size, roughly as 2^-b;
                 an accumulation floor does not.
  stochastic     SR removes the *bias* in the rounding error while leaving its
  rounding       magnitude alone. If biased error accumulating through the recurrence
                 is what remains, SR at a fixed bit-width should cut it; if the problem
                 is that the grid is too coarse, SR changes little.

Config matches results/yoke_08b.json (16 held-out articles, CTX=2048, H=2048, 256-token
continuation), so the INT4 arms reproduce the published 5.39 / 3.57 as a cross-check.
"""
import argparse, copy, json, os, statistics as st, sys, time, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_article_splits                                        # noqa: E402


def _round(y, stochastic):
    """RTNE, or unbiased stochastic rounding as in src/qwen35_common.fake_quant."""
    return torch.floor(y + torch.rand_like(y)) if stochastic else torch.round(y)


def q_self(cache, qmax, stochastic=False):
    """Scale from the quantized state: the closed loop."""
    for _, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        code = torch.clamp(_round(t.float() / s, stochastic), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def q_with_scale(cache, scales, qmax, stochastic=False):
    """Scale supplied from the lockstep FP state: the feedback edge is cut."""
    for i, t in get_rstates(cache):
        s = scales[i].to(t.device)
        code = torch.clamp(_round(t.float() / s, stochastic), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def fp_scales(fp, qmax):
    return {i: (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
            for i, t in get_rstates(fp)}


def state_norm(cache):
    # float64: under runaway self-scaling a float32 sum of squares overflows to inf
    # while the elements are still finite
    return st.mean([t.double().norm().item() for _, t in get_rstates(cache)])


@torch.no_grad()
def run(model, doc, CTX, cont, mode):
    """mode: 'unquant' | ('self'|'yoke', bits, stochastic). Returns (mean NLL, norm)."""
    dev = model.device
    cache, fp, last = None, None, None
    yoked = mode != "unquant" and mode[0] == "yoke"
    if yoked:
        fp = None

    def step(tid):
        nonlocal cache, fp, last
        o = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        if mode != "unquant":
            kind, bits, sr = mode
            qmax = 2 ** (bits - 1) - 1
            if kind == "self":
                q_self(cache, qmax, sr)
            else:
                of = model(torch.tensor([[tid]], device=dev), past_key_values=fp, use_cache=True)
                fp = of.past_key_values
                q_with_scale(cache, fp_scales(fp, qmax), qmax, sr)
        last = o.logits[:, -1].float()

    for tid in doc[:CTX]:
        step(tid)
    nm = state_norm(cache)
    cur, nll = last, 0.0
    for gt in doc[CTX:CTX + cont]:
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        step(gt)
        cur = last
    return nll / cont, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_test", type=int, default=16)
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    ap.add_argument("--sr_bits", type=int, nargs="+", default=[4, 8],
                    help="bit-widths to additionally run with stochastic rounding")
    ap.add_argument("--doc_from", type=int, default=0)
    ap.add_argument("--doc_to", type=int, default=None)
    ap.add_argument("--out", default="results/yoke_sweep_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    tok, model = load(a.model)
    sp = get_article_splits(tok, a.CTX + a.cont, {"calib": 2, "test": a.n_test})
    docs = [x[:a.CTX + a.cont] for x in sp["test"]["ids"]]
    lo, hi = a.doc_from, (a.doc_to if a.doc_to is not None else len(docs))
    docs, a.doc_ids = docs[lo:hi], list(range(lo, min(hi, len(docs))))
    print("[shard] documents %d..%d of %d" % (lo, hi - 1, a.n_test), flush=True)

    modes = {"unquant": "unquant", "self4": ("self", 4, False), "self4_sr": ("self", 4, True)}
    for b in a.bits:
        modes["yoke%d" % b] = ("yoke", b, False)
    for b in a.sr_bits:
        modes["yoke%d_sr" % b] = ("yoke", b, True)

    res = {m: [] for m in modes}
    norms = {m: [] for m in modes}
    t0 = time.time()
    for di, doc in enumerate(docs):
        for m, spec in modes.items():
            nll, nm = run(model, doc, a.CTX, a.cont, spec)
            res[m].append(nll); norms[m].append(nm)
        print("  doc %d/%d | %s | %.1f min"
              % (di + 1, len(docs),
                 " ".join("%s=%.3f" % (m, res[m][-1]) for m in modes),
                 (time.time() - t0) / 60), flush=True)
        summary = {}
        for m in modes:
            if m == "unquant":
                continue
            exc = [res[m][i] - res["unquant"][i] for i in range(len(res[m]))]
            summary[m] = dict(excess=st.mean(exc), per_doc=exc, n=len(exc),
                              norm=st.mean(norms[m]))
        json.dump(dict(summary=summary, raw=res, norms=norms, config=vars(a),
                       n_docs_done=di + 1, minutes=(time.time() - t0) / 60),
                  open(a.out, "w"), indent=2)
    print("[saved] %s | %.1f min" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
