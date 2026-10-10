"""Natural-text continuation with a matched target.

Total context is fixed and only the quantized suffix H varies, so every H scores the same
continuation and excess NLL is comparable across exposures. Articles are held out and
disjoint from calibration by title hash.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_filler_ids                                            # noqa: E402
from qwen35_range import calib_absmax                                      # noqa: E402
from qwen35_groupwise import calib_group_absmax, q_group                   # noqa: E402


def q_static(cache, absmax, qmax, tel=None):
    for i, t in get_rstates(cache):
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        raw = torch.round(t.float() / s)
        if tel is not None:
            tel["clip"].append(((raw < -(qmax + 1)) | (raw > qmax)).float().mean().item())
        code = torch.clamp(raw, -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def q_dyn(cache, qmax, tel=None):
    for i, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        code = torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def q_zero(cache, tel=None):
    for _, t in get_rstates(cache):
        t.zero_()


def state_norm(cache):
    return st.mean([t.float().norm().item() for _, t in get_rstates(cache)])


@torch.no_grad()
def run_cond(model, warm_cache, ctx_tail, cont_ids, fn, tel):
    """Clone the unquantized warm-up cache, quantize the last H steps of context, then
    score the continuation both ways: still quantizing, and with quantization off."""
    dev = model.device
    cache = copy.deepcopy(warm_cache)
    last_logits = None
    for tok_id in ctx_tail:                    # the last H steps, quantized
        out = model(torch.tensor([[tok_id]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if fn: fn(cache, tel)
        last_logits = out.logits[:, -1].float()
    snorm = state_norm(cache)
    # two readings of the continuation: still quantizing, and history damage alone
    def score(quantize):
        c = copy.deepcopy(cache); cur = last_logits; nll = 0.0
        for gt in cont_ids:
            nll -= torch.log_softmax(cur, -1)[0, gt].item()
            o = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
            c = o.past_key_values
            if quantize and fn: fn(c, None)
            cur = o.logits[:, -1].float()
        return nll / len(cont_ids)
    return score(True), score(False), snorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--Hs", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5000)      # keep the windows far apart
    ap.add_argument("--independent", action="store_true")    # one article per sample
    ap.add_argument("--out", default="results/longdoc2_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)              # the warm-up cache must be a leaf to deepcopy
    tok, model = load(a.model)
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    absmax = calib_absmax(model, calib_docs)
    g32 = calib_group_absmax(model, calib_docs, 32)
    need = a.n_docs * a.stride + a.CTX + a.cont + 100
    if getattr(a, "independent", False):
        from data import get_articles
        docs = [art[:a.CTX + a.cont] for art in
                get_articles(tok, a.n_docs, a.CTX + a.cont, part="test")]
        print(f"[docs] {len(docs)} independent articles", flush=True)
    else:
        test = get_filler_ids(tok, need, part="test")
        docs = [test[i * a.stride:i * a.stride + a.CTX + a.cont] for i in range(a.n_docs)]
    print(f"[calib] done | {a.model} Hs={a.Hs} CTX={a.CTX} n_docs={a.n_docs}", flush=True)

    conds = {
        "unquant": None,
        "zero": q_zero,
        "dynamic4": lambda c, t: q_dyn(c, 7, t),
        "static4": lambda c, t: q_static(c, absmax, 7, t),
        "static5": lambda c, t: q_static(c, absmax, 15, t),
        "dynamic5": lambda c, t: q_dyn(c, 15, t),
        "group32": lambda c, t: q_group(c, g32, 32, 7),
    }
    # res[H][cond] = dict(full=[per-doc], hist=[per-doc], snorm=[])
    res = {H: {c: dict(full=[], hist=[], snorm=[], clip=[]) for c in conds} for H in a.Hs}
    for di, doc in enumerate(docs):
        cont = doc[a.CTX:a.CTX + a.cont]
        for H in a.Hs:
            warm = a.CTX - H
            c0 = None
            for k in range(warm):              # unquantized warm-up, shared by every arm
                c0 = model(torch.tensor([[doc[k]]], device=model.device),
                           past_key_values=c0, use_cache=True).past_key_values
            ctx_tail = doc[warm:a.CTX]
            for c, fn in conds.items():
                tel = {"clip": []}
                full, hist, sn = run_cond(model, c0, ctx_tail, cont, fn, tel)
                res[H][c]["full"].append(full); res[H][c]["hist"].append(hist)
                res[H][c]["snorm"].append(sn)
                res[H][c]["clip"].append(st.mean(tel["clip"]) if tel["clip"] else 0.0)
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)

    summary = {}
    for H in a.Hs:
        summary[H] = {}
        for c in conds:
            uf = res[H]["unquant"]["full"]
            exc = [res[H][c]["full"][i] - uf[i] for i in range(a.n_docs)]
            exc_h = [res[H][c]["hist"][i] - res[H]["unquant"]["hist"][i] for i in range(a.n_docs)]
            summary[H][c] = dict(excess_full=st.mean(exc), excess_hist=st.mean(exc_h),
                                 snorm=st.mean(res[H][c]["snorm"]),
                                 clip=st.mean(res[H][c]["clip"]),
                                 excess=exc, excess_h=exc_h)
        print(f"  H={H} excess(full): "
              + " ".join(f"{c}={summary[H][c]['excess_full']:.3f}" for c in conds), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
