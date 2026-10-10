"""Scale-yoking causal experiment: is the scale-state feedback CAUSAL for the dynamic runaway?

Per-token dynamic scaling sets the quantizer step from the CURRENT (quantized) state:
s_t = max|S_t^Q|/qmax. This closes a loop (quantized state -> scale -> next residual). We cut that loop
by YOKING the scale to the FP trajectory: quantize the state each step, but with s_t = max|S_t^FP|/qmax
(the scale the *unquantized* state would have produced). Run FP and the yoked-quant model in LOCKSTEP on
the same tokens, so the FP scale is available each step with no storage.

Conditions (matched-target continuation, excess NLL vs FP + state-norm trajectory):
  dyn_self  : standard dynamic INT4 (scale from the quantized state)         -- full closed loop
  dyn_yoked : dynamic INT4 with scale from the lockstep FP state             -- feedback edge CUT
If dyn_yoked removes the norm runaway that dyn_self shows -> the quantized-state->scale feedback causes the
numerical explosion. Whether it also restores excess NLL tests numerical-vs-functional separability.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_article_splits                                        # noqa: E402

QMAX = 7  # INT4


def q_dyn_self(cache):
    for i, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / QMAX).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(QMAX + 1), QMAX) * s).to(t.dtype))


def q_with_scale(cache, scales):
    """quantize each layer's state with a provided per-layer scale (from the FP lockstep)."""
    for i, t in get_rstates(cache):
        s = scales[i].to(t.device)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(QMAX + 1), QMAX) * s).to(t.dtype))


def norm(cache):
    return st.mean([t.float().norm().item() for _, t in get_rstates(cache)])


@torch.no_grad()
def run(model, doc, CTX, H, cont, mode):
    """mode in {unquant, dyn_self, dyn_yoked}. Returns (excess-ready nll, final state norm)."""
    dev = model.device; warm = CTX - H
    # shared unquant warm-up
    cw = None
    for k in range(warm):
        cw = model(torch.tensor([[doc[k]]], device=dev), past_key_values=cw, use_cache=True).past_key_values
    # for yoked mode we also run a parallel FP state to source the scale
    cache = copy.deepcopy(cw)
    fp = copy.deepcopy(cw) if mode == "dyn_yoked" else None
    last = None
    for tid in doc[warm:CTX]:
        o = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        if mode == "dyn_self":
            q_dyn_self(cache)
        elif mode == "dyn_yoked":
            of = model(torch.tensor([[tid]], device=dev), past_key_values=fp, use_cache=True)
            fp = of.past_key_values
            scales = {i: (t.float().abs().amax(-1, keepdim=True) / QMAX).clamp_min(1e-12)
                      for i, t in get_rstates(fp)}
            q_with_scale(cache, scales)
        last = o.logits[:, -1].float()
    nm = norm(cache)
    cur = last; nll = 0.0
    for gt in doc[CTX:CTX + cont]:
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        o = model(torch.tensor([[gt]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        if mode == "dyn_self":
            q_dyn_self(cache)
        elif mode == "dyn_yoked":
            of = model(torch.tensor([[gt]], device=dev), past_key_values=fp, use_cache=True)
            fp = of.past_key_values
            scales = {i: (t.float().abs().amax(-1, keepdim=True) / QMAX).clamp_min(1e-12)
                      for i, t in get_rstates(fp)}
            q_with_scale(cache, scales)
        cur = o.logits[:, -1].float()
    return nll / cont, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--n_test", type=int, default=16)
    ap.add_argument("--out", default="results/yoke_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)
    sp = get_article_splits(tok, a.CTX + a.cont, {"calib": 2, "test": a.n_test})
    docs = [x[:a.CTX + a.cont] for x in sp["test"]["ids"]]
    modes = ["unquant", "dyn_self", "dyn_yoked"]
    res = {m: {"nll": [], "norm": []} for m in modes}
    for di, doc in enumerate(docs):
        for m in modes:
            nll, nm = run(model, doc, a.CTX, a.H, a.cont, m)
            res[m]["nll"].append(nll); res[m]["norm"].append(nm)
        print(f"  doc {di+1}/{len(docs)} | "
              + " ".join(f"{m}:norm={res[m]['norm'][-1]:.3g}" for m in modes), flush=True)
    summary = {}
    uf = res["unquant"]["nll"]
    for m in modes:
        exc = [res[m]["nll"][i] - uf[i] for i in range(len(docs))]
        summary[m] = dict(excess=st.mean(exc), norm=st.mean(res[m]["norm"]), exc=exc,
                          norms=res[m]["norm"])
    print("SUMMARY:", {m: f"excess={summary[m]['excess']:.3f} norm={summary[m]['norm']:.3g}" for m in modes})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, test_hashes=sp["test"]["hashes"], config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
