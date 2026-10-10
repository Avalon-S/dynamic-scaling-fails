"""Periodic reset of the recurrent state.

Zeroing the state every K steps interrupts the feedback in time rather than bounding it in
value. K is swept over 128, 256 and 512 against the dynamic, static and group-wise
baselines.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_filler_ids                                            # noqa: E402
from qwen35_range import calib_absmax                                      # noqa: E402
from qwen35_groupwise import calib_group_absmax, q_group                   # noqa: E402


def q_dyn(cache, qmax=7):
    for i, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_static(cache, absmax, qmax=7):
    for i, t in get_rstates(cache):
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_zero(cache):
    for _, t in get_rstates(cache):
        t.zero_()


def make_fn(name, absmax, g32):
    if name == "unquant": return None
    if name == "dynamic4": return lambda c, k: q_dyn(c)
    if name == "static4": return lambda c, k: q_static(c, absmax)
    if name == "group32": return lambda c, k: q_group(c, g32, 32, 7)
    if name.startswith("dynR"):
        K = int(name[4:])
        def fn(c, k):
            q_dyn(c)
            if (k + 1) % K == 0: q_zero(c)      # zero every K steps, interrupting the loop
        return fn
    raise ValueError(name)


@torch.no_grad()
def run(model, doc, CTX, cont, fn):
    dev = model.device; cache = None; last = None
    for k in range(CTX):
        out = model(torch.tensor([[doc[k]]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if fn: fn(cache, k)
        last = out.logits[:, -1].float()
    cur = last; nlls = []
    c = copy.deepcopy(cache)
    for j, gt in enumerate(doc[CTX:CTX + cont]):
        nlls.append(-torch.log_softmax(cur, -1)[0, gt].item())
        o = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
        c = o.past_key_values
        if fn: fn(c, CTX + j)
        cur = o.logits[:, -1].float()
    return st.mean(nlls), st.mean(nlls[:32])       # full-256, first-32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5000)
    ap.add_argument("--out", default="results/reset_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(QWEN35_08B)
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    absmax = calib_absmax(model, calib_docs)
    g32 = calib_group_absmax(model, calib_docs, 32)
    test = get_filler_ids(tok, a.n_docs * a.stride + a.CTX + a.cont + 100, part="test")
    docs = [test[i * a.stride:i * a.stride + a.CTX + a.cont] for i in range(a.n_docs)]
    conds = ["unquant", "static4", "group32", "dynamic4", "dynR128", "dynR256", "dynR512"]
    print(f"[calib] done | conds={conds}", flush=True)
    res = {c: dict(full=[], f32=[]) for c in conds}
    for di, doc in enumerate(docs):
        for c in conds:
            full, f32 = run(model, doc, a.CTX, a.cont, make_fn(c, absmax, g32))
            res[c]["full"].append(full); res[c]["f32"].append(f32)
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)
    summary = {}
    uf = res["unquant"]["full"]; uf32 = res["unquant"]["f32"]
    for c in conds:
        exc = [res[c]["full"][i] - uf[i] for i in range(a.n_docs)]
        exc32 = [res[c]["f32"][i] - uf32[i] for i in range(a.n_docs)]
        summary[c] = dict(excess=st.mean(exc), excess_f32=st.mean(exc32), exc=exc)
        print(f"  {c:9s} excess={summary[c]['excess']:.3f} first32={summary[c]['excess_f32']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
