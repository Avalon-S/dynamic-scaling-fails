"""Capped dynamic scaling and dynamic/static schedules, at full exposure.

Conditions: unquantized, per-token dynamic INT4, persistent static INT4, persistent
group-wise INT4, capped dynamic (s = min(max|S|/qmax, c) with c calibrated offline),
capped plus exact preservation of the overflowing elements, and both orders of a
dynamic/static schedule with the switch at 512 and 1024 steps.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_filler_ids                                            # noqa: E402
from qwen35_range import calib_absmax                                      # noqa: E402
from qwen35_groupwise import calib_group_absmax, q_group                   # noqa: E402


def _dyn(t, qmax):
    return (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)


def q_dyn(cache, qmax):
    for i, t in get_rstates(cache):
        s = _dyn(t, qmax)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_static(cache, absmax, qmax):
    for i, t in get_rstates(cache):
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_capped(cache, absmax, qmax, escape=False):
    for i, t in get_rstates(cache):
        cap = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)     # healthy τ
        s = torch.minimum(_dyn(t, qmax), cap)                      # capped-dynamic
        code = torch.round(t.float() / s)
        if escape:
            over = (code < -(qmax + 1)) | (code > qmax)
            deq = torch.clamp(code, -(qmax + 1), qmax) * s
            t.copy_(torch.where(over, t.float(), deq).to(t.dtype))
        else:
            t.copy_((torch.clamp(code, -(qmax + 1), qmax) * s).to(t.dtype))


def make_fn(name, absmax, g32):
    if name == "unquant": return None
    if name == "dynamic4": return lambda c, k: q_dyn(c, 7)
    if name == "static4": return lambda c, k: q_static(c, absmax, 7)
    if name == "group32": return lambda c, k: q_group(c, g32, 32, 7)
    if name == "capped": return lambda c, k: q_capped(c, absmax, 7, escape=False)
    if name == "capped_esc": return lambda c, k: q_capped(c, absmax, 7, escape=True)
    if name.startswith("hybDS"):   # dynamic for the first K steps, then static
        K = int(name[5:]); return lambda c, k: (q_dyn(c, 7) if k < K else q_static(c, absmax, 7))
    if name.startswith("hybSD"):   # static for the first K steps, then dynamic
        K = int(name[5:]); return lambda c, k: (q_static(c, absmax, 7) if k < K else q_dyn(c, 7))
    raise ValueError(name)


def all_layer_norm(cache):
    return st.mean([t.float().norm().item() for _, t in get_rstates(cache)])


@torch.no_grad()
def run(model, doc, CTX, cont, fn):
    dev = model.device
    cache = None; last = None
    for k in range(CTX):
        out = model(torch.tensor([[doc[k]]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if fn: fn(cache, k)
        last = out.logits[:, -1].float()
    nm = all_layer_norm(cache)
    # score the continuation, still quantizing, with the step counter at CTX
    c = copy.deepcopy(cache); cur = last; nll = 0.0
    for j, gt in enumerate(doc[CTX:CTX + cont]):
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        o = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
        c = o.past_key_values
        if fn: fn(c, CTX + j)
        cur = o.logits[:, -1].float()
    return nll / cont, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5000)
    ap.add_argument("--out", default="results/hybrid_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    absmax = calib_absmax(model, calib_docs)
    g32 = calib_group_absmax(model, calib_docs, 32)
    test = get_filler_ids(tok, a.n_docs * a.stride + a.CTX + a.cont + 100, part="test")
    docs = [test[i * a.stride:i * a.stride + a.CTX + a.cont] for i in range(a.n_docs)]
    conds = ["unquant", "dynamic4", "static4", "group32", "capped", "capped_esc",
             "hybDS512", "hybDS1024", "hybSD512", "hybSD1024"]
    print(f"[calib] done | conds={conds}", flush=True)
    res = {c: dict(nll=[], norm=[]) for c in conds}
    for di, doc in enumerate(docs):
        for c in conds:
            nll, nm = run(model, doc, a.CTX, a.cont, make_fn(c, absmax, g32))
            res[c]["nll"].append(nll); res[c]["norm"].append(nm)
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)
    summary = {}
    uf = res["unquant"]["nll"]
    for c in conds:
        exc = [res[c]["nll"][i] - uf[i] for i in range(a.n_docs)]
        summary[c] = dict(excess=st.mean(exc), norm=st.mean(res[c]["norm"]), exc=exc)
    for c in conds:
        print(f"  {c:11s} excess={summary[c]['excess']:.3f} norm={summary[c]['norm']:.2e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
