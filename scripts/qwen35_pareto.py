"""Rate-distortion frontier: asymmetric allocation against uniform group-wise.

Swept across bit rates so the range over which either dominates is explicit rather than
read off a single operating point.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from qwen35_groupwise import calib_group_absmax                            # noqa: E402
from data import get_filler_ids, get_articles                             # noqa: E402


def q_group_mixed(cache, gabs, G, bits):
    for i, t in get_rstates(cache):
        b = bits.get(i)
        if b is None:
            continue
        qmax = 2 ** (b - 1) - 1
        sh = t.shape
        x = t.float().reshape(*sh[:-1], sh[-1] // G, G)
        sc = (gabs[i].to(t.device) / qmax).unsqueeze(-1).clamp_min(1e-12)
        code = torch.clamp(torch.round(x / sc), -(qmax + 1), qmax)
        t.copy_((code * sc).reshape(sh).to(t.dtype))


def rate_of(bits, layers):
    return st.mean([bits[i] + 1.0 for i in layers])


@torch.no_grad()
def run_cond(model, warm_cache, ctx_tail, cont_ids, fn):
    dev = model.device
    cache = copy.deepcopy(warm_cache); last = None
    for tid in ctx_tail:
        out = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if fn: fn(cache)
        last = out.logits[:, -1].float()
    c = copy.deepcopy(cache); cur = last; nll = 0.0
    for gt in cont_ids:
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        o = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
        c = o.past_key_values
        if fn: fn(c)
        cur = o.logits[:, -1].float()
    return nll / len(cont_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--Hs", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=16)
    ap.add_argument("--sens", default="results/layersens_08b.json")
    ap.add_argument("--out", default="results/pareto_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    g32 = calib_group_absmax(model, calib, 32)
    layers = sorted(g32.keys())
    s = json.load(open(a.sens))["summary"]["2048"]
    sens = {int(c[1:]): s[c]["excess_full"] for c in s}
    ranked = sorted(layers, key=lambda i: sens.get(i, 0), reverse=True)

    plans = {}
    for b in (3, 4, 5, 8):                                   # uniform baselines
        plans[f"uniform_int{b}"] = {i: b for i in layers}
    for k in (1, 3):                                         # extreme: top-k INT8, rest r
        for r in (2, 3, 4):
            plans[f"ext_top{k}_i8_rest_i{r}"] = {i: (8 if i in ranked[:k] else r) for i in layers}

    docs = [art[:a.CTX + a.cont] for art in get_articles(tok, a.n_docs, a.CTX + a.cont, part="test")]
    for nm, pl in plans.items():
        print(f"[plan] {nm:22s} rate={rate_of(pl, layers):.2f}", flush=True)
    print(f"[calib] {a.model} | {len(docs)} independent", flush=True)

    conds = {"unquant": None}
    for nm, pl in plans.items():
        conds[nm] = (lambda P: (lambda c: q_group_mixed(c, g32, 32, P)))(pl)

    res = {H: {c: [] for c in conds} for H in a.Hs}
    for di, doc in enumerate(docs):
        cont = doc[a.CTX:a.CTX + a.cont]
        for H in a.Hs:
            warm = a.CTX - H; cw = None
            for k in range(warm):
                cw = model(torch.tensor([[doc[k]]], device=model.device),
                           past_key_values=cw, use_cache=True).past_key_values
            ctx_tail = doc[warm:a.CTX]
            for c, fn in conds.items():
                res[H][c].append(run_cond(model, cw, ctx_tail, cont, fn))
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)

    summary = {}
    for H in a.Hs:
        uf = res[H]["unquant"]; summary[H] = {}
        for c in conds:
            if c == "unquant":
                continue
            exc = [res[H][c][i] - uf[i] for i in range(a.n_docs)]
            summary[H][c] = dict(excess_full=st.mean(exc),
                                 rate=(rate_of(plans[c], layers) if c in plans else 0), excess=exc)
        pts = sorted((summary[H][c]['rate'], summary[H][c]['excess_full'], c) for c in summary[H])
        print(f"  H={H}: " + " ".join(f"{c}={e:.3f}@{r:.2f}" for r, e, c in pts), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, plans=plans, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
