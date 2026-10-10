"""Per-layer quantization sensitivity against decode horizon.

One recurrent layer is quantized at a time while the rest stay in full precision, which
locates where the damage concentrates. The ranking is recomputed at several horizons,
since a sensitivity order measured at one exposure need not hold at another.
"""
import argparse, copy, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B          # noqa: E402
from qwen35_range import calib_absmax                            # noqa: E402
from data import get_filler_ids                                  # noqa: E402


def q_static_layer(cache, absmax, qmax, L):
    """Static INT4 fake-quant applied to ONLY layer L's recurrent state."""
    for i, t in get_rstates(cache):
        if i != L:
            continue
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        code = torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


@torch.no_grad()
def run_cond(model, warm_cache, ctx_tail, cont_ids, fn):
    dev = model.device
    cache = copy.deepcopy(warm_cache)
    last_logits = None
    for tok_id in ctx_tail:
        out = model(torch.tensor([[tok_id]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if fn:
            fn(cache)
        last_logits = out.logits[:, -1].float()
    # full continuation NLL (keep quantizing during scored continuation = deployment)
    c = copy.deepcopy(cache); cur = last_logits; nll = 0.0
    for gt in cont_ids:
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        o = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
        c = o.past_key_values
        if fn:
            fn(c)
        cur = o.logits[:, -1].float()
    return nll / len(cont_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--Hs", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=12)
    ap.add_argument("--out", default="results/layersens_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)

    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    absmax = calib_absmax(model, calib_docs)

    from data import get_articles
    docs = [art[:a.CTX + a.cont] for art in
            get_articles(tok, a.n_docs, a.CTX + a.cont, part="test")]
    print(f"[docs] {len(docs)} independent articles", flush=True)

    # discover GDN layer indices from a short warm-up cache
    c0 = None
    for k in range(4):
        c0 = model(torch.tensor([[docs[0][k]]], device=model.device),
                   past_key_values=c0, use_cache=True).past_key_values
    layers = [i for i, _ in get_rstates(c0)]
    print(f"[layers] {len(layers)} GDN layers: {layers}", flush=True)

    # conditions: unquant baseline + one per layer (static INT4 leave-one-in)
    conds = {"unquant": None}
    for L in layers:
        conds[f"L{L}"] = (lambda LL: (lambda c: q_static_layer(c, absmax, 7, LL)))(L)

    res = {H: {c: [] for c in conds} for H in a.Hs}
    for di, doc in enumerate(docs):
        cont = doc[a.CTX:a.CTX + a.cont]
        for H in a.Hs:
            warm = a.CTX - H
            cw = None
            for k in range(warm):
                cw = model(torch.tensor([[doc[k]]], device=model.device),
                           past_key_values=cw, use_cache=True).past_key_values
            ctx_tail = doc[warm:a.CTX]
            for c, fn in conds.items():
                res[H][c].append(run_cond(model, cw, ctx_tail, cont, fn))
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)

    summary = {}
    for H in a.Hs:
        uf = res[H]["unquant"]
        summary[H] = {}
        for c in conds:
            if c == "unquant":
                continue
            exc = [res[H][c][i] - uf[i] for i in range(a.n_docs)]
            summary[H][c] = dict(excess_full=st.mean(exc), excess=exc)
        ranked = sorted(((summary[H][c]["excess_full"], c) for c in summary[H]), reverse=True)
        print(f"  H={H} layer sensitivity (excess NLL, desc): "
              + " ".join(f"{c}={v:.3f}" for v, c in ranked), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, layers=layers, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
