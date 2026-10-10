"""Per-layer allocation under a leak-free protocol.

Calibration, selection and test articles are disjoint by title hash. Layer selection runs
under the deployment quantizer rather than a proxy, and every reported number is on the
held-out test split, so the advantage is not read off the articles the layers were chosen
on.
"""
import argparse, copy, json, os, sys, statistics as st, random, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from qwen35_groupwise import calib_group_absmax                            # noqa: E402
from data import get_article_splits                                        # noqa: E402

G = 32


def q_mixed(cache, gabs, bits):
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
    return st.mean([bits[i] + 1.0 for i in layers])       # equal-size Qwen GDN layers


@torch.no_grad()
def eval_plans_on_doc(model, doc, CTX, H, cont, gabs, plans):
    """Warm up ONCE, then for each plan (dict{layer:bits} or None) run the quantized tail + continuation
    from a fresh deepcopy. Returns {name: nll}. 'unquant' key (None plan) is the baseline."""
    dev = model.device
    warm = CTX - H
    cw = None
    for k in range(warm):
        cw = model(torch.tensor([[doc[k]]], device=dev), past_key_values=cw, use_cache=True).past_key_values
    tail = doc[warm:CTX]; cont_ids = doc[CTX:CTX + cont]
    out = {}
    for name, bits in plans.items():
        fn = (lambda c: q_mixed(c, gabs, bits)) if bits else None
        cache = copy.deepcopy(cw); last = None
        for tid in tail:
            o = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            if fn: fn(cache)
            last = o.logits[:, -1].float()
        cur = last; nll = 0.0
        for gt in cont_ids:
            nll -= torch.log_softmax(cur, -1)[0, gt].item()
            o = model(torch.tensor([[gt]], device=dev), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            if fn: fn(cache)
            cur = o.logits[:, -1].float()
        out[name] = nll / cont
    return out


def excess_over_docs(model, docs, CTX, H, cont, gabs, plans):
    """per-doc excess NLL (plan - unquant) for each plan; shared warm-up. Returns {name:[per-doc excess]}."""
    allp = dict(plans); allp["unquant"] = None
    acc = {n: [] for n in plans}
    for di, doc in enumerate(docs):
        r = eval_plans_on_doc(model, doc, CTX, H, cont, gabs, allp)
        for n in plans:
            acc[n].append(r[n] - r["unquant"])
        print(f"    doc {di+1}/{len(docs)} done", flush=True)
    return acc


def local_mse(model, gabs, calib_ids, layers):
    """per-layer INT2 reconstruction MSE on calib (a quantizer-only, non-functional signal control)."""
    dev = model.device
    acc = {i: 0.0 for i in layers}; n = 0
    with torch.no_grad():
        cache = None
        for k in range(min(2048, len(calib_ids))):
            cache = model(torch.tensor([[calib_ids[k]]], device=dev), past_key_values=cache, use_cache=True).past_key_values
            if k in (512, 1024, 1536, 2000):
                for i, t in get_rstates(cache):
                    x = t.float(); sh = x.shape
                    xg = x.reshape(*sh[:-1], sh[-1] // G, G)
                    sc = (gabs[i].to(dev) / 1).unsqueeze(-1).clamp_min(1e-12)     # INT2 qmax=1
                    q = torch.clamp(torch.round(xg / sc), -2, 1) * sc
                    acc[i] += (q.reshape(sh) - x).pow(2).mean().item()
                n += 1
    return {i: acc[i] / max(n, 1) for i in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--n_calib", type=int, default=6)
    ap.add_argument("--n_select", type=int, default=15)
    ap.add_argument("--n_test", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/clean_alloc_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)
    sp = get_article_splits(tok, a.CTX + a.cont, {"calib": a.n_calib, "select": a.n_select, "test": a.n_test})
    calib = [x[:a.CTX + a.cont] for x in sp["calib"]["ids"]]
    select = [x[:a.CTX + a.cont] for x in sp["select"]["ids"]]
    test = [x[:a.CTX + a.cont] for x in sp["test"]["ids"]]
    gabs = calib_group_absmax(model, [c[:a.CTX] for c in calib], G)
    layers = sorted(gabs.keys()); N = len(layers)
    print(f"[split] calib={len(calib)} select={len(select)} test={len(test)} | {N} layers", flush=True)

    # ---- SELECTION on select split, using the DEPLOYMENT quantizer (G32), marginal rescue ----
    print("[select] marginal rescue (all-INT2 baseline, per-layer INT2->INT8)...", flush=True)
    base = {i: 2 for i in layers}
    sel_plans = {"base_int2": base}
    for L in layers:
        pl = dict(base); pl[L] = 8
        sel_plans[f"resc_{L}"] = pl
    sel = excess_over_docs(model, select, a.CTX, a.H, a.cont, gabs, sel_plans)
    base_nll = st.mean(sel["base_int2"])
    rescue = {L: base_nll - st.mean(sel[f"resc_{L}"]) for L in layers}
    ranked = sorted(layers, key=lambda i: rescue[i], reverse=True)
    for L in ranked:
        print(f"  rescue L{L}: {rescue[L]:+.4f}", flush=True)

    k_for_4b = round((3.0 - 2.0) * N / 6.0)               # #layers 2->8 for avg bits 3.0 (rate 4.0)
    knap = {i: 2 for i in layers}
    for L in ranked[:k_for_4b]:
        knap[L] = 8

    # Draw each selection once, here. Sampling inside the per-layer comprehension
    # would redraw it for every layer and the arms would not be comparable.
    rng = random.Random(a.seed)
    l0 = ranked[:1]
    rand3a = rng.sample(layers, 3)
    rand3b = rng.sample(layers, 3)
    l0_2r = l0 + rng.sample([x for x in layers if x not in l0], 2)
    lmse = local_mse(model, gabs, calib[0], layers)
    mse_rank = sorted(layers, key=lambda i: lmse[i], reverse=True)

    def ext(sel):
        sel = set(sel)
        return {i: (8 if i in sel else 2) for i in layers}
    plans_test = {
        "uniform_int3":        {i: 3 for i in layers},          # 4.0b
        "uniform_int4":        {i: 4 for i in layers},          # 5.0b
        "extreme_rescue_top3": ext(ranked[:3]),                 # 4.0b recipe
        "knapsack":            knap,                            # ~4.0b fair optimizer
        "l0_plus_2rand":       ext(l0_2r),                      # 4.0b control
        "rand3_a":             ext(rand3a),                     # 4.0b control
        "rand3_b":             ext(rand3b),                     # 4.0b control
        "first3":              ext(layers[:3]),                 # 4.0b control
        "inverse_bottom3":     ext(ranked[-3:]),               # 4.0b control
        "local_mse_top3":      ext(mse_rank[:3]),               # 4.0b control (non-functional signal)
    }

    # ---- EVALUATE on fresh TEST split ----
    print("[test] evaluating frozen plans on FRESH test split...", flush=True)
    texc = excess_over_docs(model, test, a.CTX, a.H, a.cont, gabs, plans_test)
    res = {n: dict(excess_full=st.mean(v), rate=rate_of(plans_test[n], layers), excess=v)
           for n, v in texc.items()}
    for n in sorted(res, key=lambda x: res[x]["excess_full"]):
        print(f"  TEST {n:22s} rate={res[n]['rate']:.2f}b  excess={res[n]['excess_full']:.3f}", flush=True)

    prov = dict(model=a.model, G=G, H=a.H, CTX=a.CTX, cont=a.cont, dtype="bfloat16",
                calib_hashes=sp["calib"]["hashes"], select_hashes=sp["select"]["hashes"],
                test_hashes=sp["test"]["hashes"], select_titles=sp["select"]["titles"],
                test_titles=sp["test"]["titles"], rescue=rescue, ranked=ranked,
                k_for_4b=k_for_4b, plans={k: v for k, v in plans_test.items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(results=res, provenance=prov), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
