"""Time to first-token mismatch under a common history.

Each policy and the full-precision reference consume the same real tokens, so any
divergence comes from state quantization alone. Reports the step at which the argmax first
differs, censored at the end of the article, for a range of confidence thresholds.
"""
import argparse, copy, json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from qwen35_range import calib_absmax                                      # noqa: E402
from qwen35_groupwise import calib_group_absmax                            # noqa: E402
from data import get_article_splits                                        # noqa: E402

G = 32


def q_dyn(cache, qmax, protect):
    for i, t in get_rstates(cache):
        if i in protect:
            continue
        s = (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_static(cache, absmax, qmax, protect):
    for i, t in get_rstates(cache):
        if i in protect:
            continue
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        t.copy_((torch.clamp(torch.round(t.float() / s), -(qmax + 1), qmax) * s).to(t.dtype))


def q_group(cache, gabs, qmax, protect):
    for i, t in get_rstates(cache):
        if i in protect:
            continue
        sh = t.shape
        x = t.float().reshape(*sh[:-1], sh[-1] // G, G)
        sc = (gabs[i].to(t.device) / qmax).unsqueeze(-1).clamp_min(1e-12)
        code = torch.clamp(torch.round(x / sc), -(qmax + 1), qmax)
        t.copy_((code * sc).reshape(sh).to(t.dtype))


MUS = [0.0, 1.0, 2.0, 4.0, 8.0]   # FP top-2 logit-gap thresholds ("confidence"): failure = confident FP flip


@torch.no_grad()
def fp_pass(model, toks):
    """Run FP once; per step store (argmax, top-2 logit margin). Naive argmax-flip is too sensitive on
    near-ties, so we gate failures on FP confidence (margin)."""
    dev = model.device; cache = None; am = []; marg = []
    for tid in toks:
        o = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        z = o.logits[:, -1].float()[0]
        top2 = torch.topk(z, 2).values
        am.append(int(z.argmax()))
        marg.append((top2[0] - top2[1]).item())
    return am, marg


@torch.no_grad()
def policy_tau(model, toks, fn, fp_am, fp_marg):
    """Quantized policy on same tokens. tau_mu = first step where quant argmax != FP argmax AND FP was
    confident (margin > mu). Returns {mu: (tau, censored)} + mean logp of FP token (graded margin)."""
    dev = model.device; cache = None; L = len(toks)
    tau = {mu: None for mu in MUS}; logps = []
    for t, tid in enumerate(toks):
        o = model(torch.tensor([[tid]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        if fn:
            fn(cache)
        z = o.logits[:, -1].float()
        mism = int(z.argmax(-1)) != fp_am[t]
        if mism:
            for mu in MUS:
                if tau[mu] is None and fp_marg[t] > mu:
                    tau[mu] = t
        logps.append(torch.log_softmax(z, -1)[0, fp_am[t]].item())
    out = {}
    for mu in MUS:
        cens = tau[mu] is None
        out[mu] = (L if cens else tau[mu], cens)
    return out, sum(logps) / len(logps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--Hmax", type=int, default=8192)
    ap.add_argument("--min_tokens", type=int, default=1024)
    ap.add_argument("--n_calib", type=int, default=6)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--out", default="results/survival_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok, model = load(a.model)
    sp = get_article_splits(tok, a.min_tokens, {"calib": a.n_calib, "test": a.n_test})
    calib = [x[:2100] for x in sp["calib"]["ids"]]
    absmax = calib_absmax(model, calib)
    g32 = calib_group_absmax(model, calib, G)
    # discover first-3 GDN layer indices
    c0 = None
    for k in range(4):
        c0 = model(torch.tensor([[sp["test"]["ids"][0][k]]], device=model.device),
                   past_key_values=c0, use_cache=True).past_key_values
    layers = [i for i, _ in get_rstates(c0)]
    first3 = set(layers[:3])
    print(f"[setup] {a.model} | {len(layers)} GDN layers | first3={sorted(first3)} | "
          f"test articles={len(sp['test']['ids'])}", flush=True)

    policies = {
        "dynamic4":       lambda c: q_dyn(c, 7, set()),
        "static4":        lambda c: q_static(c, absmax, 7, set()),
        "group32":        lambda c: q_group(c, g32, 7, set()),
        "group32_first3": lambda c: q_group(c, g32, 7, first3),
    }
    # res[policy][mu] = {"tau":[per-article], "censored":[bool]}; plus graded margin
    res = {p: {str(mu): {"tau": [], "censored": []} for mu in MUS} for p in policies}
    for p in policies:
        res[p]["margin"] = []
    lengths = []
    for di, art in enumerate(sp["test"]["ids"]):
        toks = art[:a.Hmax]
        lengths.append(len(toks))
        fp_am, fp_marg = fp_pass(model, toks)
        for p, fn in policies.items():
            taus, gmarg = policy_tau(model, toks, fn, fp_am, fp_marg)
            for mu in MUS:
                tt, cc = taus[mu]
                res[p][str(mu)]["tau"].append(tt); res[p][str(mu)]["censored"].append(cc)
            res[p]["margin"].append(gmarg)
        print(f"  article {di+1}/{len(sp['test']['ids'])} (len {len(toks)}) | "
              + " ".join(f"{p}:tau@2={res[p]['2.0']['tau'][-1]}{'*' if res[p]['2.0']['censored'][-1] else ''}"
                         for p in policies), flush=True)

    prov = dict(model=a.model, Hmax=a.Hmax, min_tokens=a.min_tokens, first3=sorted(first3), mus=MUS,
                test_hashes=sp["test"]["hashes"], calib_hashes=sp["calib"]["hashes"], lengths=lengths)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(results=res, provenance=prov), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
