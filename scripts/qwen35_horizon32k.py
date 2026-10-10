"""Confirmatory long-horizon exposure curve, 2048 -> 32768 quantized decode steps.

The exposure analysis in the paper fixes total context at 2048 and varies the quantized
suffix, so every H predicts the same continuation. That design cannot reach 32K: the
held-out WikiText split contains no article longer than 16227 tokens (probe_32k.py).

So this run uses a second protocol. Each context is quantized from step 0, and a fixed
256-token continuation is scored at several checkpoints along the way; excess NLL is
always the quantized minus unquantized loss at the *same* position, so the drift of
natural perplexity with position cancels in the difference. Contexts are built by
concatenating held-out articles, article-disjoint across contexts.

The two protocols coincide at H=2048 (quantize from token 0, score the next 256), which
is the cross-check that licenses reading the rest of the curve: if this run reproduces
the published H=2048 excess, the extension to 32K is measuring the same thing.
"""
import argparse, copy, json, os, statistics as st, sys, time, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                    # noqa: E402
from data import get_filler_ids                                           # noqa: E402
from qwen35_range import calib_absmax                                     # noqa: E402
from qwen35_groupwise import calib_group_absmax, q_group                  # noqa: E402


# ---- quantizers: identical to scripts/qwen35_longdoc2.py, so numbers stay comparable.
# `sr` selects unbiased stochastic rounding, matching src/qwen35_common.fake_quant.
def _round(y, sr):
    return torch.floor(y + torch.rand_like(y)) if sr else torch.round(y)


def q_static(cache, absmax, qmax, sr=False):
    for i, t in get_rstates(cache):
        s = (absmax[i].to(t.device) / qmax).clamp_min(1e-12)
        code = torch.clamp(_round(t.float() / s, sr), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def q_dyn(cache, qmax, sr=False):
    for _, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        code = torch.clamp(_round(t.float() / s, sr), -(qmax + 1), qmax)
        t.copy_((code * s).to(t.dtype))


def q_group_r(cache, gabs, G, qmax, sr=False):
    """Group-wise, with the rounding rule exposed. Mirrors qwen35_groupwise.q_group."""
    for i, t in get_rstates(cache):
        sh = t.shape
        tg = t.float().reshape(*sh[:-1], sh[-1] // G, G)
        sc = (gabs[i].to(t.device) / qmax).clamp_min(1e-12).unsqueeze(-1)
        code = torch.clamp(_round(tg / sc, sr), -(qmax + 1), qmax)
        t.copy_((code * sc).reshape(sh).to(t.dtype))


def q_zero(cache):
    """Zero-state control: the no-context floor, measured under this protocol."""
    for _, t in get_rstates(cache):
        t.zero_()


def state_norm(cache):
    """Accumulate in float64.

    Under runaway dynamic scaling the state reaches ~1e20 per element, and a float32
    sum of squares over 262144 of them overflows to inf while the elements themselves
    are still finite -- an artefact of the measurement, not divergence of the model.
    """
    return st.mean([t.double().norm().item() for _, t in get_rstates(cache)])


def pool_cache_path(model_id):
    """Token ids depend on the tokenizer, so the cache is keyed by model."""
    return "results/_article_pool_%s.json" % model_id.replace("/", "_")


def article_pool(tok, model_id):
    """All WikiText-103 test articles, in stable title-hash order.

    The paper's calibration/test split is by paragraph half, which leaves only ~148K
    held-out tokens: four 32K contexts. Splitting by article hash instead frees the
    whole 297K corpus while making calibration and test disjoint by *article*, which
    is the stricter of the two protocols.

    Tokenizing the corpus takes ~30 minutes, so the result is cached: the shards of one
    job would otherwise each pay it, and they must all see identical contexts anyway.
    """
    import json as _json
    cache = pool_cache_path(model_id)
    if os.path.exists(cache):
        with open(cache) as f:
            cached = _json.load(f)
        print("  [cache] %d articles from %s" % (len(cached), cache), flush=True)
        return [(h, t, ids) for h, t, ids in cached]

    import re, hashlib
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    title_re = re.compile(r"^ = [^=].* = $")
    arts, cur, title = [], [], None
    for t in df["text"]:
        s = t.rstrip("\n")
        if title_re.match(s):
            if cur:
                arts.append((title, "".join(cur)))
            cur = [t]; title = s.strip().strip("=").strip()
        else:
            cur.append(t)
    if cur:
        arts.append((title, "".join(cur)))
    out = []
    for title, art in arts:
        h = hashlib.sha1((title or "").encode()).hexdigest()
        out.append((h, title, tok(art, add_special_tokens=False).input_ids))
    out.sort(key=lambda x: x[0])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "w") as f:          # write-then-rename, so a concurrent shard never
        _json.dump(out, f)             # reads a half-written cache
    os.replace(tmp, cache)
    print("  [cache] wrote %d articles to %s" % (len(out), cache), flush=True)
    return out


def pack_contexts(pool, n_docs, ctx_len, cont):
    """Concatenate articles into article-disjoint contexts of ctx_len+cont tokens.

    No article appears in two contexts, so the contexts are independent of each other
    in the same sense the paper's single-article documents are. They are not single
    articles: the corpus has none longer than 16227 tokens.
    """
    need = ctx_len + cont
    docs, cur, used = [], [], []
    for _, title, ids in pool:
        cur.extend(ids)
        used.append(title)
        if len(cur) >= need:
            docs.append((cur[:need], list(used)))
            cur, used = [], []
            if len(docs) >= n_docs:
                break
    if not docs:
        raise RuntimeError("could not pack even one context of %d tokens" % need)
    if len(docs) < n_docs:
        print("  WARNING: corpus packed only %d of %d requested contexts"
              % (len(docs), n_docs), flush=True)
    for d, t in docs:
        print("  ctx: %d articles, %d tokens (%s ...)" % (len(t), len(d), t[0][:40]), flush=True)
    return docs


@torch.no_grad()
def run_pass(model, doc, ckpts, cont, fn):
    """One quantized pass over the context, scoring a continuation at each checkpoint.

    Scoring runs on a clone so the main trajectory is never perturbed by the tokens
    it is scored on.
    """
    dev = model.device
    cache, last, out = None, None, {}
    for i, tok_id in enumerate(doc):
        o = model(torch.tensor([[tok_id]], device=dev), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        if fn:
            fn(cache)
        last = o.logits[:, -1].float()
        H = i + 1
        if H in ckpts:
            c = copy.deepcopy(cache)
            cur, nll = last, 0.0
            for gt in doc[H:H + cont]:
                nll -= torch.log_softmax(cur, -1)[0, gt].item()
                oo = model(torch.tensor([[gt]], device=dev), past_key_values=c, use_cache=True)
                c = oo.past_key_values
                if fn:
                    fn(c)
                cur = oo.logits[:, -1].float()
            out[H] = dict(nll=nll / cont, snorm=state_norm(cache))
            del c
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--ckpts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=8)
    ap.add_argument("--n_calib_articles", type=int, default=6)
    # Single-token decode leaves the GPU ~75% idle, so the job shards by document
    # across concurrent processes rather than running one long serial pass.
    ap.add_argument("--doc_from", type=int, default=0)
    ap.add_argument("--doc_to", type=int, default=None)
    ap.add_argument("--policies", nargs="+", default=["unquant", "dynamic4", "static4", "group32"])
    ap.add_argument("--out", default="results/horizon32k_08b.json")
    # how many contexts the corpus actually yields decides how the job is sharded,
    # so it is answerable without loading the model
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    ctx_len = max(a.ckpts)
    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        pool = article_pool(tok, a.model)
        packed = pack_contexts(pool[a.n_calib_articles:], 99, ctx_len, a.cont)
        print("packs %d contexts of %d tokens from %d articles"
              % (len(packed), ctx_len + a.cont, len(pool) - a.n_calib_articles))
        return

    tok, model = load(a.model)

    print("[pool] reading articles", flush=True)
    pool = article_pool(tok, a.model)
    calib_arts, test_arts = pool[:a.n_calib_articles], pool[a.n_calib_articles:]
    print("  %d articles: %d for calibration, %d for contexts"
          % (len(pool), len(calib_arts), len(test_arts)), flush=True)

    print("[calib] on held-out articles, disjoint from every context", flush=True)
    cflat = [t for _, _, ids in calib_arts for t in ids]
    calib_docs = [cflat[i * 2200:i * 2200 + 2100] for i in range(3)]
    if len(calib_docs[-1]) < 2100:
        raise RuntimeError("calibration articles too short: %d tokens" % len(cflat))
    absmax = calib_absmax(model, calib_docs)
    g32 = calib_group_absmax(model, calib_docs, 32)

    conds = {
        "unquant": None,
        "zero": q_zero,
        "dynamic4": lambda c: q_dyn(c, 7),
        "static4": lambda c: q_static(c, absmax, 7),
        "group32": lambda c: q_group(c, g32, 32, 7),
        "group32_int5": lambda c: q_group(c, g32, 32, 15),
        "group32_int8": lambda c: q_group(c, g32, 32, 127),
        "dynamic4_sr": lambda c: q_dyn(c, 7, sr=True),
        "static4_sr": lambda c: q_static(c, absmax, 7, sr=True),
        "group32_sr": lambda c: q_group_r(c, g32, 32, 7, sr=True),
    }
    conds = {k: conds[k] for k in a.policies}

    print("[docs] packing %d contexts of %d+%d tokens" % (a.n_docs, ctx_len, a.cont), flush=True)
    docs = pack_contexts(test_arts, a.n_docs, ctx_len, a.cont)
    # pack the full set first so every shard sees the same contexts, then take this slice
    lo, hi = a.doc_from, (a.doc_to if a.doc_to is not None else len(docs))
    docs = docs[lo:hi]
    a.n_docs = len(docs)
    a.doc_ids = list(range(lo, lo + len(docs)))
    print("[shard] documents %d..%d (%d of them)" % (lo, hi - 1, len(docs)), flush=True)

    res = {p: {H: [] for H in a.ckpts} for p in conds}
    norms = {p: {H: [] for H in a.ckpts} for p in conds}
    t0 = time.time()
    for di, (doc, titles) in enumerate(docs):
        for p, fn in conds.items():
            r = run_pass(model, doc, set(a.ckpts), a.cont, fn)
            for H, v in r.items():
                res[p][H].append(v["nll"])
                norms[p][H].append(v["snorm"])
            print("  doc %d/%d %-13s %s | %.1f min elapsed"
                  % (di + 1, a.n_docs, p,
                     " ".join("H%d=%.3f" % (H, r[H]["nll"]) for H in sorted(r)),
                     (time.time() - t0) / 60), flush=True)
        # save after every document: a 4-hour job should never lose everything to a crash
        summary = {}
        for p in conds:
            if p == "unquant":
                continue
            summary[p] = {}
            for H in a.ckpts:
                n = len(res[p][H])
                exc = [res[p][H][i] - res["unquant"][H][i] for i in range(n)]
                summary[p][H] = dict(excess=st.mean(exc) if exc else None, per_doc=exc,
                                     snorm=st.mean(norms[p][H]) if norms[p][H] else None,
                                     n=n)
        json.dump(dict(summary=summary, raw_nll=res, config=vars(a),
                       n_docs_done=di + 1, minutes=(time.time() - t0) / 60),
                  open(a.out, "w"), indent=2)

    print("\nexcess NLL vs quantized exposure:", flush=True)
    for p in summary:
        print("  %-13s %s" % (p, "  ".join("H%d=%.3f" % (H, summary[p][H]["excess"])
                                           for H in a.ckpts)), flush=True)
    print("[saved] %s | %.1f min" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
