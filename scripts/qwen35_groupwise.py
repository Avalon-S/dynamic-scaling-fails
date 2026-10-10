"""Group-wise quantization against group size.

Persistent group-wise INT4 with a calibrated scale per group of G channels along d_v,
for G in {128 (which is row-wise), 64, 32, 16}. A smaller G means more scales and a
finer grid, and also more scale metadata to carry. Charged honestly, with FP32
scales, the rate per element is:
  payload 4 + 32*(128/G)/128 = 4 + 32/G b/e → G128=4.25 / G64=4.50 / G32=5.00 / G16=6.00
  rowwise static5 = 5 + 32/128 = 5.25 b/e
The comparison that matters is G=32 at 5.00 bits against row-wise static INT5 at
5.25: the question is whether the finer grid is worth its metadata at a rate no
higher. Per-seed losses and per-trial hits are both stored, so the recall difference
can be tested paired.
"""
import argparse, json, os, sys, random, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_filler_ids                                            # noqa: E402
from niah import NEEDLE_TMPL, SUFFIX_TMPL, sample_spec, score              # noqa: E402
from qwen35_range import sseed                                             # noqa: E402


@torch.no_grad()
def calib_group_absmax(model, ids_list, G):
    """Calibrated absmax per (head, d_k, group). Returns {layer: (1,16,128,128//G)}."""
    SNAP = {128, 512, 1024, 1536, 2000}
    acc = {}
    for ids in ids_list:
        cache = None
        for t in range(min(2048, len(ids))):
            cache = model(torch.tensor([[ids[t]]], device=model.device),
                          past_key_values=cache, use_cache=True).past_key_values
            if t in SNAP:
                for li, tt in get_rstates(cache):
                    x = tt[0].float().reshape(16, 128, 128 // G, G)
                    a = x.abs().amax(-1)                         # (16,128,128//G)
                    acc[li] = torch.maximum(acc[li], a) if li in acc else a
    return {li: a.unsqueeze(0).clamp_min(1e-12) for li, a in acc.items()}


def q_group(cache, gabs, G, qmax):
    for i, t in get_rstates(cache):
        sh = t.shape                                            # (1,16,128,128)
        tg = t.float().reshape(*sh[:-1], sh[-1] // G, G)        # (1,16,128,ng,G)
        sc = (gabs[i].to(t.device) / qmax).clamp_min(1e-12).unsqueeze(-1)   # (1,16,128,ng,1)
        code = torch.clamp(torch.round(tg / sc), -(qmax + 1), qmax)
        t.copy_((code * sc).reshape(sh).to(t.dtype))


@torch.no_grad()
def eval_cond(model, tok, filler, H, seeds, quant_fn):
    nlls, hits = [], []
    for s in range(seeds):
        rng = random.Random(sseed("gw", s))
        word, code = sample_spec(rng)
        nid = tok(NEEDLE_TMPL.format(word=word, code=code), add_special_tokens=False).input_ids
        qid = tok(SUFFIX_TMPL.format(word=word), add_special_tokens=False).input_ids
        gold = tok(" " + str(code), add_special_tokens=False).input_ids
        off = rng.randint(0, len(filler) - H - 200); seg = filler[off:off + H]
        dev = model.device
        cache = model(torch.tensor([nid], device=dev), use_cache=True).past_key_values
        if quant_fn: quant_fn(cache)
        for k in range(H):
            cache = model(torch.tensor([[seg[k]]], device=dev),
                          past_key_values=cache, use_cache=True).past_key_values
            if quant_fn: quant_fn(cache)
        out = model(torch.tensor([qid], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if quant_fn: quant_fn(cache)
        cur = out.logits[:, -1].float(); nll = 0.0; am = True
        # greedy recall, scored independently of teacher forcing: record the argmax chain
        gen = []
        for gt in gold:
            nll -= torch.log_softmax(cur, -1)[0, gt].item()
            if int(cur.argmax(-1)) != gt: am = False
            o = model(torch.tensor([[gt]], device=dev), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            if quant_fn: quant_fn(cache)
            cur = o.logits[:, -1].float()
        nlls.append(nll / len(gold)); hits.append(bool(am))
    return sum(hits) / seeds, sum(nlls) / len(nlls), nlls, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="results/groupwise_08b.json")
    a = ap.parse_args()
    tok, model = load(QWEN35_08B)
    cfull = get_filler_ids(tok, 10000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    test_filler = get_filler_ids(tok, a.H + 4000, part="test")
    print(f"[calib] groupwise absmax | H={a.H} seeds={a.seeds}", flush=True)

    # one calibration per group size
    gabs = {G: calib_group_absmax(model, calib_docs, G) for G in (128, 64, 32, 16)}

    def rate(bits, G):                                          # bits per element, metadata included
        return bits + 32.0 * (128 // G) / 128

    conds = [
        ("static4_G128", lambda c: q_group(c, gabs[128], 128, 7), rate(4, 128)),
        ("static5_G128", lambda c: q_group(c, gabs[128], 128, 15), rate(5, 128)),
        ("group4_G64", lambda c: q_group(c, gabs[64], 64, 7), rate(4, 64)),
        ("group4_G32", lambda c: q_group(c, gabs[32], 32, 7), rate(4, 32)),
        ("group4_G16", lambda c: q_group(c, gabs[16], 16, 7), rate(4, 16)),
    ]
    summary = {}
    for name, fn, br in conds:
        tfr, nll, per, hits = eval_cond(model, tok, test_filler, a.H, a.seeds, fn)
        summary[name] = dict(tf_recall=tfr, nll=nll, bits=br, per_seed_nll=per, per_seed_hit=hits)
        print(f"  {name:13s} tf_recall={tfr:.3f} nll={nll:.3f} bits={br:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
