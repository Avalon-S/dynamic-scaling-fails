"""Clamp range against quantizer resolution.

Spending extra bits on the widest channels changes two things at once: the step size and
the representable range. This separates them, so that targeting the most-overflowing
channels can be compared against a random subset at a matched rate, and against storing
the overflowing elements exactly.
"""
import argparse, hashlib, json, math, os, sys, random, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates, QWEN35_08B                     # noqa: E402
from data import get_filler_ids                                            # noqa: E402
from niah import NEEDLE_TMPL, SUFFIX_TMPL, sample_spec                      # noqa: E402


def sseed(*p):
    return int(hashlib.md5("|".join(map(str, p)).encode()).hexdigest()[:8], 16)


@torch.no_grad()
def calib_absmax(model, ids_list):
    SNAP = {128, 512, 1024, 1536, 2000}
    acc = {}
    for ids in ids_list:
        cache = None
        for t in range(min(2048, len(ids))):
            cache = model(torch.tensor([[ids[t]]], device=model.device),
                          past_key_values=cache, use_cache=True).past_key_values
            if t in SNAP:
                for li, tt in get_rstates(cache):
                    a = tt[0].float().abs().amax(-1)
                    acc[li] = torch.maximum(acc[li], a) if li in acc else a
    return {li: a.unsqueeze(0).unsqueeze(-1).clamp_min(1e-12) for li, a in acc.items()}


def qmax_of(b):
    return 2 ** (b - 1) - 1


def q_perchan(cache, step_map, cmax_map):
    """Quantize with a per-channel step and an independent clamp range.
    step_map and cmax_map are {layer: (1,16,128,1)}."""
    for i, t in get_rstates(cache):
        s = step_map[i].to(t.device); cm = cmax_map[i].to(t.device)
        code = torch.round(t.float() / s)
        code = torch.minimum(torch.maximum(code, -(cm + 1)), cm)
        t.copy_((code * s).to(t.dtype))


def q_dynamic4(cache):
    """Per-token dynamic INT4: the scale is the absmax of the state about to be quantized."""
    for i, t in get_rstates(cache):
        s = (t.float().abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12)
        code = torch.clamp(torch.round(t.float() / s), -8, 7)
        t.copy_((code * s).to(t.dtype))


def q_escape(cache, absmax, stat):
    """Element-wise sparse escape: INT4 everywhere, with the overflowing elements kept
    exactly. stat['over'] and stat['tot'] accumulate so the true bit rate can be charged."""
    for i, t in get_rstates(cache):
        s = (absmax[i].to(t.device) / 7).clamp_min(1e-12)
        code = torch.round(t.float() / s)
        over = (code < -8) | (code > 7)
        deq = torch.clamp(code, -8, 7) * s
        t.copy_(torch.where(over, t.float(), deq).to(t.dtype))
        stat['over'] += int(over.sum()); stat['tot'] += over.numel()


@torch.no_grad()
def select_exception_channels(model, absmax, ids_list):
    """Count per-channel overflows over a closed-loop static-INT4 calibration pass.
    Returns {layer: (16,128)}."""
    step4 = {li: absmax[li] / 7 for li in absmax}
    cmax4 = {li: torch.full_like(absmax[li], 7) for li in absmax}
    events = {li: torch.zeros(16, 128) for li in absmax}
    for ids in ids_list:
        cache = None
        for t in range(min(1024, len(ids))):
            cache = model(torch.tensor([[ids[t]]], device=model.device),
                          past_key_values=cache, use_cache=True).past_key_values
            for li, tt in get_rstates(cache):
                s = (absmax[li].to(tt.device) / 7).clamp_min(1e-12)
                code = torch.round(tt.float() / s)
                over = (code < -8) | (code > 7)                # -8 is a legal signed code
                events[li] += over.sum(-1)[0].float().cpu()
            q_perchan(cache, step4, cmax4)
    return events


def sel_mask(events, mode, K_frac, tag):
    """{layer: (16,128) bool} marking the selected channels. mode is 'top' or 'random'."""
    N = sum(e.numel() for e in events.values()); k = int(N * K_frac)
    if mode == "top":
        thresh = torch.topk(torch.cat([e.flatten() for e in events.values()]), k).values.min()
        return {li: (e >= thresh) for li, e in events.items()}
    g = torch.Generator().manual_seed(sseed("randch", K_frac, tag))
    pick = torch.zeros(N, dtype=torch.bool); pick[torch.randperm(N, generator=g)[:k]] = True
    out, off = {}, 0
    for li, e in events.items():
        n = e.numel(); out[li] = pick[off:off + n].reshape(e.shape); off += n
    return out


def build_maps(absmax, sel, hi_bits, refine):
    """Give the selected channels hi_bits and the rest 4. With refine the extra bits buy
    a finer step over the same range; otherwise they widen the range at the INT4 step."""
    step, cmax = {}, {}
    for li in absmax:
        b = torch.where(sel[li].to(absmax[li].device), hi_bits, 4).float()   # (16,128) on absmax dev
        qm = (2 ** (b - 1) - 1).unsqueeze(0).unsqueeze(-1)     # (1,16,128,1)
        cmax[li] = qm
        step[li] = absmax[li] / qm if refine else absmax[li] / 7
    return step, cmax


def payload_bits(sel, hi_bits):
    N = sum(s.numel() for s in sel.values()); nhi = sum(int(s.sum()) for s in sel.values())
    return (4 * (N - nhi) + hi_bits * nhi) / N


@torch.no_grad()
def eval_cond(model, tok, filler, H, seeds, quant_fn):
    """quant_fn(cache) quantizes in place at every step; None leaves the model alone.
    Returns (tf_recall, mean_nll, per_seed_nll)."""
    nlls, tf = [], 0
    for s in range(seeds):
        rng = random.Random(sseed("rangeeval", s))
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
        for gt in gold:
            nll -= torch.log_softmax(cur, -1)[0, gt].item()
            if int(cur.argmax(-1)) != gt: am = False
            o = model(torch.tensor([[gt]], device=dev), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            if quant_fn: quant_fn(cache)
            cur = o.logits[:, -1].float()
        nlls.append(nll / len(gold)); tf += am
    return tf / seeds, sum(nlls) / len(nlls), nlls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--hi_bits", type=int, default=8)
    ap.add_argument("--K", type=float, default=0.25)          # 25% hi @ INT8 = 5.0 bit = INT5 rate
    ap.add_argument("--n_rand", type=int, default=3)
    ap.add_argument("--out", default="results/range_08b.json")
    a = ap.parse_args()
    tok, model = load(QWEN35_08B)
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    test_filler = get_filler_ids(tok, a.H + 4000, part="test")
    absmax = calib_absmax(model, calib_docs)
    events = select_exception_channels(model, absmax, calib_docs)
    print(f"[calib] done | K={a.K} hi={a.hi_bits} rate={4+(a.hi_bits-4)*a.K:.2f}bit", flush=True)

    step4 = {li: absmax[li] / 7 for li in absmax}; cmax4 = {li: torch.full_like(absmax[li], 7) for li in absmax}
    step5 = {li: absmax[li] / 15 for li in absmax}; cmax5 = {li: torch.full_like(absmax[li], 15) for li in absmax}
    conds = [("fp16", None, 0.0),
             ("dynamic4", q_dynamic4, 4.0),
             ("static4", lambda c: q_perchan(c, step4, cmax4), 4.0),
             ("static5", lambda c: q_perchan(c, step5, cmax5), 5.0)]
    # refine and range, each against top-k and random channel selection
    pay = payload_bits(sel_mask(events, "top", a.K, 0), a.hi_bits)
    for refine in (True, False):
        tag = "refine" if refine else "range"
        st, cm = build_maps(absmax, sel_mask(events, "top", a.K, 0), a.hi_bits, refine)
        conds.append((f"{tag}_top", lambda c, st=st, cm=cm: q_perchan(c, st, cm), pay))
        for r in range(a.n_rand):
            st, cm = build_maps(absmax, sel_mask(events, "random", a.K, r), a.hi_bits, refine)
            conds.append((f"{tag}_rand{r}", lambda c, st=st, cm=cm: q_perchan(c, st, cm), pay))

    esc_stat = {'over': 0, 'tot': 0}
    conds.append(("escape_exact", lambda c: q_escape(c, absmax, esc_stat), None))

    summary = {}
    for name, fn, bits in conds:
        tfr, nll, per = eval_cond(model, tok, test_filler, a.H, a.seeds, fn)
        summary[name] = dict(tf_recall=tfr, nll=nll, bits=bits, per_seed_nll=per)
        print(f"  {name:14s} tf_recall={tfr:.3f} nll={nll:.3f} bits={bits}", flush=True)
    # what escape really costs: the overflow fraction plus its index and value overhead
    of = esc_stat['over'] / max(esc_stat['tot'], 1)
    esc_bits = 4 + of * (16 + math.log2(sum(s.numel() * 128 for s in absmax.values())))
    summary["escape_exact"]["bits"] = esc_bits
    summary["escape_exact"]["overflow_frac"] = of
    print(f"  [escape] overflow_frac={of:.4f} -> {esc_bits:.3f} bit", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
