"""NIAH recall against quantized decode horizon.

A fictional needle is planted in a short prompt, H filler tokens are then consumed with
the recurrent state quantized at every step, and the question is appended last. Scoring is
exact match on a four-digit code, so a miss means the state no longer holds the fact
rather than that the model failed to follow an instruction.
"""
import argparse, hashlib, json, os, random, sys, time
import torch


def sseed(*parts):
    """A seed that is the same in every process. Python's hash() of a str is salted
    by PYTHONHASHSEED, so it cannot be used for this."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import (load, quantize_states, get_rstates, fake_quant,  # noqa: E402
                           QWEN35_08B)
from data import get_filler_ids                                  # noqa: E402
from niah import NEEDLE_TMPL, SUFFIX_TMPL, sample_spec, score   # noqa: E402

# layer index -> a fixed scale, calibrated once. None means the dynamic absmax.
# Each bit-width is calibrated against its own qmax; sharing one calibration across
# widths would scale every arm by the wrong constant.
ABS_SCALES = {}       # {bits: {layer_idx: scale}}
CAPS = {}             # {bits: {layer_idx: ceiling}}, per channel, for capped dynamic
CAP_MULT = 1.0        # ceiling = healthy absmax/qmax * this, so a healthy state never binds
_CALIB_STATE = None   # the healthy trajectory's final state, cached across bit-widths


@torch.no_grad()
def _calib_state(model, ids):
    global _CALIB_STATE
    if _CALIB_STATE is None:
        cache = None
        for t in range(ids.shape[1]):
            cache = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True).past_key_values
        _CALIB_STATE = {i: tt.abs().amax(-1, keepdim=True).clamp_min(1e-8).float()
                        for i, tt in get_rstates(cache)}
    return _CALIB_STATE


@torch.no_grad()
def calibrate_abs(model, ids, bits, mult=1.0):
    """Calibrate a per-channel fixed scale on a healthy trajectory, against this
    bit-width's own qmax."""
    amax = _calib_state(model, ids)
    qmax = 2 ** (bits - 1) - 1
    return {i: v / qmax * mult for i, v in amax.items()}


@torch.no_grad()
def calibrate_cap(model, ids, bits, mult=1.0):
    """Per-layer, per-channel ceiling from a healthy state: absmax/qmax times mult."""
    amax = _calib_state(model, ids)
    qmax = 2 ** (bits - 1) - 1
    return {i: v / qmax * mult for i, v in amax.items()}


def quantize_abs(cache, bits, clip=True):
    """Quantize against the fixed calibrated scale: the persistent-static policy."""
    if bits is None or cache is None:
        return 0
    scales = ABS_SCALES.get(bits)
    for i, tt in get_rstates(cache):
        asc = scales.get(i) if scales else None
        tt.copy_(fake_quant(tt.float(), bits, dim=-1, abs_scale=asc, clip=clip).to(tt.dtype))
    return 1


def quantize_cap(cache, bits):
    """Capped dynamic, s = min(absmax/qmax, ceiling): dynamic resolution, with the
    scale unable to follow the state upward past the ceiling."""
    if bits is None or cache is None:
        return 0
    caps = CAPS.get(bits)
    for i, tt in get_rstates(cache):
        cp = caps.get(i) if caps else None
        tt.copy_(fake_quant(tt.float(), bits, dim=-1, cap=cp).to(tt.dtype))
    return 1


def quantize_pct(cache, bits, pct=0.999):
    """Percentile baseline: the per-channel P99.9 rather than the max sets the scale."""
    if bits is None or cache is None:
        return 0
    for i, tt in get_rstates(cache):
        tt.copy_(fake_quant(tt.float(), bits, dim=-1, pct=pct).to(tt.dtype))
    return 1


def quantize_ema(cache, bits, ema_state, beta=0.9):
    """EMA baseline: the scale is a moving average, so one step's absmax cannot move it
    far. ema_state is {layer: previous scale}, carried across the steps of one trial."""
    if bits is None or cache is None:
        return 0
    qmax = 2 ** (bits - 1) - 1
    for i, tt in get_rstates(cache):
        cur = tt.abs().amax(-1, keepdim=True).clamp_min(1e-8).float() / qmax
        prev = ema_state.get(i)
        s = cur if prev is None else beta * prev + (1 - beta) * cur
        ema_state[i] = s
        y = tt.float() / s
        q = torch.clamp(torch.round(y), -qmax - 1, qmax)
        tt.copy_((q * s).to(tt.dtype))
    return 1


def quantize_oracle(cache, ref_cache, bits, clip=True):
    """The yoked policy: the scale comes from a full-precision trajectory run in
    lockstep, so it cannot be driven by the quantized state. This is the intervention
    that cuts the feedback edge. clip=False leaves the codes unclamped, which
    separates the clipping from the scale."""
    if bits is None or cache is None:
        return 0
    qmax = 2 ** (bits - 1) - 1
    ref = dict(get_rstates(ref_cache))
    for i, tt in get_rstates(cache):
        sc = ref[i].float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
        codes = torch.round(tt.float() / sc)
        if clip:
            codes = torch.clamp(codes, -qmax - 1, qmax)
        tt.copy_((codes * sc).to(tt.dtype))
    return 1


def quantize_oracle_refresh(cache, ref_cache, bits, K, held, step):
    """Yoked, but refreshed only every K steps and held constant in between.

    This separates two things the yoked policy confounds. Its scale comes from the
    healthy reference either way, so there is no feedback in either arm, but at K=1 the
    grid moves every step and at larger K it moves K times less often while keeping the
    same healthy magnitude. If recall improves as K rises, how often the grid changes is
    the operative variable and not how large the scale is. Codes are left unclamped.

    held is {layer: the scale currently in force}, carried across one trial's steps."""
    if bits is None or cache is None:
        return 0
    qmax = 2 ** (bits - 1) - 1
    ref = dict(get_rstates(ref_cache))
    for i, tt in get_rstates(cache):
        if step % K == 0 or i not in held:            # refresh from the reference, first step included
            held[i] = ref[i].float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
        sc = held[i]
        codes = torch.round(tt.float() / sc)          # unclamped, to match the arm this is compared against
        tt.copy_((codes * sc).to(tt.dtype))
    return 1


@torch.no_grad()
def one_trial(model, tok, needle_ids, q_ids, filler_seg, horizon, bits, code, mode="rtne",
              trial_seed=0):
    dev = model.device
    torch.manual_seed(trial_seed)   # one stochastic-rounding stream per trial
    _ema, _step, _held = {}, [0], {}
    oracle = mode in ("oracle", "oracle_noclip") or mode.startswith("orefresh")
    ref_cache = [None]              # the unquantized trajectory the yoked scale reads

    def q(c):
        if mode == "abs":       return quantize_abs(c, bits, clip=True)
        if mode == "absnoclip": return quantize_abs(c, bits, clip=False)   # fixed_noclip
        if mode == "cap":  return quantize_cap(c, bits)
        if mode == "pct":  return quantize_pct(c, bits)
        if mode == "ema":  return quantize_ema(c, bits, _ema)
        if mode.startswith("orefresh"):
            K = int(mode.split(":")[1]) if ":" in mode else 1
            _step[0] += 1
            return quantize_oracle_refresh(c, ref_cache[0], bits, K, _held, _step[0] - 1)
        if oracle:         return quantize_oracle(c, ref_cache[0], bits, clip=(mode == "oracle"))
        if mode.startswith("ckpt"):
            K = int(mode.split(":")[1]) if ":" in mode else 8
            _step[0] += 1
            return quantize_states(c, bits) if _step[0] % K == 0 else 0
        return quantize_states(c, bits, stochastic=(mode == "sr"))

    def step(x):
        """Step the quantized trajectory over x, advancing the reference trajectory
        alongside it when the policy needs one. Returns the logits."""
        nonlocal cache
        if oracle:
            ref_cache[0] = model(x, past_key_values=ref_cache[0], use_cache=True).past_key_values
        out = model(x, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        q(cache)
        return out.logits

    cache = None
    prompt = needle_ids + filler_seg[:192]
    for i in range(0, len(prompt), 64):
        step(torch.tensor([prompt[i:i + 64]], device=dev))
    for k in range(horizon):
        step(torch.tensor([[filler_seg[192 + k]]], device=dev))
    logits = step(torch.tensor([q_ids], device=dev))
    ans, nxt = [], logits[:, -1].float().argmax(-1, keepdim=True)
    for _ in range(12):
        ans.append(nxt.item())
        logits = step(nxt)
        nxt = logits[:, -1].float().argmax(-1, keepdim=True)
    return score(tok.decode(ans, skip_special_tokens=True), code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QWEN35_08B)
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[0, 512, 1024, 2048, 4096, 8192])
    # one condition per "name:bits:mode"; bits='none' is the unquantized arm,
    # and mode is rtne, sr or abs
    ap.add_argument("--conds", nargs="+",
                    default=["A0:none:rtne", "INT4:4:rtne", "INT4sr:4:sr",
                             "INT3:3:rtne", "INT3sr:3:sr"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--calib_offset", type=int, default=0, help="where the calibration segment starts")
    ap.add_argument("--calib_len", type=int, default=4096, help="how long the calibration segment is")
    ap.add_argument("--calib_mult", type=float, default=1.0, help="scale the calibrated step size, for the sensitivity sweep")
    ap.add_argument("--cap_mult", type=float, default=1.0, help="headroom multiplier on the capped-dynamic ceiling")
    ap.add_argument("--out", default="results/qwen35_recall_08b.json")
    a = ap.parse_args()

    def parse(c):
        # mode may itself contain a colon, so split on the first two only
        name, b, m = c.split(":", 2)
        return name, (None if b == "none" else int(b)), m
    conds = [parse(c) for c in a.conds]

    t0 = time.time()
    tok, model = load(a.model)
    # the filler pool has to hold both the test segment at the longest horizon and the
    # calibration segment, which may sit at a far offset
    need = max(max(a.horizons) + 2000, a.calib_offset + a.calib_len + 1000)
    filler = get_filler_ids(tok, need)
    print(f"[load] {a.model} | conds={[c[0] for c in conds]}", flush=True)

    co, cl = a.calib_offset, a.calib_len
    seg = filler[co:co + cl]
    if len(seg) < 64:
        raise ValueError(f"calibration segment too short: offset={co} len={cl} -> got {len(seg)}")
    cseq = torch.tensor([seg], device=model.device)
    for _, bits, mode in conds:
        if mode in ("abs", "absnoclip") and bits not in ABS_SCALES:
            ABS_SCALES[bits] = calibrate_abs(model, cseq, bits, mult=a.calib_mult)
            print(f"[calibrate abs] INT{bits}: {len(ABS_SCALES[bits])} layers | "
                  f"offset={co} len={cl} mult={a.calib_mult}", flush=True)
        if mode == "cap" and bits not in CAPS:
            CAPS[bits] = calibrate_cap(model, cseq, bits, mult=a.cap_mult)
            print(f"[calibrate cap] INT{bits}: {len(CAPS[bits])} layers | "
                  f"τ mult={a.cap_mult}", flush=True)

    summary, records = {}, []
    for name, bits, mode in conds:
        summary[name] = {}
        for h in a.horizons:
            hits = 0
            for s in range(a.seeds):
                # the same needle in the same filler at every horizon, so the
                # comparison across horizons is paired
                rng = random.Random(sseed("q35nest", s))
                word, code = sample_spec(rng)
                nid = tok(NEEDLE_TMPL.format(word=word, code=code),
                          add_special_tokens=False).input_ids
                qid = tok(SUFFIX_TMPL.format(word=word),
                          add_special_tokens=False).input_ids
                off = rng.randint(0, len(filler) - max(a.horizons) - 400)
                seg = filler[off:off + h + 400]
                ok = one_trial(model, tok, nid, qid, seg, h, bits, code, mode=mode,
                               trial_seed=sseed(name, h, s))
                hits += ok
                records.append(dict(cond=name, horizon=h, seed=s, hit=bool(ok)))
            summary[name][h] = hits / a.seeds
            print(f"  {name:7s} horizon={h:5d} recall={hits/a.seeds:.3f} "
                  f"[{(time.time()-t0)/60:.0f}min]", flush=True)

    print("\n[recall vs decode horizon]")
    print("  h     " + "".join(f"{h:>8}" for h in a.horizons))
    for name in summary:
        print(f"  {name:5s} " + "".join(f"{summary[name][h]:8.2f}" for h in a.horizons))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(model=a.model, summary=summary, records=records,
                   seeds=a.seeds, minutes=(time.time()-t0)/60,
                   config=vars(a), seed_scheme="md5(sseed)"),   # the whole config, so a run can be repeated
              open(a.out, "w"), indent=2)
    print(f"[saved] {a.out} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
