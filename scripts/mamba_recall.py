"""The failure frontier on Mamba-2: NIAH recall across (bit-width x decode horizon).

Figure 1a is GDN only. The claim that a bit-width certified at short exposure stops
being safe as decode proceeds is the paper's framing contribution, and on an SSM it has
never been measured. This runs the same protocol on state-spaces/mamba2-1.3b.

The protocol is copied from qwen35_recall.py step for step, because the number of times
the state is quantized is the independent variable and any drift in it changes the
result. Exposure accounting, per trial:

    prompt  = needle + 192 filler tokens, quantized once every 64 tokens  (GDN: chunked
              prefill of 64, quantized after each chunk)
    horizon = H single tokens, quantized after every one
    question= ~20 tokens, quantized once at the end                       (GDN: one chunk)
    answer  = 12 greedy tokens, quantized after every one

Mamba's recurrent path takes one token at a time, so prefill here is a token loop that
quantizes every 64th step rather than a 64-wide forward that quantizes once. The count of
quantizations matches; the tokens between them see a slightly fresher state. --audit
reports the count so a change in it is visible rather than silent.

Scoring is unchanged: the answer is a 4-digit code, hit if the code appears in the
12 decoded tokens. Seeds are derived by md5 of the trial identity, so a shard run on its
own reproduces the same needles as the same shard inside a full run.

    python scripts/mamba_recall.py --smoke
    python scripts/mamba_recall.py --seeds 12 --seed_from 0 --seed_to 3 \
        --out results/mamba_bitsweep_s0.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import get_filler_ids                                            # noqa: E402
from niah import NEEDLE_TMPL, SUFFIX_TMPL, sample_spec, score              # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel             # noqa: E402
from mamba_ssm.utils.generation import InferenceParams                     # noqa: E402
from transformers import AutoTokenizer                                     # noqa: E402

DEV = "cuda"
MODEL_ID = "state-spaces/mamba2-1.3b"
TOK_ID = "EleutherAI/gpt-neox-20b"
PROMPT_FILLER = 192          # matches qwen35_recall.py's filler_seg[:192]
PREFILL_CHUNK = 64           # matches its chunked prefill
N_GEN = 12                   # matches its answer window


def sseed(*parts):
    """Deterministic across processes: Python's hash() is salted per interpreter."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def get_ssm(inf):
    return [(li, inf.key_value_memory_dict[li][1]) for li in sorted(inf.key_value_memory_dict)]


def q_dynamic(inf, bits):
    """Per-token dynamic, per (head, head_dim) along d_state — the frontier's policy.

    fake_quant(t, bits, dim=-1) on GDN is exactly this: absmax over the last axis,
    round to nearest, clamp to the signed range."""
    qm = 2 ** (bits - 1) - 1
    for _, s in get_ssm(inf):
        sc = (s.float().abs().amax(-1, keepdim=True) / qm).clamp_min(1e-12)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qm + 1), qm) * sc).to(s.dtype))


@torch.no_grad()
def one_trial(model, tok, needle_ids, q_ids, filler_seg, horizon, bits, code, audit=None):
    inf = InferenceParams(
        max_seqlen=len(needle_ids) + PROMPT_FILLER + horizon + len(q_ids) + N_GEN + 8,
        max_batch_size=1)
    nq = [0]

    def feed(tid, quantize):
        out = model(torch.tensor([[tid]], device=DEV), inference_params=inf)
        inf.seqlen_offset += 1
        if quantize and bits is not None:
            q_dynamic(inf, bits)
            nq[0] += 1
        return out.logits

    prompt = list(needle_ids) + list(filler_seg[:PROMPT_FILLER])
    logits = None
    for i, tid in enumerate(prompt):                    # prefill: quantize every 64th
        logits = feed(tid, (i + 1) % PREFILL_CHUNK == 0 or i == len(prompt) - 1)
    for k in range(horizon):                            # the exposure being varied
        logits = feed(filler_seg[PROMPT_FILLER + k], True)
    for i, tid in enumerate(q_ids):                     # question: quantize once, at the end
        logits = feed(tid, i == len(q_ids) - 1)

    ans = []
    nxt = int(logits[:, -1].float().argmax(-1).item())
    for _ in range(N_GEN):
        ans.append(nxt)
        logits = feed(nxt, True)
        nxt = int(logits[:, -1].float().argmax(-1).item())
    if audit is not None:
        audit.append(nq[0])
    return score(tok.decode(ans, skip_special_tokens=True), code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1024, 2048, 4096, 8192])
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 5, 4, 3])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed_from", type=int, default=0)
    ap.add_argument("--seed_to", type=int, default=None, help="exclusive; default = --seeds")
    ap.add_argument("--audit", action="store_true",
                    help="report how many times the state was quantized per trial")
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 2 horizons, INT4 only")
    ap.add_argument("--out", default="results/mamba_bitsweep.json")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.seed_to, a.horizons, a.bits, a.audit = 2, 2, [0, 1024], [4], True
    if a.seed_to is None:
        a.seed_to = a.seeds
    torch.set_grad_enabled(False)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEV, dtype=torch.bfloat16).eval()
    filler = get_filler_ids(tok, max(a.horizons) + 2000)
    conds = ["A0"] + ["INT%d" % b for b in a.bits]
    print("[load] %s | conds=%s | seeds %d..%d" % (MODEL_ID, conds, a.seed_from, a.seed_to),
          flush=True)

    records, audit = [], []
    for s in range(a.seed_from, a.seed_to):
        # the needle and the filler window are shared across conditions, so every
        # comparison at a given seed is paired
        rng = random.Random(sseed("mamba-bitsweep", s))
        word, code = sample_spec(rng)
        needle_ids = tok(NEEDLE_TMPL.format(word=word, code=code),
                         add_special_tokens=False).input_ids
        q_ids = tok(SUFFIX_TMPL.format(word=word), add_special_tokens=False).input_ids
        need = PROMPT_FILLER + max(a.horizons) + 8
        off = rng.randint(0, max(0, len(filler) - need - 1))
        seg = filler[off:off + need]
        for H in a.horizons:
            for name, bits in zip(conds, [None] + list(a.bits)):
                hit = one_trial(model, tok, needle_ids, q_ids, seg, H, bits, code,
                                audit if a.audit else None)
                records.append(dict(cond=name, horizon=H, seed=s, hit=bool(hit)))
        print("  seed %d done (%.1f min)" % (s, (time.time() - t0) / 60), flush=True)

    summary = {}
    for name in conds:
        summary[name] = {}
        for H in a.horizons:
            hs = [r["hit"] for r in records if r["cond"] == name and r["horizon"] == H]
            summary[name][H] = sum(hs) / len(hs) if hs else None
    for name in conds:
        print("  %-5s %s" % (name, " ".join(
            "H=%s:%.2f" % (H, summary[name][H]) for H in a.horizons)), flush=True)
    if a.audit:
        print("[audit] state quantized %s times per trial (min..max over trials)"
              % ("%d..%d" % (min(audit), max(audit)) if audit else "n/a"), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(model=MODEL_ID, summary=summary, records=records,
                       seeds=a.seed_to - a.seed_from, config=vars(a),
                       minutes=(time.time() - t0) / 60), fh, indent=2)
    print("[saved] %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
