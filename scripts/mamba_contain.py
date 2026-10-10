"""Containment on Mamba-2: the group-wise arm the paper never ran on an SSM.

The Discussion claims the exposure interaction replicates across architectures. It does.
The bit-width frontier and group-wise dominance were only ever run on GDN, and the
revision now says so. This closes that gap: the same containment comparison as
Table~\\ref{tab:contain}, on state-spaces/mamba2-1.3b.

Protocol is copied from qwen35_mamba.py deliberately, not refactored out of it, so that
file keeps producing the published numbers untouched. The two policies it already has
(dynamic4, static4) are run here as a **cross-check**: with --n_docs 16 --independent
--Hs 2048 they must reproduce results/mamba_ind_curve.json at H=2048 to a few hundredths
(the Mamba runtime is not bit-deterministic; the paper says so). --verify prints that
comparison and exits non-zero if it drifts, and nothing else in the output should be read
until it passes.

The SSM state is inf.key_value_memory_dict[layer][1], shape (1,64,64,128) =
(b, heads, head_dim, d_state). Every policy quantizes along d_state, per
(head, head_dim) — the axis that corresponds to GDN's last dimension. G=32 therefore
gives 4 groups, exactly as G=32 on GDN's 128-wide axis does.

    python scripts/mamba_contain.py --smoke
    python scripts/mamba_contain.py --verify --out results/mamba_verify_contain.json
    python scripts/mamba_contain.py --independent --n_docs 16 --Hs 2048 \
        --policies unquant dynamic4 static4 group32 capped4 zero \
        --out results/mamba_contain.json
"""
import argparse
import json
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import get_articles, get_filler_ids                              # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel             # noqa: E402
from mamba_ssm.utils.generation import InferenceParams                     # noqa: E402
from transformers import AutoTokenizer                                     # noqa: E402

DEV = "cuda"
MODEL_ID = "state-spaces/mamba2-1.3b"
TOK_ID = "EleutherAI/gpt-neox-20b"


def get_ssm(inf):
    return [(li, inf.key_value_memory_dict[li][1]) for li in sorted(inf.key_value_memory_dict)]


def qmax_of(bits):
    return 2 ** (bits - 1) - 1


# ───────────────────────────────────────────────────────────── the policies
def q_dyn(inf, absmax, bits, G=None):
    """Per-token dynamic: the scale is read from the state about to be quantized."""
    qm = qmax_of(bits)
    for _, s in get_ssm(inf):
        sc = (s.float().abs().amax(-1, keepdim=True) / qm).clamp_min(1e-12)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qm + 1), qm) * sc).to(s.dtype))


def q_static(inf, absmax, bits, G=None):
    """Persistent static: one calibrated scale per (layer, head, head_dim), never updated."""
    qm = qmax_of(bits)
    for li, s in get_ssm(inf):
        sc = (absmax[li].to(s.device) / qm).clamp_min(1e-12)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qm + 1), qm) * sc).to(s.dtype))


def q_group(inf, absmax, bits, G=32):
    """Persistent group-wise: a separate calibrated scale per G-wide group along d_state.

    Containment on the channel axis — an outlier inflates its own group's scale and no
    other. absmax here holds the per-group calibrated maxima (see calib_absmax)."""
    qm = qmax_of(bits)
    for li, s in get_ssm(inf):
        sh = s.shape                                              # (1,64,64,128)
        tg = s.float().reshape(*sh[:-1], sh[-1] // G, G)          # (1,64,64,ng,G)
        sc = (absmax[li].to(s.device) / qm).clamp_min(1e-12).unsqueeze(-1)
        code = torch.clamp(torch.round(tg / sc), -(qm + 1), qm)
        s.copy_((code * sc).reshape(sh).to(s.dtype))


def q_capped(inf, absmax, bits, G=None):
    """Capped dynamic: s = min(max|S|/qmax, ceiling), the ceiling from calibration.

    A ceiling in value rather than a fixed scale: keeps dynamic resolution while the
    scale can no longer chase the state upward."""
    qm = qmax_of(bits)
    for li, s in get_ssm(inf):
        cap = (absmax[li].to(s.device) / qm).clamp_min(1e-12)
        sc = torch.minimum((s.float().abs().amax(-1, keepdim=True) / qm).clamp_min(1e-12), cap)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qm + 1), qm) * sc).to(s.dtype))


def q_zero(inf, absmax, bits, G=None):
    """The control: discard the state entirely. Any policy that loses to this is not
    a compression of the state."""
    for _, s in get_ssm(inf):
        s.zero_()


POLICIES = {"dynamic": q_dyn, "static": q_static, "group": q_group, "capped": q_capped}


def parse_policy(name, default_G):
    """'group32' -> (q_group, 4, 32); 'dynamic4' -> (q_dyn, 4, None); 'zero' -> ..."""
    if name == "unquant":
        return None, None, None
    if name == "zero":
        return q_zero, 4, None
    for stem, fn in POLICIES.items():
        if name.startswith(stem):
            rest = name[len(stem):]
            if stem == "group":                    # group32 / group32_5 (G, then bits)
                parts = rest.split("_")
                G = int(parts[0]) if parts[0] else default_G
                bits = int(parts[1]) if len(parts) > 1 and parts[1] else 4
                return fn, bits, G
            return fn, int(rest) if rest else 4, None
    raise SystemExit("unknown policy %r" % name)


# ───────────────────────────────────────────────────────────── calibration
@torch.no_grad()
def calib_absmax(model, docs, maxlen, group_G=None):
    """Healthy-decode absmax along d_state, per (head, head_dim) or per group.

    Snapshots at several points of several sequences, exactly as qwen35_mamba.py does,
    so the static arm here is the static arm there."""
    SNAP = {128, 512, 1024, 1536, 2000}
    acc = {}
    for doc in docs:
        inf = InferenceParams(max_seqlen=maxlen, max_batch_size=1)
        for k in range(min(2048, len(doc))):
            model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf)
            inf.seqlen_offset += 1
            if k in SNAP:
                for li, s in get_ssm(inf):
                    if group_G:
                        sh = s.shape
                        a = s.float().reshape(*sh[:-1], sh[-1] // group_G, group_G) \
                             .abs().amax(-1)                       # (1,64,64,ng)
                    else:
                        a = s.float().abs().amax(-1, keepdim=True)  # (1,64,64,1)
                    acc[li] = torch.maximum(acc[li], a) if li in acc else a
    return acc


def ssm_norm(inf):
    # float64: at long horizons a float32 accumulation of a diverging state overflows to
    # inf while the elements themselves are finite, which would make "the norm diverges"
    # a statement about the accumulator.
    return st.mean([s.double().norm().item() for _, s in get_ssm(inf)])


@torch.no_grad()
def doc_nll(model, doc, CTX, H, cont, qfn, absmax, bits, G):
    inf = InferenceParams(max_seqlen=CTX + cont + 8, max_batch_size=1)
    warm = CTX - H
    out = None
    for k in range(warm):                                  # unquantized warm-up
        out = model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf)
        inf.seqlen_offset += 1
    for k in range(warm, CTX):                             # quantized exposure
        out = model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf)
        inf.seqlen_offset += 1
        if qfn:
            qfn(inf, absmax, bits, G)
    nm = ssm_norm(inf)
    cur = out.logits[:, -1].float()
    nll = 0.0
    for j in range(cont):                                  # score the continuation
        gt = doc[CTX + j]
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        out = model(torch.tensor([[gt]], device=DEV), inference_params=inf)
        inf.seqlen_offset += 1
        if qfn:
            qfn(inf, absmax, bits, G)
        cur = out.logits[:, -1].float()
    return nll / cont, nm


# ───────────────────────────────────────────────────────── the verification arm
VERIFY_REF = "mamba_ind_curve.json"
VERIFY_TOL = 0.08          # nats; the Mamba runtime is not deterministic between runs


def verify(summary, H, res_dir):
    """Compare the two policies that already exist against the published file."""
    path = os.path.join(res_dir, VERIFY_REF)
    if not os.path.exists(path):
        print("[verify] %s not found — cannot cross-check" % path)
        return 1
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]
    if str(H) not in ref:
        print("[verify] reference has no H=%d (has %s)" % (H, sorted(ref)))
        return 1
    ok = True
    print("[verify] against %s at H=%d, tolerance %.2f nats" % (VERIFY_REF, H, VERIFY_TOL))
    for pol in ("dynamic4", "static4"):
        if pol not in summary[H] or pol not in ref[str(H)]:
            print("   %-10s missing on one side — skipped" % pol)
            continue
        got = summary[H][pol]["excess"]
        want = ref[str(H)][pol]["excess"]
        drift = abs(got - want)
        flag = "ok" if drift <= VERIFY_TOL else "DRIFT"
        print("   %-10s got %.4f  published %.4f  |d|=%.4f  %s" % (pol, got, want, drift, flag))
        ok = ok and drift <= VERIFY_TOL
    print("[verify] %s" % ("PASSED — read the rest of the run" if ok else
                           "FAILED — do not read the rest of the run"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+",
                    default=["unquant", "dynamic4", "static4", "group32", "capped4", "zero"])
    ap.add_argument("--Hs", type=int, nargs="+", default=[2048])
    ap.add_argument("--G", type=int, default=32, help="default group size along d_state")
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=16)
    ap.add_argument("--stride", type=int, default=5000)
    ap.add_argument("--independent", action="store_true",
                    help="score on independent held-out articles (the paper's protocol)")
    ap.add_argument("--n_calib", type=int, default=3)
    ap.add_argument("--verify", action="store_true",
                    help="run only dynamic4/static4 and cross-check against the published file")
    ap.add_argument("--smoke", action="store_true", help="2 docs, one horizon, all policies")
    ap.add_argument("--out", default="results/mamba_contain.json")
    a = ap.parse_args()
    if a.verify:
        a.policies = ["unquant", "dynamic4", "static4"]
        a.n_docs, a.independent = 16, True
    if a.smoke:
        a.n_docs = 2
        a.Hs = a.Hs[:1]
    torch.set_grad_enabled(False)

    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=DEV, dtype=torch.bfloat16).eval()

    plans = {name: parse_policy(name, a.G) for name in a.policies}
    need_group = sorted({G for _, _, G in plans.values() if G})

    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(a.n_calib)]
    maxlen = a.CTX + a.cont + 8
    absmax = {None: calib_absmax(model, calib_docs, maxlen)}
    for G in need_group:
        absmax[G] = calib_absmax(model, calib_docs, maxlen, group_G=G)
    print("[calib] %s | per-channel + groups %s" % (MODEL_ID, need_group or "none"), flush=True)

    if a.independent:
        docs = [art[:a.CTX + a.cont] for art in
                get_articles(tok, a.n_docs, a.CTX + a.cont, part="test")]
        print("[docs] %d independent held-out articles" % len(docs), flush=True)
    else:
        test = get_filler_ids(tok, a.n_docs * a.stride + a.CTX + a.cont + 100, part="test")
        docs = [test[i * a.stride:i * a.stride + a.CTX + a.cont] for i in range(a.n_docs)]

    res = {H: {c: dict(nll=[], norm=[]) for c in a.policies} for H in a.Hs}
    for di, doc in enumerate(docs):
        for H in a.Hs:
            for name in a.policies:
                fn, bits, G = plans[name]
                nll, nm = doc_nll(model, doc, a.CTX, H, a.cont, fn, absmax.get(G), bits, G)
                res[H][name]["nll"].append(nll)
                res[H][name]["norm"].append(nm)
        print("  doc %d/%d done" % (di + 1, len(docs)), flush=True)

    summary = {}
    for H in a.Hs:
        summary[H] = {}
        uf = res[H]["unquant"]["nll"]
        for name in a.policies:
            e = [res[H][name]["nll"][i] - uf[i] for i in range(len(docs))]
            summary[H][name] = dict(excess=st.mean(e), norm=st.mean(res[H][name]["norm"]), exc=e)
        print("  H=%d: %s" % (H, " ".join(
            "%s=%.3f(norm%.2e)" % (c, summary[H][c]["excess"], summary[H][c]["norm"])
            for c in a.policies)), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, config=vars(a), model=MODEL_ID), fh, indent=2)
    print("[saved] %s" % a.out, flush=True)

    if a.verify:
        return verify(summary, a.Hs[0], res_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
