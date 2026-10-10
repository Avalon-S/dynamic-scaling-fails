"""Stochastic rounding against round-to-nearest, at every exposure checkpoint.

The two runs are separate jobs over the same seven contexts. GDN decoding here is
deterministic -- the long-horizon campaign reproduced 35 of 35 per-document values
bit-identically across independent runs -- so the unquantized reference is the same in
both and the policies pair by document. That pairing is checked, not assumed.
"""
import glob, json, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPTS = ["2048", "4096", "8192", "16384", "32768"]
PAIRS = [("dynamic4_sr", "dynamic4"), ("static4_sr", "static4"), ("group32_sr", "group32")]


def per_doc(pattern, policy, H, field="per_doc"):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        d = json.load(open(f))
        ids = d["config"]["doc_ids"]
        blk = d["summary"].get(policy)
        if not blk or H not in blk:
            continue
        for i, x in enumerate(blk[H][field]):
            out[ids[i]] = x
    return out


def raw_unquant(pattern, H):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        d = json.load(open(f))
        ids = d["config"]["doc_ids"]
        for i, x in enumerate(d["raw_nll"]["unquant"][H]):
            out[ids[i]] = x
    return out


def paired(a, b):
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks if a[k] == a[k] and b[k] == b[k]]
    if len(d) < 2:
        return (st.mean(d) if d else float("nan")), None, len(d)
    m, sd = st.mean(d), st.stdev(d)
    return m, (m / (sd / math.sqrt(len(d))) if sd else math.inf), len(d)


SR = "results/h32k_sr_08b_s*.json"
RT = "results/horizon32k_08b_s*.json"

print("== pairing check: the unquantized reference must match across the two runs ==")
worst = 0.0
for H in CKPTS:
    a, b = raw_unquant(SR, H), raw_unquant(RT, H)
    ks = sorted(set(a) & set(b))
    if not ks:
        print("  H=%-6s no overlap yet" % H)
        continue
    dmax = max(abs(a[k] - b[k]) for k in ks)
    worst = max(worst, dmax)
    print("  H=%-6s %d documents, largest difference %.2e" % (H, len(ks), dmax))
print("  -> %s\n" % ("identical, the runs pair" if worst < 1e-9 else
                     "NOT identical: pair within each run instead"))

for sr, rt in PAIRS:
    print("== %s against %s ==" % (sr, rt))
    print("  %-8s %10s %10s %10s %8s %5s" % ("H", "RTNE", "SR", "SR - RTNE", "t", "n"))
    for H in CKPTS:
        a, b = per_doc(SR, sr, H), per_doc(RT, rt, H)
        if not a or not b:
            continue
        m, t, n = paired(a, b)
        print("  %-8s %10.3f %10.3f %10.3f %8s %5d"
              % (H, st.mean(list(b.values())), st.mean(list(a.values())), m,
                 ("%.2f" % t) if t is not None else "n/a", n))
    print()
