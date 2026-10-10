"""Merge the per-document shards of the long-horizon run and test what it shows.

Each shard runs its own unquantized reference over its own documents, so excess NLL is
already paired within a shard; merging concatenates per-document values rather than
averaging averages.

The question the run exists to answer -- does the damage keep growing past the 2048 the
paper measures, or does it plateau -- is a paired comparison, not a look at a curve. So
this reports, per policy, the paired growth from H=2048 to the longest horizon with its
t statistic, and the policy contrasts at that horizon.
"""
import glob, json, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pattern = sys.argv[1] if len(sys.argv) > 1 else "results/horizon32k_08b_s*.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/horizon32k_08b.json"


def paired_t(a, b):
    """Paired t on a-b. Returns (mean difference, t, n). t is None when undefined.

    Pairs containing a NaN are dropped rather than poisoning the statistic: at 2B the
    dynamic policy diverges out of bfloat16 range before the longest horizons, and that
    is a result to report separately, not one to average over.
    """
    d = [x - y for x, y in zip(a, b) if x == x and y == y]
    n = len(d)
    if n == 0:
        return float("nan"), None, 0
    if n < 2:
        return (d[0] if d else None), None, n
    m = st.mean(d)
    sd = st.stdev(d)
    if sd == 0:
        return m, math.inf if m else 0.0, n
    return m, m / (sd / math.sqrt(n)), n


shards = sorted(glob.glob(os.path.join(ROOT, pattern)))
if not shards:
    raise SystemExit("no shards matched %s" % pattern)

merged, docs_seen, cfg = {}, [], None
for p in shards:
    with open(p) as f:
        d = json.load(f)
    cfg = cfg or d["config"]
    docs_seen.extend(d["config"].get("doc_ids", []))
    print("%-42s %d docs" % (os.path.basename(p), d.get("n_docs_done", 0)))
    for pol, by_h in d["summary"].items():
        for H, v in by_h.items():
            merged.setdefault(pol, {}).setdefault(H, {"per_doc": [], "snorm": []})
            merged[pol][H]["per_doc"].extend(v["per_doc"])
            if v.get("snorm") is not None:
                merged[pol][H]["snorm"].append(v["snorm"])

summary = {}
for pol, by_h in merged.items():
    summary[pol] = {}
    for H, v in by_h.items():
        fin = [x for x in v["per_doc"] if x == x]
        summary[pol][H] = dict(excess=st.mean(fin) if fin else float("nan"),
                               n_nan=len(v["per_doc"]) - len(fin), per_doc=v["per_doc"],
                               n=len(v["per_doc"]),
                               snorm=st.mean(v["snorm"]) if v["snorm"] else None)

Hs = sorted({int(H) for by_h in summary.values() for H in by_h})
# order the known policies, then append anything else rather than dropping it:
# a hard-coded list silently loses a policy added to the runner later
ORDER = ("zero", "dynamic4", "dynamic4_sr", "static4", "static4_sr",
         "group32", "group32_sr", "group32_int5", "group32_int8")
pols = [p for p in ORDER if p in summary] + [p for p in summary if p not in ORDER]
lo, hi = Hs[0], Hs[-1]

print("\nexcess NLL (nats) vs quantized exposure")
print("%-12s" % "policy", "".join("%10s" % ("H=%d" % h) for h in Hs), "   n")
for pol in pols:
    row = "".join("%10.3f" % summary[pol][str(h)]["excess"] if str(h) in summary[pol]
                  else "%10s" % "-" for h in Hs)
    print("%-12s" % pol, row, " %3d" % max(v["n"] for v in summary[pol].values()))

print("\ngrowth from H=%d to H=%d, paired by document" % (lo, hi))
print("%-12s %10s %8s %5s" % ("policy", "delta", "t", "n"))
growth = {}
for pol in pols:
    a = summary[pol][str(hi)]["per_doc"]
    b = summary[pol][str(lo)]["per_doc"]
    m, t, n = paired_t(a, b)
    growth[pol] = dict(delta=m, t=t, n=n)
    print("%-12s %10.3f %8s %5d" % (pol, m, ("%.2f" % t) if t is not None else "n/a", n))

print("\npolicy contrasts at H=%d, paired by document" % hi)
contrasts = {}
for i, p in enumerate(pols):
    for q in pols[i + 1:]:
        m, t, n = paired_t(summary[p][str(hi)]["per_doc"], summary[q][str(hi)]["per_doc"])
        contrasts["%s-%s" % (p, q)] = dict(delta=m, t=t, n=n)
        print("%-12s %-12s %8.3f  t=%s" % (p, q, m, ("%.2f" % t) if t is not None else "n/a"))

with open(os.path.join(ROOT, out_path), "w") as f:
    json.dump(dict(summary=summary, growth=growth, contrasts=contrasts, config=cfg,
                   shards=[os.path.basename(s) for s in shards],
                   doc_ids=sorted(docs_seen)), f, indent=2)
print("\n[saved]", out_path)
