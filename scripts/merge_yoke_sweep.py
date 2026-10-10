"""Merge the yoked-sweep shards and test what the residual is made of.

Two readings of the damage that survives yoking: a grid too coarse, or rounding error
accumulating through the recurrence. The first shrinks with bit-width and the second does
not, so the shape of excess NLL against bit-width is the discriminator. Stochastic
rounding is the second probe: it removes the bias while enlarging the per-step error.
"""
import glob, json, math, os, statistics as st, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pattern = sys.argv[1] if len(sys.argv) > 1 else "results/yoke_sweep_08b_s*.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/yoke_sweep_08b.json"


def paired_t(d):
    d = [x for x in d if x == x]
    if len(d) < 2:
        return (st.mean(d) if d else float("nan")), None, len(d)
    m, sd = st.mean(d), st.stdev(d)
    return m, (m / (sd / math.sqrt(len(d))) if sd else math.inf), len(d)


merged, cfg = {}, None
for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
    d = json.load(open(p))
    cfg = cfg or d["config"]
    print("%-36s %d docs" % (os.path.basename(p), d.get("n_docs_done", 0)))
    for mode, v in d["summary"].items():
        merged.setdefault(mode, {"per_doc": [], "norm": []})
        merged[mode]["per_doc"].extend(v["per_doc"])
        if v.get("norm") is not None:
            merged[mode]["norm"].append(v["norm"])

summary = {m: dict(excess=st.mean([x for x in v["per_doc"] if x == x]),
                   per_doc=v["per_doc"], n=len(v["per_doc"]),
                   norm=st.mean(v["norm"]) if v["norm"] else None)
           for m, v in merged.items()}

bits = sorted(int(m[4:]) for m in summary if m.startswith("yoke") and m[4:].isdigit())
print("\nyoked residual against bit-width  (excess NLL over the unquantized model)")
print("%-8s %10s %8s %10s" % ("bits", "excess", "n", "vs INT8"))
base = summary.get("yoke8", {}).get("excess")
for b in bits:
    e = summary["yoke%d" % b]
    extra = e["excess"] - base if base is not None else float("nan")
    print("%-8d %10.3f %8d %10.3f" % (b, e["excess"], e["n"], extra))

print("\nhalving test: a resolution limit should fall by roughly a constant factor per bit")
for a, c in zip(bits, bits[1:]):
    ea, ec = summary["yoke%d" % a]["excess"], summary["yoke%d" % c]["excess"]
    print("  INT%d -> INT%d   %.3f -> %.3f   ratio %.2f" % (a, c, ea, ec, (ec / ea) if ea else float("nan")))

print("\nstochastic rounding, paired by document")
for sr in sorted(m for m in summary if m.endswith("_sr")):
    plain = sr[:-3]
    if plain not in summary:
        continue
    d = [x - y for x, y in zip(summary[sr]["per_doc"], summary[plain]["per_doc"])]
    m, t, n = paired_t(d)
    print("  %-10s vs %-10s  %+7.3f nats  t=%s  n=%d"
          % (sr, plain, m, ("%.2f" % t) if t is not None else "n/a", n))

if "self4" in summary:
    print("\ncross-check against results/yoke_08b.json (published 5.39 self / 3.57 yoked)")
    print("  self4 %.3f   yoke4 %.3f" % (summary["self4"]["excess"], summary.get("yoke4", {}).get("excess", float("nan"))))

json.dump(dict(summary=summary, config=cfg), open(os.path.join(ROOT, out_path), "w"), indent=2)
print("\n[saved]", out_path)
