"""Pool the per-seed shards of mamba_recall.py into one file.

Recall is a 0/1 outcome per (condition, horizon, seed), so merging is pooling the
records and recomputing the rate. Two things are checked rather than assumed: that no
seed appears in two shards, and that every shard covered the same grid. Either would
silently bias a rate.

    python scripts/merge_mamba_recall.py results/mamba_bitsweep_s*.json \
        --out results/mamba_bitsweep.json
"""
import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default="results/mamba_bitsweep.json")
    a = ap.parse_args()

    paths = []
    for p in a.shards:
        paths.extend(sorted(glob.glob(p)) or [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("no shard files matched")

    records, seen, grids, model = [], {}, set(), None
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        model = model or d.get("model")
        grids.add((tuple(sorted(d["summary"])), tuple(sorted(
            int(h) for h in d["summary"][list(d["summary"])[0]]))))
        for r in d["records"]:
            key = (r["cond"], r["horizon"], r["seed"])
            if key in seen:
                sys.exit("seed %d appears in both %s and %s for %s at H=%d — pooling it "
                         "would weight that seed twice"
                         % (r["seed"], seen[key], os.path.basename(p), r["cond"], r["horizon"]))
            seen[key] = os.path.basename(p)
            records.append(r)
    if len(grids) > 1:
        sys.exit("shards cover different grids: %s" % sorted(grids))

    conds, Hs = grids.pop()
    summary, counts = {}, {}
    for c in conds:
        summary[c], counts[c] = {}, {}
        for H in Hs:
            hs = [r["hit"] for r in records if r["cond"] == c and r["horizon"] == H]
            summary[c][H] = (sum(hs) / len(hs)) if hs else None
            counts[c][H] = len(hs)
    n = sorted({v for c in counts for v in counts[c].values()})
    if len(n) > 1:
        print("[warn] uneven seed counts across the grid: %s" % n)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(model=model, summary=summary, records=records,
                       seeds=max(n), counts=counts,
                       shards=[os.path.basename(p) for p in paths]), fh, indent=2)
    print("merged %d shards, %d trials, %d seeds -> %s" % (len(paths), len(records), max(n), a.out))
    for c in sorted(conds, key=lambda x: (x != "A0", x)):
        print("  %-5s %s" % (c, " ".join("H=%s:%.2f" % (H, summary[c][H]) for H in Hs)))


if __name__ == "__main__":
    main()
