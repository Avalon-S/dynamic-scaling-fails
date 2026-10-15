#!/usr/bin/env python
"""Export every number the project page plots into project-page/static/data/data.js.

Nothing on the page is typed by hand. This reads results/*.json — the same files
make_figures.py reads for the paper's figures — and writes one JS object. Paired
t-statistics are recomputed here from the per-document arrays rather than copied
out of the paper, so a mismatch with the paper is visible rather than silent:
run with --check to print the recomputation next to what the paper claims.

    python scripts/make_page_data.py
    python scripts/make_page_data.py --check
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
# The working tree calls it project-page; the published repository calls it docs.
PAGE = next((d for d in ("docs", "project-page")
             if os.path.isdir(os.path.join(ROOT, d, "static"))), "docs")
OUT = os.path.join(ROOT, PAGE, "static", "data", "data.js")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def exc(entry):
    """Excess NLL, tolerating the two schema variants — same helper make_figures.py uses."""
    v = entry.get("excess_full", entry.get("excess"))
    return v if not isinstance(v, list) else (sum(v) / len(v) if v else None)


def nrm(entry):
    return entry.get("snorm", entry.get("norm"))


def per_doc(entry):
    """The per-document array, under whichever of the three names this file uses."""
    for k in ("per_doc", "exc", "excess"):
        v = entry.get(k)
        if isinstance(v, list):
            return v
    raise KeyError("no per-document array in %s" % list(entry))


def paired_t(a, b):
    """Paired t on a-b by document. Returns (mean difference, t, n)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (sum(d) / max(n, 1), float("nan"), n)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return (m, m / se if se > 0 else float("inf"), n)


def r(x, k=4):
    return None if x is None else round(x, k)


# ─────────────────────────────────────────────────────── the failure frontier
def frontier():
    out = {}
    for tag, fn in (("0.8B", "qwen35_bitsweep_08b.json"), ("2B", "qwen35_bitsweep_2b.json")):
        s = load(fn)["summary"]
        out[tag] = {k: {int(h): v for h, v in s[k].items()}
                    for k in ("INT8", "INT5", "INT4", "INT3") if k in s}
    out["seeds"] = load("qwen35_bitsweep_08b.json")["seeds"]
    return out


# ────────────────────────────────────────────── the scale-state runaway curves
def runaway():
    out = {}
    for tag, fn in (("0.8B", "longdoc2_ind_curve_08b.json"),
                    ("2B", "longdoc2_ind_curve_2b.json"),
                    ("Mamba-2-1.3B", "mamba_ind_curve.json")):
        s = load(fn)["summary"]
        Hs = sorted(int(h) for h in s)
        pols = {}
        for pol in ("unquant", "dynamic4", "static4", "group32", "zero"):
            if pol not in s[str(Hs[0])]:
                continue
            pols[pol] = {
                "excess": [r(exc(s[str(h)][pol])) for h in Hs],
                "norm": [r(nrm(s[str(h)][pol]), 2) for h in Hs],
            }
        out[tag] = {"Hs": Hs, "policies": pols, "n_docs": load(fn)["config"]["n_docs"]}
    return out


# ──────────────────────────────────────────────── the scale-yoking experiment
def yoke():
    out = {}
    for tag, fn in (("0.8B", "yoke_08b.json"), ("2B", "yoke_2b.json")):
        s = load(fn)["summary"]
        rows = {k: {"excess": r(exc(s[k])), "norm": r(nrm(s[k]), 2)}
                for k in ("unquant", "dyn_self", "dyn_yoked")}
        dm, dt, dn = paired_t(per_doc(s["dyn_self"]), per_doc(s["dyn_yoked"]))
        rows["recovered"] = {"delta": r(dm), "t": r(dt, 2), "n": dn}
        rows["n"] = len(per_doc(s["dyn_self"]))
        out[tag] = rows
    return out


# ───────────────────────── what more bits buy, with the scale held still
def bits():
    d = load("yoke_sweep_08b.json")
    s = d["summary"]
    yoked = {}
    for k, v in s.items():
        if k.startswith("yoke") and not k.endswith("_sr"):
            yoked[int(k[4:])] = {"excess": r(exc(v)), "norm": r(nrm(v), 2)}
    sr = {}
    for k, v in s.items():
        if k.endswith("_sr"):
            sr[k] = {"excess": r(exc(v)), "norm": r(nrm(v), 2)}
    # how many documents improve monotonically as bits rise, and the INT4->INT8 gap
    order = sorted(yoked)
    per = [per_doc(s["yoke%d" % b]) for b in order]
    mono = sum(1 for i in range(len(per[0]))
               if all(per[j][i] >= per[j + 1][i] for j in range(len(per) - 1)))
    gm, gt, gn = paired_t(per_doc(s["yoke4"]), per_doc(s["yoke8"]))
    srm, srt, srn = paired_t(per_doc(s["yoke4_sr"]), per_doc(s["yoke4"]))
    srm8, srt8, _ = paired_t(per_doc(s["yoke8_sr"]), per_doc(s["yoke8"]))
    return {
        "yoked": yoked, "sr": sr, "self4": r(exc(s["self4"])),
        "monotone": mono, "n": len(per[0]),
        "gap48": {"delta": r(gm), "t": r(gt, 2), "n": gn},
        "sr_cost": {"int4": {"delta": r(srm), "t": r(srt, 2)},
                    "int8": {"delta": r(srm8), "t": r(srt8, 2)}},
        "surviving": r(100.0 * exc(s["yoke8"]) / exc(s["yoke4"]), 1),
    }


# ─────────── containment: a ceiling, an interruption, and the edge removed
CONTAIN_LABEL = [
    ("dynamic4", "per-token dynamic INT4", "none", "The loop, uncontained."),
    ("dynR128", "periodic reset, K=128", "interruption",
     "Zero the state every 128 steps: the loop cannot compound past a reset."),
    ("capped", "capped dynamic INT4", "ceiling",
     "Cap the dynamic scale at a calibrated ceiling, so it cannot chase the state up."),
    ("capped_esc", "capped + exact escape", "ceiling",
     "Keep the overflowing elements exactly, on top of the cap. It adds nothing."),
    # Persistent, not confined: a calibrated scale never reads the state, so there is no
    # feedback edge left for the grouping to confine. What the extra scales buy is
    # resolution, and whether that is worth having turns out to depend on the model.
    ("group32", "group-wise INT4 (G=32)", "removed",
     "A calibrated scale never reads the state, so the edge is gone; the groups buy "
     "resolution on top of that."),
]
LOCATE_LABEL = [
    ("refine_top", "extra bits to the most-overflowing channels"),
    ("refine_rand0", "extra bits to a random channel subset"),
    ("range_top", "widen the range of the top channels"),
    ("range_rand0", "widen the range of a random subset"),
    ("escape_exact", "store the overflowing elements exactly"),
]


def contain():
    h = load("hybrid_08b.json")["summary"]
    rst = load("reset_08b.json")["summary"]
    pool = dict(h)
    pool.update({k: v for k, v in rst.items() if k.startswith("dynR")})
    rows = []
    for key, label, kind, why in CONTAIN_LABEL:
        v = pool[key]
        ref = "capped" if key == "capped_esc" else "dynamic4"
        m, t, n = paired_t(per_doc(v), per_doc(pool[ref]))
        rows.append({"key": key, "label": label, "kind": kind, "why": why,
                     "excess": r(exc(v)), "norm": r(nrm(v), 2),
                     "vs": ref, "delta": r(m), "t": r(t, 2), "n": n})
    # the hybrids: a dynamic-then-static schedule caps the runaway late
    hyb = []
    for key in ("hybDS512", "hybDS1024", "hybSD512", "hybSD1024"):
        if key not in h:
            continue
        m, t, n = paired_t(per_doc(h[key]), per_doc(h["group32"]))
        hyb.append({"key": key, "excess": r(exc(h[key])),
                    "vs_group32": {"delta": r(m), "t": r(t, 2), "n": n}})
    # locating outliers, at a matched rate
    rng = load("range_08b.json")["summary"]
    loc = [{"key": k, "label": lab, "nll": r(rng[k]["nll"]),
            "bits": r(rng[k]["bits"], 3), "recall": r(rng[k]["tf_recall"], 4)}
           for k, lab in LOCATE_LABEL if k in rng]
    for k in ("static4", "static5", "fp16", "dynamic4"):
        if k in rng:
            loc.append({"key": k, "label": k, "nll": r(rng[k]["nll"]),
                        "bits": r(rng[k]["bits"], 3), "recall": r(rng[k]["tf_recall"], 4)})
    cm, ct, cn = paired_t(per_doc(pool["capped"]), per_doc(pool["static4"]))
    return {"rows": rows, "hybrids": hyb, "locate": loc,
            "static4_norm": r(nrm(h["static4"]), 2),
            "capped_vs_static4": {"delta": r(cm), "t": r(ct, 2), "n": cn}}


# ───────────────────────────────────── out to 32K, with a zero-state control
def h32k():
    main = load("horizon32k_08b.json")
    zero = load("horizon32k_zero_08b.json")
    bits5_8 = load("horizon32k_bits_08b.json")
    ckpts = sorted(int(h) for h in main["summary"]["group32"])
    pols = {}
    for src in (main, zero, bits5_8):
        for pol, per_h in src["summary"].items():
            pols[pol] = {"excess": [r(exc(per_h[str(h)])) for h in ckpts],
                         "norm": [r(nrm(per_h[str(h)]), 2) for h in ckpts]}
    # stochastic rounding, pooled over the four shards
    sr = {}
    shards = [load("h32k_sr_08b_s%d.json" % i) for i in range(4)]
    for pol in shards[0]["summary"]:
        pooled = {h: [] for h in ckpts}
        for sh in shards:
            for h in ckpts:
                pooled[h].extend(per_doc(sh["summary"][pol][str(h)]))
        sr[pol] = {"excess": [r(sum(pooled[h]) / len(pooled[h])) for h in ckpts],
                   "n": len(pooled[ckpts[0]])}
    # group-wise INT4 against clearing the state, at every checkpoint
    gz = []
    for i, h in enumerate(ckpts):
        a = per_doc(main["summary"]["group32"][str(h)])
        b = per_doc(zero["summary"]["zero"][str(h)])
        m, t, n = paired_t(a, b)
        gz.append({"H": h, "delta": r(m), "t": r(t, 2), "n": n})
    return {"ckpts": ckpts, "policies": pols, "sr": sr,
            "growth": {k: {"delta": r(v["delta"]), "t": r(v["t"], 2), "n": v["n"]}
                       for k, v in main["growth"].items()},
            "contrasts": {k: {"delta": r(v["delta"]), "t": r(v["t"], 2), "n": v["n"]}
                          for k, v in main["contrasts"].items()},
            "vs_zero": gz, "n_docs": len(main["doc_ids"])}


# ──────────────────────────────────────────── per-layer allocation, leak-free
ALLOC_LABEL = {
    "extreme_rescue_top3": "top-3 layers INT8, the rest INT2",
    "knapsack": "matched-budget knapsack optimizer",
    "first3": "protect the first three layers",
    "l0_plus_2rand": "layer 0 plus two random layers",
    "uniform_int3": "uniform INT3",
    "uniform_int4": "uniform INT4",
    "rand3_a": "three random layers (a)",
    "rand3_b": "three random layers (b)",
    "local_mse_top3": "reconstruction-MSE top-3",
    "inverse_bottom3": "the three least sensitive layers",
}
SELECTED = ("extreme_rescue_top3", "knapsack", "first3")


def alloc():
    d = load("clean_alloc_08b.json")
    rows = []
    for key, v in d["results"].items():
        rows.append({"key": key, "label": ALLOC_LABEL.get(key, key),
                     "excess": r(exc(v)), "rate": r(v.get("rate"), 2)})
    rows.sort(key=lambda x: (x["excess"] is None, x["excess"]))
    for row in rows:
        row["selected"] = row["key"] in SELECTED
    return {"rows": rows, "provenance": {k: v for k, v in d["provenance"].items()
                                         if k in ("model", "G", "H", "CTX", "cont", "dtype")}}


# ─────────────────────────── a short evaluation certifies a policy as safe
def survival():
    d = load("survival_08b.json")
    mus = d["provenance"]["mus"]
    out = {}
    for pol, per_mu in d["results"].items():
        out[pol] = {}
        for mu in mus:
            k = str(float(mu))
            if k in per_mu:
                v = per_mu[k]
                out[pol][mu] = {kk: r(vv, 4) if isinstance(vv, float) else vv
                                for kk, vv in v.items()}
    return {"mus": mus, "policies": out, "n_docs": len(d["provenance"]["test_hashes"]),
            "Hmax": d["provenance"]["Hmax"]}


# ───────────────── what share of decode memory the state is actually worth
def memory():
    """The constants memory_share.py computes from, so the page can recompute live."""
    return {
        "Qwen3.5-0.8B": dict(linear_layers=18, attn_layers=6, kv_heads=2, head_dim=256,
                             kv_bytes=2, state_elems=16 * 128 * 128, state_bytes=4,
                             kind="hybrid"),
        "Mamba-2-1.3B": dict(linear_layers=48, attn_layers=0, kv_heads=0, head_dim=0,
                             kv_bytes=2, state_elems=64 * 64 * 128, state_bytes=4,
                             kind="pure SSM"),
    }


# ─────────────────────────────────────────────────────────────────── assembly
# ──────────────────────────────── the same containment policies on three models
CROSS_FILES = [
    ("GDN-0.8B", "longdoc2_ind_curve_08b.json"),
    ("GDN-2B", "longdoc2_ind_curve_2b.json"),
    ("Mamba-2-1.3B", "mamba_contain.json"),
]
CROSS_POL = [("dynamic4", "per-token dynamic INT4"),
             ("static4", "persistent static INT4"),
             ("zero", "zero-state control"),
             ("group32", "persistent group-wise INT4")]


def cross():
    """Table 2. Three models, one protocol, excess NLL at H=2048.

    The columns mean nothing if the runs differ, so the protocol is compared across the
    three files and a mismatch raises rather than renders."""
    protos, cols = set(), []
    for label, fname in CROSS_FILES:
        d = load(fname)
        c = d.get("config", {})
        protos.add((c.get("CTX"), c.get("cont"), c.get("n_docs"), c.get("stride"),
                    c.get("independent")))
        at = d["summary"]["2048"]
        rows = []
        for key, name in CROSS_POL:
            rows.append({"key": key, "label": name, "excess": r(exc(at[key])),
                         "norm": r(nrm(at[key]), 2)})
        # every pairing the paper's paragraph quotes, recomputed
        pairs = {}
        for a, b in (("group32", "static4"), ("static4", "zero"),
                     ("group32", "zero"), ("dynamic4", "zero")):
            m, t, n = paired_t(per_doc(at[a]), per_doc(at[b]))
            wins = sum(1 for x, y in zip(per_doc(at[a]), per_doc(at[b])) if x < y)
            pairs["%s_vs_%s" % (a, b)] = {"delta": r(m), "t": r(t, 2),
                                          "wins": wins, "n": n}
        cols.append({"model": label, "rows": rows, "pairs": pairs})
    if len(protos) != 1:
        raise SystemExit("cross(): the three runs do not share a protocol: %s" % protos)
    proto = protos.pop()
    return {"cols": cols, "policies": [p[0] for p in CROSS_POL],
            "protocol": {"CTX": proto[0], "cont": proto[1], "n_docs": proto[2],
                         "stride": proto[3], "independent": proto[4]}}


# ───────────────────────────── the recall frontier on the SSM, 12 seeds, 1 trial a cell
MAMBA_RECALL_H = [0, 1024, 2048, 4096, 8192]


def mamba_recall():
    """Appendix recall table, Mamba-2 block.

    The unquantized row is carried because it is 0.75 at 4096 and 0.00 at 8192: without it
    there is no way to tell the probe expiring from the quantizer breaking."""
    recs = load("mamba_bitsweep.json")["records"]
    conds = ["A0", "INT8", "INT5", "INT4", "INT3"]
    seeds = sorted({x["seed"] for x in recs})
    out = {}
    for c in conds:
        row = []
        for h in MAMBA_RECALL_H:
            hits = [x["hit"] for x in recs if x["cond"] == c and x["horizon"] == h]
            row.append(r(sum(hits) / float(len(hits)), 4) if hits else None)
        out[c] = row
    return {"horizons": MAMBA_RECALL_H, "conds": conds, "rows": out,
            "n_seeds": len(seeds), "trials": len(recs),
            "expired_at": 8192}


def build():
    return {
        "frontier": frontier(),
        "runaway": runaway(),
        "yoke": yoke(),
        "bits": bits(),
        "contain": contain(),
        "cross": cross(),
        "mamba_recall": mamba_recall(),
        "h32k": h32k(),
        "alloc": alloc(),
        "survival": survival(),
        "memory": memory(),
    }


PAPER_CLAIMS = [
    # (what the paper prints, where the page gets it) — recomputed, not copied
    ("yoke 0.8B dyn_self excess = 5.39", lambda d: d["yoke"]["0.8B"]["dyn_self"]["excess"]),
    ("yoke 0.8B dyn_yoked excess = 3.57", lambda d: d["yoke"]["0.8B"]["dyn_yoked"]["excess"]),
    ("yoked INT8 excess = 0.31", lambda d: d["bits"]["yoked"][8]["excess"]),
    ("yoked INT3 excess = 4.82", lambda d: d["bits"]["yoked"][3]["excess"]),
    ("INT4->INT8 yoked gap 3.26, t=32.5", lambda d: (d["bits"]["gap48"]["delta"], d["bits"]["gap48"]["t"])),
    ("8.7% survives at INT8", lambda d: d["bits"]["surviving"]),
    ("monotone in 15 of 16", lambda d: (d["bits"]["monotone"], d["bits"]["n"])),
    ("SR costs +6.71 at INT4, t=41.1", lambda d: (d["bits"]["sr_cost"]["int4"]["delta"], d["bits"]["sr_cost"]["int4"]["t"])),
    ("SR costs +0.62 at INT8, t=10.5", lambda d: (d["bits"]["sr_cost"]["int8"]["delta"], d["bits"]["sr_cost"]["int8"]["t"])),
    ("contain: dynamic INT4 = 5.47", lambda d: [r_["excess"] for r_ in d["contain"]["rows"] if r_["key"] == "dynamic4"][0]),
    ("contain: reset K=128 = 3.38, t=-6.85", lambda d: [(r_["excess"], r_["t"]) for r_ in d["contain"]["rows"] if r_["key"] == "dynR128"][0]),
    ("contain: capped = 3.58, t=-11.0", lambda d: [(r_["excess"], r_["t"]) for r_ in d["contain"]["rows"] if r_["key"] == "capped"][0]),
    ("   ... and capped vs static4, the value Table 1 used to print", lambda d: d["contain"]["capped_vs_static4"]),
    ("cross Table 2 row dynamic4 = 5.39 / 6.21 / 7.22", lambda d: [c["rows"][0]["excess"] for c in d["cross"]["cols"]]),
    ("cross Table 2 row static4  = 4.82 / 4.08 / 2.55", lambda d: [c["rows"][1]["excess"] for c in d["cross"]["cols"]]),
    ("cross Table 2 row zero     = 3.05 / 4.08 / 3.19", lambda d: [c["rows"][2]["excess"] for c in d["cross"]["cols"]]),
    ("cross Table 2 row group32  = 2.05 / 2.15 / 2.70", lambda d: [c["rows"][3]["excess"] for c in d["cross"]["cols"]]),
    ("group-static: -2.77 (t=-24.7) / -1.93 (t=-12.2) / +0.16 (t=+2.39)",
     lambda d: [(c["pairs"]["group32_vs_static4"]["delta"], c["pairs"]["group32_vs_static4"]["t"]) for c in d["cross"]["cols"]]),
    ("static-zero: +1.77 (0/16) / +0.01 (8/16) / -0.64 (16/16)",
     lambda d: [(c["pairs"]["static4_vs_zero"]["delta"], "%d/%d" % (c["pairs"]["static4_vs_zero"]["wins"], c["pairs"]["static4_vs_zero"]["n"])) for c in d["cross"]["cols"]]),
    ("Mamba INT4 recall 1.00 / 1.00 / 0.33 / 0.00 / 0.00", lambda d: d["mamba_recall"]["rows"]["INT4"]),
    ("Mamba unquantized   1.00 / 1.00 / 1.00 / 0.75 / 0.00", lambda d: d["mamba_recall"]["rows"]["A0"]),
    ("contain: +escape = 3.40, t=1.53 vs capped", lambda d: [(r_["excess"], r_["t"]) for r_ in d["contain"]["rows"] if r_["key"] == "capped_esc"][0]),
    ("contain: group-wise = 2.09, t=-13.4", lambda d: [(r_["excess"], r_["t"]) for r_ in d["contain"]["rows"] if r_["key"] == "group32"][0]),
    ("32K static4-group32 = 2.76, t=9.3", lambda d: (d["h32k"]["contrasts"]["static4-group32"]["delta"], d["h32k"]["contrasts"]["static4-group32"]["t"])),
    ("32K group32 growth 1.28, t=6.9", lambda d: (d["h32k"]["growth"]["group32"]["delta"], d["h32k"]["growth"]["group32"]["t"])),
    ("group-wise INT4 at 32K = clearing the state", lambda d: d["h32k"]["vs_zero"][-1]),
    ("alloc extreme = 1.58, uniform INT3 = 2.35", lambda d: [(x["label"], x["excess"]) for x in d["alloc"]["rows"][:4]]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the recomputation next to what the paper claims")
    a = ap.parse_args()
    d = build()
    if a.check:
        print("recomputed from results/*.json — compare against the paper by eye:\n")
        for label, get in PAPER_CLAIMS:
            try:
                print("  %-52s -> %s" % (label, get(d)))
            except Exception as exc:          # a renamed key should be loud, not silent
                print("  %-52s -> FAILED %s" % (label, exc))
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    body = json.dumps(d, indent=1, sort_keys=False, ensure_ascii=False)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("/* Generated by scripts/make_page_data.py — do not edit by hand.\n"
                 "   Every number here comes from results/*.json, the same files the\n"
                 "   paper's figures are built from. Re-run after any new result. */\n")
        fh.write("window.LQ = ")
        fh.write(body)
        fh.write(";\n")
    print("wrote %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024.0))


if __name__ == "__main__":
    sys.exit(main())
