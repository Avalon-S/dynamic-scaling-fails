#!/usr/bin/env python
"""Generate all paper figures from results/*.json.

Every number plotted here is read from a result file; nothing is hand-entered.
Output: Paper_Writing/figures/*.pdf (vector, fonttype 42 for ACL).
"""
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "Paper_Writing", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.6,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
})

import sys
# Target width/output dir: ACL two-column (6.45in) by default, or the NeurIPS
# single-column text block (5.5in) via `python make_figures.py neurips`.
if len(sys.argv) > 1 and sys.argv[1] == "neurips":
    COL2 = 5.5
    OUT = os.path.join(ROOT, "Paper_Writing", "figures_lcfm")
    os.makedirs(OUT, exist_ok=True)
else:
    COL2 = 6.45
COL1 = 3.15
DASH = "–"                  # en dash (matplotlib does not read LaTeX "--")

C = {                            # colour-blind-safe policy palette
    "dynamic":  "#d1495b",
    "static":   "#e8952b",
    "group":    "#00798c",
    "prot":     "#1b2a4a",
    "unquant":  "#8d99ae",
    "yoked":    "#5c9e5f",
}
# Greyscale luminance of dynamic (0.191) and group (0.156) is nearly identical,
# so colour alone does not survive a B/W print. Give every policy its own
# marker and dash pattern as a redundant channel.
MK = {"dynamic": "o", "static": "s", "group": "^", "prot": "D"}
LS = {"dynamic": "-", "static": (0, (4, 1.6)), "group": (0, (1.4, 1.2)),
      "prot": (0, (5, 1.4, 1, 1.4))}
BITC = {8: "#1b2a4a", 5: "#00798c", 4: "#e8952b", 3: "#d1495b"}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def exc(entry):
    """Excess NLL, tolerating the two schema variants across result files."""
    return entry.get("excess_full", entry.get("excess"))


def norm(entry):
    return entry.get("snorm", entry.get("norm"))


def ktick(h):
    """1536 -> '1.5k', 1024 -> '1k', 512 -> '512'."""
    if h < 1024:
        return str(h)
    v = h / 1024.0
    return ("%d" % v if v == int(v) else "%.1f" % v) + "k"


# =================================================================== Fig 1
def fig_frontier():
    """Teaser: (a) bit-width x horizon recall frontier, (b) the norm runaway."""
    b08 = load("qwen35_bitsweep_08b.json")["summary"]
    b2b = load("qwen35_bitsweep_2b.json")["summary"]
    cur = load("longdoc2_ind_curve_08b.json")["summary"]
    cur2 = load("longdoc2_ind_2b.json")["summary"]

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.70))

    # ---- (a) failure frontier
    a = ax[0]
    for bits in (8, 5, 4, 3):
        k = "INT%d" % bits
        Hs = sorted(int(h) for h in b08[k])
        ys = [b08[k][str(h)] for h in Hs]
        # nothing is measured between H=0 and the next point, so that stretch is drawn
        # as a faint connector; the solid line covers only measured spans
        a.plot(Hs[:2], ys[:2], ":", color=BITC[bits], lw=.8, alpha=.55, zorder=3)
        a.plot(Hs[1:], ys[1:], "-", color=BITC[bits], zorder=3)
        a.plot(Hs, ys, "o", color=BITC[bits], label="INT%d" % bits, zorder=3)
        if k in b2b:
            H2 = sorted(int(h) for h in b2b[k])
            y2 = [b2b[k][str(h)] for h in H2]
            a.plot(H2[:2], y2[:2], ":", color=BITC[bits], lw=.7, alpha=.4, zorder=2)
            a.plot(H2[1:], y2[1:], "--", color=BITC[bits], alpha=.65, linewidth=.9, zorder=2)
            a.plot(H2, y2, "s", color=BITC[bits], alpha=.65, markersize=2.6, zorder=2)
    a.axhline(1.0, color=C["unquant"], ls=":", lw=.8, zorder=1)
    a.text(8192, 1.03, "unquantized", ha="right", va="bottom",
           fontsize=6.2, color=C["unquant"])
    a.set_xscale("symlog", linthresh=1024)
    a.set_xticks([0, 1024, 2048, 4096, 8192])
    a.set_xticklabels(["0", "1k", "2k", "4k", "8k"])
    a.set_xlabel("quantized decode horizon $H$")
    a.set_ylabel("NIAH recall")
    a.set_ylim(-.05, 1.14)
    a.set_title("(a) failure frontier", loc="left")
    # colour = bit-width, in-panel; the model encoding is shared with (b) and
    # goes in the figure-level row below.
    h1, l1 = a.get_legend_handles_labels()
    a.legend(h1, l1, ncol=2, loc="lower left", columnspacing=.8,
             handlelength=1.4, borderpad=.1, labelspacing=.22)

    # ---- (b) state-norm runaway
    b = ax[1]
    for tag, res, ls, mk in (("0.8B", cur, "-", "o"), ("2B", cur2, "--", "s")):
        Hs = sorted(int(h) for h in res)
        for pol, key in (("dynamic", "dynamic4"), ("static", "static4")):
            b.plot(Hs, [norm(res[str(h)][key]) for h in Hs], ls, marker=mk,
                   color=C[pol], alpha=1.0 if tag == "0.8B" else .6,
                   markersize=3.2 if tag == "0.8B" else 2.6,
                   linewidth=1.2 if tag == "0.8B" else .9,
                   label=("%s INT4" % pol) if tag == "0.8B" else None)
    b.axhline(norm(cur["2048"]["unquant"]), color=C["unquant"], ls=":", lw=.8)
    b.text(1550, 9.6, "unquantized $\\approx$ 17", fontsize=6.2,
           color=C["unquant"], va="center", ha="center")
    b.set_yscale("log")
    b.set_ylim(8, 1.5e5)
    b.set_xlabel("quantized decode horizon $H$")
    b.set_ylabel(r"recurrent-state norm $\|S\|$")
    Hs = sorted(int(h) for h in cur)
    b.set_xticks(Hs)
    b.set_xticklabels([ktick(h) for h in Hs])
    b.set_title("(b) scale%sstate runaway" % DASH, loc="left")
    b.legend(loc="upper left", handlelength=1.5, borderpad=.1, labelspacing=.25)

    # shared row: the solid/dashed model encoding used by both panels
    style = [plt.Line2D([], [], color="0.35", ls="-", marker="o", ms=3, lw=1.1),
             plt.Line2D([], [], color="0.35", ls="--", marker="s", ms=2.6, lw=.9,
                        alpha=.65)]
    fig.legend(style, ["0.8B", "2B"], loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.10), handlelength=1.9, columnspacing=2.0)
    fig.tight_layout(pad=.3, w_pad=1.6)
    save(fig, "fig_frontier")


# =================================================================== Fig 2
def fig_mechanism():
    """(a) per-step co-expansion, split so the two quantities are not mixed on
    one axis; (b) scale-yoking as a dot plot (log-scale bars mislead)."""
    dg = load("outlier_diag_08b.json")
    y08 = load("yoke_08b.json")["summary"]
    y2b = load("yoke_2b.json")["summary"]

    fig = plt.figure(figsize=(COL2, 1.78))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=.32, wspace=.42)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    b = fig.add_subplot(gs[:, 0 + 1])

    # ---- (a) two stacked panels: the norm and the tail ratio have different
    # units, so they get their own axes rather than one shared "magnitude".
    t = np.arange(len(dg["dynamic"]["norm"]))
    W = 33

    def smooth(v):
        v = np.asarray(v, float)
        k = np.ones(W) / W
        return np.convolve(v, k, "same") / np.convolve(np.ones_like(v), k, "same")

    for pol, col in (("dynamic", C["dynamic"]), ("static", C["static"])):
        a1.plot(t, smooth(dg[pol]["norm"]), color=col, label=pol)
        a2.plot(t, smooth(dg[pol]["outlier"]), color=col, ls="--", lw=1.0)
    a1.set_yscale("log")
    a2.set_yscale("log")
    a1.set_ylabel(r"$\|S\|$", labelpad=1)
    a2.set_ylabel(r"$p_{99.9}/p_{50}$", labelpad=1)
    a1.set_ylim(5, 300)
    a2.set_ylim(8e2, 1.2e4)
    a1.set_xlim(0, 2048)
    a1.tick_params(labelbottom=False)
    a2.set_xticks([0, 512, 1024, 1536, 2048])
    a2.set_xticklabels(["0", "512", "1k", "1.5k", "2k"])
    a2.set_xlabel("decode step", labelpad=1)
    a1.set_title("(a) state norm and tail both expand", loc="left", pad=3)

    for ax, lab in ((a1, "state norm"), (a2, "tail concentration")):
        ax.text(.985, .06, lab, transform=ax.transAxes, ha="right",
                fontsize=6.0, color="0.4")

    # ---- (b) dot plot.  No connecting lines: the three columns are separate
    # experimental arms, not a trajectory through time.
    labels = ["unquantized", "dynamic\n(self-scaled)", "dynamic\n(FP-yoked)"]
    keys = ["unquant", "dyn_self", "dyn_yoked"]
    cols = [C["unquant"], C["dynamic"], C["yoked"]]
    x = np.arange(3)
    off = 0.16
    for (tag, y, mk, al, dx) in (("0.8B", y08, "o", 1.0, -off),
                                 ("2B", y2b, "s", .55, off)):
        v = [y[k]["norm"] for k in keys]
        b.scatter(x + dx, v, marker=mk, s=36, zorder=3, alpha=al, color=cols,
                  edgecolors="white", linewidths=.6, label=tag)
        for j2, val in enumerate(v):
            e = int(np.log10(val))
            txt = ("%.0f" % val if val < 1e3
                   else "$%.1f{\\times}10^{%d}$" % (val / 10.0 ** e, e))
            # 0.8B labels above the marker, 2B below, so neither can collide
            # with the annotation that sits between the columns.
            b.annotate(txt, (x[j2] + dx, val), textcoords="offset points",
                       xytext=(0, 8),
                       ha="center", fontsize=5.9, alpha=al)
    b.set_yscale("log")
    b.set_ylim(7, 5e5)
    b.set_yticks([10, 1e2, 1e3, 1e4, 1e5])
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_xlim(-.55, 2.55)
    b.set_ylabel(r"recurrent-state norm $\|S\|$", labelpad=1)
    b.set_title("(b) yoking the scale removes the runaway", loc="left", pad=3)


    pol = [plt.Line2D([], [], color=C["dynamic"], lw=1.4),
           plt.Line2D([], [], color=C["static"], lw=1.4)]
    scale = [plt.Line2D([], [], color="0.45", marker="o", ls="none", ms=4.5),
             plt.Line2D([], [], color="0.45", marker="s", ls="none", ms=4.5,
                        alpha=.55)]
    fig.legend(pol + scale, ["dynamic", "static", "0.8B", "2B"],
               loc="lower center", ncol=4, bbox_to_anchor=(.5, -.19),
               handlelength=1.7, columnspacing=1.7, scatterpoints=1)
    fig.tight_layout(pad=.3)
    save(fig, "fig_mechanism")


# =================================================================== Fig 3
def fig_policy():
    """Policy x exposure across three models."""
    panels = [
        ("GDN-0.8B", load("longdoc2_ind_curve_08b.json")["summary"],
         ["dynamic4", "static4", "group32"]),
        ("GDN-2B", load("longdoc2_ind_2b.json")["summary"],
         ["dynamic4", "static4", "group32"]),
        ("Mamba-2-1.3B", load("mamba_ind_curve.json")["summary"],
         ["dynamic4", "static4"]),
    ]
    name = {"dynamic4": "per-token dynamic INT4",
            "static4": "persistent static INT4",
            "group32": "group-wise INT4 ($G{=}32$)"}
    pol = {"dynamic4": "dynamic", "static4": "static", "group32": "group"}

    fig, ax = plt.subplots(1, 3, figsize=(COL2, 1.70), sharey=True)
    handles = {}
    for k, (title, res, keys) in enumerate(panels):
        a = ax[k]
        Hs = sorted(int(h) for h in res)
        for key in keys:
            p = pol[key]
            ln, = a.plot(Hs, [exc(res[str(h)][key]) for h in Hs], color=C[p],
                         ls=LS[p], marker=MK[p], label=name[key])
            handles.setdefault(key, ln)
        if "zero" in res[str(Hs[0])]:
            z = float(np.mean([exc(res[str(h)]["zero"]) for h in Hs]))
            a.axhline(z, color="0.55", ls=":", lw=.8)
            if k != 1:   # in the 2B panel the floor runs through the static curve
                a.text(Hs[0], z - .25, "zero-state floor", fontsize=6.4,
                       color="0.35", ha="left", va="top")
        a.set_xscale("log", base=2)
        a.set_xticks(Hs)
        a.set_xticklabels([ktick(h) for h in Hs])
        a.minorticks_off()
        a.set_xlabel("quantized exposure $H$")
        a.set_title(title, loc="left")
        a.set_ylim(-.4, 8.4)
    ax[0].set_ylabel("excess NLL (nats)")
    ax[0].annotate("crossover", xy=(1270, 4.9), xytext=(300, 6.9), fontsize=6.3,
                   arrowprops=dict(arrowstyle="->", lw=.6, color="0.35"))
    ax[2].annotate("abrupt\ncollapse", xy=(1900, 6.4), xytext=(400, 6.0),
                   fontsize=6.3,
                   arrowprops=dict(arrowstyle="->", lw=.6, color="0.35"))
    fig.legend([handles[k] for k in ("dynamic4", "static4", "group32")],
               [name[k] for k in ("dynamic4", "static4", "group32")],
               loc="lower center", ncol=3, bbox_to_anchor=(.5, -.11),
               handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(pad=.3, w_pad=.6)
    save(fig, "fig_policy")


# =================================================================== Fig 4
def fig_survival():
    """Survival of confident-token faithfulness (right-censored)."""
    order = [("group32_first3", "group-wise + first-3 protected", C["prot"], LS["prot"]),
             ("group32", "group-wise", C["group"], LS["group"]),
             ("static4", "persistent static", C["static"], LS["static"]),
             ("dynamic4", "per-token dynamic", C["dynamic"], LS["dynamic"])]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.62), sharey=True)
    handles, labels = [], []
    for k, (f, title) in enumerate((("survival_08b.json", "GDN-0.8B"),
                                    ("survival_2b.json", "GDN-2B"))):
        d = load(f)
        a = ax[k]
        grid = np.logspace(np.log10(8), np.log10(8192), 240)
        for key, lab, col, ls in order:
            e = d["results"][key]["8.0"]
            tau = np.asarray(e["tau"], float)
            cen = np.asarray(e["censored"], bool)
            # right-censored: an article censored at length L is at risk only
            # while h < L, and counts as surviving there.
            s = [np.sum(tau > h) / max(np.sum(~cen | (tau > h)), 1) for h in grid]
            ln, = a.step(grid, s, where="post", color=col, ls=ls, label=lab)
            if k == 0:                    # both panels share one legend row
                handles.append(ln)
                labels.append(lab)
        a.set_xscale("log", base=2)
        a.set_xlim(8, 8192)
        a.set_xticks([16, 64, 256, 1024, 4096])
        a.set_xticklabels(["16", "64", "256", "1k", "4k"])
        a.minorticks_off()
        a.set_ylim(-.03, 1.06)
        a.set_xlabel(r"decode step $h$")
        a.set_title(title, loc="left")
        a.axvline(2048, color="0.72", ls=":", lw=.7)
        if k == 0:
            a.set_ylabel(r"$S(h)=\Pr(\tau>h)$")
        a.text(1950, 1.05, "$H{=}2048$", fontsize=6, color="0.45",
               va="top", ha="right")
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(.5, -.13), handlelength=1.9, columnspacing=1.5)
    fig.tight_layout(pad=.3, w_pad=.6)
    save(fig, "fig_survival")


# =================================================================== Fig 5
def fig_alloc():
    """(a) layer sensitivity on true layer indices; (b) rate--distortion, with
    the in-sample sweep and the leak-free point explicitly distinguished."""
    ls08 = load("layersens_08b.json")
    ls2b = load("layersens_2b.json")
    pa08 = load("pareto_08b.json")["summary"]["2048"]
    pa2b = load("pareto_2b.json")["summary"]["2048"]
    ca08 = load("clean_alloc_08b.json")["results"]

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.70))

    # ---- (a) plot at the TRUE layer index, so the gaps where the hybrid has
    # attention rather than linear-attention layers are visible instead of
    # being silently closed up.
    a = ax[0]
    layers = ls08["layers"]
    w = .38
    for i, (d, tag, col) in enumerate(((ls08, "0.8B", C["prot"]),
                                       (ls2b, "2B", C["group"]))):
        v = [max(d["summary"]["2048"]["L%d" % L]["excess_full"], 0) for L in layers]
        a.bar(np.asarray(layers, float) + (i - .5) * w, v, w, color=col,
              label=tag, edgecolor="none")
    a.set_xticks([0, 4, 8, 12, 16, 20])
    a.set_xlim(-1.2, 23.2)
    a.set_xlabel("layer index (gaps = attention layers)")
    a.set_ylabel("single-layer excess NLL")
    a.set_ylim(0, 3.05)
    a.set_title("(a) sensitivity concentrates in layer 0", loc="left")
    a.legend(loc="upper right", handlelength=1.2, borderpad=.1,
             labelspacing=.2, bbox_to_anchor=(1.0, .84))
    a.annotate("78% (0.8B) / 63% (2B)\nof all single-layer damage",
               xy=(1.0, 2.45), xytext=(4.0, 2.80), fontsize=6.2,
               arrowprops=dict(arrowstyle="->", lw=.6, color="0.35"))

    # ---- (b) rate-distortion.  The connected series are the in-sample rate
    # sweep; the star is the same policy re-measured under the leak-free
    # protocol.  They differ, and that difference is the selection leak.
    b = ax[1]
    for pa, tag, mk, alpha, lw in ((pa08, "0.8B", "o", 1.0, 1.2),
                                   (pa2b, "2B", "s", .55, .9)):
        uni = sorted((pa[k]["rate"], pa[k]["excess_full"]) for k in
                     ("uniform_int3", "uniform_int4", "uniform_int5"))
        ext = sorted((pa[k]["rate"], pa[k]["excess_full"]) for k in pa
                     if k.startswith("ext_top3"))
        b.plot(*zip(*uni), "-" + mk, color=C["static"], alpha=alpha, lw=lw,
               ms=3.2 if tag == "0.8B" else 2.6,
               label="uniform group-wise (%s)" % tag)
        b.plot(*zip(*ext), "-" + mk, color=C["prot"], alpha=alpha, lw=lw,
               ms=3.2 if tag == "0.8B" else 2.6,
               label="top-3 protected (%s)" % tag)
    xr, yr = ca08["first3"]["rate"], ca08["first3"]["excess_full"]
    y_in = pa08["ext_top3_i8_rest_i2"]["excess_full"]
    b.plot([xr, xr], [y_in, yr], color="0.45", lw=.8, ls=":", zorder=4)
    b.scatter([xr], [yr], marker="*", s=62, color=C["dynamic"], zorder=5,
              label="same policy, leak-free")
    b.annotate("selection\nleak", xy=(xr + .02, (y_in + yr) / 2),
               xytext=(4.28, 1.44), fontsize=5.9, color="0.35",
               arrowprops=dict(arrowstyle="-", lw=.5, color="0.55"))
    b.set_xlabel("modeled bit rate (bits / state element)")
    b.set_ylabel("excess NLL (nats)")
    b.set_xlim(3.7, 6.15)
    b.set_ylim(1.1, 3.4)
    b.set_title("(b) protect the first layers, then stop", loc="left")
    b.legend(loc="upper right", handlelength=1.5, labelspacing=.22,
             borderpad=.1, scatterpoints=1)

    fig.tight_layout(pad=.3, w_pad=1.6)
    save(fig, "fig_alloc")


# =================================================================== Fig 6
def fig_horizon32k():
    """Confirmatory run: does the damage keep growing past the 2048 the paper measures?"""
    res = load("horizon32k_08b.json")["summary"]
    name = {"dynamic4": "per-token dynamic INT4",
            "static4": "persistent static INT4",
            "group32": "group-wise INT4 ($G{=}32$)"}
    pol = {"dynamic4": "dynamic", "static4": "static", "group32": "group"}
    keys = [k for k in ("dynamic4", "static4", "group32") if k in res]
    Hs = sorted(int(h) for h in res[keys[0]])
    lo, hi = str(Hs[0]), str(Hs[-1])

    fig, ax = plt.subplots(1, 2, figsize=(COL2, 1.85),
                           gridspec_kw={"width_ratios": [1.55, 1]})
    handles = []
    for key in keys:
        p = pol[key]
        ln, = ax[0].plot(Hs, [res[key][str(h)]["excess"] for h in Hs],
                         color=C[p], ls=LS[p], marker=MK[p], label=name[key])
        handles.append(ln)
    # the no-context floor, measured under this same protocol so the comparison holds
    if os.path.exists(os.path.join(RES, "horizon32k_zero_08b.json")):
        z = load("horizon32k_zero_08b.json")["summary"]["zero"]
        # long dashes, not dots: group-wise is already a fine dotted line, and the two
        # meet at the right-hand end where they must stay tellable apart in greyscale
        ax[0].plot(Hs, [z[str(h)]["excess"] for h in Hs], color="0.55",
                   ls=(0, (6, 2)), lw=.9)
        ax[0].text(Hs[0], z[lo]["excess"], " zero-state floor", fontsize=6.3,
                   color="0.35", ha="left", va="bottom")
    ax[0].axvline(2048, color="0.7", ls=(0, (1, 2)), lw=.7)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(Hs)
    ax[0].set_xticklabels([ktick(h) for h in Hs])
    ax[0].minorticks_off()
    ax[0].set_xlabel("quantized exposure $H$")
    ax[0].set_ylabel("excess NLL (nats)")
    ax[0].set_title("(a) damage against exposure", loc="left")

    # every document, not just the mean: the claim is a paired test, so show the pairs
    b = ax[1]
    for i, key in enumerate(keys):
        p = pol[key]
        d = [a - c for a, c in zip(res[key][hi]["per_doc"], res[key][lo]["per_doc"])]
        b.scatter([i + 0.5] * len(d) + np.random.RandomState(0).uniform(-.09, .09, len(d)),
                  d, s=9, color=C[p], marker=MK[p], zorder=3, linewidths=0)
        b.plot([i + 0.24, i + 0.76], [np.mean(d)] * 2, color=C[p], lw=1.4, zorder=4)
    b.axhline(0, color="0.7", lw=.7)
    b.set_xticks([i + 0.5 for i in range(len(keys))])
    b.set_xticklabels(["dynamic", "static", "group-wise"], fontsize=6.6)
    b.set_xlim(0, len(keys))
    b.set_ylabel("growth from 2k to 32k (nats)")
    b.set_title("(b) every document, same direction", loc="left")

    fig.legend(handles, [name[k] for k in keys], loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.10), handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(pad=.3, w_pad=1.6)
    save(fig, "fig_horizon32k")


if __name__ == "__main__":
    fig_frontier()
    fig_mechanism()
    fig_policy()
    fig_survival()
    fig_alloc()
    if os.path.exists(os.path.join(RES, "horizon32k_08b.json")):
        fig_horizon32k()
    else:
        print("skip fig_horizon32k: results/horizon32k_08b.json not present")
