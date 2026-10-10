/* When Dynamic Scaling Fails — project page interactions.

   Every number drawn here comes from static/data/data.js, which
   scripts/make_page_data.py exports from the same results/*.json the paper's
   figures are built from. Nothing below is a result typed by hand.

   All drawing goes through render(), which re-reads the palette from CSS each time, so
   the colours are whatever the stylesheet currently says and never a copy of them. */
(function () {
  "use strict";

  var D = window.LQ;
  var NS = "http://www.w3.org/2000/svg";
  var C = {};                                  // palette, refreshed in readPalette()

  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) {
    var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function h(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function f(v, d) { return v == null ? "—" : v.toFixed(d == null ? 2 : d); }
  function ktick(n) {
    if (n < 1024) return String(n);
    var v = n / 1024;
    return (v === Math.round(v) ? v : v.toFixed(1)) + "K";
  }
  function bigNum(v) {
    if (v == null) return "—";
    if (v >= 10000) return Math.round(v).toLocaleString("en-US");
    if (v >= 100) return v.toFixed(0);
    return v.toFixed(2);
  }
  function readPalette() {
    var s = getComputedStyle(document.documentElement);
    ["brand", "brand-2", "fail", "fail-2", "keep", "mid", "ink", "ink-2", "ink-3",
      "line", "grey", "grid"]
      .forEach(function (k) { C[k] = s.getPropertyValue("--" + k).trim(); });
  }

  // ── tooltip ────────────────────────────────────────────────────────────────
  var tip = $("tip");
  function hover(node, html) {
    node.style.cursor = "default";
    node.addEventListener("mousemove", function (ev) {
      tip.innerHTML = html; tip.classList.add("on");
      var x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + 310 > window.innerWidth) x = ev.clientX - 310;
      if (y + 120 > window.innerHeight) y = ev.clientY - 110;
      tip.style.left = x + "px"; tip.style.top = y + "px";
    });
    node.addEventListener("mouseleave", function () { tip.classList.remove("on"); });
  }

  // ── a small chart frame: equal-spaced categorical x, linear or log y ───────
  function frame(host, opts) {
    host.innerHTML = "";
    var W = opts.w || 520, H = opts.h || 250;
    var m = opts.m || { t: 10, r: 12, b: 34, l: 44 };
    var s = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img" }, host);
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var n = opts.xs.length;
    var log = !!opts.log;
    var y0 = opts.y0, y1 = opts.y1;
    function X(i) { return m.l + (n === 1 ? iw / 2 : iw * i / (n - 1)); }
    function Y(v) {
      if (v == null) return null;
      var t = log ? (Math.log(Math.max(v, y0)) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))
                  : (v - y0) / (y1 - y0);
      return m.t + ih * (1 - Math.max(0, Math.min(1, t)));
    }
    (opts.yticks || []).forEach(function (v) {
      var y = Y(v);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: C.grid, "stroke-width": 1 }, s);
      txt(s, m.l - 7, y + 3.5, opts.yfmt ? opts.yfmt(v) : String(v),
        { "text-anchor": "end", "font-size": 10.5, fill: C["ink-3"] });
    });
    opts.xs.forEach(function (lab, i) {
      txt(s, X(i), H - m.b + 15, lab, { "text-anchor": "middle", "font-size": 10.5, fill: C["ink-3"] });
    });
    el("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: C.line, "stroke-width": 1 }, s);
    if (opts.ylab) {
      txt(s, 11, m.t + ih / 2, opts.ylab, {
        "text-anchor": "middle", "font-size": 10.5, fill: C["ink-3"],
        transform: "rotate(-90 11 " + (m.t + ih / 2) + ")"
      });
    }
    return { svg: s, X: X, Y: Y, W: W, H: H, m: m, iw: iw, ih: ih };
  }
  function path(fr, vals, col, opts) {
    opts = opts || {};
    var d = "", started = false;
    vals.forEach(function (v, i) {
      if (v == null) { if (!opts.bridge) started = false; return; }
      d += (started ? " L " : " M ") + fr.X(i) + " " + fr.Y(v);
      started = true;
    });
    if (d) el("path", {
      d: d, fill: "none", stroke: col, "stroke-width": opts.w || 1.8,
      "stroke-dasharray": opts.dash || null, opacity: opts.op || 1,
      "stroke-linecap": "round", "stroke-linejoin": "round"
    }, fr.svg);
    if (opts.dots !== false) {
      vals.forEach(function (v, i) {
        if (v == null) return;
        var c = el("circle", { cx: fr.X(i), cy: fr.Y(v), r: opts.r || 3.2, fill: col,
          opacity: opts.op || 1 }, fr.svg);
        if (opts.tipFor) hover(c, opts.tipFor(i, v));
      });
    }
  }
  function legend(fr, items, x, y) {
    var g = el("g", {}, fr.svg), dy = 0;
    items.forEach(function (it) {
      el("line", { x1: x, x2: x + 16, y1: y + dy, y2: y + dy, stroke: it[1],
        "stroke-width": 2.2, "stroke-dasharray": it[2] || null }, g);
      txt(g, x + 21, y + dy + 3.5, it[0], { "font-size": 10.5, fill: C["ink-2"] });
      dy += 14;
    });
  }

  /* The same legend, in whichever corner the data leaves free.

     series is an array of value arrays, in the same index space the chart was drawn in;
     nulls are skipped. Each segment is sampled at its ends and its midpoint, so a line
     crossing a corner is caught even when no plotted point sits there. */
  function autoLegend(fr, items, series) {
    var W = 21 + 6 * Math.max.apply(null, items.map(function (i) { return i[0].length; })) + 8;
    var H = items.length * 14;
    var pad = 7;
    var corners = [
      [fr.m.l + 14, fr.m.t + 12],                             // top left
      [fr.m.l + fr.iw - W, fr.m.t + 12],                      // top right
      [fr.m.l + fr.iw - W, fr.m.t + fr.ih - H - 4],           // bottom right
      [fr.m.l + 14, fr.m.t + fr.ih - H - 4]                   // bottom left
    ];
    var pts = [];
    (series || []).forEach(function (vals) {
      if (!vals) return;
      for (var i = 0; i < vals.length; i++) {
        if (vals[i] == null) continue;
        pts.push([fr.X(i), fr.Y(vals[i])]);
        if (i + 1 < vals.length && vals[i + 1] != null) {
          pts.push([(fr.X(i) + fr.X(i + 1)) / 2, (fr.Y(vals[i]) + fr.Y(vals[i + 1])) / 2]);
        }
      }
    });
    var best = corners[0], bestHits = Infinity;
    corners.forEach(function (c) {
      var hits = 0;
      for (var k = 0; k < pts.length; k++) {
        if (pts[k][0] > c[0] - pad && pts[k][0] < c[0] + W + pad
            && pts[k][1] > c[1] - pad - 6 && pts[k][1] < c[1] + H + pad) hits++;
      }
      if (hits < bestHits) { bestHits = hits; best = c; }
    });
    legend(fr, items, best[0], best[1]);
  }
  function hbars(host, rows, opts) {
    host.innerHTML = "";
    opts = opts || {};
    var W = 560, rowH = 30, padT = 6, padB = 20;
    var H = padT + rows.length * rowH + padB;
    var s = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    var lw = opts.labelWidth || 232, bw = W - lw - 62;
    var max = opts.max || Math.max.apply(null, rows.map(function (r) { return r.v; })) * 1.06;
    rows.forEach(function (r, i) {
      var y = padT + i * rowH;
      txt(s, lw - 10, y + rowH / 2 + 3.5, r.label, {
        "text-anchor": "end", "font-size": 12,
        fill: r.hi ? C.ink : C["ink-2"], "font-weight": r.hi ? 650 : 400
      });
      var w = Math.max(1.5, bw * r.v / max);
      var bar = el("rect", { x: lw, y: y + 6, width: w, height: rowH - 15, rx: 3,
        fill: r.col, opacity: r.hi ? 1 : .82 }, s);
      if (r.tip) hover(bar, r.tip);
      txt(s, lw + w + 7, y + rowH / 2 + 3.5, f(r.v), {
        "font-size": 11.5, "font-family": "JetBrains Mono, monospace", fill: C["ink-2"] });
    });
    return s;
  }

  // ═══════════════════════════════════════════════════ state
  var S = {
    hIdx: 0, bIdx: 1, yoke: "self", round: "rtne", contain: "group32",
    evalIdx: 0, ctxIdx: 0
  };
  var MODEL = "0.8B";
  var BITS = [3, 4, 5, 6, 8];
  var CTXS = [2048, 4096, 8192, 32768, 65536, 131072];

  // ═══════════════════════════════════════════════════ headline numbers
  function stats() {
    var fr = D.frontier[MODEL];
    var yk = D.yoke[MODEL];
    var last = D.h32k.ckpts.length - 1;
    var vz = D.h32k.vs_zero[last];
    var items = [
      ["INT5", "is where the frontier runs: INT4 keeps perfect recall at H=0 and none of it by "
        + ktick(1024 * 2), ""],
      [bigNum(yk.dyn_self.norm) + "×", "state-norm inflation under per-token dynamic INT4, against "
        + f(yk.unquant.norm, 0) + " unquantized", "warn"],
      [f(yk.recovered.delta, 1) + " of " + f(yk.dyn_self.excess, 1), "nats recovered by cutting the "
        + "feedback — the runaway is not the whole failure", "dim"],
      [f(D.bits.surviving, 1) + "%", "of the yoked residual survives at INT8: what is left is resolution",
        ""],
      [(vz.delta >= 0 ? "+" : "") + f(vz.delta, 2), "nats — group-wise INT4 against clearing the state "
        + "at 32K (t=" + f(vz.t) + ", n=" + vz.n + ")", "warn"]
    ];
    var root = $("stats"); root.innerHTML = "";
    items.forEach(function (it) {
      root.appendChild(h("div", "stat " + it[2], "<b>" + it[0] + "</b><span>" + it[1] + "</span>"));
    });
  }

  function tldr() {
    var yk = D.yoke[MODEL];
    $("tldr").innerHTML =
      "Quantizing a recurrent state puts the quantizer <strong>inside the recurrence</strong>. "
      + "Per-token dynamic scaling then reads its scale from a state its own error is inflating, and "
      + "the two climb together: the state norm reaches <strong>" + bigNum(yk.dyn_self.norm)
      + "</strong> against <strong>" + f(yk.unquant.norm, 0) + "</strong> unquantized. Yoking that "
      + "scale to a full-precision reference removes the runaway completely and recovers only "
      + "<strong>" + f(yk.recovered.delta, 1) + " of " + f(yk.dyn_self.excess, 1) + " nats</strong>, so "
      + "numerical blow-up and functional damage are separable. The rest is resolution, not outliers — "
      + "and out at 32K, the best INT4 policy is worth no more than throwing the state away.";
  }

  // ═══════════════════════════════════════════════════ 1. the phenomenon
  function phenomenon() {
    var fr = D.frontier[MODEL], bits = ["INT8", "INT5", "INT4", "INT3"];
    var Hs = Object.keys(fr.INT4).map(Number).sort(function (a, b) { return a - b; });
    var col = { INT8: C.keep, INT5: C.mid, INT4: C.fail, INT3: C["fail-2"] };

    var f1 = frame($("c-recall"), {
      xs: Hs.map(ktick), y0: 0, y1: 1.08, yticks: [0, .25, .5, .75, 1],
      yfmt: function (v) { return v.toFixed(2); }, ylab: "NIAH recall", h: 236
    });
    // the unquantized reference
    el("line", { x1: f1.m.l, x2: f1.W - f1.m.r, y1: f1.Y(1), y2: f1.Y(1),
      stroke: C["ink-3"], "stroke-width": 1, "stroke-dasharray": "3 3" }, f1.svg);
    bits.forEach(function (b) {
      var vals = Hs.map(function (hh) { return fr[b][String(hh)]; });
      // nothing is measured between H=0 and H=1024: draw that span faintly
      path(f1, [vals[0], vals[1]].concat(Hs.slice(2).map(function () { return null; })),
        col[b], { dash: "2 3", op: .45, w: 1.2, dots: false });
      path(f1, [null].concat(vals.slice(1)), col[b], {
        w: 1.9, tipFor: function (i, v) {
          return "<b>" + b + "</b> at H=" + ktick(Hs[i]) + "<br>recall " + f(v, 3)
            + " over " + fr.seeds + " seeds";
        }
      });
      el("circle", { cx: f1.X(0), cy: f1.Y(vals[0]), r: 3.2, fill: col[b] }, f1.svg);
    });
    autoLegend(f1, bits.map(function (b) { return [b, col[b]]; }),
      bits.map(function (b) { return Hs.map(function (hh) { return fr[b][String(hh)]; }); })
        .concat([Hs.map(function () { return 1; })]));
    // the highlighted horizon
    el("line", { x1: f1.X(S.hIdx), x2: f1.X(S.hIdx), y1: f1.m.t, y2: f1.m.t + f1.ih,
      stroke: C.brand, "stroke-width": 1.4, opacity: .32 }, f1.svg);

    var rw = D.runaway[MODEL], pol = ["unquant", "dynamic4", "static4", "group32"];
    var pcol = { unquant: C["ink-3"], dynamic4: C.fail, static4: C.mid, group32: C.keep };
    var plab = { unquant: "unquantized", dynamic4: "dynamic INT4", static4: "static INT4",
      group32: "group-wise INT4" };
    var f2 = frame($("c-norm"), {
      xs: rw.Hs.map(ktick), log: true, y0: 8, y1: 2e5,
      yticks: [10, 100, 1000, 10000, 100000],
      yfmt: function (v) { return v >= 1000 ? (v / 1000) + "K" : String(v); },
      ylab: "state norm ||S||", h: 236
    });
    pol.forEach(function (p) {
      if (!rw.policies[p]) return;
      path(f2, rw.policies[p].norm, pcol[p], {
        w: p === "unquant" ? 1.3 : 1.9, dash: p === "unquant" ? "3 3" : null,
        tipFor: function (i, v) {
          return "<b>" + plab[p] + "</b> at H=" + ktick(rw.Hs[i]) + "<br>norm " + bigNum(v)
            + "<br>excess NLL " + f(rw.policies[p].excess[i]) + " nats";
        }
      });
    });
    autoLegend(f2, pol.filter(function (p) { return rw.policies[p]; })
      .map(function (p) { return [plab[p], pcol[p], p === "unquant" ? "3 3" : null]; }),
      pol.filter(function (p) { return rw.policies[p]; })
        .map(function (p) { return rw.policies[p].norm; }));

    // the readout under the recall panel
    var H = Hs[S.hIdx];
    var alive = bits.filter(function (b) { return fr[b][String(H)] >= 1; });
    var dead = bits.filter(function (b) { return fr[b][String(H)] === 0; });
    var msg = "At <b>H=" + ktick(H) + "</b>: ";
    if (H === 0) {
      msg += "every bit-width recalls the planted fact perfectly. This is the horizon almost "
        + "every quantization paper reports.";
    } else {
      msg += (alive.length ? "<b>" + alive.join(", ") + "</b> still perfect" : "nothing is intact")
        + (dead.length ? ", <b>" + dead.join(", ") + "</b> at zero" : "")
        + ". " + (H >= 4096
          ? "Only INT8 has survived this far; INT5 is on its way down."
          : "The same INT4 that was perfect at H=0 is already gone.");
    }
    $("recall-read").innerHTML = msg;
  }

  // ── summoning ─────────────────────────────────────────────────────────────
  // Only on the transition into the yoked arm. render() runs on every slider step, and
  // re-summoning the reference each time a reader drags something elsewhere would be
  // noise rather than feedback.
  var prevYoke = null;
  // Animations are a nicety; a host without requestAnimationFrame should still get
  // the page, so every use goes through this.
  var raf = (window.requestAnimationFrame || function (f) { return setTimeout(f, 16); })
    .bind(window);
  var slowOK = !(window.matchMedia
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  function drawIn(node, ms, delay) {
    if (!slowOK) return;
    var len;
    try { len = node.getTotalLength(); } catch (e) { return; }
    if (!len) return;
    node.style.strokeDasharray = len;
    node.style.strokeDashoffset = len;
    node.style.transition = "stroke-dashoffset " + ms + "ms cubic-bezier(.22,.61,.36,1) "
      + (delay || 0) + "ms";
    raf(function () { node.style.strokeDashoffset = "0"; });
  }

  function fadeIn(node, delay) {
    if (!slowOK) return;
    node.style.opacity = "0";
    node.style.transition = "opacity .34s ease " + (delay || 0) + "ms";
    raf(function () { node.style.opacity = ""; });
  }

  function spark(svg, x, y) {
    if (!slowOK) return;
    var c = el("circle", { cx: x, cy: y, r: 3, fill: "none", stroke: C.fail,
      "stroke-width": 2 }, svg);
    c.style.transition = "r .5s cubic-bezier(.16,1,.3,1), opacity .5s ease";
    raf(function () {
      c.setAttribute("r", "26");
      c.style.opacity = "0";
    });
    setTimeout(function () { if (c.parentNode) c.parentNode.removeChild(c); }, 700);
  }

  // ═══════════════════════════════════════════════════ 2. the loop
  function loop() {
    var yk = D.yoke[MODEL];
    var on = S.yoke === "self";
    var arm = on ? yk.dyn_self : yk.dyn_yoked;
    var host = $("c-loop"); host.innerHTML = "";
    var W = 560, H = 258, OX = 96;      // the left column is the reference's
    var s = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    var BW = 150, BH = 50;

    function box(x, y, title, sub, col, dim) {
      var g = el("g", { opacity: dim ? .34 : 1 }, s);
      el("rect", { x: x, y: y, width: BW, height: BH, rx: 9, fill: "none",
        stroke: col, "stroke-width": 1.6 }, g);
      txt(g, x + BW / 2, y + (sub ? BH / 2 - 3 : BH / 2 + 4), title,
        { "text-anchor": "middle", "font-size": 12, fill: C.ink, "font-weight": 600 });
      if (sub) txt(g, x + BW / 2, y + BH / 2 + 13, sub, {
        "text-anchor": "middle", "font-size": 10.5, fill: C["ink-2"],
        "font-family": "JetBrains Mono, monospace" });
      return g;
    }

    var defs = el("defs", {}, s);
    [["ahf", C.fail], ["ahn", C["ink-3"]], ["ahb", C.brand]].forEach(function (a) {
      var mk = el("marker", { id: a[0], viewBox: "0 0 10 10", refX: 8, refY: 5,
        markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
      el("path", { d: "M 0 1 L 9 5 L 0 9 z", fill: a[1] }, mk);
    });

    function link(d, col, dash, lab, lx, ly, dim, flow) {
      var g = el("g", { opacity: dim ? .3 : 1 }, s);
      var head = col === C.fail ? "f" : (col === C.brand ? "b" : "n");
      var p = el("path", { d: d, stroke: col, "stroke-width": 1.6, fill: "none",
        "stroke-dasharray": dash || null,
        "class": flow ? "flow" : null,
        "marker-end": "url(#ah" + head + ")" }, g);
      if (lab) txt(g, lx, ly, lab, { "text-anchor": "middle", "font-size": 10,
        fill: col, "font-weight": 600 });
      return { g: g, path: p };
    }

    box(OX + 28, 42, "recurrent state", "S", C.ink, false);
    box(OX + 282, 42, "quantizer scale", "s = max|S| / qmax",
        on ? C.fail : C.brand, !on);
    box(OX + 282, 174, "quantization error", "", on ? C.fail : C.brand, false);
    box(OX + 28, 174, "the state inflates", "", on ? C.fail : C["ink-3"], !on);

    // the closed loop, in reading order
    link("M " + (OX + 180) + " 67 L " + (OX + 278) + " 67", on ? C.fail : C["ink-3"],
         on ? null : "3 3", "measured from", OX + 229, 60, !on, on);
    link("M " + (OX + 357) + " 96 L " + (OX + 357) + " 170", on ? C.fail : C.brand,
         null, null, 0, 0, false, on);
    link("M " + (OX + 278) + " 199 L " + (OX + 182) + " 199", on ? C.fail : C["ink-3"],
         on ? null : "3 3", "inflates", OX + 230, 192, !on, on);
    link("M " + (OX + 103) + " 170 L " + (OX + 103) + " 98", on ? C.fail : C["ink-3"],
         on ? null : "3 3", null, 0, 0, !on, on);

    if (on) {
      // Nothing is summoned yet, but the column is already its own, so a faint outline
      // stands there: it keeps the figure from sitting off-centre and says where the
      // reference will appear.
      var ghost = el("g", { transform: "translate(52,122)", opacity: .1 }, s);
      el("path", {
        d: "M0 -40 l0 9 M-13 -31 C-23 -22 -25 -7 -18 5 L-23 33 L23 33 L18 5 "
         + "C25 -7 23 -22 13 -31 Z M0 33 L0 50 M-14 52 L14 52",
        fill: "none", stroke: C.ink, "stroke-width": 2, "stroke-linejoin": "round",
        "stroke-dasharray": "3 5"
      }, ghost);
    }

    if (!on) {
      // The reference is its own trajectory, running beside the quantized one for the
      // whole of the decode, so it is drawn as its own object rather than as a label on
      // an arrow. Dashed, because nothing in it is ever quantized.
      var summoned = prevYoke === "self";
      var body = el("rect", { x: 24, y: 86, width: 58, height: 74, rx: 8, fill: "none",
        stroke: C.brand, "stroke-width": 2.2, "stroke-dasharray": "5 3.5" }, s);
      var stand = el("path", { d: "M53 160 L53 176 M39 178 L67 178", fill: "none",
        stroke: C.brand, "stroke-width": 2.2, "stroke-linecap": "round" }, s);
      var cap = txt(s, 53, 196, "full-precision", { "text-anchor": "middle",
        "font-size": 10, fill: C.brand, "font-weight": 700 });
      var cap2 = txt(s, 53, 208, "reference", { "text-anchor": "middle",
        "font-size": 10, fill: C.brand, "font-weight": 700 });
      // it feeds the scale over the top, clear of the loop it replaced
      var feed = link("M 53 82 L 53 18 L " + (OX + 357) + " 18 L " + (OX + 357) + " 36",
                      C.brand, null, "scale read from the reference", 300, 12, false, false);
      // and the loop is cut where it actually breaks
      var slash = el("line", { x1: OX + 88, y1: 146, x2: OX + 118, y2: 122,
        stroke: C.brand, "stroke-width": 2.2, "stroke-linecap": "round" }, s);
      var cut = txt(s, OX + 150, 140, "loop cut", { "text-anchor": "start",
        "font-size": 10.5, fill: C.brand, "font-weight": 700 });
      if (summoned) {
        drawIn(body, 520, 60);
        drawIn(stand, 240, 460);
        fadeIn(cap, 520); fadeIn(cap2, 560);
        drawIn(feed.path, 600, 620);
        fadeIn(feed.g, 620);
        drawIn(slash, 240, 330);
        fadeIn(cut, 440);
        spark(s, OX + 103, 134);
      }
    }
    prevYoke = S.yoke;

    $("ro-norm").innerHTML = "<div class='lbl'>state norm</div><div class='big' style='color:"
      + (on ? C.fail : C.keep) + "'>" + bigNum(arm.norm) + "</div>"
      + "<div class='sub'>unquantized reference " + f(yk.unquant.norm, 1) + "</div>";
    $("ro-excess").innerHTML = "<div class='lbl'>excess NLL</div><div class='big' style='color:"
      + C.fail + "'>" + f(arm.excess) + "</div>"
      + "<div class='sub'>nats over the unquantized model, " + yk.n + " articles</div>";
    $("ro-verdict").innerHTML = on
      ? "This is the loop running. The scale is read from a state that quantization error has "
        + "already inflated, so both climb together and the norm ends "
        + bigNum(arm.norm / yk.unquant.norm) + "× above the reference."
      : "The runaway is <b>gone</b> — the norm is back to " + f(arm.norm, 1) + ", against "
        + f(yk.unquant.norm, 1) + " unquantized. And only <b>" + f(yk.recovered.delta, 2)
        + " of " + f(yk.dyn_self.excess, 2) + " nats</b> came back with it (t="
        + f(yk.recovered.t, 1) + ", n=" + yk.recovered.n + "). The feedback caused the numerical "
        + "explosion, and the numerical explosion was not the damage.";
  }

  // ═══════════════════════════════════════════════════ 3. containment
  var KIND = {
    none: "uncontained",
    ceiling: "a ceiling in value",
    interruption: "an interruption in time",
    confinement: "a confinement in space"
  };
  function contain() {
    var rows = D.contain.rows;
    var cards = $("contain-cards"); cards.innerHTML = "";
    rows.forEach(function (r) {
      var b = h("button", "ic");
      b.type = "button";
      b.setAttribute("data-key", r.key);
      b.setAttribute("aria-pressed", String(r.key === S.contain));
      b.innerHTML = "<div class='k'>" + KIND[r.kind] + "</div><div class='n'>" + r.label
        + "</div><div class='v'>" + f(r.excess) + " nats</div>";
      b.addEventListener("click", function () { S.contain = r.key; render(); });
      cards.appendChild(b);
      if (window.LQFX) window.LQFX.observe(b);
    });

    var max = Math.max.apply(null, rows.map(function (r) { return r.excess; }));
    hbars($("c-contain"), rows.map(function (r) {
      return {
        label: r.label, v: r.excess, hi: r.key === S.contain,
        col: r.kind === "none" ? C.fail : (r.key === "group32" ? C.keep : C.mid),
        tip: "<b>" + r.label + "</b><br>excess NLL " + f(r.excess) + " nats, state norm "
          + bigNum(r.norm) + "<br>vs " + (r.vs === "dynamic4" ? "dynamic INT4" : "capped dynamic")
          + ": Δ=" + f(r.delta) + ", t=" + f(r.t, 2) + ", n=" + r.n
      };
    }), { max: max * 1.08 });

    var sel = rows.filter(function (r) { return r.key === S.contain; })[0];
    var extra = "";
    if (sel.key === "capped_esc") {
      extra = "<p>Against the cap alone this moves the loss by " + f(Math.abs(sel.delta), 2)
        + " nats (t=" + f(Math.abs(sel.t), 2) + "). Keeping the outliers <em>exactly</em> adds "
        + "nothing on top of a ceiling — which is what identifies the ceiling, not the outliers, "
        + "as the operative part.</p>";
    } else if (sel.key === "group32") {
      extra = "<p>The best of the three, and nobody invented it for this problem. A per-group "
        + "scale confines an outlier's inflation to its own group: containment on the channel axis.</p>";
    } else if (sel.key !== "dynamic4") {
      extra = "<p>Against per-token dynamic INT4: Δ=" + f(sel.delta) + " nats, t=" + f(sel.t, 2)
        + " over " + sel.n + " paired articles.</p>";
    }
    $("contain-why").innerHTML = "<b>" + sel.label + "</b> — " + KIND[sel.kind] + ".<p>" + sel.why
      + "</p>" + extra
      + "<p class='note'>A fixed range, a periodic reset and a per-group scale are the same "
      + "intervention along three different axes. That is why they all work, and it is why the "
      + "schemes below, which spend their information locating outliers instead, do not.</p>";

    // locating outliers, at matched rate
    var loc = D.contain.locate.filter(function (r) {
      return ["refine_top", "refine_rand0", "range_top", "range_rand0", "escape_exact",
        "static4", "static5"].indexOf(r.key) >= 0;
    });
    var NAME = { static4: "persistent static INT4 (reference)",
      static5: "persistent static INT5 (reference)" };
    hbars($("c-locate"), loc.map(function (r) {
      var targeted = r.key === "refine_top" || r.key === "range_top" || r.key === "escape_exact";
      return {
        label: NAME[r.label] || NAME[r.key] || r.label, v: r.nll,
        hi: targeted, col: targeted ? C.fail : C.grey,
        tip: "<b>" + (NAME[r.key] || r.label) + "</b><br>teacher-forced NLL " + f(r.nll, 3)
          + "<br>modeled rate " + f(r.bits, 2) + " bits/element<br>NIAH recall " + f(r.recall, 3)
      };
    }), { labelWidth: 268 });
  }

  // ═══════════════════════════════════════════════════ 4. bits
  function bits() {
    var yk = D.bits.yoked;
    var vals = BITS.map(function (b) { return yk[String(b)] ? yk[String(b)].excess : null; });
    var srOn = S.round === "sr";
    var fr = frame($("c-bits"), {
      xs: BITS.map(function (b) { return "INT" + b; }), y0: 0, y1: 12,
      yticks: [0, 2, 4, 6, 8, 10, 12], ylab: "excess NLL (nats)", h: 236
    });
    path(fr, vals, C.keep, {
      w: 2.1, tipFor: function (i, v) {
        return "<b>INT" + BITS[i] + ", scale yoked</b><br>excess NLL " + f(v)
          + " nats<br>state norm " + bigNum(yk[String(BITS[i])].norm);
      }
    });
    if (srOn) {
      var srv = BITS.map(function (b) {
        var e = D.bits.sr["yoke" + b + "_sr"];
        return e ? e.excess : null;
      });
      path(fr, srv, C.fail, {
        w: 2.1, dash: "5 3", bridge: true, tipFor: function (i, v) {
          return "<b>INT" + BITS[i] + ", stochastic rounding</b><br>excess NLL " + f(v) + " nats";
        }
      });
      autoLegend(fr, [["round to nearest", C.keep],
        ["stochastic rounding (INT4, INT8)", C.fail, "5 3"]], [vals, srv]);
    }
    el("line", { x1: fr.X(S.bIdx), x2: fr.X(S.bIdx), y1: fr.m.t, y2: fr.m.t + fr.ih,
      stroke: C.brand, "stroke-width": 1.4, opacity: .32 }, fr.svg);

    var b = BITS[S.bIdx], e = yk[String(b)];
    $("bits-read").innerHTML = "<b>INT" + b + "</b>, scale yoked: <b>" + f(e.excess)
      + " nats</b>, state norm " + f(e.norm, 1) + " against " + f(D.yoke[MODEL].unquant.norm, 1)
      + " unquantized.";
    $("bits-why").innerHTML =
      "<p>An accumulation floor would not thin out with bit-width. This does: from <b>"
      + f(yk["3"].excess) + "</b> nats at INT3 to <b>" + f(yk["8"].excess) + "</b> at INT8, "
      + "monotone in <b>" + D.bits.monotone + " of " + D.bits.n + "</b> articles, with the "
      + "INT4→INT8 gap at Δ=" + f(D.bits.gap48.delta) + " (t=" + f(D.bits.gap48.t, 1) + "). Only <b>"
      + f(D.bits.surviving, 1) + "%</b> of the INT4 residual is still there at INT8.</p>"
      + "<p><b>More bits buy it back. A cleverer arrangement of the same bits does not.</b></p>"
      + "<p class='note'>Both arms hold the scale yoked to full precision, so the feedback cannot "
      + "confound the sweep. The INT4 arm reproduces the published value to the last digit, which "
      + "is what licenses reading the rest of the curve.</p>";

    var s4 = D.bits.sr_cost.int4, s8 = D.bits.sr_cost.int8;
    $("sr-read").innerHTML = srOn
      ? "Nemotron 3 ships stochastic rounding for the SSM cache, and the paper had never tested it. "
        + "It is unbiased, and it makes everything worse: <b>+" + f(s4.delta) + " nats</b> at yoked "
        + "INT4 (t=" + f(s4.t, 1) + "), falling to <b>+" + f(s8.delta) + "</b> at INT8 (t="
        + f(s8.t, 1) + "). The reason is mundane — unbiased or not, its per-step error is √2 larger, "
        + "and a recurrence amplifies magnitude, not sign."
      : "The curve above rounds to nearest. Switch to stochastic rounding to see what happens to a "
        + "recurrence when each step's error gets larger but unbiased.";
  }

  // ═══════════════════════════════════════════════════ 5. the horizon trap
  var H32_POL = [
    ["group32", "group-wise INT4", "keep"],
    ["static4", "persistent static INT4", "mid"],
    ["dynamic4", "per-token dynamic INT4", "fail"],
    ["group32_int5", "group-wise INT5", "keep"],
    ["group32_int8", "group-wise INT8", "keep"]
  ];
  function survivors() {
    var last = D.h32k.ckpts.length - 1, P = D.h32k.policies, z = P.zero.excess[last];
    var keep = H32_POL.filter(function (p) {
      return P[p[0]] && z - P[p[0]].excess[last] > 0.5;
    }).map(function (p) { return p[1]; });
    if (!keep.length) return "Nothing here stays clearly ahead of the control that far out";
    return (keep.length > 1 ? "The policies still clearly ahead of the control at "
      : "The only policy still clearly ahead of the control at ") + ktick(D.h32k.ckpts[last])
      + " is " + keep.join(", ").replace(/, ([^,]*)$/, " and $1")
      + (keep.length > 1 ? "" : "");
  }

  function horizon() {
    var ck = D.h32k.ckpts, P = D.h32k.policies;
    var seg = $("evalseg");
    if (!seg.childElementCount) {
      ck.forEach(function (hh, i) {
        var b = h("button", null, "certify at " + ktick(hh));
        b.type = "button";
        b.addEventListener("click", function () { S.evalIdx = i; render(); });
        seg.appendChild(b);
      });
    }
    Array.prototype.forEach.call(seg.children, function (b, i) {
      b.setAttribute("aria-pressed", String(i === S.evalIdx));
    });

    var i = S.evalIdx, zero = P.zero.excess[i];
    var vd = $("verdicts"); vd.innerHTML = "";
    H32_POL.forEach(function (p) {
      if (!P[p[0]]) return;
      var v = P[p[0]].excess[i];
      var margin = zero - v;
      var cls = margin > 0.5 ? "ok" : (margin > 0.05 ? "" : "bad");
      var says = margin > 0.5 ? "clearly better than clearing the state"
        : (margin > 0.05 ? "barely better than clearing the state"
          : "<b>no better than clearing the state</b>");
      var card = h("div", "vd " + cls);
      card.innerHTML = "<div class='n'>" + p[1] + "</div><div class='v'>" + f(v)
        + " nats over unquantized — " + says + "</div>"
        + "<div class='m'>zero-state control " + f(zero) + " · margin "
        + (Math.abs(margin) < 0.005 ? "0.00" : (margin > 0 ? "+" : "") + f(margin))
        + " · H=" + ktick(ck[i]) + "</div>";
      vd.appendChild(card);
    });

    var col = { keep: C.keep, mid: C.mid, fail: C.fail };
    var fr = frame($("c-h32k"), {
      w: 760, h: 300, xs: ck.map(ktick), y0: 0, y1: 10,
      yticks: [0, 2, 4, 6, 8, 10], ylab: "excess NLL (nats)"
    });
    el("path", {
      d: ck.map(function (_, k) { return (k ? "L " : "M ") + fr.X(k) + " " + fr.Y(P.zero.excess[k]); }).join(" "),
      fill: "none", stroke: C["ink-3"], "stroke-width": 1.5, "stroke-dasharray": "5 4"
    }, fr.svg);
    txt(fr.svg, fr.X(0) + 6, fr.Y(P.zero.excess[0]) - 8, "clear the state (zero control)",
      { "font-size": 10.5, fill: C["ink-3"] });
    H32_POL.forEach(function (p) {
      if (!P[p[0]]) return;
      path(fr, P[p[0]].excess, col[p[2]], {
        w: p[0] === "group32" ? 2.4 : 1.7,
        dash: p[0].indexOf("int") > 0 ? "3 3" : null,
        op: p[0].indexOf("int") > 0 ? .7 : 1,
        tipFor: function (k, v) {
          return "<b>" + p[1] + "</b> at H=" + ktick(ck[k]) + "<br>excess NLL " + f(v)
            + " nats<br>zero-state control " + f(P.zero.excess[k]);
        }
      });
    });
    autoLegend(fr, H32_POL.filter(function (p) { return P[p[0]]; })
      .map(function (p) { return [p[1], col[p[2]], p[0].indexOf("int") > 0 ? "3 3" : null]; }),
      H32_POL.filter(function (p) { return P[p[0]]; })
        .map(function (p) { return P[p[0]].excess; }).concat([P.zero.excess]));
    el("line", { x1: fr.X(i), x2: fr.X(i), y1: fr.m.t, y2: fr.m.t + fr.ih,
      stroke: C.brand, "stroke-width": 1.4, opacity: .32 }, fr.svg);

    var last = ck.length - 1, vz = D.h32k.vs_zero[last];
    var sg = D.h32k.contrasts["static4-group32"];
    var atEnd = i === last;
    $("h32k-read").innerHTML = "Every curve here is one continuous run, scored at each checkpoint on "
      + D.h32k.n_docs + " long articles. "
      + (atEnd
        ? "This is the far end, and it is where the recommendation dissolves: "
        : "At <b>H=" + ktick(ck[i]) + "</b> the ranking still reads like a recommendation. Out at "
          + ktick(ck[last]) + " it does not: ")
      + "group-wise INT4 sits <b>" + f(vz.delta, 2) + " nats</b> from the zero-state control (t="
      + f(vz.t, 2) + ", n=" + vz.n + ") — statistically indistinguishable from keeping no state "
      + "at all. The policy distinction the paper is built on is still real there (static vs "
      + "group-wise Δ=" + f(sg.delta) + ", t=" + f(sg.t, 1) + "), but it has become a distinction "
      + "between two unusable options. " + survivors() + ".";
  }

  // ═══════════════════════════════════════════════════ 6. how much cache is this
  function memory() {
    var host = $("c-mem"); host.innerHTML = "";
    var L = CTXS[S.ctxIdx];
    var names = Object.keys(D.memory);
    var W = 760, rowH = 74, H = names.length * rowH + 34;
    var s = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    var lw = 176, bw = W - lw - 108;
    var shares = {};
    // the widest total across the whole slider range, so the bars stay comparable
    var maxTotal = 0;
    names.forEach(function (n) {
      var m = D.memory[n];
      var st = m.linear_layers * m.state_elems * m.state_bytes / 1048576;
      CTXS.forEach(function (c) {
        var kv = 2 * m.attn_layers * m.kv_heads * m.head_dim * m.kv_bytes * c / 1048576;
        maxTotal = Math.max(maxTotal, st + kv);
      });
    });
    names.forEach(function (n, i) {
      var m = D.memory[n];
      var st = m.linear_layers * m.state_elems * m.state_bytes / 1048576;
      var kvPerTok = 2 * m.attn_layers * m.kv_heads * m.head_dim * m.kv_bytes;
      var kv = kvPerTok * L / 1048576;
      var share = (st + kv) > 0 ? st / (st + kv) : 1;
      shares[n] = share;
      var y = 16 + i * rowH;
      txt(s, lw - 12, y + 24, n, { "text-anchor": "end", "font-size": 13,
        "font-weight": 600, fill: C.ink });
      txt(s, lw - 12, y + 40, m.kind + ", " + m.linear_layers + " recurrent layers",
        { "text-anchor": "end", "font-size": 11, fill: C["ink-3"] });
      var wS = Math.max(2, bw * st / maxTotal), wK = bw * kv / maxTotal;
      var r1 = el("rect", { x: lw, y: y + 8, width: wS, height: 30, rx: 4, fill: C.brand }, s);
      hover(r1, "<b>recurrent state</b><br>" + f(st, 1) + " MB, fixed — "
        + m.linear_layers + " layers × " + m.state_elems.toLocaleString("en-US")
        + " elements × " + m.state_bytes + " bytes");
      if (wK > 0.5) {
        var r2 = el("rect", { x: lw + wS, y: y + 8, width: wK, height: 30, rx: 4,
          fill: C.grey }, s);
        hover(r2, "<b>attention KV cache</b><br>" + f(kv, 1) + " MB at " + ktick(L)
          + " tokens (" + kvPerTok + " bytes/token)");
      }
      txt(s, lw + wS + wK + 10, y + 27, (100 * share).toFixed(1) + "% state",
        { "font-size": 12, "font-family": "JetBrains Mono, monospace",
          fill: share > .5 ? C.brand : C["ink-2"], "font-weight": 650 });
    });
    txt(s, lw, H - 4, "■ recurrent state", { "font-size": 10.5, fill: C.brand });
    txt(s, lw + 108, H - 4, "■ attention KV cache", { "font-size": 10.5, fill: C["ink-3"] });

    var hyb = shares["Qwen3.5-0.8B"];
    $("mem-read").innerHTML = "At <b>" + ktick(L) + "</b> tokens the recurrent state is <b>"
      + (100 * hyb).toFixed(1) + "%</b> of the hybrid's decode cache"
      + (L >= 32768 ? " — so at the horizons this paper is about, quantizing it saves little in "
        + "<em>this</em> architecture." : ".")
      + " In a pure SSM it is <b>100%</b> at every length, because there is no KV cache beside it. "
      + "That is where the case for state quantization is strongest, and it is the honest scope of "
      + "the result.";
  }

  // ═══════════════════════════════════════════════════ does it transfer?
  /* One row per model, one shared x scale, and each model's zero-state control drawn as a
     dashed rule inside its own row. The reversal is the thing to see, so static and
     group-wise keep their colours across rows: on the GDN rows the gold sits left of the
     control and the blue-green sits right of it, and on the Mamba row they swap. */
  function crossChart(host, cols) {
    host.innerHTML = "";
    var W = 560, rowH = 58, padT = 10, lw = 104, bw = W - lw - 56;
    // The legend sits under the bars, so its height is part of the box. Leaving it out
    // is what put the last two entries on top of the paragraph below the figure.
    var LEG = 4, legTop = padT + cols.length * rowH + 8;
    var H = legTop + LEG * 14 + 6;
    var s = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    var max = 0;
    cols.forEach(function (c) {
      c.rows.forEach(function (r) { if (r.excess > max) max = r.excess; });
    });
    max *= 1.08;
    var x = function (v) { return lw + bw * v / max; };
    var SHOW = [["dynamic4", C.fail], ["static4", C.mid], ["group32", C.keep]];

    cols.forEach(function (c, i) {
      var y0 = padT + i * rowH;
      txt(s, lw - 12, y0 + 22, c.model, { "text-anchor": "end", "font-size": 11.5,
        "font-weight": 600, fill: C.ink });
      var ctl = c.rows.filter(function (r) { return r.key === "zero"; })[0];
      // under the bars and their labels: on the SSM row the control falls 25px from the
      // group-wise value and would otherwise be struck through it
      el("line", { x1: x(ctl.excess), x2: x(ctl.excess), y1: y0 + 2, y2: y0 + 50,
        stroke: C["ink-3"], "stroke-width": 1.3, "stroke-dasharray": "3 3" }, s);
      SHOW.forEach(function (p, j) {
        var r = c.rows.filter(function (q) { return q.key === p[0]; })[0];
        var y = y0 + 6 + j * 14;
        var bar = el("rect", { x: lw, y: y, width: Math.max(1.5, x(r.excess) - lw),
          height: 10, rx: 2, fill: p[1], opacity: 0.88 }, s);
        var vs = c.pairs[p[0] + "_vs_zero"];
        hover(bar, "<b>" + c.model + " &middot; " + r.label + "</b><br>excess NLL "
          + f(r.excess) + " nats, state norm " + bigNum(r.norm)
          + "<br>against the zero-state control: &Delta;=" + f(vs.delta)
          + ", t=" + f(vs.t, 2) + ", better in " + vs.wins + "/" + vs.n + " articles");
        txt(s, x(r.excess) + 6, y + 8.5, f(r.excess), { "font-size": 10.5,
          "font-family": "JetBrains Mono, monospace", fill: C["ink-2"] });
      });
    });
    legend({ svg: s },
      [["per-token dynamic", C.fail], ["persistent static", C.mid],
       ["persistent group-wise", C.keep], ["zero-state control", C["ink-3"], "3 3"]],
      lw, legTop);
    return s;
  }

  function cross() {
    var X = D.cross;
    crossChart($("c-cross"), X.cols);
    var g = {};
    X.cols.forEach(function (c) { g[c.model] = c.pairs.group32_vs_static4; });
    var m = g["Mamba-2-1.3B"], a = g["GDN-0.8B"], b = g["GDN-2B"];
    $("cross-read").innerHTML =
      "Group-wise beats a single persistent scale by <b>" + f(Math.abs(a.delta))
      + "</b> nats on GDN-0.8B (t=" + f(Math.abs(a.t), 1) + ", " + a.wins + "/" + a.n
      + " articles) and <b>" + f(Math.abs(b.delta)) + "</b> on GDN-2B (t="
      + f(Math.abs(b.t), 1) + "), so model size is not what decides it. On Mamba-2 it "
      + "<b>loses</b> by " + f(Math.abs(m.delta)) + " nats (t=" + f(Math.abs(m.t), 2)
      + ", ahead in only " + m.wins + "/" + m.n + ") &mdash; while still spending 5.0 "
      + "modeled bits per element against 4.0, so it loses holding the rate advantage. "
      + "The recall frontier transfers too, about twice as far out: INT4 recall is "
      + "<b>" + f(D.mamba_recall.rows.INT4[1]) + "</b> at H=1024, where GDN-0.8B is at "
      + "0.08, and <b>" + f(D.mamba_recall.rows.INT4[2]) + "</b> by H=2048. Past that the "
      + "probe expires: the unquantized model is itself at "
      + f(D.mamba_recall.rows.A0[3]) + " by H=4096 and " + f(D.mamba_recall.rows.A0[4])
      + " at H=8192, so nothing there is informative.";
  }

  // ═══════════════════════════════════════════════════ bibtex
  function bib() {
    $("bib").textContent =
      "@inproceedings{whendynamicscalingfails2026,\n"
      + "  title     = {When Dynamic Scaling Fails: A Scale--State Feedback Governs\n"
      + "               Recurrent-State Quantization in Linear-Attention and SSM LLMs},\n"
      + "  author    = {Jiang, Yuhang},\n"
      + "  booktitle = {NeurIPS 2026 Workshop on Long-Context Foundation Models (LCFM)},\n"
      + "  year      = {2026},\n"
      + "  url       = {https://openreview.net/forum?id=4DvSLh9jTY}\n"
      + "}";
  }

  // ═══════════════════════════════════════════════════ sliders & toggles
  function ticks(host, labels, active) {
    host.innerHTML = "";
    labels.forEach(function (l, i) {
      host.appendChild(h("span", i === active ? "on" : null, l));
    });
  }
  function wireSliders() {
    var Hs = Object.keys(D.frontier[MODEL].INT4).map(Number)
      .sort(function (a, b) { return a - b; });
    var hs = $("hslider");
    hs.max = String(Hs.length - 1);
    hs.addEventListener("input", function () { S.hIdx = +hs.value; render(); });

    var bs = $("bslider");
    bs.max = String(BITS.length - 1);
    bs.addEventListener("input", function () { S.bIdx = +bs.value; render(); });

    var cs = $("cslider");
    cs.max = String(CTXS.length - 1);
    cs.addEventListener("input", function () { S.ctxIdx = +cs.value; render(); });

    document.querySelectorAll("[data-yoke]").forEach(function (b) {
      b.addEventListener("click", function () { S.yoke = b.getAttribute("data-yoke"); render(); });
    });
    document.querySelectorAll("[data-round]").forEach(function (b) {
      b.addEventListener("click", function () { S.round = b.getAttribute("data-round"); render(); });
    });
  }
  function syncControls() {
    var Hs = Object.keys(D.frontier[MODEL].INT4).map(Number)
      .sort(function (a, b) { return a - b; });
    $("hslider").value = String(S.hIdx);
    $("hval").textContent = "H = " + ktick(Hs[S.hIdx]);
    ticks($("hticks"), Hs.map(ktick), S.hIdx);

    $("bslider").value = String(S.bIdx);
    $("bval").textContent = "INT" + BITS[S.bIdx];
    ticks($("bticks"), BITS.map(function (b) { return "INT" + b; }), S.bIdx);

    $("cslider").value = String(S.ctxIdx);
    $("cval").textContent = ktick(CTXS[S.ctxIdx]) + " tokens";
    ticks($("cticks"), CTXS.map(ktick), S.ctxIdx);

    document.querySelectorAll("[data-yoke]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-yoke") === S.yoke));
    });
    document.querySelectorAll("[data-round]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-round") === S.round));
    });
  }

  function render() {
    readPalette();
    syncControls();
    stats(); tldr();
    phenomenon(); loop(); contain(); bits(); horizon(); memory(); cross();
  }

  // Each chart host is watched once; whichever svg is in it when it first comes into
  // view animates its paths. Later redraws are instant, since redrawing on every slider
  // step would animate the reader's own input back at them.
  function watchCharts() {
    if (!("IntersectionObserver" in window)) return;
    var ids = ["c-recall", "c-norm", "c-contain", "c-locate", "c-bits", "c-h32k", "c-mem",
      "c-cross"];
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        var svg = e.target.querySelector("svg");
        if (svg && window.LQDraw) window.LQDraw(svg);
        io.unobserve(e.target);
      });
    }, { threshold: 0.15 });
    ids.forEach(function (id) { var n = $(id); if (n) io.observe(n); });
  }

  wireSliders();
  bib();
  render();
  setTimeout(watchCharts, 0);
  window.addEventListener("resize", function () { /* SVG scales itself */ });
})();
