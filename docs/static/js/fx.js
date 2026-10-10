/* The quiet motion: a thread across the top that follows the scroll, sections arriving
   rather than appearing, and a chart drawing itself in the first time it is seen.

   Nothing here is a set piece. Under prefers-reduced-motion the reveals are immediate and
   nothing draws itself in. */
(function () {
  "use strict";

  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  var TAU = Math.PI * 2;
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function rgba(hex, a) {
    var h = hex.replace("#", "");
    var n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  // ── the scroll thread across the top ──────────────────────────────────────
  (function thread() {
    var bar = document.createElement("div");
    bar.id = "spool";
    bar.setAttribute("aria-hidden", "true");
    document.body.appendChild(bar);
    function set() {
      var h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = "scaleX(" + (h > 0 ? window.scrollY / h : 0) + ")";
    }
    set();
    window.addEventListener("scroll", set, { passive: true });
    window.addEventListener("resize", set);
  })();

  // ── sections arrive rather than appear ────────────────────────────────────
  (function reveal() {
    var targets = document.querySelectorAll(
      ".block .kicker, .block h2, .block .lede, .panel, .card, .stat, .tk, .ic, .vd, .bib");
    if (reduce.matches || !("IntersectionObserver" in window)) {
      Array.prototype.forEach.call(targets, function (n) { n.classList.add("in"); });
      return;
    }
    Array.prototype.forEach.call(targets, function (n) { n.classList.add("rise"); });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add("in");
        io.unobserve(e.target);
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    Array.prototype.forEach.call(targets, function (n) { io.observe(n); });
    // cards rendered later by app.js need observing too
    window.LQFX = { observe: function (n) { n.classList.add("rise"); io.observe(n); } };
  })();

  // ── chart paths draw themselves in, once, when first seen ─────────────────
  window.LQDraw = function (svg) {
    if (reduce.matches || !svg || svg.dataset.drawn) return;
    svg.dataset.drawn = "1";
    var paths = svg.querySelectorAll("path[stroke]");
    Array.prototype.forEach.call(paths, function (p, i) {
      var len;
      try { len = p.getTotalLength(); } catch (e) { return; }
      if (!len || len > 20000) return;
      p.style.strokeDasharray = len;
      p.style.strokeDashoffset = len;
      p.style.transition = "stroke-dashoffset .85s cubic-bezier(.22,.61,.36,1) " + (i * 70) + "ms";
      requestAnimationFrame(function () { p.style.strokeDashoffset = "0"; });
    });
    var dots = svg.querySelectorAll("circle");
    Array.prototype.forEach.call(dots, function (c, i) {
      c.style.opacity = "0";
      c.style.transition = "opacity .3s ease " + (260 + i * 16) + "ms";
      requestAnimationFrame(function () { c.style.opacity = ""; });
    });
  };
})();
