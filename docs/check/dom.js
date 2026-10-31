/* A DOM stub just large enough to run the project page's app.js and catch real
   errors — missing data paths, undefined properties, bad control wiring. It is
   not a browser: it does not lay anything out. It exists because a syntax check
   cannot tell you that D.bits.sr.yoke5_sr is undefined. */
'use strict';
// The real tokens, read from style.css for whichever scheme is active, rather than a
// copy here that drifts. A token the dark block forgets reaches the charts as '', which
// is what a browser would hand them.
const { tokens } = require('./palette.js');
const SCHEME = {
  light: tokens(':root {'),
  dark: tokens(':root[data-theme="dark"]')
};

class El {
  constructor(tag) {
    this.tagName = tag; this.attrs = {}; this.kids = []; this.handlers = {};
    this.style = {}; this._html = ''; this.textContent = '';
    this.classList = {
      _s: new Set(),
      add: (c) => this.classList._s.add(c),
      remove: (c) => this.classList._s.delete(c),
      toggle: (c, on) => on ? this.classList._s.add(c) : this.classList._s.delete(c),
      contains: (c) => this.classList._s.has(c)
    };
  }
  set className(v) { this.attrs.class = v; }
  get className() { return this.attrs.class || ''; }
  set innerHTML(v) { this._html = String(v); if (v === '') this.kids = []; }
  get innerHTML() { return this._html; }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(c) { this.kids.push(c); ALL.push(c); return c; }
  addEventListener(t, fn) { (this.handlers[t] = this.handlers[t] || []).push(fn); }
  fire(t, ev) { (this.handlers[t] || []).forEach((fn) => fn(ev || { target: this })); }
  get children() { return this.kids; }
  get childElementCount() { return this.kids.length; }
  querySelectorAll(sel) { return matchIn(this.kids, sel); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

const ALL = [];
function matchIn(pool, sel) {
  const attr = sel.match(/^\[([a-z-]+)\]$/);
  const out = [];
  const walk = (nodes) => nodes.forEach((n) => {
    if (attr && n.getAttribute(attr[1]) != null) out.push(n);
    else if (!attr && sel.includes('button') && n.tagName === 'button') out.push(n);
    walk(n.kids);
  });
  walk(pool);
  const r = out; r.forEach = Array.prototype.forEach.bind(r); return r;
}

const byId = {};
function mk(id) { const e = new El('div'); e.attrs.id = id; byId[id] = e; ALL.push(e); return e; }
[
  'stats', 'tldr', 'hslider', 'hval', 'hticks', 'c-recall', 'c-norm', 'recall-read',
  'c-loop', 'ro-norm', 'ro-excess', 'ro-verdict', 'contain-cards', 'c-contain',
  'contain-why', 'c-locate', 'bslider', 'bval', 'bticks', 'c-bits', 'bits-read',
  'bits-why', 'sr-read', 'evalseg', 'verdicts', 'c-h32k', 'h32k-read',
  'cslider', 'cval', 'cticks', 'c-mem', 'mem-read', 'c-cross', 'cross-read',
  'bib', 'tip', 'themeToggle', 'loom-art', 'loomH', 'loom-read'
].forEach(mk);

// the two segmented toggles, as index.html declares them
function seg(parent, attr, values) {
  values.forEach((v, i) => {
    const b = new El('button');
    b.setAttribute(attr, v);
    b.setAttribute('aria-pressed', String(i === 0));
    parent.appendChild(b);
  });
}
const yokeHost = mk('_yoke'); seg(yokeHost, 'data-yoke', ['self', 'yoked']);
const roundHost = mk('_round'); seg(roundHost, 'data-round', ['rtne', 'sr']);

global.document = {
  documentElement: new El('html'),
  getElementById: (id) => byId[id] || null,
  createElement: (t) => new El(t),
  createElementNS: (ns, t) => new El(t),
  querySelectorAll: (sel) => matchIn(ALL.filter((e) => !e.attrs.id), sel).length
    ? matchIn(ALL.filter((e) => !e.attrs.id), sel)
    : matchIn(ALL, sel),
  addEventListener() {}
};
global.getComputedStyle = () => ({
  getPropertyValue: (k) => {
    const set = global.document.documentElement.getAttribute('data-theme') === 'dark'
      ? SCHEME.dark : SCHEME.light;
    const v = set[k.replace(/^--/, '')];
    return v === undefined ? '' : v;
  }
});
global.localStorage = {
  _d: {}, getItem(k) { return this._d[k] ?? null; }, setItem(k, v) { this._d[k] = v; }
};
/* Real frames, not counted ones, on a budget: a callback that re-arms itself at the end
   of each frame has to be stopped by something, and here that is the budget. */
const _T0 = Date.now();
let _frames = 6;
global.requestAnimationFrame = (f) => {
  if (_frames-- > 0) f(_T0 + (6 - _frames) * 16);
  return _frames;
};
global.cancelAnimationFrame = () => {};
global.performance = { now: () => _T0 };

global.window = {
  addEventListener() {},
  devicePixelRatio: 2,
  // the page asks whether motion is wanted before it animates anything
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  requestAnimationFrame: (f) => setTimeout(f, 0)
};
global.Element = El;

module.exports = { byId, yokeHost, roundHost, El, ALL };
