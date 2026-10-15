/* Does each scheme's palette actually work?

   style.css claims contrast ratios and CIE-Lab separations. A claim in a comment rots the
   moment a colour changes, and the dark scheme doubled the number of colours that have to
   hold. So the numbers are computed from the stylesheet itself.

   Two things are checked per scheme:
     contrast      every colour used for text or a 2px chart line, against the ground it
                   sits on. WCAG AA for normal text is 4.5; a chart line is a graphical
                   object, where the threshold is 3.
     separability  the three data colours against each other in CIE Lab. Two lines a
                   reader cannot tell apart are two lines that carry no data.

   Run on its own, or through run_page.js. */
'use strict';
const fs = require('fs');
const path = require('path');

const CSS = fs.readFileSync(path.join(__dirname, '..', 'static/css/style.css'), 'utf8');

/* Pull one scheme's tokens: the :root block for light, the [data-theme="dark"] block for
   dark. The media-query copy is the same values, so checking one of the two is enough. */
function tokens(selector) {
  const i = CSS.indexOf(selector);
  if (i < 0) throw new Error('no ' + selector + ' block in style.css');
  const body = CSS.slice(CSS.indexOf('{', i) + 1, CSS.indexOf('\n}', i));
  const out = {};
  for (const m of body.matchAll(/--([\w-]+)\s*:\s*([^;]+);/g)) out[m[1]] = m[2].trim();
  return out;
}

function rgb(v, onto) {
  let m = /^#([0-9a-f]{6})$/i.exec(v);
  if (m) {
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  m = /^#([0-9a-f]{3})$/i.exec(v);
  if (m) {
    const n = parseInt(m[1].replace(/./g, (c) => c + c), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  m = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,\s/]+([\d.]+))?\s*\)/i.exec(v);
  if (m) {
    const a = m[4] === undefined ? 1 : parseFloat(m[4]);
    const c = [+m[1], +m[2], +m[3]];
    if (a >= 1 || !onto) return c;
    return c.map((x, k) => Math.round(x * a + onto[k] * (1 - a)));   // composite it
  }
  return null;
}

const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
const lum = (c) => 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
function contrast(a, b) {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}
function lab(c) {
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const [r, g, b] = c.map(lin);
  const X = f((r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047);
  const Y = f(r * 0.2126 + g * 0.7152 + b * 0.0722);
  const Z = f((r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883);
  return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
}
const dE = (a, b) => Math.hypot(...lab(a).map((v, i) => v - lab(b)[i]));

/* text colour -> the ground it is read against, and the floor it must clear */
const TEXT = [
  ['ink', 'bg', 4.5], ['ink-2', 'bg', 4.5], ['ink-3', 'bg', 3.0],
  ['ink', 'surface', 4.5], ['ink-2', 'surface', 4.5],
  ['brand', 'bg', 4.5], ['brand', 'surface', 4.5],
  ['fail', 'bg', 3.0], ['mid', 'bg', 3.0],
];
/* the three data colours, which must also be distinguishable from one another */
const DATA = ['keep', 'fail', 'mid'];
const SEP = 32;

function check(name, selector) {
  const t = tokens(selector);
  const bg = rgb(t.bg);
  const fails = [];
  const rows = [];
  for (const [fg, on, floor] of TEXT) {
    if (!t[fg] || !t[on]) { fails.push(`${name}: --${fg} or --${on} is not defined`); continue; }
    const c = contrast(rgb(t[fg], bg), rgb(t[on], bg));
    rows.push(`    --${fg} on --${on}  ${c.toFixed(2)}:1 (needs ${floor})`);
    if (c < floor) fails.push(`${name}: --${fg} on --${on} is ${c.toFixed(2)}:1, under ${floor}`);
  }
  for (let i = 0; i < DATA.length; i++) {
    for (let j = i + 1; j < DATA.length; j++) {
      const d = dE(rgb(t[DATA[i]], bg), rgb(t[DATA[j]], bg));
      rows.push(`    --${DATA[i]} vs --${DATA[j]}  dE ${d.toFixed(0)} (needs ${SEP})`);
      if (d < SEP) fails.push(`${name}: --${DATA[i]} and --${DATA[j]} are only dE ${d.toFixed(0)} apart`);
    }
  }
  return { rows, fails };
}

/* app.js reads these twelve; a scheme that leaves one out keeps its daylight colour */
const CHART = ['brand', 'brand-2', 'fail', 'fail-2', 'keep', 'mid', 'ink', 'ink-2',
  'ink-3', 'line', 'grey', 'grid'];

function run(verbose) {
  const all = [];
  /* The dark scheme is written twice -- once under prefers-color-scheme for a reader who
     never touches the toggle, once under [data-theme] for one who does. Two copies of
     sixty tokens drift, and the drift shows up only for whichever half of the audience
     hits the copy that was not updated. */
  const viaMedia = tokens('@media (prefers-color-scheme: dark)');
  const attr = tokens(':root[data-theme="dark"]');
  for (const k of new Set([...Object.keys(viaMedia), ...Object.keys(attr)])) {
    if (viaMedia[k] !== attr[k]) {
      all.push(`dark: --${k} is "${viaMedia[k]}" under prefers-color-scheme and "${attr[k]}" under [data-theme]`);
    }
  }
  for (const [name, sel] of [['light', ':root {'], ['dark', ':root[data-theme="dark"]']]) {
    const t = tokens(sel);
    const missing = CHART.filter((k) => !t[k]);
    if (missing.length) all.push(`${name}: the charts read ${missing.join(', ')}, undefined here`);
    const r = check(name, sel);
    all.push(...r.fails);
    if (verbose) { console.log('  ' + name); r.rows.forEach((x) => console.log(x)); }
  }
  return all;
}

module.exports = { run, tokens };

if (require.main === module) {
  const fails = run(true);
  console.log(fails.length ? '\n' + fails.map((f) => '  FAIL  ' + f).join('\n')
    : '\nboth schemes: contrast and separation hold');
  process.exit(fails.length ? 1 : 0);
}
