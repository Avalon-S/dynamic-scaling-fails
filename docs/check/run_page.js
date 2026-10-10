/* Load the page's data and script against the DOM stub, then drive every control
   through every position. Any thrown error, or any readout that comes out empty
   or containing "undefined"/"NaN", is a bug the browser would have shown. */
'use strict';
const path = require('path');
const { byId, yokeHost, roundHost } = require('./dom.js');
const PAGE = path.resolve(__dirname, '..');

require(path.join(PAGE, 'static/data/data.js'));   // sets window.LQ
let fails = 0;
function bad(what, why) { fails++; console.log('  FAIL  ' + what + ' — ' + why); }

try {
  require(path.join(PAGE, 'static/js/app.js'));
  console.log('app.js ran, initial render OK');
} catch (e) {
  console.log('app.js THREW on load: ' + e.message + '\n' + e.stack.split('\n').slice(1, 4).join('\n'));
  process.exit(1);
}

const READOUTS = ['tldr', 'recall-read', 'ro-norm', 'ro-excess', 'ro-verdict', 'contain-why',
  'bits-read', 'bits-why', 'sr-read', 'h32k-read', 'mem-read'];
function checkReadouts(label) {
  READOUTS.forEach((id) => {
    const t = byId[id].innerHTML || '';
    if (!t.trim()) bad(label + ' #' + id, 'empty');
    if (/undefined|NaN|\[object/.test(t)) bad(label + ' #' + id, 'contains ' + (t.match(/undefined|NaN|\[object/) || [])[0]);
  });
  ['stats', 'contain-cards', 'verdicts', 'evalseg', 'hticks', 'bticks', 'cticks'].forEach((id) => {
    if (!byId[id].children.length) bad(label + ' #' + id, 'no children rendered');
  });
  ['c-recall', 'c-norm', 'c-loop', 'c-contain', 'c-locate', 'c-bits', 'c-h32k', 'c-mem'].forEach((id) => {
    if (!byId[id].children.length) bad(label + ' #' + id, 'no svg drawn');
  });
  if (!byId.bib.textContent.includes('@inproceedings')) bad(label + ' #bib', 'no bibtex');
}
checkReadouts('initial');

function drive(id, n, label) {
  for (let i = 0; i < n; i++) {
    byId[id].value = String(i);
    try { byId[id].fire('input'); } catch (e) { bad(label + '=' + i, e.message); continue; }
    checkReadouts(label + '=' + i);
  }
}
drive('hslider', 5, 'horizon');
drive('bslider', 5, 'bits');
drive('cslider', 6, 'context');

['self', 'yoked'].forEach((v) => {
  const b = yokeHost.children.find((c) => c.getAttribute('data-yoke') === v);
  try { b.fire('click'); checkReadouts('yoke=' + v); } catch (e) { bad('yoke=' + v, e.message); }
});
['rtne', 'sr'].forEach((v) => {
  const b = roundHost.children.find((c) => c.getAttribute('data-round') === v);
  try { b.fire('click'); checkReadouts('round=' + v); } catch (e) { bad('round=' + v, e.message); }
});
byId['contain-cards'].children.forEach((c, i) => {
  try { c.fire('click'); checkReadouts('contain#' + i); } catch (e) { bad('contain#' + i, e.message); }
});
byId.evalseg.children.forEach((c, i) => {
  try { c.fire('click'); checkReadouts('evalH#' + i); } catch (e) { bad('evalH#' + i, e.message); }
});
// The page repeats the paper's abstract. Two copies of a paragraph with nothing
// comparing them drifted two rounds of edits apart once, so compare them.
(() => {
  const fs = require('fs');
  // The published repository ships the page without the paper source, so there is
  // nothing to compare against there. Only skip when the file is genuinely absent.
  const texPath = path.join(PAGE, '../Paper_Writing/sec_abstract.tex');
  if (!fs.existsSync(texPath)) {
    console.log('abstract: no sec_abstract.tex alongside, nothing to compare');
    return;
  }
  const tex = fs.readFileSync(texPath, 'utf8')
    .replace(/(^|[^\\])%.*/g, '$1').split(/\s+/).join(' ').trim();
  const html = fs.readFileSync(path.join(PAGE, 'index.html'), 'utf8');
  const m = html.match(/<p class="abstract-body">(?:<!--[^]*?-->)?([^]*?)<\/p>/);
  const page = m ? m[1].split(/\s+/).join(' ').trim() : null;
  if (page === null) { bad('abstract', 'no abstract-body paragraph in index.html'); }
  else if (page !== tex) {
    let i = 0;
    while (i < tex.length && i < page.length && tex[i] === page[i]) i++;
    bad('abstract', 'the page and sec_abstract.tex diverge at character ' + i);
    console.log('    paper: ...' + tex.slice(Math.max(0, i - 30), i + 60));
    console.log('    page : ...' + page.slice(Math.max(0, i - 30), i + 60));
  } else { console.log('abstract matches sec_abstract.tex (' + tex.split(' ').length + ' words)'); }
})();

// An element drawn below its own viewBox lands on whatever follows the figure. Valid
// SVG, invisible to every other check here, and it happened.
(() => {
  for (const id of ['c-recall', 'c-norm', 'c-contain', 'c-locate', 'c-bits', 'c-h32k',
                    'c-mem', 'c-cross']) {
    const host = byId[id];
    if (!host || !host.kids.length) continue;
    const svg = host.kids.find((k) => k.tagName === 'svg');
    if (!svg) continue;
    const vb = String(svg.attrs.viewBox || '').split(' ').map(Number);
    if (vb.length !== 4) { bad(id, 'no usable viewBox'); continue; }
    let low = 0;
    (function walk(n) {
      for (const k of n.kids || []) {
        for (const a of ['y', 'y1', 'y2', 'cy']) {
          if (k.attrs[a] != null) low = Math.max(low, Number(k.attrs[a]));
        }
        walk(k);
      }
    })(svg);
    if (low > vb[3]) bad(id, 'drawn past its viewBox: lowest y=' + low + ' in a box ' + vb[3] + ' tall');
  }
})();

console.log('\nsample readouts at the end:');
console.log('  verdict : ' + byId['ro-verdict'].innerHTML.replace(/<[^>]+>/g, '').slice(0, 150));
console.log('  h32k    : ' + byId['h32k-read'].innerHTML.replace(/<[^>]+>/g, '').slice(0, 150));
console.log('  memory  : ' + byId['mem-read'].innerHTML.replace(/<[^>]+>/g, '').slice(0, 150));
console.log('  cross   : ' + byId['cross-read'].innerHTML.replace(/<[^>]+>/g, '').slice(0, 150));
console.log(fails ? '\n' + fails + ' FAILURES' : '\nall controls drove cleanly, no empty or malformed readouts');
process.exit(fails ? 1 : 0);
