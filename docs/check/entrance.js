/* Run the entrance to completion on a real clock, and report what happened.

   The page harness gives it six frames at fixed timestamps, which is enough to prove it
   does not throw and nothing else. A simulation fails in ways that needs more than six
   frames to see: it can snap on the first frame, never snap at all, or never stop.

   So this drives it at 60fps until it clears the canvas, over several plays, and reports
   the beats. Run it on its own:  node check/entrance.js */
'use strict';
const path = require('path');

const dom = require('./dom.js');
require(path.join(__dirname, '..', 'static/data/data.js'));

/* a clock the simulation cannot tell from a real one */
let T = 0;
const FRAME = 1000 / 60;
let queued = null;
global.requestAnimationFrame = (f) => { queued = f; return 1; };
global.cancelAnimationFrame = () => { queued = null; };
global.performance = { now: () => T };

const cv = dom.byId.weave;
cv.clientWidth = 1280; cv.clientHeight = 720;

require(path.join(__dirname, '..', 'static/js/app.js'));

function run(label) {
  const ctx = cv._ctx;
  ctx.ops.length = 0; ctx.colours.length = 0;
  let frames = 0;
  while (queued && frames < 1200) {
    const f = queued; queued = null;
    T += FRAME; frames += 1;
    f(T);
  }
  const n = {};
  ctx.ops.forEach((o) => { n[o] = (n[o] || 0) + 1; });
  const cols = [...new Set(ctx.colours)];
  return { label, frames, seconds: (frames * FRAME / 1000).toFixed(2), ops: n, cols };
}

const first = run('load');
const rows = [first];
/* and again, the way a reader replays it by clicking the title */
for (let i = 0; i < 3; i++) {
  dom.byId.title.fire('click');
  rows.push(run('replay ' + (i + 1)));
}

let bad = 0;
for (const r of rows) {
  const ended = r.frames < 1200;
  const flashed = (r.ops.arc || 0) > 0;
  const laid = (r.ops.ellipse || 0) > 0;
  const broke = r.cols.includes(global.getComputedStyle().getPropertyValue('--fail'));
  console.log(
    `  ${r.label.padEnd(9)} ${String(r.frames).padStart(4)} frames (${r.seconds}s)` +
    `  shuttle ${laid ? 'yes' : 'NO '}  snapped ${broke ? 'yes' : 'NO '}` +
    `  flash ${flashed ? 'yes' : 'NO '}  ended ${ended ? 'yes' : 'NO '}`);
  if (!ended) { console.log('    FAIL  it never stopped'); bad += 1; }
  if (!laid) { console.log('    FAIL  the shuttle never crossed'); bad += 1; }
  if (!broke) { console.log('    FAIL  no thread snapped'); bad += 1; }
  if (!flashed) { console.log('    FAIL  the break never lit'); bad += 1; }
  if (r.frames < 60) { console.log('    FAIL  over in ' + r.seconds + 's'); bad += 1; }
}
console.log(bad ? `\n  ${bad} problems` : '\n  every play crossed, snapped, lit and ended');
process.exit(bad ? 1 : 0);
