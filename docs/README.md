# Project page

Static site: `index.html` plus `static/`. No build step and no framework; it works opened
from disk as well as from GitHub Pages. `.nojekyll` is there so `static/` is served as-is.

## Regenerating the data

```
python scripts/make_page_data.py           # writes static/data/data.js
python scripts/make_page_data.py --check   # recomputes the paper's quoted statistics
```

Nothing on the page is typed by hand. Every chart and every number in the prose reads
`static/data/data.js`, which `make_page_data.py` exports from `results/*.json` — the same
files `scripts/make_figures.py` builds the paper's figures from. Do not edit it directly.

Paired *t*-statistics are **recomputed** there from the per-document arrays rather than
copied out of the paper, so a disagreement surfaces instead of propagating. `--check`
prints the recomputation next to what the paper prints; it is how the one error in the
submitted version was found. The three-model comparison additionally asserts that the
runs behind its columns share a protocol, and refuses to render if they do not — the
columns mean nothing otherwise.

## Checking it

```
node docs/check/run_page.js
```

This loads `data.js` and `app.js` against a DOM stub (`check/dom.js`) and drives every
slider position, every toggle and every intervention card. It fails on a thrown error, an
empty readout, or a readout containing `undefined` or `NaN`. A syntax check would not
catch a renamed key in `data.js`; this does.

## Colour

Colour carries meaning here and only one meaning each: `--brand` is the page itself and,
in the charts, persistent group-wise scaling; `--fail` is per-token dynamic scaling;
`--mid` is partial containment; grey is everything else, including unquantized references
and random baselines. Each hue has a light form for washes and a darkened one for 12px
text and chart lines, because a tint that is pleasant at 70px is unreadable at 12.

Contrast against the page background and CIE Lab separation between the data colours were
both measured, so no two chart lines can be mistaken for one another.
