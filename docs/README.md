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
submitted version was found. The three-model comparison also checks that the
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

It also checks four things a browser would show a reader and nothing else here would:

- the page's copy of the abstract still matches `sec_abstract.tex`, when the paper source
  is alongside — two copies of a paragraph with nothing comparing them drifted two rounds
  of edits apart once;
- nothing is drawn outside its own `viewBox`, which is how a legend ended up printed over
  the paragraph below its figure;
- the theme toggle runs and every chart redraws in both schemes, since a colour the dark
  scheme forgets reaches the chart as an empty string;
- the BibTeX block has a title, an author, a year and a url, whatever entry type it uses.

## Colour

Colour carries meaning here and only one meaning each: `--brand` is the page itself and,
in the charts, persistent scaling; `--fail` is per-token dynamic scaling; `--mid` is
partial containment; grey is everything else, including unquantized references and random
baselines. Each hue has a light form for washes and a darkened one for 12px text and chart
lines, because a tint that is pleasant at 70px is unreadable at 12. The hairlines, the
meander and the warp threads behind the hero are ornament and never carry data.

There are two schemes. The page follows the system setting unless the toggle in the nav is
used, and an explicit choice is remembered. `app.js` reads twelve colour tokens from CSS
at draw time, so both schemes have to define all twelve or a chart keeps its daylight
colours against the wrong ground.

```
node docs/check/palette.js
```

computes, from the stylesheet rather than from a comment: the contrast ratio of every
colour used for text or a chart line against the ground it sits on, the CIE-Lab distance
between the three data colours, and whether the two copies of the dark scheme — one under
`prefers-color-scheme`, one under `[data-theme]` — have drifted apart. `run_page.js` runs
it too, so a palette change cannot pass unnoticed.
