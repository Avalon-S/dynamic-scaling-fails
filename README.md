# When Dynamic Scaling Fails

Code and results for *When Dynamic Scaling Fails: A Scale-State Feedback Governs
Recurrent-State Quantization in Linear-Attention and SSM LLMs*
(NeurIPS 2026 Workshop on Long-Context Foundation Models, non-archival).

Paper: https://arxiv.org/abs/2610.15139
Workshop record: https://openreview.net/forum?id=4DvSLh9jTY
Project page: https://avalon-s.github.io/dynamic-scaling-fails/

The arXiv version is the one to read and cite. The workshop is non-archival, and the
OpenReview record is the submitted PDF: it predates the Mamba-2 results and the two
corrections this repository's numbers reflect.

Quantizing the recurrent state of a linear-attention or state-space model puts the
quantizer **inside** the recurrence, so its error compounds over decode steps. Per-token
dynamic scaling reads its scale from a state its own error has inflated, which closes a
feedback loop between the two. This repository holds the fake-quantization and evaluation
code, the result files every number in the paper is computed from, and the scripts that
turn those into the figures and the project page.

Nothing here trains or modifies a model. All quantization is simulated: round, clamp,
dequantize, in the dtype the model already uses.

## Layout

```
src/        state quantizers, the article splits, the NIAH task
scripts/    one runner per experiment, plus the merge and figure scripts
results/    the result file behind every number in the paper (JSON)
docs/       the static project page, built from results/ by scripts/make_page_data.py
```

## Setup

```
pip install torch transformers
pip install flash-linear-attention      # registers gated_deltanet with AutoModel
pip install mamba-ssm causal-conv1d     # only for the Mamba-2 runs
```

Models and data are public: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`,
`state-spaces/mamba2-1.3b`, and WikiText-103. Everything in the paper runs on one 24 GB
GPU; the longest single job is the 32K exposure curve at roughly eight hours.

## Reproducing the paper

Calibration, selection and test articles are disjoint by title hash
(`src/data.py: get_article_splits`), so no number below is read off the articles its
policy was chosen on.

| what | command |
|---|---|
| Figure 1a, the failure frontier | `python scripts/qwen35_recall.py --out results/qwen35_bitsweep_08b.json` |
| Figure 1b, the norm runaway | `python scripts/qwen35_longdoc2.py --independent --n_docs 16 --out results/longdoc2_ind_curve_08b.json` |
| Table 1, containment | `python scripts/qwen35_hybrid.py` and `python scripts/qwen35_reset.py` |
| Table 2, the same policies on three models | the two `longdoc2` runs above, plus `python scripts/mamba_contain.py --independent --n_docs 16 --Hs 2048 --policies unquant dynamic4 static4 group32 capped4 zero` |
| locating outliers, at matched rate | `python scripts/qwen35_range.py` |
| scale-yoking | `python scripts/qwen35_yoke.py` |
| Table 3, what the residual is made of | `python scripts/qwen35_yoke_sweep.py` then `scripts/merge_yoke_sweep.py` |
| Table 4, leak-free allocation | `python scripts/qwen35_clean_alloc.py` |
| survival analysis | `python scripts/qwen35_survival.py` |
| out to 32K | `python scripts/qwen35_horizon32k.py` then `scripts/merge_horizon32k.py` |
| stochastic rounding against exposure | `scripts/qwen35_horizon32k.py --policies unquant dynamic4_sr static4_sr group32_sr` then `scripts/compare_sr_exposure.py` |
| the Mamba-2 exposure curve | `python scripts/qwen35_mamba.py --independent --n_docs 16` |
| the Mamba-2 recall frontier | `python scripts/mamba_recall.py --seeds 12` then `scripts/merge_mamba_recall.py` |
| what share of the cache the state is | `python scripts/memory_share.py` (no GPU) |
| every figure | `python scripts/make_figures.py` |

Long runs shard by document or by seed (`--doc_from/--doc_to`, `--seed_from/--seed_to`)
and merge afterwards; the merge scripts refuse to pool a document or a seed that appears
in two shards.

`scripts/mamba_contain.py` reimplements the protocol of `scripts/qwen35_mamba.py` rather
than importing it, so that the file which produced the earlier published numbers keeps its
provenance. `--verify` runs the two policies both scripts have and compares them against
`results/mamba_ind_curve.json`, exiting non-zero if they have drifted apart; it is worth
running before anything downstream.

## Determinism

The Gated DeltaNet runs are bit-identical between independent executions: re-running the
32K sweep from scratch reproduced all 35 per-document excess-NLL values to the last digit.

The Mamba-2 runtime is not deterministic. Repeating a configuration moves a
single-document loss by a few hundredths of a nat, so those losses are reproducible in
distribution rather than exactly. State norms move much further, because a runaway
amplifies whatever differs between runs: across three executions the dynamic-INT4 norm at
H=2048 ranged from 38.9 to 42.3 while the corresponding loss moved by 0.4%. Read a norm
from a runaway condition to one significant figure.

## License

Code is MIT (`LICENSE`). The result files in `results/` are released under
CC BY 4.0. WikiText-103 is CC BY-SA 3.0; the model checkpoints carry their own licenses.
