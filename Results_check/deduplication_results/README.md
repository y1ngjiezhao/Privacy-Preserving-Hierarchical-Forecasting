# Deduplicated experiment results

This folder contains results recalculated from `fhf_dedup_gef2017` and
`fhf_dedup_ausgrid2013` using the level-wise accuracy and hierarchical
incoherence definitions in the existing `results_check` notebooks.

Accuracy levels follow the dataset-specific policy requested on 2026-11-13.
AusGrid2013 uses authoritative bottom membership from `tags_dedup.pkl`, with
retained single-child terminal series evaluated at L3: **1/60/299** nodes.
GEF2017 uses structural path depth: **1/6/3** nodes, retaining the five shallow
terminal series at L2. Structural depth is stored separately for all datasets
and continues to define parent-child incoherence.

## Publication labels

| Internal experiment name | Framework label | Unreconciled label |
|---|---|---|
| `FHF` | `FHF` | `FedAvg` |
| `HFF` | `LN-FHF` | `HierAvg` |
| `HFFPS` | `CH-FHF` | `HierAvgPS` |

Reconciled plotting columns combine the framework and estimator, for example
`FHF_fMinTs`, `BO-FHF_fMinTs`, and `CH-FHF_fMinTs`. The shared `Naive`
baseline is included once and placed after all model columns.

## Main outputs

Each dataset directory contains:

- `deduplicated_paper_summary.csv`: one row per framework, horizon, and paper method.
- `deduplicated_accuracy_by_level.csv`: level-wise MAE, MASE, and RMSSE for all estimators.
- `deduplicated_incoherence_by_level.csv`: level-wise absolute and relative
  incoherence for every method. `incoherence_ratio_pct` retains the original
  demand-weighted total ratio. `average_scale_free_incoherence_ratio_pct` is
  the arithmetic mean over parent-time comparisons of
  `2 * |parent forecast - children forecast sum| /
  (|parent forecast| + |children forecast sum|) * 100`. It is unit-free,
  bounded from 0% (perfect coherence) to 200%, and gives every comparison
  equal weight. A comparison with both forecasts equal to zero contributes 0%.
  `average_error_scaled_incoherence_ratio_pct` implements the complementary
  accuracy-conditioned diagnostic
  `2 * |parent forecast - children forecast sum| /
  (|parent forecast - parent actual| +
  |children forecast sum - parent actual|) * 100`. It also ranges from 0% to
  200%; a `0 / 0` comparison contributes 0%. Because its scaler depends on a
  method's forecast errors, it should be interpreted alongside the
  forecast-scale ratio rather than as its replacement.
- `deduplicated_incoherence_by_parent.csv`: the same measures before parents
  are combined within a hierarchy level.
- `deduplicated_average_scale_free_incoherence_by_method.csv`: one row per
  method and forecast horizon, containing both average scale-free ratios. Its
  overall averages weight hierarchy levels by their number of parent-time
  comparisons, so every comparison receives one vote regardless of series
  magnitude.
- `all_per-node_{mase,rmsse,mae}.csv`: 12 paper-method columns plus the shared
  `Naive` baseline for MCB tests.
- AusGrid: `MCB_MASE.pdf` and `MCB_RMSSE.pdf`, plus PNG previews.
- GEF2017: horizon-specific `MCB_{MASE,RMSSE}_h{1,3,6}.pdf` figures,
  plus matching PNG previews. MCB tests are calculated independently within
  each forecast horizon rather than pooling horizons.
  Obsolete pooled figures have been deleted; the renderer only generates
  horizon-specific figures for the current paper.
- `deduplication_audit.csv` and `paper_validation.json`: node-count and coherence checks.

Run `generate_deduplication_results.py` from any working directory to rebuild
the CSV outputs, then run `render_mcb_figures.R` to rebuild the MCB figures.

## Publication naming and regeneration

Reconciliation labels use `fBU`, `fMinT-cov`, `fMinT-OLS`, `fMinTs`,
`fWLSv`, and `fWLS-struct`. Accuracy headers remain `MAE`, `MASE`, and
`RMSSE`; only reconciliation methods receive the `f` prefix. These are display changes only; raw input columns, `method_internal`,
formulas, and established output filenames remain stable. `MASI` retains its
existing name as the separately defined incoherence diagnostic.

Run `render_scaled_incoherence.py` after the result generator to rebuild both
scale-free incoherence figures. MCB figures use a compact 6.4 × 5.2 inch layout;
heatmaps use narrower, taller cells with centered labels. All figures include
PDF and PNG versions.

## Current manuscript labels and provenance

The current paper displays LN-FHF (Leaf--Node FHF). Historical CSV keys may
remain `BO-FHF`; they identify the same HFF runs and do not change scores.
`paper_current/` (in this directory) exports LN-FHF labels
while retaining raw experiment keys in the original result files.
The authoritative renderer is
`PPHF_modules/results_check/deduplication_results/render_mcb_figures.R`.
All displayed rank labels use three decimals; 14-point single-row
legends span the full 6.4 × 5.2 inch canvas, with red #B44745 and blue #3675A3.
The displayed paper figures are direct copies of these source PDFs.

Cleanup on 2026-09-28 removed pooled legacy PDFs/PNGs; no current per-horizon figure or CSV was deleted.

## Unified output layout (2026-11-09)

This is the sole active result directory. Dataset folders combine complete
notebook tables (`accuracy_by_level.csv`, `coherence_*.csv`, training plots) with
publication-format tables (`deduplicated_*.csv`, MCB figures).
`validation.json` validates the complete current analysis;
`paper_validation.json` validates publication-format exports.
`error_analysis/` contains the current 12-run bootstrap analysis;
`paper_current/` contains publication exports including DP results.
Notebook files stay in their original entry directories. Old non-deduplicated
exports are archived outside `Results_check` and are not packaged.
