These results were recomputed on 2026-11-13. All outputs are inside this project and do not depend on OneDrive.
The three figures omit overall titles and subtitles, retaining panel titles and legends.
The directional phrase "lower is better" has been removed.
The legacy labels have been removed from all three figure legends.

Completed work
  Ordinary FHF, HFF, HFFPS: 3 AusGrid fh1 runs and 9 GEF fh1/fh3/fh6 runs.
  All four Results_check notebooks were executed and saved outputs without error cells.
  The 12 ordinary experiments completed 2000 block-bootstrap replicates, error redistribution analysis, and metric reproduction checks.
  Accuracy was recomputed for 39 saved AusGrid DP runs and 1 non-DP baseline.
  The 39 DP runs comprise 36 runs in the 20-round sweep and 3 runs with 100 rounds.

Levels
  AusGrid2013: evaluation levels L1/L2/L3 = 1/60/299.
  GEF2017: structural depths L1/L2/L3 = 1/6/3, following the latest requested convention.
  The 5 shallow terminal GEF nodes remain in L2.
  Coherence for both datasets still uses the actual structural parent-child relationships.

The three figures are in figures/, with 300 dpi PNG and vector PDF versions in the full experiment outputs. This GitHub snapshot retains the PDFs.
  01: MASE/RMSSE for Average, L1, L2, and L3 in a 2×2 layout.
  02: recorded privacy budgets and errors for the same four panels.
  03: forecasts for three representative series during the first 72 test hours, retaining the previously selected series.
  Color distinguishes settings; circles denote MASE and triangles denote RMSSE. The bottom explanatory text has been removed.
  All three legends use FHF (q=1), PPHF (q=1), PPHF (q=0.5), and PPHF (q=0.1).
  Figure 02 uses darker gray, blue, green, and red consistently for points, annotations, and legends.
  Its horizontal tick labels are integers: 0, 25, 50, and 75.
  Its vertical tick labels have one decimal place.
  Its Setting and Metric legends share one row.
  Its upper-left panel title is Average.
  Each q/epsilon annotation group is internally left-aligned and centered above its data point.
  Displayed results are rounded to three decimal places; calculations and original CSVs retain full precision.
  sampling_100rounds_by_scope_3dp.csv is the three-decimal display table.
  Figure 02 uses larger text: 14 pt tick labels and legends, 16 pt axis and panel titles, and 4.2 mm annotations, with additional top space.

Data scope
  The three figures use matched 100-round results with sigma=1.2, C=1.0, and delta=1e-5.
  The 20-round sweep results are stored separately in sampling_20round_sweep_by_scope.csv and are excluded from these figures.
  The historical q=1, 100-round DP baseline missing from the current directory was recovered from the local FHF_backup.
  Its source and checksums are recorded in Results_check/runs/20261113/restored_baseline_manifest.json in the full local experiment archive.
  Training targets, test keys, and observations for all DP runs were checked against the current non-DP baseline.
  The old q=1 configuration used fixed DP seed=42; its recorded epsilon is computed for the hypothetical randomized mechanism.
  Partial participation uses fresh randomness and 2× sensitivity calibration; the old q=1 configuration uses 1× calibration.
  Privacy budgets come from the original privacy.json files or existing recovery records; privacy completion markers were not rewritten.
  These budgets cover the global model, not releases of local forecasts or diagnostics.

Rerun
  Python environment: /Users/yingjiezhao/miniforge3/envs/neural-network/bin/python
  Ordinary checks: Results_check/runs/20261113/rebuild_current_results.py --datasets ausgrid2013 gef2017 --execute-notebooks
  DP checks: PPHF_modules/results_check/sampling_results_check.py
  ggplot2: Rscript PPHF_modules/results_check/plot_sampling_comparison.R
  A copy of plot_sampling_comparison.R is retained in this directory and can be run directly with Rscript.

Current archive
  The new result directories are ausgrid2013, gef2017, error_analysis, and dp_sampling.
  The older paper audits in paper_current are historical records and were not inputs to this recomputation.
