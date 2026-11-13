# Privacy-accounting dependency fix and result recovery patch

Cause of the error: after 20 training rounds, Google dp-accounting failed to import `attr`.
The `attr` module comes from the `attrs` package. This patch bundles attrs 23.2.0 and
the project's original mpmath 1.3.0, and moves dependency checks to startup in both
the notebook helper and CLI.

## Apply on Windows

This ZIP has `PPHF_modules`, `windows_gpu`, and `DP_fhf` at its top level, without
an additional `FHF_NDSH` wrapper. Extract it into the original project root, such as
`E:\yjz\FHF_NDSH`, and overwrite code files with matching names.
Keep existing datasets, forecast CSVs, and training NPZ files. The patch does not
include datasets or experiment results.

## Recover results without retraining

1. Apply the patch and restart the notebook kernel.
2. In the sampling-rate notebook that failed, run only the import and parameter
   cells. Keep the original `SAMPLING_RATE`, `SAMPLING_SEED`, and `NUM_ROUNDS`.
   The supplied log showed 20 rounds.
3. Skip the training cell and Run All. Add a cell containing:

```python
from PPHF_modules.dp_fhf.recover_dp_sweep import recover_ausgrid_sampling_sweep

recovered = recover_ausgrid_sampling_sweep(
    output_root=OUTPUT_ROOT,
    sampling_rate=SAMPLING_RATE,
    sampling_seed=SAMPLING_SEED,
    num_rounds=NUM_ROUNDS,
)
recovered
```

The tool checks available forecast CSV / training NPZ pairs against
`parameter_grid.csv`, then recovers per-series metrics, MAE/RMSE/MASE/RMSSE, and
epsilon for the corresponding configuration. It lists missing pairs without
inventing results. Recovery outputs go into `recovery/` under that sampling-rate
directory, and `recovered_all_sampling_rates_summary.csv` is added to the shared
parent directory.

Recovery does not train, alter the original CSV/NPZ files, or create the normal
`privacy.json` completion marker. Model weights cannot be reconstructed from
forecasts and residuals if they were not saved before the error. Epsilon is a
retrospective accounting value for the original global-model mechanism, based
on the supplied actual completed round count and saved client population. It is
not a DP guarantee for local forecasts. The recovered tables support research
metric analysis. Rerunning the original training cell still treats runs without
completion markers as incomplete.

## Validation

A simulated missing-attr failure stops before training, and a fully failed sweep
raises an explicit error. The failure was simulated using previously completed
7-node, 2-round outputs. Recovered metrics and epsilon matched the original
successful results for all three sampling rates, and input hashes were unchanged.
Windows results were not accessed or modified.
