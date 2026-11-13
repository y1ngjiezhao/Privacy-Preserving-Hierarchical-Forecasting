# Portable NDSH DP-FHF notebooks

All notebooks automatically locate the `FHF_NDSH` project root, the matching
deduplicated dataset, and available FHF artifacts. Set `DP_FHF_PROJECT_ROOT`
only when Jupyter is launched outside the project tree. A dataset can also be
overridden with `DATA_CSV_OVERRIDE`, `DP_FHF_GEF2017_CSV`, or
`DP_FHF_AUSGRID2013_CSV`.

## DP-FedAvg experiment sets

The sampling examples are `01_DPFedAvg_Sampling_GEF2017.ipynb` and
`01_DPFedAvg_Sampling_AusGrid2013.ipynb`. Select the `neural-network` kernel,
adjust `SAMPLING_RATE`, `SAMPLING_SEED`, and `NUM_ROUNDS`, then Run All.
Defaults are sampling rate `0.1` and 100 rounds. These examples save only under
`DP_fhf/outputs/sampling_experiments/<dataset>/amplified_v1/sampling_runs/q_0p1__seed_fresh/`
with the default settings, including participant logs and privacy metadata.
Setting their rate to `1.0` still uses the separate `sampling_experiments/`
directory. They do not overwrite historical fixed runs or sweeps.

Existing fixed/sweep notebooks expose the same sampling parameters, defaulting
to `1.0` to preserve their previous behavior. Sampling is uniform without
replacement with `ceil(q * N)` participants. For partial uniform FHF,
the accountant now applies fixed-size sampling amplification with the actual
`ceil(q*N)/N`. Partial add/remove noise is calibrated to 2C/m; q=1 retains the
original noise and accountant. `SAMPLING_SEED=None` uses fresh hidden sampling
and noise randomness for partial runs. Fixed seeds produce a clearly labeled
hypothetical research budget. Only the files in `model_release/` belong to the
accounted public model release; the other exports contain internal research
data. Completed amplified runs live in `amplified_v1/`, separate from older
partial results.

The fixed experiments each produce one FHF run with forecast horizon 1 and
`hierarchy_mode=dedup`:

1. `01_DPFedAvg_Only_GEF2017.ipynb`: 100 rounds, clip norm `1`, noise `0.3`.
2. `01_DPFedAvg_Only_AusGrid2013.ipynb`: 100 rounds, clip norm `1`, noise `1.2`.

The dataset-specific sweeps are:

1. `01_DPFedAvg_GEF2017_Sweep.ipynb`.
2. `01_DPFedAvg_AusGrid2013_Sweep_q0p1.ipynb`: sampling rate `0.1`.
3. `01_DPFedAvg_AusGrid2013_Sweep_q0p5.ipynb`: sampling rate `0.5`.
4. `01_DPFedAvg_AusGrid2013_Sweep_q1p0.ipynb`: sampling rate `1.0`.

Each sweep contains 12 FHF runs: four noise multipliers (`1.2`, `0.9`, `0.6`,
`0.3`) × three clipping norms (`1`, `0.5`, `0.2`). Every combination uses 20
rounds, forecast horizon 1, and `hierarchy_mode=dedup`.

The three AusGrid entry points each call one fixed-rate sweep; none contains
an outer training loop over sampling rates. They replace the former single
AusGrid sweep notebook. All three write beneath
`DP_fhf/outputs/ausgrid2013_sampling_sweep/`:

```text
ausgrid2013_sampling_sweep/
├── full_participation/                          # q = 1.0
│   ├── nm_<noise>__clip_<C>/
│   └── sweep_summary.csv
├── amplified_v1/sampling_runs/
│   ├── q_0p1__seed_fresh/                        # q = 0.1
│   │   ├── nm_<noise>__clip_<C>/
│   │   └── sweep_summary.csv
│   └── q_0p5__seed_fresh/                        # q = 0.5
│       ├── nm_<noise>__clip_<C>/
│       └── sweep_summary.csv
└── all_sampling_rates_summary.csv
```

Run them one at a time on a single GPU. Completed matching configurations are
resumed; `OVERWRITE=True` reruns that notebook's rate. Each completed notebook
refreshes the combined CSV from completed rate summaries, so all three give
36 rows after 12 combinations per rate finish. The table includes sampling
rate, clipping norm, noise multiplier, epsilon, base-forecast accuracy and
each run's source directory. It can be partial while other rates are pending.
`SAMPLING_SEED=None` retains the current hidden-randomness partial protocol.
The global MASE/RMSSE summary also discovers this directory. Older sweep
reconciliation entry points still read their legacy input directory.

## Recovering a late accounting dependency failure

The bundled accounting code now includes `attrs==23.2.0` (the `attr` and
`attrs` import namespaces) and `mpmath==1.3.0`. Sampling-accountant imports
are checked before training in both notebook helpers and the CLI. An entirely
failed sweep raises an error instead of returning an empty success table.

If an older run finished training but failed with `No module named 'attr'`,
do not immediately rerun its training cell. The forecast CSV and training NPZ
were exported before accounting. Apply the fix, restart the kernel, run only
the imports and original parameter cells, then execute:

```python
from PPHF_modules.dp_fhf.recover_dp_sweep import recover_ausgrid_sampling_sweep

recovered = recover_ausgrid_sampling_sweep(
    output_root=OUTPUT_ROOT,
    sampling_rate=SAMPLING_RATE,
    sampling_seed=SAMPLING_SEED,
    num_rounds=NUM_ROUNDS,  # Must match the completed original training, e.g. 20.
)
recovered
```

Recovery reads complete forecast/NPZ pairs, uses their own training scales,
and recomputes epsilon for the configured completed FHF rounds and actual
fixed sample count. It performs no training and does not overwrite source
artifacts or manufacture normal privacy-completion markers. Recovered CSVs
are saved under the rate's `recovery/` directory, with a parent-level
`recovered_all_sampling_rates_summary.csv`. Missing pairs are reported.
Model weights cannot be reconstructed from forecasts/residuals; the report
records whether a model-release file exists. The posthoc accounting concerns
the global-model mechanism, not a privacy guarantee for local forecasts.

`01_DPFedAvg_NDSH_Matrix.ipynb` is now a non-executing index. The former
24-run FHF/HFF/HFFPS matrix has been retired and cannot launch mixed-framework
experiments.

## Automatic data selection

- GEF2017: `Datasets/GEFCOM2017/GEF2017_170101_dedup.csv`.
- AusGrid2013: `Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv`.

Pre-existing cached lag columns are removed and rebuilt from the selected
deduplicated target series.

## Other notebooks

- `01_DPFedAvg_MASE_RMSSE_Summary.ipynb` discovers the fixed/legacy result sets
  and the three AusGrid sampling-rate sweeps.
- `01_DPFedAvg_100Round_Reconciliation.ipynb` computes mints, wlsv, and ols
  for the two saved 100-round FHF runs, using each run's own training residuals
  and the original reconciliation implementation. Results, detailed forecasts,
  and matrices are saved to `DP_fhf/outputs/dp_100round_reconciliation/`.
- `01_DPFedAvg_Sweep_Reconciliation.ipynb` computes mints, wlsv, and ols
  for both completed 20-round sweeps. It reuses the original FHF reconciliation
  implementation and estimates matrices from each run's own saved training
  residuals. It saves predictions, matrices, per-node metrics, comparisons,
  and verification records under
  `DP_fhf/outputs/dp_sweep_reconciliation/`. No model retraining is needed.
- `02_DP_Covariance_Estimation.ipynb` processes completed FHF inputs with
  no noise-variance correction; fixed bounds ±0.0005, epsilon=1, seed=42,
  ridge=1e-6. Its default output is `<dataset>/no_debias/`.
- `02_DP_Covariance_Multiscale_No_Debias.ipynb` reproduces retentions
  99.5%, 75%, 50%, 25%, and 0.5% (q=0.005).
- `03_DP_O2MR_Saved_P.ipynb` processes FHF only and waits until all six saved
  FHF P matrices are available.

## Output layout

- `DP_fhf/outputs/01_dp_fedavg_100round/<dataset>/`
- `DP_fhf/outputs/01_dp_fedavg_sweep_20round/<dataset>/<run_id>/`
- `DP_fhf/outputs/02_dp_covariance/<dataset>/no_debias/` (current fixed-bound paper results)
- `DP_fhf/outputs/02_dp_covariance/multiscale_no_debias/` (current quantile results)
- `DP_fhf/outputs/03_dp_o2mr/<dataset>/`

Each DP-FedAvg run stores its bounds beside its own result, preventing conflicts
with older hierarchy or dataset variants.

## Dry-run validation

```bash
DP_FHF_NOTEBOOK_DRY_RUN=1 jupyter lab
```

Use the `neural-network` kernel, then run all cells. Dry-run checks data
discovery and command construction without starting training.

Obsolete corrected result files and 0.05% snapshots were deleted on 2026-09-28. Current notebook defaults require neither.
