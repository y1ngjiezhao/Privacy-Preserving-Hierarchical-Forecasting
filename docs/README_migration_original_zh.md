# PPHF code-and-dataset migration package

This is an English translation of the earlier migration README. Its package-content
statements describe that earlier archive. See the root README for the current
GitHub snapshot, which also includes result PDFs and compressed input datasets.

The original migration package contained no completed experiment results. Preserve
the internal FHF_NDSH/ directory structure after extraction.

Included:

- PPHF_modules/: FHF, HFF/HFFPS, DP-FHF, result analysis, validation code, and
  privacy-accounting dependency source.
- Datasets/: original and prepared inputs, hierarchy matrices, deduplication
  mappings, tags, and data-source documentation.
- 42 notebook entry points with code, parameters, and Markdown retained, but
  outputs, execution counts, and execution metadata cleared.
- windows_gpu/: Windows 10/11 and RTX 4070 SUPER installation, environment checks,
  and timing tools.

The original package excluded CSVs outside the dataset directory, forecasts,
metrics, training logs, residuals, models, NPZ training artifacts, reconciliation
matrices, result figures/tables, privacy reports, old validation reports,
development backups, caches, and Python environments. ZIP_MANIFEST.json was only
a file inventory with checksums, not an experiment result.

## Windows installation

Install the NVIDIA driver and 64-bit Python 3.12, run windows_gpu/setup.bat, then
windows_gpu/start_jupyter.bat. Select the Python (FHF RTX 4070 SUPER) kernel.
See windows_gpu/README_zh.md for details. A Mac environment cannot be copied
directly to Windows.

## First-run order

1. Prepared datasets are included and can be used directly for training. To
   rebuild them, first run the preparation notebooks in Datasets/.
2. For ordinary full-participation experiments, run FHF_fh1, HFF_fh1, and HFFPS_fh1
   in fhf_dedup_ausgrid2013/, and the fh1, fh3, and fh6 notebooks for all three
   frameworks in fhf_dedup_gef2017/, giving 12 combinations. They generate
   forecasts, training residuals, reconciliation matrices, metrics, and logs for
   subsequent analyses.
3. The four Results_check/ notebooks require the complete current results from
   step 2. Run All is not appropriate before training. The code creates outputs
   in Results_check/deduplication_results/.
4. For DP-FedAvg, first run DP_fhf/notebooks/01_DPFedAvg_Only_* or the corresponding
   *_Sweep notebook. Then run MASE_RMSSE_Summary, 100Round_Reconciliation, or
   Sweep_Reconciliation as appropriate.
5. DP covariance, multiscale, and DP-O2MR read the corresponding saved FHF
   forecasts, residuals, and P matrices, so run them after step 2. See
   DP_fhf/notebooks/README.md for each entry point's prerequisites.
6. *_Sampling_* and *_Sampling_fh1 are independent sampling experiments with
   10% participation per round by default and separate output directories.
   They do not replace the default full-participation experiments required by
   the four Results_check entry points.

Training output directories are initially empty and are created at runtime.
OVERWRITE=False does not skip training results that are absent from the package.
01_DPFedAvg_NDSH_Matrix.ipynb is only an index and does not start the retired mixed
framework matrix experiment.

For manuscript-format CSVs and R MCB figures, after step 2 run:

```sh
python -m PPHF_modules.results_check.deduplication_results.generate_deduplication_results
```

Then run PPHF_modules/results_check/deduplication_results/render_mcb_figures.R in
an R environment with the required dependencies installed.

## Three independent AusGrid sweeps

Run these notebooks in DP_fhf/notebooks/ separately:

- 01_DPFedAvg_AusGrid2013_Sweep_q0p1.ipynb
- 01_DPFedAvg_AusGrid2013_Sweep_q0p5.ipynb
- 01_DPFedAvg_AusGrid2013_Sweep_q1p0.ipynb

Each notebook fixes one sampling rate and scans 4 noise multipliers × 3 clipping
norms for 20 rounds per run, giving 36 runs in total. Run the notebooks sequentially
on a single GPU. Outputs share DP_fhf/outputs/ausgrid2013_sampling_sweep/, separated
by sampling rate. Each entry point refreshes all_sampling_rates_summary.csv on
completion; the global DP-FedAvg summary can also read these outputs.
This summary reports base-forecast metrics. The original Sweep_Reconciliation
still reads the older directory by default. The original package contained no
completed results or notebook outputs.

The original package inventory, CRC, notebook format/syntax, and source consistency
were checked. Full model training and Windows/CUDA execution must be performed
on the target computer in the order above.
