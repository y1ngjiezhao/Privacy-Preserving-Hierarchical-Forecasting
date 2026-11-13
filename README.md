# Privacy-Preserving Hierarchical Forecasting

Research code and input datasets for privacy-preserving federated hierarchical forecasting (PPHF), including Federated Hierarchical Forecasting (FHF and its hierarchical variants), DP-FedAvg, reconciliations, differential privacy (with RDP accounting), and result analysis.

## Included files

- `Datasets/`: choose AusGrid2013 and GEFCOM2017, summing matrices, mappings, tags, and data preparation.
- `PPHF_modules/`: federated training, aggregation, distributed computation reconciliation, privacy accounting, analysis and summary.
- `fhf_dedup_ausgrid2013/` and `fhf_dedup_gef2017/`: FHF experiment notebooks (without privacy-preserving considerations).
- `DP_fhf/notebooks/`: Privacy experiments (DP-FedAvg, DP-based estimation for covariance matrix and DP-based forecast summation), in which DP-FedAvg considers different configurations of sampling, sweep, and summary notebooks.
- `Results_check/`: analysis notebooks, current results (PDFs), and small source scripts. Other large results are saved locally and can be accessed with authorisation.
- `windows_gpu/`: Windows setup scripts, environment checks, and dependency versions.

Note that generated forecast CSVs, residuals, trained models, NPZ training artifacts, experiment logs, caches, notebook checkpoints, and notebook cell outputs are unuploaded (since files' size). Figures are supplied as PDFs. No training runs were rerun to prepare this repository.

## Restore and verify the datasets

Six large input CSV files are stored as lossless `.csv.gz` archives. All other dataset files retain their original format. From the repository root, run:

```sh
python tools/restore_datasets.py
```

Use `python3` if that is the Python command on your machine. The script restores the original CSV filenames and verifies SHA-256 checksums. It preserves the archives and refuses to overwrite a different existing file. Restoration needs approximately 838 MiB for the complete input dataset collection, in addition to the retained compressed archives.

To verify the included archives and dataset files without restoring CSVs:

```sh
python tools/restore_datasets.py --verify-only
```

`Datasets/DATASET_MANIFEST.json` records original and uploaded file sizes and checksums. `UPLOAD_MANIFEST.json` records the original project files selected for this snapshot, including notebook output clearing, lossless compression, and English translation. Original-source checksums identify the pre-translation local files; repository checksums identify the current translated files. Repository-specific README, restore script, and Git configuration files are additional files.

## Run the experiments

1. Restore datasets as above. The prepared inputs are included; use the notebooks in `Datasets/` if you need to rebuild them.
2. Set up the Python environment. The existing dependency versions are in `windows_gpu/requirements.txt`, and privacy-accounting requirements are in `PPHF_modules/dp_fhf/requirements-privacy.txt`. PyTorch is installed separately. Windows GPU setup instructions are in `windows_gpu/README_zh.md`.
3. Run the ordinary experiments in `fhf_dedup_ausgrid2013/` and `fhf_dedup_gef2017/`. These generate the forecasts and training artifacts required by the saved-result analyses.
4. Run the analysis notebooks under `Results_check/` after generating those required results.
5. For DP-FedAvg and sampling experiments, follow `DP_fhf/notebooks/README.md`. Generate each experiment's outputs before running its summary or reconciliation notebook. DP covariance and O2MR use the saved ordinary FHF results.

Run notebooks from this repository's directory structure so their canonical `PPHF_modules` imports and relative dataset paths resolve. Generated output directories are created by the code.

An English translation of the earlier migration README is at `docs/README_migration_original_zh.md` for its detailed run sequence. Its historical package-content statements describe the earlier migration package; this GitHub snapshot additionally includes result PDFs and compressed datasets as documented above.

Dataset source documents are retained in their respective dataset folders. This repository does not add or change the original dataset or third-party dependency licensing terms.
