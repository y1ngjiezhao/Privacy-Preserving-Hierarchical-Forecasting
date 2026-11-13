# PPHF modules

- `fhf/`: model definitions, FHF and HFF/HFFPS training, aggregation,
  reconciliation and training-artifact export.
- `dp_fhf/`: DP-FedAvg CLI, notebook runtime, privacy accounting,
  DP covariance/O2MR and saved-run reconciliation.
- `results_check/`: result analysis and publication labels, with the original
  analysis subfolders retained. R scripts are grouped with the matching analyses.
- `validation/`: active notebook inventory, syntax checks and isolated execution.
- `paths.py`: separates package code locations from dataset/experiment locations.
- `notebook_paths.py`: applies canonical imports to newly generated notebooks.

Use fully qualified imports beginning with `PPHF_modules`. Notebook entry points
remain in their original directories. Generated results remain under `DP_fhf/`,
`Results_check/`, or the corresponding `fhf_dedup_*` experiment folder.

Privacy accounting's project-local dependencies and its restoration requirements
are in `dp_fhf/_privacy_dependencies/` and `dp_fhf/requirements-privacy.txt`.
The Windows dependency list also covers data preparation and analysis notebooks.
The four result notebooks use current saved deduplicated runs exclusively;
`current_results.py` provides shared tables and source fingerprints.

Examples from the project root:

```sh
python -m PPHF_modules.dp_fhf.run_dp_fedavg --help
python -m unittest discover -s PPHF_modules/dp_fhf -p 'test_*.py'
python -m PPHF_modules.validation.check_notebooks
```

This package contains no saved experiment results. Generate the required training
outputs before running saved-result analyses and their data-dependent tests.
