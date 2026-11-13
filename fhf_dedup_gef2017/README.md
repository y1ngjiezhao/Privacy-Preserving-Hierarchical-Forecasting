# fhf_dedup_gef2017

- Dataset: `Datasets/GEFCOM2017/GEF2017_univarivate.csv`
- Hierarchy mode: `dedup`
- Frameworks: FHF, HFF, HFFPS
- Forecast horizons: 1, 3, 6
- Non-DP parameters/evaluation: copied from the matching notebooks in the read-only FHF reference project.
- DP-FedAvg launcher: `DP_fhf/notebooks/01_DPFedAvg_NDSH_Matrix.ipynb`

Notebook outputs are intentionally not checked in; running a notebook writes its artifacts into this folder.

## Client sampling

Ready-to-run examples: `FHF_Sampling_fh1.ipynb`, `HFF_Sampling_fh1.ipynb`,
and `HFFPS_Sampling_fh1.ipynb`. Use the `neural-network` kernel and Run All.
Default sampling rate is `0.1`; edit `args.sampling_rate` and
`args.sampling_seed` in the parameter cell. Sampling is uniform without
replacement with `ceil(q * N)` trainable nodes per round. HFF/HFFPS select
among all trainable series nodes once per cloud round.

New examples write into `sampling_experiments/` and leave historical results
in place. Existing notebooks also support sampling, defaulting to `1.0` to
preserve their previous training. Partial runs use a separate
`sampling_runs/q_<rate>__seed_<seed>/` output directory.
