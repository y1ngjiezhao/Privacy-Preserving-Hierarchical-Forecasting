# Current paper results

Updated: 2026-09-28. This historical index describes automatically verified
manuscript-result exports and provenance in the full local project. Original
experiment data, historical results, and old manuscript backups were retained at
the time unless superseded by the cleanup below. Generated CSVs and audit JSONs
mentioned here are omitted from this GitHub code-and-data snapshot.

## Current sources

| Manuscript component | Authoritative source results | Current configuration |
|---|---|---|
| Two main result tables and MASI | `Results_check/deduplication_results/{gef2017,ausgrid2013}/` | Deduplicated hierarchy; summaries by evaluation level; MASI scaled by training naive errors |
| Figures 5–8 | MCB PDFs in the directories above | GEF h=1/3/6; Ausgrid h=1; ranks displayed to 1/3 decimal places; 14 pt legends |
| DP-FedAvg, 100 rounds | `DP_fhf/outputs/dp_100round_reconciliation/` | 2026-09-25 batch; σ=0.3/1.2, C=1, δ=1e-5 |
| DP-FedAvg, 20-round sweep | `DP_fhf/outputs/dp_sweep_reconciliation/` | 2026-09-24 batch; 24 runs, each with base/mints/wlsv/ols |
| Fixed residual bounds | `DP_fhf/outputs/02_dp_covariance/<dataset>/no_debias/` | ±0.0005, ε=1, seed=42, rho=1e-6; no noise variance correction |
| Fixed bounds versus Non-DP/OLS | `DP_fhf/outputs/02_dp_covariance/no_debias_comparison/paper_comparison.csv` | Only non_private/reference/no_debias |
| Quantile residual bounds | `DP_fhf/outputs/02_dp_covariance/multiscale_no_debias/` | 99.5%, 75%, 50%, 25%, 0.5%; non_dp_forecasts branch |
| Private forecast release | `DP_fhf/outputs/03_dp_o2mr/<dataset>/` | non_dp_forecasts; ε=1/5/10/20; MinTs and WLSv; fixed P |

Paths are relative to the FHF_NDSH project root. These CSVs are exports of
unrounded source results, rather than manuscript numbers copied back into
experiment files.

## Rules after cleanup

On 2026-09-28, the old corrected results and baseline_recheck, old 0.05% snapshots,
pooled legacy MCB figures, and old manuscript backup/staged snapshots were deleted
as requested. The full local archive records this in cleanup_20260928.json.

- Use only `<dataset>/no_debias/` for fixed bounds.
- `no_debias_comparison/paper_comparison.csv` contains only the three current
  comparisons: non_private, reference (OLS), and no_debias.
- Use retention fractions of 99.5%, 75%, 50%, 25%, and 0.5%.
  `multiscale_retention_0p5_update/` retains only current 0.5% update results and provenance.
- BO-FHF in source CSVs is a historical internal key corresponding to LN-FHF in
  the manuscript. It does not imply that the numerical results are outdated.
- The 100-round and 20-round DP-FedAvg results, private forecast releases, and
  original training inputs remain unchanged.

## Verification and reproduction

```bash
python PPHF_modules/results_check/audit_paper_results.py --paper-root /path/to/manuscript
```

Requires numpy, pandas, and pypdf. Unrounded CSV exports are updated only after all
checks pass.

To reproduce fixed bounds and reference methods fully:

```bash
python PPHF_modules/dp_fhf/run_covariance_no_debias_comparison.py
python PPHF_modules/dp_fhf/summarize_covariance_no_debias.py
```

These scripts produce only current no_debias, Non-DP, and OLS results. They neither
depend on nor rebuild old corrected configurations or manuscript snapshots.
The default covariance notebook also uses no-debias. The lower-level API/CLI
supports other explicitly configured experiments outside the current manuscript.

Earlier audit reports are historical records; statements about retaining old files
were superseded by the cleanup. Use this index and the latest verification.json
in the full local archive to identify the current data.
