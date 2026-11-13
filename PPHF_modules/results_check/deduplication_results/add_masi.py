#!/usr/bin/env python3
"""Add training-naive-scaled incoherence without changing existing metrics.

Uses the exact m=1 training-target convention of the experiment notebooks.
No forecast files need reprocessing: mean(abs(discrepancy)) is already saved.
"""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
OUT = RESULTS_DIR / "deduplication_results"
RAW = {
    'gef2017': ROOT / 'Datasets/GEFCOM2017/GEF2017_univarivate.csv',
    'ausgrid2013': ROOT / 'Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv',
}
KEYS = ['dataset', 'source_system', 'framework', 'horizon', 'method_internal', 'method']
FORECAST = 'average_scale_free_incoherence_ratio_pct'
ERROR = 'average_error_scaled_incoherence_ratio_pct'


def scaled_incoherence(total, count, scale):
    """Zero/near-zero scales are undefined, never silently scored as coherent."""
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.nan
    if count <= 0:
        raise ValueError('Comparison count must be positive')
    return total / count / scale


def training_scales(dataset, parents, horizons):
    raw = pd.read_csv(RAW[dataset], usecols=['unique_id', 'ds', 'y'])
    raw = raw.loc[raw.unique_id.isin(parents)].copy()
    raw['ds'] = pd.to_datetime(raw.ds)
    records = []
    for parent, group in raw.groupby('unique_id'):
        group = group.sort_values('ds')
        if group.ds.duplicated().any() or not np.isfinite(group.y).all():
            raise ValueError(f'Invalid raw training series: {parent}')
        y = group.y.to_numpy(float)
        train_end = int(0.8 * len(y))
        for horizon in horizons:
            # The notebooks discard 48 lag rows and construct h target columns.
            n = train_end - 48 - (horizon - 1)
            if n <= 1:
                raise ValueError(f'Insufficient training history: {parent}')
            targets = np.column_stack([y[48+j:48+j+n] for j in range(horizon)])
            scale = float(np.abs(np.diff(targets, axis=0)).mean())
            records.append(dict(parent_id=parent, horizon=horizon,
                                mase_training_scale=scale, seasonal_period=1,
                                n_training_windows=n,
                                training_last_time=str(group.ds.iloc[train_end-1])))
    scales = pd.DataFrame(records)
    if set(scales.parent_id) != set(parents):
        raise ValueError('Missing parent training series')
    # Independent audit: saved MAE / MASE must recover this training-only scale,
    # for every available method/framework, not just the base forecast.
    max_relative_error = 0.0
    for h in horizons:
        for system in ('FHF', 'HFF', 'HFFPS'):
            metrics = pd.read_csv(ROOT / f'fhf_dedup_{dataset}/fh{h}_{system}_per_node_metrics.csv')
            paired = metrics.merge(scales.loc[scales.horizon.eq(h)],
                                   left_on='unique_id', right_on='parent_id')
            recovered = paired.MAE / paired.MASE
            valid = np.isfinite(recovered) & paired.mase_training_scale.gt(1e-12)
            if not valid.any():
                raise ValueError('No saved MASE values available to validate scales')
            relative = np.abs(recovered[valid] / paired.loc[valid, 'mase_training_scale'] - 1)
            max_relative_error = max(max_relative_error, float(relative.max()))
            if not np.allclose(recovered[valid], paired.loc[valid, 'mase_training_scale'], rtol=1e-7):
                raise ValueError(f'Training scale differs from saved MASE: {dataset}/{h}/{system}')
    return scales, max_relative_error


def enrich_dataset(dataset):
    folder = OUT / dataset
    parent_path = folder / 'deduplicated_incoherence_by_parent.csv'
    original = pd.read_csv(parent_path)
    # Permit repeat execution while checking all pre-existing metric values.
    old = original.drop(columns=['MASI', 'mase_training_scale'], errors='ignore')
    scales, scale_error = training_scales(dataset, old.parent_id.unique(), sorted(old.horizon.unique()))
    parent = old.merge(scales[['parent_id', 'horizon', 'mase_training_scale']],
                       on=['parent_id', 'horizon'], validate='many_to_one', sort=False)
    parent['MASI'] = [scaled_incoherence(t, n, s) for t, n, s in
                      zip(parent.incoherence_total, parent.n_time_points, parent.mase_training_scale)]
    pd.testing.assert_frame_equal(old, parent[old.columns])
    # Parent-equal mean after time averaging; each method uses the same parents.
    def aggregate(keys):
        return parent.groupby(keys, as_index=False).agg(
            MASI=('MASI', 'mean'), masi_valid_parents=('MASI', 'count'),
            masi_total_parents=('parent_id', 'size'))
    levels = aggregate(KEYS + ['level'])
    methods = aggregate(KEYS)
    # Existing comparisons are balanced; prove this before comparing rankings.
    if parent.groupby(['horizon']).n_time_points.nunique().gt(1).any():
        raise ValueError('Unequal parent-time coverage; review weighting before comparison')
    for filename, additions, keys in [
        ('deduplicated_incoherence_by_level.csv', levels, KEYS + ['level']),
        ('deduplicated_average_scale_free_incoherence_by_method.csv', methods, KEYS),
    ]:
        path = folder / filename
        prior = pd.read_csv(path).drop(columns=['MASI', 'masi_valid_parents', 'masi_total_parents'], errors='ignore')
        updated = prior.merge(additions, on=keys, validate='one_to_one', sort=False)
        pd.testing.assert_frame_equal(prior, updated[prior.columns])
        updated.to_csv(path, index=False)
        if 'by_method' in filename:
            comparison = updated[KEYS + ['MASI', FORECAST, ERROR, 'masi_valid_parents', 'masi_total_parents']].copy()
    # Do not rank meaningless floating-point differences among coherent methods.
    for metric, rank in [('MASI', 'MASI_rank'), (FORECAST, 'forecast_scale_rank'), (ERROR, 'error_scale_rank')]:
        values = comparison[metric].mask(comparison[metric].abs().lt(1e-10), 0)
        comparison[rank] = values.groupby(comparison.horizon).rank(method='min')
        base_values = values.where(comparison.method_internal.eq('base'))
        comparison[rank + '_base_only'] = base_values.groupby(comparison.horizon).rank(method='min')
    comparison = comparison.sort_values(['horizon', 'MASI_rank', 'method'])
    comparison.to_csv(folder / 'incoherence_three_metric_comparison.csv', index=False)
    paper_path = folder / 'deduplicated_paper_summary.csv'
    paper = pd.read_csv(paper_path)
    paper = paper.drop(columns=[c for c in paper if c == 'MASI' or c.startswith('MASI_L')], errors='ignore')
    level_wide = levels.pivot(index=KEYS, columns='level', values='MASI').add_prefix('MASI_').reset_index()
    paper = paper.merge(methods[KEYS + ['MASI']], on=KEYS, validate='one_to_one')
    paper = paper.merge(level_wide, on=KEYS, validate='one_to_one')
    paper.to_csv(paper_path, index=False)
    parent.to_csv(parent_path, index=False)
    scales.to_csv(folder / 'masi_training_scale_audit.csv', index=False)
    audit = dict(primary_metric='MASI', seasonal_period=1,
                 scale_definition='mean(abs(diff(training target matrix, axis=0)))',
                 aggregation='equal parent mean of time-averaged scaled discrepancies',
                 zero_scale_policy='scale <= 1e-12: undefined; report valid and total parents',
                 max_relative_difference_from_saved_MASE=scale_error,
                 invalid_scale_count=int((~np.isfinite(scales.mase_training_scale) | scales.mase_training_scale.le(1e-12)).sum()),
                 existing_metrics_unchanged=True)
    (folder / 'masi_validation.json').write_text(json.dumps(audit, indent=2))
    return comparison


def main():
    comparisons = pd.concat([enrich_dataset(name) for name in RAW], ignore_index=True)
    base = comparisons.loc[comparisons.method_internal.eq('base')]
    lines = ['# Incoherence comparison', '',
             'Primary: MASI (multiples). Secondary: forecast-scaled and error-scaled ratios (%).', '',
             'MASI uses the same training-only m=1 naive scale as the saved experimental MASE: '
             '80% training split, 48 lag rows, mean absolute adjacent differences over the training target matrix. '
             'This is not seasonal MASE. Each horizon has its own target window, but the scale is shared across methods.', '',
             'Existing two ratios and the historical total-demand ratio are retained unchanged. '
             'The numerator is the same saved absolute parent/direct-child discrepancy, averaged in time and then equally over parents. '
             'All evaluated parents have equal time counts within each horizon. MASI is not a percentage.', '',
             '| Dataset | Horizon | Method | MASI | Forecast scale (%) | Error scale (%) | Base rank: MASI / forecast / error |',
             '|---|---:|---|---:|---:|---:|---|']
    for _, r in base.iterrows():
        ranks = ' / '.join(str(int(r[c + '_base_only'])) for c in ['MASI_rank','forecast_scale_rank','error_scale_rank'])
        lines.append(f'| {r.dataset} | {r.horizon} | {r.method} | {r.MASI:.6f} | {r[FORECAST]:.6f} | {r[ERROR]:.6f} | {ranks} |')
    lines += ['', 'All-method comparisons and level/parent results are saved per dataset. '
              'Rank values treat magnitudes below 1e-10 as numerical zero; original metric values are not rounded in CSVs. '
              'Reconciled forecasts are evaluated as their own methods; base forecasts are evaluated without reconciliation.']
    (OUT / 'MASI_comparison.md').write_text('\n'.join(lines) + '\n')
    print(base[['dataset','horizon','method','MASI',FORECAST,ERROR]].to_string(index=False))


if __name__ == '__main__':
    main()
