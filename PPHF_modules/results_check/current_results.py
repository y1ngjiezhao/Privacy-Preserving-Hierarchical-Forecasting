"""Rebuild result tables and figures exclusively from the current project runs."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PPHF_modules.paths import PROJECT_ROOT, RESULTS_DIR
from PPHF_modules.results_check.deduplication_results import generate_deduplication_results as summary

SYSTEMS = ('FHF', 'HFF', 'HFFPS')
CORE_METHODS = ('base', 'bu', 'mint_shrinkage', 'mint_var', 'naive')
COLOURS = {'FHF': '#247f9e', 'HFF': '#d8812c', 'HFFPS': '#5470a7'}
ANALYSIS_VERSION = 2


def dataset_config(dataset: str, root=PROJECT_ROOT, source_dir=None, horizons=None):
    if dataset not in summary.DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    root = Path(root).resolve()
    config = dict(summary.DATASETS[dataset])
    config['root'] = Path(source_dir).resolve() if source_dir else root / f'fhf_dedup_{dataset}'
    folder = 'GEFCOM2017' if dataset == 'gef2017' else 'AusGrid2013'
    config['bottom_tags'] = root / 'Datasets' / folder / 'tags_dedup.pkl'
    if horizons is not None:
        config['horizons'] = tuple(horizons)
    if not set(config['horizons']) <= set(summary.DATASETS[dataset]['horizons']):
        raise ValueError('Requested horizons are not available for this dataset.')
    return config


def training_scales(artifact: Path) -> pd.DataFrame:
    """Recover the same training-target m=1 scales used by saved run metrics."""
    with np.load(artifact, allow_pickle=False) as saved:
        ids = saved['client_ids'].astype(int)
        names = saved['unique_ids'].astype(str)
        targets = saved['train_targets'].astype(np.float64)
    if targets.ndim != 3 or targets.shape[0] != len(ids) or targets.shape[1] < 2:
        raise ValueError(f'Invalid training targets: {artifact}')
    differences = np.diff(targets, axis=1)
    table = pd.DataFrame({'node_id': ids, 'unique_id': names,
                          'mase_scale': np.abs(differences).mean(axis=(1, 2)),
                          'rmsse_scale': np.square(differences).mean(axis=(1, 2))})
    if table[['mase_scale', 'rmsse_scale']].le(1e-12).any().any():
        raise ValueError(f'Nonpositive training scale: {artifact}')
    if not np.isfinite(table[['mase_scale', 'rmsse_scale']]).all().all():
        raise ValueError(f'Non-finite training scale: {artifact}')
    return table


def source_manifest(paths, root=PROJECT_ROOT):
    """Content fingerprints make reuse depend on the actual input files."""
    root = Path(root).resolve()
    records = []
    for path in sorted({Path(p).resolve() for p in paths}):
        if not path.is_file():
            raise FileNotFoundError(f'Current experiment input is missing: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        records.append({'path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                        'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    return records


def _input_paths(config):
    paths = [config['bottom_tags'], Path(__file__), Path(summary.__file__)]
    for system in SYSTEMS:
        for horizon in config['horizons']:
            prefix = f'fh{horizon}_{system}'
            paths.extend(config['root'] / f'{prefix}_{suffix}' for suffix in
                         ('per_node_metrics.csv', 'forecasts.csv', 'training_artifacts.npz', 'timing.csv'))
            log = 'round_logs.csv' if system == 'FHF' else 'cloud_round_logs.csv'
            paths.append(config['root'] / f'{prefix}_{log}')
    return paths


def _aggregate_coherence(parent, extra):
    keys = ['dataset', 'source_system', 'framework', 'horizon', 'method_internal', 'method', *extra]
    result = (parent.assign(_forecast_ratio_sum=lambda x: x.average_scale_free_incoherence_ratio_pct*x.n_time_points,
                            _error_ratio_sum=lambda x: x.average_error_scaled_incoherence_ratio_pct*x.n_time_points)
              .groupby(keys, as_index=False)
              .agg(incoherence_total=('incoherence_total', 'sum'),
                   parent_actual_demand_total=('parent_actual_demand_total', 'sum'),
                   n_parent_time_comparisons=('n_time_points', 'sum'),
                   n_parents=('parent_id', 'nunique'), MASI=('MASI', 'mean'),
                   _forecast_ratio_sum=('_forecast_ratio_sum', 'sum'),
                   _error_ratio_sum=('_error_ratio_sum', 'sum')))
    result['incoherence_ratio_pct'] = 100*result.incoherence_total/result.parent_actual_demand_total
    result['average_scale_free_incoherence_ratio_pct'] = result.pop('_forecast_ratio_sum')/result.n_parent_time_comparisons
    result['average_error_scaled_incoherence_ratio_pct'] = result.pop('_error_ratio_sum')/result.n_parent_time_comparisons
    return result


def prepare_current_results(dataset, *, root=PROJECT_ROOT, output_dir=None,
                            source_dir=None, horizons=None, force=False):
    """Return freshly rebuilt or content-verified current result tables."""
    root = Path(root).resolve()
    config = dataset_config(dataset, root, source_dir, horizons)
    output = Path(output_dir).resolve() if output_dir else root/'Results_check/deduplication_results'/dataset
    output.mkdir(parents=True, exist_ok=True)
    inputs = source_manifest(_input_paths(config), root)
    signature = {'analysis_version': ANALYSIS_VERSION, 'dataset': dataset,
                 'horizons': list(config['horizons']), 'sources': inputs}
    manifest_path = output/'source_manifest.json'
    names = ['per_node', 'accuracy_by_level', 'hierarchy_audit', 'coherence_by_parent',
             'coherence_by_level', 'coherence_by_method', 'mcb_mase', 'mcb_rmsse',
             'timings', 'round_logs']
    if not force and manifest_path.exists():
        stored = json.loads(manifest_path.read_text())
        if stored.get('signature') == signature and all((output/f'{name}.csv').exists() for name in names):
            print(f'{dataset}: verified inputs unchanged; loading current result tables.', flush=True)
            return {name: pd.read_csv(output/f'{name}.csv') for name in names}
    print(f'{dataset}: rebuilding from {config["root"]}', flush=True)
    per_node, accuracy, audit = summary.load_accuracy(dataset, config)
    nodes = per_node[['unique_id', 'structural_level']].drop_duplicates().rename(columns={'structural_level':'level'})
    relations, relation_levels = summary.parent_relations(nodes)
    parent_parts, log_parts, timing_rows = [], [], []
    for system in SYSTEMS:
        for horizon in config['horizons']:
            prefix = f'fh{horizon}_{system}'
            print(f'  {prefix}: coherence and training records', flush=True)
            path = config['root']/f'{prefix}_forecasts.csv'
            parent = summary.incoherence_for_file(path, dataset, system, summary.SYSTEM_LABELS[system],
                                                  horizon, relations, relation_levels)
            scales = training_scales(config['root']/f'{prefix}_training_artifacts.npz')
            parent = parent.merge(scales[['unique_id','mase_scale']], left_on='parent_id',
                                  right_on='unique_id', validate='many_to_one').drop(columns='unique_id')
            parent['MASI'] = parent.incoherence_total / parent.n_time_points / parent.mase_scale
            parent_parts.append(parent)
            log_name = 'round_logs.csv' if system=='FHF' else 'cloud_round_logs.csv'
            log = pd.read_csv(config['root']/f'{prefix}_{log_name}')
            log['training_round'] = log['round'] if system=='FHF' else log['cloud_round']
            log['source_system'] = system; log['horizon'] = horizon; log['dataset'] = dataset
            log_parts.append(log)
            timer = pd.read_csv(config['root']/f'{prefix}_timing.csv').set_index('module')['seconds']
            timing_rows.append({'dataset': dataset, 'source_system': system, 'horizon': horizon,
                                'rounds_completed': len(log), 'training_sec': timer['fl_training_sec'],
                                'training_sec_per_round': timer['fl_training_sec']/len(log),
                                'setup_sec': timer.get('setup_sec', np.nan),
                                'reconciliation_sec': timer.get('reconciliation_sec', np.nan),
                                'total_sec': timer['total_sec']})
    parent = pd.concat(parent_parts, ignore_index=True)
    parent = parent.loc[~(parent.method_internal.eq('naive') & parent.source_system.ne('FHF'))].copy()
    tables = {'per_node': per_node, 'accuracy_by_level': accuracy, 'hierarchy_audit': audit,
              'coherence_by_parent': parent, 'coherence_by_level': _aggregate_coherence(parent, ['level']),
              'coherence_by_method': _aggregate_coherence(parent, []),
              'mcb_mase': summary.wide_mcb_table(per_node, 'MASE'),
              'mcb_rmsse': summary.wide_mcb_table(per_node, 'RMSSE'),
              'timings': pd.DataFrame(timing_rows), 'round_logs': pd.concat(log_parts, ignore_index=True)}
    numeric = parent[['incoherence_total','MASI','average_scale_free_incoherence_ratio_pct',
                      'average_error_scaled_incoherence_ratio_pct']]
    if not np.isfinite(numeric).all().all():
        raise ValueError('Current coherence tables contain non-finite values.')
    for name, table in tables.items():
        table.to_csv(output/f'{name}.csv', index=False)
    verification = {'data_scope': 'current_deduplicated_non_dp_runs', 'dataset': dataset,
                    'structural_level_counts': config['expected_structural_level_counts'],
                    'evaluation_level_counts': config['expected_evaluation_level_counts'],
                    'horizons': list(config['horizons']), 'source_systems': list(SYSTEMS),
                    'saved_metrics_not_retrained': True,
                    'accuracy_aggregation': 'equal node mean within configured evaluation level',
                    'evaluation_level_policy': config['evaluation_level_policy'],
                    'coherence_aggregation': 'structural parent-child constraints; MASI equal-parent mean',
                    'max_reconciled_MASI': float(parent.loc[parent.method_internal.isin(
                        ('bu','mint_cov','mint_shrinkage','mint_ols','mint_var','wls_structure')), 'MASI'].max())}
    manifest_path.write_text(json.dumps({'signature': signature, 'validation': verification,
                                        'generated_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    (output/'validation.json').write_text(json.dumps(verification, indent=2))
    return tables


def core_accuracy_table(tables):
    frame = tables['accuracy_by_level']
    selected = frame.loc[frame.method_internal.isin(CORE_METHODS)]
    return selected.pivot(index=['horizon','source_system','method_internal'],
                          columns='hierarchy_level', values=['MAE','MASE','RMSSE']).sort_index()


def plot_training(tables, output_dir):
    import matplotlib.pyplot as plt
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    logs = tables['round_logs']; horizons = sorted(logs.horizon.unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 3.6), squeeze=False)
    for ax, horizon in zip(axes[0], horizons):
        for system in SYSTEMS:
            rows = logs.loc[logs.horizon.eq(horizon) & logs.source_system.eq(system)]
            ax.plot(rows.training_round, rows.avg_normalised_loss, label=system, color=COLOURS[system])
        ax.set(title=f'Forecast horizon {horizon}', xlabel='Training round', ylabel='Normalised training loss')
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output/'training_loss.png', dpi=180, bbox_inches='tight')
    fig.savefig(output/'training_loss.pdf', bbox_inches='tight')
    return fig


def plot_timing(tables, output_dir):
    import matplotlib.pyplot as plt
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    pivot = tables['timings'].pivot(index='horizon', columns='source_system', values='training_sec_per_round')
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    pivot[list(SYSTEMS)].plot.bar(ax=ax, color=[COLOURS[s] for s in SYSTEMS], rot=0)
    ax.set(xlabel='Forecast horizon', ylabel='Seconds per round (saved run)')
    ax.legend(title=None, frameon=False); ax.grid(axis='y', alpha=.2); fig.tight_layout()
    fig.savefig(output/'training_time_per_round.png', dpi=180, bbox_inches='tight')
    fig.savefig(output/'training_time_per_round.pdf', bbox_inches='tight')
    return fig
