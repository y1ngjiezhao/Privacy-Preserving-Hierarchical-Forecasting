"""Generate the four current-project result notebooks without external archives."""
from __future__ import annotations
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PPHF_modules.paths import PROJECT_ROOT
from PPHF_modules.notebook_paths import BOOTSTRAP


def cell(kind, source):
    item = {'cell_type':kind,'metadata':{},'source':source.strip().splitlines(keepends=True)}
    if kind=='code':item.update(execution_count=None,outputs=[])
    return item


def write(relative, cells):
    for i, item in enumerate(cells):item['id']=f'current-{i}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (FHF RTX 4070 SUPER)',
               'language':'python','name':'fhf-4070-super'},'language_info':{'name':'python','version':'3.12'},
               'pphf_scope':'current_deduplicated_project_results'},'nbformat':4,'nbformat_minor':5}
    path=PROJECT_ROOT/relative
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print('Wrote',path)


SETUP=BOOTSTRAP+'''
ROOT = PROJECT_ROOT
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from PPHF_modules.results_check.current_results import (
    prepare_current_results, core_accuracy_table, plot_training, plot_timing,
)
'''


def overview(dataset, title, relative):
    cells=[cell('markdown',f'''# {title}

Use the completed ordinary FHF, HFF, and HFFPS experiments in this project's `fhf_dedup_{dataset}/` directory.
Show accuracy, training records, timing, and hierarchy consistency on the current deduplicated data. Include one shared Naive baseline.
Rebuild all statistics from the current per-node metrics, forecast CSVs, and training residual files, without starting model training.'''),
        cell('code',SETUP),
        cell('code',f'''DATASET = {dataset!r}
SOURCE_DIR = ROOT / 'fhf_dedup_{dataset}'
OUTPUT_DIR = ROOT / 'Results_check' / 'deduplication_results' / DATASET
RECOMPUTE = False  # Rebuild automatically when input contents change. True forces recomputation.
print('Source:', SOURCE_DIR)
print('Output:', OUTPUT_DIR)'''),
        cell('code', '''tables = prepare_current_results(DATASET, root=ROOT, source_dir=SOURCE_DIR,
                                 output_dir=OUTPUT_DIR, force=RECOMPUTE)
display(tables['hierarchy_audit'])'''),
        cell('markdown','''## Accuracy comparison

Compute MAE, MASE, and RMSSE per node, then average equally within each semantic evaluation level.
Series that become leaves after deduplication belong to the bottom level. Retain structural depth for subsequent parent-child constraint checks.
The core table includes base, BU, MinT-shrinkage, WLSv, and the shared Naive baseline. See `accuracy_by_level.csv` for all methods.'''),
        cell('code', '''core = core_accuracy_table(tables)
display(core)
core.to_csv(OUTPUT_DIR / 'core_accuracy_by_level.csv')'''),
        cell('markdown','''## All methods and MCB inputs

The table below retains all reconciliation methods. Pair MCB inputs by forecast horizon and node in the two CSVs,
`mcb_mase.csv` and `mcb_rmsse.csv`. MCB uses base, BU, MinT-shrinkage, and WLSv for each of the three frameworks,
plus the shared Naive baseline, giving 13 methods. The full accuracy table retains all reconciliation methods.'''),
        cell('code', '''display(tables['accuracy_by_level'])
print('MCB MASE shape:', tables['mcb_mase'].shape)
display(tables['mcb_mase'].head())'''),
        cell('markdown','''## Training loss and timing

Figures use logs and timings saved by the source experiments, rather than runtimes measured on the current computer.
HFF/HFFPS round counts refer to cloud rounds.'''),
        cell('code', '''figure = plot_training(tables, OUTPUT_DIR)
plt.close(figure)
display(Image(filename=str(OUTPUT_DIR / 'training_loss.png')))
display(tables['timings'])
figure = plot_timing(tables, OUTPUT_DIR)
plt.close(figure)
display(Image(filename=str(OUTPUT_DIR / 'training_time_per_round.png')))'''),
        cell('markdown','''## Hierarchy consistency

Compute parent-child discrepancies from structural levels. Normalize MASI by first-difference training-target scales and weight parents equally.
Also report discrepancies normalized by forecast and error scales. Report numerical consistency and forecast accuracy separately.'''),
        cell('code', '''coherence = tables['coherence_by_level']
display(coherence.loc[coherence.method_internal.isin(['base','bu','mint_shrinkage','mint_var','naive']),
    ['horizon','source_system','method_internal','level','incoherence_total','MASI',
     'average_scale_free_incoherence_ratio_pct','average_error_scaled_incoherence_ratio_pct']])'''),
        cell('code', '''validation = json.loads((OUTPUT_DIR / 'validation.json').read_text())
display(pd.DataFrame([validation]))
print('Source fingerprints:', OUTPUT_DIR / 'source_manifest.json')
print('Generated tables:', sorted(p.name for p in OUTPUT_DIR.glob('*.csv')))''')]
    write(relative,cells)


def main():
    overview('ausgrid2013','AusGrid2013: current deduplicated experiment results (360 nodes)',
             'Results_check/ausgrid2013_results/ausgrid2013_results.ipynb')
    overview('gef2017','GEF2017: current deduplicated experiment results (10 nodes, 1/3/6-step horizons)',
             'Results_check/gef2017_results/gef2017_results.ipynb')
    write('Results_check/full_gef2017_results/full_gef2017_results.ipynb',[
        cell('markdown','''# GEF2017: all methods and hierarchy consistency checks

This entry point retains its original filename and analyzes the current project's 10-node deduplicated hierarchy.
It covers FHF, HFF, HFFPS, 1/3/6-step horizons, and all available reconciliation methods.
Audit per-node accuracy, structural levels, semantic evaluation levels, parent-child consistency, and input provenance separately.'''),
        cell('code',SETUP),
        cell('code', '''DATASET = 'gef2017'
SOURCE_DIR = ROOT / 'fhf_dedup_gef2017'
OUTPUT_DIR = ROOT / 'Results_check' / 'deduplication_results' / DATASET
RECOMPUTE = False
tables = prepare_current_results(DATASET, root=ROOT, source_dir=SOURCE_DIR,
                                 output_dir=OUTPUT_DIR, force=RECOMPUTE)'''),
        cell('markdown','''## Nodes and levels

Use structural levels for parent-child consistency and semantic evaluation levels for accuracy summaries.
A leaf retained after deduplication may be at structural level 2 while belonging to the bottom level for accuracy evaluation.'''),
        cell('code', '''nodes = tables['per_node'][['unique_id','structural_level','level','role']].drop_duplicates()
display(nodes.sort_values(['structural_level','unique_id']))
display(tables['hierarchy_audit'])'''),
        cell('markdown','## Accuracy across all methods and per-node results'),
        cell('code', '''display(tables['accuracy_by_level'])
display(tables['per_node'].head(40))
print('All node metrics:', OUTPUT_DIR / 'per_node.csv')'''),
        cell('markdown','## Parent-child constraints and normalized consistency'),
        cell('code', '''display(tables['coherence_by_method'])
parents = tables['coherence_by_parent']
display(parents[['horizon','source_system','method_internal','parent_id','level',
                 'incoherence_total','mase_scale','MASI','n_time_points']])'''),
        cell('markdown','''## Numerical tolerance checks

Compare the MASI of each reconciliation method with the specified tolerance and retain each diagnostic result.
Assess forecast accuracy using the MAE/MASE/RMSSE tables above.'''),
        cell('code', '''COHERENCE_TOLERANCE = 1e-6
checks = tables['coherence_by_method'].loc[
    ~tables['coherence_by_method'].method_internal.isin(['base','naive'])].copy()
checks['within_tolerance'] = checks.MASI <= COHERENCE_TOLERANCE
display(checks[['horizon','source_system','method_internal','MASI','within_tolerance']])
checks.to_csv(OUTPUT_DIR / 'coherence_checks.csv', index=False)'''),
        cell('markdown','## Input provenance and content fingerprints'),
        cell('code', '''manifest = json.loads((OUTPUT_DIR / 'source_manifest.json').read_text())
display(pd.DataFrame(manifest['signature']['sources']))
display(pd.DataFrame([manifest['validation']]))
assert set(tables['hierarchy_audit'].nodes) == {10}
assert set(tables['hierarchy_audit'].horizon) == {1,3,6}
assert set(tables['hierarchy_audit'].source_system) == {'FHF','HFF','HFFPS'}''')])
    write('Results_check/error_analysis/error_redistribution_analysis.ipynb',[
        cell('markdown','''# Error redistribution in the current deduplicated experiments

Use this project's results for 360 AusGrid2013 nodes and 10 GEF2017 nodes.
AusGrid2013 covers a 1-step horizon, and GEF2017 covers 1/3/6-step horizons. Both datasets include FHF, HFF, and HFFPS.
Compare each reconciliation method only with the base from the same training approach. Obtain scales from the corresponding saved training targets,
average within the current semantic levels, and use a calendar-day block bootstrap for 95% intervals.'''),
        cell('code',BOOTSTRAP+'''
ROOT = PROJECT_ROOT
import json
import pandas as pd
from IPython.display import Image, display
from PPHF_modules.results_check.error_analysis.error_redistribution_analysis import run_all
'''),
        cell('code', '''OUTPUT_DIR = ROOT / 'Results_check' / 'deduplication_results' / 'error_analysis'
N_BOOTSTRAP = 2000
RECOMPUTE = False
manifest = run_all(root=ROOT, output_root=OUTPUT_DIR,
                   n_bootstrap=N_BOOTSTRAP, force=RECOMPUTE)
display(manifest)'''),
        cell('markdown','''## Effects by level and original metric checks

Delta is the reconciled result minus base; negative values indicate lower error.
Compute MASE/RMSSE per node, then average equally within each semantic level. Compare the check table with aggregates of the current per-node metrics.'''),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    folder = OUTPUT_DIR / dataset
    print(dataset)
    display(pd.read_csv(folder / 'level_effects_bootstrap.csv'))
    checks = pd.read_csv(folder / 'reproduction_check.csv')
    display(checks.groupby('method')[['abs_diff_MASE','abs_diff_RMSSE']].max())
    assert checks[['abs_diff_MASE','abs_diff_RMSSE']].to_numpy().max() <= 1e-6'''),
        cell('markdown','## Bottom-level node improvement rates and large-error observations'),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    folder = OUTPUT_DIR / dataset
    print(dataset)
    display(pd.read_csv(folder / 'appendix_node_improvement.csv'))
    display(pd.read_csv(folder / 'largest_error_decile.csv'))
    display(Image(filename=str(folder / 'error_redistribution_heatmap.png')))'''),
        cell('markdown','''## Interpretation

Error deciles are defined by base errors, and patterns across groups are descriptive.
Bootstrap intervals describe changes within the saved experiments, not stability across training seeds.'''),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    records = json.loads((OUTPUT_DIR / dataset / 'validation.json').read_text())
    display(pd.DataFrame(records))
assert len(manifest) == 12
print('All outputs:', OUTPUT_DIR)
print('Source fingerprints:', OUTPUT_DIR / 'source_manifest.json')''')])


if __name__=='__main__':main()
