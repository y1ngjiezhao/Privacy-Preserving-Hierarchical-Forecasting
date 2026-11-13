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

使用本项目 `fhf_dedup_{dataset}/` 中的普通 FHF、HFF、HFFPS 已完成实验。
显示当前去重数据上的精度、训练记录、耗时和层级一致性。Naive 基线只保留一份。
全部统计从当前逐节点指标、预测 CSV 和训练残差文件重建，不启动模型训练。'''),
        cell('code',SETUP),
        cell('code',f'''DATASET = {dataset!r}
SOURCE_DIR = ROOT / 'fhf_dedup_{dataset}'
OUTPUT_DIR = ROOT / 'Results_check' / 'deduplication_results' / DATASET
RECOMPUTE = False  # 输入文件内容变化时会自动重建。True 强制重新计算。
print('Source:', SOURCE_DIR)
print('Output:', OUTPUT_DIR)'''),
        cell('code', '''tables = prepare_current_results(DATASET, root=ROOT, source_dir=SOURCE_DIR,
                                 output_dir=OUTPUT_DIR, force=RECOMPUTE)
display(tables['hierarchy_audit'])'''),
        cell('markdown','''## 精度比较

MAE、MASE、RMSSE 先逐节点计算，再在语义评估层级内等权平均。
去重后成为叶节点的序列归入底层，结构深度保留用于后续父子约束检查。
核心表包含 base、BU、MinT-shrinkage、WLSv 和共享 Naive；完整方法见 `accuracy_by_level.csv`。'''),
        cell('code', '''core = core_accuracy_table(tables)
display(core)
core.to_csv(OUTPUT_DIR / 'core_accuracy_by_level.csv')'''),
        cell('markdown','''## 全方法与 MCB 输入

下表保留全部协调方法。MCB 输入按预测步数和节点配对，两个 CSV 分别为
`mcb_mase.csv` 和 `mcb_rmsse.csv`。MCB 输入采用三个框架的 base、BU、MinT-shrinkage、WLSv
以及共享 Naive，共 13 种方法；完整精度表保留全部协调方法。'''),
        cell('code', '''display(tables['accuracy_by_level'])
print('MCB MASE shape:', tables['mcb_mase'].shape)
display(tables['mcb_mase'].head())'''),
        cell('markdown','''## 训练损失与耗时

图表使用源实验保存的日志和计时，不代表当前电脑重新运行的速度。
HFF/HFFPS 的轮数以 cloud round 为准。'''),
        cell('code', '''figure = plot_training(tables, OUTPUT_DIR)
plt.close(figure)
display(Image(filename=str(OUTPUT_DIR / 'training_loss.png')))
display(tables['timings'])
figure = plot_timing(tables, OUTPUT_DIR)
plt.close(figure)
display(Image(filename=str(OUTPUT_DIR / 'training_time_per_round.png')))'''),
        cell('markdown','''## 层级一致性

父子差异根据结构层级计算。MASI 用训练目标的一阶差分尺度归一化，父节点等权平均。
另列预测尺度和误差尺度归一化差异。数值一致性与预测精度分别报告。'''),
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
    overview('ausgrid2013','AusGrid2013 当前去重实验结果（360 节点）',
             'Results_check/ausgrid2013_results/ausgrid2013_results.ipynb')
    overview('gef2017','GEF2017 当前去重实验结果（10 节点，1/3/6 步）',
             'Results_check/gef2017_results/gef2017_results.ipynb')
    write('Results_check/full_gef2017_results/full_gef2017_results.ipynb',[
        cell('markdown','''# GEF2017 全方法与层级一致性检查

本入口保留原文件名，分析范围为当前项目的 10 节点去重层级。
覆盖 FHF、HFF、HFFPS，1/3/6 步预测，以及全部已有协调方法。
逐节点精度、结构层级、语义评估层级、父子一致性和输入来源分别审计。'''),
        cell('code',SETUP),
        cell('code', '''DATASET = 'gef2017'
SOURCE_DIR = ROOT / 'fhf_dedup_gef2017'
OUTPUT_DIR = ROOT / 'Results_check' / 'deduplication_results' / DATASET
RECOMPUTE = False
tables = prepare_current_results(DATASET, root=ROOT, source_dir=SOURCE_DIR,
                                 output_dir=OUTPUT_DIR, force=RECOMPUTE)'''),
        cell('markdown','''## 节点与层级

结构层级用于父子一致性，语义评估层级用于精度汇总。
去重保留的叶节点可能在结构第 2 层，但在精度评估时归入底层。'''),
        cell('code', '''nodes = tables['per_node'][['unique_id','structural_level','level','role']].drop_duplicates()
display(nodes.sort_values(['structural_level','unique_id']))
display(tables['hierarchy_audit'])'''),
        cell('markdown','## 全方法精度与逐节点结果'),
        cell('code', '''display(tables['accuracy_by_level'])
display(tables['per_node'].head(40))
print('All node metrics:', OUTPUT_DIR / 'per_node.csv')'''),
        cell('markdown','## 父子约束与归一化一致性'),
        cell('code', '''display(tables['coherence_by_method'])
parents = tables['coherence_by_parent']
display(parents[['horizon','source_system','method_internal','parent_id','level',
                 'incoherence_total','mase_scale','MASI','n_time_points']])'''),
        cell('markdown','''## 数值阈值检查

以下将协调方法的 MASI 与指定阈值比较，逐项保留诊断结果。
精度优劣由上面的 MAE/MASE/RMSSE 表判断。'''),
        cell('code', '''COHERENCE_TOLERANCE = 1e-6
checks = tables['coherence_by_method'].loc[
    ~tables['coherence_by_method'].method_internal.isin(['base','naive'])].copy()
checks['within_tolerance'] = checks.MASI <= COHERENCE_TOLERANCE
display(checks[['horizon','source_system','method_internal','MASI','within_tolerance']])
checks.to_csv(OUTPUT_DIR / 'coherence_checks.csv', index=False)'''),
        cell('markdown','## 输入来源与内容指纹'),
        cell('code', '''manifest = json.loads((OUTPUT_DIR / 'source_manifest.json').read_text())
display(pd.DataFrame(manifest['signature']['sources']))
display(pd.DataFrame([manifest['validation']]))
assert set(tables['hierarchy_audit'].nodes) == {10}
assert set(tables['hierarchy_audit'].horizon) == {1,3,6}
assert set(tables['hierarchy_audit'].source_system) == {'FHF','HFF','HFFPS'}''')])
    write('Results_check/error_analysis/error_redistribution_analysis.ipynb',[
        cell('markdown','''# 当前去重实验的误差重新分布分析

使用本项目 AusGrid2013 的 360 节点、GEF2017 的 10 节点结果。
AusGrid2013 覆盖 1 步，GEF2017 覆盖 1/3/6 步；两套数据都包括 FHF、HFF、HFFPS。
每种协调方法只与同一训练方法的 base 比较。尺度来自对应的已保存训练目标，
按当前语义层级平均，使用日历日区块 bootstrap 计算 95% 区间。'''),
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
        cell('markdown','''## 分层效果与原指标复核

Delta 为协调后减 base，负值表示误差下降。
MASE/RMSSE 先逐节点计算，再在语义层级内等权平均；复核表与当前逐节点指标聚合结果比较。'''),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    folder = OUTPUT_DIR / dataset
    print(dataset)
    display(pd.read_csv(folder / 'level_effects_bootstrap.csv'))
    checks = pd.read_csv(folder / 'reproduction_check.csv')
    display(checks.groupby('method')[['abs_diff_MASE','abs_diff_RMSSE']].max())
    assert checks[['abs_diff_MASE','abs_diff_RMSSE']].to_numpy().max() <= 1e-6'''),
        cell('markdown','## 底层节点改善比例与大误差观测'),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    folder = OUTPUT_DIR / dataset
    print(dataset)
    display(pd.read_csv(folder / 'appendix_node_improvement.csv'))
    display(pd.read_csv(folder / 'largest_error_decile.csv'))
    display(Image(filename=str(folder / 'error_redistribution_heatmap.png')))'''),
        cell('markdown','''## 结果解释

误差十分位组由 base 误差定义，分组模式只作描述性解释。
bootstrap 区间描述当前保存实验内的变化，不等同于跨训练种子的稳定性结论。'''),
        cell('code', '''for dataset in ['ausgrid2013','gef2017']:
    records = json.loads((OUTPUT_DIR / dataset / 'validation.json').read_text())
    display(pd.DataFrame(records))
assert len(manifest) == 12
print('All outputs:', OUTPUT_DIR)
print('Source fingerprints:', OUTPUT_DIR / 'source_manifest.json')''')])


if __name__=='__main__':main()
