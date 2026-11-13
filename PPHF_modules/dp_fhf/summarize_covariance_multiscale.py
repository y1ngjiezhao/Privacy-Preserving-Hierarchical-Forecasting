"""Validate and summarise multiscale results without editing manuscript files."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import json

import numpy as np
import pandas as pd

from PPHF_modules.dp_fhf.run_covariance_multiscale import RETENTIONS

ROOT = PROJECT_ROOT
OUT = ROOT / 'DP_fhf/outputs/02_dp_covariance/multiscale_no_debias'
PERCENTAGES = [100 * q for q in RETENTIONS]


def main():
    data = pd.read_csv(OUT / 'metrics_summary.csv')
    bounds = pd.read_csv(OUT / 'client_bounds.csv')
    diagnostics = pd.read_csv(OUT / 'matrix_diagnostics.csv')
    nodes = pd.read_csv(OUT / 'per_node_metrics.csv')
    assert (len(data), len(bounds), len(diagnostics), len(nodes)) == (100, 3700, 40, 18500)
    for frame in [data, bounds, diagnostics, nodes]:
        assert set(frame.retention_target_pct) == set(PERCENTAGES)
    assert np.isfinite(data[['MASE', 'RMSSE']]).all().all()
    assert diagnostics.PS_minus_S_max_abs.max() < 1e-6
    keys = ['dataset', 'input', 'retention_target_pct', 'method']
    node_means = nodes.groupby(keys)[['MASE', 'RMSSE']].mean().sort_index()
    assert np.allclose(node_means, data.set_index(keys)[['MASE', 'RMSSE']].sort_index(), rtol=1e-12)
    for dataset, folder in [('GEF2017', 'fhf_dedup_gef2017'), ('AusGrid2013', 'fhf_dedup_ausgrid2013')]:
        for branch in ['non_dp_forecasts', 'dp_fedavg_100round']:
            artifact = (ROOT / folder / 'fh1_FHF_training_artifacts.npz' if branch == 'non_dp_forecasts'
                        else ROOT / 'DP_fhf/outputs/01_dp_fedavg_100round' / dataset / 'fh1_FHF_dp_fedavg_training_artifacts.npz')
            with np.load(artifact) as saved:
                residuals = np.abs(saved['residuals'][:, :, 0])
                ids = saved['client_ids'].astype(int)
            for q in PERCENTAGES:
                local = bounds.query('dataset == @dataset and input == @branch and retention_target_pct == @q').set_index('node_id').loc[ids]
                B = np.quantile(residuals, q / 100, axis=1, method='linear')
                assert np.allclose(B, local.B, rtol=1e-12, atol=1e-15)
                assert np.allclose(local.retained_fraction, np.mean(residuals <= B[:, None], axis=1))
                assert np.allclose(local.laplace_scale, 2 * B)
                assert np.allclose(local.noise_variance, 8 * B**2)
                label = f'retain_{q:g}pct'.replace('.', 'p')
                privacy = json.loads((OUT / dataset / branch / label / 'privacy_and_calibration.json').read_text())
                assert privacy['debias_noise_variance'] is False
                assert privacy['retention_target'] == q / 100

    main_data = data.query("input == 'non_dp_forecasts'")
    wide = main_data.pivot(index=['dataset', 'retention_target_pct'], columns='method', values=['MASE', 'RMSSE'])
    report = '# 多尺度残差剪裁：无方差校正\n\n'
    report += '保留比例：99.5%、75%、50%、25%、0.5%。每个客户端按原始尺度绝对训练残差分位数设置边界；保留比例表示未被截断的比例，所有样本仍参与估计。\n\n'
    report += 'epsilon=1，Laplace，seed=42，ridge=1e-6，debias_noise_variance=False。分位数校准非私有，这些结果是效用实验。\n\n'
    report += '以下使用非私有基础预测。DP-FedAvg 分支见 metrics_summary.csv。\n\n'
    for dataset in ['GEF2017', 'AusGrid2013']:
        report += f'## {dataset}\n\n| 保留比例 / 参照 | MinTs MASE | MinTs RMSSE | WLSv MASE | WLSv RMSSE |\n|---|---:|---:|---:|---:|\n'
        ols = main_data.query('dataset == @dataset and method == "ols"').iloc[0]
        report += f'| OLS | {ols.MASE:.6f} | {ols.RMSSE:.6f} | {ols.MASE:.6f} | {ols.RMSSE:.6f} |\n'
        for q in PERCENTAGES:
            row = wide.loc[dataset, q]
            report += f"| {q:g}% | {row['MASE','mint_shrinkage']:.6f} | {row['RMSSE','mint_shrinkage']:.6f} | {row['MASE','wlsv']:.6f} | {row['RMSSE','wlsv']:.6f} |\n"
        report += '\n'
    report += '## 最低保留比例的矩阵诊断\n\n| 数据集 | 基础预测分支 | 方法 | 低于特征值下限的数量 | 距 OLS | 距噪声加权 WLS |\n|---|---|---|---:|---:|---:|\n'
    for row in diagnostics.query('retention_target_pct == 0.5').itertuples():
        report += f'| {row.dataset} | {row.input} | {row.method} | {row.eigenvalues_below_floor} | {row.d_P_to_OLS:.6f} | {row.d_P_to_noise_WLS:.6f} |\n'
    report += '\n改变保留比例同时改变剪裁后的信号和客户端之间的相对噪声方差；这些实验不构成单独增大噪声的收敛检验。\n\n'
    report += '## 复现\n\n运行 run_covariance_multiscale.py，再运行 summarize_covariance_multiscale.py。汇总脚本仅更新实验报告和验证结果，不生成或修改论文文本。Notebook：DP_fhf/notebooks/02_DP_Covariance_Multiscale_No_Debias.ipynb。\n'
    (OUT / 'REPORT_zh.md').write_text(report)
    verification = {'configurations': 20, 'retentions': RETENTIONS, 'metrics_rows': len(data),
                    'client_bound_rows': len(bounds), 'diagnostic_rows': len(diagnostics), 'per_node_rows': len(nodes),
                    'bounds_match_training_quantiles': True, 'metrics_match_per_node_means': True,
                    'noise_variance_correction': False, 'max_PS_minus_S': float(diagnostics.PS_minus_S_max_abs.max()),
                    'max_relative_incoherence': float(diagnostics.coherence_relative.max()), 'manuscript_modified_by_summary': False}
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
