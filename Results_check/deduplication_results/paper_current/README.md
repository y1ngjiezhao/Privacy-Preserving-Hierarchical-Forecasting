# 当前论文结果索引 / Current paper results

更新：2026-09-28。此目录是经自动核验的论文结果导出与来源索引；原始实验数据、历史结果和旧论文备份继续保留。

## 当前来源

| 论文部分 | 权威源结果 | 当前配置 |
|---|---|---|
| 主结果两表、MASI | `Results_check/deduplication_results/{gef2017,ausgrid2013}/` | 去重层级；按评估层级汇总；训练 naive 尺度 MASI |
| 图 5–8 | 同上目录的 MCB PDF | GEF 按 h=1/3/6；Ausgrid h=1；rank 1/3 位小数；14 pt 图例 |
| DP-FedAvg 100 轮 | `DP_fhf/outputs/dp_100round_reconciliation/` | 2026-09-25 批次；σ=0.3/1.2，C=1，δ=1e-5 |
| DP-FedAvg 20 轮扫描 | `DP_fhf/outputs/dp_sweep_reconciliation/` | 2026-09-24 批次；24 组，每组 base/mints/wlsv/ols |
| 固定残差边界 | `DP_fhf/outputs/02_dp_covariance/<dataset>/no_debias/` | ±0.0005，ε=1，seed=42，rho=1e-6；不做噪声方差校正 |
| 固定边界与 Non-DP/OLS 对比 | `DP_fhf/outputs/02_dp_covariance/no_debias_comparison/paper_comparison.csv` | 仅保留 non_private/reference/no_debias |
| 分位数残差边界 | `DP_fhf/outputs/02_dp_covariance/multiscale_no_debias/` | 99.5%、75%、50%、25%、0.5%；使用 non_dp_forecasts 分支 |
| 私有预测发布 | `DP_fhf/outputs/03_dp_o2mr/<dataset>/` | non_dp_forecasts；ε=1/5/10/20；MinTs、WLSv；固定 P |

路径均相对于 FHF_NDSH 根目录。这里的 CSV 是未经显示舍入的源结果导出，不是将论文数字反填到实验文件。

## 清理后的使用规则

2026-09-28 已按要求删除旧 corrected 结果及 baseline_recheck、旧 0.05% 快照、pooled legacy MCB 图、旧论文 backup/staged 快照。删除清单见 cleanup_20260928.json。

- 固定边界只使用 `<dataset>/no_debias/`。
- `no_debias_comparison/paper_comparison.csv` 只含 non_private、reference（OLS）、no_debias 三类现用比较。
- 分位数使用 99.5%、75%、50%、25%、0.5%。`multiscale_retention_0p5_update/` 只保留现用 0.5% 更新的结果及来源记录。
- 源 CSV 的 BO-FHF 是内部历史键，对应论文 LN-FHF；不代表数值过期。
- DP-FedAvg 100 轮和 20 轮扫描、私有预测发布、原始训练输入保持不变。

## 核验与复现

```bash
python PPHF_modules/results_check/audit_paper_results.py --paper-root /path/to/manuscript
```

需要 numpy、pandas、pypdf。全部核验通过后才更新本目录的未舍入 CSV 导出。

固定边界及参考方法的完整复现：

```bash
python PPHF_modules/dp_fhf/run_covariance_no_debias_comparison.py
python PPHF_modules/dp_fhf/summarize_covariance_no_debias.py
```

两个脚本现仅产生现用 no_debias、Non-DP 与 OLS 结果，不依赖或重建旧校正方案和旧论文快照。默认协方差 notebook 同样使用 no-debias；底层 API/CLI 仍支持显式配置的其他实验，不属于当前论文结果。

先前审核报告是历史审计记录；其中“保留旧文件”的说明已被本次清理取代。当前数据以本索引和最新 verification.json 为准。
