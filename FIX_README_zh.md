# DP 会计依赖修复与结果恢复补丁

报错原因：20 轮训练结束后，Google dp-accounting 导入 attr 失败。
attr 来自 attrs 包。本补丁附带 attrs 23.2.0 与项目原用的 mpmath 1.3.0，
并将依赖检查移到 notebook helper 和 CLI 启动阶段。

## 应用到 Windows

本 ZIP 直接以 PPHF_modules、windows_gpu、DP_fhf 为顶层，没有额外 FHF_NDSH 包裹层。
解压到原项目根目录，例如 E:\yjz\FHF_NDSH，并覆盖同名代码文件。
不要删除已有数据集、预测 CSV 或训练 NPZ。本补丁不包含数据集或实验结果。

## 恢复结果，避免重新训练

1. 应用补丁后重启 notebook 内核。
2. 在原来报错的 sampling-rate notebook 中，只运行导入和参数两个代码单元格。
   保持原 SAMPLING_RATE、SAMPLING_SEED 和 NUM_ROUNDS；贴出日志的轮数为 20。
3. 不运行训练单元格，不使用 Run All。新增单元格执行：

```python
from PPHF_modules.dp_fhf.recover_dp_sweep import recover_ausgrid_sampling_sweep

recovered = recover_ausgrid_sampling_sweep(
    output_root=OUTPUT_ROOT,
    sampling_rate=SAMPLING_RATE,
    sampling_seed=SAMPLING_SEED,
    num_rounds=NUM_ROUNDS,
)
recovered
```

工具按 parameter_grid.csv 检查实际存在的预测 CSV/训练 NPZ 配对，恢复逐序列指标、
MAE/RMSE/MASE/RMSSE 及对应配置的 epsilon。对缺失配对会列出清单，不编造结果。
恢复输出位于该采样率目录的 recovery/，共同父目录新增
recovered_all_sampling_rates_summary.csv。

恢复过程不训练，不修改原 CSV/NPZ，不创建正常的 privacy.json 完成标记。
如果模型权重未在报错前保存，不能从预测与残差重建。epsilon 是根据提供的实际
完成轮数及保存的客户端规模，对原全局模型机制补算的会计值，不是本地预测的 DP 保证。
恢复表可用于研究指标分析；直接再次运行原训练单元格仍会把缺失完成标记的组当作未完成。

## 验证

缺少 attr 的模拟故障会在训练前停止；完整失败 sweep 会明确抛出错误。
已用之前实际运行的 7 节点、2 轮输出模拟故障，并验证三个采样率恢复后的
指标和 epsilon 与原成功结果一致，原输入文件哈希不变。未访问或修改 Windows 上的结果。
