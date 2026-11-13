本次结果重新计算于 2026-11-13，所有输出位于本项目，不依赖 OneDrive。
三张图已取消大标题和副标题，仅保留面板标题和图例。
图中已移除 lower is better 的方向性描述。
三张图图例中的 legacy 标记已移除。

已执行
  常规 FHF、HFF、HFFPS：AusGrid fh1 共 3 组，GEF fh1/fh3/fh6 共 9 组。
  四份 Results_check notebook 已执行并写入输出，无错误单元格。
  12 组常规实验完成 2000 次区块 bootstrap、误差重分布和指标复现核查。
  AusGrid 39 组 DP 已保存结果及 1 个无 DP 基线已重新计算准确度。
  39 组包括 36 组 20 轮扫描和 3 组 100 轮 DP 实验。

分层
  AusGrid2013：评价层级 L1/L2/L3 = 1/60/299。
  GEF2017：按最新要求使用结构深度 L1/L2/L3 = 1/6/3。
  GEF 的 5 个浅层终端节点保留在 L2。
  两套数据的相干性仍按实际结构父子关系计算。

三张图位于 figures/，均有 300 dpi PNG 和矢量 PDF。
  01：Average、L1、L2、L3 的 MASE/RMSSE，2×2 布局。
  02：相同四个面板的记录隐私预算与误差。
  03：三个代表序列在测试前 72 小时的预测曲线，保持此前三个序列。
  颜色区分设置，圆点为 MASE，三角为 RMSSE，取消底部说明文字。
  三张图图例统一为 FHF (q=1)、PPHF (q=1)、PPHF (q=0.5)、PPHF (q=0.1)。
  第二张图使用加深的灰、蓝、绿、红色，同步应用于数据点、标注和图例。
  第二张图横坐标刻度显示为整数 0、25、50、75。
  第二张图纵坐标刻度保留一位小数。
  第二张图 Setting 和 Metric 图例并排放在同一行。
  第二张图左上角面板标题简化为 Average。
  第二张图每组 q 与 epsilon 文字组内左对齐，整组居中放在对应数据点上方。
  展示结果按四舍五入保留三位小数，计算和原始 CSV 保留完整精度。
  sampling_100rounds_by_scope_3dp.csv 是三位小数展示表。
  第二张图已放大文字：坐标刻度和图例 14 pt，轴标题和面板标题 16 pt，图内标注 4.2 mm；增加上方留白以容纳标注。

数据范围
  三张图使用匹配的 100 轮结果，sigma=1.2、C=1.0、delta=1e-5。
  20 轮扫描结果另存 sampling_20round_sweep_by_scope.csv，没有混入这三张图。
  当前目录缺失的历史 q=1 100 轮 DP 基线从本机 FHF_backup 恢复。
  恢复文件的来源及校验值见 Results_check/runs/20261113/restored_baseline_manifest.json。
  所有 DP 结果的训练目标、测试键和真实值已与当前无 DP 基线核对一致。
  旧版 q=1 固定 DP seed=42，记录的 epsilon 为假设随机机制下的计算值。
  部分参与使用 fresh randomness 和 2 倍敏感度标定，旧版 q=1 使用 1 倍。
  隐私预算来自原 privacy.json 或原有 recovery 记录，未重写隐私完成标记。
  预算覆盖 global model，不覆盖本地预测或诊断的发布。

重新运行
  Python 环境：/Users/yingjiezhao/miniforge3/envs/neural-network/bin/python
  常规检查：Results_check/runs/20261113/rebuild_current_results.py --datasets ausgrid2013 gef2017 --execute-notebooks
  DP 检查：PPHF_modules/results_check/sampling_results_check.py
  ggplot2：Rscript PPHF_modules/results_check/plot_sampling_comparison.R
  本目录也保留 plot_sampling_comparison.R 副本，可直接用 Rscript 运行。

当前归档
  本次新结果目录为 ausgrid2013、gef2017、error_analysis、dp_sampling。
  paper_current 中的旧论文审计记录属于历史文件，未作为本次结果输入。
