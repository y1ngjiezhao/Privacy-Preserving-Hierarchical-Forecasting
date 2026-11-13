# PPHF 仅代码与数据集迁移包

本包不包含任何已运行的实验结果。解压后保留 FHF_NDSH/ 内部目录结构。

包含：
- PPHF_modules/：FHF、HFF/HFFPS、DP-FHF、结果分析模块、验证代码及隐私会计依赖源码。
- Datasets/：原始与处理后输入数据、层级结构矩阵、去重映射、标签及数据来源说明。
- 42 个 notebook 入口；代码、参数及 Markdown 内容保留，输出、执行计数与执行元数据已清空。
- windows_gpu/：Windows 10/11、RTX 4070 SUPER 安装、环境检查和计时工具。

不包含：数据集目录之外的 CSV，预测、指标、训练日志、残差、模型、NPZ 训练产物、
协调结果矩阵、结果图表、隐私实验报告、旧验证报告、开发备份、缓存或 Python 环境。
ZIP_MANIFEST.json 仅是文件清单与内容校验值，不是实验结果。

## Windows 安装

安装 NVIDIA 驱动与 64 位 Python 3.12，运行 windows_gpu/setup.bat，
再运行 windows_gpu/start_jupyter.bat。选择 Python (FHF RTX 4070 SUPER) 内核。
详细操作见 windows_gpu/README_zh.md。已有 Mac 环境不能直接搬到 Windows。

## 首次运行顺序

1. 数据集已附带处理后版本，可直接训练；需要重建数据时先运行 Datasets/ 的准备 notebook。
2. 普通全参与实验：先运行 fhf_dedup_ausgrid2013/ 的 FHF_fh1、HFF_fh1、HFFPS_fh1，
   以及 fhf_dedup_gef2017/ 三个框架的 fh1、fh3、fh6 notebook，共 12 个组合。
   它们会生成预测、训练残差、协调矩阵、指标和日志，供后续分析读取。
3. Results_check/ 的四个 notebook 依赖第 2 步生成的当前完整实验结果，不能在训练前直接 Run All。
   输出由代码新建到 Results_check/deduplication_results/。
4. DP-FedAvg：先运行 DP_fhf/notebooks/01_DPFedAvg_Only_* 或对应 *_Sweep notebook；
   完成后再运行 MASE_RMSSE_Summary、100Round_Reconciliation 或 Sweep_Reconciliation。
5. DP covariance、multiscale 和 DP-O2MR 读取相应 FHF 的保存预测、残差与 P 矩阵，
   应在第 2 步完成后执行。各入口前置要求见 DP_fhf/notebooks/README.md。
6. *_Sampling_* 和 *_Sampling_fh1 是独立的采样实验，默认每轮 10% 参与，输出在独立目录。
   它们不会代替四个 Results_check 入口需要的默认全参与实验。

训练输出目录初始为空，运行时自动生成。OVERWRITE=False 不会跳过本包中不存在的训练结果。
01_DPFedAvg_NDSH_Matrix.ipynb 仅为索引，不启动已停用的混合框架矩阵实验。

若需要论文格式 CSV 与 R MCB 图，在第 2 步完成后先运行：
python -m PPHF_modules.results_check.deduplication_results.generate_deduplication_results
再用已安装相应依赖的 R 执行 PPHF_modules/results_check/deduplication_results/render_mcb_figures.R。

## AusGrid 三个独立 sweep

分别运行 DP_fhf/notebooks/ 中的：
- 01_DPFedAvg_AusGrid2013_Sweep_q0p1.ipynb
- 01_DPFedAvg_AusGrid2013_Sweep_q0p5.ipynb
- 01_DPFedAvg_AusGrid2013_Sweep_q1p0.ipynb

每个 notebook 固定一个采样率，扫描 4 个 noise multiplier × 3 个 clipping norm，
每组 20 轮。三者共 36 组，在单张 GPU 上逐个运行 notebook。
统一输出目录为 DP_fhf/outputs/ausgrid2013_sampling_sweep/，按采样率分开保存。
每个入口完成后刷新 all_sampling_rates_summary.csv；全局 DP-FedAvg 汇总也能读取它们。
此处汇总是 base forecast 指标。原 Sweep_Reconciliation 默认仍读取旧目录。
本包不含任何已完成实验结果或 notebook 运行输出。

本包的清单、CRC、notebook 格式/语法和源码一致性均已检查。
完整模型训练与 Windows/CUDA 的运行需要在目标电脑按以上顺序执行。
