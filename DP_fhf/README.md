# DP-FHF 运行入口

模块位于 ../PPHF_modules/dp_fhf/，notebook 位于 notebooks/。
本包不附带 outputs/ 中的任何旧结果；训练和分析运行时会新建输出。
先运行固定 DP-FedAvg 或 sweep，再运行对应汇总及协调。
Covariance 和 O2MR 需要先生成普通 FHF 的预测、训练残差和协调矩阵。
具体入口、参数及依赖顺序见 notebooks/README.md 和根目录 README.md。
