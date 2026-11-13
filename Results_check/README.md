# 当前去重结果分析入口

此目录只包含 notebook 入口。实现位于 ../PPHF_modules/results_check/。
本包不含任何结果 CSV、图或保存的 notebook 输出。
先运行两套数据的普通 FHF/HFF/HFFPS 完整实验，再执行四个分析 notebook。
它们读取 fhf_dedup_* 下新生成的结果，并新建 deduplication_results/ 输出。
