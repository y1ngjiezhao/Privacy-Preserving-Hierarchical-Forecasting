# Windows 10/11 与 RTX 4070 SUPER 运行说明

将整个 `FHF_NDSH` 文件夹复制到 Windows，例如 `D:\Research\FHF_NDSH`。
保留项目内部目录结构。Mac 的 Conda 环境不能直接复制使用，需要在 Windows
重新安装 Python 与 CUDA 版 PyTorch。

本次新增的安装和计时工具放在 `windows_gpu/`。现有训练代码、notebook 参数
和已保存结果没有因本次迁移准备而修改。

## 1. 本包内容与运行依赖

本包仅含数据集、42 个已清空输出的 notebook、模块和运行/安装代码。
没有旧预测、指标、日志、残差、模型、协调矩阵或结果图。
保留解压目录结构。先训练生成结果，再运行依赖保存结果的分析 notebook。
完整运行顺序见根目录 README.md。数据集中的 CSV、层级矩阵和标签属于输入。

当前采样示例使用：

| 示例 | 数据文件 |
| --- | --- |
| 普通 FHF/HFF/HFFPS 的 GEF2017 采样示例 | `Datasets/GEFCOM2017/GEF2017_univarivate.csv` |
| DP-FHF 的 GEF2017 采样示例 | 当前 notebook 显式指向同一个 `GEF2017_univarivate.csv` |
| AusGrid2013 去重采样示例 | `Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv` |

部分其他 DP notebook 默认选择较短的 `GEF2017_170101_dedup.csv`。
比较耗时或精度前应核对实际打印的 CSV 路径。

## 2. 安装环境

先安装适用于这台电脑的 NVIDIA 显卡驱动，以及 64 位 Python 3.12。
Python 安装时保留 Windows Python Launcher。可在终端分别运行
`nvidia-smi` 和 `py -3.12 --version` 检查。

- [NVIDIA 驱动下载](https://www.nvidia.com/Download/index.aspx)
- [Python Windows 下载](https://www.python.org/downloads/windows/)
- [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)

双击 `windows_gpu\setup.bat`。它会在复制后的项目中创建 `.venv`，安装
`torch==2.13.0` 的 CUDA 12.6 版本和本项目依赖，注册独立的 Jupyter kernel，
最后执行环境检查。安装需要联网。PyTorch 版本与当前 Mac 环境一致，CUDA
安装命令来自官方 Linux/Windows 版本列表。

本项目使用预编译 PyTorch，不需要为这些 Python 训练入口单独编译 CUDA 扩展。
GPU 检查成功才开始完整训练。

如果已有 Anaconda/Miniconda，也可以在 Anaconda Prompt 中手动创建环境。
下面的命令替代 `setup.bat`，两种安装方式选一种即可：

```bat
cd /d D:\Research\FHF_NDSH
conda create -n fhf4070 python=3.12 -y
conda activate fhf4070
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r windows_gpu\requirements.txt
python -m ipykernel install --user --name fhf-4070-super --display-name "Python (FHF RTX 4070 SUPER)"
python windows_gpu\check_environment.py
python -m jupyterlab
```

项目已包含 `PPHF_modules/dp_fhf/_privacy_dependencies/` 下的纯 Python 隐私会计依赖，
须一并复制。若缺少，可在项目根目录执行原有恢复命令：

```bat
python -m pip install --no-deps --target PPHF_modules\dp_fhf\_privacy_dependencies -r PPHF_modules\dp_fhf\requirements-privacy.txt
```

## 3. 确认 notebook 使用 GPU

使用 `.venv` 安装后，双击 `windows_gpu\start_jupyter.bat`。
Conda 用户在激活 `fhf4070` 后执行 `python -m jupyterlab`。

打开 notebook，选择 **Python (FHF RTX 4070 SUPER)** kernel。现有 notebook
会使用 `"cuda" if torch.cuda.is_available() else "cpu"` 选择设备。
GPU 安装正确时无需修改训练代码。

环境检查会实际运行本项目 LSTM 的 CUDA 前向与反向计算，并检查数据文件和
隐私会计。也可以在 notebook 中确认：

```python
import sys
import torch
print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

必须看到 `True` 和目标显卡名称。若出现 `False`，先检查 kernel 对应的
Python 环境、CUDA 版 PyTorch 与 NVIDIA 驱动。

可先运行以下采样示例，默认 100 轮、每轮 10% 参与：

- `fhf_dedup_gef2017/FHF_Sampling_fh1.ipynb`
- `fhf_dedup_gef2017/HFF_Sampling_fh1.ipynb`
- `fhf_dedup_gef2017/HFFPS_Sampling_fh1.ipynb`
- `fhf_dedup_ausgrid2013/` 中的同名示例
- `DP_fhf/notebooks/01_DPFedAvg_Sampling_GEF2017.ipynb`
- `DP_fhf/notebooks/01_DPFedAvg_Sampling_AusGrid2013.ipynb`

新结果写入 notebook 原本配置的独立采样目录。注意：DP 示例默认
`OVERWRITE=False`，相同配置的已完成结果存在时会跳过训练。
需要重新运行时可在复制后的项目中设置 `OVERWRITE=True`，或更换
`output_base_dir`。仅重新显示旧结果不能用于测量 4070 SUPER 的训练速度。

## 4. 在 4070 SUPER 上估算实际耗时

先运行 3 轮短测。工具调用项目真实的共享训练模块，显式使用 CUDA，按每轮
训练均值和一次性的数据准备、预测及协调开销估算 100 轮时间。
它不导出模型或预测文件，仅新增计时 JSON，不会因旧结果存在而跳过训练。

使用 `.venv` 时，在项目根目录终端执行：

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset GEF2017 --dp-fedavg
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset AusGrid2013 --dp-fedavg
```

Conda 用户将命令开头的 `.venv\Scripts\python.exe` 换为 `python`。
普通 FHF 去掉 `--dp-fedavg`。例如，测试普通 HFFPS：

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset AusGrid2013 --framework hffps
```

全参与需要另测相应配置，不能直接使用 10% 参与的耗时：

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset GEF2017 --sampling-rate 1.0 --dp-fedavg
```

报告保存在 `output/gpu_benchmarks/`，含 GPU 型号、PyTorch/CUDA 版本、参与人数、
训练耗时和估算总耗时。该估算不含 notebook 后续文件导出和绘图。
默认预测步数为 1、LSTM 隐藏维度 64、2 层、batch size 256、每轮本地训练 1 次。

此前 Mac CPU 的耗时估计不能直接作为 4070 SUPER 的耗时。
客户端训练目前按顺序调用，数据准备和 NumPy 协调计算仍有 CPU 开销，GPU 利用率
不一定持续很高。先逐个运行实验，避免多个 notebook 同时争用同一张显卡。

## 本机验证范围

迁移工具可在 Mac 上检查导入和数据，也可用小数据验证真实训练入口。
Windows 的批处理安装、NVIDIA 驱动、CUDA 轮子及显卡执行仍须在目标电脑上
通过 `check_environment.py` 验证。跨硬件和后端运行的浮点结果可能略有差异。
