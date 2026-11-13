# Running on Windows 10/11 with an RTX 4070 SUPER

Copy the entire `FHF_NDSH` directory to Windows, for example to
`D:\Research\FHF_NDSH`, preserving its internal structure. A Mac Conda environment
cannot be copied directly: reinstall Python and CUDA-enabled PyTorch on Windows.

Installation and timing tools are in `windows_gpu/`. Preparing the original
migration package did not change training code, notebook parameters, or saved results.

## 1. Package contents and execution dependencies

The earlier migration package contained datasets, 42 notebooks with cleared
outputs, modules, and execution/installation code, with no old forecasts, metrics,
logs, residuals, models, reconciliation matrices, or result figures. The current
GitHub snapshot additionally includes result PDFs and 43 notebooks; see the root
README for its contents and restore compressed CSVs before running experiments.
Preserve the directory structure, train to generate results, then run analyses
that require saved results. Dataset CSVs, hierarchy matrices, and tags are inputs.

The current sampling examples use:

| Example | Input file |
|---|---|
| Ordinary FHF/HFF/HFFPS GEF2017 sampling | `Datasets/GEFCOM2017/GEF2017_univarivate.csv` |
| DP-FHF GEF2017 sampling | The notebook explicitly selects the same `GEF2017_univarivate.csv` |
| Deduplicated AusGrid2013 sampling | `Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv` |

Some other DP notebooks default to the shorter `GEF2017_170101_dedup.csv`.
Check the printed CSV path before comparing runtime or accuracy.

## 2. Install the environment

Install the NVIDIA driver appropriate for the target computer and 64-bit Python
3.12. Keep Windows Python Launcher enabled during Python installation. Check
`nvidia-smi` and `py -3.12 --version` in a terminal.

- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
- [Python for Windows](https://www.python.org/downloads/windows/)
- [Official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)

Double-click `windows_gpu\setup.bat`. It creates `.venv` inside the copied project,
installs the CUDA 12.6 build of `torch==2.13.0` and project dependencies, registers
a separate Jupyter kernel, and runs the environment check. Installation requires
network access. The PyTorch version matches the Mac environment used when the
package was prepared; the CUDA command came from the official Linux/Windows list.

The project uses prebuilt PyTorch and does not require compiling separate CUDA
extensions for these Python training entry points. Start full training only after
the GPU check succeeds.

If Anaconda/Miniconda is already installed, create an environment manually in
Anaconda Prompt instead of using `setup.bat`. Choose either installation method:

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

Copy the bundled pure-Python privacy-accounting dependencies in
`PPHF_modules/dp_fhf/_privacy_dependencies/` as well. If missing, restore them from
the project root with the existing command:

```bat
python -m pip install --no-deps --target PPHF_modules\dp_fhf\_privacy_dependencies -r PPHF_modules\dp_fhf\requirements-privacy.txt
```

## 3. Confirm the notebook uses the GPU

After installing into `.venv`, double-click `windows_gpu\start_jupyter.bat`.
Conda users should activate `fhf4070` and run `python -m jupyterlab`.

Open a notebook and select **Python (FHF RTX 4070 SUPER)**. Existing notebooks
select `"cuda" if torch.cuda.is_available() else "cpu"`. Correct GPU installation
does not require changing training code.

The environment check runs a CUDA forward and backward pass through the project's
LSTM and checks datasets and privacy accounting. You can also inspect:

```python
import sys
import torch
print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Expect `True` and the target GPU's name. If availability is `False`, check the
kernel's Python environment, CUDA-enabled PyTorch, and NVIDIA driver first.

Start with these sampling examples, which default to 100 rounds and 10%
participation per round:

- `fhf_dedup_gef2017/FHF_Sampling_fh1.ipynb`
- `fhf_dedup_gef2017/HFF_Sampling_fh1.ipynb`
- `fhf_dedup_gef2017/HFFPS_Sampling_fh1.ipynb`
- The corresponding examples in `fhf_dedup_ausgrid2013/`
- `DP_fhf/notebooks/z01_DPFedAvg_Sampling_GEF2017.ipynb`
- `DP_fhf/notebooks/01_0.1_DPFedAvg_Sampling_AusGrid2013.ipynb`
- `DP_fhf/notebooks/01_0.5_DPFedAvg_Sampling_AusGrid2013.ipynb`

New outputs use each notebook's configured sampling directory. DP examples default
to `OVERWRITE=False` and skip completed runs with the same configuration. For a new
run, set `OVERWRITE=True` in the copied project or change `output_base_dir`.
Redisplaying old results does not measure RTX 4070 SUPER training speed.

## 4. Estimate actual RTX 4070 SUPER runtime

Start with a 3-round benchmark. The tool calls the project's actual shared
training modules with CUDA explicitly selected. It estimates 100-round runtime
from the mean round time and one-off data preparation, forecasting, and
reconciliation overhead. It exports neither models nor forecasts, only a timing
JSON file, and does not skip training because old results exist.

With `.venv`, run these commands from the project root:

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset GEF2017 --dp-fedavg
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset AusGrid2013 --dp-fedavg
```

Conda users replace `.venv\Scripts\python.exe` with `python`. Omit `--dp-fedavg`
for ordinary FHF. For example, benchmark ordinary HFFPS with:

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset AusGrid2013 --framework hffps
```

Benchmark full participation separately rather than reusing the 10% timing:

```bat
.venv\Scripts\python.exe windows_gpu\benchmark.py --dataset GEF2017 --sampling-rate 1.0 --dp-fedavg
```

Reports in `output/gpu_benchmarks/` include GPU model, PyTorch/CUDA versions,
participant count, training time, and estimated total runtime. Estimates exclude
subsequent notebook file exports and plotting. Defaults are horizon 1, LSTM hidden
dimension 64, 2 layers, batch size 256, and 1 local training epoch per round.

Earlier Mac CPU estimates cannot be used directly as RTX 4070 SUPER runtimes.
Client training is currently sequential; data preparation and NumPy reconciliation
still incur CPU overhead, so GPU utilization may fluctuate. Run experiments one
at a time to avoid contention between notebooks on the same GPU.

## Local validation scope

The migration tools can check imports and datasets on Mac, and validate real
training entry points on small data. Windows batch installation, NVIDIA drivers,
CUDA wheels, and GPU execution still require `check_environment.py` on the target
computer. Floating-point results may differ slightly across hardware and backends.
