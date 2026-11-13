"""Verify the copied project, its Python dependencies, and real CUDA execution."""
from __future__ import annotations

import argparse
import csv
import importlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Check imports and data on a computer without CUDA.")
    args = parser.parse_args()
    privacy_dependencies = ROOT / "PPHF_modules/dp_fhf/_privacy_dependencies"
    if privacy_dependencies.is_dir():
        sys.path.insert(0, str(privacy_dependencies))
    print("Project:", ROOT)
    print("Python:", sys.executable)
    print("Platform:", platform.platform())
    failures = []
    if sys.version_info < (3, 12):
        failures.append("This installation profile requires Python 3.12 or newer.")

    modules = {
        "torch": "torch", "numpy": "numpy", "pandas": "pandas",
        "scipy": "scipy", "sklearn": "scikit-learn",
        "matplotlib": "matplotlib", "seaborn": "seaborn",
        "jupyterlab": "jupyterlab", "ipykernel": "ipykernel",
        "nbformat": "nbformat", "nbclient": "nbclient",
        "threadpoolctl": "threadpoolctl", "attrs": "attrs", "mpmath": "mpmath",
        "hierarchicalforecast": "hierarchicalforecast", "tqdm": "tqdm",
        "PIL": "Pillow", "reportlab": "reportlab", "pypdf": "pypdf",
    }
    for module, distribution in modules.items():
        try:
            importlib.import_module(module)
            print(f"OK {distribution}: {version(distribution)}")
        except Exception as exc:
            failures.append(f"{distribution}: {exc}")

    for relative in (
        "Datasets/GEFCOM2017/GEF2017_univarivate.csv",
        "Datasets/GEFCOM2017/GEF2017_170101_dedup.csv",
        "Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv",
    ):
        path = ROOT / relative
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = set(next(csv.reader(handle)))
            missing = {"partition_id", "unique_id", "ds", "y"} - columns
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")
            print("OK data:", relative)
        except Exception as exc:
            failures.append(f"{relative}: {exc}")

    sys.path.insert(0, str(ROOT))
    try:
        from PPHF_modules.fhf.FHF_runner import Args
        from PPHF_modules.fhf.FHF_models import build_model
        from PPHF_modules.fhf.FHF_HF2 import HFLArgs
        from PPHF_modules.dp_fhf.notebook_runtime import run_fixed_dpfedavg
        from PPHF_modules.dp_fhf.privacy_accounting import sampled_fedavg_epsilon

        report = sampled_fedavg_epsilon(1.2, 3, 1e-5, 100, 10)
        if not math.isfinite(report["epsilon"]):
            raise ValueError("Privacy accountant returned a non-finite epsilon.")
        print("OK FHF, HFF/HFFPS, notebook runtime, and sampled DP accounting")
    except Exception as exc:
        failures.append(f"Project imports/accountant: {exc}")

    try:
        import torch

        print("PyTorch CUDA runtime:", torch.version.cuda)
        if not torch.cuda.is_available():
            if args.allow_cpu:
                print("CPU-only check requested: GPU execution was not tested.")
            else:
                raise RuntimeError(
                    "CUDA is unavailable. Check the NVIDIA driver and install the CUDA "
                    "PyTorch wheel in the same Python environment as the notebook."
                )
        else:
            properties = torch.cuda.get_device_properties(0)
            print("GPU:", properties.name)
            print("GPU memory (GiB):", round(properties.total_memory / 1024**3, 2))
            torch.set_num_threads(1)
            model_args = Args(device="cuda", fh=0)
            model = build_model(model_args).to("cuda")
            x = torch.randn(8, model_args.lags, 1, device="cuda")
            y = torch.randn(8, 1, device="cuda")
            prediction = model(x)
            if prediction.shape != y.shape:
                raise ValueError(f"Unexpected output shape: {prediction.shape}")
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            torch.cuda.synchronize()
            print("OK actual project LSTM forward and backward on CUDA")
    except Exception as exc:
        failures.append(f"GPU verification: {exc}")

    for failure in failures:
        print("FAIL:", failure)
    if failures:
        return 1
    print("PASS: environment checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
