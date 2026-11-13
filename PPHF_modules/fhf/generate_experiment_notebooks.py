from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from PPHF_modules.notebook_paths import normalize_notebook

"""Generate clean NDSH experiment notebooks from the read-only FHF reference.

The source project is only read.  Notebook outputs and execution counters are
removed so the generated experiment matrix remains small and reproducible.
"""

import json
import re
from pathlib import Path


NDSH_ROOT = PROJECT_ROOT
REFERENCE_ROOT = NDSH_ROOT.parent / "FHF"


EXPERIMENTS = {
    "fhf_gef2017": {
        "dataset": "Datasets/GEFCOM2017/GEF2017_hierarchy_with_duplicates.csv",
        "hierarchy_mode": "duplicates",
        "horizons": [1, 3, 6],
        "sources": {
            "FHF": "fullFHF_gef2017/FHF_fh{h}.ipynb",
            "HFF": "fullHFF_gef2017/HFF_fh{h}.ipynb",
            "HFFPS": "fullHFF_gef2017/HFFPS_fh{h}.ipynb",
        },
    },
    "fhf_dedup_gef2017": {
        "dataset": "Datasets/GEFCOM2017/GEF2017_univarivate.csv",
        "hierarchy_mode": "dedup",
        "horizons": [1, 3, 6],
        "sources": {
            "FHF": "FHF_gef2017/FHF_fh{h}.ipynb",
            "HFF": "fullHFF_gef2017/HFF_fh{h}.ipynb",
            "HFFPS": "fullHFF_gef2017/HFFPS_fh{h}.ipynb",
        },
    },
    "fhf_ausgrid2013": {
        "dataset": "Datasets/AusGrid2013/AusGrid2013_univariate.csv",
        "hierarchy_mode": "duplicates",
        "horizons": [1],
        "sources": {
            "FHF": "HFF_ausgrid2013/FHF_fh{h}.ipynb",
            "HFF": "HFF_ausgrid2013/HFF_fh{h}.ipynb",
            "HFFPS": "HFF_ausgrid2013/HFFPS_fh{h}.ipynb",
        },
    },
    "fhf_dedup_ausgrid2013": {
        "dataset": "Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv",
        "hierarchy_mode": "dedup",
        "horizons": [1],
        "sources": {
            "FHF": "HFF_ausgrid2013/FHF_fh{h}.ipynb",
            "HFF": "HFF_ausgrid2013/HFF_fh{h}.ipynb",
            "HFFPS": "HFF_ausgrid2013/HFFPS_fh{h}.ipynb",
        },
    },
}


def _patch_notebook(
    source_path: Path,
    destination_path: Path,
    dataset_relative: str,
    hierarchy_mode: str,
    framework: str,
) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    dataset_expression = (
        "args.csv_path = str((module_dir.parent / "
        f"{dataset_relative!r}).resolve())"
    )
    reject_expression = f"args.reject_duplicate_series = {hierarchy_mode == 'dedup'}"
    output_expression = (
        "output_prefix = str((module_dir.parent / "
        f"{destination_path.parent.name!r} / "
        f"f'fh{{args.fh + 1}}_{framework}').resolve())"
    )

    parameter_cell_found = False
    output_prefix_found = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if re.search(r"^args\.csv_path\s*=", source, flags=re.MULTILINE):
            source = re.sub(
                r"^args\.csv_path\s*=.*$",
                dataset_expression,
                source,
                count=1,
                flags=re.MULTILINE,
            )
            if re.search(
                r"^args\.reject_duplicate_series\s*=", source, flags=re.MULTILINE
            ):
                source = re.sub(
                    r"^args\.reject_duplicate_series\s*=.*$",
                    reject_expression,
                    source,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                source = source.replace(
                    dataset_expression,
                    dataset_expression + "\n" + reject_expression,
                    1,
                )
            parameter_cell_found = True
            if "args.sampling_rate =" not in source:
                source = source.replace(
                    reject_expression,
                    reject_expression + "\nargs.sampling_rate = 1.0\nargs.sampling_seed = None\n",
                    1,
                )

        if re.search(r"^output_prefix\s*=", source, flags=re.MULTILINE):
            source = re.sub(
                r"^output_prefix\s*=.*$",
                output_expression,
                source,
                count=1,
                flags=re.MULTILINE,
            )
            output_prefix_found = True
            source = source.replace(
                output_expression,
                output_expression + "\nfrom client_sampling import sampled_output_dir\n"
                "_sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed\n"
                "output_prefix = str(sampled_output_dir(Path(output_prefix).parent, "
                "args.sampling_rate, _sampling_seed) / Path(output_prefix).name)\n"
                "Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)",
                1,
            )

        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    if not parameter_cell_found:
        raise ValueError(f"Cannot locate args.csv_path in {source_path}")
    if not output_prefix_found:
        raise ValueError(f"Cannot locate output_prefix in {source_path}")

    note = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# NDSH {framework} experiment\n",
            "\n",
            f"- Dataset: `{dataset_relative}`\n",
            f"- Hierarchy mode: `{hierarchy_mode}`\n",
            "- Parameters and evaluation logic are copied from the corresponding "
            "read-only `/Users/yingjiezhao/yjz/yjzPy/FHF` notebook.\n",
            "- All generated artifacts are written into this experiment folder.\n",
        ],
    }
    notebook["cells"].insert(0, note)
    notebook.setdefault("metadata", {})["ndsh_generation"] = {
        "reference_notebook": str(source_path),
        "dataset": dataset_relative,
        "hierarchy_mode": hierarchy_mode,
        "framework": framework,
    }
    normalize_notebook(notebook)
    destination_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _dp_launcher_notebook() -> dict:
    experiment_rows = [
        {
            "name": folder_name,
            "dataset": config["dataset"],
            "output_dir": f"{folder_name}/dp_fedavg_outputs",
            "hierarchy_mode": config["hierarchy_mode"],
            "horizons": config["horizons"],
        }
        for folder_name, config in EXPERIMENTS.items()
    ]
    setup = f'''from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path.cwd().resolve()
if not (ROOT / "PPHF_modules" / "dp_fhf" / "run_dp_fedavg.py").is_file():
    ROOT = next(
        parent for parent in (ROOT, *ROOT.parents)
        if (parent / "PPHF_modules" / "dp_fhf" / "run_dp_fedavg.py").is_file()
    )

RUNNER = ROOT / "PPHF_modules" / "dp_fhf" / "run_dp_fedavg.py"
EXPERIMENTS = {experiment_rows!r}
FRAMEWORKS = ["fhf", "hff", "hffps"]
PYTHON = sys.executable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLING_RATE = 1.0
SAMPLING_SEED = 42

for experiment in EXPERIMENTS:
    print(
        experiment["name"],
        "|", experiment["hierarchy_mode"],
        "| horizons", experiment["horizons"],
        "|", ROOT / experiment["dataset"],
    )
'''
    commands = '''commands = []
for experiment in EXPERIMENTS:
    dataset = ROOT / experiment["dataset"]
    output_dir = ROOT / experiment["output_dir"]
    for horizon in experiment["horizons"]:
        for framework in FRAMEWORKS:
            commands.append({
                "experiment": experiment["name"],
                "framework": framework,
                "horizon": horizon,
                "output_dir": output_dir,
                "command": [
                    PYTHON, str(RUNNER),
                    "--csv", str(dataset),
                    "--framework", framework,
                    "--output-dir", str(output_dir),
                    "--forecast-horizon", str(horizon),
                    "--hierarchy-mode", experiment["hierarchy_mode"],
                    "--model", "lstm",
                    "--device", DEVICE,
                    "--lags", "48",
                    "--num-rounds", "100",
                    "--local-epochs", "1",
                    "--hidden-size", "64",
                    "--num-layers", "2",
                    "--batch-size", "256",
                    "--torch-num-threads", "1",
                    "--seed", "42",
                    "--sampling-rate", str(SAMPLING_RATE),
                    "--sampling-seed", str(SAMPLING_SEED),
                    "--shared-artifact-dir", str(output_dir),
                ],
            })

print(f"Prepared {len(commands)} runs.")
for job in commands:
    print(" ".join(job["command"]))
'''
    run = '''for job in commands:
    job["output_dir"].mkdir(parents=True, exist_ok=True)
    print(
        f"Running {job['experiment']} / {job['framework'].upper()} "
        f"/ horizon {job['horizon']} ...",
        flush=True,
    )
    subprocess.run(job["command"], cwd=ROOT, check=True)
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NDSH DP-FedAvg experiment matrix\n",
                    "\n",
                    "Central launcher for duplicate and deduplicated GEF2017/AusGrid2013 "
                    "hierarchies. It runs FHF, HFF, and HFFPS; GEF2017 uses horizons "
                    "1, 3, and 6, while AusGrid2013 uses horizon 1. The final cell "
                    "starts all long-running jobs.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": setup.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": commands.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": run.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (neural-network)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate_dp_notebook() -> Path:
    destination = NDSH_ROOT / "DP_fhf" / "notebooks" / "01_DPFedAvg_NDSH_Matrix.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(normalize_notebook(_dp_launcher_notebook()), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return destination


def generate() -> None:
    if not REFERENCE_ROOT.is_dir():
        raise FileNotFoundError(f"Missing read-only reference project: {REFERENCE_ROOT}")

    for folder_name, config in EXPERIMENTS.items():
        destination_dir = NDSH_ROOT / folder_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for framework, source_pattern in config["sources"].items():
            for horizon in config["horizons"]:
                source_path = REFERENCE_ROOT / source_pattern.format(h=horizon)
                destination_path = destination_dir / f"{framework}_fh{horizon}.ipynb"
                _patch_notebook(
                    source_path=source_path,
                    destination_path=destination_path,
                    dataset_relative=config["dataset"],
                    hierarchy_mode=config["hierarchy_mode"],
                    framework=framework,
                )

        readme = f"""# {folder_name}

- Dataset: `{config['dataset']}`
- Hierarchy mode: `{config['hierarchy_mode']}`
- Frameworks: FHF, HFF, HFFPS
- Forecast horizons: {', '.join(map(str, config['horizons']))}
- Non-DP parameters/evaluation: copied from the matching notebooks in the read-only FHF reference project.
- DP-FedAvg launcher: `DP_fhf/notebooks/01_DPFedAvg_NDSH_Matrix.ipynb`

Notebook outputs are intentionally not checked in; running a notebook writes its artifacts into this folder.
"""
        (destination_dir / "README.md").write_text(readme, encoding="utf-8")

    generate_dp_notebook()


if __name__ == "__main__":
    generate()
