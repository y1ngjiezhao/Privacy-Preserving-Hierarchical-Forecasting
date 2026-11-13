"""Expose sampling in existing notebooks and create isolated runnable examples.

Existing cell outputs are retained. Sampling examples use their own output
directory even when the example's rate is changed back to 1.0.
"""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from PPHF_modules.notebook_paths import normalize_notebook

import copy
import json
import re
from pathlib import Path

ROOT = PROJECT_ROOT


def write(path: Path, notebook: dict) -> None:
    normalize_notebook(notebook)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def prepare_example(notebook: dict) -> None:
    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python (neural-network)", "language": "python", "name": "neural-network"
    }
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"sampling-{i}"


def patch_non_dp(notebook: dict) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if re.search(r"^args = (Args|HFLArgs)\(\)", source, re.M) and "args.sampling_rate =" not in source:
            source = re.sub(
                r"^(args = (?:Args|HFLArgs)\(\))", r"\1\n\n"
                "# Uniform sampling without replacement. 1.0 preserves the original run.\n"
                "args.sampling_rate = 1.0\nargs.sampling_seed = None  # Defaults to args.seed",
                source, count=1, flags=re.M,
            )
        if "output_prefix =" in source and "from PPHF_modules.fhf.client_sampling import sampled_output_dir" not in source:
            source = re.sub(
                r"^(output_prefix = .+)$", r"\1\n"
                "from PPHF_modules.fhf.client_sampling import sampled_output_dir\n"
                "_sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed\n"
                "output_prefix = str(sampled_output_dir(Path(output_prefix).parent, "
                "args.sampling_rate, _sampling_seed) / Path(output_prefix).name)\n"
                "Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)",
                source, count=1, flags=re.M,
            )
        cell["source"] = source.splitlines(keepends=True)


def main() -> None:
    count = 0
    for folder in sorted(ROOT.glob("fhf*")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.ipynb")):
            if not re.fullmatch(r"(?:FHF|HFF|HFFPS)_fh\d+\.ipynb", path.name):
                continue
            notebook = json.loads(path.read_text(encoding="utf-8"))
            patch_non_dp(notebook)
            write(path, notebook)
            if path.stem.endswith("fh1"):
                example = copy.deepcopy(notebook)
                example["cells"].insert(0, {
                    "cell_type": "markdown", "metadata": {}, "source": [
                        f"# {path.stem.split('_')[0]}: training with client sampling\n\n",
                        "The default is `sampling_rate=0.1`. Each round uniformly samples `ceil(q × N)` training nodes without replacement.\n",
                        "HFF/HFFPS sample from all trainable series nodes, including trainable edge and cloud nodes.\n",
                        "Edit the sampling rate, round count, and seeds in the parameter cell, then select Run All.\n",
                        "Write results to `sampling_experiments/` in this directory, preserving the original experiment results.\n",
                    ],
                })
                for cell in example["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    source = source.replace("args.sampling_rate = 1.0", "args.sampling_rate = 0.1")
                    # Stable round counts are easier to compare under partial participation.
                    source = source.replace("args.early_stop_enabled = True", "args.early_stop_enabled = False")
                    source = source.replace(f"/ '{folder.name}' / f'fh", f"/ '{folder.name}' / 'sampling_experiments' / f'fh")
                    cell["source"] = source.splitlines(keepends=True)
                    cell["outputs"] = []
                    cell["execution_count"] = None
                prepare_example(example)
                write(path.with_name(path.stem.replace("_fh1", "_Sampling_fh1") + ".ipynb"), example)
                count += 1

    for dataset in ("GEF2017", "AusGrid2013"):
        path = ROOT / "DP_fhf/notebooks" / f"01_DPFedAvg_Only_{dataset}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        sweep_path = ROOT / "DP_fhf/notebooks" / f"01_DPFedAvg_{dataset}_Sweep.ipynb"
        for original_path, original in [(path, notebook), (sweep_path, json.loads(sweep_path.read_text(encoding="utf-8")))]:
            for cell in original["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if "DATASET =" in source and "SAMPLING_RATE =" not in source:
                    source = re.sub(r'^(DATASET = .+)$', r'\1\nSAMPLING_RATE = 1.0\nSAMPLING_SEED = None', source, count=1, flags=re.M)
                if re.search(r"(?:result|summary) = run_(?:fixed|sweep)_dpfedavg\(", source) and "sampling_rate=" not in source:
                    source = source.replace("    dry_run=", "    sampling_rate=SAMPLING_RATE,\n    sampling_seed=SAMPLING_SEED,\n    dry_run=", 1)
                cell["source"] = source.splitlines(keepends=True)
            write(original_path, original)

        example = copy.deepcopy(notebook)
        example["cells"][0]["source"] = [
            f"# {dataset}: DP-FedAvg training with client sampling\n\n",
            "The defaults are a sampling rate of `0.1`, 100 rounds, and deduplicated data. Edit the parameters, then select Run All.\n",
            "Each round uniformly samples `ceil(q × N)` clients without replacement, with at least one client.\n",
            "For q<1, RDP accounting automatically matches fixed-size sampling without replacement. SAMPLING_SEED=None uses fresh, hidden randomness for both sampling and noise.\n",
            "For add/remove adjacency with partial participation, noise is calibrated to 2C/m. Only the global models and privacy reports in model_release/ are covered by the accountant.\n",
            "Outputs are written separately to `DP_fhf/outputs/sampling_experiments/`, including forecasts, training residuals, privacy metadata, and per-round participation logs.\n",
        ]
        for cell in example["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = source.replace("SAMPLING_RATE = 1.0", "SAMPLING_RATE = 0.1")
            source = source.replace("SAMPLING_SEED = 42", "SAMPLING_SEED = None")
            if "SAMPLING_RATE =" in source:
                source += "\nNUM_ROUNDS = 100\nOVERWRITE = False\n"
            source = source.replace("num_rounds=100,", "num_rounds=NUM_ROUNDS,")
            source = source.replace("    overwrite=False,", "    overwrite=OVERWRITE,\n    output_base_dir=ROOT / 'DP_fhf' / 'outputs' / 'sampling_experiments' / DATASET,")
            cell["source"] = source.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
        prepare_example(example)
        write(path.with_name(f"01_DPFedAvg_Sampling_{dataset}.ipynb"), example)
        count += 1
    print(f"Created {count} sampling notebooks; preserved all existing cell outputs.")


if __name__ == "__main__":
    main()
