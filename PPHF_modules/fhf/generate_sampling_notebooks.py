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
                        f"# {path.stem.split('_')[0]}：客户端采样训练\n\n",
                        "默认 `sampling_rate=0.1`，每轮均匀无放回抽取 `ceil(q × N)` 个训练节点。\n",
                        "HFF/HFFPS 在所有可训练序列节点中抽取，包括可训练的边缘节点和云节点。\n",
                        "在参数单元格中修改采样率、轮数和种子，然后 Run All。\n",
                        "结果写入本目录的 `sampling_experiments/`，保留原实验结果。\n",
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
            f"# {dataset}：DP FedAvg 客户端采样训练\n\n",
            "默认采样率 `0.1`，100 轮，使用去重数据。修改参数后 Run All。\n",
            "每轮均匀无放回抽取 `ceil(q × N)` 个客户端，至少一个。\n",
            "q<1 自动使用匹配固定大小无放回采样的 RDP 会计。SAMPLING_SEED=None 使用新鲜隐藏的采样及噪声随机性。\n",
            "add/remove 部分参与的噪声按 2C/m 标定。只有 model_release/ 中的全局模型与隐私报告属于会计覆盖的输出。\n",
            "输出写入独立的 `DP_fhf/outputs/sampling_experiments/`，包含预测、训练残差、隐私元数据和逐轮参与日志。\n",
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
