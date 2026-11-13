from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

"""Shared, FHF-only runtime used by the portable DP-FHF notebooks."""

from collections import deque
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

from PPHF_modules.fhf.client_sampling import sampled_output_dir, validate_sampling_rate, amplification_output_dir
from PPHF_modules.dp_fhf.privacy_accounting import check_sampling_accountant_dependencies


FHF_PREFIX = "fh1_FHF"
DP_PREFIX = f"{FHF_PREFIX}_dp_fedavg"


def _first_file(candidates: Iterable[str | Path | None], description: str) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Cannot locate {description}. Checked:\n- " + "\n- ".join(checked)
    )


def locate_dataset_csv(
    root: str | Path,
    dataset: str,
    override: str | Path | None = None,
) -> Path:
    """Return the canonical deduplicated long-format input for a dataset."""
    root = Path(root).resolve()
    if override:
        return _first_file([override], f"{dataset} CSV override")
    if dataset == "GEF2017":
        directory = root / "Datasets" / "GEFCOM2017"
        return _first_file(
            [
                os.environ.get("DP_FHF_GEF2017_CSV"),
                directory / "GEF2017_170101_dedup.csv",
                *sorted(directory.glob("*dedup*.csv")),
            ],
            "deduplicated GEF2017 CSV",
        )
    if dataset == "AusGrid2013":
        directory = root / "Datasets" / "AusGrid2013"
        return _first_file(
            [
                os.environ.get("DP_FHF_AUSGRID2013_CSV"),
                *sorted(directory.glob("*dedup*.csv")),
                directory / "AusGrid2013_univariate.csv",
            ],
            "deduplicated AusGrid2013 CSV",
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def locate_fhf_artifact_dir(
    root: str | Path,
    dataset: str,
    required: Sequence[str] = (),
) -> Path:
    """Find storage containing FHF-prefixed artifacts, regardless of legacy folder name."""
    root = Path(root).resolve()
    names = {
        "GEF2017": ("fhf_dedup_gef2017",),
        "AusGrid2013": ("fhf_dedup_ausgrid2013",),
    }.get(dataset)
    if names is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    checked = []
    for name in names:
        directory = root / name
        checked.append(str(directory))
        if directory.is_dir() and all((directory / filename).is_file() for filename in required):
            return directory.resolve()
    raise FileNotFoundError(
        f"Cannot locate {dataset} FHF artifacts containing {list(required)}. "
        f"Checked: {checked}"
    )


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_streaming(command: Sequence[str], root: Path) -> None:
    process = subprocess.Popen(
        list(command),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=60)
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Child process failed with exit code {return_code}.\n"
            f"Last output:\n{''.join(tail)}"
        )


def _validate_runner(root: Path) -> Path:
    runner = root / "PPHF_modules" / "dp_fhf" / "run_dp_fedavg.py"
    check = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check.returncode:
        raise RuntimeError(check.stdout)
    required = ("--clip-norm", "--noise-multiplier", "--shared-artifact-dir",
                "--sampling-rate", "--sampling-seed")
    missing = [option for option in required if option not in check.stdout]
    if missing:
        raise RuntimeError(f"Runner does not support required options: {missing}")
    return runner


def _fedavg_command(
    root: Path,
    runner: Path,
    csv_path: Path,
    output_dir: Path,
    num_rounds: int,
    clip_norm: float,
    noise_multiplier: float,
    sampling_rate: float = 1.0,
    sampling_seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(runner),
        "--framework",
        "fhf",
        "--csv",
        str(csv_path),
        "--device",
        _device(),
        "--forecast-horizon",
        "1",
        "--hierarchy-mode",
        "dedup",
        "--lags",
        "48",
        "--fh",
        "0",
        "--num-rounds",
        str(num_rounds),
        "--local-epochs",
        "1",
        "--hidden-size",
        "64",
        "--num-layers",
        "2",
        "--batch-size",
        "256",
        "--torch-num-threads",
        "1",
        "--clip-norm",
        str(clip_norm),
        "--noise-multiplier",
        str(noise_multiplier),
        "--adjacency",
        "add_remove",
        "--weighting",
        "uniform",
        "--seed",
        "42",
        "--sampling-rate",
        str(validate_sampling_rate(sampling_rate)),
        "--accounting-delta",
        "1e-5",
        "--print-every",
        "1",
        "--heartbeat-seconds",
        "60",
        # Bounds live beside their result and cannot conflict with old inputs.
        "--shared-artifact-dir",
        str(output_dir),
        "--output-dir",
        str(output_dir),
    ]
    if sampling_seed is not None:
        command += ["--sampling-seed", str(sampling_seed)]
    if sampling_rate == 1.0 or sampling_seed is not None:
        command += ["--dp-seed", "42"]
    return command


def _result_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "output_dir": output_dir,
        "forecast": output_dir / f"{DP_PREFIX}_base_forecasts.csv",
        "privacy": output_dir / f"{DP_PREFIX}_privacy.json",
        "artifacts": output_dir / f"{DP_PREFIX}_training_artifacts.npz",
    }


def _is_complete(paths: dict[str, Path]) -> bool:
    return all(paths[key].is_file() for key in ("forecast", "privacy", "artifacts"))


def _validate_completed_configuration(paths, *, num_rounds, clip_norm, noise_multiplier,
                                      sampling_rate, sampling_seed):
    metadata = json.loads(paths["privacy"].read_text(encoding="utf-8"))
    expected = {"num_rounds": int(num_rounds), "clip_norm": float(clip_norm),
                "noise_multiplier": float(noise_multiplier), "sampling_rate": float(sampling_rate)}
    actual = {key: metadata.get(key, 1.0 if key == "sampling_rate" else None) for key in expected}
    if sampling_rate < 1.0:
        expected["sampling_seed"] = None if sampling_seed is None else int(sampling_seed)
        actual["sampling_seed"] = metadata.get("sampling_seed")
        expected["mechanism_version"] = "fixed_wor_gaussian_v1"
        actual["mechanism_version"] = metadata.get("mechanism_version")
    if actual != expected:
        raise ValueError(
            f"Completed result has different parameters under {paths['output_dir']}: "
            f"saved={actual}, requested={expected}. Set overwrite=True or use a new output directory."
        )


def run_fixed_dpfedavg(
    *,
    root: str | Path,
    dataset: str,
    csv_path: str | Path,
    num_rounds: int,
    clip_norm: float,
    noise_multiplier: float,
    dry_run: bool = False,
    overwrite: bool = False,
    sampling_rate: float = 1.0,
    sampling_seed: int | None = None,
    output_base_dir: str | Path | None = None,
):
    """Run one fixed FHF DP-FedAvg experiment and return its metadata table."""
    import pandas as pd

    root = Path(root).resolve()
    csv_path = Path(csv_path).resolve()
    sampling_rate = validate_sampling_rate(sampling_rate)
    if sampling_rate < 1.0:
        check_sampling_accountant_dependencies()
    output_dir = (Path(output_base_dir).expanduser().resolve() if output_base_dir is not None
                  else root / "DP_fhf" / "outputs" / "01_dp_fedavg_100round" / dataset)
    output_dir = sampled_output_dir(amplification_output_dir(output_dir, sampling_rate < 1.0), sampling_rate, sampling_seed)
    runner = _validate_runner(root)
    command = _fedavg_command(
        root, runner, csv_path, output_dir, num_rounds, clip_norm, noise_multiplier,
        sampling_rate, sampling_seed,
    )
    paths = _result_paths(output_dir)
    print("Framework: FHF only")
    print("Rounds / clip / noise:", num_rounds, clip_norm, noise_multiplier)
    print("Sampling rate / seed:", sampling_rate, sampling_seed)
    print("Output:", output_dir)
    if dry_run:
        print("Command:", " ".join(command))
        return pd.DataFrame(
            [{
                "dataset": dataset,
                "framework": "FHF",
                "rounds": num_rounds,
                "clip_norm": clip_norm,
                "noise_multiplier": noise_multiplier,
                "status": "dry_run",
                "sampling_rate": sampling_rate,
            }]
        )
    if _is_complete(paths) and not overwrite:
        _validate_completed_configuration(
            paths, num_rounds=num_rounds, clip_norm=clip_norm, noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate, sampling_seed=sampling_seed,
        )
        print("SKIP: completed result already exists.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_streaming(command, root)
    if not _is_complete(paths):
        raise FileNotFoundError(f"Incomplete result under {output_dir}")
    metadata = json.loads(paths["privacy"].read_text(encoding="utf-8"))
    if metadata.get("framework", "").lower() != "fhf":
        raise ValueError(f"Non-FHF metadata produced: {paths['privacy']}")
    return pd.DataFrame(
        [{
            "dataset": dataset,
            "framework": "FHF",
            "rounds": num_rounds,
            "clip_norm": clip_norm,
            "noise_multiplier": noise_multiplier,
            "epsilon": metadata["accounting"]["epsilon"],
            "delta": metadata["accounting"]["delta"],
            "forecast": str(paths["forecast"]),
            "sampling_rate": metadata.get("sampling_rate", 1.0),
            "epsilon_without_sampling_amplification": metadata["accounting"].get("epsilon_without_sampling_amplification", metadata["accounting"]["epsilon"]),
            "sampling_amplification_used": metadata["accounting"]["sampling_amplification_used"],
            "budget_status": metadata["accounting"].get("budget_status", "historical"),
        }]
    )


def _number_slug(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def _evaluate_forecasts(forecasts, scale_by_series, label: str):
    import numpy as np
    import pandas as pd

    rows = []
    for unique_id, group in forecasts.groupby("unique_id", sort=True):
        unique_id = str(unique_id)
        if unique_id not in scale_by_series:
            raise KeyError(f"No training scale found for {unique_id!r}")
        error = group["base"].to_numpy(float) - group["y_true"].to_numpy(float)
        mae = float(np.nanmean(np.abs(error)))
        mse = float(np.nanmean(error**2))
        mase_denom, rmsse_denom_sq = scale_by_series[unique_id]
        rows.append(
            {
                "source": label,
                "unique_id": unique_id,
                "MAE": mae,
                "RMSE": float(np.sqrt(mse)),
                "MASE": mae / mase_denom,
                "RMSSE": float(np.sqrt(mse / rmsse_denom_sq)),
            }
        )
    per_series = pd.DataFrame(rows)
    metrics = {
        name: float(per_series[name].mean())
        for name in ("MAE", "RMSE", "MASE", "RMSSE")
    }
    return per_series, metrics


def _scales_from_artifact(path: Path):
    import numpy as np

    with np.load(path, allow_pickle=False) as stored:
        unique_ids = [str(value) for value in stored["unique_ids"]]
        targets = np.asarray(stored["train_targets"], dtype=np.float64)
    if targets.ndim == 2:
        targets = targets[:, :, None]
    scales = {}
    for index, unique_id in enumerate(unique_ids):
        differences = targets[index, 1:, :] - targets[index, :-1, :]
        scales[unique_id] = (
            float(np.nanmean(np.abs(differences))),
            float(np.nanmean(differences**2)),
        )
    return scales


def run_sweep_dpfedavg(
    *,
    root: str | Path,
    dataset: str,
    csv_path: str | Path,
    noise_multipliers: Sequence[float],
    clip_norms: Sequence[float],
    num_rounds: int = 20,
    dry_run: bool = False,
    overwrite: bool = False,
    stop_on_error: bool = False,
    sampling_rate: float = 1.0,
    sampling_seed: int | None = None,
    output_base_dir: str | Path | None = None,
):
    """Run/resume one dataset's sweep, optionally beneath a custom output base.

    Partial-participation version and sampling directories are added to the
    base automatically, as for the fixed-run notebook helper.
    """
    import numpy as np
    import pandas as pd

    root = Path(root).resolve()
    csv_path = Path(csv_path).resolve()
    sampling_rate = validate_sampling_rate(sampling_rate)
    if sampling_rate < 1.0:
        check_sampling_accountant_dependencies()
    runner = _validate_runner(root)
    sweep_root = (Path(output_base_dir).expanduser().resolve() if output_base_dir is not None else
        root / "DP_fhf" / "outputs" / "01_dp_fedavg_sweep_20round" / dataset
    )
    sweep_root = sampled_output_dir(amplification_output_dir(sweep_root, sampling_rate < 1.0), sampling_rate, sampling_seed)
    rows = []
    commands = []
    for noise, clip in product(noise_multipliers, clip_norms):
        run_id = f"nm_{_number_slug(noise)}__clip_{_number_slug(clip)}"
        output_dir = sweep_root / run_id
        rows.append({"run_id": run_id, "noise_multiplier": noise, "clip_norm": clip,
                     "sampling_rate": sampling_rate, "output_dir": str(output_dir)})
        commands.append(
            (
                run_id,
                output_dir,
                _fedavg_command(root, runner, csv_path, output_dir, num_rounds, clip, noise,
                                sampling_rate, sampling_seed),
            )
        )
    grid = pd.DataFrame(rows)
    print(grid)
    print(f"Runs: {len(grid)} ({len(noise_multipliers)} × {len(clip_norms)})")
    print("Framework / rounds: FHF /", num_rounds)
    print("Sampling rate / output:", sampling_rate, sweep_root)
    if dry_run:
        for run_id, _, command in commands:
            print(run_id, " ".join(command))
        return grid.assign(dataset=dataset, framework="FHF", num_rounds=num_rounds)

    sweep_root.mkdir(parents=True, exist_ok=True)
    grid.to_csv(sweep_root / "parameter_grid.csv", index=False)
    failures = []
    for position, (run_id, output_dir, command) in enumerate(commands, start=1):
        # The CLI isolates partial-participation runs beneath each run directory.
        paths = _result_paths(sampled_output_dir(output_dir, sampling_rate, sampling_seed))
        if _is_complete(paths) and not overwrite:
            parameters = next(row for row in rows if row["run_id"] == run_id)
            _validate_completed_configuration(
                paths, num_rounds=num_rounds, clip_norm=parameters["clip_norm"],
                noise_multiplier=parameters["noise_multiplier"], sampling_rate=sampling_rate,
                sampling_seed=sampling_seed,
            )
            print(f"[{position}/{len(commands)}] SKIP completed: {run_id}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== [{position}/{len(commands)}] {run_id} =====")
        try:
            _run_streaming(command, root)
        except Exception as exc:
            failures.append({"run_id": run_id, "error": str(exc)})
            pd.DataFrame(failures).to_csv(sweep_root / "sweep_failures.csv", index=False)
            if stop_on_error:
                raise

    completed = []
    for row in grid.itertuples(index=False):
        paths = _result_paths(sampled_output_dir(sweep_root / row.run_id, sampling_rate, sampling_seed))
        if _is_complete(paths):
            completed.append((row, paths))
    if not completed:
        if failures:
            raise RuntimeError(
                f"No sweep runs completed successfully ({len(failures)} failed). "
                f"See {sweep_root / 'sweep_failures.csv'}. "
                "Saved forecast/NPZ files may still be recoverable; do not blindly retrain."
            )
        print("No completed runs to summarize.")
        return pd.DataFrame()

    scales = _scales_from_artifact(completed[0][1]["artifacts"])
    baseline_metrics = None
    try:
        baseline_dir = locate_fhf_artifact_dir(
            root, dataset, (f"{FHF_PREFIX}_forecasts.csv",)
        )
    except FileNotFoundError:
        print(f"{dataset}: non-DP FHF baseline not found; saving DP metrics only.")
    else:
        baseline_path = baseline_dir / f"{FHF_PREFIX}_forecasts.csv"
        baseline_per_series, baseline_metrics = _evaluate_forecasts(
            pd.read_csv(baseline_path), scales, "non_dp_fhf"
        )
        baseline_per_series.to_csv(
            sweep_root / "non_dp_baseline_per_series_metrics.csv", index=False
        )

    summary_rows = []
    for row, paths in completed:
        per_series, metrics = _evaluate_forecasts(pd.read_csv(paths["forecast"]), scales, row.run_id)
        per_series.to_csv(paths["output_dir"] / "per_series_metrics.csv", index=False)
        privacy = json.loads(paths["privacy"].read_text(encoding="utf-8"))
        if privacy.get("framework", "").lower() != "fhf":
            raise ValueError(f"Non-FHF sweep result: {paths['privacy']}")
        result = {
            "dataset": dataset,
            "framework": "FHF",
            "run_id": row.run_id,
            "noise_multiplier": row.noise_multiplier,
            "clip_norm": row.clip_norm,
            "num_rounds": num_rounds,
            "sampling_rate": sampling_rate,
            "sampling_amplification_used": privacy["accounting"]["sampling_amplification_used"],
            "epsilon": float(privacy["accounting"]["epsilon"]),
            "delta": float(privacy["accounting"]["delta"]),
            **metrics,
        }
        if baseline_metrics is not None:
            if sampling_rate < 1.0:
                result["non_dp_baseline_sampling_rate"] = 1.0
            for metric in ("MAE", "RMSE", "MASE", "RMSSE"):
                result[f"non_dp_{metric}"] = baseline_metrics[metric]
                result[f"delta_{metric}"] = metrics[metric] - baseline_metrics[metric]
        pd.DataFrame([result]).to_csv(
            paths["output_dir"] / "comparison_metrics.csv", index=False
        )
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows).sort_values(["MASE", "RMSSE", "epsilon"])
    summary.to_csv(sweep_root / "sweep_summary.csv", index=False)
    return summary


def summarize_dpfedavg(root: str | Path, dry_run: bool = False):
    import pandas as pd

    root = Path(root).resolve()
    dp_dir = root / "DP_fhf"
    fixed_root = dp_dir / "outputs" / "01_dp_fedavg_100round"
    sweep_root = dp_dir / "outputs" / "01_dp_fedavg_sweep_20round"
    rows = []
    for dataset in ("GEF2017", "AusGrid2013"):
        fixed_dir = fixed_root / dataset
        privacy_path = fixed_dir / f"{DP_PREFIX}_privacy.json"
        if privacy_path.is_file():
            metadata = json.loads(privacy_path.read_text(encoding="utf-8"))
            if metadata.get("framework", "").lower() != "fhf":
                raise ValueError(f"Non-FHF result: {privacy_path}")
            rows.append(
                {
                    "dataset": dataset,
                    "experiment": "100round",
                    "run_id": "fixed",
                    "rounds": 100,
                    "clip_norm": 1.0,
                    "noise_multiplier": 0.3 if dataset == "GEF2017" else 1.2,
                    "sampling_rate": metadata.get("sampling_rate", 1.0),
                    "epsilon": metadata["accounting"]["epsilon"],
                    "delta": metadata["accounting"]["delta"],
                }
            )
        sweep_path = sweep_root / dataset / "sweep_summary.csv"
        if sweep_path.is_file():
            sweep = pd.read_csv(sweep_path)
            if "framework" in sweep and not sweep["framework"].eq("FHF").all():
                raise ValueError(f"Non-FHF result: {sweep_path}")
            for item in sweep.to_dict("records"):
                rows.append(
                    {
                        "dataset": dataset,
                        "experiment": "20round_sweep",
                        "run_id": item["run_id"],
                        "rounds": item["num_rounds"],
                        "clip_norm": item["clip_norm"],
                        "noise_multiplier": item["noise_multiplier"],
                        "sampling_rate": item.get("sampling_rate", 1.0),
                        "epsilon": item["epsilon"],
                        "delta": item["delta"],
                        "MASE": item.get("MASE"),
                        "RMSSE": item.get("RMSSE"),
                    }
                )
    sampling_root = dp_dir / "outputs" / "ausgrid2013_sampling_sweep"
    if sampling_root.is_dir():
        from PPHF_modules.dp_fhf.ausgrid_sampling_sweep import summarize_ausgrid_sampling_sweeps
        sampled = summarize_ausgrid_sampling_sweeps(sampling_root, dry_run=dry_run)
        for item in sampled.to_dict("records"):
            rows.append({
                "dataset": item["dataset"], "experiment": "ausgrid_sampling_sweep",
                "run_id": item["run_id"], "rounds": item["num_rounds"],
                "clip_norm": item["clip_norm"], "noise_multiplier": item["noise_multiplier"],
                "sampling_rate": item["sampling_rate"], "epsilon": item["epsilon"],
                "delta": item["delta"], "MASE": item.get("MASE"), "RMSSE": item.get("RMSSE"),
                "source_summary": item["source_summary"],
            })
    results = pd.DataFrame(rows)
    if results.empty:
        print("No completed DP-FedAvg results found yet.")
    else:
        results = results.sort_values(["dataset", "experiment", "sampling_rate", "run_id"])
        output = dp_dir / "outputs" / "01_dp_fedavg_summary.csv"
        if dry_run:
            print("Dry run: summary output would be", output)
        else:
            results.to_csv(output, index=False)
            print("Saved:", output)
    return results


def run_covariance_notebook(
    root: str | Path,
    datasets: Sequence[str] = ("AusGrid2013", "GEF2017"),
    dry_run: bool = False,
    debias_noise_variance: bool = False,
):
    """Run FHF covariance branches; the paper uses no noise-variance correction.

    The default writes to each dataset's no_debias/ directory. Pass
    debias_noise_variance=True only to reproduce the historical corrected
    ablation; those outputs are generated only on explicit request.
    """
    import pandas as pd

    root = Path(root).resolve()
    dp_dir = root / "DP_fhf"
    runner = DP_MODULE_DIR / "run_dp_covariance.py"
    output_base = dp_dir / "outputs" / "02_dp_covariance"
    fixed_root = dp_dir / "outputs" / "01_dp_fedavg_100round"
    commands = []
    planned = []
    for dataset in datasets:
        artifact_dir = locate_fhf_artifact_dir(root, dataset)
        branches = {}
        non_dp_pair = (
            artifact_dir / f"{FHF_PREFIX}_forecasts.csv",
            artifact_dir / f"{FHF_PREFIX}_training_artifacts.npz",
        )
        if all(path.is_file() for path in non_dp_pair):
            branches["non_dp_forecasts"] = non_dp_pair
        fixed_dir = fixed_root / dataset
        dp_pair = (
            fixed_dir / f"{DP_PREFIX}_base_forecasts.csv",
            fixed_dir / f"{DP_PREFIX}_training_artifacts.npz",
        )
        if all(path.is_file() for path in dp_pair):
            branches["dp_fedavg_100round"] = dp_pair
        if not branches:
            print(
                f"SKIP {dataset}: no completed FHF forecast/training-artifact pair yet."
            )
            continue
        output_dir = output_base / dataset
        if not debias_noise_variance:
            output_dir = output_dir / "no_debias"
        for branch, (forecasts, residuals) in branches.items():
            label = f"FHF_{branch}"
            command = [
                sys.executable, "-u", str(runner),
                "--forecasts", str(forecasts),
                "--epsilon", "1", "--mechanism", "laplace", "--delta", "1e-5",
                "--lower", "-0.0005", "--upper", "0.0005", "--seed", "42",
                "--ridge", "1e-6", "--output-dir", str(output_dir),
                "--output-label", label, "--residuals-npz", str(residuals),
            ]
            if not debias_noise_variance:
                command.append("--no-debias")
            per_node_metrics = artifact_dir / f"{FHF_PREFIX}_per_node_metrics.csv"
            if per_node_metrics.is_file():
                command.extend(["--per-node-metrics", str(per_node_metrics)])
            commands.append(command)
            planned.append((dataset, branch, label, output_dir))
    if dry_run:
        subprocess.run([sys.executable, str(runner), "--help"], cwd=root, check=True)
        for command in commands:
            print(" ".join(command))
        return pd.DataFrame(
            [{"dataset": d, "framework": "FHF", "input": b} for d, b, _, _ in planned]
        )
    if not commands:
        return pd.DataFrame()
    for command in commands:
        Path(command[command.index("--output-dir") + 1]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=root, check=True)
    tables = []
    for dataset, branch, label, output_dir in planned:
        table = pd.read_csv(output_dir / f"{label}_dp_covariance_metrics.csv")
        table.insert(0, "input", branch)
        table.insert(0, "framework", "FHF")
        table.insert(0, "dataset", dataset)
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True).groupby(
        ["dataset", "framework", "input", "method"], as_index=False
    )[["MASE", "RMSSE"]].mean()
    for dataset, group in summary.groupby("dataset"):
        summary_dir = output_base / dataset
        if not debias_noise_variance:
            summary_dir = summary_dir / "no_debias"
        group.to_csv(summary_dir / "dp_covariance_MASE_RMSSE_summary.csv", index=False)
    return summary


def run_o2mr_notebook(
    root: str | Path,
    datasets: Sequence[str] = ("AusGrid2013", "GEF2017"),
    dry_run: bool = False,
):
    """Run only FHF DP-O2MR branches, reusing saved FHF P matrices."""
    import pandas as pd

    root = Path(root).resolve()
    dp_dir = root / "DP_fhf"
    runner = DP_MODULE_DIR / "run_dp_o2mr_saved_p.py"
    output_base = dp_dir / "outputs" / "03_dp_o2mr"
    fixed_root = dp_dir / "outputs" / "01_dp_fedavg_100round"
    commands = []
    planned = []
    for dataset in datasets:
        artifact_dir = locate_fhf_artifact_dir(root, dataset)
        saved_p = list(artifact_dir.glob(f"{FHF_PREFIX}_o2mr_P_*.csv"))
        node_map = artifact_dir / f"{FHF_PREFIX}_o2mr_node_map.csv"
        if len(saved_p) < 6 or not node_map.is_file():
            print(f"SKIP {dataset}: six saved FHF P matrices are not available yet.")
            continue
        sources = {}
        non_dp_source = artifact_dir / f"{FHF_PREFIX}_forecasts.csv"
        non_dp_bounds = artifact_dir / "fh1_shared_event_dp_train_bounds_by_time_slot.csv"
        if non_dp_source.is_file() and non_dp_bounds.is_file():
            sources["non_dp_forecasts"] = (non_dp_source, non_dp_bounds)
        fixed_dir = fixed_root / dataset
        dp_source = fixed_dir / f"{DP_PREFIX}_base_forecasts.csv"
        dp_bounds = fixed_dir / "fh1_shared_event_dp_train_bounds_by_time_slot.csv"
        if dp_source.is_file() and dp_bounds.is_file():
            sources["dp_fedavg_100round"] = (dp_source, dp_bounds)
        if not sources:
            print(f"SKIP {dataset}: no matching FHF forecast/bounds pair yet.")
            continue
        output_dir = output_base / dataset
        for label, (source, bounds) in sources.items():
            for epsilon in (1, 5, 10, 20):
                file_prefix = f"FHF_{label}_epsilon_{epsilon}"
                command = [
                    sys.executable, "-u", str(runner),
                    "--base-forecasts", str(source),
                    "--artifact-dir", str(artifact_dir),
                    "--artifact-prefix", FHF_PREFIX,
                    "--bounds-file", str(bounds),
                    "--output-dir", str(output_dir),
                    "--output-label", label,
                    "--file-prefix", file_prefix,
                    "--epsilon", str(epsilon),
                    "--mechanism", "laplace", "--delta", "1e-5", "--seed", "20260727",
                ]
                per_node_metrics = artifact_dir / f"{FHF_PREFIX}_per_node_metrics.csv"
                if per_node_metrics.is_file():
                    command.extend(["--per-node-metrics", str(per_node_metrics)])
                commands.append(command)
                planned.append((dataset, label, epsilon, file_prefix, output_dir))
    if dry_run:
        subprocess.run([sys.executable, str(runner), "--help"], cwd=root, check=True)
        for command in commands:
            print(" ".join(command))
        return pd.DataFrame(
            [
                {"dataset": d, "framework": "FHF", "input": label, "epsilon": eps}
                for d, label, eps, _, _ in planned
            ]
        )
    if not commands:
        return pd.DataFrame()
    for command in commands:
        Path(command[command.index("--output-dir") + 1]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=root, check=True)
    tables = []
    for dataset, label, epsilon, prefix, output_dir in planned:
        table = pd.read_csv(output_dir / f"{prefix}_dp_o2mr_metrics.csv")
        table = table.loc[table["release"] == "dp_o2mr"].copy()
        table.insert(0, "epsilon", epsilon)
        table.insert(0, "framework", "FHF")
        table.insert(0, "dataset", dataset)
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True)[
        ["dataset", "framework", "input_label", "epsilon", "method", "MASE", "RMSSE"]
    ]
    for dataset, group in summary.groupby("dataset"):
        output_dir = output_base / dataset
        group.to_csv(output_dir / "dp_o2mr_MASE_RMSSE_summary.csv", index=False)
        for path in output_dir.glob("*_dp_o2mr_privacy.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata["P_reestimated"] is not False:
                raise AssertionError(path)
    return summary
