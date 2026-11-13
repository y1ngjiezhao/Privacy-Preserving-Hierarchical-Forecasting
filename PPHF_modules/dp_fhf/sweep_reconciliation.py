"""Reconcile saved DP-FedAvg sweeps with the project's original FHF engine.

No training is repeated. Covariance estimation uses each run's saved training
residuals; test targets are used only for evaluation. Outputs are always
recomputed and published only after the whole run passes validation.
"""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch.nn as nn

SHARED = FHF_MODULE_DIR
from PPHF_modules.fhf.FHF_Cloud import Cloud

PREFIX = "fh1_FHF_dp_fedavg"
METHODS = {"mints": "mint_shrinkage", "wlsv": "mint_var", "ols": "mint_ols"}
METRICS = ["MAE", "RMSE", "MASE", "RMSSE"]


@dataclass
class SweepRun:
    source_dir: Path
    client_ids: np.ndarray
    unique_ids: np.ndarray
    horizons: np.ndarray
    times: pd.DatetimeIndex
    base: np.ndarray
    truth: np.ndarray
    residuals: np.ndarray
    train_targets: np.ndarray
    metadata: dict


def _finite(values, label):
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing, NaN or infinite values.")


def load_sweep_run(run_dir: str | Path) -> SweepRun:
    directory = Path(run_dir).resolve()
    metadata = json.loads((directory / f"{PREFIX}_privacy.json").read_text())
    if metadata.get("framework") != "fhf" or not metadata.get("enabled"):
        raise ValueError(f"Expected an enabled FHF DP-FedAvg run: {directory}")
    with np.load(directory / f"{PREFIX}_training_artifacts.npz", allow_pickle=False) as saved:
        ids = saved["client_ids"].astype(int)
        names = saved["unique_ids"].astype(str)
        residuals = saved["residuals"].astype(float)
        targets = saved["train_targets"].astype(float)
        fitted = saved["train_predictions"].astype(float)
    if len(np.unique(ids)) != len(ids) or len(np.unique(names)) != len(ids):
        raise ValueError("Training node IDs and names must be unique and aligned.")
    if targets.ndim != 3 or targets.shape != residuals.shape or targets.shape != fitted.shape:
        raise ValueError("Training residuals, targets and predictions must share shape (nodes, times, horizons).")
    if targets.shape[0] != len(ids) or targets.shape[1] < 2:
        raise ValueError("Training artifacts have incompatible node or time dimensions.")
    for label, array in [("residuals", residuals), ("train_targets", targets), ("train_predictions", fitted)]:
        _finite(array, label)
    if not np.allclose(residuals, targets - fitted, rtol=1e-10, atol=1e-10):
        raise ValueError("Training residuals are not train_targets - train_predictions.")

    table = pd.read_csv(directory / f"{PREFIX}_base_forecasts.csv",
                        usecols=["node_id", "unique_id", "ds", "horizon_index", "y_true", "base"])
    table["ds"] = pd.to_datetime(table["ds"], errors="raise")
    mapping = table[["node_id", "unique_id"]].drop_duplicates()
    expected = dict(zip(ids.tolist(), names.tolist()))
    if len(mapping) != len(ids) or dict(zip(mapping.node_id, mapping.unique_id)) != expected:
        raise ValueError("Forecast node IDs/names do not match the training artifacts.")
    if table[["node_id", "ds", "horizon_index"]].isna().any().any():
        raise ValueError("Forecast keys contain missing values.")
    if table.duplicated(["node_id", "ds", "horizon_index"]).any():
        raise ValueError("Duplicate forecast node/time/horizon keys.")
    horizons = np.sort(table.horizon_index.unique())
    if not np.array_equal(horizons, np.arange(targets.shape[2])):
        raise ValueError("Forecast horizons do not match training artifacts.")
    times = pd.DatetimeIndex(sorted(table.ds.unique()))
    if not len(times) or len(table) != len(ids) * len(times) * len(horizons):
        raise ValueError("Forecasts must contain a complete node/time/horizon grid.")
    cubes = []
    for column in ["base", "y_true"]:
        cube = np.stack([
            table.loc[table.horizon_index == h].pivot(index="node_id", columns="ds", values=column)
            .reindex(index=ids, columns=times).to_numpy(dtype=float)
            for h in horizons
        ], axis=2)
        _finite(cube, column)
        cubes.append(cube)
    return SweepRun(directory, ids, names, horizons, times, *cubes, residuals, targets, metadata)


def estimate_matrices(run: SweepRun, ridge: float = 1e-6) -> dict[str, np.ndarray]:
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive.")
    engine = Cloud(nn.Identity(), run.client_ids.tolist())
    S, names, bottoms = engine.build_hierarchy_S(
        dict(zip(run.client_ids.tolist(), run.unique_ids.tolist())), reject_duplicate_series=True)
    Ps, Gs = engine.compute_reconciliation_matrices(
        methods=list(METHODS.values()),
        resid_by_cid=dict(zip(run.client_ids.tolist(), run.residuals)),
        ridge=ridge, dp_variance_by_cid=None)
    matrices = {"S": S, "bottom_indices": np.array([names.index(name) for name in bottoms])}
    for label, method in METHODS.items():
        matrices[f"P_{label}"] = Ps[method]
        matrices[f"G_{label}"] = Gs[method]
        matrices[f"W_{label}"] = engine.Ws_by_method[method]
    return matrices


def reconcile_forecasts(base: np.ndarray, matrices: dict) -> dict[str, np.ndarray]:
    predictions = {"base": base}
    for method in METHODS:
        predictions[method] = np.stack([
            matrices[f"P_{method}"][h] @ base[:, :, h]
            for h in range(base.shape[2])
        ], axis=2)
        _finite(predictions[method], method)
    return predictions


def evaluate_predictions(run: SweepRun, predictions: dict) -> pd.DataFrame:
    differences = np.diff(run.train_targets, axis=1)
    mae_scale = np.mean(np.abs(differences), axis=(1, 2))
    mse_scale = np.mean(differences ** 2, axis=(1, 2))
    if np.any(mae_scale <= 0) or np.any(mse_scale <= 0):
        raise ValueError("MASE/RMSSE require positive training first-difference scales.")
    rows = []
    for method, forecast in predictions.items():
        if forecast.shape != run.truth.shape:
            raise ValueError(f"Forecast shape mismatch for {method}.")
        _finite(forecast, method)
        error = forecast - run.truth
        mae = np.mean(np.abs(error), axis=(1, 2))
        mse = np.mean(error ** 2, axis=(1, 2))
        for index, (node_id, name) in enumerate(zip(run.client_ids, run.unique_ids)):
            rows.append({"method": method, "node_id": int(node_id), "unique_id": str(name),
                         "level": str(name).count("/") + 1, "MAE": float(mae[index]),
                         "RMSE": float(np.sqrt(mse[index])),
                         "MASE": float(mae[index] / mae_scale[index]),
                         "RMSSE": float(np.sqrt(mse[index] / mse_scale[index]))})
    return pd.DataFrame(rows)


def _diagnostics(run, matrices, predictions):
    rows = []
    S, bottom_indices = matrices["S"], matrices["bottom_indices"]
    for method in METHODS:
        for h in range(len(run.horizons)):
            P = matrices[f"P_{method}"][h]
            forecast = predictions[method][:, :, h]
            ps_error = float(np.max(np.abs(P @ S - S)))
            coherence = float(np.max(np.abs(forecast - S @ forecast[bottom_indices])))
            relative_coherence = coherence / max(1., float(np.max(np.abs(forecast))))
            idempotence = float(np.max(np.abs(P @ P - P)))
            if ps_error > 1e-7 or relative_coherence > 1e-9 or idempotence > 1e-7:
                raise ValueError(f"Reconciliation validation failed: {method}, PS={ps_error}, coherence={relative_coherence}.")
            rows.append({"method": method, "horizon_index": int(run.horizons[h]),
                         "PS_minus_S_max_abs": ps_error, "P_squared_minus_P_max_abs": idempotence,
                         "coherence_max_abs": coherence, "coherence_relative": relative_coherence})
    return pd.DataFrame(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_forecasts(path, run, predictions):
    n, t, h = run.base.shape
    data = {"node_id": np.repeat(run.client_ids, t * h),
            "unique_id": np.repeat(run.unique_ids, t * h),
            "ds": np.tile(np.repeat(run.times.to_numpy(), h), n),
            "horizon_index": np.tile(run.horizons, n * t),
            "y_true": run.truth.ravel()}
    data.update({method: values.ravel() for method, values in predictions.items()})
    pd.DataFrame(data).to_csv(path, index=False, chunksize=100_000,
                             compression={"method": "gzip", "compresslevel": 1, "mtime": 0})


def _publish_directory(staging: Path, destination: Path):
    """Replace only this generated run after all its files have been written."""
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        raise FileExistsError(f"Previous interrupted publication needs inspection: {backup}")
    had_previous = destination.exists()
    if had_previous:
        destination.rename(backup)
    try:
        staging.rename(destination)
    except BaseException:
        if had_previous:
            backup.rename(destination)
        raise
    if had_previous:
        shutil.rmtree(backup)


def run_sweep_reconciliation(
    root: str | Path,
    datasets=("GEF2017", "AusGrid2013"),
    output_root: str | Path | None = None,
    ridge: float = 1e-6,
    save_forecasts: bool = True,
    run_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Recompute both saved sweeps; never skip results based on file existence."""
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root / "DP_fhf" / "outputs" / "dp_sweep_reconciliation"
    source_root = root / "DP_fhf" / "outputs" / "01_dp_fedavg_sweep_20round"
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Reconciliation output must be separate from the source sweeps.")
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "manifest.json"
    # An interrupted rerun must not retain a previous success marker.
    completion.unlink(missing_ok=True)
    for name in ["metrics_summary.csv", "metrics_wide.csv", "per_node_metrics.csv", "matrix_diagnostics.csv"]:
        (output / name).unlink(missing_ok=True)
    for dataset in datasets:
        (output / dataset / "metrics_summary.csv").unlink(missing_ok=True)
    summary_frames, node_frames, diagnostic_frames, manifests = [], [], [], []
    code_hashes = {str(path.relative_to(root)): _sha256(path)
                   for path in [Path(__file__).resolve(), SHARED / "FHF_Cloud.py"]}
    with threadpool_limits(limits=1):
        for dataset in datasets:
            if dataset not in ("GEF2017", "AusGrid2013"):
                raise ValueError(f"Unsupported dataset: {dataset}")
            source = source_root / dataset
            original = pd.read_csv(source / "sweep_summary.csv")
            if original.run_id.duplicated().any():
                raise ValueError("Sweep summary contains duplicate run IDs.")
            if run_ids is not None:
                missing = set(run_ids) - set(original.run_id)
                if missing:
                    raise ValueError(f"Missing {dataset} runs: {sorted(missing)}")
                original = original.loc[original.run_id.isin(run_ids)]
            dataset_summaries = []
            for i, row in enumerate(original.sort_values(["clip_norm", "noise_multiplier"]).itertuples(index=False), 1):
                directory = source / row.run_id
                if directory.resolve().parent != source.resolve():
                    raise ValueError("Invalid sweep run directory.")
                print(f"[{dataset} {i}/{len(original)}] {row.run_id}: loading training residuals", flush=True)
                run = load_sweep_run(directory)
                meta = run.metadata
                for field in ["noise_multiplier", "clip_norm", "num_rounds"]:
                    if not np.isclose(float(meta[field]), float(getattr(row, field)), rtol=1e-12, atol=0):
                        raise ValueError(f"Summary/metadata mismatch: {row.run_id} {field}")
                if meta["aggregation_calls"] != meta["num_rounds"]:
                    raise ValueError("Unexpected aggregation count for FHF sweep.")
                for field in ["epsilon", "delta"]:
                    if not np.isclose(float(meta["accounting"][field]), float(getattr(row, field)), rtol=1e-12, atol=0):
                        raise ValueError(f"Summary/accounting mismatch: {row.run_id} {field}")
                matrices = estimate_matrices(run, ridge=ridge)
                predictions = reconcile_forecasts(run.base, matrices)
                per_node = evaluate_predictions(run, predictions)
                diagnostics = _diagnostics(run, matrices, predictions)
                summary = per_node.groupby("method", sort=False)[METRICS].mean().reset_index()
                base_metrics = summary.set_index("method").loc["base", METRICS].to_numpy(float)
                expected = np.array([getattr(row, metric) for metric in METRICS])
                if not np.allclose(base_metrics, expected, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Recomputed base metrics differ from the original sweep: {row.run_id}")
                context = {"dataset": dataset, "run_id": row.run_id,
                           "noise_multiplier": float(meta["noise_multiplier"]), "clip_norm": float(meta["clip_norm"]),
                           "num_rounds": int(meta["num_rounds"]),
                           "training_epsilon": float(meta["accounting"]["epsilon"]),
                           "training_delta": float(meta["accounting"]["delta"])}
                for frame in [summary, per_node, diagnostics]:
                    for key, value in reversed(list(context.items())):
                        frame.insert(0, key, value)
                for metric, base_metric in zip(METRICS, base_metrics):
                    summary[f"delta_{metric}_vs_base"] = summary[metric] - base_metric
                    summary[f"change_{metric}_pct_vs_base"] = 100 * (summary[metric] / base_metric - 1)
                input_hashes = {name: _sha256(directory / name) for name in [
                    f"{PREFIX}_base_forecasts.csv", f"{PREFIX}_training_artifacts.npz", f"{PREFIX}_privacy.json"]}
                manifest = {**context, "source_dir": str(directory), "input_sha256": input_hashes,
                            "implementation_sha256": code_hashes, "ridge": ridge, "methods": METHODS,
                            "matrix_source": "each_run_saved_training_residuals", "test_residuals_used": False,
                            "metric_scaling": "per-node training first differences, lag 1",
                            "metric_aggregation": "arithmetic mean of per-node metrics",
                            "epsilon_scope": "DP-FedAvg training only; covariance uses non-privatized saved training residuals",
                            "nodes": len(run.client_ids), "bottom_nodes": len(matrices["bottom_indices"]),
                            "train_shape": list(run.train_targets.shape), "test_shape": list(run.base.shape),
                            "base_metrics_verified": True, "coherence_verified": True,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat()}
                parent = output / dataset
                parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{row.run_id}-", dir=parent))
                try:
                    summary.to_csv(staging / "metrics_summary.csv", index=False)
                    per_node.to_csv(staging / "per_node_metrics.csv", index=False)
                    diagnostics.to_csv(staging / "matrix_diagnostics.csv", index=False)
                    np.savez_compressed(staging / "reconciliation_matrices.npz", **matrices,
                                        client_ids=run.client_ids, unique_ids=run.unique_ids, horizons=run.horizons)
                    if save_forecasts:
                        print(f"  Writing reconciled forecasts ({run.base.size:,} coordinates)", flush=True)
                        _write_forecasts(staging / "reconciled_forecasts.csv.gz", run, predictions)
                    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    _publish_directory(staging, parent / row.run_id)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
                summary_frames.append(summary)
                dataset_summaries.append(summary)
                node_frames.append(per_node)
                diagnostic_frames.append(diagnostics)
                manifests.append(manifest)
                print(f"  Verified base metrics and all 3 coherent reconciliations. MASE: "
                      + ", ".join(f"{r.method}={r.MASE:.6f}" for r in summary.itertuples()), flush=True)
            if dataset_summaries:
                pd.concat(dataset_summaries, ignore_index=True).to_csv(output / dataset / "metrics_summary.csv", index=False)
    if not summary_frames:
        raise ValueError("No sweep runs selected.")
    result = pd.concat(summary_frames, ignore_index=True)
    result.to_csv(output / "metrics_summary.csv", index=False)
    pd.concat(node_frames, ignore_index=True).to_csv(output / "per_node_metrics.csv", index=False)
    pd.concat(diagnostic_frames, ignore_index=True).to_csv(output / "matrix_diagnostics.csv", index=False)
    wide = result.pivot(index=["dataset", "run_id", "noise_multiplier", "clip_norm", "num_rounds"],
                        columns="method", values=METRICS)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide.reset_index().to_csv(output / "metrics_wide.csv", index=False)
    completion.write_text(json.dumps({"status": "complete", "runs_completed": len(manifests),
                                      "run_manifests": manifests}, indent=2) + "\n")
    print(f"Saved {len(manifests)} runs / {len(result)} method rows to {output}", flush=True)
    return result
