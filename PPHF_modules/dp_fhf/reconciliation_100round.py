"""Apply the existing FHF reconciliation methods to both saved 100-round runs."""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from PPHF_modules.dp_fhf import sweep_reconciliation as recon
from PPHF_modules.dp_fhf.notebook_runtime import _evaluate_forecasts, _scales_from_artifact


def run_100round_reconciliation(
    root: str | Path,
    datasets=("GEF2017", "AusGrid2013"),
    output_root: str | Path | None = None,
    ridge: float = 1e-6,
    save_forecasts: bool = True,
) -> pd.DataFrame:
    """Recompute and atomically publish a complete batch without retraining."""
    root = Path(root).resolve()
    source_root = root / "DP_fhf" / "outputs" / "01_dp_fedavg_100round"
    output = Path(output_root).resolve() if output_root else root / "DP_fhf" / "outputs" / "dp_100round_reconciliation"
    if output == source_root or source_root in output.parents or output in source_root.parents:
        raise ValueError("Output must be separate from the original 100-round results.")
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= {"GEF2017", "AusGrid2013"}:
        raise ValueError("Select unique datasets from GEF2017 and AusGrid2013.")
    code_paths = [Path(__file__), Path(recon.__file__), recon.SHARED / "FHF_Cloud.py",
                  root / "PPHF_modules" / "dp_fhf" / "notebook_runtime.py"]
    code_hashes = {str(path.resolve().relative_to(root)): recon._sha256(path) for path in code_paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    summaries, node_frames, diagnostic_frames, manifests = [], [], [], []
    try:
        with threadpool_limits(limits=1):
            for dataset in datasets:
                source = source_root / dataset
                print(f"[{dataset}] Loading saved 100-round model forecasts and training residuals", flush=True)
                filenames = [f"{recon.PREFIX}_base_forecasts.csv", f"{recon.PREFIX}_training_artifacts.npz",
                             f"{recon.PREFIX}_privacy.json"]
                input_hashes = {name: recon._sha256(source / name) for name in filenames}
                run = recon.load_sweep_run(source)
                meta = run.metadata
                if meta["num_rounds"] != 100 or meta["aggregation_calls"] != 100:
                    raise ValueError(f"{dataset} is not a completed 100-round FHF run.")
                if meta["accounting"]["aggregation_calls"] != 100:
                    raise ValueError(f"{dataset} accounting does not cover 100 aggregations.")
                matrices = recon.estimate_matrices(run, ridge=ridge)
                predictions = recon.reconcile_forecasts(run.base, matrices)
                per_node = recon.evaluate_predictions(run, predictions)
                diagnostics = recon._diagnostics(run, matrices, predictions)
                summary = per_node.groupby("method", sort=False)[recon.METRICS].mean().reset_index()
                base_metrics = summary.set_index("method").loc["base", recon.METRICS].to_numpy(float)
                # Cross-check against the metric implementation used by the old sweep.
                _, original_metrics = _evaluate_forecasts(
                    pd.read_csv(source / filenames[0]),
                    _scales_from_artifact(source / filenames[1]), "100round_base")
                if not np.allclose(base_metrics, [original_metrics[m] for m in recon.METRICS], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Base metrics do not match the original evaluation convention: {dataset}")
                context = {"dataset": dataset, "run_id": "fixed_100round", "num_rounds": 100,
                           "noise_multiplier": float(meta["noise_multiplier"]), "clip_norm": float(meta["clip_norm"]),
                           "training_epsilon": float(meta["accounting"]["epsilon"]),
                           "training_delta": float(meta["accounting"]["delta"])}
                for frame in [summary, per_node, diagnostics]:
                    for key, value in reversed(list(context.items())):
                        frame.insert(0, key, value)
                for metric, base_value in zip(recon.METRICS, base_metrics):
                    summary[f"delta_{metric}_vs_base"] = summary[metric] - base_value
                    summary[f"change_{metric}_pct_vs_base"] = 100 * (summary[metric] / base_value - 1)
                destination = staging / dataset
                destination.mkdir()
                summary.to_csv(destination / "metrics_summary.csv", index=False)
                per_node.to_csv(destination / "per_node_metrics.csv", index=False)
                diagnostics.to_csv(destination / "matrix_diagnostics.csv", index=False)
                np.savez_compressed(destination / "reconciliation_matrices.npz", **matrices,
                                    client_ids=run.client_ids, unique_ids=run.unique_ids, horizons=run.horizons)
                if save_forecasts:
                    print(f"  Writing base/mints/wlsv/ols forecasts: {run.base.size:,} coordinates", flush=True)
                    recon._write_forecasts(destination / "reconciled_forecasts.csv.gz", run, predictions)
                if any(recon._sha256(source / name) != digest for name, digest in input_hashes.items()):
                    raise ValueError(f"Source artifacts changed during calculation: {dataset}")
                manifest = {**context, "source_dir": str(source), "input_sha256": input_hashes,
                            "implementation_sha256": code_hashes, "ridge": ridge, "methods": recon.METHODS,
                            "matrix_source": "each_run_saved_training_residuals", "test_residuals_used": False,
                            "metric_scaling": "per-node training first differences, lag 1",
                            "metric_aggregation": "arithmetic mean of per-node metrics",
                            "epsilon_scope": "DP-FedAvg training only; covariance uses non-privatized saved training residuals",
                            "nodes": len(run.client_ids), "bottom_nodes": len(matrices["bottom_indices"]),
                            "train_shape": list(run.train_targets.shape), "test_shape": list(run.base.shape),
                            "base_metrics_verified": True, "coherence_verified": True, "source_files_unchanged": True,
                            "completed_at_utc": datetime.now(timezone.utc).isoformat()}
                (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                summaries.append(summary)
                node_frames.append(per_node)
                diagnostic_frames.append(diagnostics)
                manifests.append(manifest)
                print("  Verified. MASE: " + ", ".join(f"{r.method}={r.MASE:.6f}" for r in summary.itertuples()), flush=True)
        result = pd.concat(summaries, ignore_index=True)
        result.to_csv(staging / "metrics_summary.csv", index=False)
        pd.concat(node_frames, ignore_index=True).to_csv(staging / "per_node_metrics.csv", index=False)
        pd.concat(diagnostic_frames, ignore_index=True).to_csv(staging / "matrix_diagnostics.csv", index=False)
        wide = result.pivot(index=["dataset", "run_id", "num_rounds", "noise_multiplier", "clip_norm"],
                            columns="method", values=recon.METRICS)
        wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
        wide.reset_index().to_csv(staging / "metrics_wide.csv", index=False)
        (staging / "manifest.json").write_text(json.dumps(
            {"status": "complete", "runs_completed": len(manifests), "run_manifests": manifests}, indent=2) + "\n")
        (staging / "README_zh.md").write_text(
            "# Reconciliation of two 100-round runs\n\n"
            "Entry point: `DP_fhf/notebooks/01_DPFedAvg_100Round_Reconciliation.ipynb`.\n\n"
            "Reuse the original FHF Cloud: mints=mint_shrinkage, wlsv=mint_var, ols=mint_ols; ridge=1e-6. "
            "Estimate matrices using training residuals saved by the corresponding 100-round model for each dataset. Test observations are used only for scoring.\n\n"
            "MASE/RMSSE use lag-1 training-target difference denominators, computed per node and then averaged arithmetically. "
            "Negative `change_*_pct_vs_base` values indicate improvement over the matching base. "
            "training_epsilon/delta record only the original training privacy parameters, not an end-to-end guarantee for this residual estimation without additional privatization.\n\n"
            "- metrics_summary.csv: 2 datasets × 4 methods, giving 8 rows of metrics and changes relative to base.\n"
            "- metrics_wide.csv: side-by-side comparison of the four methods.\n"
            "- per_node_metrics.csv: per-series metrics.\n"
            "- matrix_diagnostics.csv: PS=S, P²=P, and hierarchical consistency of forecasts.\n"
            "- Each dataset directory: the metrics above, reconciliation_matrices.npz (S/G/P/W and node order), "
            "reconciled_forecasts.csv.gz (observations and four forecasts), and manifest.json (input and code SHA256 hashes).\n\n"
            "All results are located in `DP_fhf/outputs/dp_100round_reconciliation/`. "
            "Recompute on every run and replace this directory only after the entire batch passes validation. Input files remain unchanged.\n", encoding="utf-8")
        recon._publish_directory(staging, output)
        print(f"Saved {len(manifests)} runs / {len(result)} method rows to {output}", flush=True)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
