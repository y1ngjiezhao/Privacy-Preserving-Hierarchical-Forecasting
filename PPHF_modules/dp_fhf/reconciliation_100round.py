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
            "# 两个 100-round 结果的 reconciliation\n\n"
            "入口：`DP_fhf/notebooks/01_DPFedAvg_100Round_Reconciliation.ipynb`。\n\n"
            "复用原 FHF Cloud：mints=mint_shrinkage、wlsv=mint_var、ols=mint_ols；ridge=1e-6。"
            "每个数据集使用自身 100-round 模型保存的训练残差估计矩阵，测试真值仅用于评分。\n\n"
            "MASE/RMSSE 使用训练目标 lag-1 差分分母，先逐节点评分再取算术平均。"
            "`change_*_pct_vs_base` 负数表示相比同组 base 改善。"
            "training_epsilon/delta 仅记录原训练的隐私参数，不代表本次未额外私有化的残差估计的整体保证。\n\n"
            "- metrics_summary.csv：2 数据集 × 4 方法，共 8 行指标和相对 base 变化。\n"
            "- metrics_wide.csv：四种方法并排对比。\n"
            "- per_node_metrics.csv：逐序列指标。\n"
            "- matrix_diagnostics.csv：PS=S、P²=P、预测层级一致性。\n"
            "- 每个数据集目录：上述单独指标、reconciliation_matrices.npz（S/G/P/W 及节点顺序）、"
            "reconciled_forecasts.csv.gz（真值及四种预测）、manifest.json（输入与代码 SHA256）。\n\n"
            "所有结果统一位于 `DP_fhf/outputs/dp_100round_reconciliation/`。"
            "每次运行重新计算，完整批次验证成功后替换该目录；输入文件不变。\n", encoding="utf-8")
        recon._publish_directory(staging, output)
        print(f"Saved {len(manifests)} runs / {len(result)} method rows to {output}", flush=True)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
