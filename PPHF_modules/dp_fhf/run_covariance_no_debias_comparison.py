"""Recompute current fixed-bound no-debias results and OLS comparisons.
No archived corrected outputs or manuscript snapshots are required.
"""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.covariance import LedoitWolf

from PPHF_modules.dp_fhf.dp_covariance import DPCovarianceConfig, privatize_residuals_by_client
from PPHF_modules.dp_fhf.run_dp_covariance import _matrix_by_node


ROOT = PROJECT_ROOT
OUTPUT = ROOT / "DP_fhf/outputs/02_dp_covariance"
REPORT = OUTPUT / "no_debias_comparison"
RIDGE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    summaries, diagnostics, checks, per_nodes = [], [], [], []
    manifest = {
        "python": sys.version, "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "settings": {"epsilon": 1, "mechanism": "laplace", "lower": -0.0005,
                     "upper": 0.0005, "seed": 42, "ridge": RIDGE},
        "metric_aggregation": "arithmetic mean across node/horizon rows; original runner definitions",
        "runs": [],
    }
    for dataset, folder in [("GEF2017", "fhf_dedup_gef2017"),
                            ("AusGrid2013", "fhf_dedup_ausgrid2013")]:
        for branch in ["non_dp_forecasts", "dp_fedavg_100round"]:
            label = f"FHF_{branch}"
            original = OUTPUT / dataset
            output = original / "no_debias"
            if branch == "non_dp_forecasts":
                source = ROOT / folder / "fh1_FHF_forecasts.csv"
                residuals_path = ROOT / folder / "fh1_FHF_training_artifacts.npz"
            else:
                training = ROOT / "DP_fhf/outputs/01_dp_fedavg_100round" / dataset
                source = training / "fh1_FHF_dp_fedavg_base_forecasts.csv"
                residuals_path = training / "fh1_FHF_dp_fedavg_training_artifacts.npz"
            run = {"dataset": dataset, "input": branch,
                   "source_forecasts": str(source), "source_sha256": sha256(source),
                   "residuals": str(residuals_path), "residuals_sha256": sha256(residuals_path),
                   "commands": []}
            outputs = {"no_debias": output}
            for variant, target in outputs.items():
                target.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(ROOT / "PPHF_modules/dp_fhf/run_dp_covariance.py"),
                           "--forecasts", str(source), "--residuals-npz", str(residuals_path),
                           "--epsilon", "1", "--mechanism", "laplace", "--delta", "1e-5",
                           "--lower", "-0.0005", "--upper", "0.0005", "--seed", "42",
                           "--ridge", str(RIDGE), "--output-dir", str(target),
                           "--output-label", label]
                if variant == "no_debias":
                    command.append("--no-debias")
                run["commands"].append(command)
                print(f"RUN {dataset} {branch} {variant}", flush=True)
                with (target / f"{label}_run.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
                                                    "OMP_NUM_THREADS": "1"})

            with np.load(residuals_path) as archive:
                ids = archive["client_ids"].astype(int).tolist()
                residuals = archive["residuals"].astype(float)
                targets = archive["train_targets"].astype(float)
            private, metadata = privatize_residuals_by_client(
                dict(zip(ids, residuals)), DPCovarianceConfig(seed=42, lower=-0.0005, upper=0.0005))
            cube = np.stack([private[i] for i in ids])
            run["private_residuals_sha256"] = hashlib.sha256(cube.tobytes()).hexdigest()
            run["clipping_fraction"] = sum(m["clipped_coordinate_count"] for m in metadata.values()) / residuals.size
            table = pd.read_csv(source, usecols=["node_id", "unique_id", "ds", "horizon_index", "y_true", "base"])
            table["ds"] = pd.to_datetime(table["ds"])
            node_map = table[["node_id", "unique_id"]].drop_duplicates().sort_values("node_id")
            assert node_map.node_id.tolist() == ids
            names = node_map.unique_id.tolist()
            horizons = sorted(table.horizon_index.unique())
            differences = np.diff(targets, axis=1)
            mase_scale = np.mean(np.abs(differences), axis=(1, 2))
            rmsse_scale = np.mean(differences ** 2, axis=(1, 2))
            arrays, metric_tables = {}, {}
            for variant, target in outputs.items():
                with np.load(target / f"{label}_dp_covariance_matrices.npz") as archive:
                    arrays[variant] = {k: archive[k] for k in archive.files}
                metric_tables[variant] = pd.read_csv(target / f"{label}_dp_covariance_metrics.csv")
            S = arrays["no_debias"]["S"]
            pols = S @ np.linalg.pinv(S)
            np.savez_compressed(output / f"{label}_ols_matrices.npz", S=S, P_ols=pols)
            check = {"dataset": dataset, "input": branch,
                     "configuration": "no_debias", "archived_results_required": False}

            baseline_rows = []
            for position, h in enumerate(horizons):
                base, times = _matrix_by_node(table, ids, "base", h)
                truth, truth_times = _matrix_by_node(table, ids, "y_true", h)
                assert times.equals(truth_times)
                for method, prediction in [("base", base), ("ols", pols @ base)]:
                    error = truth - prediction
                    for n, client_id in enumerate(ids):
                        baseline_rows.append({"node_id": client_id, "unique_id": names[n],
                            "horizon_index": h, "method": method,
                            "MASE": np.mean(np.abs(error[n])) / mase_scale[n],
                            "RMSSE": np.sqrt(np.mean(error[n] ** 2) / rmsse_scale[n])})
                x = cube[:, :, position].T
                raw_cov = np.cov(x, rowvar=False, ddof=1)
                shrinkage = LedoitWolf().fit(x)
                noise_variance = np.array([metadata[i]["mean_noise_variance_by_horizon"][position] for i in ids])
                for variant in outputs:
                    for method in ["mint_shrinkage", "wlsv"]:
                        W = arrays[variant][f"W_{method}"][position]
                        P = arrays[variant][f"P_{method}"][position]
                        # WLSv takes the diagonal AFTER projection of the full sample covariance.
                        before_psd = (shrinkage.covariance_ if method == "mint_shrinkage" else raw_cov).copy()
                        eig = np.linalg.eigvalsh(before_psd)
                        mean_diag = np.trace(W) / len(ids)
                        prediction = P @ base
                        ols_prediction = pols @ base
                        row = {"dataset": dataset, "input": branch, "variant": variant,
                            "method": method, "horizon_index": int(h), "nodes": len(ids),
                            "training_times": x.shape[0], "test_times": base.shape[1],
                            "clipping_fraction": run["clipping_fraction"],
                            "noise_variance": float(noise_variance[0]),
                            "ledoit_wolf_shrinkage": float(shrinkage.shrinkage_),
                            "pre_psd_min_eigenvalue": float(eig.min()),
                            "pre_psd_max_eigenvalue": float(eig.max()),
                            "eigenvalues_below_floor": int(np.sum(eig < RIDGE)),
                            "W_condition_number": float(np.linalg.cond(W)),
                            "W_relative_distance_to_scalar_I": float(np.linalg.norm(W - mean_diag * np.eye(len(ids))) / np.linalg.norm(mean_diag * np.eye(len(ids)))),
                            "P_relative_distance_to_OLS": float(np.linalg.norm(P - pols) / np.linalg.norm(pols)),
                            "forecast_relative_distance_to_OLS": float(np.linalg.norm(prediction - ols_prediction) / np.linalg.norm(ols_prediction)),
                            "PS_minus_S_max_abs": float(np.max(np.abs(P @ S - S))),
                            "coherence_max_abs": float(np.max(np.abs(prediction - pols @ prediction)))}
                        assert row["PS_minus_S_max_abs"] < 1e-8
                        assert np.isfinite(prediction).all()
                        calculated = np.column_stack([np.mean(np.abs(truth - prediction), axis=1) / mase_scale,
                                                     np.sqrt(np.mean((truth - prediction) ** 2, axis=1) / rmsse_scale)])
                        saved = metric_tables[variant].query("method == @method and horizon_index == @h").set_index("node_id").loc[ids, ["MASE", "RMSSE"]].values
                        assert np.allclose(calculated, saved, rtol=1e-10, atol=1e-10)
                        diagnostics.append(row)
            pd.DataFrame(baseline_rows).to_csv(output / f"{label}_base_ols_metrics.csv", index=False)
            metric_tables["reference"] = pd.DataFrame(baseline_rows)
            for variant, frame in metric_tables.items():
                frame = frame.copy()
                frame["dataset"], frame["input"], frame["variant"] = dataset, branch, variant
                per_nodes.append(frame)
                for method, group in frame.groupby("method"):
                    summaries.append({"dataset": dataset, "input": branch, "variant": variant,
                                      "method": method, "MASE": group.MASE.mean(), "RMSSE": group.RMSSE.mean()})
            checks.append(check)
            manifest["runs"].append(run)
            pd.DataFrame(summaries).to_csv(REPORT / "metrics_summary.csv", index=False)
            pd.DataFrame(diagnostics).to_csv(REPORT / "matrix_diagnostics.csv", index=False)
            pd.concat(per_nodes, ignore_index=True).to_csv(REPORT / "per_node_metrics.csv", index=False)
            (REPORT / "verification.json").write_text(json.dumps(checks, indent=2))
            (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"DONE {dataset} {branch}: current no-debias outputs and OLS metrics verified", flush=True)


if __name__ == "__main__":
    main()
