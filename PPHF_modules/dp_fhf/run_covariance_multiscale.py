"""Client-specific absolute-residual quantile sweep, without variance debiasing.

Calibration is non-private and uses the same saved training residuals; outputs
are utility diagnostics, not an unconditional DP guarantee for calibration data.
Retention means the fraction inside [-B_i,h, B_i,h], not a fraction of rows kept.
"""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from PPHF_modules.dp_fhf.dp_covariance import DPCovarianceConfig, build_private_reconciliation_matrices
from PPHF_modules.dp_fhf.run_dp_covariance import _build_summing_matrix, _matrix_by_node

ROOT = PROJECT_ROOT
OUTPUT = ROOT / "DP_fhf/outputs/02_dp_covariance/multiscale_no_debias"
RETENTIONS = [0.995, 0.75, 0.50, 0.25, 0.005]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(prediction, truth, scale):
    error = truth - prediction
    return np.column_stack((np.mean(np.abs(error), axis=1) / scale[:, 0],
                            np.sqrt(np.mean(error ** 2, axis=1) / scale[:, 1])))


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["GEF2017", "AusGrid2013"], default=["GEF2017", "AusGrid2013"])
    parser.add_argument("--retentions", nargs="+", type=float, default=RETENTIONS,
                        help="Fractions left unclipped, e.g. 0.005 for 0.5 percent.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if any(not 0 < q <= 1 for q in args.retentions) or len(set(args.retentions)) != len(args.retentions):
        parser.error("Retentions must be distinct fractions in (0, 1].")
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, node_rows, bound_rows, diagnostic_rows, manifest = [], [], [], [], []
    for dataset in args.datasets:
        folder = ROOT / ("fhf_dedup_gef2017" if dataset == "GEF2017" else "fhf_dedup_ausgrid2013")
        for branch in ["non_dp_forecasts", "dp_fedavg_100round"]:
            if branch == "non_dp_forecasts":
                forecasts = folder / "fh1_FHF_forecasts.csv"
                artifact_path = folder / "fh1_FHF_training_artifacts.npz"
            else:
                training = ROOT / "DP_fhf/outputs/01_dp_fedavg_100round" / dataset
                forecasts = training / "fh1_FHF_dp_fedavg_base_forecasts.csv"
                artifact_path = training / "fh1_FHF_dp_fedavg_training_artifacts.npz"
            print(f"LOAD {dataset} {branch}", flush=True)
            with np.load(artifact_path) as a:
                ids = a["client_ids"].astype(int).tolist()
                residuals = a["residuals"].astype(float)
                targets = a["train_targets"].astype(float)
                assert np.array_equal(residuals, targets - a["train_predictions"])
            table = pd.read_csv(forecasts, usecols=["node_id", "unique_id", "ds", "horizon_index", "base", "y_true"])
            table["ds"] = pd.to_datetime(table.ds)
            mapping = table[["node_id", "unique_id"]].drop_duplicates().sort_values("node_id")
            assert ids == mapping.node_id.tolist()
            names = mapping.unique_id.tolist()
            S = _build_summing_matrix(names)
            OLS = S @ np.linalg.pinv(S)
            horizons = sorted(table.horizon_index.unique())
            assert residuals.shape[2] == len(horizons)
            delta = np.diff(targets, axis=1)
            scale = np.column_stack((np.mean(np.abs(delta), axis=(1, 2)), np.mean(delta ** 2, axis=(1, 2))))
            assert np.all(scale > 0)
            evaluation = []
            for h in horizons:
                base, times = _matrix_by_node(table, ids, "base", h)
                truth, truth_times = _matrix_by_node(table, ids, "y_true", h)
                assert times.equals(truth_times)
                evaluation.append((base, truth))
            source_info = {"dataset": dataset, "input": branch, "forecasts": str(forecasts),
                           "forecasts_sha256": fingerprint(forecasts), "residuals": str(artifact_path),
                           "residuals_sha256": fingerprint(artifact_path)}
            manifest.append(source_info)
            for q in args.retentions:
                label = f"retain_{100*q:g}pct".replace(".", "p")
                out = OUTPUT / dataset / branch / label
                out.mkdir(parents=True, exist_ok=True)
                B = np.quantile(np.abs(residuals), q, axis=1, method="linear")
                assert B.shape == (len(ids), len(horizons))
                assert np.all(np.isfinite(B)) and np.all(B > 0), "Zero quantile needs explicit calibration policy"
                config = DPCovarianceConfig(epsilon=1, lower=dict(zip(ids, -B)), upper=dict(zip(ids, B)),
                                            seed=42, ridge=1e-6, debias_noise_variance=False)
                result = build_private_reconciliation_matrices(S, ids, dict(zip(ids, residuals)), config)
                np.savez_compressed(out / "matrices.npz", S=S, P_ols=OLS,
                                    P_mint_shrinkage=result["Ps"]["mint_shrinkage"], P_wlsv=result["Ps"]["mint_var"],
                                    W_mint_shrinkage=result["Ws_by_method"]["mint_shrinkage"], W_wlsv=result["Ws_by_method"]["mint_var"])
                retained = np.mean(np.abs(residuals) <= B[:, None, :], axis=1)
                for j, client_id in enumerate(ids):
                    for hi, h in enumerate(horizons):
                        bound_rows.append({"dataset": dataset, "input": branch, "retention_target_pct": 100*q,
                            "node_id": client_id, "unique_id": names[j], "horizon_index": int(h),
                            "B": B[j, hi], "lower": -B[j, hi], "upper": B[j, hi],
                            "retained_fraction": retained[j, hi], "clipped_fraction": 1-retained[j, hi],
                            "laplace_scale": 2*B[j, hi], "noise_variance": 8*B[j, hi]**2})
                local_bounds = pd.DataFrame(bound_rows).query("dataset == @dataset and input == @branch and retention_target_pct == @q*100")
                local_bounds.to_csv(out / "client_bounds.csv", index=False)
                local_rows = []
                for hi, (base, truth) in enumerate(evaluation):
                    noise_variances = 8 * B[:, hi] ** 2
                    inverse_sqrt = np.diag(1 / np.sqrt(noise_variances))
                    P_noise = S @ np.linalg.pinv(inverse_sqrt @ S) @ inverse_sqrt
                    private = np.column_stack([result["private_residuals"][i][:, hi] for i in ids])
                    lw = LedoitWolf().fit(private)
                    projections = {"base": np.eye(len(ids)), "ols": OLS, "noise_wls": P_noise,
                                   "mint_shrinkage": result["Ps"]["mint_shrinkage"][hi], "wlsv": result["Ps"]["mint_var"][hi]}
                    for method, P in projections.items():
                        prediction = P @ base
                        values = metrics(prediction, truth, scale)
                        assert np.isfinite(values).all()
                        if method != "base":
                            assert np.max(np.abs(P @ S - S)) < 1e-6, (dataset, q, method)
                        for j, client_id in enumerate(ids):
                            item = {"dataset": dataset, "input": branch, "retention_target_pct": 100*q,
                                    "method": method, "node_id": client_id, "unique_id": names[j],
                                    "horizon_index": int(horizons[hi]), "MASE": values[j, 0], "RMSSE": values[j, 1]}
                            local_rows.append(item)
                        if method in ["mint_shrinkage", "wlsv"]:
                            W = result["Ws_by_method"]["mint_shrinkage" if method == "mint_shrinkage" else "mint_var"][hi]
                            unprojected = lw.covariance_ if method == "mint_shrinkage" else np.cov(private, rowvar=False)
                            diagnostic_rows.append({"dataset": dataset, "input": branch, "retention_target_pct": 100*q,
                                "method": method, "horizon_index": int(horizons[hi]),
                                "d_P_to_OLS": np.linalg.norm(P-OLS), "relative_d_P_to_OLS": np.linalg.norm(P-OLS)/np.linalg.norm(OLS),
                                "d_P_to_noise_WLS": np.linalg.norm(P-P_noise),
                                "relative_d_P_to_noise_WLS": np.linalg.norm(P-P_noise)/np.linalg.norm(P_noise),
                                "shrinkage": lw.shrinkage_, "W_condition": np.linalg.cond(W),
                                "pre_psd_min_eigenvalue": np.linalg.eigvalsh(unprojected).min(),
                                "eigenvalues_below_floor": int(np.sum(np.linalg.eigvalsh(unprojected)<1e-6)),
                                "PS_minus_S_max_abs": np.max(np.abs(P@S-S)),
                                "coherence_relative": np.linalg.norm(prediction-OLS@prediction)/np.linalg.norm(prediction),
                                "noise_variance_ratio_max_min": noise_variances.max()/noise_variances.min()})
                local = pd.DataFrame(local_rows)
                local.to_csv(out / "per_node_metrics.csv", index=False)
                node_rows.extend(local_rows)
                for method, frame in local.groupby("method"):
                    rows.append({"dataset": dataset, "input": branch, "retention_target_pct": 100*q,
                                 "method": method, "MASE": frame.MASE.mean(), "RMSSE": frame.RMSSE.mean(),
                                 "B_min": B.min(), "B_max": B.max(), "retained_fraction": retained.mean()})
                privacy = {**source_info, "retention_target": q, "retained_fraction": float(retained.mean()),
                    "bounds_rule": "B_i,h = linear quantile_q(abs(saved original-scale training residuals_i,h))",
                    "epsilon_conditional_on_fixed_bounds": 1, "mechanism": "laplace", "debias_noise_variance": False,
                    "ridge": 1e-6, "seed": 42, "calibration_private": False,
                    "formal_dp_for_joint_calibration_and_release": False,
                    "warning": "Utility experiment: bounds derived from the protected training residuals without DP; public fixed seed is for reproducibility only. No unconditional DP claim for these experimental releases.",
                    "all_rows_retained": True, "private_release_used_for_both_methods": True}
                (out / "privacy_and_calibration.json").write_text(json.dumps(privacy, indent=2))
                pd.DataFrame(rows).to_csv(OUTPUT / "metrics_summary.csv", index=False)
                pd.DataFrame(bound_rows).to_csv(OUTPUT / "client_bounds.csv", index=False)
                pd.DataFrame(diagnostic_rows).to_csv(OUTPUT / "matrix_diagnostics.csv", index=False)
                pd.DataFrame(node_rows).to_csv(OUTPUT / "per_node_metrics.csv", index=False)
                print(f"DONE {dataset} {branch} retained={100*q:g}% observed={100*retained.mean():.4f}% B=[{B.min():.6g}, {B.max():.6g}]", flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps({"sources": manifest, "retentions": args.retentions,
        "python": sys.version, "script_sha256": fingerprint(Path(__file__)),
        "mechanism_sha256": fingerprint(ROOT / "PPHF_modules/dp_fhf/dp_covariance.py")}, indent=2))


if __name__ == "__main__":
    main()
