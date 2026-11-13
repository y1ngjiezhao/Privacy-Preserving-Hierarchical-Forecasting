"""Build the paper-aligned comparison after run_covariance_no_debias_comparison.py."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = PROJECT_ROOT
OUTPUT = ROOT / "DP_fhf/outputs/02_dp_covariance"
REPORT = OUTPUT / "no_debias_comparison"


def markdown(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = [f"{row[c]:.6g}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    summary = pd.read_csv(REPORT / "metrics_summary.csv")
    diag = pd.read_csv(REPORT / "matrix_diagnostics.csv")
    rows = []
    for dataset, folder in [("GEF2017", "fhf_dedup_gef2017"), ("AusGrid2013", "fhf_dedup_ausgrid2013")]:
        source = ROOT / folder
        raw_residuals = np.load(source / "fh1_FHF_training_artifacts.npz")["residuals"][:, :, 0].T
        n = raw_residuals.shape[1]
        # Same centred covariance, shrinkage estimator and additive ridge as FHF_Cloud.py.
        references = {
            "mint_shrinkage": LedoitWolf().fit(raw_residuals).covariance_ + 1e-6 * np.eye(n),
            "wlsv": np.diag(np.diag(np.cov(raw_residuals, rowvar=False) + 1e-6 * np.eye(n))),
        }
        ols = np.load(OUTPUT / dataset / "no_debias/FHF_non_dp_forecasts_ols_matrices.npz")["P_ols"]
        old_metrics = pd.read_csv(source / "fh1_FHF_overall_metrics.csv").set_index("method")
        for method, stored_method in [("mint_shrinkage", "mint_shrinkage"), ("wlsv", "mint_var")]:
            saved = pd.read_csv(source / f"fh1_FHF_o2mr_P_{stored_method}.csv").sort_values("recipient_matrix_index")
            P = saved.filter(regex="^sender_col_").to_numpy()
            rows.append({"dataset": dataset, "variant": "non_private", "method": method,
                         "d_W_to_nonDP": 0.0, "d_P_to_OLS": float(np.linalg.norm(P - ols)),
                         "MASE": old_metrics.loc[stored_method, "MASE"],
                         "RMSSE": old_metrics.loc[stored_method, "RMSSE"]})
        ref = summary.query("dataset == @dataset and input == 'non_dp_forecasts' and method == 'ols'").iloc[0]
        rows.append({"dataset": dataset, "variant": "reference", "method": "ols",
                     "d_W_to_nonDP": np.nan, "d_P_to_OLS": 0.0, "MASE": ref.MASE, "RMSSE": ref.RMSSE})
        for variant, subfolder in [("no_debias", "no_debias")]:
            arrays = np.load(OUTPUT / dataset / subfolder / "FHF_non_dp_forecasts_dp_covariance_matrices.npz")
            for method, reference in references.items():
                W, P = arrays[f"W_{method}"][0], arrays[f"P_{method}"][0]
                result = summary.query("dataset == @dataset and input == 'non_dp_forecasts' and variant == @variant and method == @method").iloc[0]
                rows.append({"dataset": dataset, "variant": variant, "method": method,
                             "d_W_to_nonDP": float(np.linalg.norm(W / np.trace(W) - reference / np.trace(reference))),
                             "d_P_to_OLS": float(np.linalg.norm(P - ols)), "MASE": result.MASE, "RMSSE": result.RMSSE})
        for branch in ["non_dp_forecasts", "dp_fedavg_100round"]:
            P = np.load(OUTPUT / dataset / f"no_debias/FHF_{branch}_ols_matrices.npz")["P_ols"]
            selection = (diag.dataset == dataset) & (diag.input == branch)
            diag.loc[selection, "P_frobenius_distance_to_OLS"] = diag.loc[selection, "P_relative_distance_to_OLS"] * np.linalg.norm(P)
        local = summary.query("dataset == @dataset and variant == 'no_debias'").copy()
        local.to_csv(OUTPUT / dataset / "no_debias/dp_covariance_MASE_RMSSE_summary.csv", index=False)
    paper_table = pd.DataFrame(rows)
    paper_table.to_csv(REPORT / "paper_comparison.csv", index=False)
    diag.to_csv(REPORT / "matrix_diagnostics.csv", index=False)
    paper_table.to_latex(REPORT / "paper_comparison.tex", index=False, float_format="%.6f", na_rep="--")
    manifest_path = REPORT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["code_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [DP_MODULE_DIR / name for name in [
            "dp_covariance.py", "run_dp_covariance.py", "run_covariance_no_debias_comparison.py",
            "summarize_covariance_no_debias.py"]]
    }
    manifest["scope"] = "Current no-debias configuration, non-private and OLS references only"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    note = "# Current fixed-bound covariance results\n\n"
    note += "Configuration: bounds ±0.0005, epsilon=1, seed=42, ridge=1e-6, without noise variance correction.\n\n"
    note += "paper_comparison.csv contains the current manuscript results for non_private, reference (OLS), and no_debias. The old corrected data were removed.\n\n"
    note += markdown(paper_table, ["dataset", "variant", "method", "MASE", "RMSSE", "d_P_to_OLS"])
    note += "\n\nReproduction: run PPHF_modules/dp_fhf/run_covariance_no_debias_comparison.py, then PPHF_modules/dp_fhf/summarize_covariance_no_debias.py. No old result files are needed, and the manuscript is not modified.\n"
    (REPORT / "REPORT_zh.md").write_text(note)
    print(paper_table.to_string(index=False))


if __name__ == "__main__":
    main()
