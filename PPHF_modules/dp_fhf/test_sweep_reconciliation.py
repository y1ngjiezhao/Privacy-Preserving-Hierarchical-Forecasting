"""Numerical checks for reconciling saved sweep forecasts without retraining."""

from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn


ROOT = PROJECT_ROOT
for directory in (ROOT, FHF_MODULE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from PPHF_modules.dp_fhf.notebook_runtime import _evaluate_forecasts, _scales_from_artifact
from PPHF_modules.dp_fhf.sweep_reconciliation import (
    estimate_matrices,
    evaluate_predictions,
    load_sweep_run,
    reconcile_forecasts,
)
from PPHF_modules.fhf.FHF_Cloud import Cloud


class SweepReconciliationTests(unittest.TestCase):
    """Small unbalanced hierarchy, shuffled rows, and two forecast horizons."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "nm_0p3__clip_1"
        self.run_dir.mkdir()
        self.forecast_path = self.run_dir / "fh1_FHF_dp_fedavg_base_forecasts.csv"
        self.artifact_path = self.run_dir / "fh1_FHF_dp_fedavg_training_artifacts.npz"
        self.client_ids = np.array([100, 12, 5, 31, 7])
        self.unique_ids = np.array(
            ["Total", "Total/A", "Total/B", "Total/B/x", "Total/B/y"]
        )
        self.horizons = np.array([0, 1])
        self.times = pd.date_range("2025-01-01", periods=7, freq="h")
        self.S = np.array(
            [[1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )
        self.bottom_indices = np.array([1, 3, 4])
        rng = np.random.default_rng(930)
        train_bottom = np.cumsum(rng.normal(size=(3, 40, 2)), axis=1) + 20
        self.train_targets = np.einsum("nb,bth->nth", self.S, train_bottom)
        mixing = np.array(
            [[2, .4, 0, 0, 0], [0, 1, .2, 0, 0], [0, 0, 3, .7, 0],
             [0, 0, 0, .5, .1], [.1, 0, 0, 0, 1.4]]
        )
        self.residuals = np.einsum(
            "ij,jth->ith", mixing, rng.normal(size=(5, 40, 2))
        )
        test_bottom = rng.normal(20, 2, size=(3, 7, 2))
        self.truth = np.einsum("nb,bth->nth", self.S, test_bottom)
        self.base = self.truth + rng.normal(size=(5, 7, 2))
        self.artifacts = {
            "client_ids": self.client_ids,
            "unique_ids": self.unique_ids,
            "train_targets": self.train_targets,
            "train_predictions": self.train_targets - self.residuals,
            "residuals": self.residuals,
        }
        np.savez(self.artifact_path, **self.artifacts)
        records = []
        for node, (client_id, unique_id) in enumerate(zip(self.client_ids, self.unique_ids)):
            for time, ds in enumerate(self.times):
                for horizon, horizon_id in enumerate(self.horizons):
                    records.append({
                        "node_id": client_id,
                        "unique_id": unique_id,
                        "ds": ds,
                        "horizon_index": horizon_id,
                        "y_true": self.truth[node, time, horizon],
                        "base": self.base[node, time, horizon],
                    })
        self.forecasts = pd.DataFrame(records).sample(frac=1, random_state=7)
        self.forecasts.to_csv(self.forecast_path, index=False)
        (self.run_dir / "fh1_FHF_dp_fedavg_privacy.json").write_text(
            json.dumps({"framework": "fhf", "hierarchy_mode": "dedup", "enabled": True,
                        "forecast_horizon": 2, "num_rounds": 20,
                        "noise_multiplier": .3, "clip_norm": 1.0,
                        "aggregation_calls": 20, "training_seed": 42, "dp_seed": 42,
                        "accounting": {"epsilon": 184.9405907487698, "delta": 1e-5}}),
            encoding="utf-8",
        )

    def test_load_aligns_shuffled_rows_using_artifact_node_order(self):
        run = load_sweep_run(self.run_dir)
        np.testing.assert_array_equal(run.client_ids, self.client_ids)
        np.testing.assert_array_equal(run.unique_ids, self.unique_ids)
        np.testing.assert_array_equal(run.horizons, self.horizons)
        pd.testing.assert_index_equal(pd.DatetimeIndex(run.times), self.times, check_names=False)
        np.testing.assert_allclose(run.base, self.base, rtol=1e-14)
        np.testing.assert_allclose(run.truth, self.truth, rtol=1e-14)
        np.testing.assert_array_equal(run.residuals, self.residuals)
        np.testing.assert_array_equal(run.train_targets, self.train_targets)

    def test_matrices_match_existing_cloud_implementation(self):
        run = load_sweep_run(self.run_dir)
        actual = estimate_matrices(run)
        cloud = Cloud(nn.Linear(1, 1), self.client_ids.tolist())
        cloud.build_hierarchy_S(dict(zip(self.client_ids, self.unique_ids)), reject_duplicate_series=True)
        residuals = {int(cid): self.residuals[i] for i, cid in enumerate(self.client_ids)}
        reference_P, reference_G = cloud.compute_reconciliation_matrices(
            ["mint_shrinkage", "mint_var", "mint_ols"], residuals, ridge=1e-6
        )
        np.testing.assert_array_equal(actual["S"], self.S)
        np.testing.assert_array_equal(actual["bottom_indices"], self.bottom_indices)
        for label, reference in [("mints", "mint_shrinkage"), ("wlsv", "mint_var"), ("ols", "mint_ols")]:
            with self.subTest(method=label):
                np.testing.assert_allclose(actual[f"P_{label}"], reference_P[reference], rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(actual[f"G_{label}"], reference_G[reference], rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(actual[f"W_{label}"], cloud.Ws_by_method[reference], rtol=1e-10, atol=1e-11)

    def test_projections_preserve_coherent_forecasts_and_make_outputs_coherent(self):
        run = load_sweep_run(self.run_dir)
        matrices = estimate_matrices(run)
        predictions = reconcile_forecasts(run.base, matrices)
        coherent_predictions = reconcile_forecasts(run.truth, matrices)
        self.assertEqual(set(predictions), {"base", "mints", "wlsv", "ols"})
        np.testing.assert_array_equal(predictions["base"], run.base)
        for method in ("mints", "wlsv", "ols"):
            with self.subTest(method=method):
                P = matrices[f"P_{method}"]
                self.assertEqual(P.shape, (2, 5, 5))
                np.testing.assert_allclose(P @ self.S, np.broadcast_to(self.S, (2, 5, 3)), atol=1e-10)
                np.testing.assert_allclose(P @ P, P, atol=1e-10)
                expected = np.einsum("hij,jth->ith", P, run.base)
                np.testing.assert_allclose(predictions[method], expected, atol=1e-10)
                from_bottom = np.einsum("nb,bth->nth", self.S, predictions[method][self.bottom_indices])
                np.testing.assert_allclose(predictions[method], from_bottom, atol=1e-10)
                np.testing.assert_allclose(coherent_predictions[method], run.truth, atol=1e-10)

    def test_test_truth_cannot_change_matrices_or_reconciled_predictions(self):
        run = load_sweep_run(self.run_dir)
        matrices = estimate_matrices(run)
        predictions = reconcile_forecasts(run.base, matrices)
        altered = self.forecasts.copy()
        altered["y_true"] = np.arange(len(altered)) ** 2 + 10000
        altered.to_csv(self.forecast_path, index=False)
        changed_run = load_sweep_run(self.run_dir)
        changed_matrices = estimate_matrices(changed_run)
        changed_predictions = reconcile_forecasts(changed_run.base, changed_matrices)
        for key in matrices:
            with self.subTest(matrix=key):
                np.testing.assert_array_equal(changed_matrices[key], matrices[key])
        for method in predictions:
            np.testing.assert_array_equal(changed_predictions[method], predictions[method])
        original_metrics = evaluate_predictions(run, predictions)
        changed_metrics = evaluate_predictions(changed_run, changed_predictions)
        self.assertFalse(np.allclose(original_metrics["MAE"], changed_metrics["MAE"]))

    def test_metrics_match_old_base_metrics_and_independent_per_node_calculation(self):
        run = load_sweep_run(self.run_dir)
        predictions = reconcile_forecasts(run.base, estimate_matrices(run))
        actual = evaluate_predictions(run, predictions)
        metrics = ["MAE", "RMSE", "MASE", "RMSSE"]
        self.assertEqual(len(actual), len(self.client_ids) * len(predictions))
        self.assertTrue({"method", "node_id", "unique_id", "level", *metrics}.issubset(actual.columns))
        self.assertFalse(actual.duplicated(["method", "node_id"]).any())
        scales = _scales_from_artifact(self.artifact_path)
        reference, reference_mean = _evaluate_forecasts(self.forecasts, scales, "base")
        base_metrics = actual.loc[actual.method == "base"].set_index("unique_id")[metrics].sort_index()
        np.testing.assert_allclose(base_metrics, reference.set_index("unique_id")[metrics].sort_index(), rtol=1e-12)
        np.testing.assert_allclose(base_metrics.mean(), [reference_mean[m] for m in metrics], rtol=1e-12)
        for method, predicted in predictions.items():
            for node, cid in enumerate(self.client_ids):
                with self.subTest(method=method, node_id=cid):
                    error = predicted[node] - self.truth[node]
                    diffs = np.diff(self.train_targets[node], axis=0)
                    mae, mse = np.mean(np.abs(error)), np.mean(error ** 2)
                    expected = [mae, np.sqrt(mse), mae / np.mean(np.abs(diffs)), np.sqrt(mse / np.mean(diffs ** 2))]
                    row = actual.loc[(actual.method == method) & (actual.node_id == cid)]
                    self.assertEqual(len(row), 1)
                    self.assertEqual(row.iloc[0].unique_id, self.unique_ids[node])
                    self.assertEqual(row.iloc[0].level, self.unique_ids[node].count("/") + 1)
                    np.testing.assert_allclose(row[metrics].to_numpy(float)[0], expected, rtol=1e-12, atol=1e-12)

    def test_loader_rejects_duplicate_missing_and_misaligned_forecasts(self):
        invalid = {
            "duplicate coordinate": (pd.concat([self.forecasts, self.forecasts.iloc[:1]]), "Duplicate forecast"),
            "missing coordinate": (self.forecasts.iloc[1:], "complete node/time/horizon grid"),
            "missing node": (self.forecasts.loc[self.forecasts.node_id != self.client_ids[0]], "node IDs/names do not match"),
            "mismatched node name": (self.forecasts.assign(unique_id=lambda f: f.unique_id.where(f.node_id != self.client_ids[0], "Wrong")), "node IDs/names do not match"),
            "nonfinite prediction": (self.forecasts.assign(base=lambda f: f.base.where(f.index != f.index[0], np.inf)), "base contains missing, NaN or infinite"),
            "nonfinite truth": (self.forecasts.assign(y_true=lambda f: f.y_true.where(f.index != f.index[0], np.nan)), "y_true contains missing, NaN or infinite"),
        }
        for label, (frame, reason) in invalid.items():
            with self.subTest(problem=label):
                frame.to_csv(self.forecast_path, index=False)
                with self.assertRaisesRegex(ValueError, reason):
                    load_sweep_run(self.run_dir)

    def test_loader_rejects_invalid_training_artifacts(self):
        invalid = {}
        bad_residuals = self.residuals.copy()
        bad_residuals[0, 0, 0] += 3
        invalid["residual identity"] = ({**self.artifacts, "residuals": bad_residuals}, "not train_targets - train_predictions")
        bad_predictions = self.artifacts["train_predictions"].copy()
        bad_predictions[0, 0, 0] = np.inf
        invalid["nonfinite training prediction"] = ({**self.artifacts, "train_predictions": bad_predictions}, "train_predictions contains missing, NaN or infinite")
        bad_targets = self.train_targets.copy()
        bad_targets[0, 0, 0] = np.nan
        invalid["nonfinite training target"] = ({**self.artifacts, "train_targets": bad_targets}, "train_targets contains missing, NaN or infinite")
        invalid["duplicate client id"] = ({**self.artifacts, "client_ids": np.array([100, 100, 5, 31, 7])}, "IDs and names must be unique and aligned")
        invalid["residual horizon mismatch"] = ({**self.artifacts, "residuals": self.residuals[:, :, :1]}, "must share shape")
        for label, (arrays, reason) in invalid.items():
            with self.subTest(problem=label):
                np.savez(self.artifact_path, **arrays)
                with self.assertRaisesRegex(ValueError, reason):
                    load_sweep_run(self.run_dir)


if __name__ == "__main__":
    unittest.main()
