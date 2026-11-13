"""Prevent late dependency failures and recover saved metrics without training."""
from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from PPHF_modules.dp_fhf import notebook_runtime, run_dp_fedavg
from PPHF_modules.dp_fhf.privacy_accounting import (
    check_sampling_accountant_dependencies, gaussian_composition_epsilon,
    sampled_fedavg_epsilon,
)
from PPHF_modules.dp_fhf.recover_dp_sweep import recover_ausgrid_sampling_sweep
from PPHF_modules.fhf.client_sampling import sampled_output_dir, amplification_output_dir


class AccountingPreflightTests(unittest.TestCase):
    def test_missing_attr_is_reported_with_the_correct_distribution_name(self):
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "attr":raise ModuleNotFoundError("No module named 'attr'", name="attr")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "attrs>=22,<24") as caught:
                check_sampling_accountant_dependencies()
        self.assertIn(sys.executable, str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)

    def test_sweep_fails_before_launching_any_training(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(notebook_runtime, "check_sampling_accountant_dependencies", side_effect=RuntimeError("missing attrs")):
                with patch.object(notebook_runtime, "_run_streaming") as train:
                    with self.assertRaisesRegex(RuntimeError, "missing attrs"):
                        notebook_runtime.run_sweep_dpfedavg(root=d, dataset="AusGrid2013",
                            csv_path=Path(d)/"data.csv", noise_multipliers=[.3], clip_norms=[1.], sampling_rate=.1)
                    train.assert_not_called()
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_cli_fails_before_training_and_before_creating_an_output_folder(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/"run"
            argv = ["runner", "--csv", str(Path(d)/"data.csv"), "--output-dir", str(output), "--sampling-rate", ".1"]
            with patch.object(sys, "argv", argv), patch.object(run_dp_fedavg, "check_sampling_accountant_dependencies", side_effect=RuntimeError("missing attrs")):
                with patch.object(run_dp_fedavg, "_execute_with_heartbeat") as train:
                    with self.assertRaisesRegex(RuntimeError, "missing attrs"):
                        run_dp_fedavg.main()
                    train.assert_not_called()
            self.assertFalse(output.exists())

    def test_full_participation_accounting_does_not_require_sampled_dependencies(self):
        with patch("PPHF_modules.dp_fhf.privacy_accounting.check_sampling_accountant_dependencies", side_effect=RuntimeError("missing")):
            self.assertEqual(sampled_fedavg_epsilon(1.2, 2, 1e-5, 7, 7), gaussian_composition_epsilon(1.2, 2, 1e-5))

    def test_failed_sweep_raises_instead_of_returning_an_empty_success_table(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(notebook_runtime, "_validate_runner", return_value=Path(d)/"runner.py"), patch.object(notebook_runtime, "_device", return_value="cpu"):
                with patch.object(notebook_runtime, "_run_streaming", side_effect=RuntimeError("accounting failed")):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, "No sweep runs completed successfully"):
                            notebook_runtime.run_sweep_dpfedavg(root=d, dataset="AusGrid2013",
                                csv_path=Path(d)/"data.csv", noise_multipliers=[.3], clip_norms=[1.])


class SavedSweepRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()/"ausgrid2013_sampling_sweep"
        self.rate = .1
        self.directory = sampled_output_dir(amplification_output_dir(self.parent, True), self.rate, None)
        self.directory.mkdir(parents=True)
        rows=[]
        for clip,slug in [(1.,"1"),(.5,"0p5")]:
            run_id=f"nm_1p2__clip_{slug}";run=self.directory/run_id;run.mkdir()
            names=np.array([f"node{i}" for i in range(7)])
            targets=np.stack([np.arange(8,dtype=float)+i for i in range(7)])[:,:,None]
            np.savez_compressed(run/"fh1_FHF_dp_fedavg_training_artifacts.npz",client_ids=np.arange(7),unique_ids=names,train_targets=targets)
            pd.DataFrame([{"unique_id":name,"y_true":float(t+i),"base":float(t+i)+.25} for i,name in enumerate(names) for t in [8,9]]).to_csv(run/"fh1_FHF_dp_fedavg_base_forecasts.csv",index=False)
            rows.append(dict(run_id=run_id,noise_multiplier=1.2,clip_norm=clip,sampling_rate=self.rate))
        pd.DataFrame(rows).to_csv(self.directory/"parameter_grid.csv",index=False)

    def recover(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return recover_ausgrid_sampling_sweep(output_root=self.parent,sampling_rate=self.rate,num_rounds=2)

    def test_recovery_reproduces_metrics_and_epsilon_without_mutating_sources_or_training(self):
        paths=list(self.directory.rglob("*.npz"))+list(self.directory.glob("nm_*/*.csv"))
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with patch.object(notebook_runtime,"_run_streaming",side_effect=AssertionError("Recovery must not train")):
            result=self.recover()
        self.assertEqual(len(result),2)
        np.testing.assert_allclose(result[["MAE","RMSE","MASE","RMSSE"]],.25,rtol=0,atol=1e-12)
        expected=sampled_fedavg_epsilon(1.2,2,1e-5,7,1)["epsilon"]
        np.testing.assert_allclose(result.epsilon,expected,rtol=0,atol=1e-12)
        self.assertTrue(result.sample_size.eq(1).all())
        self.assertFalse(result.model_release_file_exists.any())
        self.assertFalse(list(self.directory.rglob("fh1_FHF_dp_fedavg_privacy.json")))
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self.assertTrue((self.parent/"recovered_all_sampling_rates_summary.csv").is_file())

    def test_missing_pairs_are_reported_without_inventing_result_rows(self):
        (self.directory/"nm_1p2__clip_0p5/fh1_FHF_dp_fedavg_base_forecasts.csv").unlink()
        result=self.recover();self.assertEqual(len(result),1)
        report=json.loads((self.directory/"recovery/recovery_report.json").read_text())
        self.assertEqual(report["recovered_runs"],1)
        self.assertEqual(len(report["missing_runs"]),1)

    def test_mismatched_saved_sampling_rate_is_rejected(self):
        grid=self.directory/"parameter_grid.csv"
        table=pd.read_csv(grid);table["sampling_rate"]=.5;table.to_csv(grid,index=False)
        with self.assertRaisesRegex(ValueError,"differs from the saved parameter grid"):
            self.recover()


if __name__ == "__main__":
    unittest.main()
