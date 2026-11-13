"""Check independent rate sweeps, shared result collection, and safe resume."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from PPHF_modules.dp_fhf.ausgrid_sampling_sweep import (
    run_ausgrid_sampling_sweep, summarize_ausgrid_sampling_sweeps,
)
from PPHF_modules.dp_fhf.notebook_runtime import run_sweep_dpfedavg, summarize_dpfedavg


class AusGridSamplingSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "DP_fhf/outputs/ausgrid2013_sampling_sweep"
        self.runner = self.root / "runner.py"
        self.options = dict(root=self.root, csv_path=self.root / "dataset.csv",
                            noise_multipliers=[1.2, .9, .6, .3],
                            clip_norms=[1., .5, .2], output_root=self.output)
        self.addCleanup(patch.stopall)
        patch("PPHF_modules.dp_fhf.notebook_runtime._validate_runner", return_value=self.runner).start()
        patch("PPHF_modules.dp_fhf.notebook_runtime._device", return_value="cpu").start()

    def test_three_rates_have_36_unique_directories_beneath_one_parent(self):
        frames = []
        with contextlib.redirect_stdout(io.StringIO()):
            for rate in (.1, .5, 1.):
                frames.append(run_ausgrid_sampling_sweep(**self.options, sampling_rate=rate, dry_run=True))
        grid = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(grid), 36)
        self.assertEqual(grid.output_dir.nunique(), 36)
        self.assertEqual(grid.groupby("sampling_rate").size().tolist(), [12, 12, 12])
        self.assertTrue(all(Path(p).is_relative_to(self.output) for p in grid.output_dir))
        self.assertTrue(all(Path(p).is_relative_to(self.output / "full_participation")
                            for p in grid.loc[grid.sampling_rate.eq(1), "output_dir"]))
        self.assertFalse(self.output.exists(), "Dry run must not create experiment outputs")

    def test_default_legacy_sweep_path_is_unchanged(self):
        with contextlib.redirect_stdout(io.StringIO()):
            grid = run_sweep_dpfedavg(root=self.root, dataset="AusGrid2013",
                csv_path=self.root / "data.csv", noise_multipliers=[.3], clip_norms=[1.], dry_run=True)
        expected = self.root / "DP_fhf/outputs/01_dp_fedavg_sweep_20round/AusGrid2013/nm_0p3__clip_1"
        self.assertEqual(Path(grid.iloc[0].output_dir), expected)

    def test_three_rates_resume_and_are_included_in_both_summaries(self):
        def complete(command, root):
            def option(name):return command[command.index(name) + 1]
            directory = Path(option("--output-dir"))
            # This is already the CLI's isolated final path, with no repeated suffix.
            from PPHF_modules.fhf.client_sampling import sampled_output_dir, amplification_output_dir
            rate = float(option("--sampling-rate"))
            self.assertEqual(directory, sampled_output_dir(
                amplification_output_dir(directory, rate < 1), rate, None))
            metadata = {"framework": "fhf", "num_rounds": 20,
                        "clip_norm": float(option("--clip-norm")),
                        "noise_multiplier": float(option("--noise-multiplier")),
                        "sampling_rate": rate, "sampling_seed": None,
                        "mechanism_version": "fixed_wor_gaussian_v1",
                        "accounting": {"epsilon": 2.0, "delta": 1e-5,
                                       "sampling_amplification_used": rate < 1}}
            (directory / "fh1_FHF_dp_fedavg_privacy.json").write_text(json.dumps(metadata))
            np.savez(directory / "fh1_FHF_dp_fedavg_training_artifacts.npz",
                client_ids=np.array([0]), unique_ids=np.array(["Total"]),
                train_targets=np.arange(6, dtype=float).reshape(1, 6, 1))
            pd.DataFrame({"unique_id": ["Total", "Total"], "y_true": [5., 6.],
                          "base": [5.2, 5.8]}).to_csv(
                directory / "fh1_FHF_dp_fedavg_base_forecasts.csv", index=False)
        options = {**self.options, "noise_multipliers": [.3], "clip_norms": [1.]}
        with patch("PPHF_modules.dp_fhf.notebook_runtime._run_streaming", side_effect=complete) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                for rate in (.1, .5, 1.):
                    run_ausgrid_sampling_sweep(**options, sampling_rate=rate)
                self.assertEqual(run.call_count, 3)
                for rate in (.1, .5, 1.):
                    run_ausgrid_sampling_sweep(**options, sampling_rate=rate)
                self.assertEqual(run.call_count, 3, "Completed runs must be resumed")
                combined = summarize_ausgrid_sampling_sweeps(self.output)
                global_summary = summarize_dpfedavg(self.root)
        self.assertEqual(len(combined), 3)
        self.assertEqual(set(combined.sampling_rate), {.1, .5, 1.})
        self.assertEqual(combined.run_directory.nunique(), 3)
        self.assertEqual(len(global_summary), 3)
        self.assertEqual(set(global_summary.sampling_rate), {.1, .5, 1.})
        self.assertTrue((self.output / "all_sampling_rates_summary.csv").is_file())

    def test_empty_summary_does_not_create_output_directory(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(summarize_ausgrid_sampling_sweeps(self.output).empty)
        self.assertFalse(self.output.exists())

    def test_combined_summary_rejects_other_datasets(self):
        directory = self.output / "wrong"
        directory.mkdir(parents=True)
        pd.DataFrame([dict(dataset="GEF2017", framework="FHF", sampling_rate=.1,
            run_id="nm_0p3__clip_1", clip_norm=1, noise_multiplier=.3,
            num_rounds=20, epsilon=2)]).to_csv(directory / "sweep_summary.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Expected AusGrid2013"):
            summarize_ausgrid_sampling_sweeps(self.output)


if __name__ == "__main__":
    unittest.main()
