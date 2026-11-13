from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = PROJECT_ROOT
from PPHF_modules.fhf.client_sampling import select_clients, validate_sampling_rate, sampled_output_dir
from PPHF_modules.fhf.FHF_Client import Client
from PPHF_modules.fhf.FHF_runner import Args, run_fhf
from PPHF_modules.fhf.HFL_runner_timed import HFLArgs, run_hfl_timed
from PPHF_modules.fhf.HFL_servers import EdgeServer, CloudServer
from PPHF_modules.dp_fhf.test_dp_fedavg import _DummyNode


class SamplingTests(unittest.TestCase):
    def test_rate_validation_and_rounding(self):
        for rate in (0, -0.1, 1.01, float("inf"), float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                validate_sampling_rate(rate)
        for size in (1, 3, 10, 11, 100):
            ids = list(range(size))
            selected = select_clients(ids, 0.1, np.random.default_rng(7))
            self.assertEqual(len(selected), math.ceil(size * .1))
            self.assertEqual(len(selected), len(set(selected)))
            self.assertEqual(selected, sorted(selected))
        self.assertEqual(len(select_clients(list(range(100)), .07, np.random.default_rng(1))), 7)
        self.assertEqual(len(select_clients(list(range(100)), .14, np.random.default_rng(1))), 14)

    def test_full_participation_does_not_consume_randomness(self):
        ids = [9, 2, 7]
        self.assertEqual(select_clients(ids, 1.0, None), ids)
        rng = np.random.default_rng(7)
        state = repr(rng.bit_generator.state)
        select_clients(ids, 1.0, rng)
        self.assertEqual(repr(rng.bit_generator.state), state)

    def test_selection_is_repeatable_and_separate_from_training_rng(self):
        def schedule(seed):
            rng = np.random.default_rng(seed)
            result = []
            for _ in range(10):
                result.append(select_clients(list(range(100)), .1, rng))
                torch.rand(100)
                np.random.random(100)
            return result
        self.assertEqual(schedule(42), schedule(42))
        self.assertNotEqual(schedule(42), schedule(43))

    def test_output_isolation_is_idempotent_and_full_path_is_unchanged(self):
        base = Path("old_results")
        self.assertEqual(sampled_output_dir(base, 1.0, 42), base)
        new = sampled_output_dir(base, .1, 42)
        self.assertNotEqual(new, base)
        self.assertEqual(sampled_output_dir(new, .1, 42), new)
        self.assertEqual(sampled_output_dir(new / "run1", .1, 42), new / "run1")
        self.assertNotEqual(sampled_output_dir(base, .1, 43), new)

    def test_inactive_branch_is_excluded_and_selected_cloud_can_train_alone(self):
        state = {"weight": torch.zeros(2), "counter": torch.tensor(0)}
        inactive = EdgeServer(1, [], state)
        nodes = {0: _DummyNode(torch.tensor([3., 4.])), 1: _DummyNode(torch.tensor([100., 100.]))}
        log = inactive.run_one_edge_round(nodes, 1.0, np.random.default_rng(1), selected_node_ids={0})
        self.assertEqual(log["num_train_events"], 0)
        self.assertEqual(inactive.last_upload_weight, 0)
        for mode in ("self_train_then_aggregate", "post_agg_local_finetune"):
            cloud = CloudServer(0, state)
            cloud.aggregate_edges({1: inactive}, nodes, mode, selected_node_ids={0})
            torch.testing.assert_close(cloud.global_sd["weight"], torch.tensor([3., 4.]))

    def test_resume_rejects_different_round_count_and_accepts_historical_full_runs(self):
        from PPHF_modules.dp_fhf.notebook_runtime import _validate_completed_configuration
        with tempfile.TemporaryDirectory() as temp:
            privacy = Path(temp) / "privacy.json"
            paths = {"privacy": privacy, "output_dir": Path(temp)}
            metadata = {"num_rounds": 100, "clip_norm": 1., "noise_multiplier": 1.2}
            privacy.write_text(json.dumps(metadata))
            expected = dict(num_rounds=100, clip_norm=1., noise_multiplier=1.2, sampling_rate=1., sampling_seed=42)
            _validate_completed_configuration(paths, **expected)
            with self.assertRaisesRegex(ValueError, "different parameters"):
                _validate_completed_configuration(paths, **(expected | {"num_rounds": 20}))
            with self.assertRaisesRegex(ValueError, "different parameters"):
                _validate_completed_configuration(paths, **(expected | {"sampling_rate": .1}))

    def test_actual_training_matches_sample_schedule_for_every_framework(self):
        # An unbalanced hierarchy includes both a terminal edge and nested edges.
        paths = ["Total", "Total/A", "Total/B", "Total/A/X", "Total/A/Y",
                 "Total/A/X/a", "Total/A/X/b"]
        t = np.arange(80)
        leaves = np.array([4 + i + np.sin(t / (3 + i)) for i in range(4)])
        values = np.vstack([leaves.sum(axis=0), leaves[1:].sum(axis=0), leaves[0],
                            leaves[2:].sum(axis=0), leaves[1], leaves[2], leaves[3]])
        with tempfile.TemporaryDirectory() as temp:
            csv = Path(temp) / "hierarchy.csv"
            pd.concat([pd.DataFrame({"partition_id": i, "unique_id": path,
                "ds": pd.date_range("2025-01-01", periods=len(t), freq="30min"), "y": values[i]})
                for i, path in enumerate(paths)]).to_csv(csv, index=False)
            for framework in ("fhf", "hff", "hffps"):
                for dp in (False, True):
                    for mode in ("self_train_then_aggregate", "post_agg_local_finetune"):
                        for execution in ("sequential", "synchronous"):
                            if framework == "fhf" and (mode != "self_train_then_aggregate" or execution != "sequential"):
                                continue
                            with self.subTest(framework=framework, dp=dp, mode=mode, execution=execution):
                                args = Args() if framework == "fhf" else HFLArgs()
                                for key, value in dict(csv_path=str(csv), truncated=None,
                                    lags=4, fh=0, hidden_size=4, num_layers=1, batch_size=24,
                                    num_rounds=4, early_stop_enabled=False, recon_method=("mint_ols",),
                                    validate_p2p=True, sampling_rate=.1, sampling_seed=42,
                                    dp_fedavg_enabled=dp, dp_fedavg_seed=89,
                                    verbose_rounds=False, verbose_cloud_rounds=False).items():
                                    setattr(args, key, value)
                                trained = []
                                original_train = Client.local_train
                                def record_train(node):
                                    trained.append(node.cid)
                                    if framework == "fhf" and dp:
                                        self.assertEqual(len(node.opt.state), 0)
                                    return original_train(node)
                                with patch.object(Client, "local_train", record_train), contextlib.redirect_stdout(io.StringIO()):
                                    if framework == "fhf":
                                        args.return_result = True
                                        result = run_fhf(args)
                                        logs = result["round_logs"]
                                        selected = [row["selected_client_ids"] for row in logs]
                                        nodes = result["clients"]
                                    else:
                                        args.edge_trainable = args.cloud_trainable = framework == "hffps"
                                        args.hier_train_mode = mode
                                        args.edge_execution_mode = execution
                                        result = run_hfl_timed(args)
                                        selected = [row["selected_node_ids"] for row in result["cloud_round_logs"]]
                                        nodes = result["nodes"]
                                self.assertEqual(trained, [cid for batch in selected for cid in batch])
                                self.assertTrue(all(len(batch) == 1 for batch in selected))
                                self.assertTrue(all(np.isfinite(node.base_forecast).all() for node in nodes.values()))
                                self.assertEqual(result["sampling"]["sampling_rate"], .1)


if __name__ == "__main__":
    unittest.main()
