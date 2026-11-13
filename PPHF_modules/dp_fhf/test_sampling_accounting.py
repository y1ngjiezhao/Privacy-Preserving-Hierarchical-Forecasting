"""Mechanism/accountant agreement, reference moments and legacy compatibility."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
import math
import sys
import unittest
import contextlib
import io
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
from PPHF_modules.fhf.aggregation import dp_fedavg_aggregate
from PPHF_modules.dp_fhf.privacy_accounting import gaussian_composition_epsilon, sampled_fedavg_epsilon
from PPHF_modules.dp_fhf.calculate_sampling_privacy import sampled_rdp
from PPHF_modules.dp_fhf.notebook_runtime import _fedavg_command
from PPHF_modules.fhf.FHF_runner import Args, run_fhf
from PPHF_modules.fhf.FHF_Client import Client
from PPHF_modules.fhf.aggregation import clip_client_update
from PPHF_modules.fhf.client_sampling import select_clients


class SamplingAccountingTests(unittest.TestCase):
    def test_full_participation_returns_exact_historical_accounting(self):
        for sigma in (.3, 1.2):
            for adjacency in ("add_remove", "replace_one"):
                self.assertEqual(sampled_fedavg_epsilon(sigma, 100, 1e-5, 100, 100, adjacency),
                                 gaussian_composition_epsilon(sigma, 100, 1e-5))

    def test_smaller_samples_give_smaller_bounds_for_add_remove(self):
        for sigma in (.3, 1.2):
            values = [sampled_fedavg_epsilon(sigma, 100, 1e-5, 100, m)["epsilon"] for m in (5, 10, 20, 50)]
            self.assertEqual(values, sorted(values))
            self.assertLess(values[1], gaussian_composition_epsilon(sigma, 100, 1e-5)["epsilon"])

    def test_reference_moment_by_independent_numerical_integration(self):
        for sigma in (.3, 1.2):
            result = sampled_fedavg_epsilon(sigma, 100, 1e-5, 13, 2)
            self.assertEqual(result["effective_sampling_rate"], 2/13)
            alpha = result["optimal_rdp_order"]
            reference = sampled_rdp(2/13, sigma, alpha, .01) * 100
            # The library computes a conservative series bound at fractional
            # orders; independent integration supplies the underlying moment.
            self.assertGreaterEqual(result["rdp_at_optimal_order"] + 1e-8, reference)
            self.assertLess(result["rdp_at_optimal_order"], reference * 1.01)
            self.assertAlmostEqual(result["epsilon"], result["rdp_at_optimal_order"] + math.log(1e5)/(alpha-1), places=6)

    def test_replace_one_uses_without_replacement_accountant(self):
        result = sampled_fedavg_epsilon(1.2, 100, 1e-5, 100, 10, "replace_one")
        self.assertTrue(result["sampling_amplification_used"])
        self.assertLess(result["epsilon"], result["epsilon_without_sampling_amplification"])
        self.assertEqual(result["adjacency"], "replace_one")

    def test_invalid_counts_and_adjacency_are_rejected(self):
        for n, m in ((10, 0), (10, 11), (0, 0), (10, 1.5)):
            with self.subTest(n=n, m=m), self.assertRaises(ValueError):
                sampled_fedavg_epsilon(1.2, 100, 1e-5, n, m)
        with self.assertRaises(ValueError):
            sampled_fedavg_epsilon(1.2, 100, 1e-5, 100, 10, "other")

    def test_actual_add_remove_noise_matches_fixed_size_sensitivity(self):
        def draw(factor, adjacency="add_remove"):
            return dp_fedavg_aggregate(
                [{"w": torch.zeros(20)}]*2, [1,1], {"w": torch.zeros(20)}, 2., .4,
                adjacency=adjacency, generator=torch.Generator().manual_seed(42),
                sensitivity_multiplier=factor,
            )["w"]
        torch.testing.assert_close(draw(2), 2*draw(1), rtol=0, atol=0)
        torch.testing.assert_close(draw(2), draw(1, "replace_one"), rtol=0, atol=0)

    def test_notebook_commands_use_fresh_randomness_for_partial_runs(self):
        def command(q, seed):
            return _fedavg_command(ROOT, ROOT/"PPHF_modules/dp_fhf/run_dp_fedavg.py", Path("data.csv"),
                                    Path("result"), 100, 1., 1.2, q, seed)
        fresh = command(.1, None)
        self.assertNotIn("--sampling-seed", fresh)
        self.assertNotIn("--dp-seed", fresh)
        historical = command(1., None)
        self.assertEqual(historical[historical.index("--dp-seed")+1], "42")
        self.assertNotIn("None", fresh)

    def test_a_clients_upload_is_independent_of_other_selected_clients(self):
        # Client 3 is trained first in one subset and second in the other.
        seeds = {}
        for seed in range(100):
            subset = select_clients(list(range(7)), .2, np.random.default_rng(seed))
            if 3 in subset:
                seeds.setdefault(subset.index(3), seed)
        self.assertEqual(set(seeds), {0, 1})
        with tempfile.TemporaryDirectory() as temp:
            csv = Path(temp)/"data.csv"
            t = np.arange(96)
            leaves = np.array([5+i+np.sin(t/(3+i)) for i in range(6)])
            values = np.vstack([leaves.sum(axis=0), leaves])
            paths = ["Total"]+[f"Total/{i}" for i in range(1,7)]
            pd.concat([pd.DataFrame({"partition_id": i, "unique_id": path,
                "ds": pd.date_range("2025-01-01",periods=96,freq="30min"), "y": values[i]})
                for i,path in enumerate(paths)]).to_csv(csv,index=False)
            uploads=[]
            for seed in seeds.values():
                args=Args(csv_path=str(csv),truncated=None,lags=4,fh=0,num_rounds=1,
                    hidden_size=4,num_layers=2,batch_size=24,early_stop_enabled=False,
                    return_result=True,recon_method=("mint_ols",),verbose_rounds=False,
                    dp_fedavg_enabled=True,dp_fedavg_seed=89,sampling_rate=.2,sampling_seed=seed)
                active=[]; captured={}; original=Client.local_train
                def train(node):
                    active[:]=[node.cid]
                    return original(node)
                def clip(*a,**kw):
                    payload=clip_client_update(*a,**kw)
                    captured[active[0]]=payload
                    return payload
                with patch.object(Client,"local_train",train),patch("PPHF_modules.fhf.FHF_runner.clip_client_update",clip),contextlib.redirect_stdout(io.StringIO()):
                    run_fhf(args)
                uploads.append(captured[3])
            for key in uploads[0]:
                torch.testing.assert_close(uploads[0][key],uploads[1][key],rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
