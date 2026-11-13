from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn


SHARED = FHF_MODULE_DIR

from PPHF_modules.fhf.aggregation import clip_client_update, dp_fedavg_aggregate
from PPHF_modules.fhf.FHF_Cloud import Cloud
from PPHF_modules.fhf.HFL_servers import EdgeServer
from PPHF_modules.fhf.HFL_topology import infer_three_level_layout


class _DummyNode:
    def __init__(self, update: torch.Tensor, trainable: bool = True) -> None:
        self.update = update
        self.trainable = trainable
        self.state = {
            "weight": torch.zeros_like(update),
            "counter": torch.tensor(0, dtype=torch.int64),
        }

    def set_state_dict(self, state) -> None:
        self.state = {key: value.detach().clone() for key, value in state.items()}

    def local_train(self) -> None:
        self.state["weight"] = self.state["weight"] + self.update

    def get_state_dict(self):
        return {key: value.detach().clone() for key, value in self.state.items()}

    def num_train_samples(self) -> int:
        return 1

    def evaluate_train_losses(self):
        return {"normalised_loss": 0.0, "actual_loss": 0.0}


class ClientSideDPFedAvgTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {
            "weight": torch.tensor([1.0, -1.0]),
            "bias": torch.tensor([0.5]),
            "counter": torch.tensor(3, dtype=torch.int64),
        }

    def test_client_upload_is_delta_only_and_globally_clipped(self) -> None:
        local = {
            "weight": torch.tensor([4.0, 3.0]),
            "bias": torch.tensor([0.5]),
            "counter": torch.tensor(9, dtype=torch.int64),
        }
        upload = clip_client_update(local, self.reference, max_update_norm=2.5)

        self.assertEqual(set(upload), {"weight", "bias"})
        total_norm = torch.sqrt(sum(torch.sum(x * x) for x in upload.values()))
        self.assertAlmostEqual(float(total_norm), 2.5, places=5)
        self.assertFalse(torch.equal(upload["weight"], local["weight"]))

    def test_server_aggregates_clipped_updates_without_absolute_models(self) -> None:
        first = {
            "weight": torch.tensor([2.0, -1.0]),
            "bias": torch.tensor([1.0]),
            "counter": torch.tensor(4, dtype=torch.int64),
        }
        second = {
            "weight": torch.tensor([0.0, 1.0]),
            "bias": torch.tensor([0.0]),
            "counter": torch.tensor(5, dtype=torch.int64),
        }
        uploads = [
            clip_client_update(first, self.reference, max_update_norm=10.0),
            clip_client_update(second, self.reference, max_update_norm=10.0),
        ]
        result = dp_fedavg_aggregate(
            clipped_updates=uploads,
            weights=[1, 1],
            reference_state_dict=self.reference,
            max_update_norm=10.0,
            noise_multiplier=0.0,
        )

        torch.testing.assert_close(result["weight"], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(result["bias"], torch.tensor([0.5]))
        self.assertEqual(int(result["counter"]), 3)

    def test_server_rejects_an_unclipped_upload(self) -> None:
        invalid_upload = {
            "weight": torch.tensor([3.0, 4.0]),
            "bias": torch.tensor([0.0]),
        }
        with self.assertRaisesRegex(ValueError, "client-side clipping"):
            dp_fedavg_aggregate(
                clipped_updates=[invalid_upload],
                weights=[1],
                reference_state_dict=self.reference,
                max_update_norm=1.0,
                noise_multiplier=1.0,
            )

    def test_hierarchical_edge_path_uploads_only_client_clipped_deltas(self) -> None:
        config = {
            "enabled": True,
            "max_update_norm": 1.0,
            "noise_multiplier": 0.0,
            "adjacency": "add_remove",
            "weighting": "uniform",
            "seed": 7,
        }
        initial = {
            "weight": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.int64),
        }
        server = EdgeServer(10, [1, 2], initial, dp_fedavg_config=config)
        nodes = {
            1: _DummyNode(torch.tensor([3.0, 4.0])),
            2: _DummyNode(torch.tensor([0.0, 2.0])),
            10: _DummyNode(torch.zeros(2), trainable=False),
        }

        captured = []

        def capture_and_aggregate(*args, **kwargs):
            uploads = kwargs["clipped_updates"]
            captured.extend(uploads)
            return dp_fedavg_aggregate(*args, **kwargs)

        with patch("PPHF_modules.fhf.HFL_servers.dp_fedavg_aggregate", side_effect=capture_and_aggregate):
            log = server.run_one_edge_round(
                nodes,
                client_frac=1.0,
                rng=np.random.default_rng(3),
                hier_train_mode="self_train_then_aggregate",
            )

        self.assertEqual(log["selected_client_ids"], [1, 2])
        self.assertEqual(len(captured), 2)
        for upload in captured:
            self.assertEqual(set(upload), {"weight"})
            self.assertLessEqual(float(torch.linalg.vector_norm(upload["weight"])), 1.00001)
        torch.testing.assert_close(server.edge_sd["weight"], torch.tensor([0.3, 0.9]))

    def test_deduplicated_unbalanced_hierarchy_has_terminal_edges(self) -> None:
        paths = {
            0: "Total",
            1: "Total/CT",
            2: "Total/MASS",
            3: "Total/ME",
            4: "Total/MASS/NEMASSBOST",
            5: "Total/MASS/SEMASS",
        }
        layout = infer_three_level_layout(list(paths), paths)

        self.assertEqual(layout.cloud_id, 0)
        self.assertEqual(layout.edge_to_clients[2], [4, 5])
        self.assertEqual(layout.terminal_edge_ids, [1, 3])
        self.assertEqual(layout.edge_to_edges[2], [])
        self.assertEqual(layout.root_edge_ids, [1, 2, 3])

    def test_multilevel_hierarchy_reuses_child_edges(self) -> None:
        paths = {
            0: "Total",
            1: "Total/RegionA",
            2: "Total/RegionB",
            3: "Total/RegionA/ZoneA",
            4: "Total/RegionA/ZoneB",
            5: "Total/RegionA/ZoneA/SiteA",
            6: "Total/RegionA/ZoneA/SiteB",
        }
        layout = infer_three_level_layout(list(paths), paths)

        self.assertEqual(layout.root_edge_ids, [1, 2])
        self.assertEqual(layout.edge_to_edges[1], [3])
        self.assertEqual(layout.edge_to_clients[1], [4])
        self.assertEqual(layout.edge_to_clients[3], [5, 6])
        self.assertEqual(layout.client_ids, [4, 5, 6])
        S, _, bottoms = Cloud(nn.Linear(1, 1), list(paths)).build_hierarchy_S(paths)
        self.assertEqual(S.shape, (7, 4))
        self.assertEqual(
            bottoms,
            [
                "Total/RegionB",
                "Total/RegionA/ZoneB",
                "Total/RegionA/ZoneA/SiteA",
                "Total/RegionA/ZoneA/SiteB",
            ],
        )

        initial = {
            "weight": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.int64),
        }
        child = EdgeServer(3, [5, 6], initial)
        parent = EdgeServer(1, [4], initial, child_edge_ids=[3])
        nodes = {
            1: _DummyNode(torch.zeros(2), trainable=False),
            2: _DummyNode(torch.zeros(2), trainable=False),
            3: _DummyNode(torch.zeros(2), trainable=False),
            4: _DummyNode(torch.tensor([5.0, 0.0])),
            5: _DummyNode(torch.tensor([1.0, 0.0])),
            6: _DummyNode(torch.tensor([3.0, 0.0])),
        }
        rng = np.random.default_rng(3)
        child.run_one_edge_round(nodes, client_frac=1.0, rng=rng)
        parent.run_one_edge_round(
            nodes,
            client_frac=1.0,
            rng=rng,
            child_edge_servers={3: child},
        )

        torch.testing.assert_close(child.edge_sd["weight"], torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(parent.edge_sd["weight"], torch.tensor([3.0, 0.0]))
        self.assertEqual(parent.last_upload_weight, 3)

    def test_terminal_edge_trains_its_own_series_once(self) -> None:
        initial = {
            "weight": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.int64),
        }
        server = EdgeServer(1, [], initial)
        node = _DummyNode(torch.tensor([1.0, 2.0]), trainable=True)

        log = server.run_one_edge_round(
            {1: node},
            client_frac=1.0,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(log["selected_client_ids"], [1])
        self.assertEqual(log["num_train_events"], 1)
        torch.testing.assert_close(server.edge_sd["weight"], torch.tensor([1.0, 2.0]))

    def test_reconciliation_allows_duplicates_by_default_and_can_reject(self) -> None:
        paths = {0: "Total", 1: "Total/CT", 2: "Total/CT/CT"}
        cloud = Cloud(nn.Linear(1, 1), list(paths))

        S, _, _ = cloud.build_hierarchy_S(paths)
        self.assertEqual(S.shape, (3, 1))
        self.assertEqual(
            cloud.duplicate_series_groups,
            [["Total", "Total/CT", "Total/CT/CT"]],
        )
        with self.assertRaisesRegex(ValueError, "duplicate series"):
            cloud.build_hierarchy_S(paths, reject_duplicate_series=True)


if __name__ == "__main__":
    unittest.main()
