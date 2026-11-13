from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from PPHF_modules.fhf.FHF_Client import Client as BaseClient
from PPHF_modules.fhf.aggregation import clip_client_update, dp_fedavg_aggregate, fedavg_aggregate


VALID_HIER_TRAIN_MODES = {"post_agg_local_finetune", "self_train_then_aggregate"}


def clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def prepare_aggregation_upload(
    local_state_dict: Dict[str, torch.Tensor],
    reference_state_dict: Dict[str, torch.Tensor],
    dp_fedavg_config: dict,
) -> Dict[str, torch.Tensor]:
    """Create the payload at the simulated sender side of one aggregation."""
    if not bool(dp_fedavg_config.get("enabled", False)):
        return local_state_dict
    return clip_client_update(
        state_dict=local_state_dict,
        reference_state_dict=reference_state_dict,
        max_update_norm=float(dp_fedavg_config["max_update_norm"]),
    )


def _safe_average(values) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _evaluate_node_losses(node: BaseClient) -> Dict[str, float]:
    """Evaluate a node's train loss on normalised and original scales."""
    if hasattr(node, "evaluate_train_losses"):
        losses = node.evaluate_train_losses()
        normalised = losses.get("normalised_loss", np.nan)
        actual = losses.get("actual_loss", np.nan)
        return {
            "normalised_loss": float(normalised) if np.isfinite(float(normalised)) else float("nan"),
            "actual_loss": float(actual) if np.isfinite(float(actual)) else float("nan"),
        }
    return {"normalised_loss": float("nan"), "actual_loss": float("nan")}


class HFLSeriesNode(BaseClient):
    def __init__(self, cid: int, series_name: str, df_one, args, role: str, parent_id: Optional[int], trainable: bool):
        super().__init__(cid=cid, series_name=series_name, df_one=df_one, args=args)
        self.role = str(role)
        self.parent_id = None if parent_id is None else int(parent_id)
        self.child_ids: List[int] = []
        self.trainable = bool(trainable)
        self.W_columns: Dict[str, np.ndarray] = {}

    def set_W_column(self, W_col: np.ndarray, method: str):
        arr = np.asarray(W_col, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.W_columns[str(method)] = arr


class EdgeServer:
    def __init__(
        self,
        edge_node_id: int,
        child_client_ids: Sequence[int],
        init_sd: Dict[str, torch.Tensor],
        dp_fedavg_config: Optional[dict] = None,
        child_edge_ids: Optional[Sequence[int]] = None,
    ):
        self.edge_node_id = int(edge_node_id)
        self.child_client_ids = [int(x) for x in child_client_ids]
        self.child_edge_ids = [int(x) for x in (child_edge_ids or [])]
        self.edge_sd = clone_state_dict(init_sd)
        self.last_upload_weight: int = max(1, len(self.child_client_ids))
        self.last_train_losses: Dict[int, float] = {}
        self.last_selected_client_ids: List[int] = []
        self.dp_fedavg_config = dict(dp_fedavg_config or {})
        self.dp_aggregation_calls = 0
        self._dp_generator = self._build_dp_generator(seed_offset=self.edge_node_id)

    def _build_dp_generator(self, seed_offset: int) -> Optional[torch.Generator]:
        if not bool(self.dp_fedavg_config.get("enabled", False)):
            return None
        generator = torch.Generator(device="cpu")
        seed = self.dp_fedavg_config.get("seed")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed) + int(seed_offset))
        return generator

    def _aggregate(
        self,
        uploads: List[Dict[str, torch.Tensor]],
        weights: List[int],
        reference_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not bool(self.dp_fedavg_config.get("enabled", False)):
            return fedavg_aggregate(uploads, weights)
        weighting = str(self.dp_fedavg_config.get("weighting", "uniform"))
        aggregation_weights = [1] * len(weights) if weighting == "uniform" else weights
        result = dp_fedavg_aggregate(
            clipped_updates=uploads,
            weights=aggregation_weights,
            reference_state_dict=reference_state_dict,
            max_update_norm=float(self.dp_fedavg_config["max_update_norm"]),
            noise_multiplier=float(self.dp_fedavg_config["noise_multiplier"]),
            adjacency=str(self.dp_fedavg_config.get("adjacency", "add_remove")),
            generator=self._dp_generator,
        )
        self.dp_aggregation_calls += 1
        return result

    def sync_from_cloud(self, global_sd: Dict[str, torch.Tensor]):
        self.edge_sd = clone_state_dict(global_sd)

    def run_one_edge_round(
        self,
        nodes: Dict[int, HFLSeriesNode],
        client_frac: float,
        rng: np.random.Generator,
        hier_train_mode: str = "post_agg_local_finetune",
        child_edge_servers: Optional[Dict[int, "EdgeServer"]] = None,
        selected_node_ids: Optional[set[int]] = None,
    ):
        hier_train_mode = str(hier_train_mode)
        if hier_train_mode not in VALID_HIER_TRAIN_MODES:
            raise ValueError(f"Unsupported hier_train_mode={hier_train_mode}. Valid options: {sorted(VALID_HIER_TRAIN_MODES)}")
        child_edge_servers = child_edge_servers or {}
        missing_child_edges = [
            eid for eid in self.child_edge_ids if eid not in child_edge_servers
        ]
        if missing_child_edges:
            raise KeyError(
                f"Edge {self.edge_node_id} is missing child edge servers "
                f"{missing_child_edges}."
            )

        if not self.child_client_ids and not self.child_edge_ids:
            # In a deduplicated, unbalanced hierarchy an edge-level series can
            # itself be a leaf (for example Total/CT after Total/CT/CT is
            # removed).  Train that colocated series exactly once and use its
            # model as the edge upload.
            edge_node = nodes[self.edge_node_id]
            if selected_node_ids is not None and self.edge_node_id not in selected_node_ids:
                self.last_upload_weight = 0
                self.last_train_losses = {}
                self.last_selected_client_ids = []
                return {"edge_id": self.edge_node_id, "selected_client_ids": [],
                        "train_losses": {}, "train_normalised_losses": {},
                        "train_actual_losses": {}, "num_train_events": 0}
            edge_node.set_state_dict(self.edge_sd)
            edge_node.local_train()
            loss_info = _evaluate_node_losses(edge_node)
            normalised_loss = float(loss_info.get("normalised_loss", float("nan")))
            actual_loss = float(loss_info.get("actual_loss", float("nan")))
            weight = max(1, edge_node.num_train_samples())
            upload = prepare_aggregation_upload(
                local_state_dict=edge_node.get_state_dict(),
                reference_state_dict=self.edge_sd,
                dp_fedavg_config=self.dp_fedavg_config,
            )
            self.edge_sd = self._aggregate(
                [upload], [weight], reference_state_dict=self.edge_sd
            )
            edge_node.set_state_dict(self.edge_sd)
            self.last_upload_weight = weight
            self.last_train_losses = {self.edge_node_id: normalised_loss}
            self.last_selected_client_ids = [self.edge_node_id]
            return {
                "edge_id": self.edge_node_id,
                "selected_client_ids": [self.edge_node_id],
                "train_losses": {self.edge_node_id: normalised_loss},
                "train_normalised_losses": {self.edge_node_id: normalised_loss},
                "train_actual_losses": {self.edge_node_id: actual_loss},
                "avg_normalised_loss": normalised_loss,
                "avg_actual_loss": actual_loss,
                "avg_train_loss": normalised_loss,
                "num_train_events": 1,
            }

        selected_num = max(1, int(round(len(self.child_client_ids) * float(client_frac)))) if self.child_client_ids else 0
        if selected_num >= len(self.child_client_ids):
            selected = list(self.child_client_ids)
        else:
            selected = list(map(int, rng.choice(self.child_client_ids, size=selected_num, replace=False).tolist()))
        if selected_node_ids is not None:
            selected = [cid for cid in selected if cid in selected_node_ids]

        client_uploads: List[Dict[str, torch.Tensor]] = []
        weights: List[int] = []
        train_normalised_losses: Dict[int, float] = {}
        train_actual_losses: Dict[int, float] = {}

        # Reuse the final model produced by every lower-level edge as one
        # child upload.  The upload is clipped relative to this edge's incoming
        # state when hierarchical DP-FedAvg is enabled.
        for child_eid in self.child_edge_ids:
            child_server = child_edge_servers[child_eid]
            if selected_node_ids is not None and child_server.last_upload_weight == 0:
                continue
            client_uploads.append(
                prepare_aggregation_upload(
                    local_state_dict=child_server.edge_sd,
                    reference_state_dict=self.edge_sd,
                    dp_fedavg_config=self.dp_fedavg_config,
                )
            )
            weights.append(max(1, child_server.last_upload_weight))

        for cid in selected:
            node = nodes[cid]
            node.set_state_dict(self.edge_sd)
            node.local_train()
            loss_info = _evaluate_node_losses(node)
            train_normalised_losses[int(cid)] = float(loss_info.get("normalised_loss", float("nan")))
            train_actual_losses[int(cid)] = float(loss_info.get("actual_loss", float("nan")))
            client_uploads.append(
                prepare_aggregation_upload(
                    local_state_dict=node.get_state_dict(),
                    reference_state_dict=self.edge_sd,
                    dp_fedavg_config=self.dp_fedavg_config,
                )
            )
            weights.append(max(1, node.num_train_samples()))

        child_agg_sd = self._aggregate(
            client_uploads, weights, reference_state_dict=self.edge_sd
        ) if client_uploads else clone_state_dict(self.edge_sd)
        total_child_weight = int(sum(weights))

        edge_node = nodes[self.edge_node_id]
        if edge_node.trainable and (selected_node_ids is None or self.edge_node_id in selected_node_ids):
            if hier_train_mode == "post_agg_local_finetune":
                edge_node.set_state_dict(child_agg_sd)
                edge_node.local_train()
                edge_loss_info = _evaluate_node_losses(edge_node)
                train_normalised_losses[int(self.edge_node_id)] = float(edge_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.edge_node_id)] = float(edge_loss_info.get("actual_loss", float("nan")))
                edge_local_sd = edge_node.get_state_dict()
                edge_weight = max(1, edge_node.num_train_samples())
                if not client_uploads:
                    self.edge_sd = self._aggregate(
                        [prepare_aggregation_upload(edge_local_sd, child_agg_sd, self.dp_fedavg_config)],
                        [edge_weight], reference_state_dict=child_agg_sd,
                    )
                else:
                    self.edge_sd = self._aggregate(
                        [
                            prepare_aggregation_upload(
                                child_agg_sd, child_agg_sd, self.dp_fedavg_config
                            ),
                            prepare_aggregation_upload(
                                edge_local_sd, child_agg_sd, self.dp_fedavg_config
                            ),
                        ],
                        [max(1, total_child_weight), edge_weight],
                        reference_state_dict=child_agg_sd,
                    )
                self.last_upload_weight = max(1, total_child_weight + edge_weight)
            else:
                # New mode: edge trains its own local model independently from the incoming shared model,
                # then participates in the same aggregation pool as its child clients.
                edge_node.set_state_dict(self.edge_sd)
                edge_node.local_train()
                edge_loss_info = _evaluate_node_losses(edge_node)
                train_normalised_losses[int(self.edge_node_id)] = float(edge_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.edge_node_id)] = float(edge_loss_info.get("actual_loss", float("nan")))
                edge_local_sd = edge_node.get_state_dict()
                edge_weight = max(1, edge_node.num_train_samples())
                self.edge_sd = self._aggregate(
                    client_uploads
                    + [
                        prepare_aggregation_upload(
                            edge_local_sd, self.edge_sd, self.dp_fedavg_config
                        )
                    ],
                    weights + [edge_weight],
                    reference_state_dict=self.edge_sd,
                )
                self.last_upload_weight = max(1, total_child_weight + edge_weight)
        else:
            self.edge_sd = clone_state_dict(child_agg_sd)
            self.last_upload_weight = total_child_weight if selected_node_ids is not None else max(1, total_child_weight)

        edge_node.set_state_dict(self.edge_sd)
        # Backward-compatible alias keeps the old name as the normalised loss.
        train_losses = dict(train_normalised_losses)
        self.last_train_losses = train_losses
        self.last_selected_client_ids = list(selected)

        return {
            "edge_id": int(self.edge_node_id),
            "selected_client_ids": list(selected),
            "train_losses": dict(train_losses),
            "train_normalised_losses": dict(train_normalised_losses),
            "train_actual_losses": dict(train_actual_losses),
            "avg_normalised_loss": _safe_average(train_normalised_losses.values()),
            "avg_actual_loss": _safe_average(train_actual_losses.values()),
            # Backward-compatible alias.
            "avg_train_loss": _safe_average(train_normalised_losses.values()),
            "num_train_events": len(train_normalised_losses),
        }


class CloudServer:
    def __init__(
        self,
        cloud_node_id: int,
        init_sd: Dict[str, torch.Tensor],
        dp_fedavg_config: Optional[dict] = None,
    ):
        self.cloud_node_id = int(cloud_node_id)
        self.global_sd = clone_state_dict(init_sd)
        self.last_upload_weight: int = 1
        self.last_train_losses: Dict[int, float] = {}
        self.dp_fedavg_config = dict(dp_fedavg_config or {})
        self.dp_aggregation_calls = 0
        self._dp_generator = self._build_dp_generator()

    def _build_dp_generator(self) -> Optional[torch.Generator]:
        if not bool(self.dp_fedavg_config.get("enabled", False)):
            return None
        generator = torch.Generator(device="cpu")
        seed = self.dp_fedavg_config.get("seed")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(seed) + 1_000_000 + self.cloud_node_id)
        return generator

    def _aggregate(
        self,
        uploads: List[Dict[str, torch.Tensor]],
        weights: List[int],
        reference_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not bool(self.dp_fedavg_config.get("enabled", False)):
            return fedavg_aggregate(uploads, weights)
        weighting = str(self.dp_fedavg_config.get("weighting", "uniform"))
        aggregation_weights = [1] * len(weights) if weighting == "uniform" else weights
        result = dp_fedavg_aggregate(
            clipped_updates=uploads,
            weights=aggregation_weights,
            reference_state_dict=reference_state_dict,
            max_update_norm=float(self.dp_fedavg_config["max_update_norm"]),
            noise_multiplier=float(self.dp_fedavg_config["noise_multiplier"]),
            adjacency=str(self.dp_fedavg_config.get("adjacency", "add_remove")),
            generator=self._dp_generator,
        )
        self.dp_aggregation_calls += 1
        return result

    def aggregate_edges(self, edge_servers: Dict[int, EdgeServer], nodes: Dict[int, HFLSeriesNode], hier_train_mode: str = "post_agg_local_finetune", selected_node_ids: Optional[set[int]] = None):
        hier_train_mode = str(hier_train_mode)
        if hier_train_mode not in VALID_HIER_TRAIN_MODES:
            raise ValueError(f"Unsupported hier_train_mode={hier_train_mode}. Valid options: {sorted(VALID_HIER_TRAIN_MODES)}")

        train_normalised_losses: Dict[int, float] = {}
        train_actual_losses: Dict[int, float] = {}
        if selected_node_ids is not None:
            edge_servers = {eid: server for eid, server in edge_servers.items() if server.last_upload_weight > 0}
        edge_uploads = [
            prepare_aggregation_upload(
                edge_servers[eid].edge_sd,
                self.global_sd,
                self.dp_fedavg_config,
            )
            for eid in edge_servers
        ]
        weights = [max(1, edge_servers[eid].last_upload_weight) for eid in edge_servers]
        tmp_global_sd = self._aggregate(
            edge_uploads, weights, reference_state_dict=self.global_sd
        ) if edge_uploads else clone_state_dict(self.global_sd)
        total_edge_weight = int(sum(weights))

        cloud_node = nodes[self.cloud_node_id]
        if cloud_node.trainable and (selected_node_ids is None or self.cloud_node_id in selected_node_ids):
            if hier_train_mode == "post_agg_local_finetune":
                cloud_node.set_state_dict(tmp_global_sd)
                cloud_node.local_train()
                cloud_loss_info = _evaluate_node_losses(cloud_node)
                train_normalised_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("actual_loss", float("nan")))
                cloud_local_sd = cloud_node.get_state_dict()
                cloud_weight = max(1, cloud_node.num_train_samples())
                if not edge_uploads:
                    self.global_sd = self._aggregate(
                        [prepare_aggregation_upload(cloud_local_sd, tmp_global_sd, self.dp_fedavg_config)],
                        [cloud_weight], reference_state_dict=tmp_global_sd,
                    )
                else:
                    self.global_sd = self._aggregate(
                        [
                            prepare_aggregation_upload(
                                tmp_global_sd, tmp_global_sd, self.dp_fedavg_config
                            ),
                            prepare_aggregation_upload(
                                cloud_local_sd, tmp_global_sd, self.dp_fedavg_config
                            ),
                        ],
                        [max(1, total_edge_weight), cloud_weight],
                        reference_state_dict=tmp_global_sd,
                    )
                self.last_upload_weight = max(1, total_edge_weight + cloud_weight)
            else:
                # New mode: cloud trains its own total-series model independently from the incoming
                # shared model before aggregating with the edge uploads.
                cloud_node.set_state_dict(self.global_sd)
                cloud_node.local_train()
                cloud_loss_info = _evaluate_node_losses(cloud_node)
                train_normalised_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("actual_loss", float("nan")))
                cloud_local_sd = cloud_node.get_state_dict()
                cloud_weight = max(1, cloud_node.num_train_samples())
                self.global_sd = self._aggregate(
                    edge_uploads
                    + [
                        prepare_aggregation_upload(
                            cloud_local_sd, self.global_sd, self.dp_fedavg_config
                        )
                    ],
                    weights + [cloud_weight],
                    reference_state_dict=self.global_sd,
                )
                self.last_upload_weight = max(1, total_edge_weight + cloud_weight)
        else:
            self.global_sd = clone_state_dict(tmp_global_sd)
            self.last_upload_weight = total_edge_weight if selected_node_ids is not None else max(1, total_edge_weight)

        cloud_node.set_state_dict(self.global_sd)
        # Backward-compatible alias keeps the old name as the normalised loss.
        train_losses = dict(train_normalised_losses)
        self.last_train_losses = train_losses

        return {
            "cloud_id": int(self.cloud_node_id),
            "train_losses": dict(train_losses),
            "train_normalised_losses": dict(train_normalised_losses),
            "train_actual_losses": dict(train_actual_losses),
            "avg_normalised_loss": _safe_average(train_normalised_losses.values()),
            "avg_actual_loss": _safe_average(train_actual_losses.values()),
            # Backward-compatible alias.
            "avg_train_loss": _safe_average(train_normalised_losses.values()),
            "num_train_events": len(train_normalised_losses),
        }
