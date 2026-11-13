from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import random

import numpy as np
import torch

from PPHF_modules.fhf.FHF_models import build_model
from PPHF_modules.fhf.FHF_nf import load_clients_from_csv
from PPHF_modules.fhf.FHF_DP import apply_prediction_dp, dp_noise_variance_by_client
from PPHF_modules.fhf.fl_timing import StageTimers
from PPHF_modules.fhf.HFL_topology import infer_hierarchy_layout, resolve_edge_trainable_flags, resolve_edge_aggregation_rounds
from PPHF_modules.fhf.HFL_servers import HFLSeriesNode, EdgeServer, CloudServer, clone_state_dict, VALID_HIER_TRAIN_MODES
from PPHF_modules.fhf.HFL_recon import ReconciliationCoordinator, PeerNetwork
from PPHF_modules.fhf.client_sampling import select_clients, validate_sampling_rate, sampling_metadata


@dataclass
class HFLArgs:
    csv_path: str = "C:/Users/n12553263/yjzPyR/Datasets/GEFCOM2017/Gef2017_170101.csv"
    partition_col: str = "partition_id"
    series_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"
    truncated: str | None = None

    lags: int = 48
    fh: int = 1
    test_ratio: float = 0.2
    batch_size: int = 256

    model_type: str = "lstm"
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    c_in: int = 1
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 3

    device: str = "cpu"
    num_rounds: int = 50
    edge_agg_rounds: int = 1
    edge_agg_rounds_by_edge: Optional[Dict[int, int]] = None
    edge_execution_mode: str = "sequential"
    local_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    client_frac_per_edge: float = 1.0

    edge_trainable: Union[bool, Sequence[int], Dict[int, bool]] = False
    cloud_trainable: bool = False
    hier_train_mode: str = "post_agg_local_finetune"

    # Early stopping is based on training loss only, not test metrics.
    # From round ``min_rounds`` onward, stop when the absolute change in
    # avg_normalised_loss versus the previous round is smaller than
    # early_stop_tol.
    early_stop_enabled: bool = True
    early_stop_tol: float = 1e-4
    early_stop_metric: str = "avg_normalised_loss_delta"
    min_rounds: int = 20

    recon_method: Union[str, Sequence[str]] = (
        "bu",
        "mint_cov",
        "mint_shrinkage",
        "mint_ols",
        "mint_var",
        "wls_structure",
    )
    ridge: float = 1e-6
    td_eps: float = 1e-8
    validate_p2p: bool = True
    p2p_atol: float = 1e-7
    reject_duplicate_series: bool = False

    # Local DP for the base forecasts released into reconciliation.
    dp_enabled: bool = False
    dp_privacy_level: str = "event"
    dp_mechanism: str = "laplace"
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_bounds_source: str = "train_predictions"
    dp_bounds_scope: str = "per_client_time_slot"
    dp_slots_per_day: int = 48
    dp_lower: object = None
    dp_upper: object = None
    dp_lower_quantile: float = 0.01
    dp_upper_quantile: float = 0.99
    dp_iqr_margin: float = 0.0
    dp_physical_lower: float | None = 0.0
    dp_bound_margin: object = 0.0
    dp_seed: int | None = None
    dp_aware_reconciliation: bool = False

    # Optional user-level DP-FedAvg at every enabled hierarchy aggregation.
    # Disabled by default, preserving HFF/HFFPS behavior.
    dp_fedavg_enabled: bool = False
    dp_fedavg_max_update_norm: float = 1.0
    dp_fedavg_noise_multiplier: float = 1.0
    dp_fedavg_adjacency: str = "add_remove"
    dp_fedavg_weighting: str = "uniform"
    dp_fedavg_seed: int | None = None

    seed: int = 42
    torch_num_threads: int = 1

    verbose_cloud_rounds: bool = True
    verbose_edge_rounds: bool = False
    print_every: int = 1
    # Global selection among trainable series nodes once per cloud round.
    # The legacy per-edge fraction remains available when sampling_rate=1.
    sampling_rate: float = 1.0
    sampling_seed: int | None = None


def set_seed(seed: int = 42, torch_num_threads: int | None = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_num_threads is not None and int(torch_num_threads) > 0:
        torch.set_num_threads(int(torch_num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def _should_print(rnd: int, total: int, every: int) -> bool:
    every = max(1, int(every))
    return ((rnd + 1) % every == 0) or ((rnd + 1) == total)


def _safe_mean(values) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _early_stop_value(avg_train_loss: float, metric: str) -> float:
    metric = str(metric).strip().lower()
    if metric in {"train_loss", "loss", "mse", "train_mse"}:
        return float(avg_train_loss)
    if metric in {"train_rmse", "rmse"}:
        return float(np.sqrt(avg_train_loss)) if np.isfinite(avg_train_loss) else float("nan")
    raise ValueError(
        f"Unsupported early_stop_metric={metric}. "
        "Use 'train_loss'/'train_mse' or 'train_rmse'."
    )


def build_hfl_nodes(args: HFLArgs):
    node_ids, cid2series, cid2df = load_clients_from_csv(
        csv_path=args.csv_path,
        partition_col=args.partition_col,
        series_col=args.series_col,
        time_col=args.time_col,
        target_col=args.target_col,
        lags=args.lags,
        fh=args.fh,
        truncated=args.truncated,
    )

    layout = infer_hierarchy_layout(node_ids=node_ids, cid2series=cid2series)
    edge_train_flags = resolve_edge_trainable_flags(layout.edge_ids, args.edge_trainable)

    nodes: Dict[int, HFLSeriesNode] = {}
    for nid in layout.node_ids:
        role = layout.role_by_id[nid]
        # A terminal edge is also the leaf series after duplicate parent/leaf
        # pairs have been removed, so it must perform local training once.
        trainable = (
            True
            if role == "client" or nid in layout.terminal_edge_ids
            else (edge_train_flags[nid] if role == "edge" else bool(args.cloud_trainable))
        )
        node = HFLSeriesNode(
            cid=nid,
            series_name=cid2series[nid],
            df_one=cid2df[nid],
            args=args,
            role=role,
            parent_id=layout.parent_by_id[nid],
            trainable=trainable,
        )
        node.prepare_data()
        node.child_ids = list(layout.children_by_id[nid])
        nodes[nid] = node

    return nodes, layout, cid2series


def run_hfl_timed(args: HFLArgs):
    sampling_rate = validate_sampling_rate(getattr(args, "sampling_rate", 1.0))
    legacy_fraction = validate_sampling_rate(getattr(args, "client_frac_per_edge", 1.0))
    if sampling_rate < 1.0 and legacy_fraction != 1.0:
        raise ValueError("Use sampling_rate or client_frac_per_edge for partial participation, not both.")
    configured_sampling_seed = getattr(args, "sampling_seed", None)
    sampling_seed = int(args.seed if configured_sampling_seed is None else configured_sampling_seed)
    sampling_rng = np.random.default_rng(sampling_seed) if sampling_rate < 1.0 else None
    timers = StageTimers()
    set_seed(int(args.seed), torch_num_threads=getattr(args, "torch_num_threads", 1))

    with timers.time_block("data_preparation_sec"):
        nodes, layout, cid2series = build_hfl_nodes(args)
    sampling_population = [nid for nid in layout.node_ids if nodes[nid].trainable]
    sampling_summary = sampling_metadata(
        sampling_rate, sampling_seed, len(sampling_population), scope="all_trainable_hfl_nodes_per_cloud_round"
    )
    sampling_summary["legacy_client_frac_per_edge"] = legacy_fraction

    with timers.time_block("model_server_initialization_sec"):
        init_model = build_model(args).to(args.device)
        init_sd = clone_state_dict(init_model.state_dict())
        dp_fedavg_config = {
            "enabled": bool(getattr(args, "dp_fedavg_enabled", False)),
            "max_update_norm": float(
                getattr(args, "dp_fedavg_max_update_norm", 1.0)
            ),
            "noise_multiplier": float(
                getattr(args, "dp_fedavg_noise_multiplier", 1.0)
            ),
            "adjacency": str(
                getattr(args, "dp_fedavg_adjacency", "add_remove")
            ),
            "weighting": str(
                getattr(args, "dp_fedavg_weighting", "uniform")
            ).strip().lower(),
            "seed": getattr(args, "dp_fedavg_seed", None),
        }
        if dp_fedavg_config["weighting"] not in {"uniform", "sample"}:
            raise ValueError("dp_fedavg_weighting must be 'uniform' or 'sample'.")
        edge_servers: Dict[int, EdgeServer] = {
            eid: EdgeServer(
                edge_node_id=eid,
                child_client_ids=layout.edge_to_clients[eid],
                init_sd=init_sd,
                dp_fedavg_config=dp_fedavg_config,
                child_edge_ids=layout.edge_to_edges[eid],
            )
            for eid in layout.edge_ids
        }
        cloud_server = CloudServer(
            cloud_node_id=layout.cloud_id,
            init_sd=init_sd,
            dp_fedavg_config=dp_fedavg_config,
        )
        edge_rounds = resolve_edge_aggregation_rounds(layout.edge_ids, int(args.edge_agg_rounds), args.edge_agg_rounds_by_edge)
        methods: List[str] = []

    timers.add(
        "setup_sec",
        timers.values.get("data_preparation_sec", 0.0)
        + timers.values.get("model_server_initialization_sec", 0.0),
    )

    round_logs: List[dict] = []
    edge_round_logs: List[dict] = []
    rng = np.random.default_rng(int(args.seed))
    execution_mode = str(getattr(args, "edge_execution_mode", "sequential")).lower()
    if execution_mode not in {"sequential", "synchronous"}:
        raise ValueError(f"Unsupported edge_execution_mode={execution_mode}. Use 'sequential' or 'synchronous'.")
    hier_train_mode = str(getattr(args, "hier_train_mode", "post_agg_local_finetune")).lower()
    if hier_train_mode not in VALID_HIER_TRAIN_MODES:
        raise ValueError(f"Unsupported hier_train_mode={hier_train_mode}. Valid options: {sorted(VALID_HIER_TRAIN_MODES)}")

    early_stop_enabled = bool(getattr(args, "early_stop_enabled", True))
    early_stop_tol = float(getattr(args, "early_stop_tol", 1e-4))
    early_stop_metric = str(getattr(args, "early_stop_metric", "avg_normalised_loss_delta")).strip().lower()
    min_rounds = max(1, int(getattr(args, "min_rounds", 20)))

    stopped_early = False
    stop_reason = "max_rounds_reached"
    stopped_round = None
    final_avg_normalised_loss = float("nan")
    final_avg_actual_loss = float("nan")
    final_normalised_loss_delta = float("nan")
    previous_avg_normalised_loss = float("nan")
    # Lower edges must finish before their parent reuses their model.
    bottom_up_edge_ids = sorted(
        layout.edge_ids,
        key=lambda eid: layout.path_by_id[eid].count("/"),
        reverse=True,
    )

    with timers.time_block("fl_training_sec"):
        for rnd in range(int(args.num_rounds)):
            selected_ids = select_clients(sampling_population, sampling_rate, sampling_rng)
            selected_node_ids = set(selected_ids) if sampling_rate < 1.0 else None
            normalised_losses_by_node: Dict[int, List[float]] = {}
            actual_losses_by_node: Dict[int, List[float]] = {}
            actual_trained_ids: set[int] = set()

            def _record_train_losses(loss_dict: Dict[int, float], actual_loss_dict: Dict[int, float] | None = None) -> None:
                actual_trained_ids.update(int(nid) for nid in loss_dict)
                actual_loss_dict = {} if actual_loss_dict is None else actual_loss_dict
                for raw_nid, raw_loss in loss_dict.items():
                    loss = float(raw_loss)
                    if np.isfinite(loss):
                        nid = int(raw_nid)
                        normalised_losses_by_node.setdefault(nid, []).append(loss)
                for raw_nid, raw_loss in actual_loss_dict.items():
                    loss = float(raw_loss)
                    if np.isfinite(loss):
                        nid = int(raw_nid)
                        actual_losses_by_node.setdefault(nid, []).append(loss)

            for eid in layout.edge_ids:
                edge_servers[eid].sync_from_cloud(cloud_server.global_sd)

            if execution_mode == "sequential":
                for eid in bottom_up_edge_ids:
                    for edge_round_idx in range(edge_rounds[eid]):
                        edge_summary = edge_servers[eid].run_one_edge_round(
                            nodes=nodes,
                            client_frac=float(args.client_frac_per_edge),
                            rng=rng,
                            hier_train_mode=hier_train_mode,
                            child_edge_servers=edge_servers,
                            selected_node_ids=selected_node_ids,
                        )
                        _record_train_losses(edge_summary.get("train_normalised_losses", edge_summary.get("train_losses", {})), edge_summary.get("train_actual_losses", {}))
                        edge_round_logs.append({
                            "cloud_round": rnd + 1,
                            "num_cloud_rounds": int(args.num_rounds),
                            "edge_id": int(eid),
                            "edge_round": edge_round_idx + 1,
                            "num_edge_rounds_for_edge": int(edge_rounds[eid]),
                            "edge_upload_weight": int(edge_servers[eid].last_upload_weight),
                            "edge_execution_mode": execution_mode,
                            "hier_train_mode": hier_train_mode,
                            "avg_normalised_loss": float(edge_summary.get("avg_normalised_loss", edge_summary.get("avg_train_loss", float("nan")))),
                            "avg_actual_loss": float(edge_summary.get("avg_actual_loss", float("nan"))),
                            # Backward-compatible alias.
                            "avg_train_loss": float(edge_summary.get("avg_normalised_loss", edge_summary.get("avg_train_loss", float("nan")))),
                            "num_train_events": int(edge_summary.get("num_train_events", 0)),
                            "selected_client_ids": list(edge_summary.get("selected_client_ids", [])),
                        })
                        if bool(args.verbose_edge_rounds) and _should_print(rnd, int(args.num_rounds), int(args.print_every)):
                            print(
                                f"[Cloud Round {rnd+1}/{args.num_rounds} | Edge {eid} "
                                f"Round {edge_round_idx+1}/{edge_rounds[eid]}] "
                                f"avg_normalised_loss={edge_summary.get('avg_normalised_loss', edge_summary.get('avg_train_loss', float('nan'))):.6g}, "
                                f"avg_actual_loss={edge_summary.get('avg_actual_loss', float('nan')):.6g}"
                            )
            else:
                max_edge_rounds = max(edge_rounds.values()) if edge_rounds else 0
                for edge_subround_idx in range(max_edge_rounds):
                    for eid in bottom_up_edge_ids:
                        if edge_subround_idx >= edge_rounds[eid]:
                            continue
                        edge_summary = edge_servers[eid].run_one_edge_round(
                            nodes=nodes,
                            client_frac=float(args.client_frac_per_edge),
                            rng=rng,
                            hier_train_mode=hier_train_mode,
                            child_edge_servers=edge_servers,
                            selected_node_ids=selected_node_ids,
                        )
                        _record_train_losses(edge_summary.get("train_normalised_losses", edge_summary.get("train_losses", {})), edge_summary.get("train_actual_losses", {}))
                        edge_round_logs.append({
                            "cloud_round": rnd + 1,
                            "num_cloud_rounds": int(args.num_rounds),
                            "edge_id": int(eid),
                            "edge_round": edge_subround_idx + 1,
                            "num_edge_rounds_for_edge": int(edge_rounds[eid]),
                            "edge_upload_weight": int(edge_servers[eid].last_upload_weight),
                            "edge_execution_mode": execution_mode,
                            "hier_train_mode": hier_train_mode,
                            "avg_normalised_loss": float(edge_summary.get("avg_normalised_loss", edge_summary.get("avg_train_loss", float("nan")))),
                            "avg_actual_loss": float(edge_summary.get("avg_actual_loss", float("nan"))),
                            # Backward-compatible alias.
                            "avg_train_loss": float(edge_summary.get("avg_normalised_loss", edge_summary.get("avg_train_loss", float("nan")))),
                            "num_train_events": int(edge_summary.get("num_train_events", 0)),
                            "selected_client_ids": list(edge_summary.get("selected_client_ids", [])),
                        })
                        if bool(args.verbose_edge_rounds) and _should_print(rnd, int(args.num_rounds), int(args.print_every)):
                            print(
                                f"[Cloud Round {rnd+1}/{args.num_rounds} | Edge {eid} "
                                f"Round {edge_subround_idx+1}/{edge_rounds[eid]}] "
                                f"avg_normalised_loss={edge_summary.get('avg_normalised_loss', edge_summary.get('avg_train_loss', float('nan'))):.6g}, "
                                f"avg_actual_loss={edge_summary.get('avg_actual_loss', float('nan')):.6g}"
                            )

            root_edge_servers = {
                eid: edge_servers[eid] for eid in layout.root_edge_ids
            }
            cloud_summary = cloud_server.aggregate_edges(edge_servers=root_edge_servers, nodes=nodes, hier_train_mode=hier_train_mode, selected_node_ids=selected_node_ids)
            _record_train_losses(cloud_summary.get("train_normalised_losses", cloud_summary.get("train_losses", {})), cloud_summary.get("train_actual_losses", {}))

            node_avg_normalised_losses = {int(nid): _safe_mean(vals) for nid, vals in normalised_losses_by_node.items()}
            node_avg_actual_losses = {int(nid): _safe_mean(vals) for nid, vals in actual_losses_by_node.items()}
            avg_normalised_loss = _safe_mean(node_avg_normalised_losses.values())
            avg_actual_loss = _safe_mean(node_avg_actual_losses.values())
            normalised_loss_delta = (
                abs(avg_normalised_loss - previous_avg_normalised_loss)
                if np.isfinite(avg_normalised_loss) and np.isfinite(previous_avg_normalised_loss)
                else float("nan")
            )
            final_avg_normalised_loss = avg_normalised_loss
            final_avg_actual_loss = avg_actual_loss
            final_normalised_loss_delta = normalised_loss_delta

            stop_this_round = (
                early_stop_enabled
                and (rnd + 1) >= min_rounds
                and np.isfinite(normalised_loss_delta)
                and normalised_loss_delta < early_stop_tol
            )

            round_logs.append({
                "sampling_rate": sampling_rate,
                "scheduled_node_ids": list(selected_ids),
                "selected_node_ids": [nid for nid in layout.node_ids if nid in actual_trained_ids],
                "num_selected_nodes": len(actual_trained_ids),
                "sampling_population_size": len(sampling_population),
                "realized_sampling_rate": len(actual_trained_ids) / len(sampling_population) if sampling_population else 0.0,
                "cloud_round": rnd + 1,
                "num_cloud_rounds": int(args.num_rounds),
                "edge_agg_rounds_by_edge": {int(eid): int(edge_rounds[eid]) for eid in layout.edge_ids},
                "edge_upload_weights": {int(eid): int(edge_servers[eid].last_upload_weight) for eid in layout.edge_ids},
                "cloud_upload_weight": int(cloud_server.last_upload_weight),
                "edge_execution_mode": execution_mode,
                "hier_train_mode": hier_train_mode,
                "avg_normalised_loss": float(avg_normalised_loss),
                "avg_actual_loss": float(avg_actual_loss),
                "normalised_loss_delta": float(normalised_loss_delta),
                # Backward-compatible alias.
                "avg_train_loss": float(avg_normalised_loss),
                "early_stop_metric": early_stop_metric,
                "early_stop_value": float(normalised_loss_delta),
                "early_stop_tol": float(early_stop_tol),
                "early_stop_enabled": bool(early_stop_enabled),
                "stopped": bool(stop_this_round),
                "num_train_nodes": int(len(node_avg_normalised_losses)),
                "node_normalised_losses": dict(node_avg_normalised_losses),
                "node_actual_losses": dict(node_avg_actual_losses),
                # Backward-compatible alias.
                "node_train_losses": dict(node_avg_normalised_losses),
                "cloud_normalised_loss": float(cloud_summary.get("avg_normalised_loss", cloud_summary.get("avg_train_loss", float("nan")))),
                "cloud_actual_loss": float(cloud_summary.get("avg_actual_loss", float("nan"))),
                # Backward-compatible alias.
                "cloud_train_loss": float(cloud_summary.get("avg_normalised_loss", cloud_summary.get("avg_train_loss", float("nan")))),
            })

            if bool(args.verbose_cloud_rounds) and (_should_print(rnd, int(args.num_rounds), int(args.print_every)) or stop_this_round):
                print(
                    f"[Cloud Round {rnd+1}/{args.num_rounds}] "
                    f"avg_normalised_loss={avg_normalised_loss:.6g}, "
                    f"avg_actual_loss={avg_actual_loss:.6g}, "
                    f"delta={normalised_loss_delta:.6g}, stopped={bool(stop_this_round)}"
                )

            if stop_this_round:
                stopped_early = True
                stopped_round = rnd + 1
                stop_reason = f"abs(avg_normalised_loss - previous_avg_normalised_loss) < {early_stop_tol}"
                if bool(args.verbose_cloud_rounds):
                    print(
                        f"[Early Stop] normalised loss delta {normalised_loss_delta:.6g} < "
                        f"{early_stop_tol} at round {rnd+1}"
                    )
                break

            previous_avg_normalised_loss = avg_normalised_loss

    for nid in layout.node_ids:
        nodes[nid].set_state_dict(cloud_server.global_sd)

    residuals_by_node: Dict[int, np.ndarray] = {}
    with timers.time_block("residual_collection_sec"):
        for nid in layout.node_ids:
            nodes[nid].compute_base_forecast_and_residuals()
            residuals_by_node[nid] = np.asarray(nodes[nid].residual_samples, dtype=np.float64)
        dp_summary = apply_prediction_dp(nodes, args)

    with timers.time_block("recon_matrix_sec"):
        coordinator = ReconciliationCoordinator(args=args, node_ids=layout.node_ids, cid2series=cid2series)
        methods = coordinator.normalize_recon_methods(args.recon_method)
        dp_variance_by_node = (
            dp_noise_variance_by_client(nodes)
            if bool(getattr(args, "dp_enabled", False))
            and bool(getattr(args, "dp_aware_reconciliation", False))
            else None
        )
        Ps, Gs = coordinator.compute_reconciliation_matrices(
            methods=methods,
            residuals_by_node=residuals_by_node,
            ridge=float(args.ridge),
            td_eps=float(args.td_eps),
            dp_variance_by_node=dp_variance_by_node,
        )

    with timers.time_block("p2p_reconcile_sec"):
        peer_net = PeerNetwork(node_ids=layout.node_ids)
        peer_net.distributed_reconcile(
            nodes=nodes,
            coordinator=coordinator,
            methods=methods,
            validate=bool(args.validate_p2p),
            atol=float(getattr(args, "p2p_atol", 1e-7)),
        )

    timers.add(
        "reconciliation_sec",
        timers.values.get("residual_collection_sec", 0.0)
        + timers.values.get("recon_matrix_sec", 0.0)
        + timers.values.get("p2p_reconcile_sec", 0.0),
    )
    timers.values.setdefault("evaluation_sec", 0.0)
    timers.add(
        "total_sec",
        timers.values.get("setup_sec", 0.0)
        + timers.values.get("fl_training_sec", 0.0)
        + timers.values.get("reconciliation_sec", 0.0)
        + timers.values.get("evaluation_sec", 0.0),
    )

    if len(methods) == 1:
        only = methods[0]
        for nid in layout.node_ids:
            nodes[nid].reconciled_forecast = nodes[nid].reconciled_forecasts[only].copy()

    early_stop_info = {
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "stopped_round": stopped_round,
        "final_avg_normalised_loss": float(final_avg_normalised_loss),
        "final_avg_actual_loss": float(final_avg_actual_loss),
        "final_normalised_loss_delta": float(final_normalised_loss_delta),
        # Backward-compatible alias.
        "final_avg_train_loss": float(final_avg_normalised_loss),
        "final_early_stop_value": float(final_normalised_loss_delta),
        "early_stop_metric": early_stop_metric,
        "early_stop_tol": float(early_stop_tol),
        "min_rounds": int(min_rounds),
    }

    return {
        "nodes": nodes,
        "layout": layout,
        "cid2series": cid2series,
        "edge_servers": edge_servers,
        "cloud_server": cloud_server,
        "reconciliation_coordinator": coordinator,
        "peer_network": peer_net,
        "Ps": Ps,
        "Gs": Gs,
        "cloud_round_logs": round_logs,
        "edge_round_logs": edge_round_logs,
        "timings": timers.values,
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "stopped_round": stopped_round,
        "final_avg_normalised_loss": float(final_avg_normalised_loss),
        "final_avg_actual_loss": float(final_avg_actual_loss),
        "final_normalised_loss_delta": float(final_normalised_loss_delta),
        # Backward-compatible alias.
        "final_avg_train_loss": float(final_avg_normalised_loss),
        "early_stop": early_stop_info,
        "dp": dp_summary,
        "sampling": sampling_summary,
        "dp_fedavg": {
            **sampling_summary,
            "enabled": bool(dp_fedavg_config["enabled"]),
            "aggregation": (
                "dp_fedavg" if dp_fedavg_config["enabled"] else "fedavg"
            ),
            "max_update_norm": (
                dp_fedavg_config["max_update_norm"]
                if dp_fedavg_config["enabled"]
                else None
            ),
            "noise_multiplier": (
                dp_fedavg_config["noise_multiplier"]
                if dp_fedavg_config["enabled"]
                else None
            ),
            "adjacency": (
                dp_fedavg_config["adjacency"]
                if dp_fedavg_config["enabled"]
                else None
            ),
            "weighting": (
                dp_fedavg_config["weighting"]
                if dp_fedavg_config["enabled"]
                else "sample"
            ),
            "participation": (
                (
                    "full"
                    if sampling_rate == 1.0 and legacy_fraction == 1.0
                    else "partial"
                )
                if dp_fedavg_config["enabled"]
                else None
            ),
            "clipping_location": (
                "sender" if dp_fedavg_config["enabled"] else None
            ),
            "uploaded_payload": (
                "clipped_model_update"
                if dp_fedavg_config["enabled"]
                else "local_model_state"
            ),
            "noise_location": (
                "aggregation_server" if dp_fedavg_config["enabled"] else None
            ),
            "privacy_model": (
                "central_user_level_dp"
                if dp_fedavg_config["enabled"]
                else None
            ),
            "server_receives_unclipped_updates": (
                False if dp_fedavg_config["enabled"] else None
            ),
            "secure_aggregation_used": (
                False if dp_fedavg_config["enabled"] else None
            ),
            "edge_aggregation_calls": int(
                sum(server.dp_aggregation_calls for server in edge_servers.values())
            ),
            "cloud_aggregation_calls": int(cloud_server.dp_aggregation_calls),
            "seed_is_fixed_for_research": (
                dp_fedavg_config["seed"] is not None
                if dp_fedavg_config["enabled"]
                else False
            ),
        },
        "dp_aware_reconciliation": dp_variance_by_node is not None,
    }
