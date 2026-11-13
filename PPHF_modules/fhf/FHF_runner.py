from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union
import random
import time

import numpy as np
import torch

from PPHF_modules.fhf.FHF_models import build_model
from PPHF_modules.fhf.FHF_nf import load_clients_from_csv
from PPHF_modules.fhf.FHF_Client import Client
from PPHF_modules.fhf.FHF_Cloud import Cloud
from PPHF_modules.fhf.FHF_DP import apply_prediction_dp, dp_noise_variance_by_client
from PPHF_modules.fhf.aggregation import clip_client_update, dp_fedavg_aggregate, fedavg_aggregate
from PPHF_modules.fhf.client_sampling import select_clients, validate_sampling_rate, sampling_metadata, fixed_sample_size


@dataclass
class Args:
    csv_path: str = "C:/Users/n12553263/yjzPyR/Datasets/GEFCOM2017/Gef2017_170101.csv"
    partition_col: str = "partition_id"
    series_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"
    truncated: str | None = "2017-01-01"

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
    num_rounds: int = 100
    local_epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"

    recon_method: Union[str, List[str]] = (
        "bu",
        "mint_cov",
        "mint_shrinkage",
        "mint_ols",
        "mint_var",
        "wls_structure",
    )
    ridge: float = 1e-6
    td_eps: float = 1e-8
    seed: int = 42
    validate_p2p: bool = True
    p2p_atol: float = 1e-8
    reject_duplicate_series: bool = False
    torch_num_threads: int = 1

    # Local DP for the base forecasts released into reconciliation. The
    # current implementation supports event replacement adjacency only.
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

    # Optional user-level DP-FedAvg. Disabled by default, preserving the
    # original FHF training path exactly.
    dp_fedavg_enabled: bool = False
    dp_fedavg_max_update_norm: float = 1.0
    dp_fedavg_noise_multiplier: float = 1.0
    dp_fedavg_adjacency: str = "add_remove"
    dp_fedavg_weighting: str = "uniform"
    dp_fedavg_seed: int | None = None


    # Early stopping / stop criteria.
    # From round ``min_rounds`` onward, stop when the absolute change in
    # avg_normalised_loss versus the previous round is smaller than
    # early_stop_tol. Test-set metrics are never used for early stopping.
    early_stop_enabled: bool = True
    early_stop_tol: float = 1e-4
    early_stop_metric: str = "avg_normalised_loss_delta"
    min_rounds: int = 20

    # Logging / return format.
    verbose_rounds: bool = True
    print_every: int = 1
    return_result: bool = False
    sampling_rate: float = 1.0
    sampling_seed: int | None = None
    dp_fedavg_sampling_amplification: bool = True
    sampled_fedavg_training_protocol: bool = False
    sampling_sample_size: int | None = None


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


def _evaluate_client_losses(client: Client) -> Dict[str, float]:
    """Return train losses on normalised and original target scales."""
    if hasattr(client, "evaluate_train_losses"):
        losses = client.evaluate_train_losses()
        normalised = losses.get("normalised_loss", np.nan)
        actual = losses.get("actual_loss", np.nan)
        return {
            "normalised_loss": float(normalised) if np.isfinite(float(normalised)) else float("nan"),
            "actual_loss": float(actual) if np.isfinite(float(actual)) else float("nan"),
        }

    # Fallback for older Client implementations: compute normalised MSE only.
    if client.train_loader is None:
        return {"normalised_loss": float("nan"), "actual_loss": float("nan")}

    model = client.model.to(client.device)
    criterion = client.criterion
    model.eval()

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in client.train_loader:
            x = batch["x"].to(client.device).float()
            y = batch["y"].to(client.device).float()
            pred = model(x)
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            loss = criterion(pred, y)
            n = int(x.shape[0])
            total_loss += float(loss.detach().cpu()) * n
            total_n += n

    normalised = float(total_loss / total_n) if total_n > 0 else float("nan")
    return {"normalised_loss": normalised, "actual_loss": float("nan")}


def _build_early_stop_summary(
    stopped_early: bool,
    stop_reason: str,
    stopped_round: int | None,
    final_avg_normalised_loss: float,
    final_avg_actual_loss: float,
    final_normalised_loss_delta: float,
) -> dict:
    return {
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "stopped_round": None if stopped_round is None else int(stopped_round),
        "final_avg_normalised_loss": float(final_avg_normalised_loss) if np.isfinite(final_avg_normalised_loss) else np.nan,
        "final_avg_actual_loss": float(final_avg_actual_loss) if np.isfinite(final_avg_actual_loss) else np.nan,
        "final_normalised_loss_delta": float(final_normalised_loss_delta) if np.isfinite(final_normalised_loss_delta) else np.nan,
        # Backward-compatible alias for older notebook cells.
        "final_avg_train_loss": float(final_avg_normalised_loss) if np.isfinite(final_avg_normalised_loss) else np.nan,
    }


def run_fhf(args: Args):
    """Run two-layer Federated Hierarchical Forecasting.

    Core reconciliation is intentionally unchanged:
    - S is built by FHF_Cloud.Cloud.build_hierarchy_S
    - P_h = S G_h is computed by FHF_Cloud.Cloud.compute_reconciliation_matrices
    - P2P reconciliation still sends P columns and weighted contributions

    Enhancements in this runner are limited to:
    - training-loss based early stopping
    - training/timing logs
    - an optional result dictionary when args.return_result=True
    """
    timers: Dict[str, float] = {}
    sampling_rate = validate_sampling_rate(getattr(args, "sampling_rate", 1.0))
    configured_sampling_seed = getattr(args, "sampling_seed", None)
    sampling_seed = int(args.seed if configured_sampling_seed is None else configured_sampling_seed)
    sampling_rng = np.random.default_rng(sampling_seed) if sampling_rate < 1.0 else None
    total_t0 = time.perf_counter()
    set_seed(int(args.seed), torch_num_threads=getattr(args, "torch_num_threads", 1))

    setup_t0 = time.perf_counter()

    data_t0 = time.perf_counter()
    client_ids, cid2series, cid2df = load_clients_from_csv(
        csv_path=args.csv_path,
        partition_col=args.partition_col,
        series_col=args.series_col,
        time_col=args.time_col,
        target_col=args.target_col,
        lags=args.lags,
        fh=args.fh,
        truncated=args.truncated,
    )

    clients: Dict[int, Client] = {}
    for cid in client_ids:
        c = Client(cid=cid, series_name=cid2series[cid], df_one=cid2df[cid], args=args)
        c.prepare_data()
        clients[cid] = c
    timers["data_preparation_sec"] = time.perf_counter() - data_t0

    init_t0 = time.perf_counter()
    calibrated_sample_size = getattr(args, "sampling_sample_size", None)
    if calibrated_sample_size is None:
        calibrated_sample_size = fixed_sample_size(len(client_ids), sampling_rate)
    if not isinstance(calibrated_sample_size, int) or not 1 <= calibrated_sample_size <= len(client_ids):
        raise ValueError("sampling_sample_size must be an integer between 1 and the population size.")
    if sampling_rate == 1.0 and calibrated_sample_size != len(client_ids):
        raise ValueError("sampling_sample_size must equal the population size when sampling_rate=1.")
    independent_initialization = (
        (bool(getattr(args, "dp_fedavg_enabled", False)) or bool(getattr(args, "sampled_fedavg_training_protocol", False))) and sampling_rate < 1.0
        and str(getattr(args, "dp_fedavg_weighting", "uniform")).strip().lower() == "uniform"
        and calibrated_sample_size < len(client_ids)
        and bool(getattr(args, "dp_fedavg_sampling_amplification", True))
    )
    if independent_initialization:
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(args.seed))
            init_model = build_model(args).to(args.device)
    else:
        init_model = build_model(args).to(args.device)
    cloud = Cloud(init_model=init_model, client_ids=client_ids)
    cloud.build_hierarchy_S(
        cid2series,
        reject_duplicate_series=bool(
            getattr(args, "reject_duplicate_series", False)
        ),
    )
    methods = cloud.normalize_recon_methods(args.recon_method)
    timers["model_server_initialization_sec"] = time.perf_counter() - init_t0
    timers["setup_sec"] = time.perf_counter() - setup_t0

    print(f"Loaded {len(client_ids)} clients.")
    print("Recon methods:", methods)

    round_logs: List[dict] = []
    stopped_early = False
    stop_reason = "max_rounds_reached"
    stopped_round = None
    final_avg_normalised_loss = np.nan
    final_avg_actual_loss = np.nan
    final_normalised_loss_delta = np.nan
    previous_avg_normalised_loss = np.nan
    dp_fedavg_enabled = bool(getattr(args, "dp_fedavg_enabled", False))
    dp_fedavg_weighting = str(
        getattr(args, "dp_fedavg_weighting", "uniform")
    ).strip().lower()
    if dp_fedavg_weighting not in {"uniform", "sample"}:
        raise ValueError("dp_fedavg_weighting must be 'uniform' or 'sample'.")
    amplified = (dp_fedavg_enabled and sampling_rate < 1.0 and dp_fedavg_weighting == "uniform"
                 and calibrated_sample_size < len(client_ids)
                 and bool(getattr(args, "dp_fedavg_sampling_amplification", True)))
    dp_fedavg_generator = None
    if dp_fedavg_enabled:
        dp_fedavg_generator = torch.Generator(device="cpu")
        configured_seed = getattr(args, "dp_fedavg_seed", None)
        if configured_seed is None:
            dp_fedavg_generator.seed()
        else:
            dp_fedavg_generator.manual_seed(int(configured_seed))

    fl_t0 = time.perf_counter()
    for rnd in range(int(args.num_rounds)):
        round_reference_sd = (
            {
                key: value.detach().cpu().clone()
                for key, value in cloud.global_sd.items()
            }
            if dp_fedavg_enabled
            else cloud.global_sd
        )
        client_uploads, weights = [], []
        normalised_losses: List[float] = []
        actual_losses: List[float] = []

        selected_client_ids = select_clients(client_ids, sampling_rate, sampling_rng, sample_size=calibrated_sample_size)
        for cid in selected_client_ids:
            client = clients[cid]
            client.set_state_dict(cloud.global_sd)
            if independent_initialization:
                # The sampled Gaussian proof needs a client's clipped update
                # to depend on its data and the incoming public model, rather
                # than on other selected clients or private optimizer history.
                client.opt.state.clear()
                cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed((int(args.seed) + 1_000_003 * rnd + 9_176 * int(cid)) % (2**63))
                    client.local_train()
                    loss_info = _evaluate_client_losses(client)
            else:
                client.local_train()
                loss_info = _evaluate_client_losses(client)
            normalised_loss = float(loss_info.get("normalised_loss", np.nan))
            actual_loss = float(loss_info.get("actual_loss", np.nan))
            if np.isfinite(normalised_loss):
                normalised_losses.append(normalised_loss)
            if np.isfinite(actual_loss):
                actual_losses.append(actual_loss)
            local_state = client.get_state_dict()
            if dp_fedavg_enabled:
                # Simulated client boundary: form and clip the model delta
                # locally.  The server-side collection never receives the
                # client's absolute locally trained model.
                client_uploads.append(
                    clip_client_update(
                        state_dict=local_state,
                        reference_state_dict=round_reference_sd,
                        max_update_norm=float(args.dp_fedavg_max_update_norm),
                    )
                )
            else:
                client_uploads.append(local_state)
            weights.append(max(1, client.num_train_samples()))

        if dp_fedavg_enabled:
            aggregation_weights = (
                [1] * len(weights)
                if dp_fedavg_weighting == "uniform"
                else weights
            )
            cloud.global_sd = dp_fedavg_aggregate(
                clipped_updates=client_uploads,
                weights=aggregation_weights,
                reference_state_dict=round_reference_sd,
                max_update_norm=float(args.dp_fedavg_max_update_norm),
                noise_multiplier=float(args.dp_fedavg_noise_multiplier),
                adjacency=str(args.dp_fedavg_adjacency),
                generator=dp_fedavg_generator,
                sensitivity_multiplier=2.0 if amplified and args.dp_fedavg_adjacency == "add_remove" else 1.0,
            )
        else:
            cloud.global_sd = fedavg_aggregate(client_uploads, weights)
        avg_normalised_loss = float(np.nanmean(normalised_losses)) if normalised_losses else np.nan
        avg_actual_loss = float(np.nanmean(actual_losses)) if actual_losses else np.nan
        normalised_loss_delta = (
            abs(avg_normalised_loss - previous_avg_normalised_loss)
            if np.isfinite(avg_normalised_loss) and np.isfinite(previous_avg_normalised_loss)
            else np.nan
        )
        final_avg_normalised_loss = avg_normalised_loss
        final_avg_actual_loss = avg_actual_loss
        final_normalised_loss_delta = normalised_loss_delta

        should_stop = (
            bool(getattr(args, "early_stop_enabled", True))
            and (rnd + 1) >= int(getattr(args, "min_rounds", 20))
            and np.isfinite(normalised_loss_delta)
            and normalised_loss_delta < float(getattr(args, "early_stop_tol", 1e-4))
        )

        log_row = {
            "round": rnd + 1,
            "sampling_rate": sampling_rate,
            "selected_client_ids": list(selected_client_ids),
            "num_selected_clients": len(selected_client_ids),
            "num_total_clients": len(client_ids),
            "realized_sampling_rate": len(selected_client_ids) / len(client_ids),
            "num_rounds": int(args.num_rounds),
            "avg_normalised_loss": avg_normalised_loss,
            "avg_actual_loss": avg_actual_loss,
            "normalised_loss_delta": normalised_loss_delta,
            # Backward-compatible alias.
            "avg_train_loss": avg_normalised_loss,
            "n_train_nodes": len(normalised_losses),
            "early_stop_enabled": bool(getattr(args, "early_stop_enabled", True)),
            "early_stop_tol": float(getattr(args, "early_stop_tol", 1e-4)),
            "early_stop_metric": str(getattr(args, "early_stop_metric", "avg_normalised_loss_delta")),
            "stopped": bool(should_stop),
        }
        round_logs.append(log_row)

        if bool(getattr(args, "verbose_rounds", True)) and _should_print(rnd, int(args.num_rounds), int(getattr(args, "print_every", 1))):
            print(
                f"[Round {rnd + 1}/{args.num_rounds}] "
                f"avg_normalised_loss={avg_normalised_loss:.6g}, "
                f"avg_actual_loss={avg_actual_loss:.6g}, "
                f"delta={normalised_loss_delta:.6g}, stopped={bool(should_stop)}"
            )

        if should_stop:
            stopped_early = True
            stopped_round = rnd + 1
            stop_reason = (
                f"abs(avg_normalised_loss - previous_avg_normalised_loss) < "
                f"{float(getattr(args, 'early_stop_tol', 1e-4))}"
            )
            print(
                f"[Early Stop] normalised loss delta {normalised_loss_delta:.6g} < "
                f"{float(getattr(args, 'early_stop_tol', 1e-4))} at round {rnd + 1}"
            )
            break

        previous_avg_normalised_loss = avg_normalised_loss

    timers["fl_training_sec"] = time.perf_counter() - fl_t0

    residuals_t0 = time.perf_counter()
    resid_by_cid: Dict[int, np.ndarray] = {}
    for cid in client_ids:
        clients[cid].set_state_dict(cloud.global_sd)
        clients[cid].compute_base_forecast_and_residuals()
        resid_by_cid[cid] = clients[cid].residual_samples
    dp_summary = apply_prediction_dp(clients, args)
    timers["residual_collection_sec"] = time.perf_counter() - residuals_t0

    recon_matrix_t0 = time.perf_counter()
    dp_variance_by_cid = (
        dp_noise_variance_by_client(clients)
        if bool(getattr(args, "dp_enabled", False))
        and bool(getattr(args, "dp_aware_reconciliation", False))
        else None
    )
    Ps, Gs = cloud.compute_reconciliation_matrices(
        methods=methods,
        resid_by_cid=resid_by_cid,
        ridge=args.ridge,
        td_eps=args.td_eps,
        dp_variance_by_cid=dp_variance_by_cid,
    )
    timers["recon_matrix_sec"] = time.perf_counter() - recon_matrix_t0

    p2p_t0 = time.perf_counter()
    for cid in client_ids:
        clients[cid].reset_reconciled_forecasts()

    # Simulated private P2P reconciliation:
    # cloud sends each sender-client only its own P column(s),
    # sender multiplies locally with its base forecasts,
    # then sends weighted elements to every recipient.
    for method, P_tensor in Ps.items():
        # P_tensor shape: (H, n_series, n_series)
        for j, cid_j in enumerate(client_ids):
            P_col = P_tensor[:, :, j].T  # (n_series, H)
            clients[cid_j].set_P_column(P_col)

        for j, cid_j in enumerate(client_ids):
            outgoing = clients[cid_j].build_weighted_contributions()  # (n_series, T_test, H)
            for i, cid_i in enumerate(client_ids):
                clients[cid_i].add_reconciled_contribution(outgoing[i], method=method)

        if bool(getattr(args, "validate_p2p", True)):
            base_stack = np.stack(
                [clients[cid].base_forecast for cid in client_ids], axis=0
            ).astype(np.float64)  # (n_series, T_test, H)
            rec_central = np.empty_like(base_stack)
            for h in range(base_stack.shape[2]):
                rec_central[:, :, h] = P_tensor[h] @ base_stack[:, :, h]
            for i, cid in enumerate(client_ids):
                diff = np.max(
                    np.abs(clients[cid].reconciled_forecasts[method] - rec_central[i])
                )
                if diff > float(getattr(args, "p2p_atol", 1e-8)):
                    raise AssertionError(
                        f"P2P reconciliation mismatch for method={method}, cid={cid}, max_diff={diff}"
                    )

    if len(methods) == 1:
        only = methods[0]
        for cid in client_ids:
            clients[cid].reconciled_forecast = clients[cid].reconciled_forecasts[only].copy()

    timers["p2p_reconcile_sec"] = time.perf_counter() - p2p_t0
    timers["reconciliation_sec"] = (
        timers.get("residual_collection_sec", 0.0)
        + timers.get("recon_matrix_sec", 0.0)
        + timers.get("p2p_reconcile_sec", 0.0)
    )
    timers["evaluation_sec"] = 0.0
    timers["total_sec"] = time.perf_counter() - total_t0

    early_stop = _build_early_stop_summary(
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        stopped_round=stopped_round,
        final_avg_normalised_loss=final_avg_normalised_loss,
        final_avg_actual_loss=final_avg_actual_loss,
        final_normalised_loss_delta=final_normalised_loss_delta,
    )

    # Keep the original three-object API usable while also exposing logs.
    cloud.timings = dict(timers)
    cloud.round_logs = list(round_logs)
    cloud.early_stop = dict(early_stop)
    cloud.methods = list(methods)
    cloud.Ps = Ps
    cloud.Gs = Gs
    cloud.dp_summary = dict(dp_summary)
    cloud.dp_aware_reconciliation = dp_variance_by_cid is not None
    cloud.sampling_summary = sampling_metadata(
        sampling_rate, sampling_seed, len(client_ids), scope="all_fhf_clients_per_round"
    )
    cloud.sampling_summary["public_sample_size"] = calibrated_sample_size
    cloud.dp_fedavg_summary = {
        **cloud.sampling_summary,
        "sampling_amplification_eligible": amplified,
        "local_training_state": "round_local_optimizer_and_client_rng" if independent_initialization else "historical",
        "sample_size": len(selected_client_ids) if round_logs else 0,
        "noise_sensitivity_multiplier": 2.0 if amplified and args.dp_fedavg_adjacency == "add_remove" else 1.0,
        "enabled": dp_fedavg_enabled,
        "aggregation": "dp_fedavg" if dp_fedavg_enabled else "fedavg",
        "max_update_norm": (
            float(args.dp_fedavg_max_update_norm) if dp_fedavg_enabled else None
        ),
        "noise_multiplier": (
            float(args.dp_fedavg_noise_multiplier) if dp_fedavg_enabled else None
        ),
        "adjacency": (
            str(args.dp_fedavg_adjacency) if dp_fedavg_enabled else None
        ),
        "weighting": dp_fedavg_weighting if dp_fedavg_enabled else "sample",
        "aggregation_calls": len(round_logs) if dp_fedavg_enabled else 0,
        "participation": ("full" if sampling_rate == 1.0 else "partial") if dp_fedavg_enabled else None,
        "clipping_location": "client" if dp_fedavg_enabled else None,
        "uploaded_payload": (
            "clipped_model_update" if dp_fedavg_enabled else "local_model_state"
        ),
        "noise_location": "central_server" if dp_fedavg_enabled else None,
        "privacy_model": "central_user_level_dp" if dp_fedavg_enabled else None,
        "server_receives_unclipped_updates": False if dp_fedavg_enabled else None,
        "secure_aggregation_used": False if dp_fedavg_enabled else None,
        "seed_is_fixed_for_research": (
            getattr(args, "dp_fedavg_seed", None) is not None
            if dp_fedavg_enabled
            else False
        ),
    }
    if bool(getattr(args, "return_result", False)):
        return {
            "clients": clients,
            "cloud": cloud,
            "cid2series": cid2series,
            "methods": methods,
            "round_logs": round_logs,
            "timings": timers,
            "early_stop": early_stop,
            "dp": dp_summary,
            "sampling": dict(cloud.sampling_summary),
            "dp_fedavg": dict(cloud.dp_fedavg_summary),
            "stopped_early": early_stop["stopped_early"],
            "stop_reason": early_stop["stop_reason"],
            "stopped_round": early_stop["stopped_round"],
            "final_avg_normalised_loss": early_stop["final_avg_normalised_loss"],
            "final_avg_actual_loss": early_stop["final_avg_actual_loss"],
            "final_normalised_loss_delta": early_stop["final_normalised_loss_delta"],
            # Backward-compatible alias.
            "final_avg_train_loss": early_stop["final_avg_train_loss"],
        }

    return clients, cloud, cid2series
