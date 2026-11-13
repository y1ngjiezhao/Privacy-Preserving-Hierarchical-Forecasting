from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import argparse
import json
import secrets
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = PROJECT_ROOT
SHARED = FHF_MODULE_DIR

from PPHF_modules.fhf.FHF_HF2 import HFLArgs, run_hfl_with_third_party_reconciliation
from PPHF_modules.fhf.FHF_runner import Args, run_fhf
from PPHF_modules.fhf.training_artifacts import (
    collect_training_artifacts,
    save_or_validate_shared_bounds,
)

from PPHF_modules.dp_fhf.privacy_accounting import (
    gaussian_composition_epsilon, sampled_fedavg_epsilon,
    check_sampling_accountant_dependencies,
)
from PPHF_modules.fhf.client_sampling import sampled_output_dir, validate_sampling_rate, amplification_output_dir


def _test_timestamps(node, count: int):
    time_col = getattr(node.args, "time_col", "ds")
    if time_col in node.df:
        return pd.to_datetime(node.df[time_col].iloc[-count:]).to_numpy()
    return np.arange(count)


def _export_base_forecasts(nodes, cid2series, output_path: Path) -> pd.DataFrame:
    frames = []
    for client_id in sorted(nodes):
        node = nodes[client_id]
        base = np.asarray(node.base_forecast, dtype=np.float64)
        truth = np.asarray(node.test_true, dtype=np.float64)
        if base.ndim == 1:
            base = base[:, None]
        if truth.ndim == 1:
            truth = truth[:, None]
        timestamps = _test_timestamps(node, base.shape[0])
        for horizon_index in range(base.shape[1]):
            frames.append(
                pd.DataFrame(
                    {
                        "node_id": int(client_id),
                        "unique_id": str(cid2series[int(client_id)]),
                        "ds": timestamps,
                        "horizon_index": horizon_index,
                        "y_true": truth[:, horizon_index],
                        "base": base[:, horizon_index],
                    }
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(output_path, index=False)
    return table


def _export_training_artifacts(nodes, cid2series, output_path: Path) -> None:
    artifacts = collect_training_artifacts(nodes=nodes, cid2series=cid2series)
    np.savez_compressed(
        output_path,
        **artifacts,
    )


def _common_args(args, parsed, dp_fedavg_enabled: bool):
    args.csv_path = str(Path(parsed.csv).expanduser().resolve())
    args.truncated = None
    args.model_type = parsed.model
    args.device = parsed.device
    args.fh = parsed.fh
    args.lags = parsed.lags
    args.num_rounds = parsed.num_rounds
    args.local_epochs = parsed.local_epochs
    args.hidden_size = parsed.hidden_size
    args.num_layers = parsed.num_layers
    args.batch_size = parsed.batch_size
    args.seed = parsed.seed
    args.sampling_rate = parsed.sampling_rate
    args.sampling_seed = parsed.sampling_seed
    args.sampling_sample_size = parsed.clients_per_round
    args.torch_num_threads = parsed.torch_num_threads
    args.print_every = parsed.print_every
    args.early_stop_enabled = False
    args.recon_method = ("mint_ols",)
    args.validate_p2p = False
    args.dp_enabled = False
    args.dp_aware_reconciliation = False
    args.dp_fedavg_enabled = bool(dp_fedavg_enabled)
    args.dp_fedavg_max_update_norm = parsed.clip_norm
    args.dp_fedavg_noise_multiplier = parsed.noise_multiplier
    args.dp_fedavg_adjacency = parsed.adjacency
    args.dp_fedavg_weighting = parsed.weighting
    args.dp_fedavg_seed = parsed.dp_seed
    args.dp_fedavg_sampling_amplification = parsed.privacy_accountant == "auto"
    args.sampled_fedavg_training_protocol = (parsed.framework == "fhf" and parsed.sampling_rate < 1.0
                                            and parsed.privacy_accountant == "auto" and parsed.weighting == "uniform")
    args.reject_duplicate_series = parsed.hierarchy_mode == "dedup"
    return args


def _execute_framework(parsed, dp_fedavg_enabled: bool):
    if parsed.framework == "fhf":
        run_args = _common_args(Args(), parsed, dp_fedavg_enabled)
        run_args.return_result = True
        run_args.verbose_rounds = True
        result = run_fhf(run_args)
        return (
            result,
            result["clients"],
            result["cid2series"],
            result["dp_fedavg"],
        )

    run_args = _common_args(HFLArgs(), parsed, dp_fedavg_enabled)
    run_args.edge_trainable = parsed.framework == "hffps"
    run_args.cloud_trainable = parsed.framework == "hffps"
    run_args.hier_train_mode = "self_train_then_aggregate"
    run_args.edge_agg_rounds = 1
    run_args.client_frac_per_edge = 1.0
    run_args.verbose_cloud_rounds = True
    run_args.verbose_edge_rounds = False
    result = run_hfl_with_third_party_reconciliation(run_args)
    return result, result["nodes"], result["cid2series"], result["dp_fedavg"]


def _execute_with_heartbeat(parsed, dp_fedavg_enabled: bool, label: str):
    """Run one long training stage with regular liveness reports."""
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        interval = max(0.01, float(parsed.heartbeat_seconds))
        while not stop.wait(interval):
            elapsed_minutes = (time.monotonic() - started) / 60.0
            print(
                f"[heartbeat] {label} is still running "
                f"({elapsed_minutes:.1f} minutes elapsed).",
                flush=True,
            )

    print(
        f"[stage] Starting {label}. Data preparation may take about 1-2 "
        "minutes before round 1.",
        flush=True,
    )
    reporter = threading.Thread(target=heartbeat, daemon=True)
    reporter.start()
    succeeded = False
    try:
        result = _execute_framework(
            parsed, dp_fedavg_enabled=dp_fedavg_enabled
        )
        succeeded = True
        return result
    finally:
        stop.set()
        reporter.join(timeout=1.0)
        elapsed_minutes = (time.monotonic() - started) / 60.0
        print(
            f"[stage] {'Finished' if succeeded else 'Stopped'} {label} after "
            f"{elapsed_minutes:.1f} minutes.",
            flush=True,
        )


def _performance_comparison(
    private_table: pd.DataFrame, baseline_table: pd.DataFrame
) -> pd.DataFrame:
    keys = ["node_id", "unique_id", "ds", "horizon_index"]
    comparison = private_table.merge(
        baseline_table[keys + ["base"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_dp_fedavg", "_fedavg"),
    )
    rows = []
    for (node_id, unique_id), group in comparison.groupby(
        ["node_id", "unique_id"], sort=True
    ):
        truth = group["y_true"].to_numpy(dtype=np.float64)
        row = {"node_id": int(node_id), "unique_id": str(unique_id)}
        for label, column in [
            ("dp_fedavg", "base_dp_fedavg"),
            ("fedavg", "base_fedavg"),
        ]:
            errors = truth - group[column].to_numpy(dtype=np.float64)
            row[f"{label}_RMSE"] = float(np.sqrt(np.mean(errors * errors)))
            row[f"{label}_MAE"] = float(np.mean(np.abs(errors)))
        row["dp_minus_non_dp_RMSE"] = (
            row["dp_fedavg_RMSE"] - row["fedavg_RMSE"]
        )
        row["dp_minus_non_dp_MAE"] = (
            row["dp_fedavg_MAE"] - row["fedavg_MAE"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run DP-FedAvg-only FHF/HFF/HFFPS with optional client sampling and export "
            "base forecasts."
        )
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--framework", choices=["fhf", "hff", "hffps"], default="fhf"
    )
    parser.add_argument("--output-dir", default=str(DP_EXPERIMENT_DIR / "outputs"))
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fh",
        type=int,
        default=0,
        help="Internal additional-target count; 0/2/5 correspond to horizons 1/3/6.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        choices=[1, 3, 6],
        default=None,
        help="Forecast horizon count. Overrides --fh when supplied.",
    )
    parser.add_argument("--lags", type=int, default=48)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print one training-round report every N rounds.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=60.0,
        help="Print a liveness message this often during long stages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-rate", type=float, default=1.0,
                        help="Uniform sampling without replacement: ceil(rate * N) trainable nodes per round. Default: 1.0.")
    parser.add_argument("--sampling-seed", type=int, default=None,
                        help="Fixed research selection seed. Default: fresh hidden randomness for q<1, --seed for q=1.")
    parser.add_argument("--clients-per-round", type=int, default=None,
                        help="Optional fixed public FHF sample count. Hold it fixed across add/remove neighbors; otherwise calibrates ceil(q*N) from the public roster.")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--adjacency", choices=["add_remove", "replace_one"], default="add_remove"
    )
    parser.add_argument("--weighting", choices=["uniform", "sample"], default="uniform")
    parser.add_argument("--dp-seed", type=int, default=None)
    parser.add_argument("--privacy-accountant", choices=["auto", "conservative"], default="auto",
                        help="auto enables matching fixed-size sampling accounting for uniform FHF runs.")
    parser.add_argument(
        "--hierarchy-mode",
        choices=["duplicates", "dedup"],
        default="duplicates",
        help=(
            "Use a historical hierarchy that may contain identical parent/leaf "
            "series, or require all hierarchy aggregation rows to be unique."
        ),
    )
    parser.add_argument("--accounting-delta", type=float, default=1e-5)
    parser.add_argument(
        "--shared-artifact-dir",
        help=(
            "Directory for the single method-independent training-target bounds "
            "file shared by FHF, HFF, HFFPS, DP, and non-DP runs."
        ),
    )
    parser.add_argument("--dp-slots-per-day", type=int, default=48)
    parser.add_argument("--dp-lower-quantile", type=float, default=0.01)
    parser.add_argument("--dp-upper-quantile", type=float, default=0.99)
    parser.add_argument("--dp-iqr-margin", type=float, default=0.0)
    parser.add_argument("--dp-physical-lower", type=float, default=0.0)
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run ordinary FedAvg with the same seed and export a comparison.",
    )
    parsed = parser.parse_args()
    try:
        validate_sampling_rate(parsed.sampling_rate)
    except ValueError as exc:
        parser.error(str(exc))
    if parsed.clients_per_round is not None and (parsed.clients_per_round <= 0 or parsed.framework != "fhf"):
        parser.error("--clients-per-round requires a positive sample count and --framework fhf.")
    public_sampling_seed = parsed.sampling_seed
    partial = parsed.sampling_rate < 1.0
    fresh_sampling = partial and parsed.sampling_seed is None
    if parsed.sampling_seed is None:
        parsed.sampling_seed = secrets.randbits(63) if partial else parsed.seed
    amplified_path = partial and parsed.privacy_accountant == "auto" and parsed.framework == "fhf" and parsed.weighting == "uniform"
    if amplified_path:
        check_sampling_accountant_dependencies()
    if parsed.forecast_horizon is not None:
        parsed.fh = int(parsed.forecast_horizon) - 1
    if parsed.print_every <= 0:
        parser.error("--print-every must be greater than zero.")
    if parsed.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be greater than zero.")

    output_dir = sampled_output_dir(
        amplification_output_dir(Path(parsed.output_dir).expanduser().resolve(), amplified_path),
        parsed.sampling_rate, None if fresh_sampling else parsed.sampling_seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result, nodes, cid2series, dp_summary = _execute_with_heartbeat(
        parsed,
        dp_fedavg_enabled=True,
        label=f"{parsed.framework.upper()} DP-FedAvg",
    )

    prefix = f"fh{parsed.fh + 1}_{parsed.framework.upper()}_dp_fedavg"
    forecast_path = output_dir / f"{prefix}_base_forecasts.csv"
    private_table = _export_base_forecasts(nodes, cid2series, forecast_path)
    training_artifacts_path = output_dir / f"{prefix}_training_artifacts.npz"
    _export_training_artifacts(nodes, cid2series, training_artifacts_path)
    if parsed.shared_artifact_dir:
        with np.load(training_artifacts_path, allow_pickle=False) as stored:
            shared_bounds_paths = save_or_validate_shared_bounds(
                artifacts={key: stored[key] for key in stored.files},
                artifact_dir=sampled_output_dir(
                    amplification_output_dir(parsed.shared_artifact_dir, amplified_path), parsed.sampling_rate,
                    None if fresh_sampling else parsed.sampling_seed
                ),
                forecast_horizon_count=parsed.fh + 1,
                slots_per_day=parsed.dp_slots_per_day,
                lower_quantile=parsed.dp_lower_quantile,
                upper_quantile=parsed.dp_upper_quantile,
                iqr_margin=parsed.dp_iqr_margin,
                physical_lower=parsed.dp_physical_lower,
            )
    else:
        shared_bounds_paths = {}

    calls = int(dp_summary.get("aggregation_calls", 0))
    calls += int(dp_summary.get("edge_aggregation_calls", 0))
    calls += int(dp_summary.get("cloud_aggregation_calls", 0))
    if dp_summary.get("sampling_amplification_eligible", False):
        accounting = sampled_fedavg_epsilon(
            parsed.noise_multiplier, calls, parsed.accounting_delta,
            int(dp_summary["population_size"]), int(dp_summary["sample_size"]), parsed.adjacency,
        )
    else:
        accounting = gaussian_composition_epsilon(
            noise_multiplier=parsed.noise_multiplier,
            aggregation_calls=calls,
            delta=parsed.accounting_delta,
        )
    if partial and not accounting["sampling_amplification_used"]:
        accounting["accounting"] = "Gaussian RDP composition without sampling amplification"
        accounting["amplification_not_used_reason"] = (
            "Requires a flat uniform-weight FHF Gaussian aggregation with an actual partial sample; "
            "hierarchical compositions and sample-dependent weights are not the same mechanism."
        )
    metadata = {
        **dp_summary,
        "framework": parsed.framework,
        "hierarchy_mode": parsed.hierarchy_mode,
        "forecast_horizon": int(parsed.fh) + 1,
        "input_csv": str(Path(parsed.csv).expanduser().resolve()),
        "num_rounds": int(parsed.num_rounds),
        "clip_norm": float(parsed.clip_norm),
        "noise_multiplier": float(parsed.noise_multiplier),
        "training_seed": int(parsed.seed),
        "dp_seed": parsed.dp_seed,
        "base_forecasts": str(forecast_path),
        "training_artifacts": str(training_artifacts_path),
        "shared_bounds": {
            key: str(path) for key, path in shared_bounds_paths.items()
        },
        "accounting": accounting,
        "warning": (
            "The accountant is conservative and assumes every aggregation call "
            "composes for the same protected client."
        ),
    }
    metadata_path = output_dir / f"{prefix}_privacy.json"
    round_logs = result.get("round_logs", result.get("cloud_round_logs", []))
    if partial:
        metadata["mechanism_version"] = "fixed_wor_gaussian_v1" if amplified_path else "conservative_sampling_v1"
        # Research artifacts remain internal. A separate model-only release
        # below excludes client identities, seeds, losses, forecasts and targets.
        metadata["sampling_seed"] = public_sampling_seed
        metadata["sampling_randomness"] = "fresh_hidden" if fresh_sampling else "fixed_research_seed"
        metadata["dp_seed"] = None
        accounting["budget_status"] = (
            "conditional_model_budget" if fresh_sampling and parsed.dp_seed is None
            else "hypothetical_budget_for_randomized_mechanism; fixed research seeds used"
        )
        metadata["warning"] = "Accounting covers the released global model only. All local forecasts, residuals, targets, losses and participation logs are internal research artifacts."
    round_log_path = output_dir / f"{prefix}_round_logs.json"
    round_log_path.write_text(json.dumps(round_logs, indent=2), encoding="utf-8")
    metadata["round_logs"] = str(round_log_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if partial:
        release_dir = output_dir / "model_release"
        release_dir.mkdir(exist_ok=True)
        server = result["cloud"] if parsed.framework == "fhf" else result["cloud_server"]
        np.savez_compressed(release_dir / "global_model.npz", **{
            key: tensor.detach().cpu().numpy() for key, tensor in server.global_sd.items()
        })
        (release_dir / "privacy.json").write_text(json.dumps({
            "accounting": accounting, "sampling_rate": parsed.sampling_rate,
            "mechanism_version": metadata["mechanism_version"],
            "adjacency": parsed.adjacency, "scope": "global_model.npz only",
        }, indent=2), encoding="utf-8")
    print(f"Saved base forecasts: {forecast_path}")
    print(f"Saved training artifacts: {training_artifacts_path}")
    for name, path in shared_bounds_paths.items():
        print(f"Shared {name}: {path}")
    print(f"Saved privacy metadata: {metadata_path}")

    if parsed.compare_baseline:
        baseline_result, baseline_nodes, baseline_names, _ = _execute_with_heartbeat(
            parsed,
            dp_fedavg_enabled=False,
            label=f"{parsed.framework.upper()} ordinary FedAvg baseline",
        )
        baseline_path = output_dir / f"{prefix}_fedavg_baseline_base_forecasts.csv"
        (output_dir / f"{prefix}_fedavg_baseline_round_logs.json").write_text(
            json.dumps(baseline_result.get("round_logs", baseline_result.get("cloud_round_logs", [])), indent=2),
            encoding="utf-8",
        )
        baseline_table = _export_base_forecasts(
            baseline_nodes, baseline_names, baseline_path
        )
        baseline_training_path = (
            output_dir / f"{prefix}_fedavg_baseline_training_artifacts.npz"
        )
        _export_training_artifacts(
            baseline_nodes, baseline_names, baseline_training_path
        )
        comparison = _performance_comparison(private_table, baseline_table)
        comparison_path = output_dir / f"{prefix}_performance_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Saved FedAvg baseline: {baseline_path}")
        print(f"Saved baseline training artifacts: {baseline_training_path}")
        print(f"Saved performance comparison: {comparison_path}")


if __name__ == "__main__":
    main()
