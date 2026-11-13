from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PPHF_modules.dp_fhf.dp_covariance import DPCovarianceConfig, build_private_reconciliation_matrices


def _coordinate_bounds_from_file(
    bounds_path: Path,
    client_ids: list[int],
    slot_indices: np.ndarray,
    residual_shape: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Map public per-client/time-slot target widths to residual coordinates.

    If both the target and a forecast are clipped to [L, U], their residual is
    bounded by [-(U-L), U-L]. We apply that same fixed bound directly to the
    saved residual before its local-DP release.
    """
    bounds = pd.read_csv(bounds_path)
    required = {"node_id", "time_slot", "sensitivity_U_minus_L"}
    missing = sorted(required - set(bounds.columns))
    if missing:
        raise ValueError(f"Bounds file is missing columns: {missing}")
    bounds = bounds[["node_id", "time_slot", "sensitivity_U_minus_L"]].copy()
    bounds["node_id"] = bounds["node_id"].astype(int)
    bounds["time_slot"] = bounds["time_slot"].astype(int)
    if bounds.duplicated(["node_id", "time_slot"]).any():
        raise ValueError("Bounds file has duplicate (node_id, time_slot) rows.")
    raw_widths = bounds["sensitivity_U_minus_L"].to_numpy(dtype=float)
    if not np.all(np.isfinite(raw_widths)) or np.any(raw_widths <= 0.0):
        raise ValueError("Every bounds width must be finite and greater than zero.")

    slots = np.asarray(slot_indices, dtype=np.int64)
    if slots.shape != residual_shape:
        raise ValueError(
            "Training time-slot indices must have the same shape as residuals: "
            f"{slots.shape} != {residual_shape}."
        )

    lower_by_client: dict[int, np.ndarray] = {}
    upper_by_client: dict[int, np.ndarray] = {}
    for position, client_id in enumerate(client_ids):
        local = (
            bounds.loc[bounds["node_id"] == int(client_id)]
            .set_index("time_slot")["sensitivity_U_minus_L"]
        )
        requested = pd.Index(np.unique(slots[position]), name="time_slot")
        missing_slots = requested.difference(local.index)
        if len(missing_slots):
            raise ValueError(
                f"Bounds file has no entries for client {client_id}, slots "
                f"{missing_slots.tolist()}."
            )
        widths = local.reindex(slots[position].ravel()).to_numpy(dtype=np.float64)
        widths = widths.reshape(slots[position].shape)
        lower_by_client[int(client_id)] = -widths
        upper_by_client[int(client_id)] = widths
    return lower_by_client, upper_by_client


def _build_summing_matrix(series_names: list[str]) -> np.ndarray:
    bottoms = [
        name
        for name in series_names
        if not any(
            other != name and other.startswith(name + "/")
            for other in series_names
        )
    ]
    return np.array(
        [
            [
                float(bottom == node or bottom.startswith(node + "/"))
                for bottom in bottoms
            ]
            for node in series_names
        ],
        dtype=np.float64,
    )


def _matrix_by_node(
    table: pd.DataFrame,
    node_ids: list[int],
    value_column: str,
    horizon_index: int,
) -> tuple[np.ndarray, pd.Index]:
    local = table.loc[table["horizon_index"] == horizon_index]
    matrix = local.pivot(index="ds", columns="node_id", values=value_column)
    matrix = matrix.reindex(columns=node_ids).dropna(axis=0)
    if matrix.empty:
        raise ValueError(f"No complete rows for horizon {horizon_index}.")
    return matrix.to_numpy(dtype=np.float64).T, matrix.index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Privatize local residuals, estimate DP covariance, and generate "
            "MinT-shrinkage/WLSV reconciliation matrices."
        )
    )
    parser.add_argument("--forecasts", required=True)
    parser.add_argument(
        "--per-node-metrics",
        default=None,
        help="Original per-node metrics used to recover MASE/RMSSE scale denominators.",
    )
    parser.add_argument(
        "--residuals-npz",
        default=None,
        help=(
            "Training artifacts exported by run_dp_fedavg.py. When omitted, "
            "y_true-base test residuals are used for diagnostic runs only."
        ),
    )
    parser.add_argument("--output-dir", default=str(DP_EXPERIMENT_DIR / "outputs"))
    parser.add_argument(
        "--output-label",
        default="",
        help="Optional filename prefix for keeping multiple runs in one directory.",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--mechanism", choices=["laplace", "gaussian"], default="laplace")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--lower",
        type=float,
        default=None,
        help="Scalar residual lower bound; use together with --upper.",
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=None,
        help="Scalar residual upper bound; use together with --lower.",
    )
    parser.add_argument(
        "--bounds-file",
        default=None,
        help=(
            "Dataset-specific per-client/time-slot target bounds. Residual "
            "bounds are set to +/-(U-L) for each saved training coordinate."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--no-debias", action="store_true")
    parser.add_argument(
        "--save-detailed-forecasts",
        action="store_true",
        help="Save coordinate-level reconciled forecasts (large for full Ausgrid).",
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Diagnostic/smoke-test limit; omit for all available residual rows.",
    )
    parsed = parser.parse_args()
    scalar_bounds_supplied = parsed.lower is not None or parsed.upper is not None
    if parsed.bounds_file and scalar_bounds_supplied:
        parser.error("Use either --bounds-file or --lower/--upper, not both.")
    if not parsed.bounds_file and not (
        parsed.lower is not None and parsed.upper is not None
    ):
        parser.error("Pass --bounds-file or both --lower and --upper.")

    source = Path(parsed.forecasts).expanduser().resolve()
    table = pd.read_csv(
        source,
        usecols=[
            "node_id",
            "unique_id",
            "ds",
            "horizon_index",
            "y_true",
            "base",
        ],
    )
    table["node_id"] = table["node_id"].astype(int)
    table["horizon_index"] = table["horizon_index"].astype(int)
    table["ds"] = pd.to_datetime(table["ds"], errors="raise")
    node_map = (
        table[["node_id", "unique_id"]]
        .drop_duplicates()
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    node_ids = node_map["node_id"].tolist()
    series_names = node_map["unique_id"].astype(str).tolist()
    summing_matrix = _build_summing_matrix(series_names)
    horizons = sorted(table["horizon_index"].unique().tolist())

    residuals_by_client: dict[int, list[np.ndarray]] = {
        client_id: [] for client_id in node_ids
    }
    base_by_horizon = {}
    truth_by_horizon = {}
    times_by_horizon = {}
    for horizon_index in horizons:
        base, times = _matrix_by_node(table, node_ids, "base", horizon_index)
        truth, truth_times = _matrix_by_node(table, node_ids, "y_true", horizon_index)
        common_times = times.intersection(truth_times)
        base_frame = pd.DataFrame(base.T, index=times).reindex(common_times)
        truth_frame = pd.DataFrame(truth.T, index=truth_times).reindex(common_times)
        if parsed.max_times is not None:
            base_frame = base_frame.iloc[: int(parsed.max_times)]
            truth_frame = truth_frame.iloc[: int(parsed.max_times)]
        base = base_frame.to_numpy(dtype=np.float64).T
        truth = truth_frame.to_numpy(dtype=np.float64).T
        if parsed.residuals_npz is None:
            for node_position, client_id in enumerate(node_ids):
                residuals_by_client[client_id].append(
                    (truth[node_position] - base[node_position])[:, None]
                )
        base_by_horizon[horizon_index] = base
        truth_by_horizon[horizon_index] = truth
        times_by_horizon[horizon_index] = base_frame.index

    artifact_targets = None
    artifact_slots = None
    residual_path = None
    if parsed.residuals_npz is None:
        residual_matrices = {
            client_id: np.concatenate(columns, axis=1)
            for client_id, columns in residuals_by_client.items()
        }
        residual_source = "saved_test_forecast_diagnostic"
    else:
        residual_path = Path(parsed.residuals_npz).expanduser().resolve()
        with np.load(residual_path, allow_pickle=False) as artifacts:
            artifact_ids = artifacts["client_ids"].astype(int).tolist()
            artifact_residuals = np.asarray(artifacts["residuals"], dtype=np.float64)
            if "train_targets" in artifacts.files:
                artifact_targets = np.asarray(
                    artifacts["train_targets"], dtype=np.float64
                )
            if "train_time_slot_indices" in artifacts.files:
                artifact_slots = np.asarray(
                    artifacts["train_time_slot_indices"], dtype=np.int64
                )
        if artifact_ids != node_ids:
            raise ValueError(
                "Training artifact client order does not match the forecast table."
            )
        if artifact_residuals.ndim != 3:
            raise ValueError(
                "Training residual artifact must have shape (clients, time, horizon)."
            )
        if artifact_residuals.shape[2] != len(horizons):
            raise ValueError(
                "Training residual artifact horizon does not match forecasts."
            )
        residual_matrices = {
            client_id: artifact_residuals[position]
            for position, client_id in enumerate(node_ids)
        }
        residual_source = "local_training_residual_artifact"

    bounds_path = None
    if parsed.bounds_file is not None:
        if residual_path is None or artifact_slots is None:
            raise ValueError(
                "--bounds-file requires --residuals-npz containing aligned "
                "train_time_slot_indices."
            )
        bounds_path = Path(parsed.bounds_file).expanduser().resolve()
        lower_bound, upper_bound = _coordinate_bounds_from_file(
            bounds_path=bounds_path,
            client_ids=node_ids,
            slot_indices=artifact_slots,
            residual_shape=artifact_residuals.shape,
        )
    else:
        lower_bound = parsed.lower
        upper_bound = parsed.upper

    config = DPCovarianceConfig(
        epsilon=parsed.epsilon,
        mechanism=parsed.mechanism,
        delta=parsed.delta,
        lower=lower_bound,
        upper=upper_bound,
        seed=parsed.seed,
        ridge=parsed.ridge,
        debias_noise_variance=not parsed.no_debias,
    )
    result = build_private_reconciliation_matrices(
        summing_matrix=summing_matrix,
        client_ids=node_ids,
        residuals_by_client=residual_matrices,
        config=config,
        methods=("mint_shrinkage", "wlsv"),
    )

    output_dir = Path(parsed.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character
        for character in str(parsed.output_label).strip()
        if character.isalnum() or character in {"-", "_"}
    )
    if safe_label != str(parsed.output_label).strip():
        raise ValueError("--output-label may contain only letters, numbers, '-' and '_'.")
    file_prefix = f"{safe_label}_" if safe_label else ""
    npz_path = output_dir / f"{file_prefix}dp_covariance_matrices.npz"
    np.savez_compressed(
        npz_path,
        S=summing_matrix,
        W_mint_shrinkage=result["Ws_by_method"]["mint_shrinkage"],
        W_wlsv=result["Ws_by_method"]["mint_var"],
        P_mint_shrinkage=result["Ps"]["mint_shrinkage"],
        P_wlsv=result["Ps"]["mint_var"],
    )

    forecast_rows = []
    metric_rows = []
    scale_by_node = {}
    scale_source = None
    if artifact_targets is not None:
        if artifact_targets.shape != artifact_residuals.shape:
            raise ValueError(
                "Training targets must have the same shape as residuals: "
                f"{artifact_targets.shape} != {artifact_residuals.shape}."
            )
        for node_position, client_id in enumerate(node_ids):
            differences = np.diff(artifact_targets[node_position], axis=0)
            mase_denominator = float(np.mean(np.abs(differences)))
            rmsse_denominator = float(np.mean(differences * differences))
            if not (
                np.isfinite(mase_denominator)
                and mase_denominator > 0.0
                and np.isfinite(rmsse_denominator)
                and rmsse_denominator > 0.0
            ):
                raise ValueError(
                    f"Invalid training scale denominators for client {client_id}."
                )
            scale_by_node[client_id] = (mase_denominator, rmsse_denominator)
        scale_source = "saved_training_targets_first_differences"
    elif parsed.per_node_metrics is not None:
        original_metrics = pd.read_csv(
            Path(parsed.per_node_metrics).expanduser().resolve()
        )
        base_metrics = (
            original_metrics.loc[original_metrics["method"] == "base"]
            .set_index("node_id")
            .loc[node_ids]
        )
        for node_position, client_id in enumerate(node_ids):
            full_truth = np.concatenate(
                [truth_by_horizon[h][node_position] for h in horizons]
            )
            full_base = np.concatenate(
                [base_by_horizon[h][node_position] for h in horizons]
            )
            base_error = full_truth - full_base
            base_mse = float(np.mean(base_error * base_error))
            row = base_metrics.loc[client_id]
            mase_denominator = float(row["MAE"]) / float(row["MASE"])
            rmsse_denominator = base_mse / float(row["RMSSE"]) ** 2
            scale_by_node[client_id] = (mase_denominator, rmsse_denominator)
        scale_source = "forecast_reconstruction_fallback"

    for horizon_position, horizon_index in enumerate(horizons):
        base = base_by_horizon[horizon_index]
        truth = truth_by_horizon[horizon_index]
        for method, p_tensor in result["Ps"].items():
            reconciled = p_tensor[horizon_position] @ base
            for node_position, client_id in enumerate(node_ids):
                errors = truth[node_position] - reconciled[node_position]
                mse = float(np.mean(errors * errors))
                mae = float(np.mean(np.abs(errors)))
                metric_row = {
                    "node_id": client_id,
                    "unique_id": series_names[node_position],
                    "horizon_index": horizon_index,
                    "method": "wlsv" if method == "mint_var" else method,
                    "RMSE": float(np.sqrt(mse)),
                    "MAE": mae,
                }
                if client_id in scale_by_node:
                    mase_denominator, rmsse_denominator = scale_by_node[client_id]
                    metric_row["MASE"] = mae / mase_denominator
                    metric_row["RMSSE"] = float(np.sqrt(mse / rmsse_denominator))
                metric_rows.append(metric_row)
                if parsed.save_detailed_forecasts:
                    forecast_rows.append(
                        pd.DataFrame(
                            {
                                "node_id": client_id,
                                "unique_id": series_names[node_position],
                                "ds": times_by_horizon[horizon_index],
                                "horizon_index": horizon_index,
                                "method": "wlsv" if method == "mint_var" else method,
                                "y_true": truth[node_position],
                                "base": base[node_position],
                                "reconciled": reconciled[node_position],
                            }
                        )
                    )
    forecasts_path = (
        output_dir / f"{file_prefix}dp_covariance_reconciled_forecasts.csv"
    )
    metrics_path = output_dir / f"{file_prefix}dp_covariance_metrics.csv"
    if parsed.save_detailed_forecasts:
        pd.concat(forecast_rows, ignore_index=True).to_csv(forecasts_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    local_dp_metadata = result["metadata_by_client"]
    clipped_count = int(
        sum(item["clipped_coordinate_count"] for item in local_dp_metadata.values())
    )
    coordinate_count = int(
        sum(item["coordinate_count"] for item in local_dp_metadata.values())
    )
    clipping_fraction = clipped_count / coordinate_count
    sensitivity_sum = float(
        sum(
            np.asarray(item["sensitivity_by_coordinate"], dtype=np.float64).sum()
            for item in local_dp_metadata.values()
        )
    )
    sensitivity_min = float(
        min(
            np.asarray(item["sensitivity_by_coordinate"], dtype=np.float64).min()
            for item in local_dp_metadata.values()
        )
    )
    sensitivity_max = float(
        max(
            np.asarray(item["sensitivity_by_coordinate"], dtype=np.float64).max()
            for item in local_dp_metadata.values()
        )
    )
    if bounds_path is not None:
        bounds_metadata = {
            "mode": "per_client_time_slot_residual_bounds",
            "source": str(bounds_path),
            "definition": "lower=-(target_U-target_L), upper=target_U-target_L",
            "sensitivity_min": sensitivity_min,
            "sensitivity_mean": sensitivity_sum / coordinate_count,
            "sensitivity_max": sensitivity_max,
        }
    else:
        bounds_metadata = {
            "mode": "scalar",
            "lower": parsed.lower,
            "upper": parsed.upper,
        }

    privacy_metadata = {
        "output_label": safe_label,
        "mechanism": parsed.mechanism,
        "epsilon_per_client_residual_release": parsed.epsilon,
        "delta_per_client_residual_release": (
            parsed.delta if parsed.mechanism == "gaussian" else 0.0
        ),
        "privacy_level": "event",
        "event_definition": "one scalar local training residual coordinate",
        "bounds": bounds_metadata,
        "bounds_must_be_public_or_independently_calibrated": True,
        "clipped_coordinate_count": clipped_count,
        "coordinate_count": coordinate_count,
        "clipping_fraction": clipping_fraction,
        "seed_is_fixed_for_research": parsed.seed is not None,
        "debias_noise_variance": not parsed.no_debias,
        "raw_residuals_uploaded": False,
        "base_contributions_private": False,
        "base_contribution_source": (
            "dp_fedavg_base_forecasts"
            if "dp_fedavg" in safe_label.lower()
            else "non_dp_base_forecasts"
        ),
        "reconciliation_methods": ["mint_shrinkage", "wlsv"],
        "contribution_exchange": (
            "sender base contributions reconciled with P estimated from "
            "locally privatized residual uploads"
        ),
        "source_forecasts": str(source),
        "residual_source": residual_source,
        "metric_scale_source": scale_source,
        "source_warning": (
            "Saved forecast CSV residuals are test-residual diagnostics. A formal "
            "training run must pass local training residuals instead."
            if parsed.residuals_npz is None
            else ""
        ),
    }
    metadata_path = output_dir / f"{file_prefix}dp_covariance_privacy.json"
    metadata_path.write_text(
        json.dumps(privacy_metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved matrices: {npz_path}")
    if parsed.save_detailed_forecasts:
        print(f"Saved forecasts: {forecasts_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved privacy metadata: {metadata_path}")
    print(
        f"Residual clipping audit: {clipped_count}/{coordinate_count} "
        f"({clipping_fraction:.4%}) coordinates clipped."
    )


if __name__ == "__main__":
    main()
