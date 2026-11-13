from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = PROJECT_ROOT
SHARED = FHF_MODULE_DIR

from PPHF_modules.fhf.FHF_DP import privatize_event_forecast


EXPECTED_P_METHODS = (
    "bu",
    "mint_cov",
    "mint_shrinkage",
    "mint_ols",
    "mint_var",
    "wls_structure",
)


def _stable_rng(label: str, seed: int | None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    label_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(np.random.SeedSequence([int(seed), label_seed]))


def _sampling_step(timestamps: pd.Series) -> pd.Timedelta:
    ordered = pd.Series(pd.to_datetime(timestamps).dropna().unique()).sort_values()
    positive = ordered.diff().dropna()
    positive = positive[positive > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("At least two distinct timestamps are required.")
    return positive.median()


def _load_p_matrices(
    artifact_dir: Path, prefix: str, node_map: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    sender_columns = node_map["sender_column"].astype(str).tolist()
    node_count = len(node_map)
    matrices = {}
    hashes = {}
    files = sorted(artifact_dir.glob(f"{prefix}_o2mr_P_*.csv"))
    if not files:
        raise FileNotFoundError(f"No saved P files found for {prefix}.")
    for path in files:
        table = pd.read_csv(path)
        methods = table["method"].astype(str).unique().tolist()
        if len(methods) != 1:
            raise ValueError(f"{path.name} must contain exactly one method.")
        method = methods[0]
        if method in matrices:
            raise ValueError(f"Duplicate saved P method: {method}.")
        by_horizon = []
        for _, group in table.groupby("horizon_index", sort=True):
            ordered = group.sort_values("recipient_matrix_index")
            matrix = ordered[sender_columns].to_numpy(dtype=np.float64)
            if matrix.shape != (node_count, node_count):
                raise ValueError(
                    f"{path.name} has P shape {matrix.shape}; "
                    f"expected {(node_count, node_count)}."
                )
            by_horizon.append(matrix)
        matrices[method] = np.stack(by_horizon)
        hashes[method] = hashlib.sha256(path.read_bytes()).hexdigest()
    missing = sorted(set(EXPECTED_P_METHODS) - set(matrices))
    unexpected = sorted(set(matrices) - set(EXPECTED_P_METHODS))
    if missing or unexpected:
        raise ValueError(
            "Saved P method set must contain all standard reconciliation methods. "
            f"Missing: {missing or 'none'}; unexpected: {unexpected or 'none'}."
        )
    return matrices, hashes


def _pivot(
    table: pd.DataFrame,
    value: str,
    node_ids: list[int],
    timestamps: pd.DatetimeIndex,
    horizons: list[int],
) -> np.ndarray:
    output = np.empty((len(node_ids), len(timestamps), len(horizons)))
    for horizon_position, horizon_index in enumerate(horizons):
        local = table.loc[table["horizon_index"] == horizon_index]
        matrix = local.pivot(index="node_id", columns="ds", values=value)
        output[:, :, horizon_position] = matrix.loc[
            node_ids, timestamps
        ].to_numpy(dtype=np.float64)
    return output


def _metrics(
    truth: np.ndarray,
    nonprivate_base: np.ndarray,
    private_base: np.ndarray,
    nonprivate_reconciled: dict[str, np.ndarray],
    private_reconciled: dict[str, np.ndarray],
    scale_by_node: dict[int, tuple[float, float]],
    node_ids: list[int],
) -> pd.DataFrame:
    rows = []
    predictions = {
        "base": (nonprivate_base, private_base),
        **{
            method: (nonprivate_reconciled[method], private_reconciled[method])
            for method in private_reconciled
        },
    }
    for method, (ordinary, private) in predictions.items():
        for label, forecast in [
            ("nonprivate", ordinary),
            ("dp_o2mr", private),
        ]:
            error = truth - forecast
            node_mae = np.mean(np.abs(error), axis=(1, 2))
            node_mse = np.mean(error * error, axis=(1, 2))
            node_mase = []
            node_rmsse = []
            for node_position, node_id in enumerate(node_ids):
                mase_denominator, rmsse_denominator = scale_by_node[node_id]
                node_mase.append(node_mae[node_position] / mase_denominator)
                node_rmsse.append(
                    np.sqrt(node_mse[node_position] / rmsse_denominator)
                )
            rows.append(
                {
                    "method": method,
                    "release": label,
                    "MAE": float(np.mean(node_mae)),
                    "RMSE": float(np.mean(np.sqrt(node_mse))),
                    "MASE": float(np.mean(node_mase)),
                    "RMSSE": float(np.mean(node_rmsse)),
                    "coordinate_count": int(error.size),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply local base-forecast DP and reconcile with previously saved P "
            "matrices. P is loaded, never re-estimated."
        )
    )
    parser.add_argument("--base-forecasts", required=True)
    parser.add_argument(
        "--per-node-metrics",
        required=True,
        help="Original per-node metrics used to recover MASE/RMSSE denominators.",
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument(
        "--bounds-file",
        help=(
            "Method-independent per-client time-slot bounds CSV. When omitted, "
            "the path declared by the framework config is used."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--mechanism", choices=["laplace", "gaussian"], default="laplace")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-label",
        default="experiment",
        help="Label distinguishing non-DP-training and DP-FedAvg inputs.",
    )
    parser.add_argument(
        "--file-prefix",
        default=None,
        help=(
            "Optional filename prefix. This is separate from --output-label so "
            "epsilon runs can share the same seeded random stream."
        ),
    )
    parser.add_argument(
        "--save-detailed-forecasts",
        action="store_true",
        help="Save coordinate-level private forecasts (large for full Ausgrid).",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    prefix = str(args.artifact_prefix)
    node_map = pd.read_csv(
        artifact_dir / f"{prefix}_o2mr_node_map.csv"
    ).sort_values("matrix_index")
    config = pd.read_csv(
        artifact_dir / f"{prefix}_event_dp_o2mr_config.csv"
    ).iloc[0]
    if args.bounds_file:
        bounds_path = Path(args.bounds_file).expanduser().resolve()
    else:
        configured_bounds = str(config.get("bounds_csv", "")).strip()
        bounds_path = (
            artifact_dir / configured_bounds
            if configured_bounds
            else artifact_dir
            / f"{prefix}_event_dp_train_bounds_by_time_slot.csv"
        )
    if not bounds_path.is_file():
        raise FileNotFoundError(f"Missing DP-O2MR bounds file: {bounds_path}")
    bounds = pd.read_csv(bounds_path)
    p_matrices, p_hashes = _load_p_matrices(artifact_dir, prefix, node_map)

    source_path = Path(args.base_forecasts).expanduser().resolve()
    forecasts = pd.read_csv(
        source_path,
        usecols=["node_id", "unique_id", "ds", "horizon_index", "y_true", "base"],
    )
    forecasts["node_id"] = forecasts["node_id"].astype(int)
    forecasts["horizon_index"] = forecasts["horizon_index"].astype(int)
    forecasts["ds"] = pd.to_datetime(forecasts["ds"], errors="raise")
    node_ids = node_map["node_id"].astype(int).tolist()
    if set(forecasts["node_id"].unique()) != set(node_ids):
        raise ValueError("Base forecast node IDs do not match the saved P node map.")
    timestamps = pd.DatetimeIndex(sorted(forecasts["ds"].unique()))
    horizons = sorted(forecasts["horizon_index"].unique().tolist())
    if any(tensor.shape[0] != len(horizons) for tensor in p_matrices.values()):
        raise ValueError("Saved P horizon count does not match base forecasts.")

    step = _sampling_step(forecasts["ds"])
    slots_per_day = int(config["dp_slots_per_day"])
    target_time = (
        forecasts["ds"]
        + forecasts["horizon_index"].map(
            lambda horizon: int(horizon) * step
        )
    )
    minute = (
        target_time.dt.hour * 60
        + target_time.dt.minute
        + target_time.dt.second / 60.0
    )
    forecasts["time_slot"] = np.clip(
        np.floor(minute.to_numpy() * slots_per_day / 1440.0).astype(int),
        0,
        slots_per_day - 1,
    )
    forecasts = forecasts.merge(
        bounds[["node_id", "time_slot", "lower_L", "upper_U"]],
        on=["node_id", "time_slot"],
        how="left",
        validate="many_to_one",
    )
    if forecasts[["lower_L", "upper_U"]].isna().any().any():
        raise ValueError("Saved bounds do not cover every base forecast coordinate.")

    base = _pivot(forecasts, "base", node_ids, timestamps, horizons)
    truth = _pivot(forecasts, "y_true", node_ids, timestamps, horizons)
    lower = _pivot(forecasts, "lower_L", node_ids, timestamps, horizons)
    upper = _pivot(forecasts, "upper_U", node_ids, timestamps, horizons)

    private_base = np.empty_like(base)
    clipped_count = 0
    for node_position, node_id in enumerate(node_ids):
        released, metadata = privatize_event_forecast(
            forecast=base[node_position],
            lower=lower[node_position],
            upper=upper[node_position],
            epsilon=args.epsilon,
            mechanism=args.mechanism,
            delta=args.delta,
            rng=_stable_rng(
                f"{args.output_label}:{prefix}:client:{node_id}", args.seed
            ),
        )
        private_base[node_position] = released
        clipped_count += int(metadata["clipped_coordinate_count"])

    original_metrics = pd.read_csv(
        Path(args.per_node_metrics).expanduser().resolve()
    )
    base_metrics = (
        original_metrics.loc[original_metrics["method"] == "base"]
        .set_index("node_id")
        .loc[node_ids]
    )
    scale_by_node = {}
    base_error = truth - base
    base_mse_by_node = np.mean(base_error * base_error, axis=(1, 2))
    for node_position, node_id in enumerate(node_ids):
        row = base_metrics.loc[node_id]
        scale_by_node[node_id] = (
            float(row["MAE"]) / float(row["MASE"]),
            float(base_mse_by_node[node_position]) / float(row["RMSSE"]) ** 2,
        )

    nonprivate_reconciled = {}
    private_reconciled = {}
    for method, p_tensor in p_matrices.items():
        ordinary = np.zeros_like(base)
        private = np.zeros_like(base)
        # This BLAS implementation is exactly the sum of saved-P sender-column
        # contributions: y_i = sum_j P[i, j] * y_j.
        for horizon_position in range(len(horizons)):
            ordinary[:, :, horizon_position] = (
                p_tensor[horizon_position] @ base[:, :, horizon_position]
            )
            private[:, :, horizon_position] = (
                p_tensor[horizon_position] @ private_base[:, :, horizon_position]
            )
        nonprivate_reconciled[method] = ordinary
        private_reconciled[method] = private

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_file_prefix = (
        str(args.output_label)
        if args.file_prefix is None
        else str(args.file_prefix)
    ).strip()
    file_prefix = "".join(
        character
        for character in raw_file_prefix
        if character.isalnum() or character in {"-", "_"}
    )
    if file_prefix != raw_file_prefix:
        raise ValueError(
            "--file-prefix may contain only letters, numbers, '-' and '_'."
        )
    metrics = _metrics(
        truth,
        base,
        private_base,
        nonprivate_reconciled,
        private_reconciled,
        scale_by_node,
        node_ids,
    )
    metrics.insert(0, "input_label", args.output_label)
    metrics_path = output_dir / f"{file_prefix}_dp_o2mr_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    if args.save_detailed_forecasts:
        rows = []
        unique_ids = node_map["unique_id"].astype(str).tolist()
        for node_position, node_id in enumerate(node_ids):
            for horizon_position, horizon_index in enumerate(horizons):
                local = pd.DataFrame(
                    {
                        "node_id": node_id,
                        "unique_id": unique_ids[node_position],
                        "ds": timestamps,
                        "horizon_index": horizon_index,
                        "y_true": truth[node_position, :, horizon_position],
                        "nonprivate_base": base[node_position, :, horizon_position],
                        "dp_base": private_base[node_position, :, horizon_position],
                    }
                )
                for method in p_matrices:
                    local[f"nonprivate_{method}"] = nonprivate_reconciled[method][
                        node_position, :, horizon_position
                    ]
                    local[f"dp_{method}"] = private_reconciled[method][
                        node_position, :, horizon_position
                    ]
                rows.append(local)
        pd.concat(rows, ignore_index=True).to_csv(
            output_dir / f"{file_prefix}_dp_o2mr_forecasts.csv",
            index=False,
        )

    metadata = {
        "input_label": args.output_label,
        "source_base_forecasts": str(source_path),
        "artifact_prefix": prefix,
        "P_reestimated": False,
        "P_sha256_by_method": p_hashes,
        "reconciliation_methods": list(p_matrices),
        "privacy_level": "event",
        "mechanism": args.mechanism,
        "epsilon_per_release": args.epsilon,
        "delta_per_release": args.delta if args.mechanism == "gaussian" else 0.0,
        "seed_is_fixed_for_research": args.seed is not None,
        "bounds_source": (
            "common_local_training_targets"
            if "shared_event_dp" in bounds_path.name
            else str(config["dp_bounds_source"])
        ),
        "bounds_scope": str(config["dp_bounds_scope"]),
        "bounds_file": str(bounds_path),
        "bounds_method_independent": "shared_event_dp" in bounds_path.name,
        "clipped_coordinate_count": clipped_count,
        "coordinate_count": int(base.size),
        "reconciliation": (
            "saved P linear post-processing; algebraically identical to the "
            "sum of sender-column contributions"
        ),
    }
    metadata["file_prefix"] = file_prefix
    metadata_path = output_dir / f"{file_prefix}_dp_o2mr_privacy.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved privacy metadata: {metadata_path}")


if __name__ == "__main__":
    main()
