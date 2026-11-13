from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from PPHF_modules.fhf.FHF_DP import calibrate_time_slot_event_bounds


def _as_2d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def collect_training_artifacts(
    nodes: Mapping[int, object],
    cid2series: Mapping[int, object],
    node_order: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Read post-training residual artifacts without mutating model state."""
    client_ids = (
        sorted(int(client_id) for client_id in nodes)
        if node_order is None
        else [int(client_id) for client_id in node_order]
    )
    residuals = []
    predictions = []
    targets = []
    slots = []

    for client_id in client_ids:
        node = nodes[client_id]
        local_residuals = _as_2d(
            getattr(node, "residual_samples", None),
            f"node {client_id} residual_samples",
        )
        local_predictions = _as_2d(
            getattr(node, "y1", None), f"node {client_id} train predictions"
        )
        local_targets = _as_2d(
            getattr(node, "y2", None), f"node {client_id} train targets"
        )
        local_slots = np.asarray(
            getattr(node, "train_time_slot_indices", None), dtype=np.int64
        )

        if not (
            local_residuals.shape
            == local_predictions.shape
            == local_targets.shape
            == local_slots.shape
        ):
            raise ValueError(
                f"Training artifact shape mismatch for node {client_id}: "
                f"residuals={local_residuals.shape}, "
                f"predictions={local_predictions.shape}, "
                f"targets={local_targets.shape}, slots={local_slots.shape}."
            )
        if not np.allclose(
            local_residuals,
            local_targets - local_predictions,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                f"Saved residuals for node {client_id} are not target - prediction."
            )

        residuals.append(local_residuals)
        predictions.append(local_predictions)
        targets.append(local_targets)
        slots.append(local_slots)

    shapes = {array.shape for array in residuals}
    if len(shapes) != 1:
        raise ValueError(
            f"All nodes must share one training artifact shape, got {sorted(shapes)}."
        )

    return {
        "client_ids": np.asarray(client_ids, dtype=np.int64),
        "unique_ids": np.asarray(
            [str(cid2series[client_id]) for client_id in client_ids], dtype=str
        ),
        "residuals": np.stack(residuals),
        "train_predictions": np.stack(predictions),
        "train_targets": np.stack(targets),
        "train_time_slot_indices": np.stack(slots),
    }


def save_training_artifacts(
    nodes: Mapping[int, object],
    cid2series: Mapping[int, object],
    output_path: str | Path,
    node_order: Sequence[int] | None = None,
) -> Path:
    """Save residuals and their aligned local training coordinates."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **collect_training_artifacts(
            nodes=nodes, cid2series=cid2series, node_order=node_order
        ),
    )
    return path


def _slot_map(slots_per_day: int) -> pd.DataFrame:
    rows = []
    for slot in range(int(slots_per_day)):
        start_minute = int(round(slot * 1440.0 / slots_per_day))
        end_minute = int(round((slot + 1) * 1440.0 / slots_per_day))
        rows.append(
            {
                "time_slot": int(slot),
                "slot_start": f"{start_minute // 60:02d}:{start_minute % 60:02d}",
                "slot_end": (
                    "24:00"
                    if end_minute == 1440
                    else f"{end_minute // 60:02d}:{end_minute % 60:02d}"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_shared_time_slot_bounds(
    artifacts: Mapping[str, np.ndarray],
    slots_per_day: int = 48,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    iqr_margin: float = 0.0,
    physical_lower: float | None = 0.0,
) -> pd.DataFrame:
    """Build method-independent bounds from the common local training targets."""
    client_ids = np.asarray(artifacts["client_ids"], dtype=np.int64)
    unique_ids = np.asarray(artifacts["unique_ids"], dtype=str)
    targets = np.asarray(artifacts["train_targets"], dtype=np.float64)
    slots = np.asarray(artifacts["train_time_slot_indices"], dtype=np.int64)
    if targets.shape != slots.shape or targets.shape[0] != client_ids.size:
        raise ValueError("Training targets, time slots, and client IDs are misaligned.")

    rows = []
    for position, client_id in enumerate(client_ids):
        lower, upper = calibrate_time_slot_event_bounds(
            calibration_forecast=targets[position],
            time_slot_indices=slots[position],
            slots_per_day=slots_per_day,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            iqr_margin=iqr_margin,
            physical_lower=physical_lower,
        )
        for slot in range(int(slots_per_day)):
            start_minute = int(round(slot * 1440.0 / slots_per_day))
            rows.append(
                {
                    "node_id": int(client_id),
                    "unique_id": str(unique_ids[position]),
                    "time_slot": int(slot),
                    "slot_start": (
                        f"{start_minute // 60:02d}:{start_minute % 60:02d}"
                    ),
                    "lower_L": float(lower[slot]),
                    "upper_U": float(upper[slot]),
                    "sensitivity_U_minus_L": float(upper[slot] - lower[slot]),
                    "n_train_coordinates": int(
                        np.count_nonzero(slots[position] == slot)
                    ),
                    "n_forecast_horizons": int(targets.shape[2]),
                    "lower_quantile": float(lower_quantile),
                    "upper_quantile": float(upper_quantile),
                    "iqr_margin": float(iqr_margin),
                    "physical_lower": physical_lower,
                }
            )
    return pd.DataFrame(rows)


def _validate_existing_bounds(existing: pd.DataFrame, expected: pd.DataFrame) -> None:
    keys = ["node_id", "time_slot"]
    existing = existing.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if list(existing.columns) != list(expected.columns):
        raise ValueError("Existing shared bounds have an incompatible schema.")
    text_columns = ["unique_id", "slot_start"]
    numeric_columns = [column for column in expected.columns if column not in text_columns]
    if not existing[text_columns].equals(expected[text_columns]):
        raise ValueError("Existing shared bounds identify different nodes or slots.")
    if not np.allclose(
        existing[numeric_columns].to_numpy(dtype=np.float64),
        expected[numeric_columns].to_numpy(dtype=np.float64),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    ):
        raise ValueError(
            "Existing shared bounds differ from the common training-target bounds."
        )


def save_or_validate_shared_bounds(
    artifacts: Mapping[str, np.ndarray],
    artifact_dir: str | Path,
    forecast_horizon_count: int,
    slots_per_day: int = 48,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    iqr_margin: float = 0.0,
    physical_lower: float | None = 0.0,
) -> dict[str, Path]:
    """Create one shared bounds release, or verify an identical existing one."""
    directory = Path(artifact_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"fh{int(forecast_horizon_count)}_shared_event_dp"
    bounds_path = directory / f"{prefix}_train_bounds_by_time_slot.csv"
    slot_map_path = directory / f"{prefix}_time_slot_map.csv"
    metadata_path = directory / f"{prefix}_bounds_metadata.json"

    bounds = build_shared_time_slot_bounds(
        artifacts=artifacts,
        slots_per_day=slots_per_day,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        iqr_margin=iqr_margin,
        physical_lower=physical_lower,
    )
    if bounds_path.is_file():
        _validate_existing_bounds(pd.read_csv(bounds_path), bounds)
    else:
        bounds.to_csv(bounds_path, index=False)

    slot_map = _slot_map(slots_per_day)
    if slot_map_path.is_file():
        if not pd.read_csv(slot_map_path).equals(slot_map):
            raise ValueError("Existing shared time-slot map is incompatible.")
    else:
        slot_map.to_csv(slot_map_path, index=False)

    metadata = {
        "method_independent": True,
        "calibration_source": "common_local_training_targets",
        "scope": "per_client_time_slot",
        "forecast_horizon_count": int(forecast_horizon_count),
        "slots_per_day": int(slots_per_day),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "iqr_margin": float(iqr_margin),
        "physical_lower": physical_lower,
        "bounds_csv": bounds_path.name,
        "time_slot_map_csv": slot_map_path.name,
        "formal_dp_note": (
            "Bounds must be public, independently calibrated, or otherwise "
            "permitted before claiming formal DP."
        ),
    }
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing shared bounds metadata is incompatible.")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return {
        "bounds": bounds_path,
        "time_slot_map": slot_map_path,
        "metadata": metadata_path,
    }
