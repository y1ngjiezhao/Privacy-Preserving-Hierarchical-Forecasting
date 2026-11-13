from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _as_horizon_vector(values, horizon: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), int(horizon))
    else:
        arr = arr.reshape(-1)
    if arr.size != int(horizon):
        raise ValueError(f"{name} must be scalar or have {horizon} values, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def validate_event_bounds(lower, upper, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    lower_h = _as_horizon_vector(lower, horizon=horizon, name="DP lower bound")
    upper_h = _as_horizon_vector(upper, horizon=horizon, name="DP upper bound")
    if np.any(upper_h < lower_h):
        bad = np.flatnonzero(upper_h < lower_h).tolist()
        raise ValueError(f"DP upper bound is smaller than lower bound at horizons {bad}.")
    return lower_h, upper_h


def _broadcast_forecast_bound(values, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.size == shape[1]:
        arr = np.broadcast_to(arr[None, :], shape).copy()
    elif arr.shape == shape:
        arr = arr.copy()
    else:
        raise ValueError(
            f"{name} must be scalar, have {shape[1]} horizon values, "
            f"or match forecast shape {shape}; got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def calibrate_event_bounds(
    calibration_forecasts: Iterable[np.ndarray],
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
    margin=0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit fixed per-horizon bounds on data that is allowed for calibration.

    The returned bounds are not themselves differentially private. Callers must
    treat the calibration forecasts and the resulting bounds as public or
    otherwise permitted inputs to the mechanism.
    """
    q_low = float(lower_quantile)
    q_high = float(upper_quantile)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "DP quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1."
        )

    matrices = [_as_2d(x, "calibration forecast") for x in calibration_forecasts]
    if not matrices:
        raise ValueError("At least one calibration forecast matrix is required.")
    horizons = {x.shape[1] for x in matrices}
    if len(horizons) != 1:
        raise ValueError(f"Calibration forecasts have inconsistent horizons: {sorted(horizons)}.")

    pooled = np.concatenate(matrices, axis=0)
    lower = np.quantile(pooled, q_low, axis=0)
    upper = np.quantile(pooled, q_high, axis=0)
    margin_h = _as_horizon_vector(margin, horizon=pooled.shape[1], name="DP bound margin")
    if np.any(margin_h < 0.0):
        raise ValueError("DP bound margin must be non-negative.")
    return lower - margin_h, upper + margin_h


def calibrate_time_slot_event_bounds(
    calibration_forecast: np.ndarray,
    time_slot_indices: np.ndarray,
    slots_per_day: int,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    iqr_margin: float = 0.0,
    physical_lower: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one robust scalar interval per local time-of-day slot."""
    values = _as_2d(calibration_forecast, "calibration forecast")
    slots = np.asarray(time_slot_indices, dtype=np.int64)
    if slots.shape != values.shape:
        raise ValueError(
            f"time_slot_indices shape {slots.shape} does not match forecast shape {values.shape}."
        )
    n_slots = int(slots_per_day)
    if n_slots <= 0:
        raise ValueError("dp_slots_per_day must be greater than zero.")
    if np.any(slots < 0) or np.any(slots >= n_slots):
        raise ValueError("time_slot_indices contains an invalid slot.")

    q_low = float(lower_quantile)
    q_high = float(upper_quantile)
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            "DP quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1."
        )
    margin = float(iqr_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("dp_iqr_margin must be finite and non-negative.")

    fallback_lower = float(np.quantile(values, q_low))
    fallback_upper = float(np.quantile(values, q_high))
    fallback_iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    lower = np.empty(n_slots, dtype=np.float64)
    upper = np.empty(n_slots, dtype=np.float64)
    for slot in range(n_slots):
        slot_values = values[slots == slot]
        if slot_values.size == 0:
            lower[slot] = fallback_lower - margin * fallback_iqr
            upper[slot] = fallback_upper + margin * fallback_iqr
        else:
            slot_iqr = float(
                np.quantile(slot_values, 0.75) - np.quantile(slot_values, 0.25)
            )
            lower[slot] = float(np.quantile(slot_values, q_low)) - margin * slot_iqr
            upper[slot] = float(np.quantile(slot_values, q_high)) + margin * slot_iqr
    if physical_lower is not None:
        floor = float(physical_lower)
        if not np.isfinite(floor):
            raise ValueError("dp_physical_lower must be finite or None.")
        lower = np.maximum(lower, floor)
    return lower, upper


def privatize_event_forecast(
    forecast: np.ndarray,
    lower,
    upper,
    epsilon: float,
    mechanism: str = "laplace",
    delta: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Clip and randomise one forecast release under event replacement adjacency.

    One event is one scalar coordinate in the released ``(time, horizon)``
    forecast matrix. Bounds may vary by horizon but must be fixed independently
    of the forecast being released.
    """
    raw = _as_2d(forecast, "forecast")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("dp_epsilon must be finite and greater than zero.")
    lower_bound = _broadcast_forecast_bound(lower, raw.shape, "DP lower bound")
    upper_bound = _broadcast_forecast_bound(upper, raw.shape, "DP upper bound")
    if np.any(upper_bound < lower_bound):
        raise ValueError("DP upper bound is smaller than lower bound.")
    widths = upper_bound - lower_bound
    clipped = np.clip(raw, lower_bound, upper_bound)
    rng = np.random.default_rng() if rng is None else rng

    mech = str(mechanism).strip().lower()
    if mech == "laplace":
        scales = widths / eps
        noise = rng.laplace(loc=0.0, scale=scales, size=raw.shape)
        privacy_delta = 0.0
    elif mech == "gaussian":
        privacy_delta = float(delta)
        if not (0.0 < privacy_delta < 1.0):
            raise ValueError("Gaussian event-level DP requires 0 < dp_delta < 1.")
        if eps > 1.0:
            raise ValueError(
                "The implemented classical Gaussian calibration requires dp_epsilon <= 1."
            )
        multiplier = np.sqrt(2.0 * np.log(1.25 / privacy_delta)) / eps
        scales = widths * multiplier
        noise = rng.normal(loc=0.0, scale=scales, size=raw.shape)
    else:
        raise ValueError("dp_mechanism must be 'laplace' or 'gaussian'.")

    private = clipped + noise
    metadata: Dict[str, object] = {
        "enabled": True,
        "privacy_level": "event",
        "adjacency": "replacement; one scalar forecast coordinate differs",
        "mechanism": mech,
        "epsilon_per_release": eps,
        "delta_per_release": privacy_delta,
        "lower": lower_bound.copy(),
        "upper": upper_bound.copy(),
        "sensitivity_by_coordinate": widths.copy(),
        "sensitivity_by_horizon": np.max(widths, axis=0),
        "max_l1_sensitivity": float(np.max(widths)),
        "max_l2_sensitivity": float(np.max(widths)),
        "noise_scale_by_coordinate": scales.copy(),
        "noise_scale_by_horizon": np.max(scales, axis=0),
        "num_scalar_coordinates": int(raw.size),
        "clipped_coordinate_count": int(np.count_nonzero(raw != clipped)),
    }
    return private.astype(np.float64), metadata


def apply_prediction_dp(clients: Dict[int, object], args) -> Dict[str, object]:
    """Apply the configured DP mechanism locally to every client's base forecast."""
    enabled = bool(getattr(args, "dp_enabled", False))
    if not enabled:
        for client in clients.values():
            previous = getattr(client, "dp_metadata", {})
            if bool(previous.get("enabled", False)):
                raw = getattr(client, "nonprivate_base_forecast", None)
                if raw is not None:
                    client.base_forecast = np.asarray(raw, dtype=np.float64).copy()
            client.reconciliation_base_forecast = client.base_forecast
            client.dp_metadata = {"enabled": False, "release_count": 0}
        return {"enabled": False, "release_count_per_client": 0}

    level = str(getattr(args, "dp_privacy_level", "event")).strip().lower()
    if level != "event":
        raise ValueError(
            "This implementation currently supports dp_privacy_level='event' only. "
            "w-event and series-level require their larger sensitivity calibration."
        )

    ordered_ids = sorted(int(cid) for cid in clients)
    if not ordered_ids:
        raise ValueError("Cannot apply prediction DP without clients.")
    first = _as_2d(clients[ordered_ids[0]].base_forecast, "base forecast")
    horizon = first.shape[1]
    source = str(getattr(args, "dp_bounds_source", "train_predictions")).strip().lower()
    scope = str(getattr(args, "dp_bounds_scope", "per_client")).strip().lower()
    if scope not in {"per_client", "per_client_time_slot", "global"}:
        raise ValueError(
            "dp_bounds_scope must be 'per_client', 'per_client_time_slot', or 'global'."
        )

    time_slot_bounds_by_cid = {}

    if source == "explicit":
        raw_lower = getattr(args, "dp_lower", None)
        raw_upper = getattr(args, "dp_upper", None)
        bounds_by_cid = {}
        for cid in ordered_ids:
            client_lower = raw_lower.get(cid) if isinstance(raw_lower, dict) else raw_lower
            client_upper = raw_upper.get(cid) if isinstance(raw_upper, dict) else raw_upper
            bounds_by_cid[cid] = validate_event_bounds(
                client_lower, client_upper, horizon=horizon
            )
        data_dependent_release_bounds = False
    elif source in {"train_predictions", "test_predictions"}:
        calibration_by_cid = {}
        calibration_slots_by_cid = {}
        for cid in ordered_ids:
            client = clients[cid]
            if source == "train_predictions":
                values = getattr(client, "y1", None)
                if values is None:
                    raise ValueError(
                        f"Client {cid} has no training predictions for DP bound calibration."
                    )
            else:
                values = getattr(client, "nonprivate_base_forecast", None)
                if values is None:
                    values = client.base_forecast
            calibration_by_cid[cid] = values
            if scope == "per_client_time_slot":
                slot_attr = (
                    "train_time_slot_indices"
                    if source == "train_predictions"
                    else "test_time_slot_indices"
                )
                slot_values = getattr(client, slot_attr, None)
                if slot_values is None:
                    raise ValueError(
                        f"Client {cid} has no {slot_attr}; a time column is required "
                        "for per-client time-slot DP bounds."
                    )
                calibration_slots_by_cid[cid] = slot_values
        if scope == "global":
            common_bounds = calibrate_event_bounds(
                calibration_by_cid.values(),
                lower_quantile=float(getattr(args, "dp_lower_quantile", 0.0)),
                upper_quantile=float(getattr(args, "dp_upper_quantile", 1.0)),
                margin=getattr(args, "dp_bound_margin", 0.0),
            )
            bounds_by_cid = {cid: common_bounds for cid in ordered_ids}
        elif scope == "per_client":
            bounds_by_cid = {
                cid: calibrate_event_bounds(
                    [calibration_by_cid[cid]],
                    lower_quantile=float(getattr(args, "dp_lower_quantile", 0.0)),
                    upper_quantile=float(getattr(args, "dp_upper_quantile", 1.0)),
                    margin=getattr(args, "dp_bound_margin", 0.0),
                )
                for cid in ordered_ids
            }
        else:
            slots_per_day = int(getattr(args, "dp_slots_per_day", 48))
            bounds_by_cid = {}
            for cid in ordered_ids:
                lower_slots, upper_slots = calibrate_time_slot_event_bounds(
                    calibration_by_cid[cid],
                    calibration_slots_by_cid[cid],
                    slots_per_day=slots_per_day,
                    lower_quantile=float(getattr(args, "dp_lower_quantile", 0.01)),
                    upper_quantile=float(getattr(args, "dp_upper_quantile", 0.99)),
                    iqr_margin=float(getattr(args, "dp_iqr_margin", 0.0)),
                    physical_lower=getattr(args, "dp_physical_lower", None),
                )
                release_slots = getattr(clients[cid], "test_time_slot_indices", None)
                if release_slots is None:
                    raise ValueError(
                        f"Client {cid} has no test_time_slot_indices for time-slot DP."
                    )
                release_slots = np.asarray(release_slots, dtype=np.int64)
                bounds_by_cid[cid] = (
                    lower_slots[release_slots],
                    upper_slots[release_slots],
                )
                time_slot_bounds_by_cid[cid] = {
                    "lower_by_time_slot": lower_slots,
                    "upper_by_time_slot": upper_slots,
                    "sensitivity_by_time_slot": upper_slots - lower_slots,
                }
        data_dependent_release_bounds = source == "test_predictions"
    else:
        raise ValueError(
            "dp_bounds_source must be 'explicit', 'train_predictions', or 'test_predictions'."
        )

    configured_seed = getattr(args, "dp_seed", None)
    mechanism = str(getattr(args, "dp_mechanism", "laplace"))
    epsilon = float(getattr(args, "dp_epsilon", 1.0))
    delta = float(getattr(args, "dp_delta", 1e-5))

    for cid in ordered_ids:
        client = clients[cid]
        previous_metadata = getattr(client, "dp_metadata", {})
        previous_count = (
            int(previous_metadata.get("release_count", 0))
            if bool(previous_metadata.get("enabled", False))
            else 0
        )
        raw_source = (
            client.nonprivate_base_forecast
            if previous_count > 0 and client.nonprivate_base_forecast is not None
            else client.base_forecast
        )
        raw = _as_2d(raw_source, "base forecast")
        if raw.shape[1] != horizon:
            raise ValueError("All clients must share a forecast horizon for common DP bounds.")
        client.nonprivate_base_forecast = raw.copy()
        release_count = previous_count + 1
        if configured_seed is None:
            # OS-backed entropy through NumPy's SeedSequence. A fixed seed is
            # useful for experiments but must not be public in deployment.
            client_seed = np.random.SeedSequence()
        else:
            client_seed = np.random.SeedSequence(
                [int(configured_seed), int(cid), int(release_count)]
            )
        private, metadata = privatize_event_forecast(
            raw,
            lower=bounds_by_cid[cid][0],
            upper=bounds_by_cid[cid][1],
            epsilon=epsilon,
            mechanism=mechanism,
            delta=delta,
            rng=np.random.default_rng(client_seed),
        )
        client.base_forecast = private
        client.reconciliation_base_forecast = private
        metadata.update(
            {
                "bounds_source": source,
                "bounds_scope": scope,
                "bounds_are_release_data_dependent": data_dependent_release_bounds,
                "release_count": release_count,
                "sequential_composition_epsilon_upper_bound": release_count * epsilon,
                "sequential_composition_delta_upper_bound": float(
                    release_count * metadata["delta_per_release"]
                ),
                "deterministic_experiment_seed": configured_seed is not None,
            }
        )
        client.dp_metadata = metadata

    return {
        "enabled": True,
        "privacy_level": "event",
        "mechanism": mechanism.lower(),
        "epsilon_per_release": epsilon,
        "delta_per_release": 0.0 if mechanism.lower() == "laplace" else delta,
        "bounds_source": source,
        "bounds_scope": scope,
        "bounds_are_release_data_dependent": data_dependent_release_bounds,
        "bounds_by_client": (
            time_slot_bounds_by_cid
            if scope == "per_client_time_slot"
            else {
                cid: {
                    "lower": bounds_by_cid[cid][0].copy(),
                    "upper": bounds_by_cid[cid][1].copy(),
                    "sensitivity_by_horizon": (
                        bounds_by_cid[cid][1] - bounds_by_cid[cid][0]
                    ).copy(),
                }
                for cid in ordered_ids
            }
        ),
        "release_count_per_client": max(
            int(clients[cid].dp_metadata["release_count"]) for cid in ordered_ids
        ),
        "deterministic_experiment_seed": configured_seed is not None,
        "protects": "released base forecast coordinates only",
    }


def dp_noise_variance_by_client(clients: Dict[int, object]) -> Dict[int, np.ndarray]:
    """Return each client's mean DP noise variance for every forecast horizon."""
    result: Dict[int, np.ndarray] = {}
    for raw_cid, client in clients.items():
        metadata = getattr(client, "dp_metadata", {})
        if not bool(metadata.get("enabled", False)):
            continue
        scales = _as_2d(metadata["noise_scale_by_coordinate"], "DP noise scale")
        mechanism = str(metadata.get("mechanism", "")).lower()
        if mechanism == "laplace":
            variance = 2.0 * scales**2
        elif mechanism == "gaussian":
            variance = scales**2
        else:
            raise ValueError(f"Unsupported DP mechanism in metadata: {mechanism}")
        result[int(raw_cid)] = np.mean(variance, axis=0)
    return result
