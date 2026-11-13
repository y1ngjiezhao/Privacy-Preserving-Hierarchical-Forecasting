from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import hashlib
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class DPCovarianceConfig:
    """Configuration for local residual releases used to estimate W."""

    epsilon: float = 1.0
    mechanism: str = "laplace"
    delta: float = 1e-5
    lower: object = -1.0
    upper: object = 1.0
    seed: int | None = None
    ridge: float = 1e-6
    debias_noise_variance: bool = True


def _as_2d(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {array.shape}.")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite.")
    return array


def _broadcast_bound(values, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=np.float64)
    elif array.ndim == 1 and array.size == shape[1]:
        array = np.broadcast_to(array[None, :], shape).copy()
    elif array.shape == shape:
        array = array.copy()
    else:
        raise ValueError(
            f"{name} must be scalar, one value per horizon, or shape {shape}; "
            f"got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _stable_rng(seed: int | None, client_id: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    digest = hashlib.sha256(f"dp-covariance:{int(client_id)}".encode()).digest()
    client_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), client_seed])
    )


def privatize_residual_matrix(
    residuals,
    lower,
    upper,
    epsilon: float,
    mechanism: str = "laplace",
    delta: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, dict]:
    """Clip and locally randomise one residual matrix before upload.

    The protected event is one scalar ``(training time, horizon)`` residual
    coordinate under replacement adjacency. Bounds must be fixed without
    consulting the release being protected for a formal DP claim.
    """
    raw = _as_2d(residuals, "residuals")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero.")
    lower_matrix = _broadcast_bound(lower, raw.shape, "lower")
    upper_matrix = _broadcast_bound(upper, raw.shape, "upper")
    if np.any(upper_matrix < lower_matrix):
        raise ValueError("upper must be greater than or equal to lower.")

    widths = upper_matrix - lower_matrix
    clipped = np.clip(raw, lower_matrix, upper_matrix)
    local_rng = np.random.default_rng() if rng is None else rng
    mechanism_name = str(mechanism).strip().lower()
    if mechanism_name == "laplace":
        scales = widths / eps
        noise = local_rng.laplace(0.0, scales, size=raw.shape)
        noise_variance = 2.0 * scales * scales
        release_delta = 0.0
    elif mechanism_name == "gaussian":
        release_delta = float(delta)
        if not (0.0 < release_delta < 1.0):
            raise ValueError("Gaussian residual DP requires 0 < delta < 1.")
        if eps > 1.0:
            raise ValueError(
                "The classical Gaussian calibration used here requires epsilon <= 1."
            )
        multiplier = np.sqrt(2.0 * np.log(1.25 / release_delta)) / eps
        scales = widths * multiplier
        noise = local_rng.normal(0.0, scales, size=raw.shape)
        noise_variance = scales * scales
    else:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'.")

    private = clipped + noise
    metadata = {
        "enabled": True,
        "privacy_level": "event",
        "event_definition": "one scalar local training residual coordinate",
        "adjacency": "replacement",
        "mechanism": mechanism_name,
        "epsilon_per_release": eps,
        "delta_per_release": release_delta,
        "lower": lower_matrix,
        "upper": upper_matrix,
        "sensitivity_by_coordinate": widths,
        "noise_scale_by_coordinate": scales,
        "noise_variance_by_coordinate": noise_variance,
        "mean_noise_variance_by_horizon": np.mean(noise_variance, axis=0),
        "clipped_coordinate_count": int(np.count_nonzero(raw != clipped)),
        "coordinate_count": int(raw.size),
        "release_count": 1,
    }
    return private.astype(np.float64), metadata


def _client_bound(specification, client_id: int):
    if isinstance(specification, Mapping):
        if client_id in specification:
            return specification[client_id]
        if str(client_id) in specification:
            return specification[str(client_id)]
        raise KeyError(f"No residual bound configured for client {client_id}.")
    return specification


def privatize_residuals_by_client(
    residuals_by_client: Mapping[int, np.ndarray],
    config: DPCovarianceConfig,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Simulate local residual releases; only returned matrices are uploaded."""
    private_residuals: Dict[int, np.ndarray] = {}
    metadata: Dict[int, dict] = {}
    for raw_client_id in sorted(residuals_by_client):
        client_id = int(raw_client_id)
        private, local_metadata = privatize_residual_matrix(
            residuals=residuals_by_client[raw_client_id],
            lower=_client_bound(config.lower, client_id),
            upper=_client_bound(config.upper, client_id),
            epsilon=config.epsilon,
            mechanism=config.mechanism,
            delta=config.delta,
            rng=_stable_rng(config.seed, client_id),
        )
        local_metadata["client_id"] = client_id
        local_metadata["seed_is_fixed_for_research"] = config.seed is not None
        private_residuals[client_id] = private
        metadata[client_id] = local_metadata
    if not private_residuals:
        raise ValueError("residuals_by_client must not be empty.")
    return private_residuals, metadata


def _prepare_cube(
    client_ids: Sequence[int],
    residuals_by_client: Mapping[int, np.ndarray],
) -> np.ndarray:
    matrices = []
    for client_id in client_ids:
        if int(client_id) not in residuals_by_client:
            raise KeyError(f"Missing residual release for client {client_id}.")
        matrices.append(
            _as_2d(residuals_by_client[int(client_id)], "private residuals")
        )
    horizons = {matrix.shape[1] for matrix in matrices}
    if len(horizons) != 1:
        raise ValueError("All clients must have the same residual horizon.")
    common_length = min(matrix.shape[0] for matrix in matrices)
    return np.stack(
        [matrix[-common_length:] for matrix in matrices], axis=0
    )


def _nearest_psd(matrix: np.ndarray, eigenvalue_floor: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(eigenvalue_floor))
    return (eigenvectors * clipped) @ eigenvectors.T


def estimate_private_covariances(
    client_ids: Sequence[int],
    private_residuals: Mapping[int, np.ndarray],
    metadata_by_client: Mapping[int, dict],
    ridge: float = 1e-6,
    debias_noise_variance: bool = True,
) -> np.ndarray:
    """Estimate one PSD residual covariance matrix per forecast horizon."""
    ids = [int(client_id) for client_id in client_ids]
    cube = _prepare_cube(ids, private_residuals)
    _, time_length, horizon = cube.shape
    estimates = []
    for horizon_index in range(horizon):
        residual_matrix = cube[:, :, horizon_index].T
        if time_length <= 1:
            covariance = np.eye(len(ids), dtype=np.float64)
        else:
            covariance = np.asarray(
                np.cov(residual_matrix, rowvar=False), dtype=np.float64
            )
        if debias_noise_variance:
            noise_variances = np.array(
                [
                    np.asarray(
                        metadata_by_client[client_id][
                            "mean_noise_variance_by_horizon"
                        ],
                        dtype=np.float64,
                    )[horizon_index]
                    for client_id in ids
                ]
            )
            covariance = covariance - np.diag(noise_variances)
        estimates.append(_nearest_psd(covariance, eigenvalue_floor=ridge))
    return np.stack(estimates, axis=0)


def estimate_private_shrinkage_covariances(
    client_ids: Sequence[int],
    private_residuals: Mapping[int, np.ndarray],
    metadata_by_client: Mapping[int, dict],
    ridge: float = 1e-6,
    debias_noise_variance: bool = True,
) -> np.ndarray:
    """Estimate one PSD Ledoit-Wolf covariance per forecast horizon.

    The estimator is fitted only to locally privatized residual uploads.
    Noise-variance correction and PSD projection are DP post-processing.
    """
    from sklearn.covariance import LedoitWolf

    ids = [int(client_id) for client_id in client_ids]
    cube = _prepare_cube(ids, private_residuals)
    _, time_length, horizon = cube.shape
    estimates = []
    for horizon_index in range(horizon):
        residual_matrix = cube[:, :, horizon_index].T
        if time_length <= 1:
            covariance = np.eye(len(ids), dtype=np.float64)
        else:
            covariance = np.asarray(
                LedoitWolf().fit(residual_matrix).covariance_,
                dtype=np.float64,
            )
        if debias_noise_variance:
            noise_variances = np.array(
                [
                    np.asarray(
                        metadata_by_client[client_id][
                            "mean_noise_variance_by_horizon"
                        ],
                        dtype=np.float64,
                    )[horizon_index]
                    for client_id in ids
                ]
            )
            covariance = covariance - np.diag(noise_variances)
        estimates.append(_nearest_psd(covariance, eigenvalue_floor=ridge))
    return np.stack(estimates, axis=0)


def _bottom_selection_matrix(
    summing_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(summing_matrix, dtype=np.float64)
    bottom_count = matrix.shape[1]
    selector = np.zeros((bottom_count, matrix.shape[0]), dtype=np.float64)
    for bottom_index in range(bottom_count):
        candidates = np.flatnonzero(
            np.all(
                np.isclose(
                    matrix,
                    np.eye(bottom_count, dtype=np.float64)[bottom_index],
                ),
                axis=1,
            )
        )
        if candidates.size != 1:
            raise ValueError(
                "Could not identify exactly one row for every bottom-level series."
            )
        selector[bottom_index, candidates[0]] = 1.0
    return selector


def _mint_projection(
    summing_matrix: np.ndarray, covariance: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    s_matrix = np.asarray(summing_matrix, dtype=np.float64)
    inverse = np.linalg.pinv(covariance)
    g_matrix = np.linalg.pinv(s_matrix.T @ inverse @ s_matrix) @ s_matrix.T @ inverse
    return s_matrix @ g_matrix, g_matrix


def build_private_reconciliation_matrices(
    summing_matrix: np.ndarray,
    client_ids: Sequence[int],
    residuals_by_client: Mapping[int, np.ndarray],
    config: DPCovarianceConfig,
    methods: Sequence[str] = ("mint_shrinkage", "wlsv"),
) -> dict:
    """Local-DP residual upload, private W estimation, then P-column download."""
    private_residuals, metadata = privatize_residuals_by_client(
        residuals_by_client, config
    )
    covariance_by_horizon = estimate_private_covariances(
        client_ids=client_ids,
        private_residuals=private_residuals,
        metadata_by_client=metadata,
        ridge=config.ridge,
        debias_noise_variance=config.debias_noise_variance,
    )
    shrinkage_covariance_by_horizon = estimate_private_shrinkage_covariances(
        client_ids=client_ids,
        private_residuals=private_residuals,
        metadata_by_client=metadata,
        ridge=config.ridge,
        debias_noise_variance=config.debias_noise_variance,
    )
    canonical_methods = []
    aliases = {
        "mint": "mint_cov",
        "mint_cov": "mint_cov",
        "mints": "mint_shrinkage",
        "mint_shr": "mint_shrinkage",
        "mint_shrinkage": "mint_shrinkage",
        "wlsv": "mint_var",
        "mint_var": "mint_var",
    }
    for raw_method in methods:
        key = str(raw_method).strip().lower().replace("-", "_")
        if key not in aliases:
            raise ValueError(
                "DP covariance supports 'mint_shrinkage'/'mints', "
                "'wlsv'/'mint_var', and 'mint_cov'."
            )
        canonical = aliases[key]
        if canonical not in canonical_methods:
            canonical_methods.append(canonical)

    ps: Dict[str, np.ndarray] = {}
    gs: Dict[str, np.ndarray] = {}
    ws: Dict[str, np.ndarray] = {}
    for method in canonical_methods:
        if method == "mint_cov":
            method_covariances = covariance_by_horizon
        elif method == "mint_shrinkage":
            method_covariances = shrinkage_covariance_by_horizon
        else:
            method_covariances = np.stack(
                [np.diag(np.diag(matrix)) for matrix in covariance_by_horizon]
            )
        p_list, g_list = [], []
        for covariance in method_covariances:
            p_matrix, g_matrix = _mint_projection(summing_matrix, covariance)
            p_list.append(p_matrix)
            g_list.append(g_matrix)
        ps[method] = np.stack(p_list)
        gs[method] = np.stack(g_list)
        ws[method] = method_covariances

    ids = [int(client_id) for client_id in client_ids]
    downloaded_columns = {
        client_id: {
            method: tensor[:, :, sender_index].transpose(1, 0)
            for method, tensor in ps.items()
        }
        for sender_index, client_id in enumerate(ids)
    }
    return {
        "private_residuals": private_residuals,
        "metadata_by_client": metadata,
        "covariance_by_horizon": covariance_by_horizon,
        "shrinkage_covariance_by_horizon": shrinkage_covariance_by_horizon,
        "Ws_by_method": ws,
        "Gs": gs,
        "Ps": ps,
        "downloaded_P_columns_by_client": downloaded_columns,
        "communication": {
            "upload": "one locally privatized residual matrix per client",
            "download": "one P column per client, method, and horizon",
            "raw_residuals_uploaded": False,
        },
    }


def install_and_distribute_private_reconciliation(
    nodes: Mapping[int, object],
    client_ids: Sequence[int],
    private_result: dict,
    validate: bool = True,
    atol: float = 1e-8,
) -> None:
    """Install private W/G/P estimates and reconcile via downloaded P columns."""
    ids = [int(client_id) for client_id in client_ids]
    if set(ids) != {int(client_id) for client_id in nodes}:
        raise ValueError("nodes must contain exactly the configured client IDs.")
    ps = private_result["Ps"]
    for client_id in ids:
        nodes[client_id].reset_reconciled_forecasts()

    for method, p_tensor in ps.items():
        for sender_index, sender_id in enumerate(ids):
            p_column = p_tensor[:, :, sender_index].T
            nodes[sender_id].set_P_column(p_column)
        for sender_id in ids:
            outgoing = nodes[sender_id].build_weighted_contributions()
            for recipient_index, recipient_id in enumerate(ids):
                nodes[recipient_id].add_reconciled_contribution(
                    outgoing[recipient_index], method=method
                )

        if validate:
            base_stack = np.stack(
                [
                    np.asarray(
                        getattr(nodes[client_id], "reconciliation_base_forecast", None)
                        if getattr(
                            nodes[client_id],
                            "reconciliation_base_forecast",
                            None,
                        )
                        is not None
                        else nodes[client_id].base_forecast,
                        dtype=np.float64,
                    )
                    for client_id in ids
                ]
            )
            central = np.empty_like(base_stack)
            for horizon_index in range(base_stack.shape[2]):
                central[:, :, horizon_index] = (
                    p_tensor[horizon_index] @ base_stack[:, :, horizon_index]
                )
            for recipient_index, recipient_id in enumerate(ids):
                difference = np.max(
                    np.abs(
                        nodes[recipient_id].reconciled_forecasts[method]
                        - central[recipient_index]
                    )
                )
                if difference > float(atol):
                    raise AssertionError(
                        f"Private P-column reconciliation mismatch for {method}, "
                        f"client {recipient_id}: {difference} > {atol}."
                    )

    if len(ps) == 1:
        only_method = next(iter(ps))
        for client_id in ids:
            nodes[client_id].reconciled_forecast = nodes[
                client_id
            ].reconciled_forecasts[only_method].copy()


def run_local_dp_covariance_reconciliation(
    nodes: Mapping[int, object],
    client_ids: Sequence[int],
    summing_matrix: np.ndarray,
    config: DPCovarianceConfig,
    methods: Sequence[str] = ("mint_shrinkage", "wlsv"),
    validate: bool = True,
) -> dict:
    """End-to-end local residual DP, coordinator estimation, and P download."""
    ids = [int(client_id) for client_id in client_ids]
    local_residuals = {}
    for client_id in ids:
        residuals = getattr(nodes[client_id], "residual_samples", None)
        if residuals is None:
            raise ValueError(
                f"Client {client_id} has no local residual_samples to privatize."
            )
        local_residuals[client_id] = np.asarray(residuals, dtype=np.float64)
    private_result = build_private_reconciliation_matrices(
        summing_matrix=summing_matrix,
        client_ids=ids,
        residuals_by_client=local_residuals,
        config=config,
        methods=methods,
    )
    install_and_distribute_private_reconciliation(
        nodes=nodes,
        client_ids=ids,
        private_result=private_result,
        validate=validate,
    )
    return private_result
