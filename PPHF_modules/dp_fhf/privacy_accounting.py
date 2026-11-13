from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import math
import logging as python_logging
import sys
from pathlib import Path
from typing import Iterable


def check_sampling_accountant_dependencies():
    """Load the sampled accountant before spending time on model training."""
    dependency_dir = Path(__file__).with_name("_privacy_dependencies")
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))
    try:
        import attr  # Provided by the attrs distribution, not a package named attr.
        from dp_accounting import dp_event, privacy_accountant
        from dp_accounting.rdp import rdp_privacy_accountant
    except ImportError as exc:
        raise RuntimeError(
            "Sampling privacy-accounting dependencies are unavailable. Training has not started. "
            f"Python: {sys.executable}. Original error: {exc}. "
            "In this notebook kernel run: import subprocess, sys; "
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', "
            "'attrs>=22,<24', 'mpmath==1.3.0']). "
            "The import name attr belongs to the attrs distribution."
        ) from exc
    return dp_event, privacy_accountant, rdp_privacy_accountant


def gaussian_composition_epsilon(
    noise_multiplier: float,
    aggregation_calls: int,
    delta: float,
    orders: Iterable[float] | None = None,
) -> dict:
    """Conservative RDP accounting for repeated full Gaussian mechanisms.

    This deliberately does not claim privacy amplification from client
    sampling. Each call is treated as a Gaussian mechanism with sensitivity
    calibrated by the client-side clipping bound and server-side
    ``dp_fedavg_aggregate`` sensitivity, with standard deviation
    ``noise_multiplier * sensitivity``.
    """
    sigma = float(noise_multiplier)
    calls = int(aggregation_calls)
    target_delta = float(delta)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("noise_multiplier must be finite and greater than zero.")
    if calls <= 0:
        raise ValueError("aggregation_calls must be greater than zero.")
    if not (0.0 < target_delta < 1.0):
        raise ValueError("delta must satisfy 0 < delta < 1.")

    candidate_orders = (
        [1.25, 1.5, 1.75]
        + list(range(2, 65))
        + [96, 128, 192, 256, 384, 512]
        if orders is None
        else [float(order) for order in orders]
    )
    if not candidate_orders or any(order <= 1.0 for order in candidate_orders):
        raise ValueError("Every RDP order must be greater than one.")

    candidates = []
    for order in candidate_orders:
        rdp = calls * order / (2.0 * sigma * sigma)
        epsilon = rdp + math.log(1.0 / target_delta) / (order - 1.0)
        candidates.append((epsilon, order, rdp))
    epsilon, optimal_order, rdp = min(candidates, key=lambda item: item[0])
    return {
        "epsilon": float(epsilon),
        "delta": target_delta,
        "noise_multiplier": sigma,
        "aggregation_calls": calls,
        "optimal_rdp_order": float(optimal_order),
        "rdp_at_optimal_order": float(rdp),
        "sampling_amplification_used": False,
        "accounting": "full-participation Gaussian RDP composition",
    }


def sampled_fedavg_epsilon(noise_multiplier: float, aggregation_calls: int, delta: float,
                           population_size: int, sample_size: int, adjacency: str = "add_remove") -> dict:
    """Fixed-size WOR Gaussian RDP, with the actual m/N (not requested q).

    Add/remove: Birrell et al. (2024), Theorem 3.1, Eq. (6). For clipped
    vectors bounded by C and noise std s, its dominating Gaussian mixture has
    multiplier s/(2C/m). Our amplified training path sets s=2*sigma*C/m.
    The mixture moment is evaluated by Google's Poisson Gaussian accountant;
    the sampler itself remains fixed-size WOR. This is a theorem reduction,
    not an assumption that WOR samples are Poisson samples.

    Replace-one: Google's SampledWithoutReplacementDpEvent (WBK 2019), where
    sigma is noise std divided by the replace-one sensitivity 2C/m.
    """
    base = gaussian_composition_epsilon(noise_multiplier, aggregation_calls, delta)
    if (not isinstance(population_size, int) or not isinstance(sample_size, int)
            or not 1 <= sample_size <= population_size):
        raise ValueError("Require integer sample_size and population_size with 1 <= m <= N.")
    if adjacency not in {"add_remove", "replace_one"}:
        raise ValueError("Unknown adjacency.")
    if sample_size == population_size:
        return base
    dp_event, privacy_accountant, rdp = check_sampling_accountant_dependencies()
    orders = [1 + x / 100 for x in range(1, 100)] + [2 + x / 10 for x in range(100)] + list(range(12, 65)) + [96, 128, 192, 256, 384, 512]
    q = sample_size / population_size
    if adjacency == "add_remove":
        event = dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(noise_multiplier))
        relation = privacy_accountant.NeighboringRelation.ADD_OR_REMOVE_ONE
        reference = "https://arxiv.org/html/2408.10456v1#S3.SS2.SSS1"
    else:
        event = dp_event.SampledWithoutReplacementDpEvent(population_size, sample_size, dp_event.GaussianDpEvent(noise_multiplier))
        relation = privacy_accountant.NeighboringRelation.REPLACE_ONE
        reference = "https://arxiv.org/abs/1808.00087"
    accountant = rdp.RdpAccountant(orders=orders, neighboring_relation=relation)
    # Nonconvergent fractional orders are rejected by the library (infinity),
    # so only their unsampled Gaussian bound is used. Record them rather than printing one
    # library warning per candidate order into every notebook.
    from absl import logging
    logger = logging.get_absl_logger()
    previous_level = logger.level
    try:
        logger.setLevel(python_logging.ERROR)
        accountant.compose(event, count=aggregation_calls)
    finally:
        logger.setLevel(previous_level)
    # Same basic RDP-to-DP conversion as the historical accountant, with a
    # denser order grid for the small-sigma regime. Keep q=1 exactly unchanged.
    full_rdps = [aggregation_calls * float(a) / (2 * noise_multiplier**2) for a in accountant.orders]
    used_rdps = [min(float(r), full) for r, full in zip(accountant.rdp, full_rdps)]
    candidates = [r + math.log(1 / delta) / (a - 1) for a, r in zip(accountant.orders, used_rdps)]
    best = min(range(len(candidates)), key=candidates.__getitem__)
    same_grid_without_sampling = min(r + math.log(1 / delta) / (a - 1) for a, r in zip(accountant.orders, full_rdps))
    return {
        "epsilon": candidates[best], "delta": delta, "noise_multiplier": noise_multiplier,
        "aggregation_calls": aggregation_calls, "optimal_rdp_order": float(accountant.orders[best]),
        "rdp_at_optimal_order": used_rdps[best], "sampling_amplification_used": True,
        "accounting": "fixed-size without-replacement Gaussian RDP composition",
        "adjacency": adjacency, "population_size": population_size, "sample_size": sample_size,
        "effective_sampling_rate": q, "epsilon_without_sampling_amplification": same_grid_without_sampling,
        "historical_full_participation_epsilon": base["epsilon"],
        "noise_std_formula": "2 * noise_multiplier * clip_norm / sample_size",
        "reference": reference, "accountant_library": "Google dp-accounting 0.6.0",
        "excluded_nonconvergent_orders": [float(a) for a, r in zip(accountant.orders, accountant.rdp) if not math.isfinite(r)],
        "scope": "released global model trajectory only; excludes local forecasts, residuals, targets and diagnostics",
        "assumptions": ["Fixed public sample size and public population-size calibration",
                        "Uniform weights and independently drawn hidden subsets each round",
                        "Fresh hidden sampling and Gaussian randomness", "Trusted central aggregation server"],
    }
