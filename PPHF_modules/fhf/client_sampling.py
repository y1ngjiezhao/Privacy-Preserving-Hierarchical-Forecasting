"""Client selection shared by ordinary and DP federated training.

Sampling is uniform without replacement with ceil(q * N) participants. The
caller must use an accountant and noise calibration matching this scheme.
The q=1 path keeps the original order and consumes no random numbers.
"""
from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Sequence

def validate_sampling_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("sampling_rate must be finite and satisfy 0 < sampling_rate <= 1.")
    return rate


def select_clients(client_ids: Sequence[int], sampling_rate: float, rng, *, sample_size: int | None = None) -> list[int]:
    rate = validate_sampling_rate(sampling_rate)
    ids = list(client_ids)
    if not ids or rate == 1.0:
        return ids
    # Decimal avoids e.g. ceil(0.07 * 100) accidentally becoming 8.
    count = fixed_sample_size(len(ids), rate) if sample_size is None else sample_size
    if not isinstance(count, int) or not 1 <= count <= len(ids):
        raise ValueError("sample_size must be an integer between 1 and the population size.")
    if count == len(ids):
        return ids
    selected = set(rng.choice(ids, size=count, replace=False).tolist())
    # Keep training and floating-point aggregation in the population's order.
    return [cid for cid in ids if cid in selected]


def fixed_sample_size(population_size: int, sampling_rate: float) -> int:
    return min(population_size, max(1, math.ceil(Decimal(str(validate_sampling_rate(sampling_rate))) * population_size)))


def amplification_output_dir(path, enabled: bool) -> Path:
    path = Path(path)
    if not enabled or any(p.name == "amplified_v1" for p in (path, *path.parents)):
        return path
    return path / "amplified_v1"


def sampling_run_tag(sampling_rate: float, sampling_seed: int) -> str:
    rate = validate_sampling_rate(sampling_rate)
    slug = repr(rate).replace(".", "p")
    return f"q_{slug}__seed_{'fresh' if sampling_seed is None else int(sampling_seed)}"


def sampled_output_dir(path, sampling_rate: float, sampling_seed: int) -> Path:
    """Keep partial-participation runs away from historical result paths."""
    path = Path(path)
    if validate_sampling_rate(sampling_rate) == 1.0:
        return path
    tag = sampling_run_tag(sampling_rate, sampling_seed)
    # Both the notebook helper and CLI use this function.
    if any(p.name == tag and p.parent.name == "sampling_runs" for p in (path, *path.parents)):
        return path
    return path / "sampling_runs" / tag


def sampling_metadata(rate: float, seed: int, population_size: int, *, scope: str) -> dict:
    return {
        "sampling_rate": rate,
        "sampling_seed": seed,
        "sampling_scheme": "uniform_without_replacement",
        "sampling_scope": scope,
        "population_size": population_size,
        "sample_size_rule": "ceil(q * N), at least one for nonempty populations",
    }
