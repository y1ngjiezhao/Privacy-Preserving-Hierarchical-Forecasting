"""Counterfactual Poisson-sampling RDP budgets for saved 100-round runs.

This does not change training or certify existing fixed-seed releases.
Uses the sampled Gaussian log moment A_alpha (Mironov et al., 2019),
numerically integrated, and the project's basic RDP-to-DP conversion.
Only Python's standard library is required.
"""
from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import csv
import json
import math
from pathlib import Path

ROOT = DP_EXPERIMENT_DIR


def logadd(a: float, b: float) -> float:
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def sampled_rdp(q: float, z: float, alpha: float, step: float) -> float:
    if q == 1:
        return alpha / (2 * z * z)
    # A_alpha = E_X[(1-q + q exp(X/z - 1/(2z^2)))^alpha], X~N(0,1).
    # Include both Gaussian peaks with at least 12 standard deviations margin.
    lower, upper = -12.0, max(12.0, alpha / z + 12.0)
    n = math.ceil((upper - lower) / step)
    n += n % 2
    h = (upper - lower) / n
    values = []
    for i in range(n + 1):
        x = lower + i * h
        mixture = logadd(math.log1p(-q), math.log(q) + x / z - 1 / (2*z*z))
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        values.append(-x*x/2 + alpha*mixture + math.log(weight))
    peak = max(values)
    log_a = peak + math.log(sum(math.exp(v-peak) for v in values)*h/3)
    log_a -= math.log(2*math.pi)/2
    return log_a / (alpha - 1)


def budget(q: float, z: float, rounds: int, delta: float, step: float) -> tuple[float, float]:
    if q == 1:
        alpha = 1 + math.sqrt(2*z*z*math.log(1/delta)/rounds)
        return rounds*alpha/(2*z*z) + math.log(1/delta)/(alpha-1), alpha
    def objective(a):
        return rounds*sampled_rdp(q, z, a, step) + math.log(1/delta)/(a-1)
    lo, hi = 1.0001, 64.0
    ratio = (math.sqrt(5)-1)/2
    left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fl, fr = objective(left), objective(right)
    for _ in range(65):
        if fl < fr:
            hi, right, fr = right, left, fl
            left = hi-ratio*(hi-lo)
            fl = objective(left)
        else:
            lo, left, fl = left, right, fr
            right = lo+ratio*(hi-lo)
            fr = objective(right)
    alpha = (lo+hi)/2
    return objective(alpha), alpha


def main():
    out = ROOT / 'outputs' / 'sampling_privacy_accounting'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ('GEF2017', 'AusGrid2013'):
        source = ROOT/'outputs'/'01_dp_fedavg_100round'/dataset/'fh1_FHF_dp_fedavg_privacy.json'
        metadata = json.loads(source.read_text())
        z = metadata['noise_multiplier']
        rounds = metadata['aggregation_calls']
        delta = metadata['accounting']['delta']
        for q in (.1, .3, .5, .7, 1.0):
            epsilon, alpha = budget(q, z, rounds, delta, .02)
            check, _ = budget(q, z, rounds, delta, .01)
            assert abs(epsilon-check) < 1e-6, (dataset, q, epsilon, check)
            # Validate integration against the exact integer-order moment.
            if q < 1:
                exact = logadd(logadd(2*math.log1p(-q), math.log(2*q*(1-q))),
                               2*math.log(q)+1/(z*z))
                assert abs(sampled_rdp(q,z,2,.01)-exact) < 1e-8
            rows.append(dict(dataset=dataset, q=q, noise_multiplier=z,
                clip_norm=metadata['clip_norm'], rounds=rounds, delta=delta,
                epsilon=check, optimal_rdp_order=alpha,
                original_reported_epsilon=metadata['accounting']['epsilon'],
                numerical_check_absolute_difference=abs(epsilon-check)))
    with (out/'privacy_by_sampling_rate.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out/'privacy_by_sampling_rate.json').write_text(json.dumps(dict(
        accounting='Poisson-sampled Gaussian RDP; basic conversion; continuous order optimization',
        status='Counterfactual budgets, not guarantees for existing full-participation fixed-seed runs',
        assumptions=['Independent Bernoulli client sampling each round',
                     'User add/remove adjacency; bounded contribution and fixed public normalization',
                     'Noise multiplier calibrated to sensitivity',
                     'Sampling identities not included in public release; trusted central server',
                     'Fresh privacy randomness; no unaccounted private releases'],
        source='https://arxiv.org/abs/1908.10530', results=rows),indent=2))
    for row in rows:
        print(f"{row['dataset']:12} q={row['q']:.1f} epsilon={row['epsilon']:.6f}")


if __name__ == '__main__':
    main()
