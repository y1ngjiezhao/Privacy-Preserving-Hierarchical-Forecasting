"""Differential-privacy research extensions for FHF, HFF, and HFFPS."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

from .dp_covariance import (
    DPCovarianceConfig,
    build_private_reconciliation_matrices,
    install_and_distribute_private_reconciliation,
    privatize_residual_matrix,
    privatize_residuals_by_client,
    run_local_dp_covariance_reconciliation,
)
from .privacy_accounting import gaussian_composition_epsilon

__all__ = [
    "DPCovarianceConfig",
    "build_private_reconciliation_matrices",
    "gaussian_composition_epsilon",
    "install_and_distribute_private_reconciliation",
    "privatize_residual_matrix",
    "privatize_residuals_by_client",
    "run_local_dp_covariance_reconciliation",
]
