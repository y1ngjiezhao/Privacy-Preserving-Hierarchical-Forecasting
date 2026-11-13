"""Locations shared by modules, notebooks, and command-line entry points."""
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FHF_MODULE_DIR = PROJECT_ROOT / "PPHF_modules" / "fhf"
DP_MODULE_DIR = PROJECT_ROOT / "PPHF_modules" / "dp_fhf"
RESULTS_MODULE_DIR = PROJECT_ROOT / "PPHF_modules" / "results_check"
DP_EXPERIMENT_DIR = PROJECT_ROOT / "DP_fhf"
RESULTS_DIR = PROJECT_ROOT / "Results_check"

def locate_project_root(start=None):
    override = os.environ.get("PPHF_PROJECT_ROOT") or os.environ.get("DP_FHF_PROJECT_ROOT")
    base = Path(start or Path.cwd()).resolve()
    candidates = ([Path(override).expanduser()] if override else []) + [base, *base.parents]
    for candidate in candidates:
        if (candidate / "PPHF_modules" / "fhf" / "FHF_runner.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Cannot locate PPHF_modules. Start inside the project or set PPHF_PROJECT_ROOT.")
