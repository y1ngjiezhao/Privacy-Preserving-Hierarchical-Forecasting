from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from PPHF_modules.notebook_paths import normalize_notebook

"""Generate the portable DP-FHF notebooks using only the standard library."""

import json
from pathlib import Path
from textwrap import dedent


HERE = DP_EXPERIMENT_DIR
NOTEBOOK_DIR = HERE / "notebooks"


def cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def write_notebook(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (neural-network)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    normalize_notebook(notebook)
    path = NOTEBOOK_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", path)


BOOTSTRAP = r'''
from pathlib import Path
import os
import sys

PROJECT_ROOT = os.environ.get("DP_FHF_PROJECT_ROOT")

def locate_project_root():
    candidates = [Path(PROJECT_ROOT).expanduser()] if PROJECT_ROOT else []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    for candidate in candidates:
        if (candidate / "PPHF_modules" / "fhf").is_dir() and (candidate / "DP_fhf").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot locate the FHF project root. Set DP_FHF_PROJECT_ROOT or PROJECT_ROOT."
    )

ROOT = locate_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PPHF_modules.dp_fhf.notebook_runtime import (
    locate_dataset_csv,
    run_covariance_notebook,
    run_fixed_dpfedavg,
    run_o2mr_notebook,
    run_sweep_dpfedavg,
    summarize_dpfedavg,
)

DRY_RUN = os.environ.get("DP_FHF_NOTEBOOK_DRY_RUN", "0") == "1"
print("Project root:", ROOT)
print("Python:", sys.executable)
print("Dry run:", DRY_RUN)
'''


def fixed_cells(dataset: str, noise: float) -> list[dict]:
    return [
        cell(
            "markdown",
            f'''
            # {dataset}: FHF DP-FedAvg, 100 rounds

            Runs exactly one FHF experiment on the automatically discovered
            deduplicated {dataset} data. Configuration: 100 rounds, clipping
            norm 1, noise multiplier {noise:g}, forecast horizon 1, and
            hierarchy mode `dedup`. HFF/HFFPS are not used.
            Results are isolated under `outputs/01_dp_fedavg_100round/{dataset}`.
            ''',
        ),
        cell("code", BOOTSTRAP),
        cell(
            "code",
            f'''
            DATASET = "{dataset}"
            SAMPLING_RATE = 1.0  # Full participation preserves the original experiment.
            SAMPLING_SEED = None  # Fresh hidden sampling for q<1; q=1 keeps the original seed.
            DATA_CSV_OVERRIDE = None  # Optional explicit CSV path
            CSV_PATH = locate_dataset_csv(ROOT, DATASET, DATA_CSV_OVERRIDE)
            print("Deduplicated input:", CSV_PATH)
            ''',
        ),
        cell(
            "code",
            f'''
            result = run_fixed_dpfedavg(
                root=ROOT,
                dataset=DATASET,
                csv_path=CSV_PATH,
                num_rounds=100,
                clip_norm=1.0,
                noise_multiplier={noise:g},
                sampling_rate=SAMPLING_RATE,
                sampling_seed=SAMPLING_SEED,
                dry_run=DRY_RUN,
                overwrite=False,
            )
            result
            ''',
        ),
    ]


def sweep_cells(dataset: str) -> list[dict]:
    return [
        cell(
            "markdown",
            f'''
            # {dataset}: FHF DP-FedAvg, 20-round sweep

            Runs 12 FHF-only combinations on the automatically discovered
            deduplicated {dataset} data: 4 noise multipliers × 3 clipping norms.
            Every combination uses 20 rounds, forecast horizon 1, hierarchy
            mode `dedup`, and completed runs are resumable.
            ''',
        ),
        cell("code", BOOTSTRAP),
        cell(
            "code",
            f'''
            DATASET = "{dataset}"
            SAMPLING_RATE = 1.0
            SAMPLING_SEED = None
            DATA_CSV_OVERRIDE = None  # Optional explicit CSV path
            CSV_PATH = locate_dataset_csv(ROOT, DATASET, DATA_CSV_OVERRIDE)
            NOISE_MULTIPLIERS = [1.2, 0.9, 0.6, 0.3]
            CLIP_NORMS = [1.0, 0.5, 0.2]
            print("Deduplicated input:", CSV_PATH)
            print("Runs:", len(NOISE_MULTIPLIERS) * len(CLIP_NORMS))
            ''',
        ),
        cell(
            "code",
            '''
            summary = run_sweep_dpfedavg(
                root=ROOT,
                dataset=DATASET,
                csv_path=CSV_PATH,
                noise_multipliers=NOISE_MULTIPLIERS,
                clip_norms=CLIP_NORMS,
                num_rounds=20,
                sampling_rate=SAMPLING_RATE,
                sampling_seed=SAMPLING_SEED,
                dry_run=DRY_RUN,
                overwrite=False,
                stop_on_error=False,
            )
            summary
            ''',
        ),
    ]


def ausgrid_sampling_sweep_cells(sampling_rate: float) -> list[dict]:
    return [
        cell("markdown", f'''
            # AusGrid2013 DP-FedAvg sweep: sampling rate {sampling_rate:g}

            This notebook runs only sampling rate **{sampling_rate:g}**.
            It sweeps 4 noise multipliers × 3 clipping norms, with 20 rounds
            per combination, using the deduplicated AusGrid2013 input.
            There is no outer loop over sampling rates.

            All three notebooks share `DP_fhf/outputs/ausgrid2013_sampling_sweep/`.
            Full participation is stored under `full_participation/`;
            partial runs use `amplified_v1/sampling_runs/q_<rate>__seed_fresh/`.
            Each rate has its own `sweep_summary.csv`. After training, this
            notebook also refreshes the parent's `all_sampling_rates_summary.csv`
            from whichever rates have completed. Run the three notebooks one
            at a time on a single GPU.

            Summaries evaluate base forecasts. Existing reconciliation notebooks
            for the older sweep directory do not automatically read this new
            experiment. If a non-DP reference is available, its sampling rate is
            1.0; it is not a matched partial-participation baseline.
            '''),
        cell("code", BOOTSTRAP + '''
from PPHF_modules.dp_fhf.ausgrid_sampling_sweep import (
    run_ausgrid_sampling_sweep,
    summarize_ausgrid_sampling_sweeps,
)
'''),
        cell("code", f'''
            DATASET = "AusGrid2013"
            SAMPLING_RATE = {sampling_rate!r}
            SAMPLING_SEED = None
            NUM_ROUNDS = 20
            NOISE_MULTIPLIERS = [1.2, 0.9, 0.6, 0.3]
            CLIP_NORMS = [1.0, 0.5, 0.2]
            OVERWRITE = False
            # DRY_RUN = True  # Uncomment to preview commands without training.

            CSV_PATH = locate_dataset_csv(ROOT, DATASET)
            OUTPUT_ROOT = ROOT / "DP_fhf" / "outputs" / "ausgrid2013_sampling_sweep"
            print("Input:", CSV_PATH)
            print("Sampling rate:", SAMPLING_RATE)
            print("Runs:", len(NOISE_MULTIPLIERS) * len(CLIP_NORMS))
            print("Common output folder:", OUTPUT_ROOT)
            '''),
        cell("code", '''
            summary = run_ausgrid_sampling_sweep(
                root=ROOT,
                csv_path=CSV_PATH,
                sampling_rate=SAMPLING_RATE,
                noise_multipliers=NOISE_MULTIPLIERS,
                clip_norms=CLIP_NORMS,
                num_rounds=NUM_ROUNDS,
                sampling_seed=SAMPLING_SEED,
                output_root=OUTPUT_ROOT,
                dry_run=DRY_RUN,
                overwrite=OVERWRITE,
                stop_on_error=False,
            )
            summary
            '''),
        cell("code", '''
            all_rates_summary = summarize_ausgrid_sampling_sweeps(
                OUTPUT_ROOT, dry_run=DRY_RUN,
            )
            all_rates_summary
            '''),
    ]


def write_ausgrid_sampling_sweeps() -> None:
    for rate, suffix in ((0.1, "q0p1"), (0.5, "q0p5"), (1.0, "q1p0")):
        write_notebook(f"01_DPFedAvg_AusGrid2013_Sweep_{suffix}.ipynb",
                       ausgrid_sampling_sweep_cells(rate))


def main() -> None:
    write_notebook("01_DPFedAvg_Only_GEF2017.ipynb", fixed_cells("GEF2017", 0.3))
    write_notebook("01_DPFedAvg_Only_AusGrid2013.ipynb", fixed_cells("AusGrid2013", 1.2))
    write_notebook("01_DPFedAvg_GEF2017_Sweep.ipynb", sweep_cells("GEF2017"))
    write_ausgrid_sampling_sweeps()

    write_notebook(
        "01_DPFedAvg_MASE_RMSSE_Summary.ipynb",
        [
            cell(
                "markdown",
                '''
                # FHF DP-FedAvg results summary

                Automatically discovers the two 100-round experiments, legacy
                20-round sweeps, and the three AusGrid sampling-rate sweeps.
                Any non-FHF metadata is rejected.
                ''',
            ),
            cell("code", BOOTSTRAP),
            cell("code", "results = summarize_dpfedavg(ROOT, dry_run=DRY_RUN)\nresults"),
        ],
    )

    write_notebook(
        "02_DP_Covariance_Estimation.ipynb",
        [
            cell(
                "markdown",
                '''
                # Local-DP covariance estimation (FHF only)

                Automatically discovers the matching AusGrid2013 and GEF2017 FHF
                artifacts. The fixed 100-round DP-FedAvg branch is included when it
                exists; HFF and HFFPS branches are never constructed.
                Uses the current paper configuration: no noise-variance correction,
                fixed bounds +/-0.0005, epsilon=1, seed=42, ridge=1e-6.
                Results are stored in each dataset's no_debias/ directory.
                ''',
            ),
            cell("code", BOOTSTRAP),
            cell(
                "code",
                '''
                covariance_summary = run_covariance_notebook(
                    ROOT, datasets=("AusGrid2013", "GEF2017"), dry_run=DRY_RUN,
                    debias_noise_variance=False
                )
                covariance_summary
                ''',
            ),
        ],
    )

    write_notebook(
        "03_DP_O2MR_Saved_P.ipynb",
        [
            cell(
                "markdown",
                '''
                # DP-O2MR with saved P matrices (FHF only)

                Automatically discovers each dataset's saved FHF P matrices. The
                fixed DP-FedAvg branch uses its own matching bounds when available.
                ''',
            ),
            cell("code", BOOTSTRAP),
            cell(
                "code",
                '''
                o2mr_summary = run_o2mr_notebook(
                    ROOT, datasets=("AusGrid2013", "GEF2017"), dry_run=DRY_RUN
                )
                o2mr_summary
                ''',
            ),
        ],
    )

    # Preserve the historical filename as a safe index. The old version prepared
    # 24 mixed FHF/HFF/HFFPS runs and must not remain executable.
    write_notebook(
        "01_DPFedAvg_NDSH_Matrix.ipynb",
        [
            cell(
                "markdown",
                '''
                # NDSH DP-FedAvg experiment index (FHF only)

                The former 24-run mixed-framework matrix has been retired. Use the
                two 100-round fixed notebooks and the 20-round sweeps below:

                - `01_DPFedAvg_Only_GEF2017.ipynb`
                - `01_DPFedAvg_Only_AusGrid2013.ipynb`
                - `01_DPFedAvg_GEF2017_Sweep.ipynb`
                - `01_DPFedAvg_AusGrid2013_Sweep_q0p1.ipynb`
                - `01_DPFedAvg_AusGrid2013_Sweep_q0p5.ipynb`
                - `01_DPFedAvg_AusGrid2013_Sweep_q1p0.ipynb`

                All generated commands use `framework=fhf`,
                `hierarchy_mode=dedup`, and forecast horizon 1.
                ''',
            ),
            cell("code", BOOTSTRAP),
            cell(
                "code",
                '''
                print("The old 24-run FHF/HFF/HFFPS matrix is disabled.")
                print("Run one of the dataset-specific notebooks listed above.")
                ''',
            ),
        ],
    )


if __name__ == "__main__":
    main()
