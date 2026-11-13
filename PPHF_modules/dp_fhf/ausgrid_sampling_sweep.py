"""Independent fixed-rate AusGrid sweeps sharing one experiment directory."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PPHF_modules.fhf.client_sampling import validate_sampling_rate
from PPHF_modules.dp_fhf.notebook_runtime import run_sweep_dpfedavg


def run_ausgrid_sampling_sweep(
    *, root: str | Path, csv_path: str | Path, sampling_rate: float,
    noise_multipliers: Sequence[float], clip_norms: Sequence[float],
    num_rounds: int = 20, sampling_seed: int | None = None,
    output_root: str | Path | None = None, dry_run: bool = False,
    overwrite: bool = False, stop_on_error: bool = False,
):
    """Run one rate's C/noise grid; never launch a loop over sampling rates."""
    root = Path(root).resolve()
    rate = validate_sampling_rate(sampling_rate)
    parent = (Path(output_root).expanduser().resolve() if output_root is not None
              else root / "DP_fhf/outputs/ausgrid2013_sampling_sweep")
    # The partial-run helper adds its version/selection directories itself.
    # Keep full participation in its own child of the same experiment parent.
    base = parent / "full_participation" if rate == 1.0 else parent
    return run_sweep_dpfedavg(
        root=root, dataset="AusGrid2013", csv_path=csv_path,
        noise_multipliers=noise_multipliers, clip_norms=clip_norms,
        num_rounds=num_rounds, sampling_rate=rate, sampling_seed=sampling_seed,
        output_base_dir=base, dry_run=dry_run, overwrite=overwrite,
        stop_on_error=stop_on_error,
    )


def summarize_ausgrid_sampling_sweeps(output_root: str | Path, *, dry_run: bool = False):
    """Combine completed rate summaries, retaining each run's source directory."""
    import pandas as pd

    root = Path(output_root).expanduser().resolve()
    frames = []
    for path in sorted(root.rglob("sweep_summary.csv")):
        frame = pd.read_csv(path)
        required = {"dataset", "framework", "sampling_rate", "run_id",
                    "clip_norm", "noise_multiplier", "num_rounds", "epsilon"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Incomplete sweep summary: {path}")
        if not frame.dataset.eq("AusGrid2013").all() or not frame.framework.eq("FHF").all():
            raise ValueError(f"Expected AusGrid2013 FHF results: {path}")
        frame["sampling_rate"] = frame.sampling_rate.map(validate_sampling_rate)
        if frame.sampling_rate.nunique() != 1 or frame.run_id.duplicated().any():
            raise ValueError(f"Expected one sampling rate and unique run IDs: {path}")
        frame["source_summary"] = path.relative_to(root).as_posix()
        frame["run_directory"] = [str(path.parent / run_id) for run_id in frame.run_id]
        frames.append(frame)
    if not frames:
        print("No completed AusGrid sampling sweeps yet.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["sampling_rate", "clip_norm", "noise_multiplier", "source_summary"]
    ).reset_index(drop=True)
    output = root / "all_sampling_rates_summary.csv"
    if not dry_run:
        combined.to_csv(output, index=False)
    print("Combined summary:", output)
    print("Completed runs by sampling rate:")
    print(combined.groupby("sampling_rate").size())
    return combined
