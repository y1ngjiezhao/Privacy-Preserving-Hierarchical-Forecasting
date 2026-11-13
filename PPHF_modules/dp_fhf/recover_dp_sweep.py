"""Recover sweep accuracy/accounting tables from saved forecasts, without training.

Recovery never creates a normal privacy-completion marker or reconstructs model
weights. The supplied round count must match the completed original training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from PPHF_modules.dp_fhf.notebook_runtime import (
    DP_PREFIX, _evaluate_forecasts, _number_slug, _scales_from_artifact,
)
from PPHF_modules.dp_fhf.privacy_accounting import (
    check_sampling_accountant_dependencies, sampled_fedavg_epsilon,
)
from PPHF_modules.fhf.client_sampling import (
    amplification_output_dir, fixed_sample_size, sampled_output_dir,
    validate_sampling_rate,
)


def recover_ausgrid_sampling_sweep(
    *, output_root: str | Path, sampling_rate: float, num_rounds: int,
    sampling_seed: int | None = None, delta: float = 1e-5,
):
    """Recover the fixed-rate AusGrid notebook's available CSV/NPZ pairs.

    These notebooks disable early stopping and aggregate once per round.
    Accounting is recomputed from the supplied completed round count and saved
    population size. It applies to the global-model mechanism, not to the
    saved local forecasts/residuals. Existing source artifacts are read only.
    """
    parent = Path(output_root).expanduser().resolve()
    rate = validate_sampling_rate(sampling_rate)
    if not isinstance(num_rounds, int) or isinstance(num_rounds, bool) or num_rounds < 1:
        raise ValueError("num_rounds must be the positive integer used in the completed training.")
    if not 0 < delta < 1:
        raise ValueError("delta must lie between zero and one.")
    if rate < 1:
        check_sampling_accountant_dependencies()
    base = parent / "full_participation" if rate == 1 else parent
    directory = sampled_output_dir(amplification_output_dir(base, rate < 1), rate, sampling_seed)
    grid_path = directory / "parameter_grid.csv"
    grid = pd.read_csv(grid_path)
    if not {"run_id", "noise_multiplier", "clip_norm"}.issubset(grid.columns):
        raise ValueError(f"Missing grid configuration columns: {grid_path}")
    if grid.run_id.duplicated().any():
        raise ValueError(f"Duplicate run IDs: {grid_path}")
    if "sampling_rate" in grid and not np.allclose(grid.sampling_rate, rate, rtol=0, atol=1e-12):
        raise ValueError("Requested sampling rate differs from the saved parameter grid.")

    recovered, missing, node_tables, source_stats = [], [], [], {}
    accounting_cache = {}
    for item in grid.itertuples(index=False):
        noise, clip = float(item.noise_multiplier), float(item.clip_norm)
        if not math.isfinite(noise) or noise <= 0 or not math.isfinite(clip) or clip <= 0:
            raise ValueError(f"Invalid noise/clipping configuration: {item.run_id}")
        expected_id = f"nm_{_number_slug(noise)}__clip_{_number_slug(clip)}"
        if item.run_id != expected_id:
            raise ValueError(f"Run ID/configuration mismatch: {item.run_id}")
        run = directory / item.run_id
        forecast = run / f"{DP_PREFIX}_base_forecasts.csv"
        artifact = run / f"{DP_PREFIX}_training_artifacts.npz"
        if not forecast.is_file() or not artifact.is_file():
            missing.append({"run_id": item.run_id, "missing_files": [str(p) for p in
                            (forecast, artifact) if not p.is_file()]})
            continue
        for path in (forecast, artifact):
            stat = path.stat()
            source_stats[path] = (stat.st_size, stat.st_mtime_ns)
        with np.load(artifact, allow_pickle=False) as stored:
            names = stored["unique_ids"].astype(str)
            ids = stored["client_ids"]
        population = len(ids)
        if (population == 0 or len(names) != population or len(set(names)) != population
                or len(set(ids.tolist())) != population):
            raise ValueError(f"Invalid saved client identities: {artifact}")
        sample_size = fixed_sample_size(population, rate)
        scales = _scales_from_artifact(artifact)
        if any(not np.isfinite(value) or value <= 0 for pair in scales.values() for value in pair):
            raise ValueError(f"Nonpositive or nonfinite training scales: {artifact}")
        predictions = pd.read_csv(forecast)
        if not {"unique_id", "y_true", "base"}.issubset(predictions.columns):
            raise ValueError(f"Invalid forecast columns: {forecast}")
        if set(predictions.unique_id.astype(str)) != set(names):
            raise ValueError(f"Forecast/artifact series mismatch: {run}")
        if not np.isfinite(predictions[["y_true", "base"]].to_numpy(float)).all():
            raise ValueError(f"Nonfinite saved predictions: {forecast}")
        # If a normal completion marker exists, verify the supplied configuration.
        privacy_path = run / f"{DP_PREFIX}_privacy.json"
        if privacy_path.exists():
            saved = json.loads(privacy_path.read_text(encoding="utf-8"))
            expected = {"num_rounds": num_rounds, "sampling_rate": rate,
                        "noise_multiplier": noise, "clip_norm": clip}
            for key, value in expected.items():
                if saved.get(key, 1.0 if key == "sampling_rate" else None) != value:
                    raise ValueError(f"Recovery configuration differs from saved metadata: {run}: {key}")
            if saved.get("framework", "").lower() != "fhf":
                raise ValueError(f"Expected FHF metadata: {privacy_path}")
            if saved.get("aggregation_calls", num_rounds) != num_rounds:
                raise ValueError(f"Saved aggregation count differs from supplied rounds: {privacy_path}")
        per_node, metrics = _evaluate_forecasts(predictions, scales, item.run_id)
        key = (noise, num_rounds, delta, population, sample_size)
        if key not in accounting_cache:
            accounting_cache[key] = sampled_fedavg_epsilon(
                noise, num_rounds, delta, population, sample_size, "add_remove",
            )
        accounting = accounting_cache[key]
        recovered.append({
            "dataset": "AusGrid2013", "framework": "FHF", "run_id": item.run_id,
            "sampling_rate": rate, "effective_sampling_rate": sample_size / population,
            "population_size": population, "sample_size": sample_size,
            "noise_multiplier": noise, "clip_norm": clip, "num_rounds": num_rounds,
            "epsilon": float(accounting["epsilon"]), "delta": delta,
            "sampling_amplification_used": accounting["sampling_amplification_used"],
            "status": "recovered_saved_forecast_metrics_and_posthoc_accounting",
            "round_count_source": "supplied_completed_rounds; original notebook disables early stopping",
            "sampling_seed": sampling_seed,
            "budget_status": ("posthoc_accounting_for_configured_randomized_model_mechanism"
                              if sampling_seed is None or rate == 1 else
                              "hypothetical_budget_for_randomized_mechanism; fixed research seeds used"),
            "model_release_file_exists": (run / "model_release/global_model.npz").is_file(),
            "accounting_scope": "global-model mechanism only; excludes saved local forecasts and residuals",
            "run_directory": str(run), **metrics,
        })
        node_tables.append((item.run_id, per_node))
    if not recovered:
        raise FileNotFoundError(f"No complete forecast/NPZ pairs found beneath {directory}")
    for path, before in source_stats.items():
        stat = path.stat()
        if before != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Source changed during recovery: {path}")
    recovery = directory / "recovery"
    recovery.mkdir(exist_ok=True)
    for run_id, table in node_tables:
        table.to_csv(recovery / f"{run_id}_per_series_metrics.csv", index=False)
    result = pd.DataFrame(recovered).sort_values(["MASE", "RMSSE", "epsilon"])
    summary_path = recovery / "recovered_sweep_summary.csv"
    result.to_csv(summary_path, index=False)
    (recovery / "recovery_report.json").write_text(json.dumps({
        "training_executed": False, "source_artifacts_modified": False,
        "original_privacy_completion_markers_created": False,
        "model_weights_reconstructed": False,
        "completed_rounds_supplied": num_rounds, "sampling_rate": rate,
        "recovered_runs": len(result), "grid_runs": len(grid), "missing_runs": missing,
        "summary": str(summary_path),
    }, indent=2), encoding="utf-8")
    frames = [pd.read_csv(p) for p in sorted(parent.rglob("recovered_sweep_summary.csv"))]
    combined_path = parent / "recovered_all_sampling_rates_summary.csv"
    pd.concat(frames, ignore_index=True).sort_values(
        ["sampling_rate", "clip_norm", "noise_multiplier"]
    ).to_csv(combined_path, index=False)
    print(f"Recovered {len(result)}/{len(grid)} runs without training.")
    print("Recovered rate summary:", summary_path)
    print("Combined recovered summary:", combined_path)
    print("Model weights are not reconstructed; original privacy-completion markers are untouched.")
    return result
