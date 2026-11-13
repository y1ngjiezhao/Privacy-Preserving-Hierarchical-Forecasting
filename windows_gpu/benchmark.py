"""Time a few real training rounds and estimate the requested total on this PC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("GEF2017", "AusGrid2013"), default="GEF2017")
    parser.add_argument("--framework", choices=("fhf", "hff", "hffps"), default="fhf")
    parser.add_argument("--sampling-rate", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target-rounds", type=int, default=100)
    parser.add_argument("--dp-fedavg", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--csv", type=Path, help="Override the dataset; relative paths use the project root.")
    parser.add_argument("--output", type=Path, help="JSON report; relative paths use the project root.")
    parsed = parser.parse_args()
    if parsed.rounds < 1 or parsed.target_rounds < parsed.rounds:
        parser.error("Require 1 <= rounds <= target-rounds.")
    if not 0 < parsed.sampling_rate <= 1:
        parser.error("sampling-rate must be in (0, 1].")

    import torch
    from PPHF_modules.fhf.FHF_runner import Args, run_fhf
    from PPHF_modules.fhf.HFL_runner_timed import HFLArgs, run_hfl_timed

    if parsed.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run windows_gpu/check_environment.py first.")
    relative = {
        "GEF2017": "Datasets/GEFCOM2017/GEF2017_univarivate.csv",
        "AusGrid2013": "Datasets/AusGrid2013/AusGrid2013_univariate_dedup.csv",
    }[parsed.dataset]
    csv_path = parsed.csv or Path(relative)
    csv_path = (ROOT / csv_path).resolve()
    if not csv_path.is_file():
        parser.error(f"Dataset not found: {csv_path}")

    args = Args() if parsed.framework == "fhf" else HFLArgs()
    for name, value in {
        "csv_path": str(csv_path), "truncated": None, "device": parsed.device,
        "fh": 0, "lags": 48, "batch_size": 256, "hidden_size": 64,
        "num_layers": 2, "local_epochs": 1, "num_rounds": parsed.rounds,
        "sampling_rate": parsed.sampling_rate, "sampling_seed": 42, "seed": 42,
        "torch_num_threads": 1, "early_stop_enabled": False,
        "return_result": True, "reject_duplicate_series": True,
        "dp_enabled": False, "dp_aware_reconciliation": False,
        "dp_fedavg_enabled": parsed.dp_fedavg, "dp_fedavg_seed": 42,
        "dp_fedavg_max_update_norm": 1.0,
        "dp_fedavg_noise_multiplier": 0.3 if parsed.dataset == "GEF2017" else 1.2,
        "dp_fedavg_adjacency": "add_remove", "dp_fedavg_weighting": "uniform",
        "verbose_rounds": True, "verbose_cloud_rounds": True, "print_every": 1,
        "validate_p2p": False,
    }.items():
        setattr(args, name, value)
    if parsed.dp_fedavg:
        args.recon_method = ("mint_ols",)
    if parsed.framework != "fhf":
        args.edge_trainable = args.cloud_trainable = parsed.framework == "hffps"
        args.hier_train_mode = "self_train_then_aggregate"
        args.edge_agg_rounds = 1
        args.client_frac_per_edge = 1.0
        args.verbose_edge_rounds = False

    print("Dataset:", csv_path, flush=True)
    print("Framework / DP / device:", parsed.framework, parsed.dp_fedavg, parsed.device, flush=True)
    print("Sampling rate / rounds:", parsed.sampling_rate, parsed.rounds, flush=True)
    if parsed.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = run_fhf(args) if parsed.framework == "fhf" else run_hfl_timed(args)
    if parsed.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    timings = result["timings"]
    logs = result.get("round_logs", result.get("cloud_round_logs", []))
    completed = len(logs)
    training = float(timings["fl_training_sec"])
    fixed = max(0.0, wall - training)
    estimate = fixed + training / completed * parsed.target_rounds
    timestamp = datetime.now(timezone.utc)
    label = f"{parsed.dataset}_{parsed.framework}_{'dp' if parsed.dp_fedavg else 'non_dp'}"
    output = parsed.output or (
        Path("output/gpu_benchmarks") / f"{label}_{timestamp:%Y%m%dT%H%M%S%fZ}.json"
    )
    output = (ROOT / output).resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing report: {output}")
    report = {
        "created_utc": timestamp.isoformat(), "input_csv": str(csv_path),
        "framework": parsed.framework, "dp_fedavg": parsed.dp_fedavg,
        "device": parsed.device, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if parsed.device == "cuda" else None,
        "forecast_horizon": 1, "lags": 48, "local_epochs": 1,
        "hidden_size": 64, "num_layers": 2, "batch_size": 256,
        "sampling_rate": parsed.sampling_rate, "rounds_completed": completed,
        "selected_per_round": [
            row.get("num_selected_clients", row.get("num_selected_nodes")) for row in logs
        ],
        "timings": timings, "wall_sec": wall,
        "mean_training_round_sec": training / completed,
        "fixed_setup_and_forecast_sec": fixed,
        "target_rounds": parsed.target_rounds, "estimated_total_sec": estimate,
        "gpu_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if parsed.device == "cuda" else None
        ),
        "note": "Estimate uses this sampling rate and input. Notebook exports and plots are excluded. "
                "Benchmark seeds are fixed. No trained model or forecast is exported.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Mean training round: {training / completed:.2f} seconds")
    print(f"Estimated {parsed.target_rounds} rounds: {estimate / 60:.1f} minutes ({estimate / 3600:.2f} hours)")
    print("Report:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
