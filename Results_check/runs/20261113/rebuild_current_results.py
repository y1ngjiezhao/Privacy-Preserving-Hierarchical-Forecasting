"""Force fresh result tables, then execute the corresponding result notebooks."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fhf_results_mpl_20261113")
os.environ.setdefault("IPYTHONDIR", "/private/tmp/fhf_results_ipy_20261113")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/private/tmp/fhf_results_jupyter_20261113")
os.environ["MPLBACKEND"] = "Agg"
for name in ("MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["ausgrid2013", "gef2017"], required=True)
    parser.add_argument("--execute-notebooks", action="store_true")
    options = parser.parse_args()
    from PPHF_modules.results_check.current_results import prepare_current_results

    run_dir = Path(__file__).parent
    record_path = run_dir / "execution_manifest.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    notebook_paths = []
    for dataset in options.datasets:
        start = time.time()
        print(f"Rebuilding {dataset} from current saved results", flush=True)
        tables = prepare_current_results(dataset, root=ROOT, force=True)
        folder = ROOT / "Results_check/deduplication_results" / dataset
        record[dataset] = {
            "status": "fresh_tables_completed",
            "elapsed_seconds": time.time() - start,
            "source": str(ROOT / f"fhf_dedup_{dataset}"),
            "output": str(folder),
            "audit": tables["hierarchy_audit"].to_dict(orient="records"),
        }
        record_path.write_text(json.dumps(record, indent=2))
        print(tables["hierarchy_audit"].to_string(index=False), flush=True)
        entry = f"{dataset}_results"
        notebook_paths.append(ROOT / "Results_check" / entry / f"{entry}.ipynb")
        if dataset == "gef2017":
            notebook_paths.append(ROOT / "Results_check/full_gef2017_results/full_gef2017_results.ipynb")
    if options.execute_notebooks:
        command = [sys.executable, str(ROOT / "PPHF_modules/results_check/execute_notebooks.py"),
                   *map(str, notebook_paths)]
        subprocess.run(command, cwd=ROOT, check=True)
        for dataset in options.datasets:
            record[dataset]["status"] = "fresh_tables_and_executed_notebooks_completed"
        record_path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
