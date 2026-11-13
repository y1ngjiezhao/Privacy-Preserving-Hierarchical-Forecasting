from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

from pathlib import Path

import nbformat


ROOT = PROJECT_ROOT
NOTEBOOKS = {
    ROOT / "fhf_dedup_ausgrid2013" / "FHF_fh1.ipynb": (
        "clients",
        "list(cloud.client_ids)",
    ),
    ROOT / "fhf_dedup_ausgrid2013" / "HFF_fh1.ipynb": (
        "nodes",
        "list(layout.node_ids)",
    ),
    ROOT / "fhf_dedup_ausgrid2013" / "HFFPS_fh1.ipynb": (
        "nodes",
        "list(layout.node_ids)",
    ),
}


def _patch_artifact_cell(source: str, nodes_name: str, node_order: str) -> str:
    if "from PPHF_modules.fhf.training_artifacts import (" in source:
        return source
    source = source.replace(
        "from PPHF_modules.fhf.FHF_DP import calibrate_time_slot_event_bounds",
        """from PPHF_modules.fhf.training_artifacts import (
    save_or_validate_shared_bounds,
    save_training_artifacts,
)


# Post-training export only: these reads do not update model weights, forecasts,
# reconciliation matrices, random state, or evaluation metrics.
training_artifacts_path = save_training_artifacts(
    nodes=NODES_PLACEHOLDER,
    cid2series=cid2series,
    node_order=NODE_ORDER_PLACEHOLDER,
    output_path=f'{output_prefix}_training_artifacts.npz',
)
with np.load(training_artifacts_path, allow_pickle=False) as stored:
    shared_bounds_paths = save_or_validate_shared_bounds(
        artifacts={key: stored[key] for key in stored.files},
        artifact_dir=Path(training_artifacts_path).parent,
        forecast_horizon_count=int(args.fh) + 1,
        slots_per_day=int(args.dp_slots_per_day),
        lower_quantile=float(args.dp_lower_quantile),
        upper_quantile=float(args.dp_upper_quantile),
        iqr_margin=float(args.dp_iqr_margin),
        physical_lower=args.dp_physical_lower,
    )""".replace("NODES_PLACEHOLDER", nodes_name).replace(
            "NODE_ORDER_PLACEHOLDER", node_order
        ),
    )

    bounds_start = source.find("    bounds_rows = []")
    config_start = source.find("    config_path =", bounds_start)
    if bounds_start < 0 or config_start < 0:
        raise ValueError("Cannot locate the method-specific bounds block.")
    source = (
        source[:bounds_start]
        + """    # All frameworks and training modes reference one method-independent
    # release calibrated from the common local training targets.
    bounds_path = str(shared_bounds_paths['bounds'])
    slot_map_path = str(shared_bounds_paths['time_slot_map'])

"""
        + source[config_start:]
    )
    source = source.replace(
        "'dp_bounds_source': str(args.dp_bounds_source),",
        "'dp_bounds_source': 'common_local_training_targets',",
    )
    source = source.replace(
        "'bounds_csv': bounds_path,",
        "'bounds_csv': Path(bounds_path).name,",
    )
    source = source.replace(
        "'time_slot_map_csv': slot_map_path,",
        "'time_slot_map_csv': Path(slot_map_path).name,",
    )
    source = source.replace(
        "output_paths.update(event_dp_o2mr_paths)\n",
        """output_paths.update(event_dp_o2mr_paths)
output_paths['training_artifacts'] = str(training_artifacts_path)
output_paths['shared_event_dp_bounds_metadata'] = str(
    shared_bounds_paths['metadata']
)
""",
    )
    return source


def patch_notebook(path: Path, nodes_name: str, node_order: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    artifact_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "def save_event_dp_o2mr_artifacts" in cell.source
    ]
    if len(artifact_cells) != 1:
        raise ValueError(f"Expected one O2MR artifact cell in {path}.")

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.source = cell.source.replace(
            'args.dp_bounds_source = "train_predictions"',
            'args.dp_bounds_source = "common_local_training_targets"',
        )
        cell.source = cell.source.replace(
            "# time-slot bounds are exported after evaluation.",
            "# one shared training-target time-slot bounds file is exported after evaluation.",
        )

    artifact_cells[0].source = _patch_artifact_cell(
        artifact_cells[0].source,
        nodes_name=nodes_name,
        node_order=node_order,
    )
    nbformat.write(notebook, path)
    print(f"Updated {path}")


def main() -> None:
    for path, (nodes_name, node_order) in NOTEBOOKS.items():
        patch_notebook(path, nodes_name=nodes_name, node_order=node_order)


if __name__ == "__main__":
    main()
