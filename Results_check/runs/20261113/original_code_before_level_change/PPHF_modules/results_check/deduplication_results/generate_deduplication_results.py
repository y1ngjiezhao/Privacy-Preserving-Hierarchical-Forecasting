#!/usr/bin/env python3
"""Generate publication-ready summaries for the deduplicated experiments.

The calculations follow Results_check/full_gef2017_results:
series-level MASE/RMSSE/MAE are averaged within each hierarchy level, and
incoherence is the sum of absolute parent-minus-children discrepancies.  A
second, scale-free measure averages the symmetric relative discrepancy at
each parent-time comparison.
"""

from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd


ROOT = PROJECT_ROOT
OUTPUT_ROOT = RESULTS_DIR / "deduplication_results"
from PPHF_modules.results_check.publication_labels import publication_table

DATASETS = {
    "gef2017": {
        "root": ROOT / "fhf_dedup_gef2017",
        "horizons": (1, 3, 6),
        "bottom_tags": ROOT / "Datasets" / "GEFCOM2017" / "tags_dedup.pkl",
        "expected_structural_level_counts": {1: 1, 2: 6, 3: 3},
        "expected_evaluation_level_counts": {1: 1, 2: 1, 3: 8},
    },
    "ausgrid2013": {
        "root": ROOT / "fhf_dedup_ausgrid2013",
        "horizons": (1,),
        "bottom_tags": ROOT / "Datasets" / "AusGrid2013" / "tags_dedup.pkl",
        "expected_structural_level_counts": {1: 1, 2: 100, 3: 259},
        "expected_evaluation_level_counts": {1: 1, 2: 60, 3: 299},
    },
}

# Internal experiment names are retained only in source_system for auditing.
SYSTEM_LABELS = {
    "FHF": "FHF",
    "HFF": "BO-FHF",
    "HFFPS": "CH-FHF",
}
BASE_LABELS = {
    "FHF": "FedAvg",
    "HFF": "HierAvg",
    "HFFPS": "HierAvgPS",
}
METHOD_LABELS = {
    "bu": "fBU",
    "mint_cov": "fMinT-cov",
    "mint_ols": "fMinT-OLS",
    "mint_shrinkage": "fMinTs",
    "mint_var": "fWLSv",
    "wls_structure": "fWLS-struct",
    "naive": "Naive",
}
PAPER_METHODS = ("base", "bu", "mint_shrinkage", "mint_var")
MCB_METHODS = (*PAPER_METHODS, "naive")
METRICS = ("MAE", "MASE", "RMSSE")
KEY_COLUMNS = ["ds", "horizon_index", "target_col"]
NON_FORECAST_COLUMNS = {
    "node_id", "unique_id", "role", "ds", "horizon_index",
    "target_col", "y_true",
}


def symmetric_scale_free_incoherence(
    parent_forecast: np.ndarray,
    children_sum_forecast: np.ndarray,
) -> np.ndarray:
    """Return a bounded, scale-free incoherence ratio per comparison.

    The ratio is ``2 * |parent - children_sum| / (|parent| +
    |children_sum|)``.  A comparison where both forecasts are exactly zero is
    defined as coherent (ratio zero).  The result is in [0, 2], or [0, 200]
    after conversion to percent.
    """
    parent = np.asarray(parent_forecast, dtype=float)
    children = np.asarray(children_sum_forecast, dtype=float)
    denominator = np.abs(parent) + np.abs(children)
    numerator = 2.0 * np.abs(parent - children)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def symmetric_error_scaled_incoherence(
    parent_forecast: np.ndarray,
    children_sum_forecast: np.ndarray,
    parent_actual: np.ndarray,
) -> np.ndarray:
    """Return incoherence relative to the mean error of both forecasts.

    For each parent-time comparison this is ``2 * |parent - children_sum| /
    (|parent - actual| + |children_sum - actual|)``.  By the triangle
    inequality it is in [0, 2].  When both forecasts equal the actual value,
    numerator and denominator are zero and the comparison contributes zero.

    Unlike :func:`symmetric_scale_free_incoherence`, this is deliberately an
    accuracy-conditioned diagnostic: inaccurate methods receive a larger
    scaler.  It is therefore reported alongside, rather than in place of, the
    forecast-scale measure.
    """
    parent = np.asarray(parent_forecast, dtype=float)
    children = np.asarray(children_sum_forecast, dtype=float)
    actual = np.asarray(parent_actual, dtype=float)
    denominator = np.abs(parent - actual) + np.abs(children - actual)
    numerator = 2.0 * np.abs(parent - children)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def display_method(source_system: str, method: str) -> str:
    if method == "base":
        return BASE_LABELS[source_system]
    if method == "naive":
        return "Naive"
    return f"{SYSTEM_LABELS[source_system]}_{METHOD_LABELS.get(method, method)}"


def method_display_order(methods: tuple[str, ...]) -> list[str]:
    """Return publication order, with the shared Naive baseline last."""
    order = [
        display_method(source_system, method)
        for source_system in SYSTEM_LABELS
        for method in methods
        if method != "naive"
    ]
    if "naive" in methods:
        order.append("Naive")
    return order


def sort_with_naive_last(frame: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Sort numerical outputs consistently within each horizon."""
    all_methods = ("base", *METHOD_LABELS.keys())
    order = method_display_order(all_methods)
    result = frame.copy()
    result["_method_order"] = pd.Categorical(result["method"], categories=order, ordered=True)
    sort_columns = ["horizon", "_method_order", *(extra or [])]
    return result.sort_values(sort_columns).drop(columns="_method_order").reset_index(drop=True)


def parent_relations(node_levels: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Infer direct parent-child relations from slash-delimited unique IDs."""
    nodes = set(node_levels["unique_id"].astype(str))
    level_by_node = node_levels.set_index("unique_id")["level"].astype(int).to_dict()
    relations: dict[str, list[str]] = {}
    for node in sorted(nodes):
        if "/" not in node:
            continue
        parent = node.rsplit("/", 1)[0]
        if parent in nodes:
            relations.setdefault(parent, []).append(node)
    relation_levels = {parent: f"L{level_by_node[parent]}" for parent in relations}
    return relations, relation_levels


def load_bottom_series(config: dict) -> set[str]:
    """Load the authoritative bottom-series membership from preprocessing."""
    path = config["bottom_tags"]
    with path.open("rb") as file:
        tags = pickle.load(file)
    if "bottom" not in tags:
        raise KeyError(f"{path} does not contain a 'bottom' tag")
    bottom = {str(value) for value in tags["bottom"]}
    if not bottom:
        raise ValueError(f"{path} contains an empty 'bottom' tag")
    return bottom


def assign_evaluation_levels(
    frame: pd.DataFrame,
    bottom_series: set[str],
) -> pd.DataFrame:
    """Assign semantic levels while retaining structural depth separately.

    Deduplication keeps the shallower identifier when a parent has one child.
    Such a retained node is structurally at the parent depth but is also an
    authoritative bottom series.  Accuracy summaries classify it as bottom;
    parent-child incoherence continues to use its structural depth.
    """
    result = frame.copy()
    result["structural_level"] = result["level"].astype(int)
    bottom_level = int(result["structural_level"].max())
    is_bottom = result["unique_id"].astype(str).isin(bottom_series)
    result.loc[is_bottom, "level"] = bottom_level
    result["level"] = result["level"].astype(int)
    result["role"] = np.where(
        result["level"].eq(result["level"].min()),
        "top",
        np.where(result["level"].eq(bottom_level), "bottom", "middle"),
    )
    return result


def load_accuracy(dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    audits = []
    node_reference = None
    bottom_series = load_bottom_series(config)
    for source_system, framework in SYSTEM_LABELS.items():
        for horizon in config["horizons"]:
            path = config["root"] / f"fh{horizon}_{source_system}_per_node_metrics.csv"
            frame = pd.read_csv(path)
            required = {"unique_id", "level", "method", *METRICS}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} missing columns: {sorted(missing)}")
            duplicate_rows = int(frame.duplicated(["unique_id", "method"]).sum())
            if duplicate_rows:
                raise ValueError(f"{path} has {duplicate_rows} duplicate node/method rows")
            structural_counts = frame.groupby("level")["unique_id"].nunique().to_dict()
            if structural_counts != config["expected_structural_level_counts"]:
                raise ValueError(
                    f"{path} structural level counts {structural_counts}, expected "
                    f"{config['expected_structural_level_counts']}"
                )
            current_nodes = frame[["unique_id", "level"]].drop_duplicates().sort_values("unique_id")
            if node_reference is None:
                node_reference = current_nodes
            elif not current_nodes.reset_index(drop=True).equals(node_reference.reset_index(drop=True)):
                raise ValueError(f"{path} node set differs from the other runs")

            missing_bottom = bottom_series.difference(set(frame["unique_id"].astype(str)))
            if missing_bottom:
                raise ValueError(f"{path} is missing bottom series: {sorted(missing_bottom)}")
            frame = assign_evaluation_levels(frame, bottom_series)
            evaluation_counts = frame.groupby("level")["unique_id"].nunique().to_dict()
            if evaluation_counts != config["expected_evaluation_level_counts"]:
                raise ValueError(
                    f"{path} evaluation level counts {evaluation_counts}, expected "
                    f"{config['expected_evaluation_level_counts']}"
                )

            frame["dataset"] = dataset
            frame["source_system"] = source_system
            frame["framework"] = framework
            frame["horizon"] = horizon
            frame["method_internal"] = frame["method"]
            frame["method"] = [display_method(source_system, method) for method in frame["method"]]
            frame["reconciliation"] = np.where(
                frame["method_internal"].isin(("base", "naive")), "None",
                frame["method_internal"].map(METHOD_LABELS).fillna(frame["method_internal"]),
            )
            frame["hierarchy_level"] = "L" + frame["level"].astype(int).astype(str)
            frames.append(frame)
            audits.append({
                "dataset": dataset,
                "source_system": source_system,
                "framework": framework,
                "horizon": horizon,
                "rows": len(frame),
                "nodes": frame["unique_id"].nunique(),
                "duplicate_node_method_rows": duplicate_rows,
                **{f"structural_L{level}_nodes": count for level, count in structural_counts.items()},
                **{f"L{level}_nodes": count for level, count in evaluation_counts.items()},
            })

    per_node = pd.concat(frames, ignore_index=True)
    # Naive is shared by all three experiment files. Keep one copy so it is a
    # single baseline in the numerical outputs and MCB tests.
    per_node = per_node.loc[
        ~(
            per_node["method_internal"].eq("naive")
            & per_node["source_system"].ne("FHF")
        )
    ].copy()
    accuracy = (
        per_node.groupby(
            ["dataset", "source_system", "framework", "horizon", "method_internal",
             "method", "reconciliation", "hierarchy_level"], as_index=False
        )
        .agg(MAE=("MAE", "mean"), MASE=("MASE", "mean"), RMSSE=("RMSSE", "mean"),
             n_nodes=("unique_id", "nunique"))
    )
    accuracy = sort_with_naive_last(accuracy, ["hierarchy_level"])
    return per_node, accuracy, pd.DataFrame(audits)


def incoherence_for_file(
    path: Path,
    dataset: str,
    source_system: str,
    framework: str,
    horizon: int,
    relations: dict[str, list[str]],
    relation_levels: dict[str, str],
    chunksize: int = 150_000,
) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    methods = [column for column in columns if column not in NON_FORECAST_COLUMNS]
    usecols = ["unique_id", "y_true", *KEY_COLUMNS, *methods]
    relevant_nodes = {node for parent, children in relations.items() for node in (parent, *children)}
    partials = []

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.loc[chunk["unique_id"].isin(relevant_nodes)]
        for parent, children in relations.items():
            part = chunk.loc[chunk["unique_id"].isin({parent, *children})]
            if part.empty:
                continue
            is_parent = part["unique_id"].eq(parent)
            work = part[KEY_COLUMNS].copy()
            work["parent_id"] = parent
            work["parent_actual"] = part["y_true"].where(is_parent, 0.0)
            signs = np.where(is_parent, 1.0, -1.0)
            for method in methods:
                work[f"diff__{method}"] = part[method] * signs
                work[f"parent_forecast__{method}"] = part[method].where(is_parent, 0.0)
                work[f"children_sum_forecast__{method}"] = part[method].where(~is_parent, 0.0)
            partials.append(work.groupby(["parent_id", *KEY_COLUMNS], as_index=False).sum(numeric_only=True))

    totals = (
        pd.concat(partials, ignore_index=True)
        .groupby(["parent_id", *KEY_COLUMNS], as_index=False)
        .sum(numeric_only=True)
    )
    records = []
    for method in methods:
        for parent, rows in totals.groupby("parent_id"):
            incoherence = float(rows[f"diff__{method}"].abs().sum())
            actual = float(rows["parent_actual"].abs().sum())
            scale_free_ratios = symmetric_scale_free_incoherence(
                rows[f"parent_forecast__{method}"].to_numpy(),
                rows[f"children_sum_forecast__{method}"].to_numpy(),
            )
            error_scaled_ratios = symmetric_error_scaled_incoherence(
                rows[f"parent_forecast__{method}"].to_numpy(),
                rows[f"children_sum_forecast__{method}"].to_numpy(),
                rows["parent_actual"].to_numpy(),
            )
            records.append({
                "dataset": dataset,
                "source_system": source_system,
                "framework": framework,
                "horizon": horizon,
                "method_internal": method,
                "method": display_method(source_system, method),
                "level": relation_levels[parent],
                "parent_id": parent,
                "incoherence_total": incoherence,
                "parent_actual_demand_total": actual,
                "incoherence_ratio_pct": 100.0 * incoherence / actual if actual else np.nan,
                "average_scale_free_incoherence_ratio_pct": 100.0 * float(scale_free_ratios.mean()),
                "average_error_scaled_incoherence_ratio_pct": (
                    100.0 * float(error_scaled_ratios.mean())
                ),
                "n_time_points": len(rows),
            })
    return pd.DataFrame(records)


def wide_mcb_table(per_node: pd.DataFrame, metric: str) -> pd.DataFrame:
    paper = per_node.loc[per_node["method_internal"].isin(MCB_METHODS)].copy()
    wide = paper.pivot(
        index=["dataset", "horizon", "unique_id", "level"],
        columns="method",
        values=metric,
    ).reset_index()
    method_order = method_display_order(MCB_METHODS)
    missing = set(method_order).difference(wide.columns)
    if missing:
        raise ValueError(f"Missing {metric} MCB columns: {sorted(missing)}")
    return wide[["dataset", "horizon", "unique_id", "level", *method_order]]


def pivot_summary(accuracy: pd.DataFrame) -> pd.DataFrame:
    paper = accuracy.loc[accuracy["method_internal"].isin(MCB_METHODS)].copy()
    result = paper.pivot(
        index=["dataset", "source_system", "framework", "horizon", "method_internal", "method", "reconciliation"],
        columns="hierarchy_level",
        values=["MAE", "MASE", "RMSSE", "n_nodes"],
    ).reset_index()
    result.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple) else column
        for column in result.columns
    ]
    return sort_with_naive_last(result)


def scale_free_summary_by_method(incoherence_level: pd.DataFrame) -> pd.DataFrame:
    """Combine hierarchy levels with one vote per parent-time comparison."""
    keys = [
        "dataset", "source_system", "framework", "horizon",
        "method_internal", "method",
    ]
    result = (
        incoherence_level.assign(
            _weighted_ratio=lambda frame: (
                frame["average_scale_free_incoherence_ratio_pct"]
                * frame["n_parent_time_comparisons"]
            ),
            _weighted_error_scaled_ratio=lambda frame: (
                frame["average_error_scaled_incoherence_ratio_pct"]
                * frame["n_parent_time_comparisons"]
            ),
        )
        .groupby(keys, as_index=False)
        .agg(
            _weighted_ratio=("_weighted_ratio", "sum"),
            _weighted_error_scaled_ratio=("_weighted_error_scaled_ratio", "sum"),
            n_parent_time_comparisons=("n_parent_time_comparisons", "sum"),
        )
    )
    result["average_scale_free_incoherence_ratio_pct"] = (
        result.pop("_weighted_ratio") / result["n_parent_time_comparisons"]
    )
    result["average_error_scaled_incoherence_ratio_pct"] = (
        result.pop("_weighted_error_scaled_ratio")
        / result["n_parent_time_comparisons"]
    )
    return sort_with_naive_last(result)


def generate_dataset(dataset: str, config: dict) -> dict:
    output_dir = OUTPUT_ROOT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    per_node, accuracy, audit = load_accuracy(dataset, config)
    node_levels = (
        per_node[["unique_id", "structural_level"]]
        .drop_duplicates()
        .rename(columns={"structural_level": "level"})
    )
    relations, relation_levels = parent_relations(node_levels)

    incoherence_parts = []
    for source_system, framework in SYSTEM_LABELS.items():
        for horizon in config["horizons"]:
            path = config["root"] / f"fh{horizon}_{source_system}_forecasts.csv"
            print(f"[{dataset}] incoherence: {path.name}", flush=True)
            incoherence_parts.append(incoherence_for_file(
                path, dataset, source_system, framework, horizon, relations, relation_levels
            ))
    incoherence_parent = pd.concat(incoherence_parts, ignore_index=True)
    incoherence_parent = incoherence_parent.loc[
        ~(
            incoherence_parent["method_internal"].eq("naive")
            & incoherence_parent["source_system"].ne("FHF")
        )
    ].copy()
    incoherence_parent = sort_with_naive_last(incoherence_parent, ["level", "parent_id"])
    incoherence_level = (
        incoherence_parent.assign(
            _scale_free_ratio_sum=lambda frame: (
                frame["average_scale_free_incoherence_ratio_pct"] * frame["n_time_points"]
            ),
            _error_scaled_ratio_sum=lambda frame: (
                frame["average_error_scaled_incoherence_ratio_pct"] * frame["n_time_points"]
            ),
        ).groupby(
            ["dataset", "source_system", "framework", "horizon", "method_internal", "method", "level"],
            as_index=False,
        )
        .agg(incoherence_total=("incoherence_total", "sum"),
             parent_actual_demand_total=("parent_actual_demand_total", "sum"),
             n_parent_time_comparisons=("n_time_points", "sum"),
             _scale_free_ratio_sum=("_scale_free_ratio_sum", "sum"),
             _error_scaled_ratio_sum=("_error_scaled_ratio_sum", "sum"))
    )
    incoherence_level["incoherence_ratio_pct"] = (
        100.0 * incoherence_level["incoherence_total"] / incoherence_level["parent_actual_demand_total"]
    )
    incoherence_level["average_scale_free_incoherence_ratio_pct"] = (
        incoherence_level.pop("_scale_free_ratio_sum")
        / incoherence_level["n_parent_time_comparisons"]
    )
    incoherence_level["average_error_scaled_incoherence_ratio_pct"] = (
        incoherence_level.pop("_error_scaled_ratio_sum")
        / incoherence_level["n_parent_time_comparisons"]
    )
    incoherence_level = sort_with_naive_last(incoherence_level, ["level"])
    scale_free_by_method = scale_free_summary_by_method(incoherence_level)

    paper_summary = pivot_summary(accuracy)
    incoherence_summary = incoherence_level.pivot(
        index=["dataset", "source_system", "framework", "horizon", "method_internal", "method"],
        columns="level",
        values=[
            "incoherence_total",
            "incoherence_ratio_pct",
            "average_scale_free_incoherence_ratio_pct",
            "average_error_scaled_incoherence_ratio_pct",
        ],
    ).reset_index()
    incoherence_summary.columns = [
        "_".join(str(part) for part in column if part).strip("_")
        if isinstance(column, tuple) else column
        for column in incoherence_summary.columns
    ]
    paper_summary = paper_summary.merge(
        incoherence_summary,
        on=["dataset", "source_system", "framework", "horizon", "method_internal", "method"],
        how="left",
        validate="one_to_one",
    )

    outputs = {
        "paper_summary": output_dir / "deduplicated_paper_summary.csv",
        "accuracy_by_level": output_dir / "deduplicated_accuracy_by_level.csv",
        "incoherence_by_level": output_dir / "deduplicated_incoherence_by_level.csv",
        "incoherence_by_parent": output_dir / "deduplicated_incoherence_by_parent.csv",
        "scale_free_incoherence_by_method": (
            output_dir / "deduplicated_average_scale_free_incoherence_by_method.csv"
        ),
        "audit": output_dir / "deduplication_audit.csv",
        "mase": output_dir / "all_per-node_mase.csv",
        "rmsse": output_dir / "all_per-node_rmsse.csv",
        "mae": output_dir / "all_per-node_mae.csv",
    }
    paper_summary.to_csv(outputs["paper_summary"], index=False)
    accuracy.to_csv(outputs["accuracy_by_level"], index=False)
    incoherence_level.to_csv(outputs["incoherence_by_level"], index=False)
    incoherence_parent.to_csv(outputs["incoherence_by_parent"], index=False)
    scale_free_by_method.to_csv(outputs["scale_free_incoherence_by_method"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    for metric, key in (("MASE", "mase"), ("RMSSE", "rmsse"), ("MAE", "mae")):
        wide_mcb_table(per_node, metric).to_csv(outputs[key], index=False)

    coherent_methods = incoherence_level.loc[
        incoherence_level["method_internal"].isin(("bu", "mint_shrinkage", "mint_var")),
        "incoherence_total",
    ]
    coherent_scale_free = incoherence_level.loc[
        incoherence_level["method_internal"].isin(("bu", "mint_shrinkage", "mint_var")),
        "average_scale_free_incoherence_ratio_pct",
    ]
    validation = {
        "dataset": dataset,
        "frameworks": sorted(accuracy["framework"].unique()),
        "base_methods": sorted(per_node.loc[per_node["method_internal"].eq("base"), "method"].unique()),
        "horizons": list(config["horizons"]),
        "structural_level_node_counts": {
            f"L{k}": v for k, v in config["expected_structural_level_counts"].items()
        },
        "level_node_counts": {
            f"L{k}": v for k, v in config["expected_evaluation_level_counts"].items()
        },
        "mcb_method_columns": wide_mcb_table(per_node, "MASE").columns[4:].tolist(),
        "max_selected_coherent_incoherence": float(coherent_methods.max()),
        "max_selected_coherent_average_scale_free_incoherence_ratio_pct": (
            float(coherent_scale_free.max())
        ),
        "all_selected_reconciled_numerically_coherent": bool((coherent_methods < 1e-6).all()),
        "all_average_scale_free_ratios_within_bounds": bool(
            incoherence_level["average_scale_free_incoherence_ratio_pct"].between(
                0.0, 200.0 + 1e-10
            ).all()
        ),
        "all_average_error_scaled_ratios_within_bounds": bool(
            incoherence_level["average_error_scaled_incoherence_ratio_pct"].between(
                0.0, 200.0 + 1e-10
            ).all()
        ),
    }
    (output_dir / "paper_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    validations = [generate_dataset(dataset, config) for dataset, config in DATASETS.items()]
    # Retain the historical metrics and append the primary training-scaled one.
    from PPHF_modules.results_check.deduplication_results.add_masi import main as add_masi_main
    add_masi_main()
    # Publish names only after all computations and scale audits finish.
    for dataset in DATASETS:
        for path in (OUTPUT_ROOT / dataset).glob("*.csv"):
            publication_table(pd.read_csv(path)).to_csv(path, index=False)
    (OUTPUT_ROOT / "paper_validation.json").write_text(json.dumps(validations, indent=2), encoding="utf-8")
    print(json.dumps(validations, indent=2))


if __name__ == "__main__":
    main()
