#!/usr/bin/env python3
"""Compact error-redistribution analysis not duplicated by the MCB tests.

MCB already compares node-level method ranks. This module instead asks where
errors move after reconciliation. fMinTs and fWLSv are each compared with the
base forecast from the same approach and forecast file.
"""

from __future__ import annotations

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = PROJECT_ROOT
OUT_ROOT = RESULTS_DIR / "deduplication_results" / "error_analysis"
from PPHF_modules.results_check.publication_labels import publication_table, publication_text
from PPHF_modules.results_check.current_results import dataset_config, training_scales, source_manifest
from PPHF_modules.results_check.deduplication_results.generate_deduplication_results import assign_evaluation_levels, load_bottom_series
SEED = 20260808
N_BOOTSTRAP = 2000
LAGS = 48
TEST_RATIO = 0.20
EPS = 1e-12
RECONCILIATIONS = {"fMinTs": "mint_shrinkage", "fWLSv": "mint_var"}


@dataclass(frozen=True)
class Experiment:
    dataset: str
    horizon: int
    approach: str
    forecasts: Path
    training_artifacts: Path
    bottom_tags: Path


def current_experiments(root=ROOT):
    experiments = []
    for dataset in ('ausgrid2013', 'gef2017'):
        config = dataset_config(dataset, root)
        for horizon in config['horizons']:
            for system in ('FHF', 'HFF', 'HFFPS'):
                prefix = f'fh{horizon}_{system}'
                experiments.append(Experiment(dataset, horizon, system,
                    config['root']/f'{prefix}_forecasts.csv',
                    config['root']/f'{prefix}_training_artifacts.npz', config['bottom_tags']))
    return experiments


EXPERIMENTS = current_experiments()


def _load_observations(exp: Experiment) -> pd.DataFrame:
    keys = ["unique_id", "ds", "horizon_index", "target_col"]
    data = pd.read_csv(
        exp.forecasts,
        usecols=keys + ["node_id", "y_true", "base", *RECONCILIATIONS.values()],
    ).rename(
        columns={
            "base": "pred_base",
            "mint_shrinkage": "pred_mints",
            "mint_var": "pred_wlsv",
        }
    )
    data["unique_id"] = data["unique_id"].astype(str)
    data["level"] = data["unique_id"].str.count("/") + 1
    bottom = load_bottom_series({"bottom_tags": exp.bottom_tags})
    data = assign_evaluation_levels(data, bottom, dataset_config(exp.dataset)["evaluation_level_policy"])
    data["ds"] = pd.to_datetime(data["ds"], errors="raise")
    data["block"] = data["ds"].dt.floor("D")
    if data.duplicated(keys).any():
        raise ValueError(f"duplicate paired observations: {exp.forecasts}")
    return data.sort_values(keys).reset_index(drop=True)


def _training_scales(exp: Experiment, unique_ids: list[str], n_horizons: int) -> pd.DataFrame:
    scales = training_scales(exp.training_artifacts)
    if set(scales['unique_id']) != set(unique_ids):
        raise ValueError(f'Training artifacts and forecasts contain different nodes: {exp}')
    if n_horizons != exp.horizon:
        raise ValueError(f'Forecast horizon mismatch: {exp}')
    return scales[['unique_id', 'mase_scale', 'rmsse_scale']]


def _add_scaled_errors(exp: Experiment, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scales = _training_scales(exp, data["unique_id"].unique().tolist(), data["horizon_index"].nunique())
    data = data.merge(scales, on="unique_id", validate="many_to_one")
    for method in ("base", "mints", "wlsv"):
        error = data["y_true"] - data[f"pred_{method}"]
        data[f"ase_{method}"] = error.abs() / data["mase_scale"]
        data[f"sse_{method}"] = error.pow(2) / data["rmsse_scale"]
    return data, scales


def _node_metrics(data: pd.DataFrame, method: str) -> pd.DataFrame:
    return (
        data.groupby(["node_id", "unique_id", "level"], as_index=False)
        .agg(MASE=(f"ase_{method}", "mean"), mean_SSE=(f"sse_{method}", "mean"), n_obs=("y_true", "size"))
        .assign(RMSSE=lambda x: np.sqrt(x["mean_SSE"]))
    )


def _bootstrap_weights(data: pd.DataFrame, seed: int, n_bootstrap=N_BOOTSTRAP) -> tuple[np.ndarray, list]:
    blocks = sorted(data["block"].unique())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=n_bootstrap)
    return weights.astype(float), blocks


def _block_node_matrix(data: pd.DataFrame, blocks: list, nodes: list[int], column: str) -> tuple[np.ndarray, np.ndarray]:
    sums = data.pivot_table(index="block", columns="node_id", values=column, aggfunc="sum").reindex(index=blocks, columns=nodes).fillna(0)
    counts = data.assign(_n=1).pivot_table(index="block", columns="node_id", values="_n", aggfunc="sum").reindex(index=blocks, columns=nodes).fillna(0)
    return sums.to_numpy(float), counts.to_numpy(float)


def _bootstrap_level_metrics(data: pd.DataFrame, method: str, weights: np.ndarray, blocks: list) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    nodes = sorted(data["node_id"].unique())
    levels = data[["node_id", "level"]].drop_duplicates().set_index("node_id")["level"]
    ase, counts = _block_node_matrix(data, blocks, nodes, f"ase_{method}")
    sse, _ = _block_node_matrix(data, blocks, nodes, f"sse_{method}")
    denom = weights @ counts
    node_mase = (weights @ ase) / denom
    node_rmsse = np.sqrt((weights @ sse) / denom)
    return {
        int(level): (
            node_mase[:, np.array([levels.loc[n] == level for n in nodes])].mean(axis=1),
            node_rmsse[:, np.array([levels.loc[n] == level for n in nodes])].mean(axis=1),
        )
        for level in sorted(levels.unique())
    }


def _level_results(exp: Experiment, data: pd.DataFrame, weights: np.ndarray, blocks: list) -> pd.DataFrame:
    observed = {m: _node_metrics(data, m) for m in ("base", "mints", "wlsv")}
    bootstrap = {m: _bootstrap_level_metrics(data, m, weights, blocks) for m in observed}
    rows = []
    for label, method in (("fMinTs", "mints"), ("fWLSv", "wlsv")):
        for level in sorted(data["level"].unique()):
            base = observed["base"].query("level == @level")
            target = observed[method].query("level == @level")
            dm = bootstrap[method][int(level)][0] - bootstrap["base"][int(level)][0]
            dr = bootstrap[method][int(level)][1] - bootstrap["base"][int(level)][1]
            rows.append(
                {
                    "dataset": exp.dataset,
                    "forecast_horizon": exp.horizon,
                    "approach": exp.approach,
                    "reconciliation": label,
                    "level": int(level),
                    "base_MASE": base.MASE.mean(),
                    "reconciled_MASE": target.MASE.mean(),
                    "delta_MASE": target.MASE.mean() - base.MASE.mean(),
                    "delta_MASE_ci_low": np.quantile(dm, 0.025),
                    "delta_MASE_ci_high": np.quantile(dm, 0.975),
                    "base_RMSSE": base.RMSSE.mean(),
                    "reconciled_RMSSE": target.RMSSE.mean(),
                    "delta_RMSSE": target.RMSSE.mean() - base.RMSSE.mean(),
                    "delta_RMSSE_ci_low": np.quantile(dr, 0.025),
                    "delta_RMSSE_ci_high": np.quantile(dr, 0.975),
                    "n_nodes": base.node_id.nunique(),
                    "n_observations": int((data.level == level).sum()),
                }
            )
    return pd.DataFrame(rows)


def _node_appendix(exp: Experiment, data: pd.DataFrame) -> pd.DataFrame:
    level3 = data[data.level == 3]
    base = _node_metrics(level3, "base").set_index("node_id")
    rows = []
    for label, method in (("fMinTs", "mints"), ("fWLSv", "wlsv")):
        target = _node_metrics(level3, method).set_index("node_id")
        dm = target.MASE - base.MASE
        dr = target.RMSSE - base.RMSSE
        rows.append(
            {
                "dataset": exp.dataset,
                "forecast_horizon": exp.horizon,
                "approach": exp.approach,
                "reconciliation": label,
                "n_nodes": len(dm),
                "MASE_improved_count": int((dm < 0).sum()),
                "MASE_improved_proportion": float((dm < 0).mean()),
                "RMSSE_improved_count": int((dr < 0).sum()),
                "RMSSE_improved_proportion": float((dr < 0).mean()),
                "both_improved_proportion": float(((dm < 0) & (dr < 0)).mean()),
                "MASE_worse_RMSSE_better_proportion": float(((dm > 0) & (dr < 0)).mean()),
                "both_worse_proportion": float(((dm > 0) & (dr > 0)).mean()),
                "median_delta_MASE": dm.median(),
                "IQR_delta_MASE": dm.quantile(.75) - dm.quantile(.25),
                "median_delta_RMSSE": dr.median(),
                "IQR_delta_RMSSE": dr.quantile(.75) - dr.quantile(.25),
            }
        )
    return pd.DataFrame(rows)


def _decile_results(exp: Experiment, data: pd.DataFrame, weights: np.ndarray, blocks: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    level3 = data[data.level == 3].copy()
    level3["decile"] = pd.qcut(level3.ase_base.rank(method="first"), 10, labels=False).astype(int) + 1
    decile_rows, tail_rows = [], []
    for label, method in (("fMinTs", "mints"), ("fWLSv", "wlsv")):
        delta_ase = f"delta_ase_{method}"
        delta_sse = f"delta_sse_{method}"
        level3[delta_ase] = level3[f"ase_{method}"] - level3.ase_base
        level3[delta_sse] = level3[f"sse_{method}"] - level3.sse_base
        grouped = level3.groupby("decile", as_index=False).agg(
            base_ASE_min=("ase_base", "min"),
            base_ASE_max=("ase_base", "max"),
            n_observations=(delta_ase, "size"),
            mean_delta_ASE=(delta_ase, "mean"),
            median_delta_ASE=(delta_ase, "median"),
            improvement_proportion=(delta_ase, lambda x: float((x < 0).mean())),
            mean_base_ASE=("ase_base", "mean"),
            mean_reconciled_ASE=(f"ase_{method}", "mean"),
            mean_base_SSE=("sse_base", "mean"),
            mean_reconciled_SSE=(f"sse_{method}", "mean"),
        )
        sums = level3.pivot_table(index="block", columns="decile", values=delta_ase, aggfunc="sum").reindex(index=blocks, columns=range(1, 11)).fillna(0)
        counts = level3.assign(_n=1).pivot_table(index="block", columns="decile", values="_n", aggfunc="sum").reindex(index=blocks, columns=range(1, 11)).fillna(0)
        boot = (weights @ sums.to_numpy(float)) / (weights @ counts.to_numpy(float))
        grouped["mean_delta_ASE_ci_low"] = np.quantile(boot, .025, axis=0)
        grouped["mean_delta_ASE_ci_high"] = np.quantile(boot, .975, axis=0)
        grouped.insert(0, "reconciliation", label)
        grouped.insert(0, "approach", exp.approach)
        grouped.insert(0, "forecast_horizon", exp.horizon)
        grouped.insert(0, "dataset", exp.dataset)
        decile_rows.append(grouped)

        top = level3[level3.decile == 10]
        total_sse_change = level3[delta_sse].sum()
        tail_sse_change = top[delta_sse].sum()
        tail_rows.append(
            {
                "dataset": exp.dataset,
                "forecast_horizon": exp.horizon,
                "approach": exp.approach,
                "reconciliation": label,
                "n_observations": len(top),
                "improvement_proportion": float((top[delta_ase] < 0).mean()),
                "mean_ASE_reduction": float(top.ase_base.mean() - top[f"ase_{method}"].mean()),
                "top10_delta_SSE_sum": tail_sse_change,
                "overall_delta_SSE_sum": total_sse_change,
                "share_of_overall_SSE_change": tail_sse_change / total_sse_change if abs(total_sse_change) > EPS else np.nan,
            }
        )
    return pd.concat(decile_rows, ignore_index=True), pd.DataFrame(tail_rows)


def _reproduction_check(exp: Experiment, data: pd.DataFrame) -> pd.DataFrame:
    saved_path = exp.forecasts.with_name(exp.forecasts.name.replace("_forecasts", "_per_node_metrics"))
    saved = pd.read_csv(saved_path)
    bottom = load_bottom_series({"bottom_tags": exp.bottom_tags})
    saved = assign_evaluation_levels(saved, bottom, dataset_config(exp.dataset)["evaluation_level_policy"])
    saved = saved.groupby(["method", "level"], as_index=False)[["MASE", "RMSSE"]].mean()
    rows = []
    for label, method, saved_method in (("base", "base", "base"), ("fMinTs", "mints", "mint_shrinkage"), ("fWLSv", "wlsv", "mint_var")):
        calc = _node_metrics(data, method).groupby("level", as_index=False).agg(MASE=("MASE", "mean"), RMSSE=("RMSSE", "mean"))
        reference = saved.query("method == @saved_method")[["level", "MASE", "RMSSE"]]
        merged = calc.merge(reference, on="level", suffixes=("_calculated", "_saved"), validate="one_to_one")
        for row in merged.itertuples():
            rows.append(
                {
                    "dataset": exp.dataset,
                    "forecast_horizon": exp.horizon,
                    "approach": exp.approach,
                    "method": label,
                    "level": row.level,
                    "abs_diff_MASE": abs(row.MASE_calculated - row.MASE_saved),
                    "abs_diff_RMSSE": abs(row.RMSSE_calculated - row.RMSSE_saved),
                }
            )
    check = pd.DataFrame(rows)
    if check[["abs_diff_MASE", "abs_diff_RMSSE"]].to_numpy().max() > 1e-6:
        raise ValueError(f"saved metrics not reproduced: {exp}")
    return check


def _validation(exp: Experiment, data: pd.DataFrame, scales: pd.DataFrame, n_bootstrap=N_BOOTSTRAP) -> dict:
    required = ["y_true", "pred_base", "pred_mints", "pred_wlsv", "mase_scale", "rmsse_scale"]
    result = {
        "dataset": exp.dataset,
        "forecast_horizon": exp.horizon,
        "approach": exp.approach,
        "forecast_file": str(exp.forecasts),
        "non_dp_source": "Non-DP experiment forecast CSV; no DP output is read",
        "n_rows": len(data),
        "n_nodes": data.node_id.nunique(),
        "n_origins": data.ds.nunique(),
        "n_time_blocks": data.block.nunique(),
        "n_horizons": data.horizon_index.nunique(),
        "missing_values": int(data[required].isna().sum().sum()),
        "nonpositive_mase_scales": int((scales.mase_scale <= EPS).sum()),
        "nonpositive_rmsse_scales": int((scales.rmsse_scale <= EPS).sum()),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": n_bootstrap,
        "data_scope": "current_deduplicated_non_dp_runs",
        "training_artifacts": str(exp.training_artifacts),
        "structural_level_counts": data.groupby("structural_level").unique_id.nunique().to_dict(),
        "evaluation_level_counts": data.groupby("level").unique_id.nunique().to_dict(),
        "bootstrap_unit": "calendar-day test-time block retaining all nodes and horizons",
    }
    if any(result[k] for k in ("missing_values", "nonpositive_mase_scales", "nonpositive_rmsse_scales")):
        raise ValueError(f"validation failed: {result}")
    return result


def _colour(value: float, limit: float) -> tuple[int, int, int]:
    ratio = min(abs(value) / limit, 1.0) if limit else 0
    white = np.array([248, 248, 248], float)
    endpoint = np.array([38, 112, 170] if value < 0 else [202, 64, 55], float)
    return tuple(np.rint(white * (1 - ratio) + endpoint * ratio).astype(int))


def _save_pdf(png: Path, pdf: Path, width_pt: float, height_pt: float) -> None:
    # Reuse Matplotlib, already needed by the project, for portable PDF export.
    import matplotlib.pyplot as plt
    with Image.open(png) as source:
        figure = plt.figure(figsize=(width_pt / 72, height_pt / 72), dpi=300)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.imshow(source)
        axes.set_axis_off()
        figure.savefig(pdf, format='pdf', dpi=300)
        plt.close(figure)


def _heatmap(deciles: pd.DataFrame, out: Path, dataset: str) -> None:
    row_keys = deciles[["forecast_horizon", "approach", "reconciliation"]].drop_duplicates().itertuples(index=False, name=None)
    rows = list(row_keys)
    cell_w, cell_h = 130, 108
    left, top, right, bottom = 350, 145, 30, 105
    width, height = left + 10 * cell_w + right, top + len(rows) * cell_h + bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    from matplotlib import font_manager
    font_path = font_manager.findfont('DejaVu Sans')
    bold_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
    font = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 22)
    bold = ImageFont.truetype(bold_path, 25)
    limit = float(np.nanquantile(np.abs(deciles.mean_delta_ASE), .95)) or 1.0
    draw.text((25, 25), f"{dataset}: mean Delta ASE after reconciliation", fill="black", font=font)
    draw.text((25, 72), "Cell: mean Delta ASE / share improved; blue = improvement; bold border = 95% CI excludes zero", fill="#444444", font=small)
    for d in range(1, 11):
        draw.text((left + (d - 0.5) * cell_w, top - 42), str(d), fill="black", font=font, anchor="mt")
    draw.text((left + 5 * cell_w, top + len(rows) * cell_h + 32), "Pre-reconciliation base ASE decile", fill="black", font=font, anchor="mt")
    for r, (horizon, approach, reconciliation) in enumerate(rows):
        display_approach = {"HFF": "HFF", "HFFPS": "HFFPS"}.get(approach, approach)
        label = f"{display_approach}  h={horizon}  {reconciliation}"
        subset = deciles.query("forecast_horizon == @horizon and approach == @approach and reconciliation == @reconciliation").sort_values("decile")
        for c, row in enumerate(subset.itertuples()):
            x0, y0 = left + c * cell_w, top + r * cell_h
            excludes_zero = row.mean_delta_ASE_ci_low > 0 or row.mean_delta_ASE_ci_high < 0
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=_colour(row.mean_delta_ASE, limit), outline="black", width=5 if excludes_zero else 1)
            text = f"{row.mean_delta_ASE:+.3f}\n{row.improvement_proportion:.0%}"
            draw.multiline_text((x0 + cell_w / 2, y0 + cell_h / 2), text, fill="black", font=bold if excludes_zero else small, spacing=6, align="center", anchor="mm")
        draw.text((left - 18, top + r * cell_h + cell_h / 2), label, fill="black", font=small, anchor="rm")
    png = out / "error_redistribution_heatmap.png"
    img.save(png, dpi=(300, 300))
    _save_pdf(png, out / "error_redistribution_heatmap.pdf", width / 300 * 72, height / 300 * 72)


def _write_narratives(dataset: str, level: pd.DataFrame, tail: pd.DataFrame, out: Path) -> None:
    lines_en = [f"# Error redistribution analysis ({dataset})", "", "All comparisons are strict within-approach comparisons of a reconciled forecast against its own base forecast. Negative differences indicate improvement.", ""]
    lines_zh = [f"# Error redistribution analysis ({dataset})", "", "All comparisons are between a reconciled result and its own base within the same training approach. Negative differences indicate improvement.", ""]
    for row in level.query("level == 3").itertuples():
        t = tail.query("forecast_horizon == @row.forecast_horizon and approach == @row.approach and reconciliation == @row.reconciliation").iloc[0]
        lines_en.append(
            f"- {row.approach} fh{row.forecast_horizon} {row.reconciliation}: level-3 delta MASE {row.delta_MASE:+.4f} (95% CI [{row.delta_MASE_ci_low:+.4f}, {row.delta_MASE_ci_high:+.4f}]) and delta RMSSE {row.delta_RMSSE:+.4f} (95% CI [{row.delta_RMSSE_ci_low:+.4f}, {row.delta_RMSSE_ci_high:+.4f}]). In the largest base-error decile, {t.improvement_proportion:.1%} improved and mean ASE fell by {t.mean_ASE_reduction:.4f}."
        )
        lines_zh.append(
            f"- {row.approach} fh{row.forecast_horizon} {row.reconciliation}: level 3 Delta MASE is {row.delta_MASE:+.4f} (95% CI [{row.delta_MASE_ci_low:+.4f}, {row.delta_MASE_ci_high:+.4f}]), and Delta RMSSE is {row.delta_RMSSE:+.4f} (95% CI [{row.delta_RMSSE_ci_low:+.4f}, {row.delta_RMSSE_ci_high:+.4f}]); in the decile with the largest base errors, {t.improvement_proportion:.1%} of observations improve, with a mean ASE reduction of {t.mean_ASE_reduction:.4f}."
        )
    lines_en += ["", "The decile pattern is descriptive. Because groups are defined by the base error, regression to the mean may contribute and the results should not be interpreted as a causal mechanism."]
    lines_zh += ["", "Deciles are defined by base errors before reconciliation, so regression to the mean may contribute to the observed patterns. These patterns are descriptive evidence, not a causal mechanism."]
    (out / "paper_analysis_en.md").write_text(publication_text("\n".join(lines_en) + "\n"), encoding="utf-8")
    (out / "results_interpretation_zh.md").write_text(publication_text("\n".join(lines_zh) + "\n"), encoding="utf-8")


def run_all(*, root=ROOT, output_root=None, n_bootstrap=N_BOOTSTRAP, force=False) -> pd.DataFrame:
    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be positive.')
    output_root = Path(output_root).resolve() if output_root else Path(root)/'Results_check/deduplication_results/error_analysis'
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = current_experiments(root)
    paths = [Path(__file__), Path(__file__).parents[1]/'current_results.py',
             Path(__file__).parents[1]/'deduplication_results/generate_deduplication_results.py']
    for exp in experiments:
        paths += [exp.forecasts, exp.training_artifacts, exp.bottom_tags,
                  exp.forecasts.with_name(exp.forecasts.name.replace('_forecasts', '_per_node_metrics'))]
    signature = {'data_scope':'current_deduplicated_non_dp_runs', 'seed':SEED,
                 'evaluation_level_policies': {d: dataset_config(d)['evaluation_level_policy']
                                              for d in ('ausgrid2013','gef2017')},
                 'bootstrap_replicates':n_bootstrap, 'sources':source_manifest(paths, root)}
    manifest_path = output_root/'source_manifest.json'
    expected = [output_root/dataset/name for dataset in ('ausgrid2013','gef2017')
                for name in ('level_effects_bootstrap.csv','error_deciles.csv','largest_error_decile.csv',
                             'appendix_node_improvement.csv','reproduction_check.csv','validation.json',
                             'error_redistribution_heatmap.png','error_redistribution_heatmap.pdf')]
    if not force and manifest_path.exists() and all(path.exists() for path in expected):
        if json.loads(manifest_path.read_text()) == signature:
            print('Verified current inputs unchanged; loading current error analysis.', flush=True)
            return pd.read_csv(output_root/'analysis_manifest.csv')
    collected: dict[str, dict[str, list]] = {}
    manifest_rows = []
    for i, exp in enumerate(experiments):
        print(f"[{exp.dataset}] {exp.approach} h={exp.horizon}: bootstrap and reproduction", flush=True)
        data = _load_observations(exp)
        data, scales = _add_scaled_errors(exp, data)
        validation = _validation(exp, data, scales, n_bootstrap)
        weights, blocks = _bootstrap_weights(data, SEED + i, n_bootstrap)
        level = _level_results(exp, data, weights, blocks)
        appendix = _node_appendix(exp, data)
        deciles, tail = _decile_results(exp, data, weights, blocks)
        reproduction = _reproduction_check(exp, data)
        bucket = collected.setdefault(exp.dataset, {"level": [], "appendix": [], "deciles": [], "tail": [], "reproduction": [], "validation": []})
        bucket["level"].append(level)
        bucket["appendix"].append(appendix)
        bucket["deciles"].append(deciles)
        bucket["tail"].append(tail)
        bucket["reproduction"].append(reproduction)
        bucket["validation"].append(validation)
        manifest_rows.append({"dataset": exp.dataset, "forecast_horizon": exp.horizon, "approach": exp.approach, "forecast_file": str(exp.forecasts), **{k: validation[k] for k in ("n_rows", "n_nodes", "n_origins", "n_time_blocks", "n_horizons")}})

    for dataset, bucket in collected.items():
        out = output_root / dataset
        out.mkdir(parents=True, exist_ok=True)
        level = pd.concat(bucket["level"], ignore_index=True)
        appendix = pd.concat(bucket["appendix"], ignore_index=True)
        deciles = pd.concat(bucket["deciles"], ignore_index=True)
        tail = pd.concat(bucket["tail"], ignore_index=True)
        reproduction = pd.concat(bucket["reproduction"], ignore_index=True)
        publication_table(level).to_csv(out / "level_effects_bootstrap.csv", index=False)
        publication_table(deciles).to_csv(out / "error_deciles.csv", index=False)
        publication_table(tail).to_csv(out / "largest_error_decile.csv", index=False)
        publication_table(appendix).to_csv(out / "appendix_node_improvement.csv", index=False)
        publication_table(reproduction).to_csv(out / "reproduction_check.csv", index=False)
        (out / "validation.json").write_text(json.dumps(bucket["validation"], indent=2, default=str), encoding="utf-8")
        _heatmap(deciles, out, dataset)
        _write_narratives(dataset, level, tail, out)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_root / "analysis_manifest.csv", index=False)
    manifest_path.write_text(json.dumps(signature, indent=2))
    return manifest


if __name__ == "__main__":
    print(run_all().to_string(index=False))
