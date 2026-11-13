"""Recalculate every saved AusGrid DP run and prepare the three sampling figures.

No models are trained. Accuracy uses saved training targets and semantic bottom
membership. Privacy budgets retain their recorded accounting provenance.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from PPHF_modules.results_check.current_results import training_scales, dataset_config
from PPHF_modules.results_check.deduplication_results.generate_deduplication_results import load_bottom_series

OUT = ROOT / "Results_check/deduplication_results/dp_sampling"
KEYS = ["node_id", "unique_id", "ds", "horizon_index"]
METRICS = ["MAE", "RMSE", "MASE", "RMSSE"]


def read_forecast(path):
    frame = pd.read_csv(path, usecols=KEYS + ["y_true", "base"])
    if frame.duplicated(KEYS).any() or not np.isfinite(frame[["y_true", "base"]]).all().all():
        raise ValueError(f"Invalid forecast rows: {path}")
    return frame.sort_values(KEYS).reset_index(drop=True)


def evaluate(frame, scales, membership, run):
    work = frame.assign(_ae=(frame.base-frame.y_true).abs(), _se=(frame.base-frame.y_true)**2)
    node = work.groupby(["node_id", "unique_id"], as_index=False).agg(
        MAE=("_ae", "mean"), MSE=("_se", "mean"), n_test=("base", "size"))
    node = node.merge(scales, on=["node_id", "unique_id"], validate="one_to_one")
    node["RMSE"] = np.sqrt(node.MSE)
    node["MASE"] = node.MAE/node.mase_scale
    node["RMSSE"] = np.sqrt(node.MSE/node.rmsse_scale)
    node = node.merge(membership, on="unique_id", validate="one_to_one")
    node["run_key"] = run["run_key"]
    rows=[]
    for scope, subset in [("Node average",node), *((f"L{i}",node[node.level.eq(i)]) for i in (1,2,3))]:
        rows.append({**run,"scope":scope,"n_nodes":len(subset),**subset[METRICS].mean().to_dict()})
    return node,rows


def metadata(folder):
    privacy=folder/"fh1_FHF_dp_fedavg_privacy.json"
    relative=folder.relative_to(ROOT).as_posix()
    if privacy.exists():
        p=json.loads(privacy.read_text())
        result=dict(sampling_rate=p.get("sampling_rate",1.0), num_rounds=p["num_rounds"],
                    noise_multiplier=p["noise_multiplier"],clip_norm=p["clip_norm"],
                    epsilon=p["accounting"]["epsilon"],delta=p["accounting"]["delta"],
                    budget_source=str(privacy.relative_to(ROOT)),
                    fixed_research_seed=p.get("seed_is_fixed_for_research",False))
    else:
        recovery=folder.parent/"recovery/recovered_sweep_summary.csv"
        table=pd.read_csv(recovery)
        row=table.loc[table.run_id.eq(folder.name)]
        if len(row)!=1: raise ValueError(f"Missing recovered metadata for {folder}")
        p=row.iloc[0]
        result={k:p[k].item() if hasattr(p[k],"item") else p[k] for k in
                ["sampling_rate","num_rounds","noise_multiplier","clip_norm","epsilon","delta"]}
        result.update(budget_source=str(recovery.relative_to(ROOT)),fixed_research_seed=False)
    result.update(run_key=relative,run_directory=str(folder),dp_enabled=True)
    return result


def representative_series(group):
    """Use a stable UID tie break for the two nodes around an even median."""
    distance=(group.MASE-group.MASE.median()).abs()
    tied=np.isclose(distance,distance.min(),rtol=0,atol=1e-12)
    return sorted(group.loc[tied,"unique_id"].astype(str))[0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bottom=load_bottom_series(dataset_config("ausgrid2013",ROOT))
    baseline_dir=ROOT/"fhf_dedup_ausgrid2013"
    baseline_file=baseline_dir/"fh1_FHF_forecasts.csv"
    artifact=baseline_dir/"fh1_FHF_training_artifacts.npz"
    scales=training_scales(artifact)
    ids=scales.unique_id.astype(str)
    membership=pd.DataFrame({"unique_id":ids,"structural_level":ids.str.count("/")+1})
    membership["level"]=np.where(ids.eq("GC"),1,np.where(ids.isin(bottom),3,2))
    membership["scope"]="L"+membership.level.astype(str)
    counts=membership.groupby("level").size().to_dict()
    if counts!={1:1,2:60,3:299}: raise ValueError(f"Invalid semantic hierarchy: {counts}")
    membership.to_csv(OUT/"hierarchy_membership.csv",index=False)
    reference=read_forecast(baseline_file)
    with np.load(artifact,allow_pickle=False) as z:
        reference_targets=z["train_targets"]
        reference_names=z["unique_ids"].astype(str)
    base=dict(run_key="non_dp_q1_100rounds",run_directory=str(baseline_dir),dp_enabled=False,
              sampling_rate=1.0,num_rounds=100,noise_multiplier=np.nan,clip_norm=np.nan,
              epsilon=np.nan,delta=np.nan,budget_source="non_private",fixed_research_seed=False)
    baseline_node,summary=evaluate(reference,scales,membership,base)
    per_nodes=[baseline_node]
    validation=[]
    examples=[]
    baseline_reps=baseline_node.sort_values("unique_id").groupby("level").apply(
        representative_series,include_groups=False).to_dict()
    test_start=pd.Timestamp(reference.ds.min());test_end=test_start+pd.Timedelta(hours=72)
    selected=set(baseline_reps.values())

    def example(frame,key):
        keep=frame.unique_id.isin(selected) & frame.ds.lt(str(test_end))
        result=frame.loc[keep,["unique_id","ds","y_true","base"]].copy()
        result["run_key"]=key
        return result
    examples.append(example(reference,base["run_key"]))
    folders=sorted({p.parent for p in (ROOT/"DP_fhf/outputs").rglob("fh1_FHF_dp_fedavg_base_forecasts.csv")})
    source_rows=[]
    for position,folder in enumerate(folders,1):
        run=metadata(folder)
        print(f"[{position}/{len(folders)}] q={run['sampling_rate']} rounds={run['num_rounds']} {folder.name}",flush=True)
        forecast=folder/"fh1_FHF_dp_fedavg_base_forecasts.csv"
        saved=folder/"fh1_FHF_dp_fedavg_training_artifacts.npz"
        run_scales=training_scales(saved)
        with np.load(saved,allow_pickle=False) as z:
            if not np.array_equal(z["unique_ids"].astype(str),reference_names):
                raise ValueError(f"Training series order mismatch: {folder}")
            max_target_diff=float(np.max(np.abs(z["train_targets"]-reference_targets)))
            if max_target_diff>1e-10: raise ValueError(f"Different training targets: {folder}: {max_target_diff}")
        frame=read_forecast(forecast)
        if not frame[KEYS].equals(reference[KEYS]): raise ValueError(f"Test key mismatch: {folder}")
        max_test_diff=float(np.max(np.abs(frame.y_true.to_numpy()-reference.y_true.to_numpy())))
        if max_test_diff>1e-10: raise ValueError(f"Test target mismatch: {folder}: {max_test_diff}")
        node,rows=evaluate(frame,run_scales,membership,run)
        per_nodes.append(node);summary.extend(rows)
        validation.append({"run_key":run["run_key"],"n_rows":len(frame),"n_nodes":len(node),
                           "level_counts":counts,"max_train_target_difference":max_target_diff,
                           "max_test_target_difference":max_test_diff})
        if run["num_rounds"]==100:
            if run["noise_multiplier"]!=1.2 or run["clip_norm"]!=1: raise ValueError("100-round comparison parameters differ")
            examples.append(example(frame,run["run_key"]))
        source_rows.extend({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,
                            "mtime_ns":p.stat().st_mtime_ns} for p in (forecast,saved))
    all_scopes=pd.DataFrame(summary)
    all_scopes.to_csv(OUT/"all_saved_runs_by_scope.csv",index=False)
    pd.concat(per_nodes,ignore_index=True).to_csv(OUT/"all_saved_runs_per_series.csv",index=False)
    hundred=all_scopes[all_scopes.num_rounds.eq(100)].copy()
    hundred["condition"]=np.where(hundred.dp_enabled,
        "dp_q"+hundred.sampling_rate.astype(str).str.replace(".","p",regex=False),"non_dp_q1")
    if len(hundred)!=16: raise ValueError(f"Expected four comparable 100-round settings, got {len(hundred)} rows")
    hundred.to_csv(OUT/"sampling_100rounds_by_scope.csv",index=False)
    twenty=all_scopes[all_scopes.num_rounds.eq(20)].copy()
    twenty.to_csv(OUT/"sampling_20round_sweep_by_scope.csv",index=False)
    pd.concat(examples,ignore_index=True).merge(membership[["unique_id","level"]],on="unique_id").to_csv(
        OUT/"forecast_examples_72h.csv",index=False)
    (OUT/"validation.json").write_text(json.dumps({"training_executed":False,
        "source_root":str(ROOT),"dp_runs_evaluated":len(folders),"non_dp_runs_evaluated":1,
        "evaluation_level_counts":counts,"representative_series":baseline_reps,
        "privacy_budget_note":"Recorded privacy files or recovered posthoc accounting; model scope only",
        "runs":validation},indent=2))
    pd.DataFrame(source_rows).to_csv(OUT/"source_files.csv",index=False)
    print(f"Completed {len(folders)} DP runs and non-DP baseline. Output: {OUT}",flush=True)


if __name__=="__main__": main()
