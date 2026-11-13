"""Execute all active notebooks in isolated workspaces and report exact coverage.

Training uses a complete small hierarchy and two rounds. Other notebook cells
still execute, including exports, reconciliation, accounting, and plots. Dataset
preparation uses copied original input data. Result notebooks use actual saved
current-project forecasts, metrics and training artifacts.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PPHF_modules.paths import PROJECT_ROOT


def inventory(root):
    return sorted(p for p in root.rglob('*.ipynb') if not any(
        part in {'output', '.ipynb_checkpoints', '.venv', '__pycache__'} for part in p.relative_to(root).parts))


def make_fixture(root):
    import numpy as np
    import pandas as pd
    t = np.arange(192)
    leaves = np.array([5+i+np.sin(t/(3+i))+.01*t for i in range(4)])
    values = np.vstack([leaves.sum(axis=0), leaves[:2].sum(axis=0), leaves[2:].sum(axis=0), leaves])
    paths = ['Total','Total/A','Total/B','Total/A/x','Total/A/y','Total/B/z','Total/B/w']
    table = pd.concat([pd.DataFrame({'partition_id':i,'unique_id':path,
        'ds':pd.date_range('2025-01-01',periods=len(t),freq='30min'),'y':values[i]})
        for i,path in enumerate(paths)])
    for dataset, names in {'GEFCOM2017':['GEF2017_univarivate.csv','GEF2017_170101_dedup.csv'],
                           'AusGrid2013':['AusGrid2013_univariate_dedup.csv']}.items():
        folder=root/'Datasets'/dataset;folder.mkdir(parents=True,exist_ok=True)
        for name in names:table.to_csv(folder/name,index=False)
    return root/'Datasets/GEFCOM2017/GEF2017_univarivate.csv'


def prepare_workspace(source, destination):
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'PPHF_modules',destination/'PPHF_modules',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for folder in ['fhf_dedup_gef2017','fhf_dedup_ausgrid2013','DP_fhf/notebooks',
                   'Results_check/ausgrid2013_results','Results_check/gef2017_results',
                   'Results_check/full_gef2017_results','Results_check/error_analysis']:
        (destination/folder).mkdir(parents=True,exist_ok=True)


def worker(job_path):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    job=json.loads(Path(job_path).read_text())
    source=Path(job['source']);workspace=Path(job['workspace']);relative=Path(job['relative'])
    notebook=nbformat.read(source,as_version=4)
    os.chdir(workspace/relative.parent)
    fixture=workspace/'Datasets/GEFCOM2017/GEF2017_univarivate.csv'
    if job['kind']=='training_smoke':
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            s=cell.source
            if re.search(r'^args = (Args|HFLArgs)\(\)',s,re.M):
                s += f'\nargs.csv_path = {str(fixture)!r}\nargs.num_rounds = 2\nargs.early_stop_enabled = False\nargs.min_rounds = 2\nargs.device = "cpu"\n'
            cell.source=s
    elif job['kind']=='dp_pipeline_smoke':
        fixed_rate_sweep = any('run_ausgrid_sampling_sweep(' in c.source
                               for c in notebook.cells if c.cell_type=='code')
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            s=cell.source
            s=re.sub(r'num_rounds=(?:100|20),','num_rounds=2,',s)
            s=re.sub(r'NUM_ROUNDS = (?:100|20)','NUM_ROUNDS = 2',s)
            if fixed_rate_sweep:
                s=re.sub(r'(?m)^NOISE_MULTIPLIERS = .*','NOISE_MULTIPLIERS = [1.2]',s)
                s=re.sub(r'(?m)^CLIP_NORMS = .*','CLIP_NORMS = [1.0]',s)
            cell.source=s
    shell=InteractiveShell()
    started=time.monotonic();result={'path':str(relative),'kind':job['kind'],'status':'passed','executed_cells':0}
    failure=None
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type!='code':continue
        try:
            with capture_output() as captured:
                execution=shell.run_cell(cell.source,store_history=True)
            cell.execution_count=result['executed_cells']+1
            outputs=[]
            if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for rich in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
            cell.outputs=outputs
            result['executed_cells']+=1
            if not execution.success:
                failure=execution.error_before_exec or execution.error_in_exec
                raise failure
        except BaseException as exc:
            result.update(status='failed',cell=i,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            if isinstance(exc,FileNotFoundError):result['status']='missing_input'
            if isinstance(exc,ModuleNotFoundError):result['status']='missing_dependency'
            break
    result['elapsed_sec']=time.monotonic()-started
    target=Path(job['executed_notebook']);target.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,target)
    Path(job['result']).write_text(json.dumps(result,indent=2))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'output/module_reorganization_20261108/validation')
    parser.add_argument('--only',nargs='*',help='Optional notebook path fragments.')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for name in ['matplotlib','ipython']:(output/'runtime'/name).mkdir(parents=True,exist_ok=True)
    notebooks=inventory(PROJECT_ROOT)
    if args.only:notebooks=[p for p in notebooks if any(term in str(p.relative_to(PROJECT_ROOT)) for term in args.only)]
    static=[];transform=TransformerManager()
    for path in notebooks:
        item={'path':str(path.relative_to(PROJECT_ROOT)),'status':'passed'}
        try:
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
            for i,c in enumerate(nb.cells):
                if c.cell_type=='code':compile(transform.transform_cell(c.source),f'{path}:cell{i}','exec')
        except Exception as exc:item.update(status='failed',error=str(exc))
        static.append(item)
    (output/'static.json').write_text(json.dumps(static,indent=2))
    smoke=output/'smoke_workspace';prepare_workspace(PROJECT_ROOT,smoke);make_fixture(smoke)
    data=output/'data_workspace';prepare_workspace(PROJECT_ROOT,data)
    if any(p.relative_to(PROJECT_ROOT).parts[0]=='Datasets' for p in notebooks):
        for relative in ['Datasets/GEFCOM2017/Gef2017.csv',
                         'Datasets/AusGrid2013/2012-2013 Solar home electricity data v2.csv',
                         'Datasets/AusGrid2013/AusGrid2013_univariate.csv',
                         'Datasets/AusGrid2013/AusGrid_S.csv']:
            target=data/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(PROJECT_ROOT/relative,target)
    full=output/'saved_run_workspace';prepare_workspace(PROJECT_ROOT,full)
    source=PROJECT_ROOT/'DP_fhf/outputs/01_dp_fedavg_100round'
    target=full/'DP_fhf/outputs/01_dp_fedavg_100round';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:target.symlink_to(source,target_is_directory=True)
        except OSError:shutil.copytree(source,target)
    for name in ['fhf_dedup_gef2017','fhf_dedup_ausgrid2013']:
        target=full/name
        if target.is_dir() and not target.is_symlink() and not any(target.iterdir()):target.rmdir()
        if not target.exists():
            try:target.symlink_to(PROJECT_ROOT/name,target_is_directory=True)
            except OSError:shutil.copytree(PROJECT_ROOT/name,target)
    for dataset in ['GEFCOM2017','AusGrid2013']:
        target=full/'Datasets'/dataset/'tags_dedup.pkl'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT_ROOT/'Datasets'/dataset/'tags_dedup.pkl',target)
    # Populate fixture baselines before DP covariance/reconciliation notebooks.
    def order(path):
        r=str(path.relative_to(PROJECT_ROOT));name=path.name
        if r.startswith('fhf_'):return (0,r)
        if name.startswith('01_DPFedAvg_Only_'):return (1,r)
        if '_Sweep' in name and 'Reconciliation' not in name:return (2,r)
        if '_Sampling_' in name:return (3,r)
        if '100Round_Reconciliation' in name:return (4,r)
        if r.startswith('DP_fhf'):return (5,r)
        if r.startswith('Results_check'):return (6,r)
        return (7,r)
    records=[]
    for index,path in enumerate(sorted(notebooks,key=order),1):
        rel=path.relative_to(PROJECT_ROOT)
        kind='training_smoke' if rel.parts[0].startswith('fhf_') else 'dp_pipeline_smoke'
        workspace=smoke
        if rel.parts[0]=='Datasets':kind='data_preparation_full';workspace=data
        elif rel.parts[0]=='Results_check':kind='current_results_full';workspace=full
        elif '100Round_Reconciliation' in path.name:kind='saved_100round_analysis';workspace=full
        elif path.name=='02_DP_Covariance_Multiscale_No_Debias.ipynb':kind='saved_residual_analysis';workspace=full
        stem='__'.join(rel.parts)
        job={'source':str(path),'relative':str(rel),'workspace':str(workspace),'kind':kind,
             'executed_notebook':str(output/'executed'/rel),'result':str(output/'jobs'/f'{stem}.json')}
        job_path=output/'jobs'/f'{stem}.job.json';job_path.parent.mkdir(exist_ok=True)
        job_path.write_text(json.dumps(job,indent=2))
        env={**os.environ,'PPHF_PROJECT_ROOT':str(workspace),'DP_FHF_PROJECT_ROOT':str(workspace),
             'MPLBACKEND':'Agg','MPLCONFIGDIR':str(output/'runtime/matplotlib'),
             'IPYTHONDIR':str(output/'runtime/ipython'),'PYTHONDONTWRITEBYTECODE':'1',
             'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        command=[sys.executable,str(workspace/'PPHF_modules/validation/check_notebooks.py'),'--worker',str(job_path)]
        print(f'[{index}/{len(notebooks)}] {rel} ({kind})',flush=True)
        log=output/'jobs'/f'{stem}.log'
        try:
            with log.open('w') as handle:completed=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=1800)
            if Path(job['result']).exists():item=json.loads(Path(job['result']).read_text())
            else:item={'path':str(rel),'kind':kind,'status':'worker_failed','error':log.read_text()[-3500:]}
        except subprocess.TimeoutExpired:item={'path':str(rel),'kind':kind,'status':'timeout','error':'Worker exceeded 1800 seconds.'}
        records.append(item);(output/'results.json').write_text(json.dumps(records,indent=2))
        print(' ',item['status'],item.get('error','')[:350],flush=True)
    counts={status:sum(r['status']==status for r in records) for status in sorted({r['status'] for r in records})}
    lines=['# Notebook validation', '',f'Active notebooks checked: {len(records)}. Static checks: {sum(x["status"]=="passed" for x in static)}/{len(static)}.',
           '',f'Execution counts: {counts}.','',
           'Training and DP pipeline checks use a seven-node synthetic hierarchy and two training rounds. Dataset-preparation notebooks use copied original input files. The 100-round reconciliation and multiscale covariance notebooks read actual saved runs/residuals. Full 100-round retraining and Windows/CUDA execution are not certified by this run.',
           '', '| Notebook | Mode | Status | Detail |','| --- | --- | --- | --- |']
    for r in records:
        detail=r.get('error','').replace('\n',' ').replace('|','/')[:450]
        lines.append(f'| {r["path"]} | {r["kind"]} | {r["status"]} | {detail} |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Execution summary:',counts,flush=True)
    return 0 if all(r['status']=='passed' for r in records) and all(x['status']=='passed' for x in static) else 1

if __name__=='__main__':raise SystemExit(main())
