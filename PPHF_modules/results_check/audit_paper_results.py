"""Verify current paper tables against source results without changing observations.

Usage: python PPHF_modules/results_check/audit_paper_results.py --paper-root /path/to/paper
The report is written only after all table/source checks have been recorded.
"""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paper-root',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,default=PROJECT_ROOT)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(); root=args.source_root.resolve();paper=args.paper_root.resolve()
    out=args.output or root/'Results_check/deduplication_results/paper_current';out.mkdir(parents=True,exist_ok=True)
    dedup=root/'Results_check/deduplication_results';dp=root/'DP_fhf/outputs'
    checks=[];sources={};exports={}
    def fingerprint(p):
        p=Path(p);h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    def record_source(p):
        p=Path(p);sources[str(p)]={'sha256':fingerprint(p),'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
    def read(p):
        record_source(p);return pd.read_csv(p)
    def text(p):
        record_source(p);return re.sub(r'(?m)(?<!\\)%.*','',Path(p).read_text())
    def check(name,shown,expected,digits=3):
        ok=bool(abs(float(shown)-float(expected))<=.5*10**(-digits)+1e-10)
        checks.append(dict(check=name,paper_value=shown,source_value=float(expected),digits=digits,passed=ok))
    def eq(name,ok):checks.append(dict(check=name,passed=bool(ok)))
    def nums(line):
        return [float(re.search(r'-?\d[\d,]*(?:\.\d+)?',c)[0].replace(',','')) for c in line.split('&')]
    def block(t,label):
        return next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',t,re.S) if '\\label{'+label+'}' in b)
    def rows(t,n):
        out=[]
        for l in t.split(r'\\'):
            c=l.split('&')
            if len(c)<n+1:continue
            values=[]
            for cell in c[-n:]:
                m=re.fullmatch(r'\s*(?:\\(?:textbf|underline)\{)?([\d.,]+)\}?\s*(?:\\\\)?\s*',cell)
                if not m:break
                values.append(float(m[1].replace(',','')))
            if len(values)==n:out.append((c[0].strip(),values))
        return out
    summaries={};baselines={}
    for ds,folder,table in [('gef2017','fhf_dedup_gef2017','results_gef_all'),('ausgrid2013','fhf_dedup_ausgrid2013','results_aus_fhf_vs_hf2')]:
        a=read(dedup/ds/'deduplicated_paper_summary.csv');summaries[ds]=a
        actual=rows(text(paper/'tables'/f'{table}.tex'),8)
        expected=[]
        if ds=='ausgrid2013':expected.append(a[a.method_internal.eq('naive')].iloc[0])
        for system in ['FHF','HFF','HFFPS']:
            for h in ([1,3,6] if ds=='gef2017' else [1]):
                for method in ['base','bu','mint_shrinkage','mint_var']:
                    expected.append(a[(a.source_system==system)&(a.horizon==h)&(a.method_internal==method)].iloc[0])
        eq(table+'_rows',len(actual)==len(expected))
        for i,((_,shown),r) in enumerate(zip(actual,expected)):
            cols=[f'{m}_L{l}' for m in ['MASE','RMSSE'] for l in [1,2,3]]+['MASI_L1','MASI_L2']
            for c,v in zip(cols,shown):check(f'{table}:{i}:{c}',v,r[c])
        # Verify accuracy summaries from the original per-node experiment scores.
        for system in ['FHF','HFF','HFFPS']:
            for h in ([1,3,6] if ds=='gef2017' else [1]):
                raw=read(root/folder/f'fh{h}_{system}_per_node_metrics.csv')
                for method,g in raw.groupby('method'):
                    selected=a[(a.source_system==system)&(a.horizon==h)&(a.method_internal==method)]
                    if selected.empty:continue
                    rr=selected.iloc[0]
                    # Overall means are invariant to the evaluation-level regrouping.
                    for metric in ['MASE','RMSSE']:
                        expected_mean=sum(rr[f'{metric}_L{l}']*rr[f'n_nodes_L{l}'] for l in [1,2,3])/sum(rr[f'n_nodes_L{l}'] for l in [1,2,3])
                        eq(f'raw_mean:{ds}:{system}:h{h}:{method}:{metric}',np.isclose(g[metric].mean(),expected_mean,rtol=1e-11,atol=1e-12))
                if system=='FHF' and h==1:baselines['GEF2017' if ds=='gef2017' else 'AusGrid2013']=raw.groupby('method')[['MASE','RMSSE']].mean()
        exports[f'nonprivate_{ds}.csv']=a
    # Historical appendix is no longer included but remains numerically verified.
    ds='gef2017';h=1
    for l in text(paper/'tables/appendix_incoherence.tex').splitlines():
        if '\\multicolumn{7}' in l:
            ds='gef2017' if 'GEF2017' in l else 'ausgrid2013';h=int(re.search(r'h=(\d)',l)[1])
        if not re.match(r'^(FedAvg|HierAvgPS|HierAvg) &',l):continue
        method=l.split('&')[0].strip();r=summaries[ds][(summaries[ds].horizon==h)&(summaries[ds].method==method)].iloc[0]
        cols=[f'{m}_L{lev}' for m in ['incoherence_total','average_scale_free_incoherence_ratio_pct','average_error_scaled_incoherence_ratio_pct'] for lev in [1,2]]
        for col,v in zip(cols,nums(l.split('&',1)[1])):check(f'archived_appendix:{ds}:h{h}:{method}:{col}',v,r[col])
    t=text(paper/'sections/7differential privacy.tex')
    main_data=read(dp/'dp_100round_reconciliation/metrics_summary.csv');exports['dp_fedavg_100round.csv']=main_data
    b=block(t,'tab:dp-fedavg-main');dataset='GEF2017';methods=['base','mints','wlsv','ols'];i=0
    for l in b.splitlines():
        if 'textbf{AusGrid2013}' in l:dataset='AusGrid2013';i=0
        rr=rows(l,4)
        if not rr:continue
        method=methods[i];i+=1;shown=rr[0][1];source_method={'base':'base','mints':'mint_shrinkage','wlsv':'mint_var','ols':'mint_ols'}[method]
        r=main_data[(main_data.dataset==dataset)&(main_data.method==method)].iloc[0]
        expected=list(baselines[dataset].loc[source_method])+[r.MASE,r.RMSSE]
        for j,(v,e) in enumerate(zip(shown,expected)):check(f'dp_main:{dataset}:{method}:{j}',v,e)
    sweep=read(dp/'dp_sweep_reconciliation/metrics_summary.csv');exports['dp_fedavg_sweep.csv']=sweep
    b=block(t,'tab:dp-fedavg-sweep');dataset='GEF2017';clip=None
    from PPHF_modules.dp_fhf.privacy_accounting import gaussian_composition_epsilon
    for l in b.splitlines():
        if 'textbf{AusGrid2013}' in l:dataset='AusGrid2013'
        if len(l.split('&'))!=12 or 'multirow' not in l and not l.lstrip().startswith('&'):continue
        c=l.split('&')
        if 'multirow' in c[0]:clip=float(re.findall(r'\{([\d.]+)\}',c[0])[-1])
        v=nums('&'.join(c[1:]));sigma,eps,sd=v[:3]
        check(f'sweep:{dataset}:{clip}:{sigma}:epsilon',eps,gaussian_composition_epsilon(sigma,20,1e-5)['epsilon'])
        check(f'sweep:{dataset}:{clip}:{sigma}:noise_sd',sd,sigma*clip/(10 if dataset=='GEF2017' else 360),3 if dataset=='GEF2017' else 5)
        records=sweep[(sweep.dataset==dataset)&np.isclose(sweep.clip_norm,clip)&np.isclose(sweep.noise_multiplier,sigma)].set_index('method')
        for j,(metric,method) in enumerate((m,k) for m in ['MASE','RMSSE'] for k in methods):check(f'sweep:{dataset}:{clip}:{sigma}:{metric}:{method}',v[3+j],records.loc[method,metric])
    # Saved summary rows must match the complete per-node metrics, not just rounded tables.
    for folder,summary in [('dp_100round_reconciliation',main_data),('dp_sweep_reconciliation',sweep)]:
        nodes=read(dp/folder/'per_node_metrics.csv');keys=['dataset','run_id','method']
        means=nodes.groupby(keys)[['MASE','RMSSE']].mean().sort_index();saved=summary.set_index(keys)[['MASE','RMSSE']].sort_index()
        eq(folder+':per_node_means',means.shape==saved.shape and np.allclose(means,saved,rtol=1e-11,atol=1e-12))
    fixed=read(dp/'02_dp_covariance/no_debias_comparison/paper_comparison.csv')
    multi=read(dp/'02_dp_covariance/multiscale_no_debias/metrics_summary.csv')
    exports['dp_covariance_fixed.csv']=fixed;exports['dp_covariance_quantiles.csv']=multi[multi.input.eq('non_dp_forecasts')]
    b=block(t,'tab:dp-covariance-results');dataset='GEF2017'
    for l in b.splitlines():
        if 'textbf{AusGrid2013}' in l:dataset='AusGrid2013'
        rr=rows(l,4)
        if not rr:continue
        name,v=rr[0];expected=[]
        for method in ['mint_shrinkage','wlsv']:
            if name in ['Non-DP','OLS','Fixed bound']:
                variant={'Non-DP':'non_private','OLS':'reference','Fixed bound':'no_debias'}[name]
                m='ols' if name=='OLS' else method
                r=fixed[(fixed.dataset==dataset)&(fixed.variant==variant)&(fixed.method==m)].iloc[0]
            else:
                q=float(re.search(r'[\d.]+',name)[0]);r=multi[(multi.dataset==dataset)&multi.input.eq('non_dp_forecasts')&(multi.retention_target_pct==q)&(multi.method==method)].iloc[0]
            expected.extend([r.MASE,r.RMSSE])
        for j,(a,e) in enumerate(zip(v,expected)):check(f'covariance:{dataset}:{name}:{j}',a,e)
    # Validate current retention and coherence metadata.
    eq('quantiles:correct_five_retentions',set(multi.retention_target_pct)=={99.5,75,50,25,.5})
    diag=read(dp/'02_dp_covariance/multiscale_no_debias/matrix_diagnostics.csv')
    eq('quantiles:PS_equals_S',diag.PS_minus_S_max_abs.max()<1e-6)
    floored=diag[(diag.dataset=='AusGrid2013')&diag.input.eq('non_dp_forecasts')&(diag.retention_target_pct==.5)]
    match=re.search(r'(\d+) \(MinTs\) and (\d+) \(WLSv\) of 360 eigenvalues',t)
    eq('quantiles:floor_counts_present',match is not None)
    if match:
        for method,count in zip(['mint_shrinkage','wlsv'],map(int,match.groups())):
            eq('quantiles:floor_count:'+method,count==int(floored[floored.method==method].iloc[0].eigenvalues_below_floor))
    for dataset in ['GEF2017','AusGrid2013']:
        # Summaries and private matrices select the uncorrected source branch.
        p=dp/'02_dp_covariance'/dataset/'no_debias/FHF_non_dp_forecasts_dp_covariance_privacy.json';record_source(p);meta=json.loads(p.read_text())
        eq('fixed:'+dataset+':no_debias',meta['debias_noise_variance'] is False)
    # Fresh fixed-bound notebook results must agree with the published comparison.
    for dataset in ['GEF2017','AusGrid2013']:
        direct=read(dp/'02_dp_covariance'/dataset/'no_debias/FHF_non_dp_forecasts_dp_covariance_metrics.csv')
        direct_means=direct.groupby('method')[['MASE','RMSSE']].mean()
        for internal,display in [('mint_shrinkage','mint_shrinkage'),('mint_var','wlsv')]:
            r=fixed[(fixed.dataset==dataset)&fixed.variant.eq('no_debias')&fixed.method.eq(display)].iloc[0]
            eq('fixed:'+dataset+':raw_metrics:'+internal,np.allclose(direct_means.loc[display],[r.MASE,r.RMSSE],rtol=1e-9,atol=1e-11))
    nodes=read(dp/'02_dp_covariance/multiscale_no_debias/per_node_metrics.csv')
    keys=['dataset','input','retention_target_pct','method']
    eq('quantiles:per_node_means',np.allclose(nodes.groupby(keys)[['MASE','RMSSE']].mean().sort_index(),multi.set_index(keys)[['MASE','RMSSE']].sort_index(),rtol=1e-11,atol=1e-12))
    # Forecast perturbation experiment.
    b=block(t,'tab:private-base-forecast-cwdr-results');dataset='GEF2017'
    forecast={ds:read(dp/'03_dp_o2mr'/ds/'dp_o2mr_MASE_RMSSE_summary.csv') for ds in ['GEF2017','AusGrid2013']}
    for l in b.splitlines():
        if 'multirow' in l and 'AusGrid2013' in l:dataset='AusGrid2013'
        if not l.lstrip().startswith('&') or len(l.split('&'))!=6:continue
        vals=nums(l.split('&',1)[1]);eps=vals[0];expected=[]
        for method in ['mint_shrinkage','mint_var']:
            r=forecast[dataset][forecast[dataset].input_label.eq('non_dp_forecasts')&(forecast[dataset].epsilon==eps)&(forecast[dataset].method==method)].iloc[0]
            expected.extend([r.MASE,r.RMSSE])
        for j,(v,e) in enumerate(zip(vals[1:],expected)):check(f'forecast_dp:{dataset}:{eps}:{j}',v,e)
    exports['dp_forecast_release.csv']=pd.concat([x[x.input_label.eq('non_dp_forecasts')&x.method.isin(['mint_shrinkage','mint_var'])] for x in forecast.values()])
    # Independent RDP accounting matches the current appendix.
    for rounds,sigma,shown in [(100,.3,740.496),(100,1.2,75.109),(20,.3,184.941),(20,.6,63.962),(20,.9,36.204),(20,1.2,25.402)]:
        check(f'RDP:{rounds}:{sigma}',shown,gaussian_composition_epsilon(sigma,rounds,1e-5)['epsilon'])
    # Verify source figure identity and independently computed ranks.
    from pypdf import PdfReader
    for ds in ['gef2017','ausgrid2013']:
        for metric in ['MASE','RMSSE']:
            a=read(dedup/ds/f'all_per-node_{metric.lower()}.csv')
            for h,g in a.groupby('horizon'):
                scores=g.drop(columns=[c for c in ['node','unique_id','series','level','Level','h','horizon','dataset'] if c in g]).select_dtypes('number').dropna()
                ranks=scores.rank(axis=1,method='average').mean()
                src=dedup/ds/(f'MCB_{metric}_h{h}.pdf' if ds=='gef2017' else f'MCB_{metric}.pdf')
                dst=paper/'plots'/(f'MCB_GEF2017_{metric}_h{h}.pdf' if ds=='gef2017' else f'MCB_{metric}.pdf')
                record_source(src);eq(f'figure_identical:{ds}:{metric}:{h}',fingerprint(src)==fingerprint(dst))
                pt=PdfReader(src).pages[0].extract_text().replace('−','-').replace('–','-')
                for m,val in ranks.items():
                    label=m.replace('BO-FHF','LN-FHF');match=re.search(r'(?<![\w-])'+re.escape(label)+r' \(([\d.]+)\)',pt)
                    eq(f'rank_label:{ds}:{metric}:{h}:{label}',match is not None)
                    if match:check(f'rank:{ds}:{metric}:{h}:{label}',float(match[1]),val,1 if ds=='gef2017' else 3)
    for prefix,expected_count in [('results_gef_all:',288),('results_aus_fhf_vs_hf2:',104),('dp_main:',32),('sweep:',240),('covariance:',64),('forecast_dp:',32),('rank:',104),('archived_appendix:',72)]:
        eq('coverage:'+prefix,sum(c['check'].startswith(prefix) for c in checks)==expected_count)
    failures=[c for c in checks if not c['passed']]
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'paper_root':str(paper),'source_root':str(root),'checks':len(checks),'failures':failures,'sources':sources,'scope':'Current main result tables, four DP numeric tables, inactive incoherence appendix, ranks/figures; saved-source and per-node consistency; no model retraining.'}
    pd.DataFrame(checks).to_csv(out/'checks.csv',index=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    if not failures:
        for name,df in exports.items():
            df=df.copy()
            for col in df.select_dtypes('object'):
                df[col]=df[col].str.replace('BO-FHF','LN-FHF',regex=False)
            df.to_csv(out/name,index=False)
    print(json.dumps({'checks':len(checks),'failures':failures,'output':str(out)},indent=2))
    return bool(failures)

if __name__=='__main__':sys.exit(main())
