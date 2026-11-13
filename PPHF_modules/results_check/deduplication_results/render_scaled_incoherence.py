#!/usr/bin/env python3
"""Regenerate both scale-free incoherence figures from the audited CSVs."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE = RESULTS_DIR / "deduplication_results"
PANELS = [('gef2017', 1), ('gef2017', 3), ('gef2017', 6), ('ausgrid2013', 1)]
COLOURS = {'FHF': '#247f9e', 'BO-FHF': '#d8812c', 'CH-FHF': '#35937f'}
FIGURES = [
    ('average_scale_free_incoherence_ratio_pct', 'forecast_scaled_incoherence',
     'Forecast-scaled incoherence', r'$100\,\mathrm{mean}_{p,t}\!\left[\frac{2|P-C|}{|P|+|C|}\right]$'),
    ('average_error_scaled_incoherence_ratio_pct', 'error_scaled_incoherence',
     'Error-scaled incoherence', r'$100\,\mathrm{mean}_{p,t}\!\left[\frac{2|P-C|}{|P-Y|+|C-Y|}\right]$'),
]


def main():
    tables = {d: pd.read_csv(HERE / d / 'deduplicated_average_scale_free_incoherence_by_method.csv')
              for d in {d for d, _ in PANELS}}
    output = HERE / 'scaled_incoherence_figures'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False})
    for metric, stem, title, formula in FIGURES:
        # Shared method rows, but dataset-specific ranges; no nonlinear rescaling.
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 7.6), sharey=True)
        fig.subplots_adjust(left=.195, right=.985, bottom=.145, top=.81, wspace=.16)
        fig.text(.025, .965, title, fontsize=16, weight='bold', va='top')
        fig.text(.025, .9, formula, fontsize=14, va='center')
        fig.text(.38, .9, 'P: parent forecast   C: child forecast sum   Y: parent actual',
                 color='#555555', fontsize=9, va='center')
        fig.legend(handles=[Patch(color=c, label=l) for l, c in COLOURS.items()],
                   loc='upper right', bbox_to_anchor=(.989, .978), ncol=3, frameon=False,
                   handlelength=1.2, columnspacing=1.3)
        gef_limit = tables['gef2017'][metric].max() * 1.35
        for ax, (dataset, horizon) in zip(axes, PANELS):
            table = tables[dataset].query('horizon == @horizon').reset_index(drop=True)
            y = np.arange(len(table))
            for start in [0, 14]:
                ax.axhspan(start-.5, start+6.5, color='#f2f4f6', zorder=0)
            base = table.method_internal.eq('base').to_numpy()
            colours = [COLOURS.get(f, '#8b969e') for f in table.framework]
            ax.barh(y[base], table.loc[base, metric], height=.56,
                    color=np.array(colours)[base], zorder=3)
            ax.scatter(table.loc[~base, metric], y[~base], color='#8b969e', s=13,
                       zorder=4, clip_on=False)
            for i in np.where(base)[0]:
                ax.annotate(f'{table.loc[i, metric]:.3f}%',
                            (table.loc[i, metric], i), xytext=(4, 0),
                            textcoords='offset points', va='center', fontsize=8.5)
            upper = gef_limit if dataset == 'gef2017' else table[metric].max()*1.4
            ax.set_xlim(0, max(upper, 1e-6))
            ax.set_ylim(len(table)-.4, -.7)
            ax.set_yticks(y, table.method, fontsize=8.5)
            ax.tick_params(axis='y', length=0, pad=7)
            ax.tick_params(axis='x', labelsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='x', color='#dce0e4', linewidth=.7, zorder=1)
            ax.set_title(f'{"GEF2017" if dataset == "gef2017" else "AusGrid2013"}\nh = {horizon}',
                         fontsize=11, weight='bold', pad=10)
            ax.set_xlabel('Mean ratio (%)', labelpad=5)
        fig.text(.195, .055, 'All 22 methods shown. Grey points: reconciled methods and Naive, numerically ≈ 0%.', fontsize=9)
        fig.text(.195, .025, 'Equal weight per parent–time comparison. GEF2017 panels share a scale; AusGrid2013 has its own scale.',
                 fontsize=8.5, color='#555555')
        for suffix in ['pdf', 'png']:
            fig.savefig(output / f'{stem}.{suffix}', dpi=220, facecolor='white')
        plt.close(fig)
        print(f'Saved {stem}', flush=True)


if __name__ == '__main__':
    main()
