"""Publication-only labels; raw experiment keys and metric formulas stay stable."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
import re

_METRICS = re.compile(r'(?<![A-Za-z0-9])f(MAE|MASE|RMSSE)(?![A-Za-z0-9])')
_RECONCILIATIONS = {'bu': 'fBU', 'mint_cov': 'fMinT-cov',
                   'mint_ols': 'fMinT-OLS', 'mint_shrinkage': 'fMinTs',
                   'mint_var': 'fWLSv', 'wls_structure': 'fWLS-struct'}


def publication_text(text):
    return _METRICS.sub(lambda match: match.group(1), text.replace('CHT', 'CH'))


def publication_table(frame):
    frame = frame.rename(columns=publication_text).copy()
    for column in ('method', 'reconciliation'):
        if column in frame:
            frame[column] = frame[column].replace(_RECONCILIATIONS)
    return frame
