#!/usr/bin/env python3
"""Execute result notebooks in their own folders with the invoking Python kernel."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

ROOT = RESULTS_DIR

def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]] or [
        ROOT / name / ('error_redistribution_analysis.ipynb' if name == 'error_analysis' else f'{name}.ipynb')
        for name in ['ausgrid2013_results', 'gef2017_results', 'full_gef2017_results', 'error_analysis']]
    for path in paths:
        print(f'Executing {path}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, kernel_name='python3', timeout=1800, resources={'metadata': {'path': str(path.parent)}})
        km = client.create_kernel_manager()
        km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f'Completed {path.name}', flush=True)

if __name__ == '__main__':
    main()
