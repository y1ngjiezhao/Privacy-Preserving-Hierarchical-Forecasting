"""Keep generated notebook imports aligned with the consolidated package."""
from __future__ import annotations
from pathlib import Path
import re

BOOTSTRAP = '''# Locate the project package from a notebook folder or the project root.
from pathlib import Path
import os
import sys
_pphf_start = Path.cwd().resolve()
_pphf_override = os.environ.get("PPHF_PROJECT_ROOT") or os.environ.get("DP_FHF_PROJECT_ROOT")
_pphf_candidates = ([Path(_pphf_override).expanduser()] if _pphf_override else []) + [_pphf_start, *_pphf_start.parents]
PROJECT_ROOT = next((p.resolve() for p in _pphf_candidates if (p / "PPHF_modules" / "fhf" / "FHF_runner.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Cannot locate PPHF_modules. Set PPHF_PROJECT_ROOT or start Jupyter inside the project.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
'''


def normalize_notebook(notebook: dict) -> dict:
    package = Path(__file__).resolve().parent
    imports = {p.stem: f'PPHF_modules.{folder}.{p.stem}'
               for folder in ('fhf', 'dp_fhf') for p in (package / folder).glob('*.py')
               if p.stem != '__init__'}
    imports.update({'DP_fhf':'PPHF_modules.dp_fhf',
                    'FHF_HFL_modular':'PPHF_modules.fhf.FHF_HF2'})
    first = True
    for index, cell in enumerate(notebook['cells']):
        cell.setdefault('id', f'pphf-{index}')
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        source = re.sub(r'# --- shared module path setup \(generated\) ---.*?# --- end shared module path setup ---',
                        'project_dir = Path.cwd()\nmodule_dir = PROJECT_ROOT / "PPHF_modules" / "fhf"',source,flags=re.S)
        for name, qualified in imports.items():
            source = re.sub(r'\bfrom '+re.escape(name)+r'(?=\s+import|\.)','from '+qualified,source)
        source = source.replace('module_dir.parent','PROJECT_ROOT')
        source = source.replace('/ "shared_modules"','/ "PPHF_modules" / "fhf"')
        source = source.replace("/ 'shared_modules'","/ 'PPHF_modules' / 'fhf'")
        if first:
            if '# Locate the project package from a notebook folder or the project root.' not in source:
                source = BOOTSTRAP+'\n'+source
            first = False
        future = 'from __future__ import annotations'
        if future in source:
            source = future+'\n'+source.replace(future+'\n','').lstrip('\n')
        cell['source'] = source.splitlines(keepends=True)
    return notebook
