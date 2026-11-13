"""Regression checks for the paper's covariance notebook output routing."""

# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
import pandas as pd
from PPHF_modules.dp_fhf.notebook_runtime import run_covariance_notebook

class CovariancePaperDefaultsTests(unittest.TestCase):
    def exercise(self, corrected=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();art=root/'fhf_dedup_gef2017';art.mkdir()
            for name in ['fh1_FHF_forecasts.csv','fh1_FHF_training_artifacts.npz','fh1_FHF_per_node_metrics.csv']:(art/name).touch()
            historical=root/'DP_fhf/outputs/02_dp_covariance/GEF2017';historical.mkdir(parents=True)
            marker=historical/'historical_results.txt';marker.write_text('preserve original corrected results')
            commands=[]
            def execute(command,**kwargs):
                commands.append(command)
                output=Path(command[command.index('--output-dir')+1]);label=command[command.index('--output-label')+1]
                pd.DataFrame([{'method':'mint_shrinkage','MASE':.17,'RMSSE':.18},{'method':'mint_var','MASE':.18,'RMSSE':.19}]).to_csv(output/f'{label}_dp_covariance_metrics.csv',index=False)
            with patch('PPHF_modules.dp_fhf.notebook_runtime.locate_fhf_artifact_dir',return_value=art),patch('PPHF_modules.dp_fhf.notebook_runtime.subprocess.run',side_effect=execute):
                kw={'debias_noise_variance':True} if corrected else {}
                summary=run_covariance_notebook(root,datasets=('GEF2017',),**kw)
            self.assertEqual(len(commands),1);self.assertEqual(len(summary),2)
            c=commands[0];self.assertEqual('--no-debias' in c,not corrected)
            output=Path(c[c.index('--output-dir')+1])
            self.assertEqual(output,historical if corrected else historical/'no_debias')
            self.assertTrue((output/'dp_covariance_MASE_RMSSE_summary.csv').is_file())
            self.assertEqual(marker.read_text(),'preserve original corrected results')
            self.assertEqual(c[c.index('--lower')+1],'-0.0005');self.assertEqual(c[c.index('--seed')+1],'42')
    def test_default_uses_current_paper_and_preserves_history(self):self.exercise()
    def test_corrected_ablation_requires_explicit_selection(self):self.exercise(corrected=True)

if __name__=='__main__':unittest.main()
