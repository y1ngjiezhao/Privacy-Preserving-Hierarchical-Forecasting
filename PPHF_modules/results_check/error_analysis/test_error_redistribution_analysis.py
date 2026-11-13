
# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
import json
import unittest
from pathlib import Path

import pandas as pd

from PPHF_modules.results_check.error_analysis.error_redistribution_analysis import EXPERIMENTS, OUT_ROOT, RECONCILIATIONS


HERE = RESULTS_DIR / "error_analysis"


class ErrorAnalysisTests(unittest.TestCase):
    def test_current_project_experiment_grid(self):
        expected = {(dataset, horizon, system)
                    for dataset, horizons in [('ausgrid2013', (1,)), ('gef2017', (1, 3, 6))]
                    for horizon in horizons for system in ('FHF', 'HFF', 'HFFPS')}
        self.assertEqual({(exp.dataset, exp.horizon, exp.approach) for exp in EXPERIMENTS}, expected)
        for exp in EXPERIMENTS:
            self.assertEqual(exp.forecasts.parent, PROJECT_ROOT / f'fhf_dedup_{exp.dataset}')
            self.assertTrue(exp.forecasts.is_file())
            self.assertTrue(exp.training_artifacts.is_file())
            self.assertTrue(exp.bottom_tags.is_file())

    def test_generated_outputs_pass_validation_and_reproduce_metrics(self):
        self.assertEqual(RECONCILIATIONS, {"fMinTs": "mint_shrinkage", "fWLSv": "mint_var"})
        for dataset in {exp.dataset for exp in EXPERIMENTS}:
            out = OUT_ROOT / dataset
            validation = json.loads((out / "validation.json").read_text())
            for run in validation:
                for key in ("missing_values", "nonpositive_mase_scales", "nonpositive_rmsse_scales"):
                    self.assertEqual(run[key], 0, f"{dataset}: {key}")
                self.assertIn("no DP output", run["non_dp_source"])
                self.assertEqual(run["data_scope"], "current_deduplicated_non_dp_runs")
                self.assertEqual(run["n_nodes"], 360 if dataset == "ausgrid2013" else 10)
            check = pd.read_csv(out / "reproduction_check.csv")
            self.assertLessEqual(check[["abs_diff_MASE", "abs_diff_RMSSE"]].to_numpy().max(), 1e-6)
            for name in ("error_redistribution_heatmap.png", "error_redistribution_heatmap.pdf"):
                self.assertGreater((out / name).stat().st_size, 1000)
            appendix = pd.read_csv(out / "appendix_node_improvement.csv")
            self.assertEqual(set(appendix["reconciliation"]), {"fMinTs", "fWLSv"})

    def test_notebook_is_valid_json_and_code_compiles(self):
        notebook = json.loads((HERE / "error_redistribution_analysis.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook-cell", "exec")


if __name__ == "__main__":
    unittest.main()
