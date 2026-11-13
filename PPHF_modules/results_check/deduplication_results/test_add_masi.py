
# Allow direct CLI execution as well as python -m PPHF_modules....
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from PPHF_modules.paths import PROJECT_ROOT, FHF_MODULE_DIR, DP_MODULE_DIR, DP_EXPERIMENT_DIR, RESULTS_DIR
import unittest
import numpy as np
import pandas as pd
from PPHF_modules.results_check.deduplication_results.add_masi import scaled_incoherence, OUT, RAW, FORECAST, ERROR


class TestMASI(unittest.TestCase):
    def test_definition(self):
        self.assertAlmostEqual(scaled_incoherence(60, 3, 25), 0.8)
        self.assertEqual(scaled_incoherence(0, 3, 25), 0)

    def test_scale_invariance(self):
        self.assertEqual(scaled_incoherence(60, 3, 25), scaled_incoherence(600, 3, 250))

    def test_zero_scale_is_not_zero_incoherence(self):
        for scale in (0, 1e-14, np.nan):
            self.assertTrue(np.isnan(scaled_incoherence(20, 1, scale)))
            self.assertTrue(np.isnan(scaled_incoherence(0, 1, scale)))

    def test_saved_results(self):
        for dataset in RAW:
            parent = pd.read_csv(OUT / dataset / 'deduplicated_incoherence_by_parent.csv')
            table = pd.read_csv(OUT / dataset / 'incoherence_three_metric_comparison.csv')
            self.assertTrue(np.isfinite(parent.MASI).all())
            self.assertTrue((parent.MASI >= 0).all())
            for _, row in table.iterrows():
                sub = parent.loc[parent.method.eq(row.method) & parent.horizon.eq(row.horizon)]
                self.assertAlmostEqual(row.MASI, sub.MASI.mean(), places=12)
                for metric in (FORECAST, ERROR):
                    self.assertAlmostEqual(row[metric], np.average(sub[metric], weights=sub.n_time_points), places=10)
                self.assertEqual(row.masi_valid_parents, row.masi_total_parents)
            coherent = table.loc[table.method_internal.isin(['bu', 'mint_shrinkage', 'mint_var'])]
            self.assertTrue((coherent.MASI < 1e-10).all())


if __name__ == '__main__':
    unittest.main()
