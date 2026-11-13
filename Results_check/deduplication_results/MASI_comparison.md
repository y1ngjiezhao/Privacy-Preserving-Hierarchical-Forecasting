# Incoherence comparison

Primary: MASI (multiples). Secondary: forecast-scaled and error-scaled ratios (%).

MASI uses the same training-only m=1 naive scale as the saved experimental MASE: 80% training split, 48 lag rows, mean absolute adjacent differences over the training target matrix. This is not seasonal MASE. Each horizon has its own target window, but the scale is shared across methods.

Existing two ratios and the historical total-demand ratio are retained unchanged. The numerator is the same saved absolute parent/direct-child discrepancy, averaged in time and then equally over parents. All evaluated parents have equal time counts within each horizon. MASI is not a percentage.

| Dataset | Horizon | Method | MASI | Forecast scale (%) | Error scale (%) | Base rank: MASI / forecast / error |
|---|---:|---|---:|---:|---:|---|
| gef2017 | 1 | FedAvg | 0.018870 | 0.077296 | 32.359760 | 1 / 1 / 2 |
| gef2017 | 1 | HierAvgPS | 0.019842 | 0.082651 | 34.654054 | 2 / 2 / 3 |
| gef2017 | 1 | HierAvg | 0.019907 | 0.082740 | 30.082770 | 3 / 3 / 1 |
| gef2017 | 3 | HierAvg | 0.049761 | 0.207627 | 37.575577 | 1 / 1 / 1 |
| gef2017 | 3 | FedAvg | 0.049913 | 0.207754 | 38.602862 | 2 / 2 / 2 |
| gef2017 | 3 | HierAvgPS | 0.052801 | 0.221619 | 41.296348 | 3 / 3 / 3 |
| gef2017 | 6 | FedAvg | 0.075194 | 0.313068 | 38.718723 | 1 / 1 / 2 |
| gef2017 | 6 | HierAvg | 0.077450 | 0.321618 | 38.108288 | 2 / 2 / 1 |
| gef2017 | 6 | HierAvgPS | 0.082574 | 0.344727 | 40.968244 | 3 / 3 / 3 |
| ausgrid2013 | 1 | HierAvgPS | 0.178807 | 4.311912 | 37.099691 | 1 / 2 / 2 |
| ausgrid2013 | 1 | FedAvg | 0.179102 | 4.291164 | 36.912801 | 2 / 1 / 1 |
| ausgrid2013 | 1 | HierAvg | 0.191851 | 4.704446 | 37.810361 | 3 / 3 / 3 |

All-method comparisons and level/parent results are saved per dataset. Rank values treat magnitudes below 1e-10 as numerical zero; original metric values are not rounded in CSVs. Reconciled forecasts are evaluated as their own methods; base forecasts are evaluated without reconciliation.
