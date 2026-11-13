# DP-FHF entry points

Modules are in `../PPHF_modules/dp_fhf/`, and notebooks are in `notebooks/`.
This package does not include previous results from `outputs/`; training and analysis create new outputs.
Run a fixed DP-FedAvg experiment or sweep before its corresponding summary and reconciliation.
Covariance and O2MR require forecasts, training residuals, and reconciliation matrices from ordinary FHF runs first.
See `notebooks/README.md` and the root `README.md` for entry points, parameters, and dependency order.
