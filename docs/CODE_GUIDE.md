# Model and analysis guide

The main entry point is `python -m tree_dynamics`. Model tasks are `predict`, `predict-tif` and `evaluate-masks`; integrity checking is `verify-files`. Run `--help` for arguments. The model architecture and numerical preprocessing are retained with their released checkpoint.

Training and augmentation source from the earlier public v2 archive is in `reference/historical_model_v2/`. Its original Python files are retained; the published training examples have compatibility issues described in `HISTORICAL_CODE_NOTES.md`. They do not identify the exact historical training membership or establish a new retraining workflow.

Photometric transforms are in `tree_dynamics/photometric.py`, with scenarios in `config/photometric_scenarios.json`. They operate on supplied arrays; the complete imagery-based sensitivity experiment also requires source inputs and evaluation windows.

Optional statistics, climate fitting and plot reconstruction are documented in `../analysis/README.txt`. Their scope is listed in `../analysis/analysis_scope.csv`. They are independent of downloading the Zenodo sample collection and do not need to be run before model prediction.

GitHub hosts code and the separate weight release asset. Zenodo hosts the 956-pair sample dataset. Publication source tables and derived numerical inputs are separate from that sample collection.
