Optional manuscript analyses

These commands reproduce derived-input statistics and figures. They are separate from model prediction and from the 956-pair Zenodo sample collection.

Input: Supplementary_source_data.zip accompanying the manuscript materials. Extract the contents of its inner tree_dynamics directory into the repository root to provide data/, data_manifest.json and accompanying data documentation. The original archive also includes the same released weight; its identity is checked if used. This is a compatibility input format, not the current destination for downloading model weights.

Run from the repository root:
python -m pip install -r requirements.txt
python -B -m tree_dynamics verify-files --with-data
python -B -m tree_dynamics all --output outputs/statistics
python analysis/scripts/validate_training_manifest.py
python analysis/scripts/validate_mosaic_inventory.py

Expected numerical checks: 13,440 validation values and 5,068 climate values across 88 cases, with 308 coefficient and 240 spatial-validation rows. Archived object statistics contain 6,221 reference and 6,294 predicted objects. Source definitions and scope are described in the accompanying data dictionary and analysis_scope.csv.

To reconstruct the saved Figure 4 coefficients:
python -m pip install -r requirements-figures.txt
python analysis/scripts/replot_figure4.py --source data/source_data/Fig_S2_all_32_coefficients.json --out outputs/figures

These workflows do not download raw climate products, regenerate regional mosaics or establish the original training split. No extra analysis archive is needed for the prediction and mask-evaluation commands in the main README.
