# Installation and demonstration

Run commands from the repository root. Python 3.10+ is supported by the source syntax; the recorded executions use Python 3.12.7 on Windows with existing environments. A fresh installation on other operating systems has not been tested.

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then install the model dependencies and optionally GeoTIFF support:

```console
python -m pip install -r requirements-model.txt
python -m pip install -r requirements-geotiff.txt
python -B -m tree_dynamics verify-files
```

Download the weight from GitHub Releases into `models/`, then:

```console
python -B -m tree_dynamics verify-files --with-model
python scripts/check_core.py
python scripts/demo_inference.py --output outputs/demo
python -B -m tree_dynamics predict --input outputs/demo/synthetic_four_band_256.npy --output outputs/predict
```

Expected demo outputs: synthetic four-band NumPy inputs and masks at 256 and 512 pixels, plus `demo_report.json` with `passed: true`. The masks are uint8 with values 0 and 255. CPU inference uses four threads. The demo checks software interfaces rather than ecological accuracy. Installing dependencies depends on the environment and package source; no installation-time benchmark is supplied.

`verify-files` checks code. Add `--with-model` to check the weight and `--with-data` to check optional manuscript-derived inputs. Missing optional data do not block model commands. `--code-only` remains an explicit synonym for code-only verification.

GeoTIFF prediction and mask evaluation commands are in the main README. Run `python scripts/check_model_io.py` for channel mapping, grid-preservation and mask-evaluation contract checks after installing model/GeoTIFF dependencies and the weight. This uses synthetic arrays and temporary files, not published accuracy estimates.

The preserved training source has separate requirements and compatibility notes. There is no new training run in these installation checks. Exact environment versions and check scope are recorded in config/release_test_environment.json.
