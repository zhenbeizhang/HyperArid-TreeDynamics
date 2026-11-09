# Historical source compatibility notes

Source: https://zenodo.org/records/18170005 (v2), archive MD5 `7e7e9eb6625e5ea588961c97f63fade6`. The original Python files remain unchanged. The reference/historical_model_v2 directory is retained for provenance; it is not the supported retraining pipeline for the supplied checkpoint.

| Finding | Source evidence | Consequence and treatment |
|---|---|---|
| Independent-validation return mismatch | `validation.py` unpacks six fields from `evaluate`; `train_utils/train_and_eval.py` returns seven (precision, recall, F1, OA, IoU, Dice, kappa) | The independent validation entry cannot complete as written. Use the current revision evaluation functions for the packaged analyses. |
| Probability/logit interface mismatch | `src/UNET.py` applies sigmoid at output; `train_utils/loss/BCE_TverskyLoss.py` applies another sigmoid and BCEWithLogits | The loss consumes probability values as logits. Do not start a new training run from this archive without reviewing the intended loss. No historical-training effect is inferred from an exported source archive. |
| Example spectral configuration | `train.py` defaults to RGB/no NDVI; `validation.py` and `predict_stride.py` request five channels | These examples do not reconstruct the supplied checkpoint's historical training configuration. |
| Directory role differs from the author's reserved-test account | `train.py` selects checkpoints using a directory named `Test`; the author-designated reserved collection in this study is `predict` | The directory name in published sample code does not establish the role of the original reserved set. Do not use this archive to assert historical sample membership or independence. |
| Installation filename differs | The archived README names `requirements.txt`; the supplied file is `requirement.txt` | Follow the actual filename if inspecting the old environment. Current supported environments are documented separately. |

A strict parameter load of the supplied state dictionary into the old five-channel architecture has been verified. Parameter compatibility alone does not establish identity of training, preprocessing or full-image prediction settings. Current revision commands retain the checkpoint-bound architecture and preprocessing.

The original archive is available at its persistent public record. Fixing its training loss would define a changed training implementation, so this release does not substitute a new loss and claim that it generated the existing checkpoint.
