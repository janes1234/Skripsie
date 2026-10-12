# results/

## Current round (`CNN/pipeline.py`)

Clarifiers are trained on the relabelled data with the supervisor's review applied. Aerobic
zones are trained on the original labels (never relabelled). Both components use letterbox
padding (not stretching) and the chronological split (train before 2020-01, test after) in
place of the old random 70/20/10 stratified split -- see `CNN/split_dataset_chronological.py`.

| Folder | What |
|---|---|
| `clarifier_aug/` | Clarifiers, tuned hyperparameters, with augmentation: LOFO (5 facilities) + chronological |
| `clarifier_noaug/` | Clarifiers, same hyperparameters, no train-time augmentation: chronological only |
| `aerobic_zone_aug/` | Aerobic zones, hyperparameters tuned for aerobic zones (`CNN/tuned_hyperparams_aerobic_zone.json`), with augmentation: LOFO (3 facilities) + chronological |
| `aerobic_zone_noaug/` | Aerobic zones, same hyperparameters, no augmentation: chronological only |

With vs without augmentation is compared on the chronological split only; LOFO is run once, with
augmentation.

Each folder has `results_<component>_<model>_<test set>.pkl`, where the test set is the held-out
facility for leave-one-facility-out runs or `chronological` for the chronological split. It also
has `figures/`, `logs/` and `summary_*.csv` (written by the results notebooks). Open them with
`CNN/results_clarifier.ipynb` and `CNN/results_aerobic_zone.ipynb`.

## Archived results

Everything superseded by this round -- the random-stratified-split era, the separate padded-only
aerobic experiment, AlexNet, and pre-convention loose files -- has been moved into `archive/` (see
`archive/README.md`), not deleted, so earlier numbers stay recoverable if needed.
