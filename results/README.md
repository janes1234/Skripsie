# results/

## Current round (from 2026-09-28, `CNN/pipeline.py`)

Clarifiers were trained on the relabelled data with the supervisor's review applied
(`relabel_review/supervisor_decisions_applied.csv`). Aerobic zones were trained on the original labels.

| Folder | What |
|---|---|
| `clarifier_aug/` | Clarifiers, tuned hyperparameters, with augmentation: LOFO (5 facilities) + stratified 70/20/10 |
| `clarifier_noaug/` | Clarifiers, same hyperparameters, no train-time augmentation: stratified 70/20/10 only |
| `aerobic_zone_aug/` | Aerobic zones, hyperparameters tuned for aerobic zones (`CNN/tuned_hyperparams_aerobic_zone.json`), with augmentation: LOFO (3 facilities) + stratified |
| `aerobic_zone_noaug/` | Aerobic zones, same hyperparameters, no augmentation: stratified only |

With vs without augmentation is compared on the stratified split only; LOFO is run once, with augmentation.

Each folder has `results_<component>_<model>_<test set>.pkl`, where the test set is the held-out
facility for leave-one-facility-out runs or `stratified` for the 70/20/10 split. It also has
`figures/`, `logs/` and `summary_*.csv` (written by the results notebooks). Open them with
`CNN/results_clarifier.ipynb` and `CNN/results_aerobic_zone.ipynb`.

## Older results

The `*.pkl`, `figures/` and `logs/` directly in this folder are the previous clarifier round,
from before the supervisor's review. That round had AlexNet retrained on 2026-09-28 and the other
six architectures trained on 2026-09-24. `CNN/results_comparison.ipynb` reads these. Earlier
phases are in `archive/` (see `archive/README.md`).
