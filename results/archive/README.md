# results/archive/

Snapshots of `results/*.pkl` taken before each phase of the hyperparameter
retuning work overwrote them, so every phase's numbers stay comparable
after the fact.

- `phase0_before_retune/` — the original results, before any of this work:
  ResNet-18 as a fixed untuned baseline, ResNet-50/DenseNet-121/InceptionNet
  v3 on the old (now-unreproducible) tuning sweep's values, EfficientNet-B0/
  ConvNeXt-Tiny/AlexNet on hand-picked defaults. `ADAM_BETA2` was 0.9 at
  this point (see wwtw_utils.py git history).
- `phase1_tuned/` — after retuning all 6 architectures (everything except
  AlexNet) with tune.py (Optuna), using `TUNED_ARCH_HYPERPARAMS` in
  wwtw_utils.py.
- `phase2_no_tuning/` (once that pass finishes) — the same 6 architectures
  trained with one uniform, un-tuned default recipe
  (`NO_TUNING_ARCH_HYPERPARAMS`), to isolate whether the Optuna search in
  phase1 actually helped.

AlexNet's results are included in every snapshot but are unchanged across
all of them -- it's deliberately excluded from the retuning/ablation work.
- `phase3_tuned_aug_hue0.1/` — snapshot of the tuned results (`ARCH_HYPERPARAMS`
  = `TUNED_ARCH_HYPERPARAMS` + AlexNet defaults) trained with the full
  augmentation pipeline, i.e. `ColorJitter(..., hue=0.1)`, taken before the
  hue=0 retrain overwrote `results/`. Includes `summary.txt`
  (summarize_phase.py output) and `per_run_accuracy_dysfunctional.txt`
  (accuracy + Dysfunctional recall/false positives per run).
- `phase4_tuned_aug_hue0/` — same as phase3 (tuned hyperparameters, full
  augmentation) but with `ColorJitter(..., hue=0.0)`, to test whether hue
  jitter was washing out the green cue of the Dysfunctional class. Same
  `summary.txt` and `per_run_accuracy_dysfunctional.txt` files as phase3.
