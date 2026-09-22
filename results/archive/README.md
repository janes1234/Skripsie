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
