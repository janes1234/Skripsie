# CNN/archive/

Not live code/data for the current pipeline -- kept for reference, not deleted.

- `pre_pad_retune/` — `optuna_tuning.db` and `tuned_hyperparams*.json` from the tuning round done
  under stretch resize mode and (for aerobic zones) a Northern-Works holdout that validated
  nothing (Northern Works has no aerobic zone images). Superseded by the padding + chronological-
  split retune, which holds Atlantis out for aerobic zone tuning instead. See
  `results/archive/README.md` for the matching trained-model results.
- `executed_notebooks_pre_convention/` — executed notebooks that used to sit directly under
  `CNN/executed_notebooks/` (not in a `clarifier_aug/`-style subfolder), from before that
  results-dir convention existed.
