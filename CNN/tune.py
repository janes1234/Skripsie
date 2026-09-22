"""
tune.py
=======
Retuning sweep for the WWTW CNN classifiers, using Optuna.

Jointly searches (hidden_layers, neurons, lr, beta1, step_size, batch_size)
per architecture -- one Optuna study per architecture (TPE sampler +
MedianPruner), optimizing validation accuracy at the epoch with the lowest
validation loss (matching how every train_*.ipynb already picks its "best
checkpoint" -- see train_model() in wwtw_utils.py).

Tuned on a single held-out-facility split (NoordelikeWerke held out for
test; train/val drawn from the other four facilities), then the winning
hyperparameters get applied uniformly across all 5 leave-one-facility-out
folds when the architecture is actually trained via run.py -- the same
convention the original (now-unreproducible) tuning sweep used.

AlexNet is deliberately excluded: it's trained from scratch (no ImageNet
pretraining) and is being kept exactly as-is.

Usage:
    # Tune everything (cheapest architectures first), appending results to
    # tuned_hyperparams.json as each architecture finishes:
    python tune.py --all

    # Tune just one architecture:
    python tune.py --arch resnet18

    # Fewer/more trials per architecture (default: 20):
    python tune.py --all --n-trials 15

Progress -- every trial's sampled hyperparameters, every epoch's train/val
loss+acc (via wwtw_utils's own per-epoch prints), and each architecture's
winning params -- goes to stdout. Redirect it to a file to track progress
from outside this process, e.g.:

    python tune.py --all > tune_progress.log 2>&1 &
    tail -f tune_progress.log
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Must be set before importing wwtw_utils -- it reads these at import time.
os.environ.setdefault("DATASET_ROOT", "../cnn_dataset_split_facility_test-NoordelikeWerke")
os.environ.setdefault("COMPONENT", "clarifier")

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from wwtw_utils import (
    device, num_classes, ADAM_BETA2, WEIGHT_DECAY, EPOCHS, EARLY_STOP_PATIENCE,
    MICRO_BATCH_CAP, IMG_TARGET_SIZE, get_dataloaders, make_class_weights,
    build_classifier_head, train_model,
)

RESULTS_JSON = Path("tuned_hyperparams.json")

# Cheapest/fastest architectures first, so partial progress (and the log)
# is useful even if this gets interrupted partway through.
ARCHITECTURES = [
    "resnet18", "efficientnet_b0", "densenet121",
    "resnet50", "convnext_tiny", "inception_v3",
]

IMG_SIZE = {arch: ((299, 299) if arch == "inception_v3" else IMG_TARGET_SIZE) for arch in ARCHITECTURES}

# Search space shared by every architecture. img_size is NOT searched --
# it's an architectural constraint (Inception v3 needs 299x299), fixed per
# architecture exactly as it is in ARCH_HYPERPARAMS.
HIDDEN_LAYERS_CHOICES = [0, 1, 2, 3]
NEURONS_CHOICES = [512, 1024, 2048]
BATCH_SIZE_CHOICES = [8, 16, 24, 32, 48, 64, 96, 128]
LR_BOUNDS = (1e-6, 1e-2)          # log-uniform
BETA1_BOUNDS = (0.5, 0.999)       # beta2 is fixed at ADAM_BETA2 (0.999)
STEP_SIZE_BOUNDS = (5, 40)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_model(arch: str, hidden_layers: int, neurons: int):
    """Mirrors the per-architecture model-definition block in each
    train_<arch>.ipynb, parameterized by the (hidden_layers, neurons) being
    tried this trial. Returns (model_on_device, is_inception)."""
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        is_inception = False
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        is_inception = False
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        is_inception = False
    elif arch == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        in_f = m.fc.in_features
        m.fc = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        # Aux classifier always stays a single Linear, same as every
        # train_inception_v3.ipynb run regardless of tuned head shape.
        in_f_aux = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(in_f_aux, num_classes)
        is_inception = True
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        is_inception = False
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.avgpool = nn.AdaptiveMaxPool2d(1)  # kept as-is, per current design
        in_f = m.classifier[2].in_features
        m.classifier[2] = build_classifier_head(in_f, hidden_layers, neurons, num_classes)
        is_inception = False
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m.to(device), is_inception


def make_objective(arch: str):
    img_size = IMG_SIZE[arch]

    def objective(trial: optuna.Trial) -> float:
        hidden_layers = trial.suggest_categorical("hidden_layers", HIDDEN_LAYERS_CHOICES)
        neurons = trial.suggest_categorical("neurons", NEURONS_CHOICES)
        lr = trial.suggest_float("lr", *LR_BOUNDS, log=True)
        beta1 = trial.suggest_float("beta1", *BETA1_BOUNDS)
        step_size = trial.suggest_int("step_size", *STEP_SIZE_BOUNDS)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)

        micro_batch = min(batch_size, MICRO_BATCH_CAP)
        accum_steps = max(1, round(batch_size / micro_batch))

        log(f"[{arch}] trial {trial.number}: H={hidden_layers} N={neurons} lr={lr:.2e} "
            f"beta1={beta1:.3f} step_size={step_size} batch_size={batch_size} "
            f"(micro_batch={micro_batch}, accum_steps={accum_steps})")

        train_ds_arch, val_ds_arch, _, train_loader_arch, val_loader_arch, _ = \
            get_dataloaders(img_size, micro_batch)
        class_weights_arch = make_class_weights(train_ds_arch)

        model, is_inception = build_model(arch, hidden_layers, neurons)
        criterion = nn.CrossEntropyLoss(weight=class_weights_arch)
        optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=(beta1, ADAM_BETA2), weight_decay=WEIGHT_DECAY,
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

        best = {"acc": 0.0, "loss": float("inf")}

        def epoch_cb(epoch, val_loss, val_acc):
            if val_loss < best["loss"]:
                best["loss"] = val_loss
                best["acc"] = val_acc
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            train_model(
                model, train_loader_arch, val_loader_arch, optimizer, scheduler,
                criterion, EPOCHS, EARLY_STOP_PATIENCE, f"{arch} trial{trial.number}",
                is_inception=is_inception, accum_steps=accum_steps,
                epoch_callback=epoch_cb,
            )
        finally:
            del model, optimizer, scheduler, criterion
            del train_loader_arch, val_loader_arch, train_ds_arch, val_ds_arch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log(f"[{arch}] trial {trial.number} finished: best val_acc={best['acc']:.4f} "
            f"(at val_loss={best['loss']:.4f})")
        return best["acc"]

    return objective


def save_result_json(arch: str, result: dict):
    data = {}
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
    data[arch] = result
    RESULTS_JSON.write_text(json.dumps(data, indent=2))
    log(f"Saved tuned hyperparameters for {arch} to {RESULTS_JSON}")


STORAGE = "sqlite:///optuna_tuning.db"


def tune_architecture(arch: str, n_trials: int):
    # Persistent (SQLite) storage + load_if_exists means a crashed process
    # (e.g. the GPU driver's RC watchdog killing a kernel -- see module
    # docstring note below) can be relaunched and pick up exactly where it
    # left off instead of losing all trial progress for the architecture it
    # was on. See run_tuning_with_retries.sh for the auto-restart loop that
    # relies on this.
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=f"{arch}_tune", storage=STORAGE, load_if_exists=True,
    )

    # A trial that was RUNNING when a prior process died is stuck in that
    # state forever otherwise (Optuna never sees it fail) -- mark those as
    # failed so they're excluded from best-trial selection and don't block
    # anything.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            log(f"[{arch}] marking stale RUNNING trial {t.number} (from a prior crashed process) as failed")
            study.add_trial(
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.FAIL, params=t.params,
                    distributions=t.distributions, value=None,
                )
            )

    already_run = len(study.trials)
    remaining = max(0, n_trials - already_run)
    log(f"=== {arch}: {already_run}/{n_trials} trials already recorded, {remaining} remaining, "
        f"img_size={IMG_SIZE[arch]} ===")
    if remaining > 0:
        study.optimize(make_objective(arch), n_trials=remaining)

    best = study.best_trial
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    log(f"=== {arch} done: {n_complete} completed, {n_pruned} pruned. "
        f"Best trial #{best.number}: val_acc={best.value:.4f} params={best.params} ===")

    result = dict(best.params)
    result["val_acc"] = best.value
    result["img_size"] = list(IMG_SIZE[arch])
    save_result_json(arch, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=ARCHITECTURES, default=None,
                         help="Tune just this architecture")
    parser.add_argument("--all", action="store_true",
                         help="Tune all 6 architectures (everything except alexnet)")
    parser.add_argument("--n-trials", type=int, default=20,
                         help="Optuna trials per architecture (default: 20)")
    args = parser.parse_args()

    if not args.arch and not args.all:
        parser.error("Pass --arch <name> or --all")

    archs = ARCHITECTURES if args.all else [args.arch]

    log(f"Dataset root: {os.environ['DATASET_ROOT']}  Component: {os.environ['COMPONENT']}")
    log(f"Architectures to tune ({len(archs)}): {archs}")
    log(f"Trials per architecture: {args.n_trials}")

    existing = {}
    if RESULTS_JSON.exists():
        existing = json.loads(RESULTS_JSON.read_text())

    t0 = time.time()
    for arch in archs:
        if arch in existing:
            log(f"[{arch}] already has a saved result in {RESULTS_JSON} "
                f"(val_acc={existing[arch]['val_acc']:.4f}) -- skipping.")
            continue
        arch_t0 = time.time()
        tune_architecture(arch, args.n_trials)
        log(f"[{arch}] finished in {(time.time() - arch_t0) / 60:.1f} min")
    log(f"All requested architectures tuned in {(time.time() - t0) / 60:.1f} min total.")


if __name__ == "__main__":
    main()
