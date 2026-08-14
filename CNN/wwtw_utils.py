"""
wwtw_utils.py
=============
Shared imports, configuration, dataset loading and helper functions for the
WWTW CNN classifier notebooks.

Every training notebook (train_resnet18.ipynb, train_resnet50.ipynb,
train_densenet121.ipynb, train_inception_v3.ipynb) starts with:

    from wwtw_utils import *

This guarantees every notebook sees exactly the same config, data pipeline
and helper functions, and removes the risk of one notebook's leftover
GPU state bleeding into the next one (each notebook now runs in its own
kernel / process).
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- CONFIG -------------------------

DATASET_ROOT = Path("../cnn_dataset_split")   # adjust if your output folder is elsewhere
COMPONENT = "aerobic_zone"              # "aerobic_zone" or "clarifier"

# (height, width) — aerobic zones are ~2:1 (wide rectangles), clarifiers are
# closer to square since they're circular tank crops. Adjust if your actual
# crop aspect ratios differ noticeably — check a few images if unsure.
IMG_TARGET_SIZE = {
    "aerobic_zone": (112, 224),
    "clarifier": (224, 224),
}[COMPONENT]

BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4     # L2 regularization — helps reduce the train/val gap
EPOCHS = 100
EARLY_STOP_PATIENCE = 10   # stop if val loss hasn't improved for this many epochs
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

MODEL_SAVE_PATH = Path(f"../models/{COMPONENT}_cnn.pt")
RESULTS_DIR = Path("../results")   # where per-model results_*.pkl files are written

# ------------------------------------------------------------------

data_dir = DATASET_ROOT / COMPONENT
assert data_dir.exists(), f"Dataset folder not found: {data_dir}. Run prepare_cnn_dataset.py first."

torch.manual_seed(SEED)


# ------------------------- DEFAULT DATASET / LOADERS -------------------------
# Built once at import time, at the "default" IMG_TARGET_SIZE / BATCH_SIZE
# configured above. This is what the ResNet-18 baseline trains on directly.
# The other architectures build their own loaders via get_dataloaders()
# below (different image size and/or batch size per the tuned hyperparameters).

train_transform = transforms.Compose([
    transforms.Resize(IMG_TARGET_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # Aggressive color jitter to stop it from memorizing water color or sun glare
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMG_TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# Load explicitly from the predefined split folders
train_ds = datasets.ImageFolder(root=data_dir / "train", transform=train_transform)
val_ds = datasets.ImageFolder(root=data_dir / "val", transform=eval_transform)
test_ds = datasets.ImageFolder(root=data_dir / "test", transform=eval_transform)

class_names = train_ds.classes
num_classes = len(class_names)
print(f"Classes found ({num_classes}): {class_names}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


# ------------------------- TRAINING / EVAL HELPERS -------------------------

# Collects the results (trained model + metrics) for whichever architecture(s)
# are trained in *this* notebook/process. Each training notebook now only
# ever trains one architecture, so this dict will hold a single entry — it
# gets pickled to disk at the end of each training notebook (see the last
# cell of each train_*.ipynb) so the results notebook can later load and
# compare all four without needing the GPU or the trained models in memory.
results = {}

# Mixed precision roughly halves activation memory and speeds up training on
# GPU, with negligible accuracy impact — a good default when GPU memory is
# tight (e.g. the 7-8 GB cards common on shared/university machines).
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.float16


def report_gpu_memory(label=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        tag = f" — {label}" if label else ""
        print(f"[GPU memory{tag}] allocated: {alloc:.2f} GB | reserved: {reserved:.2f} GB")


def free_gpu(*names, scope):
    """Deletes the given variable names from `scope` (pass globals()) and
    releases their GPU memory. Needed because optimizers/schedulers hold
    references to GPU tensors (Adam's momentum buffers, in particular) and
    stay alive as notebook globals until explicitly deleted — otherwise each
    architecture's optimizer keeps accumulating in memory across sections."""
    for name in names:
        if name in scope:
            del scope[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report_gpu_memory("after cleanup")


def build_classifier_head(in_features, hidden_layers, neurons, num_classes):
    """FC classifier head: `hidden_layers` ReLU-activated hidden layers of
    `neurons` units each, ending in a Linear(-> num_classes) layer.

    This mirrors the classifier design swept in the architecture grid search
    (H = number of hidden layers, N = neurons per layer) — see
    ARCH_HYPERPARAMS below. hidden_layers=0 just gives a single
    Linear(in_features, num_classes) layer, which is what the ResNet-18
    baseline uses.
    """
    layers = []
    in_f = in_features
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_f, neurons))
        layers.append(nn.ReLU(inplace=True))
        in_f = neurons
    layers.append(nn.Linear(in_f, num_classes))
    return nn.Sequential(*layers)


def get_dataloaders(img_size, batch_size):
    """Builds train/val/test ImageFolder datasets + loaders for a given
    image size and batch size. Separate from the module-level train_loader
    /val_loader/test_loader above because each architecture uses its own
    image size (InceptionNet v3 needs 299x299) and its own tuned batch size."""
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    tr_ds = datasets.ImageFolder(root=data_dir / "train", transform=train_tf)
    va_ds = datasets.ImageFolder(root=data_dir / "val", transform=eval_tf)
    te_ds = datasets.ImageFolder(root=data_dir / "test", transform=eval_tf)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    return tr_ds, va_ds, te_ds, tr_loader, va_loader, te_loader


def make_class_weights(train_ds_):
    """Inverse-frequency class weights, normalized to sum to num_classes."""
    counts = np.bincount([label for _, label in train_ds_.samples], minlength=num_classes)
    w = 1.0 / torch.tensor(counts, dtype=torch.float32)
    w = w / w.sum() * num_classes
    return w.to(device)


def run_epoch(model, loader, optimizer, criterion, train_mode, is_inception=False,
              accum_steps=1, scaler=None):
    """accum_steps > 1 implements gradient accumulation: the loader uses a
    small "micro batch" that comfortably fits in GPU memory, but gradients
    are accumulated over `accum_steps` micro-batches before each optimizer
    step — so the effective batch size (and the resulting gradient
    statistics) matches the larger tuned batch size without ever holding
    that many samples' activations in memory at once."""
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    if train_mode:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train_mode):
        step = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                if is_inception and train_mode:
                    # Inception's aux classifier gets its own loss term, weighted
                    # down (0.4x), which is the standard way to train it — it
                    # acts as an auxiliary gradient signal earlier in the network.
                    y_hat, aux_hat = model(images)
                    loss = criterion(y_hat, labels) + 0.4 * criterion(aux_hat, labels)
                else:
                    y_hat = model(images)
                    loss = criterion(y_hat, labels)

            if train_mode:
                scaled_loss = loss / accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (step + 1) % accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * images.size(0)
            correct += (y_hat.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            step += 1

        # Flush any leftover accumulated gradients from a final partial group
        if train_mode and total > 0 and step % accum_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                 epochs, patience, model_name, is_inception=False, accum_steps=1):
    """Training loop with early stopping on validation loss: track val loss
    each epoch, keep the best-performing weights in memory, stop once val
    loss hasn't improved for `patience` epochs, then restore the best
    checkpoint."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    if accum_steps > 1:
        print(f"[{model_name}] Using gradient accumulation: {accum_steps} micro-batches "
              f"per optimizer step (effective batch size ≈ micro_batch x {accum_steps}).")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, True,
                                           is_inception, accum_steps, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, False,
                                       is_inception, accum_steps, scaler)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = " *" if improved else ""
        print(f"[{model_name}] Epoch {epoch:3d}/{epochs} | LR: {current_lr:.2e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}{flag}")

        if epochs_no_improve >= patience:
            print(f"\n[{model_name}] No val loss improvement for {patience} epochs — stopping early at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"\n[{model_name}] Restored best checkpoint (val loss = {best_val_loss:.4f})")
    return history


def plot_training_curves(history, model_name):
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].axvline(best_epoch - 1, color="gray", linestyle="--", linewidth=1, label="restored checkpoint")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].axvline(best_epoch - 1, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def collect_predictions(model, loader):
    """Runs the test set through the model once, returning predicted
    labels, true labels and full class-probability vectors (the last one
    is needed for the ROC/AUC curves)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            y_hat = model(images)
            probs = torch.softmax(y_hat, dim=1).cpu().numpy()
            all_preds.extend(probs.argmax(axis=1))
            all_labels.extend(labels.numpy())
            all_probs.append(probs)
    return np.array(all_preds), np.array(all_labels), np.concatenate(all_probs, axis=0)


def plot_confusion_and_report(all_labels, all_preds, class_names, model_name):
    test_acc = np.mean(all_preds == all_labels)
    print(f"[{model_name}] Test accuracy: {test_acc:.3f}")

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30)
    plt.title(f"Confusion matrix — {model_name} ({COMPONENT} test set)")
    plt.tight_layout()
    plt.show()

    print(classification_report(
        all_labels,
        all_preds,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=3,
        zero_division=0
    ))
    return test_acc


def plot_roc_auc(all_labels, all_probs, class_names, num_classes, model_name):
    y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    plt.figure(figsize=(7, 6))
    roc_aucs = {}
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        roc_aucs[name] = auc
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.500)")
    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    plt.title(f"ROC curves — {model_name} ({COMPONENT} test set)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("AUC (area under ROC curve) per class:")
    for name, auc in roc_aucs.items():
        print(f"  {name:15s}: {auc:.3f}")
    print(f"\nMean AUC: {np.mean(list(roc_aucs.values())):.3f}")
    return roc_aucs


def evaluate_and_record(model, test_loader, model_name, history, img_size, hidden_layers, neurons):
    """Runs the full evaluation block (confusion matrix, report, ROC/AUC)
    for one trained model and stores everything in `results` for the
    save/pickle cell at the end of the training notebook."""
    preds, labels_, probs = collect_predictions(model, test_loader)
    test_acc = plot_confusion_and_report(labels_, preds, class_names, model_name)
    roc_aucs = plot_roc_auc(labels_, probs, class_names, num_classes, model_name)

    # Move the finished model to CPU before storing it. Keeping the trained
    # model resident on the GPU is what causes later cells to run out of
    # memory — we only need the weights (for saving), not for the model to
    # stay on GPU. It can be moved back with .to(device) for any further
    # inference.
    model_cpu = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results[model_name] = dict(
        model=model_cpu, history=history, test_acc=test_acc,
        mean_auc=float(np.mean(list(roc_aucs.values()))),
        img_size=img_size, hidden_layers=hidden_layers, neurons=neurons,
    )
    return test_acc, roc_aucs


def save_result(model_key, save_name):
    """Pickles the metrics/history for `model_key` (from `results`) to
    RESULTS_DIR/results_<save_name>.pkl, and separately saves the model
    weights via torch.save to MODEL_SAVE_PATH.parent. Call this as the last
    cell of each training notebook.

    This is the "crucial step" that lets the results notebook later
    reconstruct training curves, test accuracy and ROC/AUC for every
    architecture without needing the GPU or any of the trained models
    loaded in memory at once.
    """
    r = results[model_key]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pickle_payload = {
        "model_name": model_key,
        "history": r["history"],
        "test_acc": r["test_acc"],
        "mean_auc": r["mean_auc"],
        "img_size": r["img_size"],
        "hidden_layers": r["hidden_layers"],
        "neurons": r["neurons"],
        "class_names": class_names,
        "component": COMPONENT,
    }
    import pickle
    pkl_path = RESULTS_DIR / f"results_{COMPONENT}_{save_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pickle_payload, f)
    print(f"Saved results to {pkl_path}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_PATH.parent / f"{COMPONENT}_{save_name}_cnn.pt"
    torch.save({
        "model_state_dict": r["model"].state_dict(),
        "class_names": class_names,
        "img_target_size": r["img_size"],
        "component": COMPONENT,
        "hidden_layers": r["hidden_layers"],
        "neurons": r["neurons"],
        "test_accuracy": r["test_acc"],
        "mean_auc": r["mean_auc"],
    }, model_path)
    print(f"Saved model weights to {model_path}")


# ------------------------- ARCHITECTURE HYPERPARAMETERS -------------------------
# Bundles everything each architecture's training notebook needs to build
# its classifier head, optimizer and dataloaders. img_size defaults to
# IMG_TARGET_SIZE (set above) except for InceptionNet v3, which requires
# larger inputs (it downsamples aggressively and expects ~299x299).

ARCH_HYPERPARAMS = {
    "resnet50": dict(
        hidden_layers=3, neurons=1024,
        lr=2.68014042e-05, beta1=0.1788129, step_size=34, batch_size=121,
        img_size=IMG_TARGET_SIZE,
    ),
    "densenet121": dict(
        hidden_layers=3, neurons=2048,
        lr=2.74235159e-04, beta1=4.149011e-17, step_size=7, batch_size=14,
        img_size=IMG_TARGET_SIZE,
    ),
    "inception_v3": dict(
        hidden_layers=3, neurons=2048,
        lr=1.42756669e-04, beta1=5.190841e-17, step_size=21, batch_size=17,
        img_size=(299, 299),
    ),
    # EfficientNet-B0 wasn't part of the original grid search / Bayesian
    # tuning sweep, so these are sensible defaults rather than tuned values:
    # a middling classifier head (matching the other architectures), a
    # standard Adam beta1, and a StepLR schedule. Adjust freely if you want
    # to run your own tuning pass for it.
    "efficientnet_b0": dict(
        hidden_layers=3, neurons=1024,
        lr=1e-4, beta1=0.9, step_size=15, batch_size=32,
        img_size=IMG_TARGET_SIZE,
    ),
    # Also not part of the original tuning sweep — sensible defaults again.
    # ConvNeXt is a heavier/slower backbone than the others, so it gets a
    # smaller default batch size and a slightly lower LR to start.
    "convnext_tiny": dict(
        hidden_layers=3, neurons=1024,
        lr=5e-5, beta1=0.9, step_size=15, batch_size=24,
        img_size=IMG_TARGET_SIZE,
    ),
}

# beta2 (decay rate for the squared-gradient average) is held fixed at 0.9
# for all models, per the tuning setup.
ADAM_BETA2 = 0.9

# Upper bound on the *actual* per-step batch size used on the GPU. Some of
# the tuned batch sizes above (e.g. 121 for ResNet-50) are too large to fit
# a deeper model's activations in memory alongside everything else — the
# training loop uses gradient accumulation to still reach the tuned
# effective batch size without needing that many samples in memory at once.
# Lower this (e.g. to 8) if you still see CUDA out-of-memory errors.
MICRO_BATCH_CAP = 4

# Reference dataframes from the hyperparameter tuning sweep (grid search +
# Bayesian optimization). Not used programmatically anywhere else — kept
# here for documentation/reference. Call print_tuning_summary() from a
# notebook cell if you want to see them.
_architecture_grid_search = pd.DataFrame([
    {"model": "ResNet-50",       "hidden_layers (H)": 3, "neurons (N)": 1024, "val_accuracy_%": 97.37},
    {"model": "DenseNet-121",    "hidden_layers (H)": 3, "neurons (N)": 2048, "val_accuracy_%": 96.05},
    {"model": "InceptionNet v3", "hidden_layers (H)": 3, "neurons (N)": 2048, "val_accuracy_%": 97.70},
]).set_index("model")

_bayesian_opt_results = pd.DataFrame([
    {"model": "ResNet-50",       "learning_rate": 2.68014042e-05, "beta1": 0.1788129,   "step_size_Es": 34, "batch_size": 121},
    {"model": "DenseNet-121",    "learning_rate": 2.74235159e-04, "beta1": 4.149011e-17, "step_size_Es": 7,  "batch_size": 14},
    {"model": "InceptionNet v3", "learning_rate": 1.42756669e-04, "beta1": 5.190841e-17, "step_size_Es": 21, "batch_size": 17},
]).set_index("model")


def print_tuning_summary():
    """Optional: call from a notebook cell to display the hyperparameter
    tuning sweep results that ARCH_HYPERPARAMS above was derived from."""
    print("Architecture grid search (best H, N per model):")
    print(_architecture_grid_search)
    print("\nBayesian-optimized learning hyperparameters:")
    print(_bayesian_opt_results)
