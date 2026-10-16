"""
wwtw_utils.py
=============
Shared config, data loading, training and evaluation code for the train_*.ipynb
notebooks, which all start with `from wwtw_utils import *`.

DATASET_ROOT, COMPONENT, AUGMENT and RESULTS_DIR are read from the environment
at import time, so run.py / pipeline.py pick the split and component per run.
"""

import os
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR

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

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "../cnn_dataset_split_facility_test-Waterval"))
COMPONENT = os.environ.get("COMPONENT", "clarifier")  # "clarifier" or "aerobic_zone"

# (height, width). Aerobic zones are wide rectangles, clarifiers roughly square.
IMG_TARGET_SIZE = {
    "aerobic_zone": (112, 224),
    "clarifier": (224, 224),
}[COMPONENT]

WEIGHT_DECAY = 1e-4
EPOCHS = 100
EARLY_STOP_PATIENCE = 10
SEED = 42
ADAM_BETA2 = 0.999

# Largest batch actually put on the GPU. Bigger tuned batch sizes are reached
# with gradient accumulation (see run_epoch).
MICRO_BATCH_CAP = 16

# AUGMENT=0 turns off all train-time augmentation, for the with/without comparison.
AUGMENT = os.environ.get("AUGMENT", "1") != "0"

MODEL_SAVE_DIR = Path("../models/pad")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "../results"))
FIGURES_DIR = RESULTS_DIR / "figures"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_test_set_id():
    """Test set name from DATASET_ROOT, e.g. '..._facility_test-Waterval' ->
    'Waterval' and '..._split-chronological' -> 'chronological'."""
    return DATASET_ROOT.name.split("-")[-1]


TEST_SET_ID = get_test_set_id()
print(f"Test set ID: {TEST_SET_ID}")

data_dir = DATASET_ROOT / COMPONENT
assert data_dir.exists(), f"Dataset folder not found: {data_dir}"

torch.manual_seed(SEED)
print(f"Train-time augmentation: {'on' if AUGMENT else 'off'}")

# The canonical class -> index mapping, taken from the training split. Every
# other split is forced onto it (see align_dataset_to_classes).
class_names, class_to_idx = datasets.folder.find_classes(data_dir / "train")
num_classes = len(class_names)
print(f"Classes found ({num_classes}): {class_names}")


# ------------------------- DATA -------------------------

class LetterboxResize:
    """Scales a PIL image to fit inside `size` (h, w) without changing its
    aspect ratio, then pads it with black to exactly `size`, centred."""

    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, img):
        th, tw = self.size
        w, h = img.size
        scale = min(th / h, tw / w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        left, top = (tw - nw) // 2, (th - nh) // 2
        canvas = Image.new(img.mode, (tw, th))
        canvas.paste(img, (left, top))
        return canvas

    def __repr__(self):
        return f"LetterboxResize(size={self.size})"


def augmentation_transforms():
    """The train-time augmentations, in order. Also used for the augmentation
    figure in data_exploration.ipynb."""
    return [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    ]


def make_transform(img_size, train):
    augment = augmentation_transforms() if (train and AUGMENT) else []
    return transforms.Compose([
        LetterboxResize(img_size),
        *augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def align_dataset_to_classes(ds, class_to_idx):
    """ImageFolder numbers classes by the folders present in that split, so a
    split missing a class (e.g. no Empty clarifiers at Waterval) would shift
    every later class onto the wrong index. This remaps `ds` onto the
    canonical `class_to_idx`."""
    idx_remap = {old_idx: class_to_idx[name] for name, old_idx in ds.class_to_idx.items()}
    ds.samples = [(path, idx_remap[old_idx]) for path, old_idx in ds.samples]
    ds.targets = [idx_remap[old_idx] for old_idx in ds.targets]
    ds.class_to_idx = dict(class_to_idx)
    ds.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    return ds


# Python 3.14 defaults to "forkserver" on Linux, which crashes once CUDA is
# initialised in the parent. The workers only do CPU-side image loading, so
# "fork" is safe.
DATALOADER_MP_CONTEXT = "fork"


def get_dataloaders(img_size, batch_size):
    """Train/val/test ImageFolder datasets and loaders for one architecture."""
    tr_ds = datasets.ImageFolder(root=data_dir / "train", transform=make_transform(img_size, train=True))
    va_ds = datasets.ImageFolder(root=data_dir / "val", transform=make_transform(img_size, train=False))
    te_ds = datasets.ImageFolder(root=data_dir / "test", transform=make_transform(img_size, train=False))
    for ds in (tr_ds, va_ds, te_ds):
        align_dataset_to_classes(ds, class_to_idx)
    print(f"Train: {len(tr_ds)} | Val: {len(va_ds)} | Test: {len(te_ds)}")

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True,
                       multiprocessing_context=DATALOADER_MP_CONTEXT)
    # A last training batch of one image crashes Inception-v3's aux classifier
    # (BatchNorm on a 1x1 feature map), so it is dropped in that case only.
    tr_loader = DataLoader(tr_ds, shuffle=True, drop_last=(len(tr_ds) % batch_size == 1), **loader_args)
    va_loader = DataLoader(va_ds, shuffle=False, **loader_args)
    te_loader = DataLoader(te_ds, shuffle=False, **loader_args)
    return tr_ds, va_ds, te_ds, tr_loader, va_loader, te_loader


def make_class_weights(train_ds_):
    """Inverse-frequency class weights, normalised to sum to num_classes."""
    counts = np.bincount([label for _, label in train_ds_.samples], minlength=num_classes)
    w = 1.0 / torch.tensor(counts, dtype=torch.float32)
    w = w / w.sum() * num_classes
    return w.to(device)


# ------------------------- MODEL -------------------------

def build_classifier_head(in_features, hidden_layers, neurons, num_classes):
    """`hidden_layers` Linear+ReLU layers of `neurons` units, then a final
    Linear to num_classes. hidden_layers=0 gives a single Linear layer."""
    layers = []
    in_f = in_features
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_f, neurons))
        layers.append(nn.ReLU(inplace=True))
        in_f = neurons
    layers.append(nn.Linear(in_f, num_classes))
    return nn.Sequential(*layers)


# ------------------------- TRAINING -------------------------

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.float16


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, is_inception=False, accum_steps=1):
    """One pass over `loader`. Trains when an optimizer is given, otherwise
    only evaluates. With accum_steps > 1, gradients are summed over that many
    micro-batches before each optimizer step, so the effective batch size
    matches the tuned one. Returns (mean loss, accuracy)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    if train_mode:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train_mode):
        step = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                if is_inception and train_mode:
                    # Inception-v3's auxiliary classifier adds a 0.4-weighted loss term.
                    y_hat, aux_hat = model(images)
                    loss = criterion(y_hat, labels) + 0.4 * criterion(aux_hat, labels)
                else:
                    y_hat = model(images)
                    loss = criterion(y_hat, labels)

            if train_mode:
                scaler.scale(loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            total_loss += loss.item() * images.size(0)
            correct += (y_hat.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            step += 1

        # Apply the gradients of a final, incomplete accumulation group.
        if train_mode and total > 0 and step % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                epochs, patience, model_name, is_inception=False, accum_steps=1,
                epoch_callback=None):
    """Trains with early stopping on validation loss and restores the weights
    of the epoch with the lowest validation loss. epoch_callback(epoch,
    val_loss, val_acc) is called after every epoch; tune.py uses it for Optuna
    pruning."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    if accum_steps > 1:
        print(f"[{model_name}] Gradient accumulation: {accum_steps} micro-batches per optimizer step.")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                          is_inception, accum_steps)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, is_inception=is_inception)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

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

        if epoch_callback is not None:
            epoch_callback(epoch, val_loss, val_acc)

        if patience is not None and epochs_no_improve >= patience:
            print(f"\n[{model_name}] No val loss improvement for {patience} epochs, stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    print(f"\n[{model_name}] Restored best checkpoint (val loss = {best_val_loss:.4f})")
    return history


# ------------------------- EVALUATION -------------------------

def _slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _save_figure(fig, kind: str, model_name: str):
    """Saves `fig` to FIGURES_DIR, since headless runs never display plt.show()."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{COMPONENT}_{_slugify(model_name)}_{TEST_SET_ID}_{kind}.png"
    fig.savefig(FIGURES_DIR / fname, dpi=150, bbox_inches="tight")


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
    _save_figure(fig, "training_curves", model_name)
    plt.show()


def collect_predictions(model, loader):
    """Returns (predicted labels, true labels, softmax probabilities)."""
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
    _save_figure(fig, "confusion_matrix", model_name)
    plt.show()

    report_args = dict(labels=range(len(class_names)), target_names=class_names, digits=3, zero_division=0)
    print(classification_report(all_labels, all_preds, **report_args))
    report_dict = classification_report(all_labels, all_preds, output_dict=True, **report_args)
    return test_acc, cm, report_dict


def plot_roc_auc(all_labels, all_probs, class_names, num_classes, model_name):
    """One-vs-rest ROC curve and AUC per class. A class with no test images
    gets an AUC of NaN, since AUC is undefined without positives."""
    y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    fig = plt.figure(figsize=(7, 6))
    roc_aucs = {}
    for i, name in enumerate(class_names):
        if y_true_bin[:, i].sum() == 0:
            roc_aucs[name] = float("nan")
            print(f"  [warn] '{name}' has 0 examples in this test set — AUC undefined, skipping.")
            continue
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
    _save_figure(fig, "roc_auc", model_name)
    plt.show()

    print("AUC (area under ROC curve) per class:")
    for name, auc in roc_aucs.items():
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "undefined (0 examples)"
        print(f"  {name:15s}: {auc_str}")
    valid_aucs = [a for a in roc_aucs.values() if not np.isnan(a)]
    print(f"\nMean AUC (over {len(valid_aucs)}/{len(roc_aucs)} classes with test examples): {np.mean(valid_aucs):.3f}")
    return roc_aucs


# Results of the architecture trained in this process, written out by save_result().
results = {}


def per_sample_records(loader, labels_, preds, probs):
    """One record per image (path, true and assigned class, softmax
    probabilities). The loader must not be shuffled, so dataset.samples is
    in the same order as the outputs."""
    sample_paths = [str(p) for p, _ in loader.dataset.samples]
    return [
        {
            "path": path,
            "true": class_names[true_idx],
            "pred": class_names[pred_idx],
            "correct": bool(true_idx == pred_idx),
            "probs": {name: float(p) for name, p in zip(class_names, prob_vec)},
        }
        for path, true_idx, pred_idx, prob_vec in zip(sample_paths, labels_, preds, probs)
    ]


def evaluate_and_record(model, test_loader, model_name, history, img_size, hidden_layers, neurons,
                        dev_loader):
    """Evaluates `model` on the test set and stores the metrics, plus a
    per-image record for the test set and the development set. The results
    notebooks choose the binary OK/not-OK threshold on the development
    records and report it on the test records."""
    for loader in (test_loader, dev_loader):
        assert loader.dataset.class_to_idx == class_to_idx, \
            "split is not aligned to the canonical class_to_idx"

    preds, labels_, probs = collect_predictions(model, test_loader)
    test_acc, cm, report_dict = plot_confusion_and_report(labels_, preds, class_names, model_name)
    roc_aucs = plot_roc_auc(labels_, probs, class_names, num_classes, model_name)
    per_sample = per_sample_records(test_loader, labels_, preds, probs)

    dev_preds, dev_labels, dev_probs = collect_predictions(model, dev_loader)
    dev_per_sample = per_sample_records(dev_loader, dev_labels, dev_preds, dev_probs)

    model_cpu = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results[model_name] = dict(
        model=model_cpu, history=history, test_acc=test_acc,
        mean_auc=float(np.nanmean(list(roc_aucs.values()))),
        img_size=img_size, hidden_layers=hidden_layers, neurons=neurons,
        per_sample=per_sample,
        dev_per_sample=dev_per_sample,
        confusion_matrix=cm,
        classification_report=report_dict,
        roc_aucs=roc_aucs,
    )
    return test_acc, roc_aucs


def save_result(model_key, save_name):
    """Writes the results of `model_key` to
    RESULTS_DIR/results_<COMPONENT>_<save_name>_<TEST_SET_ID>.pkl and its
    weights to MODEL_SAVE_DIR/<RESULTS_DIR name>/<COMPONENT>_<save_name>_<TEST_SET_ID>.pt,
    so every run keeps its own weights."""
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
        "test_set_id": TEST_SET_ID,
        "per_sample": r["per_sample"],
        "dev_per_sample": r["dev_per_sample"],
        "confusion_matrix": r["confusion_matrix"],
        "classification_report": r["classification_report"],
        "roc_aucs": r["roc_aucs"],
    }
    pkl_path = RESULTS_DIR / f"results_{COMPONENT}_{save_name}_{TEST_SET_ID}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(pickle_payload, f)
    print(f"Saved results to {pkl_path}")

    model_dir = MODEL_SAVE_DIR / RESULTS_DIR.name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{COMPONENT}_{save_name}_{TEST_SET_ID}.pt"
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


# ------------------------- HYPERPARAMETERS -------------------------

def _load_tuned_hyperparams(json_path):
    """Reads a tune.py results json into {arch: {hidden_layers, neurons, lr,
    beta1, step_size, batch_size, img_size}}."""
    raw = json.loads(Path(json_path).read_text())
    return {
        arch: dict(
            hidden_layers=r["hidden_layers"], neurons=r["neurons"], lr=r["lr"],
            beta1=r["beta1"], step_size=r["step_size"], batch_size=r["batch_size"],
            img_size=tuple(r["img_size"]),
        )
        for arch, r in raw.items()
    }


# Tuned values per architecture, written by tune.py (one json per component).
_tuned_json = Path(__file__).resolve().parent / (
    "tuned_hyperparams.json" if COMPONENT == "clarifier" else f"tuned_hyperparams_{COMPONENT}.json")
ARCH_HYPERPARAMS = _load_tuned_hyperparams(_tuned_json) if _tuned_json.exists() else {}
if not ARCH_HYPERPARAMS:
    print(f"[warn] {_tuned_json.name} not found -- no tuned hyperparameters for {COMPONENT} yet.")

# Fixed, untuned values for ResNet-18 (scratch).
SCRATCH_HYPERPARAMS = dict(
    hidden_layers=3, neurons=1024,
    lr=1e-4, beta1=0.9, step_size=15, batch_size=32,
    img_size=IMG_TARGET_SIZE,
)
