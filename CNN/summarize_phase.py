"""
summarize_phase.py
===================
Standalone summary for one snapshot of results_clarifier_*.pkl files: mean
AUC per architecture (averaged across the 5 held-out facilities) and the
OK ("Functional") vs not-OK binary ROC-AUC + recall-driven threshold
table, per architecture -- averaged across the 5 held-out facilities
(one ROC curve per facility, then macro-averaged), NOT pooled. See the
"NOTE ON POOLING vs. AVERAGING" comment below for why.

This duplicates the relevant logic from results_comparison.ipynb (cells 1,
3, 13, 16) rather than running that notebook directly, because it needs to
point at an archived results/archive/phaseN_*/ snapshot instead of the live
results/ directory -- results_comparison.ipynb hardcodes RESULTS_DIR =
"../results", which is exactly the directory a training batch may currently
be overwriting.

Usage:
    python summarize_phase.py ../results/archive/phase0_before_retune
    python summarize_phase.py ../results/archive/phase1_tuned
    python summarize_phase.py ../results          # only when nothing is writing to it
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

COMPONENT = "clarifier"
MODELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "inception_v3": "InceptionNet v3",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "alexnet": "AlexNet",
}
ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]
OK_CLASS = "Functional"
TARGET_RECALL = 0.95


def load(results_dir: Path):
    results = {}
    missing = []
    for facility in ALL_FACILITIES:
        results[facility] = {}
        for save_name in MODELS:
            path = results_dir / f"results_{COMPONENT}_{save_name}_{facility}.pkl"
            if not path.exists():
                missing.append(str(path))
                continue
            with open(path, "rb") as f:
                results[facility][save_name] = pickle.load(f)
    return results, missing


def mean_auc_table(results):
    rows = []
    for save_name, display_name in MODELS.items():
        aucs = [results[fac][save_name]["mean_auc"] for fac in ALL_FACILITIES if save_name in results[fac]]
        if aucs:
            rows.append((display_name, np.mean(aucs), len(aucs)))
    return rows


# NOTE ON POOLING vs. AVERAGING: earlier versions of this analysis (and of
# results_comparison.ipynb) pooled every held-out facility's predictions
# into one combined array before computing a single ROC/AUC. That's wrong
# for this experimental design -- each held-out facility is a *different*
# trained model evaluated on a non-overlapping test set (leave-one-
# facility-out), so pooling lets facility size dominate (CapeFlats alone
# has ~5x Atlantis's clarifier test images) and breaks the independence a
# single ROC curve assumes. Everything below instead computes one ROC
# curve/threshold-table row per facility and *averages* across facilities,
# so each facility counts as one equally-weighted generalization test --
# matching "average of 5 ROCs, not pooled" as already described to Thomas.

FPR_GRID = np.linspace(0.0, 1.0, 200)      # common x-axis for macro-averaging ROC curves
THRESH_GRID = np.linspace(0.0, 1.0, 500)   # common threshold axis for macro-averaging recall


def per_facility_binary_scores(results, save_name):
    """facility -> (y_true, y_score) for one architecture. NOT pooled --
    each facility's array stays separate."""
    out = {}
    for facility, by_model in results.items():
        r = by_model.get(save_name)
        if r is None:
            continue
        y_true = np.array([0 if s["true"] == OK_CLASS else 1 for s in r["per_sample"]])
        y_score = np.array([1.0 - s["probs"][OK_CLASS] for s in r["per_sample"]])
        out[facility] = (y_true, y_score)
    return out


def per_facility_roc(per_facility_scores):
    """facility -> {fpr, tpr, auc, interp_tpr} via sklearn's roc_curve on
    that facility alone, plus tpr interpolated onto the shared FPR_GRID so
    curves from different facilities (with different raw fpr/tpr points)
    can be averaged pointwise. Facilities with only one class present in
    their not-OK/OK split (ROC/AUC undefined) are skipped, same spirit as
    wwtw_utils.py's own "0 examples of this class" AUC handling."""
    curves = {}
    for facility, (y_true, y_score) in per_facility_scores.items():
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        interp_tpr = np.interp(FPR_GRID, fpr, tpr)
        interp_tpr[0] = 0.0
        curves[facility] = dict(fpr=fpr, tpr=tpr, thresh=thresh, auc=auc(fpr, tpr), interp_tpr=interp_tpr)
    return curves


def macro_average_roc(curves):
    """Mean TPR over FPR_GRID for plotting, and the mean of the individual
    per-facility AUCs (not the AUC of the mean curve) -- same convention
    mean_auc_table() already uses for the 4-class metric."""
    interp_tprs = np.array([c["interp_tpr"] for c in curves.values()])
    aucs = np.array([c["auc"] for c in curves.values()])
    return dict(
        mean_tpr=interp_tprs.mean(axis=0),
        mean_auc=aucs.mean(), n=len(curves),
    )


def per_facility_recall_curve(per_facility_scores):
    """facility -> recall (TPR) at every point of THRESH_GRID, swept
    directly on that facility's own scores (not via roc_curve's uneven
    threshold spacing), so recall-vs-threshold can be averaged across
    facilities on a shared threshold axis."""
    out = {}
    for facility, (y_true, y_score) in per_facility_scores.items():
        n_pos = y_true.sum()
        if n_pos == 0:
            continue
        out[facility] = np.array([
            ((y_score >= t) & (y_true == 1)).sum() / n_pos for t in THRESH_GRID
        ])
    return out


def choose_averaged_threshold(per_facility_scores, target_recall):
    """Most permissive threshold (highest on THRESH_GRID, i.e. flags the
    fewest images as not-OK) whose *mean* recall across facilities still
    reaches target_recall. Falls back to threshold 0.0 (flag everything) if
    no threshold reaches it on average."""
    recall_curves = per_facility_recall_curve(per_facility_scores)
    mean_recall = np.mean(list(recall_curves.values()), axis=0)
    valid = np.where(mean_recall >= target_recall)[0]
    idx = int(valid.max()) if len(valid) else 0
    return float(THRESH_GRID[idx]), mean_recall


def binary_table(results):
    rows = []
    for save_name, display_name in MODELS.items():
        per_facility_scores = per_facility_binary_scores(results, save_name)
        curves = per_facility_roc(per_facility_scores)
        if not curves:
            continue
        macro = macro_average_roc(curves)
        t, _ = choose_averaged_threshold(per_facility_scores, TARGET_RECALL)

        recalls, precisions, specificities = [], [], []
        for facility, (y_true, y_score) in per_facility_scores.items():
            y_pred = (y_score >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            if (tp + fn) > 0:
                recalls.append(tp / (tp + fn))
            if (tp + fp) > 0:
                precisions.append(tp / (tp + fp))
            if (tn + fp) > 0:
                specificities.append(tn / (tn + fp))

        rows.append(dict(
            model=display_name, binary_auc=macro["mean_auc"],
            threshold=t,
            recall=np.mean(recalls) if recalls else float("nan"),
            precision=np.mean(precisions) if precisions else float("nan"),
            specificity=np.mean(specificities) if specificities else float("nan"),
            n_facilities=macro["n"],
        ))
    return rows


def per_facility_rate_confusion(y_true, y_pred):
    """Row-normalized 2x2 confusion matrix (each row sums to 1, i.e. rates
    not counts) for one facility -- averaging *these* across facilities
    (rather than summing raw counts) is what keeps the aggregate figure
    from pooling."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def save_figures(results, figures_dir: Path, phase_label: str):
    """Regenerates results_comparison.ipynb's two OK-vs-not-OK figures
    (cells 14 and 17), from an arbitrary results snapshot, averaging across
    facilities instead of pooling them. Saved with phase_label in the
    filename so they never collide with (or silently overwrite) another
    phase's figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_model_scores = {sn: per_facility_binary_scores(results, sn) for sn in MODELS}
    per_model_curves = {sn: per_facility_roc(s) for sn, s in per_model_scores.items() if s}
    per_model_curves = {sn: c for sn, c in per_model_curves.items() if c}
    per_model_macro = {sn: macro_average_roc(c) for sn, c in per_model_curves.items()}
    per_model_threshold = {
        sn: choose_averaged_threshold(per_model_scores[sn], TARGET_RECALL)[0]
        for sn in per_model_curves
    }

    # --- Mean ROC curve across facilities, one line per model (average only). ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for save_name, macro in per_model_macro.items():
        ax.plot(
            FPR_GRID, macro["mean_tpr"],
            label=f"{MODELS[save_name]} (mean AUC = {macro['mean_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.axhline(TARGET_RECALL, color="black", linestyle=":", alpha=0.5,
               label=f"Target recall = {TARGET_RECALL:.0%}")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity / Recall (TPR) — fraction of malfunctioning images caught")
    ax.set_title(f"OK ({OK_CLASS}) vs not-OK ROC — mean of 5 per-facility ROCs ({COMPONENT})",
                 fontsize=10)
    ax.text(0.5, 1.06, phase_label, transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = figures_dir / f"{COMPONENT}_ok_vs_notok_roc_{phase_label}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)

    # --- Confusion matrices at chosen threshold: average RATE across
    # facilities (not pooled counts), one 2x2 matrix per model ---
    models_with_curves = list(per_model_curves.keys())
    n = len(models_with_curves)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, save_name in zip(axes, models_with_curves):
        t = per_model_threshold[save_name]
        rate_cms = [
            per_facility_rate_confusion(y_true, (y_score >= t).astype(int))
            for y_true, y_score in per_model_scores[save_name].values()
        ]
        mean_rate_cm = np.mean(rate_cms, axis=0)
        ConfusionMatrixDisplay(mean_rate_cm, display_labels=["OK", "not OK"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".2f"
        )
        ax.set_title(f"{MODELS[save_name]}\nthreshold = {t:.3f} (mean rate over {len(rate_cms)} facilities)",
                     fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{phase_label} — cell values are rates (row-normalized), averaged across facilities, not pooled counts")
    plt.tight_layout()
    cm_path = figures_dir / f"{COMPONENT}_ok_vs_notok_confusion_at_threshold_{phase_label}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    return roc_path, cm_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--label", default=None, help="Label for this snapshot in the printed output")
    parser.add_argument("--save-figures", action="store_true",
                         help="Also regenerate the OK-vs-not-OK ROC + confusion-matrix figures "
                              "for this snapshot, saved into --figures-dir with the label in "
                              "the filename")
    parser.add_argument("--figures-dir", type=Path, default=Path("../results/figures"),
                         help="Where to save figures (default: ../results/figures, same place "
                              "results_comparison.ipynb saves everything else)")
    args = parser.parse_args()

    label = args.label or args.results_dir.name
    results, missing = load(args.results_dir)
    n_loaded = sum(len(v) for v in results.values())
    n_total = len(ALL_FACILITIES) * len(MODELS)
    print(f"=== {label} ({args.results_dir}) === loaded {n_loaded}/{n_total}")
    if missing:
        print(f"  missing {len(missing)}:")
        for m in missing:
            print(f"    {m}")

    print(f"\n{'model':<18}{'mean_auc':>10}{'n_facilities':>14}")
    for name, mean_a, n in mean_auc_table(results):
        print(f"{name:<18}{mean_a:>10.4f}{n:>14}")

    print(f"\nOK(Functional) vs not-OK, averaged over facilities (not pooled), target recall={TARGET_RECALL:.0%}:")
    print(f"{'model':<18}{'mean_auc':>10}{'thresh':>9}{'recall':>10}{'precision':>11}{'specificity':>13}")
    for r in binary_table(results):
        print(f"{r['model']:<18}{r['binary_auc']:>10.4f}{r['threshold']:>9.3f}"
              f"{r['recall']:>10.4f}{r['precision']:>11.4f}{r['specificity']:>13.4f}")

    if args.save_figures:
        phase_label = args.label.replace(" ", "_") if args.label else args.results_dir.name
        roc_path, cm_path = save_figures(results, args.figures_dir, phase_label)
        print(f"\nSaved figures:\n  {roc_path}\n  {cm_path}")


if __name__ == "__main__":
    main()
