"""
make_results_notebooks.py
=========================
Writes results_clarifier.ipynb and results_aerobic_zone.ipynb from one shared
cell template, so both notebooks report exactly the same things:

  for the augmented experiment on both the stratified 70/20/10 split and
  leave-one-facility-out (LOFO), and for the no-augmentation experiment on
  the stratified split only:
    - comparison table (accuracy, mean AUC, macro precision/recall/F1)
      (+ mean/std across held-out facilities for LOFO)
    - accuracy / AUC bar charts
    - per-class precision / recall / F1
    - binary OK (Functional) vs not-OK ROC, a recall-driven threshold, the
      confusion matrix at that threshold, and a per-test-set check of it
    - every run's training curves / confusion matrix / ROC figure
  plus a summary table per experiment (also saved as CSV) and a
  with-vs-without augmentation comparison on the stratified split.

Only the config cell differs between the two. Re-run this script after
editing the template, then execute the notebooks (pipeline.py does that).
"""

import json
from pathlib import Path

NOTEBOOKS = {
    "clarifier": dict(
        path="results_clarifier.ipynb",
        title="Clarifiers",
        experiments={
            "aug": ("Augmented training", "../results/clarifier_aug", ["stratified", "lofo"]),
            "noaug": ("No augmentation", "../results/clarifier_noaug", ["stratified"]),
        },
        lofo_facilities=["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"],
        classes_note="`Functional`, `Dysfunctional`, `Scum`, `Empty`. Trained on the relabelled "
                     "data (`relabel/` with the supervisor's review applied).",
    ),
    "aerobic_zone": dict(
        path="results_aerobic_zone.ipynb",
        title="Aerobic zones",
        experiments={
            "aug": ("Augmented training", "../results/aerobic_zone_aug", ["stratified", "lofo"]),
            "noaug": ("No augmentation", "../results/aerobic_zone_noaug", ["stratified"]),
        },
        lofo_facilities=["Atlantis", "CapeFlats", "Waterval"],
        classes_note="`Functional`, `Suboptimal`, `Dysfunctional`. Trained on the original "
                     "labels (aerobic zones were not relabelled). Only Atlantis, Cape Flats and "
                     "Waterval have aerobic zones, so leave-one-facility-out has 3 folds. "
                     "Hyperparameters were tuned for aerobic zones separately "
                     "(`tuned_hyperparams_aerobic_zone.json`).",
    ),
}

HELPERS = r'''
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# save_name -> display name, matches TRAIN_NOTEBOOKS in run.py
MODELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "inception_v3": "InceptionNet v3",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "alexnet": "AlexNet",
}
STRATIFIED_ID = "stratified"   # test-set id of the stratified split, see run.py
SPLITS = {"stratified": "Stratified 70/20/10", "lofo": "Leave-one-facility-out"}

OK_CLASS = "Functional"
TARGET_RECALL = 0.95   # fraction of not-OK images the chosen threshold must catch (on average)
FPR_GRID = np.linspace(0.0, 1.0, 200)
THRESH_GRID = np.linspace(0.0, 1.0, 500)

pd.set_option("display.precision", 3)


def slugify(name):
    # Must match _slugify() in wwtw_utils.py (used for the saved figure names).
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def test_ids(split):
    return [STRATIFIED_ID] if split == "stratified" else LOFO_FACILITIES


def load_results(exp):
    """results[split][test_id][save_name] = payload written by save_result()."""
    out, missing = {}, []
    for split in EXPERIMENTS[exp][2]:
        out[split] = {}
        for tid in test_ids(split):
            out[split][tid] = {}
            for save_name in MODELS:
                path = EXPERIMENTS[exp][1] / f"results_{COMPONENT}_{save_name}_{tid}.pkl"
                if path.exists():
                    with open(path, "rb") as f:
                        out[split][tid][save_name] = pickle.load(f)
                else:
                    missing.append(path.name)
    return out, missing


def comparison_table(exp, split):
    rows = []
    for tid, by_model in RESULTS[exp][split].items():
        for save_name, r in by_model.items():
            macro = (r.get("classification_report") or {}).get("macro avg", {})
            rows.append({
                "test set": tid, "model": MODELS[save_name],
                "test_accuracy": r["test_acc"], "mean_auc": r["mean_auc"],
                "macro_precision": macro.get("precision", np.nan),
                "macro_recall": macro.get("recall", np.nan),
                "macro_f1": macro.get("f1-score", np.nan),
                "n_test_images": len(r["per_sample"]),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=list(MODELS.values()), ordered=True)
    df["test set"] = pd.Categorical(df["test set"], categories=test_ids(split), ordered=True)
    return df.set_index(["test set", "model"]).sort_index()


def show_comparison(exp, split):
    comp = comparison_table(exp, split)
    if comp is None:
        print("No results yet.")
        return
    display(comp)
    if split == "lofo":
        display(Markdown("**Mean ± std across held-out facilities** (each facility counts once):"))
        metrics = ["test_accuracy", "mean_auc", "macro_precision", "macro_recall", "macro_f1"]
        agg = comp.reset_index().groupby("model", observed=True)[metrics].agg(["mean", "std"])
        agg[("n_facilities", "")] = comp.reset_index().groupby("model", observed=True).size()
        display(agg)


def plot_bars(exp, split):
    comp = comparison_table(exp, split)
    if comp is None:
        print("No results yet.")
        return
    df = comp.reset_index()
    tids = [t for t in test_ids(split) if t in set(df["test set"])]
    groups = tids + (["Mean"] if split == "lofo" else [])
    models_present = [m for m in MODELS.values() if m in set(df["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.2 * len(groups) + 4), 4.8))
    x = np.arange(len(groups))
    width = 0.8 / max(len(models_present), 1)
    for i, model_name in enumerate(models_present):
        sub = df[df["model"] == model_name].set_index("test set")
        offset = (i - (len(models_present) - 1) / 2) * width
        for ax, col in zip(axes, ["test_accuracy", "mean_auc"]):
            vals = [sub[col].get(t, np.nan) for t in tids]
            if split == "lofo":
                vals.append(np.nanmean(vals))
            ax.bar(x + offset, vals, width, label=model_name)
    for ax, title in zip(axes, ["Test accuracy", "Mean AUC (one-vs-rest, 4-/3-class)"]):
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} - {SPLITS[split]}")
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS[exp][1] / "figures" / f"{COMPONENT}_{split}_accuracy_auc_bars.png", dpi=150)
    plt.show()


def show_per_class(exp, split):
    any_shown = False
    for tid, by_model in RESULTS[exp][split].items():
        for save_name in MODELS:
            r = by_model.get(save_name)
            report = r and r.get("classification_report")
            if not report:
                continue
            class_rows = {c: v for c, v in report.items() if c not in ("accuracy", "macro avg", "weighted avg")}
            display(Markdown(f"**{tid} - {MODELS[save_name]}**"))
            display(pd.DataFrame(class_rows).T[["precision", "recall", "f1-score", "support"]])
            any_shown = True
    if not any_shown:
        print("No results yet.")


def show_figures(exp, split):
    figures_dir = EXPERIMENTS[exp][1] / "figures"
    for tid, by_model in RESULTS[exp][split].items():
        if not by_model:
            continue
        display(Markdown(f"#### {'Stratified test set' if tid == STRATIFIED_ID else tid + ' (held out)'}"))
        for save_name in MODELS:
            r = by_model.get(save_name)
            if r is None:
                continue
            display(Markdown(f"**{MODELS[save_name]}** - test accuracy {r['test_acc']:.3f}, mean AUC {r['mean_auc']:.3f}"))
            for kind in ["training_curves", "confusion_matrix", "roc_auc"]:
                path = figures_dir / f"{COMPONENT}_{slugify(r['model_name'])}_{tid}_{kind}.png"
                if path.exists():
                    display(Image(filename=str(path)))
                else:
                    print(f"  [missing] {path}")


# ---------------- binary OK vs not-OK ----------------

def binary_scores(exp, split, save_name):
    """test_id -> (y_true, y_score) with y=1 for not-OK and score = 1 - P(OK).
    Kept per test set (NOT pooled): each LOFO fold is scored by a different model."""
    out = {}
    for tid, by_model in RESULTS[exp][split].items():
        r = by_model.get(save_name)
        if r is None:
            continue
        y_true = np.array([0 if s["true"] == OK_CLASS else 1 for s in r["per_sample"]])
        y_score = np.array([1.0 - s["probs"][OK_CLASS] for s in r["per_sample"]])
        out[tid] = (y_true, y_score)
    return out


def roc_per_test_set(scores):
    curves = {}
    for tid, (y_true, y_score) in scores.items():
        if len(np.unique(y_true)) < 2:
            print(f"  [warn] {tid}: only one of OK/not-OK present -- ROC undefined, skipped")
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(FPR_GRID, fpr, tpr)
        interp_tpr[0] = 0.0
        curves[tid] = dict(auc=auc(fpr, tpr), interp_tpr=interp_tpr)
    return curves


def choose_threshold(scores):
    """Most permissive threshold on P(not OK) whose MEAN recall across test
    sets still reaches TARGET_RECALL (0.0 = flag everything, if none does)."""
    recall_curves = []
    for y_true, y_score in scores.values():
        n_pos = y_true.sum()
        if n_pos:
            recall_curves.append([((y_score >= t) & (y_true == 1)).sum() / n_pos for t in THRESH_GRID])
    mean_recall = np.mean(recall_curves, axis=0)
    valid = np.where(mean_recall >= TARGET_RECALL)[0]
    return float(THRESH_GRID[int(valid.max()) if len(valid) else 0])


def rate_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    sums = cm.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return cm / sums


BINARY = {}   # (exp, split) -> dict(curves=..., thresholds=DataFrame, per_test_set=DataFrame)


def binary_analysis(exp, split):
    figures_dir = EXPERIMENTS[exp][1] / "figures"
    scores, curves = {}, {}
    for save_name in MODELS:
        s = binary_scores(exp, split, save_name)
        c = roc_per_test_set(s)
        if c:
            scores[save_name], curves[save_name] = s, c
    if not curves:
        print("No results yet.")
        return
    n_sets = len(test_ids(split))

    # ROC (mean of per-test-set ROCs; for the stratified split that's just the one ROC)
    fig, ax = plt.subplots(figsize=(7, 6))
    summary = {}
    for save_name, c in curves.items():
        aucs = np.array([v["auc"] for v in c.values()])
        mean_tpr = np.mean([v["interp_tpr"] for v in c.values()], axis=0)
        summary[save_name] = dict(auc=aucs.mean(), auc_std=aucs.std(), n=len(c))
        label = (f"{MODELS[save_name]} (AUC = {aucs.mean():.3f})" if split == "stratified"
                 else f"{MODELS[save_name]} (mean AUC = {aucs.mean():.3f} ± {aucs.std():.3f})")
        ax.plot(FPR_GRID, mean_tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.axhline(TARGET_RECALL, color="black", linestyle=":", alpha=0.5, label=f"Target recall = {TARGET_RECALL:.0%}")
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity / Recall - fraction of not-OK images caught")
    ax.set_title(f"OK vs not-OK ROC - {SPLITS[split]}" + ("" if split == "stratified" else f" (mean of {n_sets} facilities)"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{COMPONENT}_{split}_ok_vs_notok_roc.png", dpi=150)
    plt.show()

    # Threshold at the target recall + mean metrics at it
    rows, per_set_rows, thresholds = [], [], {}
    for save_name in curves:
        t = choose_threshold(scores[save_name])
        thresholds[save_name] = t
        rec, spec, prec = [], [], []
        for tid, (y_true, y_score) in scores[save_name].items():
            tn, fp, fn, tp = confusion_matrix(y_true, (y_score >= t).astype(int), labels=[0, 1]).ravel()
            r_ = tp / (tp + fn) if (tp + fn) else np.nan
            s_ = tn / (tn + fp) if (tn + fp) else np.nan
            p_ = tp / (tp + fp) if (tp + fp) else np.nan
            rec.append(r_); spec.append(s_); prec.append(p_)
            per_set_rows.append({"model": MODELS[save_name], "test set": tid, "recall": r_,
                                 "specificity": s_, "precision": p_,
                                 "n_not_ok": int(y_true.sum()), "n_images": len(y_true)})
        rows.append({"model": MODELS[save_name],
                     "OK/not-OK AUC": summary[save_name]["auc"],
                     "OK/not-OK AUC std": summary[save_name]["auc_std"] if split == "lofo" else np.nan,
                     "threshold on P(not OK)": t,
                     "recall (sensitivity)": np.nanmean(rec),
                     "specificity": np.nanmean(spec),
                     "precision": np.nanmean(prec),
                     "n_test_sets": len(scores[save_name])})
    thr_table = pd.DataFrame(rows).set_index("model")
    if split == "stratified":
        thr_table = thr_table.drop(columns=["OK/not-OK AUC std"])
    display(Markdown(f"**Operating point per model** - most permissive threshold reaching "
                     f"{TARGET_RECALL:.0%} {'recall' if split == 'stratified' else 'mean recall across facilities'}:"))
    display(thr_table)

    # Confusion matrix at that threshold (row-normalised rates, averaged across test sets)
    n = len(curves)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, save_name in zip(axes, curves):
        t = thresholds[save_name]
        cms = [rate_confusion(y, (s >= t).astype(int)) for y, s in scores[save_name].values()]
        ConfusionMatrixDisplay(np.mean(cms, axis=0), display_labels=["OK", "not OK"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"{MODELS[save_name]}\nthreshold = {t:.3f}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("OK vs not-OK at the chosen threshold - row-normalised rates"
                 + ("" if split == "stratified" else ", averaged across facilities (not pooled)"))
    plt.tight_layout()
    plt.savefig(figures_dir / f"{COMPONENT}_{split}_ok_vs_notok_confusion_at_threshold.png", dpi=150)
    plt.show()

    per_set = pd.DataFrame(per_set_rows).set_index(["model", "test set"])
    if split == "lofo":
        display(Markdown("**Does the one threshold hold up on each held-out facility?**"))
        display(per_set)
        display(Markdown("Recall spread across held-out facilities at each model's threshold:"))
        display(per_set["recall"].groupby("model", sort=False).agg(["mean", "std", "min", "max"]))
    BINARY[(exp, split)] = dict(thresholds=thr_table, per_test_set=per_set)


def summary_table(exp):
    rows = {}
    for save_name, name in MODELS.items():
        row = {}
        strat = RESULTS[exp]["stratified"][STRATIFIED_ID].get(save_name)
        if strat:
            row["stratified accuracy"] = strat["test_acc"]
            row["stratified mean AUC"] = strat["mean_auc"]
            row["stratified macro F1"] = strat["classification_report"]["macro avg"]["f1-score"]
        lofo = [by_model[save_name] for by_model in RESULTS[exp].get("lofo", {}).values() if save_name in by_model]
        if lofo:
            accs = [r["test_acc"] for r in lofo]
            row["LOFO accuracy mean"] = np.mean(accs)
            row["LOFO accuracy std"] = np.std(accs, ddof=1) if len(accs) > 1 else np.nan
            row["LOFO mean AUC"] = np.nanmean([r["mean_auc"] for r in lofo])
            row["LOFO macro F1"] = np.mean([r["classification_report"]["macro avg"]["f1-score"] for r in lofo])
            row["LOFO folds"] = len(lofo)
        for split in SPLITS:
            b = BINARY.get((exp, split))
            if b is not None and name in b["thresholds"].index:
                t = b["thresholds"].loc[name]
                row[f"{split} OK/not-OK AUC"] = t["OK/not-OK AUC"]
                row[f"{split} recall @thr"] = t["recall (sensitivity)"]
                row[f"{split} specificity @thr"] = t["specificity"]
        if row:
            rows[name] = row
    if not rows:
        print("No results yet.")
        return None
    df = pd.DataFrame(rows).T
    out = EXPERIMENTS[exp][1] / f"summary_{COMPONENT}_{exp}.csv"
    df.to_csv(out)
    display(df)
    print(f"Saved to {out}")
    return df
'''.strip("\n")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build(component, cfg):
    exps = cfg["experiments"]
    exp_lines = "\n".join(f'    "{k}": ("{label}", Path("{d}"), {splits!r}),' for k, (label, d, splits) in exps.items())
    cells = [
        md(f"""
# WWTW CNN results - {cfg['title']}

Every architecture's results for **{component}**, for both evaluation approaches:

- **Stratified 70/20/10** - one train/val/test split, stratified by (facility, unit, class).
- **Leave-one-facility-out (LOFO)** - one model per held-out facility; the test facility is never
  seen during training.

Training with vs without augmentation is compared on the stratified split only; LOFO is run
once, with augmentation.

Classes: {cfg['classes_note']}

For each approach: comparison table, accuracy/AUC charts, per-class precision/recall/F1, the binary
**OK (`Functional`) vs not-OK** (every other class pooled) ROC with a recall-driven threshold,
and every run's training curves / confusion matrix / ROC figure. The summary table at the end
of each experiment is also saved as CSV next to its results.

Generated by `make_results_notebooks.py` (the clarifier and aerobic zone notebooks share the
same cells; only the config cell differs). Needs no GPU: everything comes from the pickles and
PNGs the training runs saved.
"""),
        code(f"""
from pathlib import Path

COMPONENT = "{component}"
# experiment key -> (label, results folder written by run.py / pipeline.py, splits run)
EXPERIMENTS = {{
{exp_lines}
}}
LOFO_FACILITIES = {cfg['lofo_facilities']!r}
"""),
        code(HELPERS),
        md("## Load the results"),
        code("""
RESULTS = {}
for exp, (label, folder, _) in EXPERIMENTS.items():
    RESULTS[exp], missing = load_results(exp)
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    n_loaded = sum(len(m) for split in RESULTS[exp].values() for m in split.values())
    print(f"{label}: loaded {n_loaded} results from {folder}")
    if missing:
        print(f"  missing ({len(missing)}): " + ", ".join(missing))
class_names = next((r["class_names"] for exp in RESULTS.values() for split in exp.values()
                    for by_model in split.values() for r in by_model.values()), None)
print("Classes:", class_names)
"""),
    ]
    split_labels = {"stratified": "Stratified 70/20/10", "lofo": "Leave-one-facility-out"}
    for exp, (label, _, splits) in exps.items():
        cells.append(md(f"# {label}"))
        for split in splits:
            split_label = split_labels[split]
            lofo_note = (" One row per (held-out facility, model), followed by the mean ± std across "
                         "facilities." if split == "lofo" else "")
            bin_note = ("One ROC per held-out facility, then averaged (interpolated onto a common FPR "
                        "grid) rather than pooled, so each facility counts once; the threshold is the "
                        "most permissive one whose *mean* recall across facilities reaches the target."
                        if split == "lofo" else
                        "One test set, so this is a single ROC and the threshold is the most permissive "
                        "one that reaches the target recall on it.")
            cells += [
                md(f"## {split_label} - {label}"),
                md(f"### Comparison table\n\nMacro-averaged precision/recall/F1 (every class counts equally).{lofo_note}"),
                code(f'show_comparison("{exp}", "{split}")'),
                md("### Accuracy and AUC"),
                code(f'plot_bars("{exp}", "{split}")'),
                md("### Per-class precision / recall / F1"),
                code(f'show_per_class("{exp}", "{split}")'),
                md(f"""
### OK vs not-OK: ROC and a recall-driven threshold

Collapses the problem to **OK** (`Functional`) vs **not OK** (all other classes), scored by
`P(not OK) = 1 - P(Functional)`. What matters operationally is catching a component that is
no longer working properly, even at the cost of some false alarms, so the operating point is
chosen for a target recall rather than the default argmax rule. {bin_note}
"""),
                code(f'binary_analysis("{exp}", "{split}")'),
                md("### Training curves, confusion matrices and ROC curves per run"),
                code(f'show_figures("{exp}", "{split}")'),
            ]
        cells += [
            md(f"## Summary - {label}\n\nOne row per architecture: stratified results"
               + (", LOFO mean across held-out facilities," if "lofo" in splits else "")
               + " and the OK/not-OK operating point."),
            code(f'SUMMARY_{exp.upper()} = summary_table("{exp}")'),
        ]
    if len(exps) > 1:
        keys = list(exps)
        cells += [
            md("# With vs without augmentation\n\nStratified 70/20/10 results side by side "
               "(same tuned hyperparameters and split; only the train-time augmentation differs)."),
            code(f"""
_parts = {{EXPERIMENTS[k][0]: t for k, t in [{", ".join(f'("{k}", SUMMARY_{k.upper()})' for k in keys)}] if t is not None}}
if _parts:
    _cols = ["stratified accuracy", "stratified mean AUC", "stratified macro F1",
             "stratified OK/not-OK AUC", "stratified recall @thr", "stratified specificity @thr"]
    display(pd.concat({{k: v.reindex(columns=_cols) for k, v in _parts.items()}}, axis=1))
"""),
        ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    for component, cfg in NOTEBOOKS.items():
        nb = build(component, cfg)
        Path(cfg["path"]).write_text(json.dumps(nb, indent=1))
        print(f"wrote {cfg['path']} ({len(nb['cells'])} cells)")
