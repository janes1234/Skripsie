"""Redraws the multi-class confusion matrices used in the report, row-normalised.

Each cell shows the fraction of the true class (row) assigned to the predicted
class (column), with the count below it, and each row label gives the number
of test images of that class. The colour scale is fixed to [0, 1], so the
matrices of different facilities can be compared directly.

Reads the results pickles in ../results/<component>_aug and overwrites the
copies in ../Report/figures/results. The original count matrices saved during
training stay in ../results/<component>_aug/figures.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 15})

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "Report" / "figures" / "results"

FIGURES = {
    "clarifier": ("convnext_tiny", ["Atlantis", "CapeFlats", "Fisantekraal", "NoordelikeWerke", "Waterval"]),
    "aerobic_zone": ("densenet_121", ["Atlantis", "CapeFlats", "Waterval"]),
}
PICKLE_MODEL = {"convnext_tiny": "convnext_tiny", "densenet_121": "densenet121"}


def plot_row_normalised(cm, class_names, out_path):
    totals = cm.sum(axis=1)
    rates = np.divide(cm, totals[:, None], out=np.zeros(cm.shape), where=totals[:, None] > 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = "white" if rates[i, j] > 0.5 else "black"
            ax.text(j, i, f"{rates[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", color=colour, fontsize=16)

    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), [f"{c}\n(n={n})" for c, n in zip(class_names, totals)])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for component, (model, facilities) in FIGURES.items():
        for facility in facilities:
            pkl = RESULTS_ROOT / f"{component}_aug" / f"results_{component}_{PICKLE_MODEL[model]}_{facility}.pkl"
            with open(pkl, "rb") as f:
                r = pickle.load(f)
            out_path = OUT_DIR / f"{component}_{model}_{facility}_confusion_matrix.png"
            plot_row_normalised(np.asarray(r["confusion_matrix"]), r["class_names"], out_path)
            print(f"{pkl.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
