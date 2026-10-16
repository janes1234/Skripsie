"""
run.py
======
Runs the training notebooks for every requested (architecture, test set)
combination. pipeline.py uses run_one() from here; the command line below is
for running single combinations by hand.

Each run executes the notebook with `jupyter nbconvert --execute`, with
DATASET_ROOT, COMPONENT, RESULTS_DIR and AUGMENT set in the environment (which
wwtw_utils.py reads at import time), and saves the executed notebook to
./executed_notebooks/. A combination whose results_*.pkl already exists is
skipped unless --force is given.

Usage:
    python run.py                                        # all architectures x all facilities
    python run.py --models resnet18,convnext_tiny --facilities Waterval,CapeFlats
    python run.py --component aerobic_zone
    python run.py --split chronological
    python run.py --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ------------------------- CONFIG -------------------------

# save_name (as in results_{component}_{save_name}_{facility}.pkl) -> notebook.
TRAIN_NOTEBOOKS = {
    "resnet18": "train_resnet18.ipynb",
    "resnet18_scratch": "train_resnet18_scratch.ipynb",
    "resnet50": "train_resnet50.ipynb",
    "densenet121": "train_densenet121.ipynb",
    "inception_v3": "train_inception_v3.ipynb",
    "efficientnet_b0": "train_efficientnet_b0.ipynb",
    "convnext_tiny": "train_convnext_tiny.ipynb",
}

# Matches ALL_FACILITIES in split_dataset_facility.py.
ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]
# Only these three facilities have aerobic zone labels.
AEROBIC_FACILITIES = ["Atlantis", "CapeFlats", "Waterval"]

# The chronological split's results use "chronological" in place of a facility name.
CHRONOLOGICAL_ID = "chronological"
CHRONOLOGICAL_ROOT = "../cnn_dataset_split-chronological"

RESULTS_DIR = Path("../results")
EXECUTED_NOTEBOOKS_DIR = Path("./executed_notebooks")

# No cell timeout: a training cell runs far longer than nbconvert's default.
CELL_TIMEOUT_SECONDS = -1

# ------------------------------------------------------------------------


def check_prerequisites(kernel_name: str):
    missing = []
    for tool in ("jupyter",):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"[error] Required command(s) not found on PATH: {', '.join(missing)}")
        print("        Install with: pip install nbconvert ipykernel jupyter_client --break-system-packages")
        sys.exit(1)

    # nbconvert runs notebooks in a registered Jupyter kernel, which is not
    # necessarily the active venv. Check up front that it can import torch.
    kernelspec = subprocess.run(["jupyter", "kernelspec", "list"], capture_output=True, text=True)
    if kernel_name not in kernelspec.stdout:
        print(f"[error] No Jupyter kernel named '{kernel_name}' is registered.")
        print(f"        Available kernels:\n{kernelspec.stdout}")
        print(
            "        If you're using a venv, register it as a kernel first (with the venv active):\n"
            "          pip install ipykernel --break-system-packages\n"
            f"          python -m ipykernel install --user --name={kernel_name} --display-name \"{kernel_name}\"\n"
            f"        Or pass the name of a kernel that already exists: --kernel-name <name>"
        )
        sys.exit(1)

    probe_nb = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["import numpy, torch\n", "print('KERNEL_PROBE_OK')\n"],
        }],
        "metadata": {"kernelspec": {"display_name": kernel_name, "language": "python", "name": kernel_name}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe_path = Path(tmp) / "kernel_probe.ipynb"
        with open(probe_path, "w") as f:
            _json.dump(probe_nb, f)
        probe = subprocess.run(
            [
                "jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=30",
                f"--ExecutePreprocessor.kernel_name={kernel_name}",
                str(probe_path), "--output", str(probe_path),
            ],
            capture_output=True, text=True,
        )
    if probe.returncode != 0:
        print(f"[error] Kernel '{kernel_name}' can't import numpy/torch -- it is probably not "
              f"the same Python environment as your venv.")
        print(f"        Probe output:\n{probe.stdout}\n{probe.stderr}")
        print(
            "        Fix (with your venv active):\n"
            "          pip install ipykernel --break-system-packages\n"
            f"          python -m ipykernel install --user --name={kernel_name} --display-name \"{kernel_name}\"\n"
            f"        Then re-run with: python run.py --kernel-name {kernel_name}"
        )
        sys.exit(1)


def dataset_root_for(facility: str) -> str:
    if facility == CHRONOLOGICAL_ID:
        return CHRONOLOGICAL_ROOT
    return f"../cnn_dataset_split_facility_test-{facility}"


def results_pkl_path(component: str, save_name: str, facility: str,
                     results_dir: Path = RESULTS_DIR) -> Path:
    # Same naming as save_result() in wwtw_utils.py.
    return Path(results_dir) / f"results_{component}_{save_name}_{facility}.pkl"


def run_one(component: str, save_name: str, notebook_name: str, facility: str,
            kernel_name: str, dry_run: bool, results_dir: Path = RESULTS_DIR,
            augment: bool = True) -> tuple[bool, str]:
    """Runs one (architecture, facility) combination. `facility` is the
    held-out facility, or CHRONOLOGICAL_ID for the chronological split. Results,
    figures and logs go to results_dir; executed notebooks to a subfolder of
    EXECUTED_NOTEBOOKS_DIR named after it (unless it's the default
    ../results). Returns (success, message)."""
    notebook_path = Path(notebook_name)
    if not notebook_path.exists():
        return False, f"notebook not found: {notebook_path}"

    results_dir = Path(results_dir)
    dataset_root = dataset_root_for(facility)
    if not Path(dataset_root).exists():
        return False, (
            f"dataset split not found: {dataset_root} "
            f"(run prep.py and the split scripts first, or python pipeline.py --prep-only)"
        )

    if dry_run:
        return True, f"[dry-run] would train {save_name} on {component}, held-out facility {facility}"

    executed_dir = EXECUTED_NOTEBOOKS_DIR
    if results_dir.resolve() != RESULTS_DIR.resolve():
        executed_dir = EXECUTED_NOTEBOOKS_DIR / results_dir.name
    logs_dir = results_dir / "logs"
    executed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    executed_path = executed_dir / f"{component}_{save_name}_{facility}.ipynb"
    log_path = logs_dir / f"{component}_{save_name}_{facility}.log"

    env = os.environ.copy()
    env["DATASET_ROOT"] = dataset_root
    env["COMPONENT"] = component
    env["RESULTS_DIR"] = str(results_dir)
    env["AUGMENT"] = "1" if augment else "0"

    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={CELL_TIMEOUT_SECONDS}",
        f"--ExecutePreprocessor.kernel_name={kernel_name}",
        str(notebook_path),
        "--output", str(executed_path.resolve()),
    ]

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    if proc.returncode != 0:
        return False, f"FAILED after {elapsed/60:.1f} min -- see {log_path}"

    expected_pkl = results_pkl_path(component, save_name, facility, results_dir)
    if not expected_pkl.exists():
        return False, (
            f"notebook finished but {expected_pkl} was never written -- "
            f"check the notebook actually calls save_result(...) as its last cell. See {log_path}"
        )

    return True, f"OK in {elapsed/60:.1f} min -- {expected_pkl}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", default="clarifier", choices=["clarifier", "aerobic_zone"],
                         help="Component to train (default: clarifier)")
    parser.add_argument("--models", default=None,
                         help=f"Comma-separated save_names to run (default: all). Choices: {', '.join(TRAIN_NOTEBOOKS)}")
    parser.add_argument("--facilities", default=None,
                         help=f"Comma-separated facilities to hold out (default: all). Choices: {', '.join(ALL_FACILITIES)}")
    parser.add_argument("--split", default="lofo", choices=["lofo", "chronological"],
                         help="lofo = leave-one-facility-out folds (default); chronological = "
                              f"train-before/test-after-cutoff split in {CHRONOLOGICAL_ROOT}")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR),
                         help=f"Where results/figures/logs go (default: {RESULTS_DIR})")
    parser.add_argument("--no-augment", action="store_true",
                         help="Train without any train-time augmentation")
    parser.add_argument("--kernel-name", default="python3",
                         help="Jupyter kernel name to execute notebooks with (default: python3)")
    parser.add_argument("--force", action="store_true",
                         help="Re-run even if a results_*.pkl already exists for that combination")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would run without training anything")
    args = parser.parse_args()

    check_prerequisites(args.kernel_name)

    models = args.models.split(",") if args.models else list(TRAIN_NOTEBOOKS)
    default_facilities = AEROBIC_FACILITIES if args.component == "aerobic_zone" else ALL_FACILITIES
    facilities = args.facilities.split(",") if args.facilities else list(default_facilities)
    if args.split == "chronological":
        facilities = [CHRONOLOGICAL_ID]

    unknown_models = [m for m in models if m not in TRAIN_NOTEBOOKS]
    if unknown_models:
        parser.error(f"Unknown model(s): {unknown_models}. Choices: {list(TRAIN_NOTEBOOKS)}")
    unknown_facilities = [f for f in facilities if f not in ALL_FACILITIES + [CHRONOLOGICAL_ID]]
    if unknown_facilities:
        parser.error(f"Unknown facility(ies): {unknown_facilities}. Choices: {ALL_FACILITIES}")

    combinations = [(m, f) for f in facilities for m in models]
    print(f"Component: {args.component}")
    print(f"Models ({len(models)}): {models}")
    print(f"Facilities ({len(facilities)}): {facilities}")
    print(f"Total combinations: {len(combinations)}\n")

    summary = []
    for i, (save_name, facility) in enumerate(combinations, start=1):
        notebook_name = TRAIN_NOTEBOOKS[save_name]
        pkl_path = results_pkl_path(args.component, save_name, facility, args.results_dir)

        print(f"[{i}/{len(combinations)}] {save_name} | held-out facility: {facility}")

        if pkl_path.exists() and not args.force:
            print(f"    skipped (already exists: {pkl_path}, use --force to re-run)")
            summary.append((save_name, facility, "skipped"))
            continue

        success, message = run_one(
            args.component, save_name, notebook_name, facility,
            args.kernel_name, args.dry_run, args.results_dir, not args.no_augment,
        )
        print(f"    {'OK' if success else 'FAILED'}: {message}")
        summary.append((save_name, facility, "ok" if success else "failed"))

    print("\n=== Run summary ===")
    n_ok = sum(1 for *_, status in summary if status == "ok")
    n_skipped = sum(1 for *_, status in summary if status == "skipped")
    n_failed = sum(1 for *_, status in summary if status == "failed")
    print(f"OK: {n_ok}  Skipped: {n_skipped}  Failed: {n_failed}  Total: {len(summary)}")
    if n_failed:
        print("\nFailed combinations:")
        for save_name, facility, status in summary:
            if status == "failed":
                print(f"  - {save_name} / {facility}")
        sys.exit(1)


if __name__ == "__main__":
    main()