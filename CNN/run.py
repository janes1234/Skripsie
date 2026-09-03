"""
run.py
======
Batch-runs every training notebook (train_resnet18.ipynb, train_resnet50.ipynb,
train_densenet121.ipynb, train_inception_v3.ipynb, train_efficientnet_b0.ipynb,
train_convnext_tiny.ipynb) against every held-out facility's leave-one-
facility-out split, so you end up with a full 6 architectures x 5 facilities
results grid without babysitting each run by hand.

For each (architecture, facility) combination this script:
  1. Exports the training notebook to a plain .py script (via
     `jupyter nbconvert --to script`) into ./generated_scripts/, so you have
     a plain, version-controllable script on disk for every run -- this is
     purely a saved artifact though; see the note below on why execution
     itself goes through nbconvert's --execute path instead of just running
     that script directly.
  2. Actually executes the notebook via `jupyter nbconvert --to notebook
     --execute`, with DATASET_ROOT and COMPONENT set as environment
     variables for that run. wwtw_utils.py reads both of those at import
     time, so this is the one place that controls which facility is held
     out and which component is being trained -- nothing in the notebooks
     themselves needs to change.
  3. Saves the fully executed notebook (every printed line and every plot,
     inline, exactly as if you'd run it in Jupyter yourself) to
     ./executed_notebooks/{component}_{save_name}_{facility}.ipynb.
  4. Skips the combination if its results_*.pkl already exists in
     ../results, unless --force is given, so an interrupted batch run can
     just be re-launched and it'll pick up where it left off.

Why execute via nbconvert --execute rather than just running the plain .py
script from step 1: notebooks routinely contain things like
`get_ipython().run_line_magic(...)` (from magics like %matplotlib inline)
and `display(df)` calls, which only work inside a real IPython kernel.
Running the exported .py with a plain `python script.py` would crash on
those. --execute runs the notebook through an actual Jupyter kernel, so all
of that works exactly as it does interactively, and you additionally get an
executed notebook back with every plot preserved inline -- handy for
browsing results without having to dig through separate image files.

Usage:
    # Everything: 6 architectures x 5 facilities x 1 component
    python run.py

    # Just two architectures, two facilities
    python run.py --models resnet18,convnext_tiny --facilities Waterval,CapeFlats

    # aerobic_zone instead of the default clarifier
    python run.py --component aerobic_zone

    # Re-run even if a results_*.pkl already exists for that combination
    python run.py --force

    # See what would run without actually running anything
    python run.py --dry-run

Run this from the same directory the notebooks live in (the one containing
train_resnet18.ipynb etc.), same as you'd run any of the training notebooks
by hand -- DATASET_ROOT/RESULTS_DIR inside wwtw_utils.py are relative paths
(e.g. "../results"), so this script's own working directory matters.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ------------------------- CONFIG -------------------------

# save_name (matches results_comparison.ipynb's naming, and what save_result()
# writes into results_{component}_{save_name}_{facility}.pkl) -> notebook file.
TRAIN_NOTEBOOKS = {
    "resnet18": "train_resnet18.ipynb",
    "resnet50": "train_resnet50.ipynb",
    "densenet121": "train_densenet121.ipynb",
    "inception_v3": "train_inception_v3.ipynb",
    "efficientnet_b0": "train_efficientnet_b0.ipynb",
    "convnext_tiny": "train_convnext_tiny.ipynb",
}

# Matches ALL_FACILITIES in split_dataset_facility.py.
ALL_FACILITIES = ["Atlantis", "CapeFlats", "Waterval", "Fisantekraal", "NoordelikeWerke"]

RESULTS_DIR = Path("../results")
GENERATED_SCRIPTS_DIR = Path("./generated_scripts")
EXECUTED_NOTEBOOKS_DIR = Path("./executed_notebooks")
LOGS_DIR = RESULTS_DIR / "logs"

# No cell timeout by default -- training runs can easily take well over the
# 10 minute default nbconvert would otherwise kill a cell at.
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

    # The single most common way this whole script fails: nbconvert --execute
    # runs notebooks through a *registered Jupyter kernel*, which is a
    # separate concept from "whatever python/venv is currently active in
    # your shell". If the kernel you're pointing at isn't the one with
    # numpy/torch/etc. installed (e.g. the default "python3" kernel is some
    # other interpreter), every single run fails identically on
    # `from wwtw_utils import *` with ModuleNotFoundError, only visible if
    # you go dig through a log file. Check this once, up front, instead.
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
        print(f"[error] Kernel '{kernel_name}' can't import numpy/torch -- it's likely not the same "
              f"Python environment as your venv (this is what caused the ModuleNotFoundError you just saw).")
        print(f"        Probe output:\n{probe.stdout}\n{probe.stderr}")
        print(
            "        Fix (with your venv active):\n"
            "          pip install ipykernel --break-system-packages\n"
            f"          python -m ipykernel install --user --name={kernel_name} --display-name \"{kernel_name}\"\n"
            f"        Then re-run with: python run.py --kernel-name {kernel_name}"
        )
        sys.exit(1)


def results_pkl_path(component: str, save_name: str, facility: str) -> Path:
    # Must match the naming save_result() in wwtw_utils.py writes.
    return RESULTS_DIR / f"results_{component}_{save_name}_{facility}.pkl"


def export_script(notebook_path: Path, save_name: str) -> Path:
    """Exports notebook_path to a plain .py script under
    GENERATED_SCRIPTS_DIR, purely as a saved artifact (see module docstring
    for why this isn't what actually gets executed)."""
    GENERATED_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = f"train_{save_name}"
    subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "script",
            str(notebook_path),
            "--output", out_stem,
            "--output-dir", str(GENERATED_SCRIPTS_DIR),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return GENERATED_SCRIPTS_DIR / f"{out_stem}.py"


def run_one(component: str, save_name: str, notebook_name: str, facility: str,
            kernel_name: str, dry_run: bool) -> tuple[bool, str]:
    """Runs one (architecture, facility) combination. Returns (success, message)."""
    notebook_path = Path(notebook_name)
    if not notebook_path.exists():
        return False, f"notebook not found: {notebook_path}"

    dataset_root = f"../cnn_dataset_split_facility_test-{facility}"
    if not Path(dataset_root).exists():
        return False, (
            f"dataset split not found: {dataset_root} "
            f"(run: python split_data_facility.py --test {facility} --component {component})"
        )

    if dry_run:
        return True, f"[dry-run] would train {save_name} on {component}, held-out facility {facility}"

    export_script(notebook_path, save_name)  # saved artifact, see module docstring

    EXECUTED_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    executed_path = EXECUTED_NOTEBOOKS_DIR / f"{component}_{save_name}_{facility}.ipynb"
    log_path = LOGS_DIR / f"{component}_{save_name}_{facility}.log"

    env = os.environ.copy()
    env["DATASET_ROOT"] = dataset_root
    env["COMPONENT"] = component

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

    expected_pkl = results_pkl_path(component, save_name, facility)
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
    parser.add_argument("--kernel-name", default="python3",
                         help="Jupyter kernel name to execute notebooks with (default: python3)")
    parser.add_argument("--force", action="store_true",
                         help="Re-run even if a results_*.pkl already exists for that combination")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would run without training anything")
    args = parser.parse_args()

    check_prerequisites(args.kernel_name)

    models = args.models.split(",") if args.models else list(TRAIN_NOTEBOOKS)
    facilities = args.facilities.split(",") if args.facilities else list(ALL_FACILITIES)

    unknown_models = [m for m in models if m not in TRAIN_NOTEBOOKS]
    if unknown_models:
        parser.error(f"Unknown model(s): {unknown_models}. Choices: {list(TRAIN_NOTEBOOKS)}")
    unknown_facilities = [f for f in facilities if f not in ALL_FACILITIES]
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
        pkl_path = results_pkl_path(args.component, save_name, facility)

        print(f"[{i}/{len(combinations)}] {save_name} | held-out facility: {facility}")

        if pkl_path.exists() and not args.force:
            print(f"    skipped (already exists: {pkl_path}, use --force to re-run)")
            summary.append((save_name, facility, "skipped"))
            continue

        success, message = run_one(
            args.component, save_name, notebook_name, facility,
            args.kernel_name, args.dry_run,
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