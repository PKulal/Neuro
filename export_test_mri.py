"""
export_test_mri.py
------------------
Convenience helper for testing the GUI.

MAIN_Testing1/ holds the JPG slices the model is scored on. The GUI, however,
takes a WHOLE volume (.nii/.nii.gz), and those live scattered across the raw
ABIDEII-* folders at paths like:

    ABIDEII-GU_1\\28752\\session_1\\anat_1\\anat.nii.gz

This script copies the 40 held-out patients' whole MRIs into one flat folder
with self-describing names, so they are easy to pick in a file dialog:

    MAIN_Testing1_WholeMRI/
        Autism_29006.nii.gz
        Healthy_29020.nii.gz
        ...

Nothing here affects training or evaluation -- it only makes manual testing
convenient.

Usage:
    python export_test_mri.py
    python export_test_mri.py --list        # print paths, copy nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_DIR)
MANIFEST = os.path.join(PROJECT_DIR, "dataset_manifest.json")
DEFAULT_OUT = os.path.join(DATA_ROOT, "MAIN_Testing1_WholeMRI")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect held-out whole MRIs")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true",
                    help="only print the source paths")
    args = ap.parse_args()

    if not os.path.exists(MANIFEST):
        sys.exit(f"ERROR: {MANIFEST} not found. Run create_dataset.py first.")

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    test_patients = [p for p in manifest["patients"] if p["split"] == "test"]
    test_patients.sort(key=lambda p: (p["label"], p["sub_id"]))

    if args.list:
        for p in test_patients:
            print(f"{p['label']:<8} {p['sub_id']:<8} {p['mri']}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    print(f"Copying {len(test_patients)} held-out whole MRIs to:\n  {args.out}\n")

    copied = 0
    for p in test_patients:
        src = p["mri"]
        if not os.path.isfile(src):
            print(f"  MISSING {p['sub_id']}: {src}")
            continue
        ext = ".nii.gz" if src.lower().endswith(".nii.gz") else ".nii"
        name = f"{p['label'].capitalize()}_{p['sub_id']}{ext}"
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        print(f"  {name:<28} <- {os.path.relpath(src, DATA_ROOT)}")
        copied += 1

    n_a = sum(1 for p in test_patients if p["label"] == "autism")
    print(f"\nDone: {copied} files ({n_a} Autism, {copied - n_a} Healthy)")
    print("\nThe diagnosis is in each filename so you can check the model's")
    print("answer. These 40 patients were never seen during training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
