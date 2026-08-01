"""
create_dataset.py
-----------------
STEP 4-6 of NeuroConnect AI v2.

Turns the raw ABIDE-II archives into a patient-wise JPG dataset:

    raw .nii.gz volumes + phenotypic CSVs
                |
        match SUB_ID -> DX_GROUP
                |
        validate every patient
                |
    hold out 20 Autism + 20 Healthy patients
                |
    MAIN_Training1/{A1,H1}/Patient_<id>/slice_###.jpg
    MAIN_Testing1/{A1,H1}/Patient_<id>/slice_###.jpg

The split is at PATIENT level. Every slice of a patient lands in exactly one
of training or testing -- never both.

Usage:
    python create_dataset.py
    python create_dataset.py --raw-root "D:\\data" --out-root "D:\\out"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from neuro_utils import (
    IMG_SIZE,
    SLICES_PER_PATIENT,
    find_mri_file,
    mri_to_slices,
)

# ==========================================================================
# CONFIGURATION -- edit these if the data lives elsewhere on your machine.
# Paths are NOT hard-coded to any one computer: everything is resolved
# relative to RAW_ROOT, and can be overridden on the command line.
# ==========================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder holding the ABIDEII-* site directories and the ABIDEII-*.csv files
RAW_ROOT = os.path.dirname(PROJECT_DIR)
PHENO_ROOT = RAW_ROOT              # phenotypic CSVs (same folder by default)
OUT_ROOT = RAW_ROOT                # where MAIN_Training1 / MAIN_Testing1 go

TRAIN_DIR_NAME = "MAIN_Training1"
TEST_DIR_NAME = "MAIN_Testing1"
AUTISM_DIR_NAME = "A1"
HEALTHY_DIR_NAME = "H1"

TEST_PATIENTS_PER_CLASS = 20       # 20 Autism + 20 Healthy held out
JPEG_QUALITY = 95
SEED = 42

# DX_GROUP encoding used by ABIDE
DX_AUTISM = 1
DX_HEALTHY = 2

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")


# ==========================================================================
# Phenotypic files
# ==========================================================================
def read_csv_any_encoding(path: str) -> list[dict]:
    """Read a CSV trying several encodings.

    ABIDE phenotypic files are not consistently UTF-8 (GU_1 is Latin-1), and
    some use bare CR line endings, so newlines are normalised too.
    """
    raw = open(path, "rb").read()

    text = None
    used = None
    for enc in CSV_ENCODINGS:
        try:
            text = raw.decode(enc)
            used = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin-1", errors="replace")
        used = "latin-1(replace)"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.DictReader(text.splitlines()))
    print(f"  {os.path.basename(path):<26} encoding={used:<14} rows={len(rows)}")
    return rows


def load_phenotypes(pheno_root: str) -> dict[str, dict]:
    """Build {SUB_ID: {label, dx, site}} from every ABIDEII-*.csv found.

    Patient IDs are never typed by hand -- they come only from the CSVs.
    """
    csv_paths = sorted(
        os.path.join(pheno_root, f)
        for f in os.listdir(pheno_root)
        if f.lower().startswith("abideii-") and f.lower().endswith(".csv")
    )
    if not csv_paths:
        sys.exit(f"ERROR: no ABIDEII-*.csv phenotypic files found in {pheno_root}")

    print(f"\nReading phenotypic files from: {pheno_root}")
    table: dict[str, dict] = {}
    for path in csv_paths:
        for row in read_csv_any_encoding(path):
            # Header names occasionally carry stray whitespace
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            sub_id = row.get("SUB_ID", "")
            dx_raw = row.get("DX_GROUP", "")
            if not sub_id or not dx_raw:
                continue
            try:
                dx = int(float(dx_raw))
            except ValueError:
                continue
            if dx not in (DX_AUTISM, DX_HEALTHY):
                continue

            table[sub_id] = {
                "dx": dx,
                "label": "autism" if dx == DX_AUTISM else "healthy",
                "site": row.get("SITE_ID", "UNKNOWN"),
            }

    n_a = sum(1 for v in table.values() if v["dx"] == DX_AUTISM)
    n_h = len(table) - n_a
    print(f"  -> {len(table)} patients with a valid diagnosis "
          f"({n_a} Autism, {n_h} Healthy)")
    return table


# ==========================================================================
# Raw MRI discovery
# ==========================================================================
def discover_patients(raw_root: str, pheno: dict[str, dict]) -> list[dict]:
    """Walk the ABIDEII-* site folders and pair each patient dir with its label.

    Matching is done on SUB_ID, not on folder names, so a site directory whose
    name differs from its CSV name (e.g. ABIDEII-STANFORD holding the subjects
    listed in ABIDEII-SU_2.csv) is resolved automatically.
    """
    site_dirs = sorted(
        os.path.join(raw_root, d)
        for d in os.listdir(raw_root)
        if d.lower().startswith("abideii-") and os.path.isdir(os.path.join(raw_root, d))
    )
    if not site_dirs:
        sys.exit(f"ERROR: no ABIDEII-* dataset folders found in {raw_root}")

    print(f"\nScanning raw MRI folders in: {raw_root}")
    patients: list[dict] = []
    unmatched: list[str] = []

    for site_dir in site_dirs:
        found = 0
        for entry in sorted(os.listdir(site_dir)):
            pdir = os.path.join(site_dir, entry)
            if not os.path.isdir(pdir):
                continue
            sub_id = entry.strip()
            if sub_id not in pheno:
                unmatched.append(f"{os.path.basename(site_dir)}/{sub_id}")
                continue
            patients.append({
                "sub_id": sub_id,
                "dir": pdir,
                "site_dir": os.path.basename(site_dir),
                **pheno[sub_id],
            })
            found += 1
        print(f"  {os.path.basename(site_dir):<22} {found} matched patients")

    if unmatched:
        print(f"  NOTE: {len(unmatched)} patient folders had no phenotypic entry: "
              f"{', '.join(unmatched[:6])}{' ...' if len(unmatched) > 6 else ''}")

    # A subject ID must never appear twice -- that would be silent leakage
    seen = defaultdict(list)
    for p in patients:
        seen[p["sub_id"]].append(p["site_dir"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        sys.exit(f"ERROR: duplicate patient IDs across sites: {dupes}")

    return patients


# ==========================================================================
# Slice extraction with per-patient validation
# ==========================================================================
def process_patients(patients: list[dict]):
    """Extract slices for every patient. Returns (valid, skipped).

    A patient that fails any check is skipped with a printed reason; it never
    aborts the run. Slices are kept in memory as encoded JPGs so that nothing
    is written to disk before the train/test split is decided.
    """
    print(f"\nExtracting brain slices ({SLICES_PER_PATIENT} per patient, "
          f"{IMG_SIZE}x{IMG_SIZE})")
    valid, skipped = [], []

    for p in tqdm(patients, desc="  patients", ncols=78, file=sys.stdout):
        mri_path = find_mri_file(p["dir"])
        if mri_path is None:
            skipped.append((p["sub_id"], "MRI not found"))
            continue

        slices, reason = mri_to_slices(mri_path, n_slices=SLICES_PER_PATIENT)
        if not slices:
            skipped.append((p["sub_id"], reason))
            continue

        encoded = []
        for img in slices:
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                encoded.append(buf.tobytes())
        if not encoded:
            skipped.append((p["sub_id"], "no valid JPGs could be encoded"))
            continue

        valid.append({**p, "mri": mri_path, "jpgs": encoded})

    if skipped:
        print(f"\n  Skipped {len(skipped)} patient(s):")
        for sub_id, reason in skipped:
            print(f"    Skipped Patient: {sub_id}")
            print(f"    Reason: {reason}")

    return valid, skipped


# ==========================================================================
# Patient-wise split
# ==========================================================================
def split_patients(valid: list[dict], rng: random.Random):
    """Hold out TEST_PATIENTS_PER_CLASS Autism + the same number of Healthy.

    Selection is spread across sites so the held-out set is not dominated by a
    single scanner, then shuffled with a fixed seed for reproducibility.
    """
    def pick(label: str) -> list[dict]:
        pool = [p for p in valid if p["label"] == label]
        by_site: dict[str, list[dict]] = defaultdict(list)
        for p in pool:
            by_site[p["site_dir"]].append(p)
        for group in by_site.values():
            rng.shuffle(group)

        # Round-robin across sites until the quota is filled
        chosen, sites = [], sorted(by_site)
        while len(chosen) < TEST_PATIENTS_PER_CLASS and any(by_site[s] for s in sites):
            for s in sites:
                if by_site[s] and len(chosen) < TEST_PATIENTS_PER_CLASS:
                    chosen.append(by_site[s].pop())
        return chosen

    test = pick("autism") + pick("healthy")
    test_ids = {p["sub_id"] for p in test}
    train = [p for p in valid if p["sub_id"] not in test_ids]

    for p in test:
        p["split"] = "test"
    for p in train:
        p["split"] = "train"

    return train, test


# ==========================================================================
# Writing
# ==========================================================================
def write_split(patients: list[dict], root: str) -> int:
    """Write Patient_<id>/slice_###.jpg folders. Returns image count."""
    total = 0
    for p in patients:
        cls_dir = AUTISM_DIR_NAME if p["label"] == "autism" else HEALTHY_DIR_NAME
        pdir = os.path.join(root, cls_dir, f"Patient_{p['sub_id']}")
        os.makedirs(pdir, exist_ok=True)
        for i, blob in enumerate(p["jpgs"], start=1):
            with open(os.path.join(pdir, f"slice_{i:03d}.jpg"), "wb") as fh:
                fh.write(blob)
            total += 1
    return total


def verify_output(train_root: str, test_root: str) -> dict:
    """Re-read what was actually written and prove the split is clean."""
    def scan(root: str):
        out = {}
        for cls in (AUTISM_DIR_NAME, HEALTHY_DIR_NAME):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                out[cls] = {}
                continue
            out[cls] = {
                d: sorted(f for f in os.listdir(os.path.join(cdir, d))
                          if f.lower().endswith(".jpg"))
                for d in sorted(os.listdir(cdir))
                if os.path.isdir(os.path.join(cdir, d))
            }
        return out

    tr, te = scan(train_root), scan(test_root)
    tr_ids = {p for cls in tr.values() for p in cls}
    te_ids = {p for cls in te.values() for p in cls}
    overlap = tr_ids & te_ids

    empty = [p for side in (tr, te) for cls in side.values()
             for p, files in cls.items() if not files]

    # Confirm every written JPG is readable and the right size
    bad = []
    for root, side in ((train_root, tr), (test_root, te)):
        for cls, pats in side.items():
            for pat, files in pats.items():
                for f in files:
                    img = cv2.imread(os.path.join(root, cls, pat, f),
                                     cv2.IMREAD_GRAYSCALE)
                    if img is None or img.shape != (IMG_SIZE, IMG_SIZE):
                        bad.append(os.path.join(cls, pat, f))

    return {
        "train_autism": len(tr.get(AUTISM_DIR_NAME, {})),
        "train_healthy": len(tr.get(HEALTHY_DIR_NAME, {})),
        "test_autism": len(te.get(AUTISM_DIR_NAME, {})),
        "test_healthy": len(te.get(HEALTHY_DIR_NAME, {})),
        "train_images": sum(len(f) for cls in tr.values() for f in cls.values()),
        "test_images": sum(len(f) for cls in te.values() for f in cls.values()),
        "overlap": sorted(overlap),
        "empty_patients": empty,
        "invalid_jpgs": bad,
    }


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Build the NeuroConnect AI v2 dataset")
    ap.add_argument("--raw-root", default=RAW_ROOT,
                    help="folder containing the ABIDEII-* site directories")
    ap.add_argument("--pheno-root", default=None,
                    help="folder containing the ABIDEII-*.csv files "
                         "(defaults to --raw-root)")
    ap.add_argument("--out-root", default=OUT_ROOT,
                    help="where MAIN_Training1 / MAIN_Testing1 are created")
    ap.add_argument("--slices", type=int, default=SLICES_PER_PATIENT,
                    help="slices extracted per patient")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    raw_root = os.path.abspath(args.raw_root)
    pheno_root = os.path.abspath(args.pheno_root or args.raw_root)
    out_root = os.path.abspath(args.out_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 60)
    print("NeuroConnect AI v2  --  DATASET CREATION")
    print("=" * 60)
    print(f"Raw MRI root   : {raw_root}")
    print(f"Phenotypic root: {pheno_root}")
    print(f"Output root    : {out_root}")
    print(f"Seed           : {args.seed}")

    pheno = load_phenotypes(pheno_root)
    patients = discover_patients(raw_root, pheno)
    if not patients:
        sys.exit("ERROR: no patients could be matched to the phenotypic files")

    valid, skipped = process_patients(patients)

    n_a = sum(1 for p in valid if p["label"] == "autism")
    n_h = len(valid) - n_a
    print(f"\nValid patients: {len(valid)}  ({n_a} Autism, {n_h} Healthy)")
    if n_a < TEST_PATIENTS_PER_CLASS or n_h < TEST_PATIENTS_PER_CLASS:
        sys.exit(f"ERROR: need at least {TEST_PATIENTS_PER_CLASS} valid patients "
                 f"per class to build the held-out test set")

    train, test = split_patients(valid, rng)

    train_root = os.path.join(out_root, TRAIN_DIR_NAME)
    test_root = os.path.join(out_root, TEST_DIR_NAME)
    for root in (train_root, test_root):
        if os.path.isdir(root):
            shutil.rmtree(root)
        for cls in (AUTISM_DIR_NAME, HEALTHY_DIR_NAME):
            os.makedirs(os.path.join(root, cls), exist_ok=True)

    print(f"\nWriting {TRAIN_DIR_NAME} and {TEST_DIR_NAME} ...")
    write_split(train, train_root)
    write_split(test, test_root)

    stats = verify_output(train_root, test_root)

    # Manifest: the record of exactly which patient went where
    manifest = {
        "seed": args.seed,
        "image_size": IMG_SIZE,
        "slices_per_patient": args.slices,
        "train_root": train_root,
        "test_root": test_root,
        "patients": [
            {k: p[k] for k in ("sub_id", "label", "dx", "site", "site_dir", "split")}
            | {"n_slices": len(p["jpgs"]), "mri": p["mri"]}
            for p in sorted(valid, key=lambda x: x["sub_id"])
        ],
        "skipped": [{"sub_id": s, "reason": r} for s, r in skipped],
    }
    manifest_path = os.path.join(PROJECT_DIR, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print("=" * 40)
    print("DATASET CREATION COMPLETE")
    print("=" * 40)
    print(f"Autism Training Patients  : {stats['train_autism']}")
    print(f"Healthy Training Patients : {stats['train_healthy']}")
    print(f"Autism Testing Patients   : {stats['test_autism']}")
    print(f"Healthy Testing Patients  : {stats['test_healthy']}")
    print(f"Training Images           : {stats['train_images']}")
    print(f"Testing Images            : {stats['test_images']}")
    print(f"Skipped Patients          : {len(skipped)}")
    print(f"Patient Leakage           : {len(stats['overlap'])}")
    print(f"Empty Patient Folders     : {len(stats['empty_patients'])}")
    print(f"Invalid JPGs              : {len(stats['invalid_jpgs'])}")
    print("=" * 40)
    print(f"Manifest: {manifest_path}")

    failures = []
    if stats["overlap"]:
        failures.append(f"patient leakage: {stats['overlap']}")
    if stats["empty_patients"]:
        failures.append(f"empty folders: {stats['empty_patients']}")
    if stats["invalid_jpgs"]:
        failures.append(f"invalid JPGs: {stats['invalid_jpgs'][:5]}")
    if stats["test_autism"] != TEST_PATIENTS_PER_CLASS:
        failures.append(f"expected {TEST_PATIENTS_PER_CLASS} autism test patients, "
                        f"got {stats['test_autism']}")
    if stats["test_healthy"] != TEST_PATIENTS_PER_CLASS:
        failures.append(f"expected {TEST_PATIENTS_PER_CLASS} healthy test patients, "
                        f"got {stats['test_healthy']}")

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nVALIDATION PASSED - dataset is ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
