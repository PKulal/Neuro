"""
test_model.py
-------------
STEP 10-11 of NeuroConnect AI v2.

Evaluates the trained model on the 20 Autism + 20 Healthy patients held out in
MAIN_Testing1. These patients were never seen during training, and were not
used to pick the aggregation rule or the decision threshold -- both of those
came from the validation patients inside MAIN_Training1.

Two modes:

    python test_model.py                 # full held-out evaluation
    python test_model.py --mri scan.nii  # one whole MRI, patient-level result
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

from predictor import NeuroPredictor, aggregate

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_DIR)

TEST_ROOT = os.path.join(DATA_ROOT, "MAIN_Testing1")
REPORTS_DIR = os.path.join(PROJECT_DIR, "Reports")
SPLIT_PATH = os.path.join(PROJECT_DIR, "train_val_split.json")

AUTISM_DIR_NAME = "A1"
HEALTHY_DIR_NAME = "H1"
CLASS_NAMES = ["Healthy", "Autism"]


# ==========================================================================
# Held-out evaluation
# ==========================================================================
def collect_test_patients(root: str) -> list[dict]:
    patients = []
    for cls_dir, label in ((HEALTHY_DIR_NAME, 0), (AUTISM_DIR_NAME, 1)):
        cdir = os.path.join(root, cls_dir)
        if not os.path.isdir(cdir):
            sys.exit(f"ERROR: missing {cdir}. Run create_dataset.py first.")
        for pname in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, pname)
            if not os.path.isdir(pdir):
                continue
            files = sorted(os.path.join(pdir, f) for f in os.listdir(pdir)
                           if f.lower().endswith(".jpg"))
            if files:
                patients.append({"patient": pname, "label": label, "files": files})
    return patients


def assert_unseen(test_patients: list[dict]) -> None:
    """Fail loudly if any test patient was used during training."""
    if not os.path.exists(SPLIT_PATH):
        print("  WARNING: train_val_split.json not found; cannot verify overlap.")
        return
    with open(SPLIT_PATH, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    seen = set(split["train_patients"]) | set(split["val_patients"])
    overlap = {p["patient"] for p in test_patients} & seen
    if overlap:
        sys.exit(f"ERROR: test patients were seen during training: {sorted(overlap)}")
    print(f"  Leakage check: 0 of {len(test_patients)} test patients "
          f"appear in the training or validation sets.")


def load_patient_slices(files: list[str]) -> list[np.ndarray]:
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imgs.append(img)
    return imgs


def plot_confusion(cm: np.ndarray, path: str, title: str) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES)
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=15)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, auc: float, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5.5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Held-out test ROC (patient-level)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def evaluate(predictor: NeuroPredictor, test_root: str) -> int:
    print("=" * 60)
    print("NeuroConnect AI v2  --  HELD-OUT TEST EVALUATION")
    print("=" * 60)
    print(f"Model      : {predictor.model_path}")
    print(f"Aggregation: {predictor.method}")
    print(f"Threshold  : {predictor.threshold:.4f}")
    print(f"Test data  : {test_root}\n")

    patients = collect_test_patients(test_root)
    n_a = sum(p["label"] for p in patients)
    print(f"  Test patients: {len(patients)} ({n_a} Autism, {len(patients) - n_a} Healthy)")
    assert_unseen(patients)

    print("\nPredicting (patient-level aggregation of slice scores)...")
    rows = []
    for p in patients:
        imgs = load_patient_slices(p["files"])
        if not imgs:
            print(f"  WARNING: no readable slices for {p['patient']}, skipped")
            continue
        slice_probs = predictor.predict_slices(imgs)
        prob = aggregate(slice_probs, predictor.method)
        rows.append({
            "patient": p["patient"],
            "true": p["label"],
            "prob": prob,
            "pred": int(prob >= predictor.threshold),
            "n_slices": len(imgs),
        })

    y_true = np.array([r["true"] for r in rows])
    y_prob = np.array([r["prob"] for r in rows])
    y_pred = np.array([r["pred"] for r in rows])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   zero_division=0, digits=4)

    print("\nPer-patient results:")
    print(f"  {'patient':<20}{'true':>9}{'autism p':>11}{'predicted':>12}")
    for r in sorted(rows, key=lambda x: (x["true"], -x["prob"])):
        mark = " " if r["true"] == r["pred"] else "X"
        print(f"{mark} {r['patient']:<20}{CLASS_NAMES[r['true']]:>9}"
              f"{r['prob']:>11.3f}{CLASS_NAMES[r['pred']]:>12}")

    print("\n" + "=" * 40)
    print("HELD-OUT TEST RESULTS (patient-level)")
    print("=" * 40)
    print(f"Patients        : {len(rows)}")
    print(f"Accuracy        : {acc:.4f}  ({int(acc * len(rows))}/{len(rows)})")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1              : {f1:.4f}")
    print(f"AUC             : {auc:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(f"                 pred_Healthy  pred_Autism")
    print(f"  true_Healthy   {cm[0, 0]:>12}  {cm[0, 1]:>11}")
    print(f"  true_Autism    {cm[1, 0]:>12}  {cm[1, 1]:>11}")
    print("=" * 40)
    print("\n" + report)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    plot_confusion(cm, os.path.join(REPORTS_DIR, "confusion_matrix.png"),
                   "Held-out test confusion matrix (patient-level)")
    plot_roc(y_true, y_prob, auc, os.path.join(REPORTS_DIR, "roc_curve.png"))

    txt_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("NeuroConnect AI v2 - HELD-OUT TEST SET (patient-level)\n")
        fh.write("=" * 58 + "\n")
        fh.write(f"model        : {predictor.model_path}\n")
        fh.write(f"aggregation  : {predictor.method}\n")
        fh.write(f"threshold    : {predictor.threshold:.4f}\n")
        fh.write(f"patients     : {len(rows)} "
                 f"({int(y_true.sum())} Autism, {int((1 - y_true).sum())} Healthy)\n\n")
        fh.write(f"Accuracy : {acc:.4f}\n")
        fh.write(f"Precision: {prec:.4f}\n")
        fh.write(f"Recall   : {rec:.4f}\n")
        fh.write(f"F1       : {f1:.4f}\n")
        fh.write(f"AUC      : {auc:.4f}\n\n")
        fh.write("Confusion matrix (rows = true, cols = predicted)\n")
        fh.write(f"               pred_Healthy  pred_Autism\n")
        fh.write(f"  true_Healthy {cm[0, 0]:>12}  {cm[0, 1]:>11}\n")
        fh.write(f"  true_Autism  {cm[1, 0]:>12}  {cm[1, 1]:>11}\n\n")
        fh.write(report)
        fh.write("\n\nPer-patient scores\n")
        for r in sorted(rows, key=lambda x: (x["true"], -x["prob"])):
            fh.write(f"  {r['patient']:<20} true={CLASS_NAMES[r['true']]:<8}"
                     f"autism_p={r['prob']:.4f}  pred={CLASS_NAMES[r['pred']]}\n")
        fh.write("\nThis is a research prototype. It is NOT a clinically "
                 "validated diagnostic system.\n")

    with open(os.path.join(REPORTS_DIR, "test_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "threshold": predictor.threshold,
            "aggregation": predictor.method,
            "patients": rows,
        }, fh, indent=2)

    print(f"Reports written to: {REPORTS_DIR}")
    if acc >= 0.85:
        print(f"\nTarget of 85% accuracy MET on the held-out set ({acc:.1%}).")
    else:
        print(f"\nTarget of 85% accuracy NOT met: actual held-out accuracy is "
              f"{acc:.1%}. This is the real measured number and has not been "
              f"adjusted.")
    return 0


# ==========================================================================
# Single whole-MRI mode
# ==========================================================================
def predict_single(predictor: NeuroPredictor, mri_path: str) -> int:
    result = predictor.predict_mri(mri_path)

    if not result["valid"]:
        print("\nInvalid MRI file.")
        print("Please select a valid whole-brain MRI (.nii or .nii.gz).")
        print(f"Reason: {result['reason']}")
        return 1

    print()
    if result["label"] == "AUTISM":
        print("=" * 32)
        print("AUTISM DETECTED")
        print("=" * 32)
        print(f"Autism Probability: {result['autism_probability'] * 100:.1f}%")
    else:
        print("=" * 32)
        print("HEALTHY")
        print("=" * 32)
        print(f"Healthy Probability: {result['healthy_probability'] * 100:.1f}%")

    print(f"Confidence: {result['confidence_percent']:.1f}% "
          f"({result['confidence']})")

    print()
    print("These probabilities are the model's estimated scores for its Autism")
    print("and Healthy classes. They are not a measure of how much autism a")
    print("person has, and they are not a clinical diagnosis.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate NeuroConnect AI v2")
    ap.add_argument("--test-root", default=TEST_ROOT)
    ap.add_argument("--model", default=None, help="path to a .keras model")
    ap.add_argument("--mri", default=None,
                    help="predict a single whole MRI instead of the test set")
    args = ap.parse_args()

    predictor = NeuroPredictor(model_path=args.model)

    if args.mri:
        return predict_single(predictor, args.mri)
    return evaluate(predictor, args.test_root)


if __name__ == "__main__":
    sys.exit(main())
