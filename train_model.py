"""
train_model.py
--------------
STEP 7-9 of NeuroConnect AI v2.

Trains EfficientNetB3 on the slice JPGs produced by create_dataset.py.

Design rules enforced here:
  * The 20+20 held-out testing patients in MAIN_Testing1 are NEVER opened.
  * The training patients are split into train/validation BY PATIENT, so no
    patient contributes slices to both sides.
  * Two-stage transfer learning: frozen backbone first, then a partial
    unfreeze at a much smaller learning rate.
  * The patient-level aggregation rule and the decision threshold are chosen
    on the VALIDATION patients only.

Usage:
    python train_model.py
    python train_model.py --stage1-epochs 20 --stage2-epochs 12
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from neuro_utils import IMG_SIZE

# ==========================================================================
# CONFIGURATION
# ==========================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(PROJECT_DIR)

TRAIN_ROOT = os.path.join(DATA_ROOT, "MAIN_Training1")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "Reports")

AUTISM_DIR_NAME = "A1"
HEALTHY_DIR_NAME = "H1"
CLASS_NAMES = ["Healthy", "Autism"]     # index 0, 1 -- label 1 == Autism

VAL_PATIENT_FRACTION = 0.20
BATCH_SIZE = 32
SEED = 42

STAGE1_EPOCHS = 20
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 12
STAGE2_LR = 1e-5

# Fine-tune only the upper blocks. Unfreezing the whole backbone is ~11x
# slower on CPU and overfits badly on a few hundred patients.
UNFREEZE_FROM = "block6a"

DROPOUT = 0.4
LABEL_SMOOTHING = 0.05

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "autism_detector.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
MODEL_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
SPLIT_PATH = os.path.join(PROJECT_DIR, "train_val_split.json")


# ==========================================================================
# Reproducibility
# ==========================================================================
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


# ==========================================================================
# Patient-wise data collection
# ==========================================================================
def collect_patients(root: str) -> list[dict]:
    """Read MAIN_Training1 into a list of patients with their slice paths."""
    patients = []
    for cls_dir, label in ((HEALTHY_DIR_NAME, 0), (AUTISM_DIR_NAME, 1)):
        cdir = os.path.join(root, cls_dir)
        if not os.path.isdir(cdir):
            sys.exit(f"ERROR: missing {cdir}. Run create_dataset.py first.")
        for pname in sorted(os.listdir(cdir)):
            pdir = os.path.join(cdir, pname)
            if not os.path.isdir(pdir):
                continue
            files = sorted(
                os.path.join(pdir, f) for f in os.listdir(pdir)
                if f.lower().endswith(".jpg")
            )
            if files:
                patients.append({
                    "patient": pname,
                    "label": label,
                    "files": files,
                })
    if not patients:
        sys.exit(f"ERROR: no patients found under {root}")
    return patients


def split_by_patient(patients: list[dict], val_fraction: float, seed: int):
    """Class-stratified split at PATIENT level -- never at slice level."""
    rng = random.Random(seed)
    train, val = [], []
    for label in (0, 1):
        group = [p for p in patients if p["label"] == label]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_fraction)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    overlap = {p["patient"] for p in train} & {p["patient"] for p in val}
    if overlap:
        sys.exit(f"ERROR: patient leakage between train and validation: {overlap}")
    return train, val


def flatten(patients: list[dict]):
    """Patient list -> (paths, labels, patient_ids) at slice level."""
    paths, labels, pids = [], [], []
    for p in patients:
        for f in p["files"]:
            paths.append(f)
            labels.append(p["label"])
            pids.append(p["patient"])
    return paths, np.array(labels, dtype="float32"), pids


# ==========================================================================
# tf.data input pipeline
# ==========================================================================
def _decode(path, label):
    """JPG -> (IMG_SIZE, IMG_SIZE, 1) uint8. Cached before augmentation."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return tf.cast(img, tf.uint8), label


def build_augmenter() -> keras.Sequential:
    """Mild, anatomy-preserving augmentation.

    Horizontal flip is valid here because the brain is roughly symmetric
    about the midline. Rotations/zooms stay small so anatomy is not warped.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.04, fill_mode="constant", seed=SEED),
        layers.RandomZoom(0.08, fill_mode="constant", seed=SEED),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=SEED),
        layers.RandomContrast(0.15, seed=SEED),
    ], name="augmentation")


def make_dataset(paths, labels, training: bool, augmenter=None,
                 batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Build the input pipeline.

    Decoded slices are cached in RAM once (~270 MB), so the JPGs are read and
    decoded a single time for the whole run instead of once per epoch.
    EfficientNet carries its own normalisation, so images stay on the 0-255
    scale and are simply tiled to 3 channels.
    """
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    # Augmentation layers operate on floats, so cast before augmenting
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (tf.repeat(x, 3, axis=-1), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ==========================================================================
# Model
# ==========================================================================
def build_model() -> tuple[keras.Model, keras.Model]:
    """EfficientNetB3 + ImageNet weights + binary classification head."""
    base = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="mri_slice")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="autism_score")(x)

    model = keras.Model(inputs, outputs, name="NeuroConnect_EfficientNetB3")
    return model, base


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def unfreeze_top(base: keras.Model, from_block: str) -> int:
    """Unfreeze the upper blocks, keeping BatchNorm layers frozen.

    Re-estimating BatchNorm statistics on a small medical dataset is a common
    cause of fine-tuning collapse, so those layers stay in inference mode.
    """
    base.trainable = True
    reached = False
    n_trainable = 0
    for layer in base.layers:
        if layer.name.startswith(from_block):
            reached = True
        if not reached or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            n_trainable += 1
    return n_trainable


# ==========================================================================
# Patient-level aggregation
# ==========================================================================
def aggregate(probs: np.ndarray, method: str) -> float:
    """Combine a patient's slice probabilities into one patient probability."""
    if method == "mean":
        return float(np.mean(probs))
    if method == "median":
        return float(np.median(probs))
    if method == "trimmed_mean":
        # Drop the most extreme 20% at each end -- robust to a few odd slices
        p = np.sort(probs)
        k = int(len(p) * 0.2)
        core = p[k:len(p) - k] if len(p) - 2 * k > 0 else p
        return float(np.mean(core))
    if method == "top_k_mean":
        # Average the most autism-like third of slices: any focal signal is
        # unlikely to be present on every single slice
        p = np.sort(probs)[::-1]
        k = max(1, len(p) // 3)
        return float(np.mean(p[:k]))
    if method == "mean_logit":
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        return float(1.0 / (1.0 + np.exp(-np.mean(np.log(p / (1 - p))))))
    raise ValueError(f"unknown aggregation method: {method}")


AGGREGATIONS = ["mean", "median", "trimmed_mean", "top_k_mean", "mean_logit"]


def patient_probabilities(model: keras.Model, patients: list[dict],
                          method: str = "mean"):
    """Predict every slice, then aggregate per patient."""
    paths, labels, pids = flatten(patients)
    ds = make_dataset(paths, labels, training=False)
    slice_probs = model.predict(ds, verbose=0).ravel()

    by_patient = defaultdict(list)
    for pid, prob in zip(pids, slice_probs):
        by_patient[pid].append(prob)

    order = [p["patient"] for p in patients]
    y_true = np.array([p["label"] for p in patients], dtype=int)
    y_prob = np.array([aggregate(np.array(by_patient[p]), method) for p in order])
    return y_true, y_prob, by_patient, order


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising Youden's J on the validation patients."""
    candidates = np.unique(np.round(np.concatenate([y_prob, [0.5]]), 4))
    best_t, best_j = 0.5, -1.0
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


# ==========================================================================
# Reporting
# ==========================================================================
def plot_history(hist: dict, path: str, key: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(hist[key], label=f"train {key}")
    plt.plot(hist[f"val_{key}"], label=f"validation {key}")
    if hist.get("stage2_start"):
        plt.axvline(hist["stage2_start"] - 0.5, color="grey", ls="--",
                    label="stage 2 (fine-tune)")
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


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
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Train EfficientNetB3 on MRI slices")
    ap.add_argument("--train-root", default=TRAIN_ROOT)
    ap.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    ap.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seeds(args.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("NeuroConnect AI v2  --  TRAINING (EfficientNetB3)")
    print("=" * 60)
    print(f"TensorFlow {tf.__version__}")
    print(f"Training data: {args.train_root}")
    print("NOTE: MAIN_Testing1 is not opened by this script.\n")

    # ---- patients ------------------------------------------------------
    patients = collect_patients(args.train_root)
    train_p, val_p = split_by_patient(patients, VAL_PATIENT_FRACTION, args.seed)

    tr_paths, tr_labels, _ = flatten(train_p)
    va_paths, va_labels, _ = flatten(val_p)

    print(f"Patients      : {len(patients)} "
          f"({sum(p['label'] for p in patients)} Autism, "
          f"{len(patients) - sum(p['label'] for p in patients)} Healthy)")
    print(f"  training    : {len(train_p)} patients / {len(tr_paths)} slices")
    print(f"  validation  : {len(val_p)} patients / {len(va_paths)} slices")
    print("  patient overlap: 0 (verified)\n")

    with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "seed": args.seed,
            "train_patients": sorted(p["patient"] for p in train_p),
            "val_patients": sorted(p["patient"] for p in val_p),
        }, fh, indent=2)

    # ---- data ----------------------------------------------------------
    augmenter = build_augmenter()
    train_ds = make_dataset(tr_paths, tr_labels, True, augmenter, args.batch_size)
    val_ds = make_dataset(va_paths, va_labels, False, None, args.batch_size)

    n_pos = float(tr_labels.sum())
    n_neg = float(len(tr_labels) - n_pos)
    class_weight = {
        0: len(tr_labels) / (2.0 * n_neg),
        1: len(tr_labels) / (2.0 * n_pos),
    }
    print(f"Class weights : {{0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}}}\n")

    # ---- model ---------------------------------------------------------
    model, base = build_model()
    print(f"EfficientNetB3 parameters: {base.count_params():,}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1),
    ]

    history: dict[str, list] = defaultdict(list)

    # ---- stage 1: frozen backbone --------------------------------------
    print("\n" + "-" * 60)
    print(f"STAGE 1  |  frozen backbone, lr={STAGE1_LR}")
    print("-" * 60)
    compile_model(model, STAGE1_LR)
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage1_epochs,
                   class_weight=class_weight, callbacks=callbacks, verbose=2)
    print(f"Stage 1 finished in {(time.time() - t0) / 60:.1f} min")
    for k, v in h1.history.items():
        history[k].extend(v)
    stage2_start = len(h1.history["loss"])

    # ---- stage 2: partial unfreeze -------------------------------------
    if args.stage2_epochs > 0:
        n_trainable = unfreeze_top(base, UNFREEZE_FROM)
        print("\n" + "-" * 60)
        print(f"STAGE 2  |  unfrozen from '{UNFREEZE_FROM}' "
              f"({n_trainable} layers, BatchNorm kept frozen), lr={STAGE2_LR}")
        print("-" * 60)
        compile_model(model, STAGE2_LR)
        t0 = time.time()
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage2_epochs,
                       class_weight=class_weight, callbacks=callbacks, verbose=2)
        print(f"Stage 2 finished in {(time.time() - t0) / 60:.1f} min")
        for k, v in h2.history.items():
            history[k].extend(v)
    else:
        stage2_start = 0

    history["stage2_start"] = stage2_start

    # ---- curves --------------------------------------------------------
    plot_history(history, os.path.join(REPORTS_DIR, "accuracy.png"),
                 "accuracy", "Slice-level accuracy")
    plot_history(history, os.path.join(REPORTS_DIR, "loss.png"),
                 "loss", "Binary cross-entropy loss")

    # ---- choose aggregation + threshold on VALIDATION patients ---------
    print("\n" + "-" * 60)
    print("Selecting patient-level aggregation on VALIDATION patients")
    print("(the held-out test set is not involved in this choice)")
    print("-" * 60)

    if os.path.exists(BEST_MODEL_PATH):
        model = keras.models.load_model(BEST_MODEL_PATH)
        print(f"Loaded best checkpoint: {BEST_MODEL_PATH}")

    results = {}
    print(f"\n{'method':<15}{'AUC':>8}{'accuracy':>11}{'threshold':>11}")
    for method in AGGREGATIONS:
        y_true, y_prob, _, _ = patient_probabilities(model, val_p, method)
        thr = best_threshold(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        acc = accuracy_score(y_true, (y_prob >= thr).astype(int))
        results[method] = {"auc": float(auc), "accuracy": float(acc),
                           "threshold": thr}
        print(f"{method:<15}{auc:>8.3f}{acc:>11.3f}{thr:>11.3f}")

    best_method = max(results, key=lambda m: (results[m]["auc"], results[m]["accuracy"]))
    best_thr = results[best_method]["threshold"]
    print(f"\nSelected aggregation: {best_method}  (threshold {best_thr:.3f})")

    # ---- validation report ---------------------------------------------
    y_true, y_prob, _, _ = patient_probabilities(model, val_p, best_method)
    y_pred = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nVALIDATION (patient-level):")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  AUC      : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Confusion matrix (rows true, cols pred):\n{cm}")

    plot_confusion(cm, os.path.join(REPORTS_DIR, "confusion_matrix_val.png"),
                   "Validation confusion matrix (patient-level)")
    with open(os.path.join(REPORTS_DIR, "classification_report_val.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("NeuroConnect AI v2 - VALIDATION patients (patient-level)\n")
        fh.write(f"aggregation={best_method}  threshold={best_thr:.4f}\n\n")
        fh.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                       zero_division=0))
        fh.write(f"\nAUC: {roc_auc_score(y_true, y_prob):.4f}\n")

    # ---- save ----------------------------------------------------------
    model.save(FINAL_MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump({"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]}, fh, indent=2)

    config = {
        "model": "EfficientNetB3",
        "weights_init": "imagenet",
        "image_size": IMG_SIZE,
        "channels": 3,
        "input_scale": "0-255 (EfficientNet applies its own normalisation)",
        "positive_class": "Autism",
        "aggregation": best_method,
        "threshold": best_thr,
        "aggregation_comparison": results,
        "validation": {
            "patients": len(val_p),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": cm.tolist(),
        },
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "stage1_epochs_run": stage2_start,
            "stage1_lr": STAGE1_LR,
            "stage2_epochs_run": len(history["loss"]) - stage2_start,
            "stage2_lr": STAGE2_LR,
            "unfreeze_from": UNFREEZE_FROM,
            "train_patients": len(train_p),
            "val_patients": len(val_p),
        },
    }
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model : {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"Config     : {MODEL_CONFIG_PATH}")
    print(f"Reports    : {REPORTS_DIR}")
    print("\nNext: python test_model.py   (evaluates the untouched 20+20 patients)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
