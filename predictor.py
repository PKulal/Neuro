"""
predictor.py
------------
The single place where "model + aggregation rule + threshold" turn into a
patient-level answer. test_model.py and app.py both use it, so the CLI and the
GUI can never disagree about how a prediction is made.
"""

from __future__ import annotations

import json
import os

import numpy as np

from neuro_utils import IMG_SIZE, SLICES_PER_PATIENT, mri_to_slices

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "autism_detector.keras")
MODEL_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
# Inference-only model shipped with the repository (see export_model.py)
EXPORT_MODEL_PATH = os.path.join(MODELS_DIR, "neuroconnect_model.keras")


def aggregate(probs: np.ndarray, method: str) -> float:
    """Combine slice probabilities into one patient probability.

    Kept identical to the version used during training so that the rule
    selected on the validation patients is the rule applied at predict time.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        return float(np.mean(probs))
    if method == "median":
        return float(np.median(probs))
    if method == "trimmed_mean":
        p = np.sort(probs)
        k = int(len(p) * 0.2)
        core = p[k:len(p) - k] if len(p) - 2 * k > 0 else p
        return float(np.mean(core))
    if method == "top_k_mean":
        p = np.sort(probs)[::-1]
        k = max(1, len(p) // 3)
        return float(np.mean(p[:k]))
    if method == "mean_logit":
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        return float(1.0 / (1.0 + np.exp(-np.mean(np.log(p / (1 - p))))))
    raise ValueError(f"unknown aggregation method: {method}")


def confidence(slice_probs, patient_prob: float,
               threshold: float) -> tuple[float, str]:
    """How much the model's own evidence backs the verdict it just gave.

    Returns (percent, band).

    The percentage is SLICE AGREEMENT: the share of the patient's brain slices
    whose individual scores land on the same side of the decision threshold as
    the final answer. 25 slices voting 21-4 gives 84%. This is a real measured
    quantity, and deliberately not a second copy of the class probability --
    it answers "how consistent was the evidence", which the probability alone
    does not tell you.

    The band downgrades on the weaker of two signals:
      * agreement  -- were the slices consistent?
      * margin     -- did the aggregate score clear the threshold decisively,
                      or only just scrape past it?
    A verdict where every slice agrees but the score sits a hair above the
    threshold is not a confident verdict, and neither is a decisive score
    built on slices that flatly contradicted each other.

    None of this is a statement of clinical certainty.
    """
    probs = np.asarray(slice_probs, dtype=np.float64)
    verdict = patient_prob >= threshold
    agreement = float(np.mean((probs >= threshold) == verdict)) if probs.size else 0.0
    margin = abs(patient_prob - threshold)

    # Agreement below 50% means most slices contradict the verdict, which can
    # only happen if the threshold is not on the same scale as an individual
    # slice score (see SCALE_COMPATIBLE in train_model.py). Reporting it as
    # "confidence" would be nonsense, so fall back to measuring each slice
    # against the patient's own aggregate and force the Low band.
    if agreement < 0.5 and probs.size:
        agreement = float(np.mean((probs >= patient_prob) == verdict))
        return round(max(agreement, 1.0 - agreement) * 100.0, 1), "Low"

    if agreement >= 0.85:
        agree_band = 2
    elif agreement >= 0.70:
        agree_band = 1
    else:
        agree_band = 0

    if margin >= 0.25:
        margin_band = 2
    elif margin >= 0.10:
        margin_band = 1
    else:
        margin_band = 0

    band = ("Low", "Moderate", "High")[min(agree_band, margin_band)]
    return round(agreement * 100.0, 1), band


class NeuroPredictor:
    """Loads the trained model once, then answers whole-MRI queries."""

    def __init__(self, model_path: str | None = None,
                 config_path: str = MODEL_CONFIG_PATH):
        # Imported lazily so that importing this module stays cheap
        from tensorflow import keras

        if model_path is None:
            # Prefer the locally trained checkpoints; fall back to the exported
            # model that ships with the repository, so a fresh clone can
            # predict without retraining.
            for candidate in (BEST_MODEL_PATH, FINAL_MODEL_PATH, EXPORT_MODEL_PATH):
                if os.path.exists(candidate):
                    model_path = candidate
                    break
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found in {MODELS_DIR}. Either run "
                f"train_model.py, or place a shared "
                f"{os.path.basename(EXPORT_MODEL_PATH)} there."
            )

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        self.config = cfg
        self.method = cfg.get("aggregation", "mean")
        self.threshold = float(cfg.get("threshold", 0.5))

    # ------------------------------------------------------------------
    def predict_slices(self, slices: list[np.ndarray],
                       batch_size: int = 32) -> np.ndarray:
        """Slice images -> per-slice autism probabilities."""
        arr = np.stack(slices).astype("float32")          # (n, H, W)
        arr = np.repeat(arr[..., None], 3, axis=-1)       # -> 3 channels
        return self.model.predict(arr, batch_size=batch_size, verbose=0).ravel()

    def predict_mri(self, path: str) -> dict:
        """Whole .nii/.nii.gz volume -> one patient-level result.

        Returns a dict with 'valid'. When invalid, 'reason' explains why and
        NO prediction is produced.
        """
        if not os.path.isfile(path):
            return {"valid": False, "reason": "file does not exist"}

        lower = path.lower()
        if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            return {"valid": False,
                    "reason": "not a NIfTI file (expected .nii or .nii.gz)"}

        slices, reason = mri_to_slices(path, n_slices=SLICES_PER_PATIENT)
        if not slices:
            return {"valid": False, "reason": reason}

        slice_probs = self.predict_slices(slices)
        prob = aggregate(slice_probs, self.method)
        is_autism = prob >= self.threshold
        conf_pct, conf_band = confidence(slice_probs, prob, self.threshold)
        n_agree = int(round(conf_pct / 100.0 * len(slice_probs)))

        return {
            "valid": True,
            "reason": "ok",
            "label": "AUTISM" if is_autism else "HEALTHY",
            "autism_probability": prob,
            # Complement of the Autism score -- the model's score for Healthy.
            # Note the decision threshold is not necessarily 0.5, so a HEALTHY
            # verdict can carry a Healthy score slightly under 50%. That is the
            # model's real number and is reported as-is rather than rescaled.
            "healthy_probability": 1.0 - prob,
            "confidence_percent": conf_pct,
            "confidence": conf_band,
            "slices_agreeing": n_agree,
            "n_slices": len(slices),
            "threshold": self.threshold,
            "aggregation": self.method,
            # kept for logging/debugging only; never surfaced in the GUI
            "slice_probabilities": slice_probs.tolist(),
        }
