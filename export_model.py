"""
export_model.py
---------------
Packages the trained model for sharing, so whoever clones this repository can
run predictions immediately instead of retraining for three hours.

Why this script exists
    Models/autism_detector.keras is 112 MB. That is not the network's weights
    -- the network has 10.8M parameters, which is 44 MB of float32. The extra
    68 MB is Adam optimizer state: two momentum buffers for every fine-tuned
    parameter, saved inside the weights file so training could be resumed.

    Those buffers are useless for prediction, and 112 MB exceeds GitHub's
    100 MB per-file hard limit, so a push containing it is rejected.

    Rebuilding the architecture and loading the weights by topology produces a
    44.5 MB inference-only model with bit-identical predictions, which commits
    to an ordinary git repository with no Git LFS and no release asset.

Output
    Models/neuroconnect_model.keras   the shareable model
    Models/model_config.json          already written by train_model.py

Usage
    python export_model.py
    python export_model.py --verify        also check predictions match
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
SOURCE_PATH = os.path.join(MODELS_DIR, "autism_detector.keras")
EXPORT_PATH = os.path.join(MODELS_DIR, "neuroconnect_model.keras")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

GITHUB_FILE_LIMIT_MB = 100


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a shareable trained model")
    ap.add_argument("--source", default=SOURCE_PATH)
    ap.add_argument("--out", default=EXPORT_PATH)
    ap.add_argument("--verify", action="store_true",
                    help="confirm the export predicts identically to the source")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"ERROR: {args.source} not found. Run train_model.py first.")

    from tensorflow import keras
    import train_model as tm

    print("=" * 58)
    print("EXPORTING SHAREABLE MODEL")
    print("=" * 58)

    src_mb = os.path.getsize(args.source) / 1e6
    print(f"Source: {os.path.basename(args.source)}  ({src_mb:.1f} MB)")

    # A freshly built architecture has no optimizer attached, so loading the
    # weights into it drops the momentum buffers.
    model, _ = tm.build_model()
    model.load_weights(args.source)
    model.save(args.out)

    out_mb = os.path.getsize(args.out) / 1e6
    print(f"Export: {os.path.basename(args.out)}  ({out_mb:.1f} MB)")
    print(f"Saved {src_mb - out_mb:.1f} MB "
          f"({(1 - out_mb / src_mb) * 100:.0f}% smaller)")

    if args.verify:
        print("\nVerifying predictions match...")
        src_model = keras.models.load_model(args.source)
        x = np.random.RandomState(0).rand(8, tm.IMG_SIZE, tm.IMG_SIZE, 3)
        x = (x * 255).astype("float32")
        a = src_model.predict(x, verbose=0).ravel()
        b = model.predict(x, verbose=0).ravel()
        diff = float(np.abs(a - b).max())
        print(f"  max difference over 8 random inputs: {diff:.3e}")
        if diff > 1e-6:
            sys.exit("ERROR: exported model does not match the source")
        print("  identical")

    print()
    if out_mb < GITHUB_FILE_LIMIT_MB:
        print(f"OK: {out_mb:.1f} MB is under GitHub's {GITHUB_FILE_LIMIT_MB} MB "
              f"limit and can be committed directly.")
    else:
        print(f"WARNING: {out_mb:.1f} MB exceeds GitHub's "
              f"{GITHUB_FILE_LIMIT_MB} MB limit. Use Git LFS or a Release "
              f"asset instead of committing it.")

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
        print(f"\nShip alongside it: model_config.json "
              f"(aggregation={cfg.get('aggregation')}, "
              f"threshold={cfg.get('threshold', 0):.3f}) and class_names.json.")
        print("predictor.py needs all three to reproduce the same answers.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
