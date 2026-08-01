"""
neuro_utils.py
--------------
Shared MRI handling for NeuroConnect AI v2.

Every stage of the project (dataset creation, testing, GUI) must turn a raw
whole-brain MRI volume into exactly the same kind of 224x224 slice images,
otherwise the model sees different data at train time and at predict time.
That conversion lives here, in one place, so it cannot drift.

Public API
    load_volume(path)              -> (volume_ras, nibabel_image)
    validate_volume(volume)        -> (is_valid, reason)
    extract_useful_slices(volume)  -> list[np.uint8 array of IMG_SIZE x IMG_SIZE]
    mri_to_slices(path)            -> (slices, reason)   # convenience wrapper
    find_mri_file(patient_dir)     -> path or None
"""

from __future__ import annotations

import os
import glob

import cv2
import numpy as np
import nibabel as nib

# --------------------------------------------------------------------------
# Slice-extraction configuration (shared by create_dataset / test / app)
# --------------------------------------------------------------------------
IMG_SIZE = 224          # network input size
SLICES_PER_PATIENT = 25  # evenly sampled across the informative brain range

# A slice counts as "brain-containing" when at least this fraction of its
# pixels sit above the tissue threshold. Background/air slices fall far below.
MIN_BRAIN_FRACTION = 0.05
TISSUE_THRESHOLD = 0.12  # on the 0..1 percentile-normalised volume

# Fraction of the brain's superior-inferior extent kept. Dropping the extreme
# top of the skull and the bottom of the cerebellum removes slices that are
# mostly skull/neck and carry little tissue signal.
CENTRAL_RANGE = 0.80

# Sanity bounds for "is this plausibly a brain MRI volume at all"
MIN_DIM = 32
MAX_DIM = 1024
MIN_USEFUL_SLICES = 10


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def find_mri_file(patient_dir: str, prefer: str = "anat") -> str | None:
    """Recursively find a patient's anatomical MRI.

    Filenames are NOT assumed to be identical across sites, so we search for
    any *.nii / *.nii.gz below the patient folder and prefer anatomical scans
    over functional/diffusion ones.
    """
    candidates = []
    for pattern in ("*.nii", "*.nii.gz"):
        candidates.extend(
            glob.glob(os.path.join(patient_dir, "**", pattern), recursive=True)
        )
    if not candidates:
        return None

    anat = [c for c in candidates if prefer in c.lower()]
    pool = anat if anat else candidates

    # Exclude obvious non-anatomical modalities when anything else is available
    filtered = [
        c for c in pool
        if not any(bad in c.lower() for bad in ("dti", "rest", "func", "bold"))
    ]
    pool = filtered if filtered else pool

    # Largest file is the highest-resolution structural scan
    return max(pool, key=os.path.getsize)


def load_volume(path: str):
    """Load a NIfTI file and reorient it to canonical RAS.

    Sites store their voxels in different axis orders. Reorienting to RAS
    guarantees axis 2 is always superior-inferior, so "axial slice" means the
    same thing for every site and for any file a user later drops into the GUI.
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = np.asanyarray(img.dataobj, dtype=np.float32)

    # Some scans carry a trailing singleton/time axis
    while vol.ndim > 3:
        vol = vol[..., 0]

    return vol, img


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_volume(vol: np.ndarray) -> tuple[bool, str]:
    """Reject anything that is not a usable 3D brain volume."""
    if vol is None:
        return False, "volume could not be read"
    if vol.ndim != 3:
        return False, f"not a 3D volume (ndim={vol.ndim})"
    if any(d < MIN_DIM or d > MAX_DIM for d in vol.shape):
        return False, f"implausible dimensions {vol.shape}"
    if not np.isfinite(vol).any():
        return False, "volume contains no finite values"

    finite = vol[np.isfinite(vol)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return False, "volume is empty or constant"
    if float(np.count_nonzero(finite)) / finite.size < 0.01:
        return False, "volume is almost entirely background"

    return True, "ok"


# --------------------------------------------------------------------------
# Slice extraction
# --------------------------------------------------------------------------
def _normalise(vol: np.ndarray) -> np.ndarray:
    """Percentile-based intensity normalisation to 0..1.

    MRI intensities are not calibrated: raw values differ wildly between
    scanners. Clipping at the 1st/99th percentile of foreground voxels removes
    scanner-specific scaling and bright outliers.
    """
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    fg = vol[vol > 0]
    if fg.size == 0:
        return np.zeros_like(vol)

    lo, hi = np.percentile(fg, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
        if hi <= lo:
            return np.zeros_like(vol)

    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0)


def _brain_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """In-plane bounding box of the brain, unioned over all slices.

    Cropping to the head removes background padding, which differs per site,
    and makes head size comparable across scanners.
    """
    rows = np.any(mask, axis=(1, 2))
    cols = np.any(mask, axis=(0, 2))
    if not rows.any() or not cols.any():
        return 0, mask.shape[0], 0, mask.shape[1]

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def _to_square(img: np.ndarray) -> np.ndarray:
    """Pad to square with background so the resize does not distort anatomy."""
    h, w = img.shape
    side = max(h, w)
    out = np.zeros((side, side), dtype=img.dtype)
    top = (side - h) // 2
    left = (side - w) // 2
    out[top:top + h, left:left + w] = img
    return out


def extract_useful_slices(
    vol: np.ndarray,
    n_slices: int = SLICES_PER_PATIENT,
) -> list[np.ndarray]:
    """Turn a 3D volume into evenly spaced axial slices of brain tissue.

    Whole volume -> normalise -> find brain-containing slices -> keep the
    central portion -> sample n_slices evenly -> crop to brain -> 224x224.

    Returns an empty list when the volume holds too little brain to be usable.
    """
    norm = _normalise(vol)
    mask = norm > TISSUE_THRESHOLD

    # Which axial slices actually contain brain?
    frac = mask.mean(axis=(0, 1))
    useful = np.where(frac >= MIN_BRAIN_FRACTION)[0]
    if useful.size < MIN_USEFUL_SLICES:
        return []

    first, last = int(useful[0]), int(useful[-1])
    span = last - first + 1

    # Keep the central CENTRAL_RANGE of the brain's vertical extent
    margin = int(round(span * (1.0 - CENTRAL_RANGE) / 2.0))
    lo, hi = first + margin, last - margin
    if hi - lo + 1 < MIN_USEFUL_SLICES:
        lo, hi = first, last

    # Evenly spaced indices; deterministic, never random
    count = min(n_slices, hi - lo + 1)
    indices = np.unique(np.linspace(lo, hi, count).round().astype(int))

    r0, r1, c0, c1 = _brain_bbox(mask)

    slices = []
    for k in indices:
        plane = norm[r0:r1, c0:c1, k]
        # RAS axis0 = left->right, axis1 = posterior->anterior. rot90 puts
        # anterior at the top, giving the conventional axial view.
        plane = np.rot90(plane)
        plane = _to_square(plane)
        img = (plane * 255.0).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Drop any slice that ended up essentially blank after cropping
        if img.mean() < 5.0:
            continue
        slices.append(img)

    return slices


def mri_to_slices(path: str, n_slices: int = SLICES_PER_PATIENT):
    """Full path -> slices, with validation. Returns (slices, reason)."""
    try:
        vol, _ = load_volume(path)
    except Exception as exc:  # unreadable / corrupted file
        return [], f"MRI could not be loaded ({type(exc).__name__}: {exc})"

    ok, reason = validate_volume(vol)
    if not ok:
        return [], reason

    slices = extract_useful_slices(vol, n_slices=n_slices)
    if len(slices) < MIN_USEFUL_SLICES:
        return [], f"insufficient useful slices ({len(slices)})"

    return slices, "ok"
