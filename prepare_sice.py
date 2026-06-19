"""
Prepare SICE Part2 dataset — extract overexposed images.

SICE Part2 structure (expected):
  dataset/sice_raw/
    1/
      1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg   (multi-exposure, dark → bright)
    2/
      1.jpg, 2.jpg, ...
    ...
    Label/
      1.jpg, 2.jpg, ...                    (ground truth, ignored)

Strategy:
  For each sequence folder, identify the BRIGHTEST image by measuring
  mean L-channel brightness. If mean > OVER_THRESHOLD → overexposed.
  Convert to PNG, save to dataset/train/over/.

Usage:
  python prepare_sice.py
  python prepare_sice.py --sice_dir dataset/sice_raw --out_dir dataset/train/over --limit 485
"""

import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm


OVER_THRESHOLD  = 155   # L-channel mean (0-255) above this = overexposed
NORMAL_LOW      = 80    # below this = underexposed, skip
IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mean_brightness(bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(cv2.split(lab)[0].mean())


def get_image_files(folder: Path) -> list[Path]:
    files = [f for f in sorted(folder.iterdir())
             if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
    return files


def prepare_sice(sice_dir: str, out_dir: str, limit: int = 485) -> dict:
    sice_path = Path(sice_dir)
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not sice_path.exists():
        raise FileNotFoundError(f"SICE directory not found: {sice_path}")

    # Get all sequence subdirectories (skip Label/ folder)
    seq_dirs = sorted([
        d for d in sice_path.iterdir()
        if d.is_dir() and d.name.lower() != "label"
    ])

    if not seq_dirs:
        raise ValueError(f"No sequence folders found in {sice_path}. "
                         "Make sure SICE Part2 is extracted there.")

    print(f"Found {len(seq_dirs)} sequences in {sice_path}")
    print(f"Output: {out_path}")
    print(f"Limit: {limit} images\n")

    saved       = 0
    skipped_dim = 0  # too dark / not overexposed
    skipped_err = 0  # read error

    for seq in tqdm(seq_dirs, desc="Extracting overexposed"):
        if saved >= limit:
            break

        imgs = get_image_files(seq)
        if not imgs:
            continue

        # Read all images in sequence, pick brightest
        best_img  = None
        best_mean = -1.0
        best_path = None

        for img_path in imgs:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skipped_err += 1
                continue
            m = mean_brightness(bgr)
            if m > best_mean:
                best_mean  = m
                best_img   = bgr
                best_path  = img_path

        if best_img is None:
            skipped_err += 1
            continue

        if best_mean < OVER_THRESHOLD:
            skipped_dim += 1
            continue

        # Save as PNG
        out_name = f"sice_{seq.name}_{best_path.stem}.png"
        out_file = out_path / out_name
        cv2.imwrite(str(out_file), best_img)
        saved += 1

    print(f"\nDone.")
    print(f"  Saved      : {saved} overexposed images → {out_path}")
    print(f"  Skipped    : {skipped_dim} (brightness < {OVER_THRESHOLD}, not overexposed)")
    print(f"  Errors     : {skipped_err} (unreadable files)")

    if saved < 100:
        print(f"\n  WARNING: Only {saved} images saved. "
              f"Check SICE folder structure or lower OVER_THRESHOLD (current={OVER_THRESHOLD}).")

    return {"saved": saved, "skipped_dim": skipped_dim, "errors": skipped_err}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SICE overexposed images")
    parser.add_argument("--sice_dir", default="dataset/sice_raw",
                        help="Path to extracted SICE Part2 folder")
    parser.add_argument("--out_dir",  default="dataset/train/over",
                        help="Output folder for overexposed PNGs")
    parser.add_argument("--limit",    type=int, default=485,
                        help="Max number of images to extract (default: 485)")
    args = parser.parse_args()

    prepare_sice(args.sice_dir, args.out_dir, args.limit)
