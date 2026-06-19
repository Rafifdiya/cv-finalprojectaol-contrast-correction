# processing/clahe.py
# Contrast Limited Adaptive Histogram Equalization
# Input/Output contract: numpy array, BGR, uint8, shape (H, W, 3)

import cv2
import numpy as np
from processing.color_utils import bgr_to_lab, lab_to_bgr


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE on the L channel (LAB space).
    
    Args:
        image     : BGR numpy array, uint8, shape (H, W, 3)
        clip_limit: Controls contrast amplification limit.
                    Higher = more contrast but more noise risk.
                    Recommended range: 1.0 – 4.0
        tile_size : Grid size for local equalization (rows, cols).
                    Smaller = more local, risk of tiling artifacts.
                    Recommended: (8, 8)
    
    Returns:
        Enhanced BGR numpy array, uint8, shape (H, W, 3)
    """
    L, A, B = bgr_to_lab(image)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    L_eq = clahe.apply(L)

    enhanced = lab_to_bgr(L_eq, A, B)
    return enhanced


def run_clahe_grid_experiment(
    image: np.ndarray,
    clip_limits: list[float] = [1.0, 2.0, 3.0, 4.0],
    tile_sizes: list[tuple] = [(4, 4), (8, 8), (16, 16)],
) -> dict:
    """
    Run all combinations of clip_limit x tile_size.
    Returns a dict keyed by (clip_limit, tile_size) → enhanced image.
    
    Usage:
        results = run_clahe_grid_experiment(image)
        for params, enhanced_img in results.items():
            ...
    """
    results = {}
    for clip in clip_limits:
        for tile in tile_sizes:
            key = (clip, tile)
            results[key] = apply_clahe(image, clip_limit=clip, tile_size=tile)
    return results
