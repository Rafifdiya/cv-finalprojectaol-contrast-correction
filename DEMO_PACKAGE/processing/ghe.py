# processing/ghe.py
# Global Histogram Equalization
# Input/Output contract: numpy array, BGR, uint8, shape (H, W, 3)

import cv2
import numpy as np
from processing.color_utils import bgr_to_lab, lab_to_bgr


def apply_ghe(image: np.ndarray) -> np.ndarray:
    """
    Apply Global Histogram Equalization on the L channel (LAB space).
    
    Why LAB? Equalizing directly on BGR shifts hue/saturation.
    By isolating the L (Lightness) channel, we only affect brightness.
    
    Args:
        image: BGR numpy array, uint8, shape (H, W, 3)
    
    Returns:
        Enhanced BGR numpy array, uint8, shape (H, W, 3)
    """
    L, A, B = bgr_to_lab(image)

    # OpenCV's equalizeHist works on single-channel uint8
    L_eq = cv2.equalizeHist(L)

    enhanced = lab_to_bgr(L_eq, A, B)
    return enhanced
