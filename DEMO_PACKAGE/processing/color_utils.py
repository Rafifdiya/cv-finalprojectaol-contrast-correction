# processing/color_utils.py
# Utility functions for color space conversion
# Input/Output contract: numpy array, BGR, uint8, shape (H, W, 3)

import cv2
import numpy as np


def bgr_to_lab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert BGR image to LAB and return (L, A, B) channels separately.
    We only equalize the L channel to preserve original hue/saturation.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    return L, A, B


def lab_to_bgr(L: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Merge L, A, B channels back and convert to BGR.
    """
    merged = cv2.merge([L, A, B])
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return bgr


def load_image(path: str) -> np.ndarray:
    """
    Load image from path. Returns BGR numpy array.
    Raises FileNotFoundError if image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save BGR numpy array to disk.
    """
    cv2.imwrite(path, image)


def pil_to_bgr(pil_image) -> np.ndarray:
    """
    Convert PIL Image (from Streamlit uploader) to BGR numpy array.
    """
    arr = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR to RGB — for display in Streamlit / matplotlib.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
