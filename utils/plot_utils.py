# utils/plot_utils.py
# Histogram plotting utilities for Streamlit display

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for Streamlit


def plot_histogram(image: np.ndarray, title: str = "Histogram") -> plt.Figure:
    """
    Plot grayscale histogram of an image's L channel (LAB).
    
    Args:
        image: BGR numpy array
        title: Plot title string
    
    Returns:
        matplotlib Figure object (pass to st.pyplot())
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(L.ravel(), bins=256, range=(0, 256), color="#378ADD", alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Pixel Intensity (L channel)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xlim([0, 255])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_comparison_histograms(
    original: np.ndarray,
    ghe_result: np.ndarray,
    clahe_result: np.ndarray,
) -> plt.Figure:
    """
    Side-by-side histogram comparison: Original | GHE | CLAHE.
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    datasets = [
        (original, "Original", "#888780"),
        (ghe_result, "After GHE", "#378ADD"),
        (clahe_result, "After CLAHE", "#1D9E75"),
    ]

    for ax, (img, label, color) in zip(axes, datasets):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab)
        ax.hist(L.ravel(), bins=256, range=(0, 256), color=color, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlim([0, 255])
        ax.tick_params(labelsize=7)

    fig.suptitle("Histogram Comparison (L channel)", fontsize=11)
    fig.tight_layout()
    return fig
