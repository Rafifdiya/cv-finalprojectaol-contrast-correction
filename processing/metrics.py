# processing/metrics.py
# Evaluation metrics: PSNR and SSIM
# Both functions expect BGR numpy arrays, uint8

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(ground_truth: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better. Typical good range: > 20 dB.
    
    Args:
        ground_truth: Normal-light reference image (BGR, uint8)
        enhanced    : Output of GHE or CLAHE (BGR, uint8)
    
    Returns:
        PSNR value in dB (float)
    """
    return peak_signal_noise_ratio(ground_truth, enhanced, data_range=255)


def calculate_ssim(ground_truth: np.ndarray, enhanced: np.ndarray) -> float:
    """
    Structural Similarity Index (0 to 1).
    Higher is better. Project target: > 0.80.
    
    Args:
        ground_truth: Normal-light reference image (BGR, uint8)
        enhanced    : Output of GHE or CLAHE (BGR, uint8)
    
    Returns:
        SSIM value (float, 0–1)
    """
    return structural_similarity(
        ground_truth,
        enhanced,
        channel_axis=2,   # tell skimage this is a color image (H, W, C)
        data_range=255,
    )


def evaluate(ground_truth: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Run both metrics at once.
    
    Returns:
        {"psnr": float, "ssim": float}
    """
    return {
        "psnr": calculate_psnr(ground_truth, enhanced),
        "ssim": calculate_ssim(ground_truth, enhanced),
    }
