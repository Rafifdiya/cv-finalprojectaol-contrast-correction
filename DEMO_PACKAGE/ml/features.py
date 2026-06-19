import cv2
import numpy as np
from scipy import stats

FEATURE_NAMES = [
    "mean", "std", "skewness", "kurtosis",
    "p10", "p25", "p75", "p90",
    "contrast_ratio", "entropy",
]


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract 10 histogram features from L channel of a BGR image.
    Returns float32 array of shape (10,).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = cv2.split(lab)[0].ravel().astype(np.float32)

    mean      = float(np.mean(L))
    std       = float(np.std(L))
    skewness  = float(stats.skew(L))
    kurt      = float(stats.kurtosis(L))
    p10, p25, p75, p90 = (float(v) for v in np.percentile(L, [10, 25, 75, 90]))
    contrast  = p90 - p10

    hist, _ = np.histogram(L, bins=256, range=(0, 256), density=True)
    hist    = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))

    return np.array(
        [mean, std, skewness, kurt, p10, p25, p75, p90, contrast, entropy],
        dtype=np.float32,
    )
