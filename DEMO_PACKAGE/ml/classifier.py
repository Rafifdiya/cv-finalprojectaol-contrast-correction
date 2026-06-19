import cv2
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from ml.features import extract_features

LABELS      = ["underexposed", "normal", "overexposed"]
MODEL_PATH  = Path(__file__).parent / "models" / "exposure_classifier.pkl"

# L-channel mean thresholds (0–255 scale)
THRESHOLD_LOW  = 80
THRESHOLD_HIGH = 130


def _synthesize_overexposed(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    lut = np.array(
        [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def train_classifier(dataset_dir: str = "dataset") -> dict:
    """
    Train Random Forest exposure classifier.
    - train/low   → underexposed
    - train/normal → normal
    - train/over  → overexposed (real SICE images, if folder exists & non-empty)
    - fallback: gamma-brightened train/normal → overexposed (synthesized)
    Saves model to ml/models/exposure_classifier.pkl.
    Returns training report dict.
    """
    train_low    = Path(dataset_dir) / "train" / "low"
    train_normal = Path(dataset_dir) / "train" / "normal"
    train_over   = Path(dataset_dir) / "train" / "over"

    # Determine overexposed source
    real_over_imgs = sorted(train_over.glob("*.png")) if train_over.exists() else []
    use_real_over  = len(real_over_imgs) >= 50  # require at least 50 real images
    over_source    = "real (SICE)" if use_real_over else "synthesized (gamma)"
    print(f"  Overexposed source: {over_source} "
          f"({'%d images' % len(real_over_imgs) if use_real_over else 'from train/normal'})")

    X, y = [], []

    for p in tqdm(sorted(train_low.glob("*.png")), desc="underexposed"):
        img = cv2.imread(str(p))
        if img is not None:
            X.append(extract_features(img))
            y.append("underexposed")

    for p in tqdm(sorted(train_normal.glob("*.png")), desc="normal"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        X.append(extract_features(img))
        y.append("normal")
        if not use_real_over:
            X.append(extract_features(_synthesize_overexposed(img)))
            y.append("overexposed")

    if use_real_over:
        for p in tqdm(real_over_imgs, desc="overexposed (SICE)"):
            img = cv2.imread(str(p))
            if img is not None:
                X.append(extract_features(img))
                y.append("overexposed")

    X = np.array(X)
    y = np.array(y)

    clf    = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    clf.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    return {
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_std":  round(float(scores.std()), 4),
        "n_samples":        int(len(y)),
        "label_counts":     {l: int((y == l).sum()) for l in LABELS},
        "model_path":       str(MODEL_PATH),
        "over_source":      over_source,
    }


def predict_exposure(image: np.ndarray) -> str:
    """
    Predict exposure class. Returns 'underexposed', 'normal', or 'overexposed'.
    Uses hybrid ML + rule-based fallback for low-confidence predictions.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_classifier() first.")
    clf = joblib.load(MODEL_PATH)
    features = extract_features(image).reshape(1, -1)
    proba = clf.predict_proba(features)[0]
    label = clf.classes_[proba.argmax()]
    confidence = proba.max()

    # Rule-based fallback when confidence is low or overexposed is misclassified
    # Real overexposed: high mean + high p90 + negative skewness
    mean    = float(features[0][0])
    p90     = float(features[0][7])
    skew    = float(features[0][2])

    if label == "normal":
        # Blown highlights (p90 > 240) + bright mean = strong overexposed signal
        if mean > 128 and p90 > 240:
            return "overexposed"
        # Lower-confidence prediction + softer overexposed features
        if confidence < 0.85 and mean > 128 and p90 > 205 and skew < 0.1:
            return "overexposed"

    return str(label)


def load_classifier():
    """Load trained classifier. Returns None if not trained yet."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None
