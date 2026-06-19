import cv2
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from ml.features import extract_features
from processing import apply_clahe, apply_ghe, evaluate

CLIP_LIMITS = [1.0, 2.0, 3.0, 4.0]
TILE_SIZES  = [(4, 4), (8, 8), (16, 16)]

MODEL_PATH_CLIP = Path(__file__).parent / "models" / "predictor_clip.pkl"
MODEL_PATH_TILE = Path(__file__).parent / "models" / "predictor_tile.pkl"
RESULTS_CSV     = Path("experiments") / "results.csv"


def train_predictor(dataset_dir: str = "dataset") -> dict:
    """
    For each train/low image, run CLAHE grid + GHE and find params with best SSIM.
    Trains RF regressor (clip_limit) + RF classifier (tile_size).
    Saves models and experiment results to experiments/results.csv.
    Returns training report dict.
    """
    train_low    = Path(dataset_dir) / "train" / "low"
    train_normal = Path(dataset_dir) / "train" / "normal"

    X, y_clip, y_tile = [], [], []
    rows = []

    low_images = sorted(train_low.glob("*.png"))

    for p in tqdm(low_images, desc="grid experiment"):
        img_low    = cv2.imread(str(p))
        img_normal = cv2.imread(str(train_normal / p.name))

        if img_low is None or img_normal is None:
            continue

        # GHE baseline
        ghe_result   = apply_ghe(img_low)
        ghe_metrics  = evaluate(img_normal, ghe_result)
        rows.append({
            "image": p.name, "method": "GHE",
            "clip_limit": None, "tile_size": None,
            "psnr": round(ghe_metrics["psnr"], 4),
            "ssim": round(ghe_metrics["ssim"], 4),
        })

        best_ssim = -1.0
        best_clip = 2.0
        best_tile = (8, 8)

        for clip in CLIP_LIMITS:
            for tile in TILE_SIZES:
                result  = apply_clahe(img_low, clip_limit=clip, tile_size=tile)
                metrics = evaluate(img_normal, result)
                rows.append({
                    "image": p.name, "method": "CLAHE",
                    "clip_limit": clip,
                    "tile_size": f"{tile[0]}x{tile[1]}",
                    "psnr": round(metrics["psnr"], 4),
                    "ssim": round(metrics["ssim"], 4),
                })
                if metrics["ssim"] > best_ssim:
                    best_ssim = metrics["ssim"]
                    best_clip = clip
                    best_tile = tile

        X.append(extract_features(img_low))
        y_clip.append(best_clip)
        y_tile.append(best_tile[0])  # encode tile as N in (N, N)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    X      = np.array(X)
    y_clip = np.array(y_clip, dtype=np.float32)
    y_tile = np.array(y_tile, dtype=np.int32)

    reg        = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    clip_scores = cross_val_score(reg, X, y_clip, cv=5, scoring="neg_mean_absolute_error")
    reg.fit(X, y_clip)

    tile_clf    = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    tile_scores = cross_val_score(tile_clf, X, y_tile, cv=5, scoring="accuracy")
    tile_clf.fit(X, y_tile)

    MODEL_PATH_CLIP.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg,      MODEL_PATH_CLIP)
    joblib.dump(tile_clf, MODEL_PATH_TILE)

    return {
        "clip_mae":     round(float(-clip_scores.mean()), 4),
        "tile_accuracy": round(float(tile_scores.mean()), 4),
        "n_samples":    int(len(X)),
        "results_csv":  str(RESULTS_CSV),
    }


def predict_params(image: np.ndarray) -> dict:
    """
    Predict best CLAHE parameters for input BGR image.
    Returns {"clip_limit": float, "tile_size": tuple[int,int]}.
    """
    if not MODEL_PATH_CLIP.exists() or not MODEL_PATH_TILE.exists():
        raise FileNotFoundError("Predictor models not found. Run train_predictor() first.")

    reg      = joblib.load(MODEL_PATH_CLIP)
    tile_clf = joblib.load(MODEL_PATH_TILE)

    features = extract_features(image).reshape(1, -1)

    clip_raw   = float(reg.predict(features)[0])
    clip_limit = float(np.clip(round(clip_raw * 2) / 2, 1.0, 8.0))  # snap to 0.5 steps

    tile_n    = int(tile_clf.predict(features)[0])
    tile_size = (tile_n, tile_n)

    return {"clip_limit": clip_limit, "tile_size": tile_size}


def load_predictor():
    """Load trained predictor. Returns (reg, tile_clf) or (None, None)."""
    if MODEL_PATH_CLIP.exists() and MODEL_PATH_TILE.exists():
        return joblib.load(MODEL_PATH_CLIP), joblib.load(MODEL_PATH_TILE)
    return None, None
