"""
ML Visualization — generate all training plots.
Run: python ml/visualize.py

Output: experiments/plots/
  - classifier_confusion_matrix.png
  - classifier_feature_importance.png
  - classifier_cv_scores.png
  - predictor_clip_scatter.png
  - predictor_feature_importance.png
  - predictor_param_distribution.png
  - experiment_ssim_heatmap.png
  - experiment_method_comparison.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from ml.features import extract_features, FEATURE_NAMES
from ml.classifier import _synthesize_overexposed, LABELS

# ─── Paths ────────────────────────────────────────────────────────────────────
PLOTS_DIR       = Path("experiments") / "plots"
RESULTS_CSV     = Path("experiments") / "results.csv"
MODEL_CLF       = Path("ml") / "models" / "exposure_classifier.pkl"
MODEL_CLIP      = Path("ml") / "models" / "predictor_clip.pkl"
MODEL_TILE      = Path("ml") / "models" / "predictor_tile.pkl"
DATASET_DIR     = Path("dataset")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
BLUE   = "#2368B4"
CYAN   = "#00B4D8"
GRAY   = "#CCCCCC"
RED    = "#E05252"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
})

LABEL_COLORS = {
    "underexposed": BLUE,
    "normal":       GREEN,
    "overexposed":  ORANGE,
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_classifier_data():
    """Reload training data exactly as classifier.py does."""
    train_low    = DATASET_DIR / "train" / "low"
    train_normal = DATASET_DIR / "train" / "normal"
    X, y = [], []

    for p in tqdm(sorted(train_low.glob("*.png")), desc="loading underexposed"):
        img = cv2.imread(str(p))
        if img is not None:
            X.append(extract_features(img))
            y.append("underexposed")

    for p in tqdm(sorted(train_normal.glob("*.png")), desc="loading normal+overexposed"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        X.append(extract_features(img))
        y.append("normal")
        X.append(extract_features(_synthesize_overexposed(img)))
        y.append("overexposed")

    return np.array(X), np.array(y)


def _load_predictor_data():
    """Load feature vectors + best clip/tile labels from results.csv."""
    if not RESULTS_CSV.exists():
        print(f"[SKIP] {RESULTS_CSV} not found — run train_predictor() first")
        return None, None, None

    df = pd.read_csv(RESULTS_CSV)
    clahe_df = df[df["method"] == "CLAHE"].copy()

    train_low = DATASET_DIR / "train" / "low"
    X, y_clip, y_tile = [], [], []

    best = (
        clahe_df.loc[clahe_df.groupby("image")["ssim"].idxmax()]
        .set_index("image")
    )

    for p in tqdm(sorted(train_low.glob("*.png")), desc="loading predictor data"):
        if p.name not in best.index:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        row = best.loc[p.name]
        X.append(extract_features(img))
        y_clip.append(float(row["clip_limit"]))
        tile_n = int(str(row["tile_size"]).split("x")[0])
        y_tile.append(tile_n)

    return np.array(X), np.array(y_clip, dtype=np.float32), np.array(y_tile)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(X, y):
    print("Generating: confusion matrix...")
    if not MODEL_CLF.exists():
        print("[SKIP] classifier model not found")
        return

    clf = joblib.load(MODEL_CLF)
    y_pred = cross_val_predict(clf, X, y, cv=5)

    cm = confusion_matrix(y, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title("Confusion Matrix — Exposure Classifier\n(5-fold cross-validated predictions)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    out = PLOTS_DIR / "classifier_confusion_matrix.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLASSIFIER FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_classifier_feature_importance():
    print("Generating: classifier feature importance...")
    if not MODEL_CLF.exists():
        print("[SKIP] classifier model not found")
        return

    clf = joblib.load(MODEL_CLF)
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        range(len(FEATURE_NAMES)),
        importances[idx],
        color=[BLUE if i < 3 else GRAY for i in range(len(FEATURE_NAMES))],
        edgecolor="none",
    )
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Importance (Gini)")
    ax.set_title("Feature Importance — Exposure Classifier\n(top 3 in blue)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    out = PLOTS_DIR / "classifier_feature_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIER CV FOLD SCORES
# ══════════════════════════════════════════════════════════════════════════════

def plot_classifier_cv_scores(X, y):
    print("Generating: classifier CV fold scores...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 6)
    bars = ax.bar(x, scores, color=BLUE, edgecolor="none", width=0.5)
    ax.axhline(scores.mean(), color=RED, linestyle="--", linewidth=1.5,
               label=f"Mean = {scores.mean():.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.set_title(f"5-Fold CV Accuracy — Exposure Classifier\nMean: {scores.mean():.4f} ± {scores.std():.4f}")
    ax.legend(fontsize=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    out = PLOTS_DIR / "classifier_cv_scores.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTOR — PREDICTED VS ACTUAL CLIP_LIMIT
# ══════════════════════════════════════════════════════════════════════════════

def plot_clip_scatter(X, y_clip):
    print("Generating: predicted vs actual clip_limit...")
    if not MODEL_CLIP.exists():
        print("[SKIP] predictor model not found")
        return

    reg = joblib.load(MODEL_CLIP)
    y_pred = cross_val_predict(reg, X, y_clip, cv=5)

    mae = float(np.mean(np.abs(y_pred - y_clip)))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_clip, y_pred, alpha=0.35, s=18, color=BLUE, edgecolors="none")
    lims = [min(y_clip.min(), y_pred.min()) - 0.2,
            max(y_clip.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, color=RED, linewidth=1.5, linestyle="--", label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual clip_limit")
    ax.set_ylabel("Predicted clip_limit")
    ax.set_title(f"Predicted vs Actual — clip_limit\nMAE = {mae:.4f} (5-fold CV)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = PLOTS_DIR / "predictor_clip_scatter.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTOR — FEATURE IMPORTANCE (clip regressor)
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictor_feature_importance():
    print("Generating: predictor feature importance...")
    if not MODEL_CLIP.exists():
        print("[SKIP] predictor model not found")
        return

    reg = joblib.load(MODEL_CLIP)
    importances = reg.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        range(len(FEATURE_NAMES)),
        importances[idx],
        color=[CYAN if i < 3 else GRAY for i in range(len(FEATURE_NAMES))],
        edgecolor="none",
    )
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=30, ha="right")
    ax.set_ylabel("Importance (Gini)")
    ax.set_title("Feature Importance — clip_limit Predictor\n(top 3 in cyan)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    out = PLOTS_DIR / "predictor_feature_importance.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICTOR — BEST PARAMETER DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_param_distribution(y_clip, y_tile):
    print("Generating: best parameter distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # clip_limit distribution
    clip_vals, clip_counts = np.unique(y_clip, return_counts=True)
    axes[0].bar(
        [str(v) for v in clip_vals], clip_counts,
        color=BLUE, edgecolor="none", width=0.5
    )
    axes[0].set_xlabel("clip_limit")
    axes[0].set_ylabel("Count (images)")
    axes[0].set_title("Best clip_limit Distribution\nacross 485 training images")
    for x, cnt in zip(range(len(clip_vals)), clip_counts):
        axes[0].text(x, cnt + 2, str(cnt), ha="center", va="bottom", fontsize=11)

    # tile_size distribution
    tile_vals, tile_counts = np.unique(y_tile, return_counts=True)
    tile_labels = [f"{v}×{v}" for v in tile_vals]
    axes[1].bar(
        tile_labels, tile_counts,
        color=CYAN, edgecolor="none", width=0.5
    )
    axes[1].set_xlabel("tile_size")
    axes[1].set_ylabel("Count (images)")
    axes[1].set_title("Best tile_size Distribution\nacross 485 training images")
    for x, cnt in zip(range(len(tile_vals)), tile_counts):
        axes[1].text(x, cnt + 2, str(cnt), ha="center", va="bottom", fontsize=11)

    plt.suptitle("How often each CLAHE parameter is optimal (by SSIM)", y=1.02, fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / "predictor_param_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPERIMENT — SSIM HEATMAP (clip × tile)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ssim_heatmap():
    print("Generating: SSIM heatmap...")
    if not RESULTS_CSV.exists():
        print(f"[SKIP] {RESULTS_CSV} not found")
        return

    df = pd.read_csv(RESULTS_CSV)
    clahe = df[df["method"] == "CLAHE"].copy()
    clahe["tile_label"] = clahe["tile_size"].astype(str)

    pivot = clahe.groupby(["clip_limit", "tile_label"])["ssim"].mean().unstack()
    tile_order = ["4x4", "8x8", "16x16"]
    pivot = pivot[[c for c in tile_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=pivot.values.min() * 0.95)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("tile_size")
    ax.set_ylabel("clip_limit")
    ax.set_title("Mean SSIM Heatmap — CLAHE Grid Search\n(avg over 485 training images)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > pivot.values.mean() else "black")

    plt.colorbar(im, ax=ax, label="Mean SSIM")
    plt.tight_layout()
    out = PLOTS_DIR / "experiment_ssim_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. EXPERIMENT — GHE vs CLAHE BAR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_method_comparison():
    print("Generating: GHE vs CLAHE comparison...")
    if not RESULTS_CSV.exists():
        print(f"[SKIP] {RESULTS_CSV} not found")
        return

    df = pd.read_csv(RESULTS_CSV)

    ghe_mean_psnr = df[df["method"] == "GHE"]["psnr"].mean()
    ghe_mean_ssim = df[df["method"] == "GHE"]["ssim"].mean()

    best_clahe = (
        df[df["method"] == "CLAHE"]
        .groupby("image")[["psnr", "ssim"]]
        .max()
    )
    clahe_mean_psnr = best_clahe["psnr"].mean()
    clahe_mean_ssim = best_clahe["ssim"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, metric, ghe_val, clahe_val, ylabel in [
        (axes[0], "PSNR (dB)", ghe_mean_psnr, clahe_mean_psnr, "Mean PSNR (dB)"),
        (axes[1], "SSIM",      ghe_mean_ssim,  clahe_mean_ssim,  "Mean SSIM"),
    ]:
        vals = [ghe_val, clahe_val]
        colors = [GRAY, BLUE]
        bars = ax.bar(["GHE", "CLAHE (best)"], vals, color=colors,
                      edgecolor="none", width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Mean {metric} — GHE vs CLAHE\n(best per image, 485 train)")
        base = min(vals) * 0.9
        ax.set_ylim(base, max(vals) * 1.08)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(vals) - min(vals)) * 0.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.suptitle("GHE (baseline) vs Best CLAHE — Average Metrics", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "experiment_method_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ML Visualization — loading data...")
    print("=" * 60)

    # Load classifier data (needed for plots 1, 2, 3)
    X_clf, y_clf = _load_classifier_data()

    # Load predictor data (needed for plots 4, 5, 6)
    X_pred, y_clip, y_tile = _load_predictor_data()

    print("\nGenerating plots...")
    print("-" * 60)

    # Classifier plots
    plot_confusion_matrix(X_clf, y_clf)
    plot_classifier_feature_importance()
    plot_classifier_cv_scores(X_clf, y_clf)

    # Predictor plots
    if X_pred is not None:
        plot_clip_scatter(X_pred, y_clip)
        plot_predictor_feature_importance()
        plot_param_distribution(y_clip, y_tile)

    # Experiment plots (from results.csv — no model needed)
    plot_ssim_heatmap()
    plot_method_comparison()

    print("\n" + "=" * 60)
    print(f"Done. All plots saved to: {PLOTS_DIR.resolve()}")
    print("=" * 60)
