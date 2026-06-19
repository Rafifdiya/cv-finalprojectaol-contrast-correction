"""
Train all ML models for the Automated Contrast Correction project.
Run once before launching the Streamlit app.

Usage:
    python train.py
"""

import time
from ml.classifier import train_classifier
from ml.predictor import train_predictor


def main():
    print("=" * 55)
    print("  Automated Contrast Correction — Model Training")
    print("=" * 55)

    print("\n[1/2] Training Exposure Classifier...")
    t0 = time.time()
    report = train_classifier()
    print(f"  CV Accuracy : {report['cv_accuracy_mean']:.4f} ± {report['cv_accuracy_std']:.4f}")
    print(f"  Samples     : {report['n_samples']} {report['label_counts']}")
    print(f"  Over source : {report['over_source']}")
    print(f"  Saved to    : {report['model_path']}")
    print(f"  Time        : {time.time() - t0:.1f}s")

    print("\n[2/2] Training Parameter Predictor (grid experiment on all 485 train images)...")
    t0 = time.time()
    report = train_predictor()
    print(f"  Clip MAE    : {report['clip_mae']:.4f}")
    print(f"  Tile Acc    : {report['tile_accuracy']:.4f}")
    print(f"  Samples     : {report['n_samples']}")
    print(f"  Results CSV : {report['results_csv']}")
    print(f"  Time        : {time.time() - t0:.1f}s")

    print("\nAll models trained. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
