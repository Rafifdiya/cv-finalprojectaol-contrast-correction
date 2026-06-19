import io
import cv2
import streamlit as st
from PIL import Image as PILImage
import numpy as np

from processing import apply_ghe, apply_clahe, evaluate, pil_to_bgr, bgr_to_rgb
from utils import plot_histogram, plot_comparison_histograms

# ── ML models (optional — graceful fallback if not trained yet) ────────────────
_ml_available = False
try:
    from ml.classifier import predict_exposure, load_classifier
    from ml.predictor import predict_params, load_predictor
    _clf           = load_classifier()
    _reg, _tile_clf = load_predictor()
    _ml_available  = (_clf is not None) and (_reg is not None)
except Exception:
    pass

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contrast Correction — CV AOL",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Automated Contrast Correction")
st.caption("GHE vs CLAHE — Computer Vision AOL | Binus University")
st.divider()

# ── Sidebar: controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    input_mode = st.radio("Input source", ["Upload from file", "Take from camera"])

    st.subheader("CLAHE Mode")
    if _ml_available:
        clahe_mode = st.radio("Mode", ["Manual (slider)", "Auto (ML)"])
    else:
        clahe_mode = "Manual (slider)"
        st.caption("⚠️ ML models not trained yet. Run `python train.py` first.")

    st.subheader("CLAHE Parameters")
    if clahe_mode == "Manual (slider)":
        clip_limit  = st.slider("Clip Limit", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
        tile_size_n = st.select_slider("Tile Size", options=[4, 8, 16], value=8)
        tile_size   = (tile_size_n, tile_size_n)
    else:
        st.info("Parameters will be set automatically by ML model.")
        clip_limit  = 2.0
        tile_size_n = 8
        tile_size   = (8, 8)

    show_histogram  = st.checkbox("Show histograms", value=True)
    has_ground_truth = st.checkbox("I have a ground truth image (for PSNR/SSIM)", value=False)

# ── Heuristic scorer (no GT needed) ───────────────────────────────────────────
def _heuristic_score(bgr_img: np.ndarray) -> float:
    """
    Score image quality without ground truth.
    Combines: entropy (detail), brightness balance, contrast, noise penalty.
    Higher = better.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L_2d = cv2.split(lab)[0].astype(np.float32)
    L    = L_2d.ravel()

    # Entropy (detail richness)
    hist, _ = np.histogram(L, bins=256, range=(0, 256), density=True)
    hist    = hist + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))

    # Brightness balance: penalize if mean too dark (<80) or too bright (>180)
    mean = float(np.mean(L))
    brightness_score = 1.0 - abs(mean - 128.0) / 128.0

    # Contrast (p90 - p10)
    p10, p90 = float(np.percentile(L, 10)), float(np.percentile(L, 90))
    contrast = (p90 - p10) / 255.0

    # Noise penalty: high Laplacian variance = noisy = lower score
    # Use soft sigmoid-like: 1/(1 + var/800) so sensitive at typical noise levels
    lap_var     = float(cv2.Laplacian(L_2d, cv2.CV_32F).var())
    noise_score = 1.0 / (1.0 + lap_var / 800.0)

    return (
        0.30 * entropy
        + 0.20 * brightness_score * 10
        + 0.15 * contrast * 10
        + 0.35 * noise_score * 10
    )


def _no_ref_metrics(bgr_img: np.ndarray) -> dict:
    """Compute no-reference metrics: histogram uniformity + entropy (L channel)."""
    lab  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L    = cv2.split(lab)[0].ravel().astype(np.float32)
    hist, _ = np.histogram(L, bins=256, range=(0, 256), density=True)
    hist = hist + 1e-10

    # Entropy: higher = more information spread
    entropy = float(-np.sum(hist * np.log2(hist)))

    # Uniformity: L1 distance from ideal uniform distribution
    # 1.0 = perfectly flat, 0.0 = all pixels same value
    uniform = np.ones(256) / 256.0
    l1_dist = float(np.sum(np.abs(hist - uniform)))
    uniformity = float(np.clip(1.0 - l1_dist / 2.0, 0.0, 1.0))

    return {"entropy": round(entropy, 4), "uniformity": round(uniformity, 4)}


def to_bytes(bgr_img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(bgr_to_rgb(bgr_img)).save(buf, format="PNG")
    return buf.getvalue()


# ── Image input ────────────────────────────────────────────────────────────────
input_image = None

if input_mode == "Upload from file":
    uploaded = st.file_uploader(
        "Upload a low-light or poorly exposed image", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        input_image = PILImage.open(uploaded)
else:
    camera_shot = st.camera_input("Take a photo")
    if camera_shot:
        input_image = PILImage.open(camera_shot)

# ── Ground truth input (optional) ─────────────────────────────────────────────
gt_image = None
if has_ground_truth:
    gt_uploaded = st.file_uploader(
        "Upload ground truth (normal-light) image", type=["jpg", "jpeg", "png"], key="gt"
    )
    if gt_uploaded:
        gt_image = PILImage.open(gt_uploaded)

# ── Processing ─────────────────────────────────────────────────────────────────
if input_image is not None:
    bgr = pil_to_bgr(input_image)

    # Auto mode: predict params + exposure label
    exposure_label = None
    if clahe_mode == "Auto (ML)":
        with st.spinner("Running ML inference..."):
            try:
                params      = predict_params(bgr)
                clip_limit  = params["clip_limit"]
                tile_size   = params["tile_size"]
                tile_size_n = tile_size[0]
                exposure_label = predict_exposure(bgr)
            except Exception as e:
                st.warning(f"ML inference failed: {e}. Falling back to manual params.")

    with st.spinner("Enhancing image..."):
        ghe_result   = apply_ghe(bgr)
        clahe_result = apply_clahe(bgr, clip_limit=clip_limit, tile_size=tile_size)

    # ── Exposure label badge ───────────────────────────────────────────────────
    if exposure_label:
        badge_color = {
            "underexposed": "🔵",
            "normal":       "🟢",
            "overexposed":  "🟡",
        }.get(exposure_label, "⚪")
        st.info(
            f"{badge_color} ML detected: **{exposure_label.upper()}** "
            f"| Auto CLAHE params: clip={clip_limit}, tile={tile_size_n}×{tile_size_n}"
        )
        if exposure_label == "normal":
            st.warning("⚠️ Image is already well-exposed. Enhancement will be minimal — results may look similar to the original.")

    # ── Display: Before / After ────────────────────────────────────────────────
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(bgr_to_rgb(bgr), caption="Original", use_column_width=True)
    with col2:
        st.image(bgr_to_rgb(ghe_result), caption="After GHE", use_column_width=True)
        st.download_button(
            "⬇️ Download GHE", data=to_bytes(ghe_result),
            file_name="ghe_result.png", mime="image/png",
            use_container_width=True, key="dl_ghe",
        )
    with col3:
        st.image(
            bgr_to_rgb(clahe_result),
            caption=f"After CLAHE (clip={clip_limit}, tile={tile_size_n}×{tile_size_n})",
            use_column_width=True,
        )
        st.download_button(
            "⬇️ Download CLAHE", data=to_bytes(clahe_result),
            file_name="clahe_result.png", mime="image/png",
            use_container_width=True, key="dl_clahe",
        )

    # ── Metrics ────────────────────────────────────────────────────────────────
    ghe_metrics   = None
    clahe_metrics = None

    ghe_nr   = _no_ref_metrics(ghe_result)
    clahe_nr = _no_ref_metrics(clahe_result)
    orig_nr  = _no_ref_metrics(bgr)

    st.subheader("Evaluation Metrics")

    if gt_image is not None:
        gt_bgr = pil_to_bgr(gt_image)

        if gt_bgr.shape != bgr.shape:
            gt_bgr = cv2.resize(gt_bgr, (bgr.shape[1], bgr.shape[0]))

        ghe_metrics   = evaluate(gt_bgr, ghe_result)
        clahe_metrics = evaluate(gt_bgr, clahe_result)

        st.caption("**vs Ground Truth (PSNR / SSIM)**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("GHE — PSNR",   f"{ghe_metrics['psnr']:.2f} dB")
        m2.metric("GHE — SSIM",   f"{ghe_metrics['ssim']:.4f}")
        m3.metric("CLAHE — PSNR", f"{clahe_metrics['psnr']:.2f} dB")
        m4.metric("CLAHE — SSIM", f"{clahe_metrics['ssim']:.4f}")

        if clahe_metrics["ssim"] > 0.80:
            st.success("CLAHE SSIM target (> 0.80) achieved!")
        else:
            st.warning(
                f"CLAHE SSIM {clahe_metrics['ssim']:.4f} — below 0.80 target. "
                "Try adjusting parameters."
            )

    st.caption("**No-Reference Metrics (Histogram Uniformity & Entropy)**")
    n1, n2, n3, n4, n5, n6 = st.columns(6)
    n1.metric("Original — Uniformity", f"{orig_nr['uniformity']:.4f}")
    n2.metric("GHE — Uniformity",      f"{ghe_nr['uniformity']:.4f}",
              delta=f"{ghe_nr['uniformity'] - orig_nr['uniformity']:+.4f}")
    n3.metric("CLAHE — Uniformity",    f"{clahe_nr['uniformity']:.4f}",
              delta=f"{clahe_nr['uniformity'] - orig_nr['uniformity']:+.4f}")
    n4.metric("Original — Entropy",    f"{orig_nr['entropy']:.4f}")
    n5.metric("GHE — Entropy",         f"{ghe_nr['entropy']:.4f}",
              delta=f"{ghe_nr['entropy'] - orig_nr['entropy']:+.4f}")
    n6.metric("CLAHE — Entropy",       f"{clahe_nr['entropy']:.4f}",
              delta=f"{clahe_nr['entropy'] - orig_nr['entropy']:+.4f}")

    # ── Best Result ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⭐ Best Result")

    if ghe_metrics is not None and clahe_metrics is not None:
        # Objective: pick by SSIM (GT available)
        if clahe_metrics["ssim"] >= ghe_metrics["ssim"]:
            best_img    = clahe_result
            best_label  = f"CLAHE (clip={clip_limit}, tile={tile_size_n}×{tile_size_n})"
            best_reason = f"SSIM {clahe_metrics['ssim']:.4f} > GHE {ghe_metrics['ssim']:.4f}"
            best_file   = "best_clahe.png"
        else:
            best_img    = ghe_result
            best_label  = "GHE"
            best_reason = f"SSIM {ghe_metrics['ssim']:.4f} > CLAHE {clahe_metrics['ssim']:.4f}"
            best_file   = "best_ghe.png"

        st.success(f"**Winner: {best_label}** — {best_reason} *(based on SSIM vs ground truth)*")

    else:
        # Heuristic: no GT available
        ghe_score   = _heuristic_score(ghe_result)
        clahe_score = _heuristic_score(clahe_result)

        # Tiebreak: if GHE wins by < 5%, prefer CLAHE (less noisy perceptually)
        TIEBREAK_MARGIN = 0.15
        ghe_wins_by = (ghe_score - clahe_score) / max(clahe_score, 1e-6)

        if clahe_score >= ghe_score or ghe_wins_by < TIEBREAK_MARGIN:
            best_img    = clahe_result
            best_label  = f"CLAHE (clip={clip_limit}, tile={tile_size_n}×{tile_size_n})"
            tiebreak_note = " *(tiebreak: CLAHE preferred — less noise)*" if ghe_wins_by < TIEBREAK_MARGIN and ghe_score > clahe_score else ""
            best_reason = f"heuristic score {clahe_score:.3f} vs GHE {ghe_score:.3f}{tiebreak_note}"
            best_file   = "best_clahe.png"
        else:
            best_img    = ghe_result
            best_label  = "GHE"
            best_reason = f"heuristic score {ghe_score:.3f} vs CLAHE {clahe_score:.3f}"
            best_file   = "best_ghe.png"

        st.info(f"**Recommended: {best_label}** — {best_reason} *(heuristic: entropy + brightness + contrast + noise penalty)*")

    b1, b2 = st.columns([1.9, 1.1])
    with b1:
        st.image(bgr_to_rgb(best_img), caption=f"Best: {best_label}", use_column_width=True)
    with b2:
        st.download_button(
            label="⬇️ Download Best Result",
            data=to_bytes(best_img),
            file_name=best_file,
            mime="image/png",
            use_container_width=True,
        )

    # ── Histograms ─────────────────────────────────────────────────────────────
    if show_histogram:
        st.subheader("Histogram Comparison")
        fig = plot_comparison_histograms(bgr, ghe_result, clahe_result)
        st.pyplot(fig)


else:
    st.info("Upload an image or take a photo using the sidebar to get started.")
