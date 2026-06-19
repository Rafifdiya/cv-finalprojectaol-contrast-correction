# AUTOMATED CONTRAST CORRECTION SYSTEM
### Untuk Fotografi Under/Over-Exposed Menggunakan Optimized Histogram Equalization

**Mata Kuliah:** COMP7116001 — Computer Vision  
**Dosen:** Fiqri Ramadhan Tambunan, S.Kom., M.Kom  

**Group 2:**
- Ahmad Daffa Hidayatullah
- Andrey Apriliady
- Ezra Mayurga
- Jason Firenze Trianto
- Keanu Stadeva Reeves
- Rafifdiya

---

## Slide 2 — Problem Statement

### Masalah yang Diangkat

- **Foto Underexposed**  
  Foto terlihat terlalu gelap akibat kurangnya cahaya yang masuk ke sensor kamera.

- **Foto Overexposed**  
  Gambar tampak sangat cerah atau pucat akibat terlalu banyak cahaya yang masuk ke sensor kamera.

- **Histogram Equalization Klasik**  
  Butuh parameter manual tiap gambar.

---

## Slide 3 — Objektif & Dataset

### Tujuan
- Bangun pipeline koreksi kontras otomatis (GHE + CLAHE)
- ML Classifier: deteksi kondisi exposure secara otomatis
- ML Predictor: pilih parameter CLAHE optimal per gambar

### Dataset
- **Wei et al., BMVC 2018** — LOL Low-Light Dataset  
  via Kaggle: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset/data
- 485 training pairs + 15 test pairs
- Format: pasangan PNG (low-light ↔ normal-light)

- **SICE Part2** — Single Image Contrast Enhancement Dataset  
  https://github.com/csjcai/SICE  
- 186 real overexposed images (untuk training ML Classifier kelas overexposed)

---

## Slide 4 — Metode CV: GHE & CLAHE

### GHE (Global Histogram Equalization)
- Equalizes intensitas piksel secara global
- Formula: `s = (L−1) × CDF(r)`
- Cepat dan mudah, dipakai sebagai baseline
- **Kelemahan:** rawan over-enhance & noise

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Equalize per tile lokal (grid N×N)
- **clip limit:** batas redistribusi histogram
- **tile size:** ukuran grid (4×4, 8×8, 16×16)
- Lebih natural dan noise terkontrol

---

## Slide 5 — Pipeline Sistem

```
Input Foto
    ↓
Ekstrak 10 Fitur Histogram (L channel)
    ↓
ML Classifier → Under / Over / Normal
    ↓
ML Predictor → clip limit + tile size
    ↓
CLAHE & GHE di L channel
    ↓
Output + Metrics (PSNR, SSIM, Entropy)
```

---

## Slide 6 — Hasil Eksperimen

| Method         | clip limit | tile size | PSNR (dB) | SSIM   |
|----------------|------------|-----------|-----------|--------|
| GHE (baseline) | —          | —         | 15.33     | 0.4431 |
| CLAHE          | 1          | 4×4       | 8.61      | 0.2933 |
| CLAHE          | 2          | 4×4       | 9.27      | 0.3659 |
| CLAHE          | 3          | 4×4       | 9.91      | 0.4171 |
| **CLAHE ★ BEST** | **4**    | **4×4**   | **10.54** | **0.4519** |

> **CLAHE clip=4.0 tile=4×4 unggul SSIM (0.4519 > 0.4431 GHE)**  
> SSIM lebih relevan untuk kualitas visual manusia.

---

## Slide 7 — Machine Learning Component

### Auto Exposure CLASSIFIER
| Parameter | Value |
|-----------|-------|
| Model     | Random Forest (100 trees) |
| Input     | 10 fitur histogram (L channel) |
| Output    | under / over / normal |
| Akurasi   | **96.28%** ✓ |

### Auto Parameter PREDICTOR
| Parameter | Value |
|-----------|-------|
| Model     | RF Regressor + RF Classifier |
| Input     | 10 fitur histogram (L channel) |
| Output    | clip limit + tile size |
| MAE clip  | **0.0837** |
| Tile Acc  | **99.59%** ✓ |

---

## Slide 8 — Output Demo Snapshot

Fitur yang ditampilkan pada antarmuka demo:

- **Label Exposure Otomatis** — sistem mendeteksi kondisi gambar (contoh: `ML detected: UNDEREXPOSED | Auto CLAHE params: clip=4.0, tile=4×4`)
- **Perbandingan Hasil** — tampilan tiga gambar: Original / After GHE / After CLAHE
- **Evaluation Metrics (vs Ground Truth)** *(1 sample image)*
  - GHE — PSNR: 10.46 dB | GHE — SSIM: 0.3172
  - CLAHE — PSNR: 10.17 dB | CLAHE — SSIM: 0.2970
- **No-Reference Metrics (Histogram Uniformity & Entropy)**

| Metric      | Original | GHE    | CLAHE  |
|-------------|----------|--------|--------|
| Uniformity  | 0.3647   | 0.3497 | 0.5329 |
| Entropy     | 5.4963   | 5.4124 | 6.7161 |

---

## Slide 9 — Kesimpulan

### Hasil
- Pipeline GHE & CLAHE berhasil koreksi kontras otomatis
- Parameter optimal: CLAHE clip 4.0 dan tile 4×4 (SSIM = 0.4519)
- ML Classifier akurasi: **96.28%** (data overexposed = real SICE Part2)
- ML Predictor MAE: **0.08** → parameter dipilih tanpa tuning manual
- Web app Streamlit berjalan: upload, kamera, download, Auto/Manual mode

### Limitasi
1. **Class imbalance** — Data overexposed (SICE): 186 sampel vs 485 untuk kelas under/normal
2. **Gap ke deep learning** — SSIM 0.4519 vs metode DL state-of-the-art (0.55–0.75)
3. **Distribusi terbatas** — Hanya diuji pada LOL dataset (indoor low-light); belum divalidasi di kondisi lain

**Future Work:**  
Deep learning (RetinexNet/EnlightenGAN), data overexposed lebih besar, dukungan video real-time

---

## Slide 10 — Thank You

**DEMO:**
```bash
python -m streamlit run app.py
```
