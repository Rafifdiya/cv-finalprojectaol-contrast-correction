# Automated Contrast Correction System

Sistem koreksi kontras otomatis untuk foto *under-exposed* dan *over-exposed* menggunakan **Global Histogram Equalization (GHE)** dan **CLAHE**, dilengkapi klasifikasi eksposur berbasis Machine Learning (Random Forest).

> Tugas Akhir COMP7116001 — Computer Vision | Binus University Semester 4

---

## Fitur

- Upload foto atau ambil via kamera
- Mode **Auto (ML)** — parameter CLAHE dipilih otomatis oleh model
- Mode **Manual** — atur `clip_limit` dan `tile_size` via slider
- Tampilkan hasil GHE & CLAHE secara berdampingan
- Evaluasi metrik: PSNR, SSIM, Histogram Uniformity, Entropy
- Download hasil langsung dari app

## Pipeline

```
Input (BGR) → LAB → Equalize L channel (GHE / CLAHE) → LAB → Output (BGR)
```

## ML Model

| Model | Metode |
|---|---|
| Exposure Classifier | Random Forest |
| Clip Limit Predictor | Random Forest Regressor |
| Tile Size Predictor | Random Forest Classifier |

## Dataset

Dataset tidak diikutkan di repo. Download manual:

| Dataset | Digunakan untuk | Link |
|---|---|---|
| LOL (Low-Light Object) | Underexposed + normal pairs (485 train, 15 test) | [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) |
| SICE Part 2 | Overexposed images (186 gambar) | [GitHub](https://github.com/csjcai/SICE) |


## Cara Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan app
streamlit run app.py #atau menyesuaikan environment device
```

## Tim

| NIM | Nama |
|---|---|
| 2802486980 | Ahmad Daffa Hidayatullah |
| 2802493752 | Andrey Apriliady |
| 2802481626 | Ezra Mayurga |
| 2802484470 | Jason Firenze Trianto |
| 2802488185 | Keanu Stadeva |
| 2802484483 | Rafifdiya |

## Limitasi

- **Class imbalance** — Data overexposed (SICE): 186 sampel vs 485 untuk kelas under/normal
- **Gap ke deep learning** — SSIM 0.4519 vs metode DL state-of-the-art (0.55–0.75)
- **Distribusi terbatas** — Hanya diuji pada LOL dataset (indoor low-light); belum divalidasi di kondisi lain

## Future Work

- Integrasi modul neural enhancement ringan (Zero-DCE) sebagai mode opsional
- Ekspansi data overexposed di luar SICE Part2 untuk generalisasi classifier
- Ekstensi ke video stream dengan temporal consistency
- Mobile deployment via ONNX / TFLite
