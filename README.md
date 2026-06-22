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

| Model | Metode | Performa |
|---|---|---|
| Exposure Classifier | Random Forest | Accuracy 96.28% |
| Clip Limit Predictor | Random Forest Regressor | MAE 0.0837 |
| Tile Size Predictor | Random Forest Classifier | Accuracy 99.59% |

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
