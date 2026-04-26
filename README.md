# FDE MedLogic — Full System Guide

## Project Structure

```
FDE_System/
├── backend/
│   ├── app.py                          ← Flask entry point (updated: registers image_bp)
│   ├── database.py                     ← SQLite init (updated: adds image_predictions table)
│   ├── fde_system.db                   ← SQLite database (auto-created)
│   ├── models/                         ← ★ CNN weights live here after training
│   │   ├── fde_cnn_model.h5            ←   created by training script
│   │   └── model_meta.json             ←   class indices + accuracy metrics
│   ├── ml/
│   │   ├── fde_model.py                ← WCIS algorithm (unchanged)
│   │   ├── medicine_data.py            ← 1,600-medicine loader (unchanged)
│   │   └── image_recognition/          ← ★ NEW module
│   │       ├── __init__.py             ←   public API exports
│   │       ├── fde_cnn_model.py        ←   MobileNetV2 architecture + training pipeline
│   │       ├── predictor.py            ←   inference engine + GRAD-CAM
│   │       └── train_fde_cnn.py        ←   standalone training CLI script
│   └── routes/
│       ├── auth.py                     ← (unchanged)
│       ├── patient.py                  ← (unchanged)
│       ├── predict.py                  ← WCIS endpoints (unchanged)
│       └── image.py                    ← ★ NEW — CNN image API endpoints
├── frontend/
│   └── index.html                      ← updated: added Image Scan tab + JS
├── data/
│   └── medicines_1600_detailed.xlsx
└── requirements.txt                    ← updated: added tensorflow, pillow, matplotlib
```

---

## Part 1 — Standard Backend (no CNN)

```bash
cd FDE_System/backend
pip install -r ../requirements.txt --break-system-packages
python app.py
```

The server runs fine without a trained CNN. Image endpoints return a clear
"model not trained" message with setup instructions if no weights exist.

---

## Part 2 — CNN Image Recognition Setup

### Step 1 — Install ML dependencies

```bash
pip install tensorflow pillow matplotlib numpy --break-system-packages
```

GPU strongly recommended. Without one, full DermNet training takes ~2-4 hours.
Free GPU option: Kaggle Notebooks (T4) or Google Colab.

### Step 2 — Download DermNet

```bash
pip install kaggle --break-system-packages
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d shubhamgoel27/dermnet
unzip dermnet.zip -d /path/to/dermnet
```

### Step 3 — Validate dataset

```bash
python backend/ml/image_recognition/train_fde_cnn.py \
  --dataset /path/to/dermnet --check-only
```

### Step 4 — Train

```bash
python backend/ml/image_recognition/train_fde_cnn.py \
  --dataset /path/to/dermnet
# Quick test: add --epochs 5 --fine-tune-epochs 3
```

Weights saved to `backend/models/fde_cnn_model.h5` automatically.

---

## Part 3 — API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/image/model_status` | Model readiness + accuracy |
| POST | `/api/image/predict_fde` | Predict FDE from uploaded image |
| POST | `/api/image/predict_fde_base64` | Predict from base64 string |
| POST | `/api/image/gradcam` | Prediction + GRAD-CAM heatmap |
| POST | `/api/image/predict_fde_combined` | CNN + WCIS combined score |

```bash
# Quick curl tests
curl http://localhost:5000/api/image/model_status
curl -X POST http://localhost:5000/api/image/predict_fde -F "file=@skin.jpg"
curl -X POST http://localhost:5000/api/image/gradcam -F "file=@skin.jpg"
```

---

## Part 4 — CNN Architecture

```
Input 224x224x3
  → MobileNetV2 (ImageNet pretrained, frozen Phase 1 / top-30 unfrozen Phase 2)
  → GlobalAveragePooling2D
  → Dense(256) + BatchNorm + ReLU + Dropout(0.4)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(num_classes) + Softmax
  → FDE Score = sum(P(class_i) * weight_i)
       FDE_HIGH=1.0  FDE_MODERATE=0.6  DIFFERENTIAL=0.2  NOT_FDE=0.0
```

Risk thresholds: HIGH >= 0.65 | MODERATE >= 0.40 | LOW < 0.40

---

## Part 5 — Combined Scoring

```
combined = 0.5 * CNN_score + 0.5 * (WCIS_score / 10)
```

Use `/api/image/predict_fde_combined` with `patient_id` + `components`
to get the most complete FDE assessment — skin image evidence fused with
personalized drug history from WCIS.

---

## Disclaimer

Decision-support tool only. All outputs require clinician review.
Not validated for clinical use. Do not use as sole basis for treatment.
