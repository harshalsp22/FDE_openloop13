"""
FDE Image Preprocessor & Inference Engine
==========================================
Handles:
  - Raw image decoding (JPEG, PNG, WEBP)
  - Resizing + normalisation to MobileNetV2 spec (224×224, [0,1])
  - Raw-probability → FDE risk score mapping
  - GRAD-CAM heatmap generation (explains which skin region triggered prediction)
  - Model status reporting
"""

import io
import base64
import numpy as np
from pathlib import Path

from .fde_cnn_model import (
    IMG_SIZE, FDE_CLASS_MAPPING,
    load_model, load_meta,
)

# ── FDE confidence thresholds ────────────────────────────────────────────────
CONF_HIGH     = 0.65   # ≥ 65 % → HIGH
CONF_MODERATE = 0.40   # ≥ 40 % → MODERATE
# < 40 % → LOW


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> "np.ndarray":
    """
    Decode raw image bytes → normalised numpy array ready for CNN.

    Steps:
        1. Decode JPEG / PNG / WEBP via Pillow
        2. Convert to RGB (handles RGBA, greyscale)
        3. Resize to 224×224 (MobileNetV2 standard)
        4. Normalise pixel values to [0, 1]
        5. Add batch dimension → shape (1, 224, 224, 3)

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32
    """
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def preprocess_base64(b64_string: str) -> "np.ndarray":
    """Convenience wrapper — decode a base64 data-URI then preprocess."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    return preprocess_image(raw)


# ─────────────────────────────────────────────────────────────────────────────
def map_predictions_to_fde(class_indices: dict, raw_probs: "np.ndarray") -> dict:
    """
    Convert raw DermNet class probabilities → FDE-specific risk score.

    Aggregation formula:
        fde_score = Σ p(class_i) × weight(class_i)

        Weights:
            FDE_HIGH     → 1.0
            FDE_MODERATE → 0.6
            DIFFERENTIAL → 0.2
            NOT_FDE      → 0.0

    Args:
        class_indices : dict mapping class_name → index (from training generator)
        raw_probs     : probability array of shape (num_classes,)

    Returns:
        dict with fde_score, risk_level, recommendation, top_predictions, etc.
    """
    WEIGHTS = {
        "FDE_HIGH":     1.0,
        "FDE_MODERATE": 0.6,
        "DIFFERENTIAL": 0.2,
        "NOT_FDE":      0.0,
    }

    idx_to_class = {v: k for k, v in class_indices.items()}

    fde_score   = 0.0
    top_classes = []
    fde_classes = []

    for idx, prob in enumerate(raw_probs):
        class_name = idx_to_class.get(idx, f"class_{idx}")
        fde_cat    = FDE_CLASS_MAPPING.get(class_name, "NOT_FDE")
        weight     = WEIGHTS.get(fde_cat, 0.0)
        fde_score += float(prob) * weight

        entry = {
            "class":        class_name,
            "probability":  round(float(prob) * 100, 2),
            "fde_category": fde_cat,
            "weight":       weight,
        }
        top_classes.append(entry)
        if fde_cat in ("FDE_HIGH", "FDE_MODERATE"):
            fde_classes.append(entry)

    top_classes.sort(key=lambda x: x["probability"], reverse=True)
    fde_classes.sort(key=lambda x: x["probability"], reverse=True)

    # ── Risk classification ───────────────────────────────────────────────────
    if fde_score >= CONF_HIGH:
        risk_level     = "HIGH"
        risk_color     = "#ef4444"
        recommendation = (
            "⛔ HIGH suspicion of Fixed Drug Eruption. "
            "Skin pattern strongly correlates with a drug-induced eruption. "
            "Immediate dermatologist review and medicine causality assessment recommended. "
            "Suspected drug should be withdrawn pending evaluation."
        )
    elif fde_score >= CONF_MODERATE:
        risk_level     = "MODERATE"
        risk_color     = "#f59e0b"
        recommendation = (
            "⚠️ MODERATE suspicion. Image shows features consistent with FDE "
            "but differential diagnoses cannot be excluded. "
            "Clinical correlation with detailed drug history is essential. "
            "Consider dermatology referral."
        )
    else:
        risk_level     = "LOW"
        risk_color     = "#22c55e"
        recommendation = (
            "✅ LOW FDE suspicion from image alone. "
            "Pattern is more consistent with a non-drug-related dermatosis. "
            "Continue standard clinical evaluation."
        )

    top3 = top_classes[:3]
    explanation_parts = [f"{e['class']} ({e['probability']:.1f}%)" for e in top3]
    explanation = (
        f"CNN identified top matches: {', '.join(explanation_parts)}. "
        f"Weighted FDE score: {fde_score:.3f} "
        f"(thresholds — HIGH ≥ {CONF_HIGH}, MODERATE ≥ {CONF_MODERATE})."
    )

    return {
        "fde_score":       round(fde_score, 4),
        "risk_level":      risk_level,
        "risk_color":      risk_color,
        "recommendation":  recommendation,
        "top_predictions": top_classes[:5],
        "fde_indicators":  fde_classes[:3],
        "explanation":     explanation,
        "algorithm_note":  (
            "MobileNetV2 CNN trained on DermNet 23-class dataset. "
            "FDE score = weighted sum of class probabilities by FDE relevance."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
def predict_fde(image_bytes: bytes, model=None, meta: dict = None) -> dict:
    """
    End-to-end FDE prediction from raw image bytes.

    Args:
        image_bytes : raw bytes of a skin-lesion image (JPEG / PNG / WEBP)
        model       : pre-loaded Keras model (avoids repeated disk I/O)
        meta        : pre-loaded model metadata dict

    Returns:
        Full prediction dict — fde_score, risk_level, top_predictions, etc.
    """
    if model is None:
        model = load_model()
    if meta is None:
        meta = load_meta()

    class_indices = meta.get("class_indices", {})
    img_array     = preprocess_image(image_bytes)

    raw_output = model.predict(img_array, verbose=0)   # (1, num_classes)
    probs      = raw_output[0]

    result = map_predictions_to_fde(class_indices, probs)
    result["model_info"] = {
        "architecture": "MobileNetV2 + Custom Dense Head",
        "dataset":      "DermNet (23 skin conditions)",
        "img_size":     f"{IMG_SIZE}×{IMG_SIZE}",
        "test_accuracy": meta.get("test_accuracy", "N/A"),
        "test_auc":      meta.get("test_auc", "N/A"),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
def generate_gradcam(image_bytes: bytes, model=None):  # -> str | None (py3.10+)
    """
    Generate a GRAD-CAM heatmap overlay showing which skin region triggered
    the FDE prediction — critical for clinical explainability.

    GRAD-CAM Algorithm:
        1. Forward pass → record last conv-layer activations
        2. Compute gradient of predicted class score w.r.t. those activations
        3. Global-average-pool gradients → per-channel importance weights
        4. Weighted sum of activation maps → coarse heat map
        5. Upsample to input resolution, overlay on original image (jet colormap)

    Returns:
        Base64-encoded PNG data-URI, or None if generation fails.
    """
    try:
        import tensorflow as tf
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm

        if model is None:
            model = load_model()

        # Find last conv layer inside the MobileNetV2 sub-model
        mobilenet_layer = None
        for lyr in model.layers:
            if "mobilenetv2" in lyr.name.lower():
                mobilenet_layer = lyr
                break

        if mobilenet_layer is None:
            return None

        last_conv_name = None
        for lyr in reversed(mobilenet_layer.layers):
            if hasattr(lyr, "filters") or "conv" in lyr.name.lower():
                last_conv_name = lyr.name
                break

        if last_conv_name is None:
            return None

        # Build GRAD-CAM sub-model
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                mobilenet_layer.get_layer(last_conv_name).output,
                model.output,
            ],
        )

        img_array = preprocess_image(image_bytes)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_score     = predictions[:, predicted_class]

        grads       = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        # Overlay on original image
        img     = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        img     = img.resize((IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img)

        hm_img  = PILImage.fromarray(np.uint8(heatmap * 255))
        hm_img  = hm_img.resize((IMG_SIZE, IMG_SIZE), PILImage.LANCZOS)
        hm_arr  = np.array(hm_img) / 255.0

        colormap = cm.get_cmap("jet")
        hm_color = np.uint8(colormap(hm_arr)[:, :, :3] * 255)

        blended  = np.uint8(0.6 * img_arr + 0.4 * hm_color)

        buf = io.BytesIO()
        PILImage.fromarray(blended).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as exc:
        print(f"[GRAD-CAM] Failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def get_model_status() -> dict:
    """Return current model status — trained flag, accuracy, thresholds, paths."""
    meta         = load_meta()
    model_exists = (Path(__file__).parent.parent.parent / "models" / "fde_cnn_model.h5").exists()

    return {
        "model_trained":  model_exists,
        "test_accuracy":  meta.get("test_accuracy"),
        "test_auc":       meta.get("test_auc"),
        "num_classes":    meta.get("num_classes"),
        "architecture":   "MobileNetV2 + Custom Dense Head",
        "dataset":        "DermNet (Kaggle: shubhamgoel27/dermnet)",
        "input_size":     f"{IMG_SIZE}×{IMG_SIZE}×3",
        "fde_thresholds": {
            "HIGH":     f"≥ {CONF_HIGH}",
            "MODERATE": f"≥ {CONF_MODERATE}",
            "LOW":      f"< {CONF_MODERATE}",
        },
        "setup_instructions": (
            "1. kaggle datasets download -d shubhamgoel27/dermnet\n"
            "2. unzip dermnet.zip -d /path/to/dermnet\n"
            "3. pip install tensorflow pillow matplotlib --break-system-packages\n"
            "4. python backend/ml/image_recognition/train_fde_cnn.py --dataset /path/to/dermnet"
        ) if not model_exists else None,
    }
