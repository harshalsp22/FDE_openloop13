"""
FDE Image Recognition API Routes
==================================
Endpoints:
    GET  /api/image/model_status            → model readiness, accuracy, thresholds
    POST /api/image/predict_fde             → predict FDE from uploaded image file
    POST /api/image/predict_fde_base64      → predict FDE from base64 image
    POST /api/image/gradcam                 → GRAD-CAM heatmap for explainability
    POST /api/image/predict_fde_combined    → CNN image result + WCIS drug score

All prediction endpoints return:
    {
        "fde_score":       float  (0–1 weighted FDE probability),
        "risk_level":      "HIGH" | "MODERATE" | "LOW",
        "risk_color":      "#ef4444" | "#f59e0b" | "#22c55e",
        "recommendation":  str,
        "top_predictions": [...],
        "fde_indicators":  [...],
        "explanation":     str,
        "model_info":      {...}
    }
"""

from flask import Blueprint, request, jsonify
from database import get_db

image_bp = Blueprint("image", __name__, url_prefix="/api/image")

# Lazy-load model to avoid slowing app startup
_model_cache = None
_meta_cache  = None


def _get_model():
    """Lazy-load and cache the trained CNN model + metadata."""
    global _model_cache, _meta_cache
    if _model_cache is None:
        try:
            from ml.image_recognition import load_model, load_meta
            _model_cache = load_model()
            _meta_cache  = load_meta()
        except FileNotFoundError as exc:
            return None, None, str(exc)
        except Exception as exc:
            return None, None, f"Model load error: {exc}"
    return _model_cache, _meta_cache, None


def _patient_exists(pid: str) -> bool:
    conn = get_db()
    row  = conn.execute("SELECT id FROM patients WHERE patient_id=?", (pid,)).fetchone()
    conn.close()
    return row is not None


def _not_trained_response():
    return jsonify({
        "error":   "CNN model not trained yet.",
        "detail":  "No model weights found at backend/models/fde_cnn_model.h5",
        "action":  (
            "1. Download DermNet: kaggle datasets download -d shubhamgoel27/dermnet\n"
            "2. unzip dermnet.zip -d /path/to/dermnet\n"
            "3. pip install tensorflow pillow matplotlib --break-system-packages\n"
            "4. python backend/ml/image_recognition/train_fde_cnn.py --dataset /path/to/dermnet"
        )
    }), 503


# ─────────────────────────────────────────────────────────────────────────────
@image_bp.route("/model_status", methods=["GET"])
def model_status():
    """
    Check if the CNN model is trained and return its accuracy, architecture,
    dataset info, and FDE risk thresholds.
    """
    try:
        from ml.image_recognition import get_model_status
        return jsonify(get_model_status())
    except Exception as exc:
        return jsonify({"error": str(exc), "model_trained": False}), 500


# ─────────────────────────────────────────────────────────────────────────────
@image_bp.route("/predict_fde", methods=["POST"])
def predict_fde_upload():
    """
    Predict FDE risk from an uploaded image file.

    Request: multipart/form-data
        file        : image (JPEG / PNG / WEBP)  ← required
        patient_id  : patient ID                  ← optional (logs result to DB)

    Returns: prediction dict (see module docstring above)
    """
    if "file" not in request.files:
        return jsonify({
            "error": "No image file. Send multipart/form-data with key 'file'."
        }), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        return jsonify({
            "error": f"Unsupported type: {file.content_type}. Use JPEG, PNG, or WEBP."
        }), 400

    image_bytes = file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10 MB."}), 413

    model, meta, err = _get_model()
    if err:
        return _not_trained_response()

    try:
        from ml.image_recognition import predict_fde
        result = predict_fde(image_bytes, model=model, meta=meta)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    patient_id = request.form.get("patient_id")
    if patient_id and _patient_exists(patient_id):
        _log_image_prediction(patient_id, result, file.filename)

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
@image_bp.route("/predict_fde_base64", methods=["POST"])
def predict_fde_base64():
    """
    Predict FDE risk from a base64-encoded image (useful for web/mobile clients).

    Request JSON:
        {
            "image":      "data:image/jpeg;base64,/9j/4AAQ...",
            "patient_id": "P001"    ← optional
        }
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "JSON body with 'image' key required."}), 400

    try:
        import base64 as b64lib
        b64_str = data["image"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        image_bytes = b64lib.b64decode(b64_str)
    except Exception:
        return jsonify({"error": "Invalid base64 image data."}), 400

    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10 MB."}), 413

    model, meta, err = _get_model()
    if err:
        return _not_trained_response()

    try:
        from ml.image_recognition import predict_fde
        result = predict_fde(image_bytes, model=model, meta=meta)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    patient_id = data.get("patient_id")
    if patient_id and _patient_exists(patient_id):
        _log_image_prediction(patient_id, result, "base64_upload")

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
@image_bp.route("/gradcam", methods=["POST"])
def gradcam():
    """
    Generate GRAD-CAM visualisation showing which skin region the CNN
    is responding to — critical for clinical trust and explainability.

    Request: multipart/form-data
        file : image (JPEG / PNG / WEBP)

    Response:
        {
            "gradcam_image": "data:image/png;base64,...",
            "prediction":    { ...same as predict_fde... }
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    image_bytes = request.files["file"].read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10 MB."}), 413

    model, meta, err = _get_model()
    if err:
        return _not_trained_response()

    try:
        from ml.image_recognition import predict_fde, generate_gradcam
        prediction  = predict_fde(image_bytes, model=model, meta=meta)
        gradcam_b64 = generate_gradcam(image_bytes, model=model)
    except Exception as exc:
        return jsonify({"error": f"GRAD-CAM failed: {exc}"}), 500

    return jsonify({
        "gradcam_image": gradcam_b64,
        "prediction":    prediction,
    })


# ─────────────────────────────────────────────────────────────────────────────
@image_bp.route("/predict_fde_combined", methods=["POST"])
def predict_fde_combined():
    """
    Combined analysis — CNN image score  +  WCIS medicine risk score.
    Produces the most complete FDE assessment available.

    Request: multipart/form-data
        file           : image file         ← required
        patient_id     : patient ID         ← required
        medicine_name  : medicine name      ← optional
        components     : comma-separated    ← optional (enables WCIS scoring)

    Response:
        {
            "image_analysis":   { CNN result },
            "wcis_risk":        { WCIS result or null },
            "combined_verdict": "HIGH" | "MODERATE" | "LOW",
            "combined_score":   float,
            "combined_color":   hex,
            "explanation":      str,
            "scoring_method":   str
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    patient_id = request.form.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id required for combined analysis."}), 400
    if not _patient_exists(patient_id):
        return jsonify({"error": "Patient not found."}), 404

    image_bytes = request.files["file"].read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large. Max 10 MB."}), 413

    model, meta, err = _get_model()
    if err:
        return _not_trained_response()

    # ── CNN image prediction ──────────────────────────────────────────────────
    try:
        from ml.image_recognition import predict_fde
        img_result = predict_fde(image_bytes, model=model, meta=meta)
    except Exception as exc:
        return jsonify({"error": f"Image prediction failed: {exc}"}), 500

    # ── WCIS medicine risk (only if components provided) ─────────────────────
    wcis_result   = None
    components_raw = request.form.get("components", "")
    if components_raw:
        components = [c.strip() for c in components_raw.split(",") if c.strip()]
        try:
            from ml.fde_model import calculate_risk
            wcis_result = calculate_risk(patient_id, components)
        except Exception as exc:
            wcis_result = {"error": str(exc)}

    # ── Combine scores ────────────────────────────────────────────────────────
    img_score = img_result.get("fde_score", 0.0)

    if wcis_result and "risk_score" in wcis_result:
        wcis_norm = min(wcis_result["risk_score"] / 10.0, 1.0)
        combined  = round(0.5 * img_score + 0.5 * wcis_norm, 4)
        method    = "Image CNN (50 %) + WCIS medicine score (50 %)"
    else:
        combined = round(img_score, 4)
        method   = "Image CNN only (no medicine components provided)"

    if combined >= 0.65:
        verdict       = "HIGH"
        verdict_color = "#ef4444"
        verdict_text  = (
            "⛔ HIGH combined FDE risk. Both image pattern and drug history "
            "indicate Fixed Drug Eruption. Immediate clinical review required."
        )
    elif combined >= 0.40:
        verdict       = "MODERATE"
        verdict_color = "#f59e0b"
        verdict_text  = (
            "⚠️ MODERATE combined risk. Clinical correlation with detailed "
            "drug history strongly recommended. Dermatology referral advised."
        )
    else:
        verdict       = "LOW"
        verdict_color = "#22c55e"
        verdict_text  = (
            "✅ LOW combined risk from image and drug history. "
            "Routine monitoring advised."
        )

    return jsonify({
        "image_analysis":   img_result,
        "wcis_risk":        wcis_result,
        "combined_verdict": verdict,
        "combined_score":   combined,
        "combined_color":   verdict_color,
        "explanation":      verdict_text,
        "scoring_method":   method,
    })


# ─────────────────────────────────────────────────────────────────────────────
def _log_image_prediction(patient_id: str, result: dict, filename: str):
    """Persist an image prediction to the DB for audit trail (non-critical)."""
    try:
        conn = get_db()
        conn.execute(
            """INSERT INTO image_predictions
               (patient_id, filename, fde_score, risk_level, top_class, created_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                patient_id,
                filename,
                result.get("fde_score", 0),
                result.get("risk_level", "UNKNOWN"),
                (result.get("top_predictions") or [{}])[0].get("class", "unknown"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass   # Non-critical — never fail a prediction because of a logging error
