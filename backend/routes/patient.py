from flask import Blueprint, request, jsonify
from database import get_db

patient_bp = Blueprint("patient", __name__, url_prefix="/api/patient")


@patient_bp.route("/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    conn = get_db()
    c = conn.cursor()

    patient = c.execute(
        "SELECT patient_id, name, age, gender, blood_group, contact, created_at FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()

    if not patient:
        conn.close()
        return jsonify({"error": "Patient not found"}), 404

    # Get dosage history
    dosages = c.execute(
        "SELECT * FROM dosage_history WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()

    # Get component risk scores
    scores = c.execute(
        "SELECT component, score, fde_count FROM component_scores WHERE patient_id = ? ORDER BY score DESC",
        (patient_id,)
    ).fetchall()

    # Get FDE events
    fde_events = c.execute(
        "SELECT * FROM fde_events WHERE patient_id = ? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()

    conn.close()

    return jsonify({
        "patient": dict(patient),
        "dosage_history": [dict(d) for d in dosages],
        "risk_profile": [dict(s) for s in scores],
        "fde_events": [dict(e) for e in fde_events]
    })


@patient_bp.route("/<patient_id>/dosage", methods=["POST"])
def add_dosage(patient_id):
    """Add a new medication dosage entry (without FDE marking - just logging)."""
    data = request.json
    required = ["medicine_name", "components"]
    for f in required:
        if not data.get(f):
            return jsonify({"error": f"'{f}' is required"}), 400

    components_str = ",".join([c.strip().lower() for c in data["components"]])

    conn = get_db()
    c = conn.cursor()

    c.execute('''
        INSERT INTO dosage_history 
        (patient_id, medicine_name, components, dosage, frequency, duration, fde_occurred, prescribed_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
    ''', (
        patient_id,
        data["medicine_name"],
        components_str,
        data.get("dosage", ""),
        data.get("frequency", ""),
        data.get("duration", ""),
        data.get("prescribed_date", ""),
        data.get("notes", "")
    ))

    new_id = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"message": "Dosage added", "dosage_id": new_id}), 201


@patient_bp.route("/<patient_id>/risk_profile", methods=["GET"])
def get_risk_profile(patient_id):
    """Return sorted component risk scores for the patient."""
    conn = get_db()
    scores = conn.execute(
        "SELECT component, score, fde_count FROM component_scores WHERE patient_id = ? ORDER BY score DESC",
        (patient_id,)
    ).fetchall()
    conn.close()
    return jsonify({"risk_profile": [dict(s) for s in scores]})
