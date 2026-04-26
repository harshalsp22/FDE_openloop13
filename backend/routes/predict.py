from flask import Blueprint, request, jsonify
from database import get_db
from ml.fde_model import (
    calculate_risk, process_fde_event,
    find_safe_alternatives, get_component_scores
)
from ml.medicine_data import medicine_df, col_name, col_components, col_use

predict_bp = Blueprint("predict", __name__, url_prefix="/api/predict")

def _patient_exists(pid):
    conn = get_db()
    p = conn.execute("SELECT id FROM patients WHERE patient_id=?", (pid,)).fetchone()
    conn.close()
    return p is not None

# ── Search medicines from real DB ──────────────────────
@predict_bp.route("/search_medicine", methods=["GET"])
def search_medicine():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify({"results": []})
    matches = medicine_df[medicine_df[col_name].str.lower().str.contains(q, na=False)]
    results = []
    for _, row in matches.head(10).iterrows():
        results.append({
            "name":       row[col_name],
            "used_for":   row[col_use],
            "components": row[col_components][:8],
            "risk_level": str(row.get("Risk_Level", "")).strip(),
            "type":       str(row.get("Type", "")).strip()
        })
    return jsonify({"results": results})

# ── Get medicine details ───────────────────────────────
@predict_bp.route("/medicine_info", methods=["GET"])
def medicine_info():
    name = request.args.get("name", "").strip()
    row = medicine_df[medicine_df[col_name].str.lower() == name.lower()]
    if row.empty:
        return jsonify({"error": "Not found"}), 404
    r = row.iloc[0]
    return jsonify({
        "name":        r[col_name],
        "used_for":    r[col_use],
        "components":  r[col_components],
        "allergens":   r.get("Allergen_Flags", []),
        "risk_level":  str(r.get("Risk_Level", "")),
        "type":        str(r.get("Type", ""))
    })

# ── Check FDE risk ──────────────────────────────────────
@predict_bp.route("/check_risk", methods=["POST"])
def check_risk():
    data = request.json
    pid  = data.get("patient_id")
    med  = data.get("medicine_name", "")
    comps = data.get("components", [])

    if not pid or not comps:
        return jsonify({"error": "patient_id and components required"}), 400
    if not _patient_exists(pid):
        return jsonify({"error": "Patient not found"}), 404

    risk = calculate_risk(pid, comps)

    # lookup used_for for alternative search
    med_row   = medicine_df[medicine_df[col_name].str.lower() == med.lower()]
    target_use = med_row.iloc[0][col_use] if not med_row.empty else ""

    alternatives = []
    if risk["risk_level"] in ["MODERATE", "HIGH"]:
        alternatives = find_safe_alternatives(
            pid, target_use, medicine_df, col_name, col_components, col_use
        )

    rec_map = {
        "HIGH":     "⛔ HIGH RISK — This drug contains components that caused FDE before. Avoid and use an alternative.",
        "MODERATE": "⚠️ MODERATE RISK — Some suspicious components found. Monitor carefully or consider alternatives.",
        "LOW":      "✅ LOW RISK — No previously flagged components detected for this patient."
    }

    return jsonify({
        "medicine":       med,
        "risk":           risk,
        "alternatives":   alternatives,
        "recommendation": rec_map[risk["risk_level"]]
    })

# ── Report FDE event ────────────────────────────────────
@predict_bp.route("/report_fde", methods=["POST"])
def report_fde():
    data     = request.json
    pid      = data.get("patient_id")
    dose_id  = data.get("dosage_id")
    severity = data.get("severity", "moderate")

    if not pid or not dose_id:
        return jsonify({"error": "patient_id and dosage_id required"}), 400
    if not _patient_exists(pid):
        return jsonify({"error": "Patient not found"}), 404

    conn   = get_db()
    dosage = conn.execute(
        "SELECT * FROM dosage_history WHERE id=? AND patient_id=?", (dose_id, pid)
    ).fetchone()

    if not dosage:
        conn.close()
        return jsonify({"error": "Dosage record not found"}), 404

    conn.execute("UPDATE dosage_history SET fde_occurred=1 WHERE id=?", (dose_id,))
    conn.commit()

    components = [c.strip() for c in dosage["components"].split(",")]
    analysis   = process_fde_event(pid, components, dose_id)

    culprit_str = ",".join(analysis["culprit_components"])
    conn.execute(
        "INSERT INTO fde_events (patient_id, dosage_id, medicine_name, culprit_components, severity, event_date) VALUES (?,?,?,?,?,DATE('now'))",
        (pid, dose_id, dosage["medicine_name"], culprit_str, severity)
    )
    conn.commit()
    conn.close()

    scores       = get_component_scores(pid)
    risk_profile = sorted(
        [{"component": k, "score": round(v["score"],2), "fde_count": v["fde_count"]}
         for k, v in scores.items()],
        key=lambda x: x["score"], reverse=True
    )

    return jsonify({
        "message":            "FDE event recorded. Model updated via WCIS.",
        "analysis":           analysis,
        "updated_risk_profile": risk_profile[:15]
    })

# ── Full patient analysis ───────────────────────────────
@predict_bp.route("/analyze/<patient_id>", methods=["GET"])
def analyze(patient_id):
    if not _patient_exists(patient_id):
        return jsonify({"error": "Not found"}), 404

    scores = get_component_scores(patient_id)
    sorted_scores = sorted(
        [{"component": k, "score": round(v["score"],2), "fde_count": v["fde_count"]}
         for k, v in scores.items()],
        key=lambda x: x["score"], reverse=True
    )
    conn       = get_db()
    fde_count  = conn.execute("SELECT COUNT(*) as c FROM fde_events WHERE patient_id=?", (patient_id,)).fetchone()["c"]
    last_fde   = conn.execute("SELECT * FROM fde_events WHERE patient_id=? ORDER BY created_at DESC LIMIT 1", (patient_id,)).fetchone()
    conn.close()

    return jsonify({
        "patient_id":          patient_id,
        "algorithm":           "WCIS — Weighted Component Intersection Scoring",
        "total_fde_events":    fde_count,
        "top_risky_components": sorted_scores[:15],
        "last_fde":            dict(last_fde) if last_fde else None
    })
