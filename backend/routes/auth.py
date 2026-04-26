from flask import Blueprint, request, jsonify
import hashlib
from database import get_db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.json
    required = ["patient_id", "name", "password", "age", "gender", "blood_group"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    conn = get_db()
    c = conn.cursor()

    # Check if patient_id already exists
    existing = c.execute(
        "SELECT id FROM patients WHERE patient_id = ?", (data["patient_id"],)
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({"error": "Patient ID already registered"}), 409

    c.execute('''
        INSERT INTO patients (patient_id, name, age, gender, blood_group, contact, password)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data["patient_id"],
        data["name"],
        data.get("age"),
        data.get("gender"),
        data.get("blood_group"),
        data.get("contact", ""),
        hash_password(data["password"])
    ))

    conn.commit()
    conn.close()
    return jsonify({"message": "Registration successful", "patient_id": data["patient_id"]}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    pid = data.get("patient_id")
    password = data.get("password")

    if not pid or not password:
        return jsonify({"error": "patient_id and password required"}), 400

    conn = get_db()
    c = conn.cursor()
    patient = c.execute(
        "SELECT * FROM patients WHERE patient_id = ? AND password = ?",
        (pid, hash_password(password))
    ).fetchone()
    conn.close()

    if not patient:
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful",
        "patient": {
            "patient_id": patient["patient_id"],
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"],
            "blood_group": patient["blood_group"],
            "contact": patient["contact"]
        }
    }), 200
