import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "fde_system.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Patients table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            contact TEXT,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Dosage/medication history per patient
    c.execute('''
        CREATE TABLE IF NOT EXISTS dosage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            medicine_name TEXT NOT NULL,
            components TEXT NOT NULL,  -- comma-separated
            dosage TEXT,
            frequency TEXT,
            duration TEXT,
            fde_occurred INTEGER DEFAULT 0,  -- 0=No, 1=Yes
            prescribed_date TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # Per-patient component risk scores (the ML model state)
    c.execute('''
        CREATE TABLE IF NOT EXISTS component_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            component TEXT NOT NULL,
            score REAL DEFAULT 0,
            fde_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id, component),
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # FDE events log
    c.execute('''
        CREATE TABLE IF NOT EXISTS fde_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            dosage_id INTEGER NOT NULL,
            medicine_name TEXT NOT NULL,
            culprit_components TEXT,
            severity TEXT DEFAULT 'moderate',
            event_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # Image prediction audit log (CNN results stored per patient)
    c.execute('''
        CREATE TABLE IF NOT EXISTS image_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            filename TEXT,
            fde_score REAL,
            risk_level TEXT,
            top_class TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")
