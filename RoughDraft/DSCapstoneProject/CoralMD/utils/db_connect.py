from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "coralmd.sqlite3"
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def ensure_schema():
    """Create tables if they don't exist yet."""
    with sqlite3.connect(DB_PATH) as conn, open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())


def get_conn():
    """Get a connection with Row objects."""
    ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def add_patient(name: str) -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO patients(name) VALUES (?)", (name,))
        conn.commit()
        return cur.lastrowid


def list_patients():
    """Return all patients with their most recent risk."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id,
                   p.name,
                   (
                       SELECT risk_score
                       FROM predictions pr
                       WHERE pr.patient_id = p.id
                       ORDER BY pr.id DESC
                       LIMIT 1
                   ) AS last_risk
            FROM patients p
            ORDER BY p.id DESC
            """
        ).fetchall()
    return rows



def add_upload(patient_id: int, kind: str, filename: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO uploads(patient_id, kind, filename) VALUES (?,?,?)",
            (patient_id, kind, filename),
        )
        conn.commit()


def add_prediction(patient_id: int, model_name: str, risk_score: float, details_json: str = "{}"):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO predictions(patient_id, model_name, risk_score, details_json)
            VALUES (?,?,?,?)
            """,
            (patient_id, model_name, risk_score, details_json),
        )
        conn.commit()
