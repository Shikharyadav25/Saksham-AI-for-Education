import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "saksham.sqlite"


@contextmanager
def get_db(db_path: Path = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS learners (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        learner_id TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        FOREIGN KEY (learner_id) REFERENCES learners(id)
    );

    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        learner_id TEXT NOT NULL,
        question_id TEXT NOT NULL,
        skill_id TEXT NOT NULL,
        is_correct INTEGER NOT NULL,
        response_time REAL NOT NULL,
        hint_requested INTEGER NOT NULL DEFAULT 0,
        confidence_rating REAL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (learner_id) REFERENCES learners(id)
    );

    CREATE TABLE IF NOT EXISTS cognitive_states (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        learner_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        mastery_json TEXT NOT NULL,
        ddm_params_json TEXT NOT NULL,
        fatigue_prob REAL NOT NULL,
        calibration_error REAL,
        root_cause_skill TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (learner_id) REFERENCES learners(id)
    );

    CREATE INDEX IF NOT EXISTS idx_interactions_learner ON interactions(learner_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_interactions_skill ON interactions(skill_id);
    """
    with get_db(db_path) as conn:
        conn.executescript(schema)
        # Handle schema migration for age and gender columns
        cursor = conn.execute("PRAGMA table_info(learners)")
        columns = [row[1] for row in cursor.fetchall()]
        if "age" not in columns:
            conn.execute("ALTER TABLE learners ADD COLUMN age INTEGER")
        if "gender" not in columns:
            conn.execute("ALTER TABLE learners ADD COLUMN gender TEXT")


def ensure_learner(
    learner_id: str,
    name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO learners (id, name, age, gender, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                age = COALESCE(excluded.age, learners.age),
                gender = COALESCE(excluded.gender, learners.gender)
            """,
            (learner_id, name, age, gender, datetime.now(timezone.utc).isoformat()),
        )


def create_learner(
    name: str,
    age: int,
    gender: str,
    learner_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    import re
    import uuid

    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", name.lower().strip()).strip("_")
    if not clean_name:
        clean_name = "learner"
    suffix = uuid.uuid4().hex[:6]
    lid = learner_id or f"usr_{clean_name}_{suffix}"

    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO learners (id, name, age, gender, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (lid, name.strip(), int(age), gender.strip(), datetime.now(timezone.utc).isoformat()),
        )
    start_session(f"sess_{lid}", lid, db_path)
    return lid


def get_all_learners(db_path: Path = DEFAULT_DB_PATH, include_demo: bool = True) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, age, gender, created_at FROM learners ORDER BY created_at DESC"
        ).fetchall()
        learners = []
        for r in rows:
            d = dict(r)
            # Filter out deprecated demo profiles
            if d["id"] in ["bob_impulsive", "charlie_fatigued", "student_demo_01", "live_student"]:
                continue
            if d["id"] == "alice_prereq_gap":
                if include_demo:
                    d["name"] = "Alice (Demo Profile)"
                    d["age"] = d["age"] or 16
                    d["gender"] = d["gender"] or "Female"
                    learners.append(d)
            else:
                learners.append(d)
        return learners


def get_learner(learner_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, age, gender, created_at FROM learners WHERE id = ?",
            (learner_id,),
        ).fetchone()
        if row:
            d = dict(row)
            if d["id"] == "alice_prereq_gap":
                d["name"] = "Alice (Demo Profile)"
                d["age"] = d["age"] or 16
                d["gender"] = d["gender"] or "Female"
            return d
        return None


def delete_learner(learner_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    if learner_id == "alice_prereq_gap":
        return False
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM interactions WHERE learner_id = ?", (learner_id,))
        conn.execute("DELETE FROM cognitive_states WHERE learner_id = ?", (learner_id,))
        conn.execute("DELETE FROM sessions WHERE learner_id = ?", (learner_id,))
        conn.execute("DELETE FROM learners WHERE id = ?", (learner_id,))
        return True


def start_session(session_id: str, learner_id: str, db_path: Path = DEFAULT_DB_PATH) -> None:
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (id, learner_id, started_at) VALUES (?, ?, ?)",
            (session_id, learner_id, datetime.now(timezone.utc).isoformat()),
        )


def record_interaction(
    session_id: str,
    learner_id: str,
    question_id: str,
    skill_id: str,
    is_correct: bool,
    response_time: float,
    hint_requested: bool = False,
    confidence_rating: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO interactions (
                session_id, learner_id, question_id, skill_id,
                is_correct, response_time, hint_requested,
                confidence_rating, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                learner_id,
                question_id,
                skill_id,
                1 if is_correct else 0,
                float(response_time),
                1 if hint_requested else 0,
                confidence_rating,
                ts,
            ),
        )
        return int(cursor.lastrowid)


def get_learner_interactions(
    learner_id: str,
    skill_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM interactions WHERE learner_id = ?"
    params: List[Any] = [learner_id]
    if skill_id:
        query += " AND skill_id = ?"
        params.append(skill_id)
    query += " ORDER BY id ASC"

    with get_db(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def save_cognitive_state(
    session_id: str,
    learner_id: str,
    mastery: Dict[str, float],
    ddm_params: Dict[str, float],
    fatigue_prob: float,
    calibration_error: Optional[float] = None,
    root_cause_skill: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cognitive_states (
                session_id, learner_id, timestamp, mastery_json,
                ddm_params_json, fatigue_prob, calibration_error, root_cause_skill
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                learner_id,
                datetime.utcnow().isoformat(),
                json.dumps(mastery),
                json.dumps(ddm_params),
                fatigue_prob,
                calibration_error,
                root_cause_skill,
            ),
        )


def seed_sample_learners_if_empty(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Ensures Alice exists as a reference demo profile and purges deprecated demo entries.
    New learners are created manually through the frontend with custom attributes.
    """
    init_db(db_path)
    with get_db(db_path) as conn:
        # Purge deprecated demo profiles
        conn.execute("DELETE FROM interactions WHERE learner_id IN ('bob_impulsive', 'charlie_fatigued', 'student_demo_01', 'live_student')")
        conn.execute("DELETE FROM cognitive_states WHERE learner_id IN ('bob_impulsive', 'charlie_fatigued', 'student_demo_01', 'live_student')")
        conn.execute("DELETE FROM sessions WHERE learner_id IN ('bob_impulsive', 'charlie_fatigued', 'student_demo_01', 'live_student')")
        conn.execute("DELETE FROM learners WHERE id IN ('bob_impulsive', 'charlie_fatigued', 'student_demo_01', 'live_student')")

        # Update Alice metadata if already seeded
        conn.execute("UPDATE learners SET name = 'Alice (Demo Profile)', age = 16, gender = 'Female' WHERE id = 'alice_prereq_gap'")

        count = conn.execute("SELECT COUNT(*) FROM interactions WHERE learner_id = 'alice_prereq_gap'").fetchone()[0]
        if count > 0:
            return

    # Keep Alice as the single authentic reference demo profile for show
    ensure_learner("alice_prereq_gap", "Alice (Demo Profile)", age=16, gender="Female", db_path=db_path)
    start_session("sess_alice", "alice_prereq_gap", db_path)
    alice_data = [
        ("q_int_01", "arithmetic_integers", False, 11.2, False, 0.70),
        ("q_int_02", "arithmetic_integers", False, 10.5, True, 0.60),
        ("q_pemdas_01", "order_of_operations", False, 13.1, True, 0.65),
        ("q_pemdas_02", "order_of_operations", True, 9.8, False, 0.80),
        ("q_frac_01", "fraction_operations", True, 12.4, False, 0.75),
        ("q_frac_02", "fraction_operations", True, 11.0, False, 0.85),
        ("q_lin_01", "linear_equations_1var", False, 14.5, True, 0.55),
        ("q_lin_02", "linear_equations_1var", False, 13.8, False, 0.60),
        ("q_fact_01", "factoring_polynomials", True, 12.0, False, 0.80),
        ("q_fact_02", "factoring_polynomials", True, 11.5, False, 0.85),
        ("q_quad_01", "quadratic_equations", False, 15.2, True, 0.50),
        ("q_quad_02", "quadratic_equations", False, 16.0, True, 0.45),
        ("q_quad_01", "quadratic_equations", False, 14.8, True, 0.40),
        ("q_quad_02", "quadratic_equations", False, 17.1, False, 0.50),
        ("q_fn_01", "function_notation", True, 10.2, False, 0.80),
        ("q_fn_02", "function_notation", True, 11.4, False, 0.85),
    ]
    for q_id, s_id, corr, rt, hint, conf in alice_data:
        record_interaction("sess_alice", "alice_prereq_gap", q_id, s_id, corr, rt, hint, conf, db_path=db_path)

