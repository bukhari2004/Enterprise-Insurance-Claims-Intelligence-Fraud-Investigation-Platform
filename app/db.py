"""Tiny portable DB layer: SQLite (default, zero setup) or PostgreSQL (DATABASE_URL=postgresql://...).

SQL is written with '?' placeholders and plain TEXT/DOUBLE PRECISION/INTEGER types so the same
statements run on both engines. JSON columns are stored as TEXT and decoded automatically.
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

JSON_COLS = {
    "claims": {"explanation", "indicators", "features"},
    "documents": {"extracted", "issues"},
    "images": {"assessment"},
    "audit_log": {"detail"},
}

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS policies (
        policy_no TEXT PRIMARY KEY, customer_id TEXT, policy_type TEXT,
        coverage_limit DOUBLE PRECISION, deductible DOUBLE PRECISION,
        start_date TEXT, end_date TEXT, status TEXT)""",
    """CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY, claim_no TEXT UNIQUE, claim_type TEXT, policy_no TEXT,
        customer_id TEXT, customer_name TEXT, national_id TEXT, phone TEXT, address TEXT,
        bank_account TEXT, provider_id TEXT, provider_name TEXT, provider_type TEXT,
        doctor TEXT, vehicle_plate TEXT, claim_amount DOUBLE PRECISION,
        incident_date TEXT, filed_date TEXT, description TEXT, status TEXT,
        fraud_score DOUBLE PRECISION, risk_level TEXT, recommendation TEXT,
        est_damage_cost DOUBLE PRECISION, est_settlement DOUBLE PRECISION,
        explanation TEXT, indicators TEXT, features TEXT, investigator TEXT, ring_id TEXT, priority DOUBLE PRECISION,
        created_at TEXT, updated_at TEXT, decided_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY, claim_id TEXT, doc_type TEXT, filename TEXT, path TEXT,
        text TEXT, extracted TEXT, issues TEXT, tamper_score DOUBLE PRECISION, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS images (
        img_id TEXT PRIMARY KEY, claim_id TEXT, filename TEXT, path TEXT, phash TEXT,
        assessment TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY, claim_id TEXT, alert_type TEXT, severity TEXT,
        message TEXT, acknowledged INTEGER, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS audit_log (
        audit_id TEXT PRIMARY KEY, ts TEXT, actor TEXT, action TEXT,
        entity TEXT, entity_id TEXT, detail TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_claims_customer ON claims (customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_claims_policy ON claims (policy_no)",
    "CREATE INDEX IF NOT EXISTS ix_claims_provider ON claims (provider_id)",
    "CREATE INDEX IF NOT EXISTS ix_claims_status ON claims (status)",
    "CREATE INDEX IF NOT EXISTS ix_docs_claim ON documents (claim_id)",
    "CREATE INDEX IF NOT EXISTS ix_img_hash ON images (phash)",
    "CREATE INDEX IF NOT EXISTS ix_alert_claim ON alerts (claim_id)",
]


def now_iso():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def new_id():
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, url):
        self.url = url
        self.is_pg = url.startswith("postgres")
        self._lock = threading.RLock()
        if self.is_pg:
            import psycopg2  # noqa: pip install psycopg2-binary
            import psycopg2.extras
            self._extras = psycopg2.extras
            self._conn = psycopg2.connect(url)
            self._conn.autocommit = True
        else:
            path = url.replace("sqlite:///", "")
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        for stmt in SCHEMA:
            self.execute(stmt)

    # ---- low level -------------------------------------------------
    def execute(self, sql, params=()):
        with self._lock:
            if self.is_pg:
                cur = self._conn.cursor(cursor_factory=self._extras.RealDictCursor)
                cur.execute(sql.replace("?", "%s"), params)
            else:
                cur = self._conn.cursor()
                cur.execute(sql, params)
                self._conn.commit()
            return cur

    def query(self, sql, params=(), table=None):
        with self._lock:
            cur = self.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        return [self._decode(table, r) for r in rows] if table else rows

    def one(self, sql, params=(), table=None):
        rows = self.query(sql, params, table)
        return rows[0] if rows else None

    def scalar(self, sql, params=()):
        row = self.one(sql, params)
        return list(row.values())[0] if row else None

    # ---- helpers ---------------------------------------------------
    @staticmethod
    def _decode(table, row):
        for col in JSON_COLS.get(table, ()):
            v = row.get(col)
            if isinstance(v, str):
                try:
                    row[col] = json.loads(v)
                except ValueError:
                    pass
        return row

    @staticmethod
    def _encode(v):
        return json.dumps(v, default=str) if isinstance(v, (dict, list)) else v

    def insert(self, table, data):
        cols = list(data)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        self.execute(sql, [self._encode(data[c]) for c in cols])
        return data

    def update(self, table, key_col, key, data):
        if not data:
            return
        sets = ",".join(f"{c}=?" for c in data)
        self.execute(f"UPDATE {table} SET {sets} WHERE {key_col}=?",
                     [self._encode(v) for v in data.values()] + [key])

    def get_claim(self, claim_ref):
        """Accepts claim_id or claim_no."""
        return self.one("SELECT * FROM claims WHERE claim_id=? OR claim_no=?",
                        (claim_ref, claim_ref), "claims")
