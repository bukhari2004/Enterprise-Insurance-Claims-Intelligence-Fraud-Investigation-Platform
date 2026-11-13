"""Central configuration. Every integration is optional; defaults run locally."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(key, default=""):
    return os.environ.get(key, default)


class Settings:
    database_url = _env("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'claims.db'}")
    redis_url = _env("REDIS_URL")
    neo4j_uri = _env("NEO4J_URI")
    neo4j_user = _env("NEO4J_USER", "neo4j")
    neo4j_password = _env("NEO4J_PASSWORD", "password")
    auth_enabled = _env("AUTH_ENABLED", "false").lower() == "true"
    api_keys = _env("API_KEYS", "dev-admin-key:admin,dev-inv-key:investigator,dev-view-key:viewer")
    currency = _env("CURRENCY", "USD")
    ocr_engine = _env("OCR_ENGINE", "tesseract").lower()
    yolo_weights = _env("YOLO_WEIGHTS")
    llm_provider = _env("LLM_PROVIDER", "none").lower()
    anthropic_api_key = _env("ANTHROPIC_API_KEY")
    llm_model = _env("LLM_MODEL", "claude-sonnet-5-5")
    upload_dir = BASE_DIR / "data" / "uploads"
    auto_approve_limit = float(_env("AUTO_APPROVE_LIMIT", "5000"))
    field_inspection_amount = float(_env("FIELD_INSPECTION_AMOUNT", "20000"))
    escalation_amount = float(_env("ESCALATION_AMOUNT", "100000"))

    def key_roles(self):
        out = {}
        for pair in self.api_keys.split(","):
            if ":" in pair:
                k, r = pair.split(":", 1)
                out[k.strip()] = r.strip()
        return out


settings = Settings()
