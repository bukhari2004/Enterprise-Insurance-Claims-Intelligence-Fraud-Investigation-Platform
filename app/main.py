"""REST API gateway (FastAPI).  Run:  uvicorn app.main:app --reload   ->  http://localhost:8000"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline import Platform

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"
ROLE_RANK = {"viewer": 1, "investigator": 2, "admin": 3}
platform: Platform = None  # initialised at startup


@asynccontextmanager
async def lifespan(app):
    global platform
    platform = Platform()
    yield


app = FastAPI(title="Insurance Claims Intelligence & Fraud Investigation Platform (AI-235)",
              version="1.0.0", lifespan=lifespan)


# ------------------------------------------------------------------ security (API key + roles)
class Principal(BaseModel):
    name: str
    role: str


def auth(min_role):
    def dep(x_api_key: Optional[str] = Header(default=None)):
        if not settings.auth_enabled:
            return Principal(name="local-dev", role="admin")
        role = settings.key_roles().get(x_api_key or "")
        if not role:
            raise HTTPException(401, "Missing or invalid X-API-Key")
        if ROLE_RANK[role] < ROLE_RANK[min_role]:
            raise HTTPException(403, f"Requires role {min_role}")
        return Principal(name=f"{role}:{(x_api_key or '')[:4]}***", role=role)
    return dep


# ------------------------------------------------------------------ schemas
class ClaimIn(BaseModel):
    claim_type: str = Field(..., examples=["vehicle"])
    policy_no: str = Field(..., examples=["POL-100000"])
    customer_id: str
    customer_name: str
    claim_amount: float = Field(..., gt=0)
    incident_date: str = Field(..., examples=["2026-11-01"])
    filed_date: Optional[str] = None
    description: str = ""
    national_id: str = ""
    phone: str = ""
    address: str = ""
    bank_account: str = ""
    provider_id: str = ""
    provider_name: str = ""
    provider_type: str = ""
    doctor: str = ""
    vehicle_plate: str = ""


class PolicyIn(BaseModel):
    policy_no: str
    customer_id: str
    policy_type: str
    coverage_limit: float
    deductible: float = 0
    start_date: str
    end_date: str


class DecisionIn(BaseModel):
    decision: str = Field(..., examples=["APPROVED"])
    note: str = ""


class CopilotIn(BaseModel):
    question: str
    claim_ref: Optional[str] = None


def _claim_or_404(ref):
    c = platform.claim_detail(ref)
    if not c:
        raise HTTPException(404, f"Claim {ref} not found")
    return c


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "db": "postgres" if platform.db.is_pg else "sqlite",
            "neo4j": platform.graph.driver is not None, "redis": platform.bus.redis is not None,
            "copilot_engine": "langgraph" if platform.copilot.graph is not None else "python-dispatch",
            "yolo": bool(settings.yolo_weights), "ocr": settings.ocr_engine}


@app.post("/api/policies")
def create_policy(p: PolicyIn, user: Principal = Depends(auth("investigator"))):
    platform.create_policy(p.model_dump())
    platform.audit(user.name, "policy.created", "policy", p.policy_no)
    return {"ok": True}


@app.post("/api/claims", status_code=201)
def submit_claim(claim: ClaimIn, user: Principal = Depends(auth("investigator"))):
    """Claim Intake Engine: validates, stores, then triggers the full AI pipeline (event-driven)."""
    try:
        c = platform.submit_claim(claim.model_dump(), actor=user.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return platform.claim_detail(c["claim_id"])


@app.get("/api/claims")
def list_claims(status: Optional[str] = None, risk: Optional[str] = None, claim_type: Optional[str] = None,
                limit: int = 50, offset: int = 0, order: str = "priority", user: Principal = Depends(auth("viewer"))):
    return platform.list_claims(status, risk, claim_type, min(limit, 500), offset, order)


@app.get("/api/claims/{ref}")
def get_claim(ref: str, user: Principal = Depends(auth("viewer"))):
    return _claim_or_404(ref)


@app.post("/api/claims/{ref}/documents")
async def upload_document(ref: str, file: UploadFile = File(...), doc_type: Optional[str] = Form(None),
                          user: Principal = Depends(auth("investigator"))):
    """OCR + extraction + validation + tamper check, then the claim is automatically re-scored."""
    _claim_or_404(ref)
    doc = platform.add_document(ref, file.filename, await file.read(), doc_type, actor=user.name)
    return {"document": {k: doc[k] for k in ("doc_id", "doc_type", "extracted", "issues", "tamper_score")},
            "claim": platform.claim_detail(ref)}


@app.post("/api/claims/{ref}/images")
async def upload_image(ref: str, file: UploadFile = File(...), user: Principal = Depends(auth("investigator"))):
    """Computer-vision damage assessment + repair-cost estimate, then re-score."""
    _claim_or_404(ref)
    img = platform.add_image(ref, file.filename, await file.read(), actor=user.name)
    return {"assessment": img["assessment"], "claim": platform.claim_detail(ref)}


@app.post("/api/claims/{ref}/decision")
def decide(ref: str, d: DecisionIn, user: Principal = Depends(auth("investigator"))):
    _claim_or_404(ref)
    try:
        return platform.decide(ref, d.decision, user.name, d.note)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/api/claims/{ref}/rescore")
def rescore(ref: str, user: Principal = Depends(auth("investigator"))):
    c = _claim_or_404(ref)
    platform.bus.publish("claim.analyzed", {"claim_id": c["claim_id"], "actor": user.name})
    return platform.claim_detail(ref)


@app.get("/api/dashboard")
def dashboard(user: Principal = Depends(auth("viewer"))):
    return platform.dashboard()


@app.get("/api/alerts")
def alerts(include_acknowledged: bool = False, user: Principal = Depends(auth("viewer"))):
    return platform.alerts(only_open=not include_acknowledged)


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: str, user: Principal = Depends(auth("investigator"))):
    platform.ack_alert(alert_id, user.name)
    return {"ok": True}


@app.get("/api/graph/rings")
def rings(user: Principal = Depends(auth("viewer"))):
    return platform.graph.rings()


@app.get("/api/graph/claim/{ref}")
def claim_graph(ref: str, depth: int = 2, user: Principal = Depends(auth("viewer"))):
    c = _claim_or_404(ref)
    return platform.graph.subgraph(c["claim_no"], min(depth, 3))


@app.post("/api/graph/rescore-rings")
def rescore_rings(user: Principal = Depends(auth("admin"))):
    return {"rescored_claims": platform.rescore_ring_members(user.name)}


@app.post("/api/copilot")
def copilot(q: CopilotIn, user: Principal = Depends(auth("viewer"))):
    platform.audit(user.name, "copilot.query", "claim", q.claim_ref or "-", {"question": q.question})
    return platform.copilot.ask(q.question, q.claim_ref)


@app.get("/api/audit")
def audit_log(limit: int = 100, entity_id: Optional[str] = None, user: Principal = Depends(auth("admin"))):
    sql, params = "SELECT * FROM audit_log", []
    if entity_id:
        sql += " WHERE entity_id=?"
        params.append(entity_id)
    return platform.db.query(sql + " ORDER BY ts DESC LIMIT ?", params + [min(limit, 1000)], "audit_log")


@app.get("/api/events")
def events(user: Principal = Depends(auth("admin"))):
    return platform.bus.history[-50:]


# ------------------------------------------------------------------ dashboard UI
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "dashboard.html")
