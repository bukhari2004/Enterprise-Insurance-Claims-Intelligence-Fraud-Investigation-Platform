# Enterprise Insurance Claims Intelligence & Fraud Investigation Platform (AI-235)

Ezitech Engineering Framework case study. An event-driven AI platform that validates claims, reads
documents (OCR), assesses damage from photos (computer vision), detects fraud (explainable rules +
anomaly detection + knowledge graph), recommends the next investigative step, and lets claim officers
interrogate any decision through an AI copilot.

**Runs out-of-the-box on a laptop** - SQLite, in-memory graph, OpenCV vision, Tesseract OCR, no GPU, no API keys.
PostgreSQL / Redis / Neo4j / YOLO / PaddleOCR / LangGraph / an LLM are *optional* plug-ins that switch on automatically.

## 1. Quick start (VS Code)

```bash
# Python 3.10+  (3.11 recommended)
python -m venv .venv
.venv\Scripts\activate            # Windows        |  source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```
Install **Tesseract OCR** for document images (optional - text/PDF and typed claims work without it):
Windows: https://github.com/UB-Mannheim/tesseract/wiki (add to PATH) - macOS: `brew install tesseract` - Linux: `sudo apt install tesseract-ocr`.

```bash
python scripts/seed_data.py --reset      # ~400 synthetic claims incl. injected fraud (about 30 s)
uvicorn app.main:app --reload            # http://localhost:8000  (dashboard)   /docs  (Swagger)
python -m unittest discover -s tests -v  # tests
```
In VS Code press **F5** and pick *API (uvicorn, reload)* (config in `.vscode/launch.json`).

## 2. What you will see
The seed contains realistic normal traffic plus six planted fraud scenarios; the platform finds all of them:

| Planted scenario | How the platform catches it |
|---|---|
| 7-customer ring sharing 2 bank accounts + an address, one garage "Speedy Fix Garage", reused photo | Knowledge-graph ring `RING-xxxx`, SHARED_BANK_ACCOUNT, REUSED_IMAGE, SUSPICIOUS_PROVIDER, escalation |
| "City Care Clinic" inflating hospital bills (~3x invoice) | DOC_AMOUNT_INFLATED + provider risk concentration alert |
| Resubmitted duplicate claims | DUPLICATE_CLAIM (similar incident/amount/description) |
| Same national ID, different people | IDENTITY_MISMATCH |
| Serial claimant (7 claims in 3 months) | UNUSUAL_FREQUENCY + Isolation-Forest anomaly |
| Edited documents (dates before incident, wrong name, totals that don't add up) | Document validation + tamper analysis |

The seed yields two suspected rings: the planted garage ring and a second provider-centred cluster (City Care Clinic customers plus the identity-fraud pairs).

Dashboard: KPIs, charts, investigation queue, alert center, fraud rings, claim drill-down (score decomposition,
document findings, vision result, settlement estimate, network graph, approve/reject), copilot chat, new-claim form.
Try uploading `samples/repair_estimate_scan.png` (OCR) and `samples/car_severe_damage.png` (vision) to a claim.

## 3. Module map (maps 1:1 to the case-study brief)

| Brief module | Code |
|---|---|
| Claim Intake Engine (6 claim types) | `app/pipeline.py::submit_claim`, `POST /api/claims` |
| AI Document Intelligence | `app/services/document_intel.py` (OCR -> classify -> extract -> validate -> tamper/ELA) |
| Computer Vision Damage Assessment + repair cost | `app/services/vision.py` (YOLO11 or OpenCV baseline, cost bands, perceptual hash) |
| Fraud Intelligence Engine | `app/services/fraud.py` (explainable rules + Isolation Forest) |
| Fraud Knowledge Graph + rings | `app/services/graph.py` (NetworkX, optional Neo4j mirror) |
| Investigation Recommendation Engine | `app/services/recommender.py` (6 actions, priority score) |
| AI Claims Copilot (multi-agent) | `app/services/copilot.py` (LangGraph if installed; grounded answers; optional LLM wording) |
| Alert Center | `recommender.build_alerts`, `GET /api/alerts` |
| Executive Dashboard | `app/static/dashboard.html`, `GET /api/dashboard` |
| Event-driven architecture | `app/events.py` (in-process bus, Redis Streams when `REDIS_URL` set) |
| Audit logs / enterprise security | `audit_log` table, API-key RBAC (`AUTH_ENABLED=true`) |

## 4. Key API calls (`/docs` has all)
```bash
curl -X POST localhost:8000/api/claims -H "Content-Type: application/json" -d '{
  "claim_type":"vehicle","policy_no":"POL-100000","customer_id":"C0001","customer_name":"Ali Raza",
  "claim_amount":30000,"incident_date":"2026-11-01","provider_name":"Speedy Fix Garage","description":"rear bumper damage"}'
curl -F "file=@samples/repair_estimate_scan.png" localhost:8000/api/claims/CLM-000385/documents
curl -F "file=@samples/car_severe_damage.png"    localhost:8000/api/claims/CLM-000385/images
curl -X POST localhost:8000/api/copilot -H "Content-Type: application/json" \
     -d '{"question":"Why was this claim flagged?","claim_ref":"CLM-000385"}'
```
The sample scan totals 13,000 and is dated before the incident, so a 30,000 claim triggers AMOUNT_INFLATED + DOC_BEFORE_INCIDENT and the score jumps live.
Use a policy number that exists (seeded ones are `POL-100000` upward; `GET /api/claims` shows real ones), or create one via `POST /api/policies`.

## 5. Optional integrations (set in `.env`, see `.env.example`)
| Want | Do |
|---|---|
| PostgreSQL | `pip install psycopg2-binary`, `DATABASE_URL=postgresql://user:pw@host/db` |
| Redis event stream | `pip install redis`, `REDIS_URL=redis://localhost:6379/0` |
| Neo4j mirror | `pip install neo4j`, `NEO4J_URI=bolt://localhost:7687` |
| YOLO11 damage detection | `pip install ultralytics`, `YOLO_WEIGHTS=path/to/best.pt` (see docs/ARCHITECTURE.md for training) |
| PaddleOCR | `pip install paddleocr paddlepaddle`, `OCR_ENGINE=paddle` |
| LangGraph agent graph | `pip install langgraph` (auto-detected; `/api/health` shows the engine) |
| LLM-polished copilot | `pip install anthropic`, `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` |
| Full stack | `docker compose up --build` |
| Auth | `AUTH_ENABLED=true`; send `X-API-Key` (roles: viewer < investigator < admin) |

## 6. Honest limitations (say these in your presentation - it scores well)
* **Vision** defaults to an OpenCV colour/edge *baseline* so everything runs without weights. For real accuracy fine-tune YOLO11 on a
  car-damage dataset (plan in docs/ARCHITECTURE.md); the interface does not change.
* **Data is synthetic.** Fraud weights are expert-set and tunable in `fraud.py`; with labelled history, train a supervised model
  (e.g. XGBoost + SHAP) and blend it in next to the rules.
* LangChain / LlamaIndex / FAISS / Hugging Face are not hard dependencies: the copilot uses deterministic grounded agents (LangGraph optional),
  TF-IDF retrieval for similar claims, and an optional LLM only for wording. Swap points are documented in the architecture doc.
