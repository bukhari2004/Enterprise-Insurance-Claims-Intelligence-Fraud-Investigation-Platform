# Deployment Guide

## A. Local (no Docker) - recommended for demo
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
python scripts/seed_data.py --reset
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Health: `GET /api/health` shows which integrations are active (db, neo4j, redis, yolo, ocr, copilot engine).

## B. Docker Compose (Postgres + Redis + Neo4j + API)
```bash
docker compose up --build -d
docker compose exec api python scripts/seed_data.py --reset
open http://localhost:8000      # API key header required: dev-admin-key (enter it in the dashboard header)
```
Neo4j browser: http://localhost:7474 (neo4j / password).

## C. Production checklist
1. Replace `API_KEYS` with OIDC/JWT (FastAPI dependency in `app/main.py::auth`).
2. Run `uvicorn --workers N` behind NGINX/ALB with TLS; set `AUTH_ENABLED=true`.
3. Postgres managed service; add a connection pool (pgbouncer / psycopg pool) - `app/db.py` uses one connection for simplicity.
4. Move pipeline handlers to workers consuming the Redis/Kafka stream; scale OCR/CV workers on GPU.
5. Object storage for uploads (replace `Platform._save_upload`).
6. Schedule `POST /api/graph/rescore-rings` (cron / Airflow) nightly or hourly.
7. Ship `audit_log` to your SIEM; back up Postgres + Neo4j.
8. Train YOLO11 weights, set `YOLO_WEIGHTS`; calibrate fraud weights/thresholds against labelled outcomes (precision/recall at the HIGH threshold).

## D. Troubleshooting
| Symptom | Fix |
|---|---|
| `OCR_EMPTY` issue on uploaded image | Install Tesseract and make sure it is on PATH (`tesseract --version`). |
| `psycopg2` import error | `pip install psycopg2-binary` (only for PostgreSQL). |
| 401 from dashboard | `AUTH_ENABLED=true`: type an API key in the header box. |
| Fresh start | stop the server, delete `data/claims.db`, run the seed script. |
