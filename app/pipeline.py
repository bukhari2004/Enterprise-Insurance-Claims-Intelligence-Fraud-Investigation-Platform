"""Claims Intelligence Pipeline - wires every service together over the event bus.

claim.received -> (intake checks) -> claim.analyzed -> (fraud scoring + graph) -> claim.scored
               -> (recommendation + settlement + explanation) -> claim.decided -> (alerts + audit)
Uploading a document or image re-publishes claim.analyzed, so the claim is re-scored automatically.
"""
import re
from datetime import date, datetime
from pathlib import Path

from app.config import settings
from app.db import Database, new_id, now_iso
from app.events import EventBus
from app.services import document_intel, recommender, vision
from app.services.copilot import Copilot
from app.services.fraud import FraudEngine
from app.services.graph import FraudGraph

INVESTIGATORS = [f"INV-{i:02d}" for i in range(1, 9)]
REQUIRED = ["claim_type", "policy_no", "customer_id", "customer_name", "claim_amount", "incident_date"]
CLAIM_TYPES = {"vehicle", "health", "property", "travel", "marine", "life"}


class Platform:
    def __init__(self, db_url=None):
        self.db = Database(db_url or settings.database_url)
        self.bus = EventBus(settings.redis_url)
        self.graph = FraudGraph(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        self.fraud = FraudEngine(self.db, self.graph)
        self.copilot = Copilot(self)
        self.graph.rebuild(self.db.query("SELECT * FROM claims", (), "claims"))
        self._refresh_suspects()
        self.bus.subscribe("claim.received", self.on_received)
        self.bus.subscribe("claim.analyzed", self.on_analyzed)
        self.bus.subscribe("claim.scored", self.on_scored)
        self.bus.subscribe("claim.decided", self.on_decided)

    # ------------------------------------------------------------ audit
    def audit(self, actor, action, entity, entity_id, detail=None):
        self.db.insert("audit_log", {"audit_id": new_id(), "ts": now_iso(), "actor": actor, "action": action,
                                     "entity": entity, "entity_id": entity_id, "detail": detail or {}})

    # ------------------------------------------------------------ intake
    def create_policy(self, p):
        self.db.execute("DELETE FROM policies WHERE policy_no=?", (p["policy_no"],))
        self.db.insert("policies", {"status": "active", "deductible": 0, **p})

    def submit_claim(self, data, actor="system"):
        missing = [k for k in REQUIRED if data.get(k) in (None, "")]
        if missing:
            raise ValueError("Missing fields: " + ", ".join(missing))
        if data["claim_type"] not in CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {sorted(CLAIM_TYPES)}")
        if float(data["claim_amount"]) <= 0:
            raise ValueError("claim_amount must be positive")
        nxt = (self.db.scalar("SELECT COUNT(*) FROM claims") or 0) + 1
        claim_no = data.get("claim_no") or f"CLM-{nxt:06d}"
        while self.db.scalar("SELECT COUNT(*) FROM claims WHERE claim_no=?", (claim_no,)):
            nxt += 1
            claim_no = f"CLM-{nxt:06d}"
        cols = ["claim_type", "policy_no", "customer_id", "customer_name", "national_id", "phone", "address",
                "bank_account", "provider_id", "provider_name", "provider_type", "doctor", "vehicle_plate",
                "incident_date", "filed_date", "description"]
        row = {c: data.get(c) or "" for c in cols}
        row["filed_date"] = data.get("filed_date") or date.today().isoformat()
        row.update({"claim_id": new_id(), "claim_no": claim_no, "claim_amount": float(data["claim_amount"]),
                    "status": "RECEIVED", "created_at": data.get("created_at") or now_iso(), "updated_at": now_iso()})
        self.db.insert("claims", row)
        self.graph.add_claim(row)
        self.audit(actor, "claim.submitted", "claim", row["claim_id"], {"claim_no": claim_no, "amount": row["claim_amount"]})
        self.bus.publish("claim.received", {"claim_id": row["claim_id"], "actor": actor})
        return self.db.get_claim(row["claim_id"])

    # ------------------------------------------------------------ documents & images
    def _save_upload(self, claim_no, filename, content):
        d = Path(settings.upload_dir) / claim_no
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
        path = d / f"{new_id()[:6]}_{safe}"
        path.write_bytes(content)
        return path

    def add_document(self, claim_ref, filename="document.txt", content=None, doc_type=None, text=None, actor="system",
                     rescore=True):
        c = self.db.get_claim(claim_ref)
        if not c:
            raise KeyError(claim_ref)
        path = self._save_upload(c["claim_no"], filename, content) if content is not None else None
        if text is None:
            text = document_intel.read_document(path) if path else ""
        dtype = doc_type if doc_type in document_intel.DOC_TYPES else document_intel.classify(text)
        ex = document_intel.extract(text, dtype)
        issues = document_intel.validate(c, ex)
        tamper, reasons = document_intel.tamper_analysis(path) if path else (0.0, [])
        if tamper >= 0.5:
            issues.append({"code": "TAMPERING_SUSPECTED", "severity": "high", "message": "; ".join(reasons)})
        if not text.strip():
            issues.append({"code": "OCR_EMPTY", "severity": "medium",
                           "message": "No text could be read (install Tesseract or upload a clearer scan)"})
        doc = {"doc_id": new_id(), "claim_id": c["claim_id"], "doc_type": dtype, "filename": filename,
               "path": str(path or ""), "text": text, "extracted": ex, "issues": issues,
               "tamper_score": tamper, "created_at": now_iso()}
        self.db.insert("documents", doc)
        self.audit(actor, "document.analyzed", "claim", c["claim_id"], {"type": dtype, "issues": len(issues)})
        if rescore:
            self.bus.publish("claim.analyzed", {"claim_id": c["claim_id"], "actor": actor})
        return doc

    def add_image(self, claim_ref, filename="photo.jpg", content=None, path=None, actor="system", rescore=True):
        c = self.db.get_claim(claim_ref)
        if not c:
            raise KeyError(claim_ref)
        if content is not None:
            path = self._save_upload(c["claim_no"], filename, content)
        assessment = vision.assess(path, c["claim_type"])
        img = {"img_id": new_id(), "claim_id": c["claim_id"], "filename": filename, "path": str(path),
               "phash": vision.dhash(path), "assessment": assessment, "created_at": now_iso()}
        self.db.insert("images", img)
        costs = [i["assessment"]["estimated_cost"]["expected"] for i in
                 self.db.query("SELECT assessment FROM images WHERE claim_id=?", (c["claim_id"],), "images")
                 if isinstance(i["assessment"], dict) and "estimated_cost" in i["assessment"]]
        self.db.update("claims", "claim_id", c["claim_id"], {"est_damage_cost": max(costs) if costs else None})
        self.audit(actor, "image.assessed", "claim", c["claim_id"], {"severity": assessment.get("severity_label")})
        if rescore:
            self.bus.publish("claim.analyzed", {"claim_id": c["claim_id"], "actor": actor})
        return img

    # ------------------------------------------------------------ event handlers
    def on_received(self, ev):
        c = self.db.get_claim(ev["claim_id"])
        pol = self.db.one("SELECT * FROM policies WHERE policy_no=?", (c["policy_no"],))
        if not pol:
            self.audit("system", "intake.warning", "claim", c["claim_id"], {"warning": "policy not found"})
        self.bus.publish("claim.analyzed", {"claim_id": c["claim_id"], "actor": ev.get("actor", "system")})

    def _refresh_suspects(self):
        stats = self.graph.provider_stats()
        self.graph.set_suspect_providers([p for p, s in stats.items() if s["claims"] >= 6 and s["high_share"] >= 0.4])

    def on_analyzed(self, ev):
        c = self.db.get_claim(ev["claim_id"])
        result = self.fraud.evaluate(c)
        ring = result["ring"]
        upd = {"fraud_score": result["score"], "risk_level": result["level"], "indicators": result["indicators"],
               "features": result["features"], "ring_id": ring["ring_id"] if ring else None, "updated_at": now_iso()}
        self.db.update("claims", "claim_id", c["claim_id"], upd)
        self.graph.add_claim({**c, **upd})
        self._refresh_suspects()
        self.bus.publish("claim.scored", {"claim_id": c["claim_id"], "evaluation": result, "actor": ev.get("actor")})

    def on_scored(self, ev):
        c = self.db.get_claim(ev["claim_id"])
        result = ev["evaluation"]
        docs = self.db.query("SELECT * FROM documents WHERE claim_id=?", (c["claim_id"],), "documents")
        imgs = self.db.query("SELECT * FROM images WHERE claim_id=?", (c["claim_id"],), "images")
        rec = recommender.recommend(c, result, docs, imgs)
        policy = self.db.one("SELECT * FROM policies WHERE policy_no=?", (c["policy_no"],))
        hold = rec["action"] != "AUTO_APPROVE"
        settle = recommender.estimate_settlement(c, docs, policy, c.get("est_damage_cost"), hold)
        explanation = recommender.build_explanation(c, result, rec, docs, imgs, settle)
        upd = {"recommendation": rec["action"], "priority": rec["priority"], "est_settlement": settle["indicative_amount"],
               "explanation": explanation, "updated_at": now_iso()}
        if c["status"] not in recommender.FINAL_STATUSES:
            upd["status"] = rec["status"]
            if rec["status"] in ("INVESTIGATING", "ESCALATED") and not c.get("investigator"):
                n = self.db.scalar("SELECT COUNT(*) FROM claims WHERE investigator<>''") or 0
                upd["investigator"] = INVESTIGATORS[n % len(INVESTIGATORS)]
        self.db.update("claims", "claim_id", c["claim_id"], upd)
        self.graph.add_claim({**c, **upd})
        self.bus.publish("claim.decided", {"claim_id": c["claim_id"], "evaluation": result, "recommendation": rec,
                                           "actor": ev.get("actor")})

    def on_decided(self, ev):
        c = self.db.get_claim(ev["claim_id"])
        for atype, sev, msg in recommender.build_alerts(c, ev["evaluation"]):
            exists = self.db.scalar("SELECT COUNT(*) FROM alerts WHERE claim_id=? AND alert_type=?", (c["claim_id"], atype))
            if not exists:
                self.db.insert("alerts", {"alert_id": new_id(), "claim_id": c["claim_id"], "alert_type": atype,
                                          "severity": sev, "message": msg, "acknowledged": 0, "created_at": now_iso()})
        self.audit("ai-engine", "claim.decision", "claim", c["claim_id"],
                   {"score": c["fraud_score"], "level": c["risk_level"], "recommendation": ev["recommendation"]["action"]})

    # ------------------------------------------------------------ human actions
    def decide(self, claim_ref, decision, actor, note=""):
        decision = decision.upper()
        if decision not in {"APPROVED", "REJECTED", "SETTLED"}:
            raise ValueError("decision must be APPROVED, REJECTED or SETTLED")
        c = self.db.get_claim(claim_ref)
        if not c:
            raise KeyError(claim_ref)
        self.db.update("claims", "claim_id", c["claim_id"], {"status": decision, "decided_at": now_iso(), "updated_at": now_iso()})
        self.audit(actor, f"claim.{decision.lower()}", "claim", c["claim_id"], {"note": note, "ai_recommendation": c["recommendation"]})
        return self.db.get_claim(c["claim_id"])

    def ack_alert(self, alert_id, actor):
        self.db.update("alerts", "alert_id", alert_id, {"acknowledged": 1})
        self.audit(actor, "alert.acknowledged", "alert", alert_id)

    # ------------------------------------------------------------ read models
    def claim_detail(self, ref):
        c = self.db.get_claim(ref)
        if not c:
            return None
        cid = c["claim_id"]
        c["documents"] = self.db.query("SELECT doc_id, doc_type, filename, extracted, issues, tamper_score, created_at "
                                       "FROM documents WHERE claim_id=?", (cid,), "documents")
        c["images"] = self.db.query("SELECT img_id, filename, phash, assessment, created_at FROM images WHERE claim_id=?",
                                    (cid,), "images")
        c["alerts"] = self.db.query("SELECT * FROM alerts WHERE claim_id=? ORDER BY created_at", (cid,))
        c["policy"] = self.db.one("SELECT * FROM policies WHERE policy_no=?", (c["policy_no"],))
        return c

    def list_claims(self, status=None, risk=None, claim_type=None, limit=50, offset=0, order="priority"):
        where, params = [], []
        for col, val in (("status", status), ("risk_level", risk), ("claim_type", claim_type)):
            if val:
                where.append(f"{col}=?")
                params.append(val)
        sql = "SELECT * FROM claims" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY " + ("priority DESC" if order == "priority" else "created_at DESC") + " LIMIT ? OFFSET ?"
        rows = self.db.query(sql, params + [limit, offset], "claims")
        for r in rows:  # keep list payload light
            r.pop("features", None); r.pop("explanation", None)
        return rows

    def similar_claims(self, claim, k=5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        rows = self.db.query("SELECT claim_no, description, claim_amount, status, recommendation, fraud_score, provider_name "
                             "FROM claims WHERE claim_type=? AND claim_id<>? LIMIT 5000", (claim["claim_type"], claim["claim_id"]))
        if len(rows) < 3:
            return []
        mk = lambda r: f"{r.get('description') or ''} {r.get('provider_name') or ''}"
        vec = TfidfVectorizer(stop_words="english").fit([mk(r) for r in rows] + [mk(claim)])
        sims = cosine_similarity(vec.transform([mk(claim)]), vec.transform([mk(r) for r in rows]))[0]
        amt = float(claim["claim_amount"] or 1)
        out = []
        for r, s in zip(rows, sims):
            closeness = 1 - min(1.0, abs(r["claim_amount"] - amt) / max(amt, 1))
            out.append({**r, "similarity": float(0.6 * s + 0.4 * closeness)})
        return sorted(out, key=lambda x: -x["similarity"])[:k]

    def dashboard(self):
        q = self.db.query
        by_status = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) n FROM claims GROUP BY status")}
        by_risk = {r["risk_level"] or "UNSCORED": r["n"] for r in q("SELECT risk_level, COUNT(*) n FROM claims GROUP BY risk_level")}
        by_type = {r["claim_type"]: r["n"] for r in q("SELECT claim_type, COUNT(*) n FROM claims GROUP BY claim_type")}
        by_rec = {r["recommendation"] or "NONE": r["n"] for r in q("SELECT recommendation, COUNT(*) n FROM claims GROUP BY recommendation")}
        total = self.db.scalar("SELECT COUNT(*) FROM claims") or 0
        open_rows = q("SELECT claim_amount, fraud_score FROM claims WHERE status NOT IN ('SETTLED','REJECTED','APPROVED')")
        exposure = sum(r["claim_amount"] or 0 for r in open_rows)
        at_risk = sum((r["claim_amount"] or 0) * (r["fraud_score"] or 0) for r in open_rows)
        decided = q("SELECT filed_date, decided_at FROM claims WHERE decided_at IS NOT NULL AND status IN ('SETTLED','APPROVED')")
        days = []
        for r in decided:
            try:
                days.append((datetime.fromisoformat(r["decided_at"][:19]).date() - date.fromisoformat(r["filed_date"][:10])).days)
            except Exception:
                pass
        vol = {}
        for r in q("SELECT filed_date, COUNT(*) n FROM claims GROUP BY filed_date ORDER BY filed_date DESC LIMIT 30"):
            vol[r["filed_date"]] = r["n"]
        queue = q("SELECT claim_no, claim_type, customer_name, claim_amount, fraud_score, risk_level, recommendation, status, "
                  "investigator, priority FROM claims WHERE status IN ('INVESTIGATING','ESCALATED','PENDING_DOCUMENTS') "
                  "ORDER BY priority DESC LIMIT 15")
        return {
            "total_claims": total, "currency": settings.currency,
            "open_alerts": self.db.scalar("SELECT COUNT(*) FROM alerts WHERE acknowledged=0") or 0,
            "avg_settlement_days": round(sum(days) / len(days), 1) if days else None,
            "auto_approved_share": round(by_status.get("AUTO_APPROVED", 0) / total, 3) if total else 0,
            "financial_exposure": round(exposure), "fraud_adjusted_exposure": round(at_risk),
            "prevented_loss": round(self.db.scalar("SELECT COALESCE(SUM(claim_amount),0) FROM claims WHERE status='REJECTED'") or 0),
            "by_status": by_status, "by_risk": by_risk, "by_type": by_type, "by_recommendation": by_rec,
            "volume_by_day": dict(sorted(vol.items())), "investigation_queue": queue,
            "rings": len(self.graph.rings()),
        }

    def rescore_ring_members(self, actor="graph-batch"):
        """Periodic graph-analytics job: rings only become visible once enough members have filed, so
        re-evaluate every claim of every suspected ring (earlier members get flagged retroactively)."""
        n = 0
        for ring in self.graph.rings():
            rows = self.db.query("SELECT claim_id FROM claims WHERE customer_name IN (%s)" % ",".join("?" * len(ring["members"])),
                                 ring["members"])
            for r in rows:
                self.bus.publish("claim.analyzed", {"claim_id": r["claim_id"], "actor": actor})
                n += 1
        return n

    def alerts(self, only_open=True, limit=100):
        sql = ("SELECT a.*, c.claim_no, c.customer_name FROM alerts a JOIN claims c ON a.claim_id=c.claim_id"
               + (" WHERE a.acknowledged=0" if only_open else "") + " ORDER BY a.created_at DESC LIMIT ?")
        return self.db.query(sql, (limit,))
