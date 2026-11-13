"""Fraud Intelligence Engine: explainable rule scoring + Isolation-Forest anomaly detection.

Every point of the score is attributable to a named indicator (explainable by construction).
The anomaly model adds an unsupervised "this looks unlike normal claims" signal and reports which
features deviate most (z-scores) so even the ML part is explained.
"""
import math
from datetime import date, timedelta
from difflib import SequenceMatcher
from statistics import median

import numpy as np

from app.services import vision

FEATURES = ["log_amount_ratio", "filing_delay_days", "cust_claims_365", "provider_claims_30d",
            "doc_issue_count", "amount_to_coverage"]


def _d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def risk_level(score):
    return "LOW" if score < 0.25 else "MEDIUM" if score < 0.5 else "HIGH" if score < 0.75 else "CRITICAL"


class FraudEngine:
    def __init__(self, db, graph):
        self.db, self.graph = db, graph
        self._model, self._mu, self._sd, self._trained_at = None, None, None, 0

    # ------------------------------------------------------------ features
    def _type_median(self, claim_type, exclude_id):
        rows = self.db.query("SELECT claim_amount FROM claims WHERE claim_type=? AND claim_id<>?",
                             (claim_type, exclude_id))
        vals = [r["claim_amount"] for r in rows if r["claim_amount"]]
        return median(vals) if len(vals) >= 5 else None

    def features(self, c):
        db, cid = self.db, c["claim_id"]
        filed, inc = _d(c.get("filed_date")), _d(c.get("incident_date"))
        med = self._type_median(c["claim_type"], cid)
        amt = float(c.get("claim_amount") or 0)
        pol = db.one("SELECT * FROM policies WHERE policy_no=?", (c.get("policy_no"),))
        since = (filed - timedelta(days=365)).isoformat() if filed else "0000"
        cust_n = db.scalar("SELECT COUNT(*) FROM claims WHERE customer_id=? AND claim_id<>? AND filed_date>=?",
                           (c["customer_id"], cid, since)) or 0
        since30 = (filed - timedelta(days=30)).isoformat() if filed else "0000"
        prov_n = db.scalar("SELECT COUNT(*) FROM claims WHERE provider_id=? AND provider_id<>'' AND claim_id<>? "
                           "AND filed_date>=?", (c.get("provider_id") or "", cid, since30)) or 0
        docs = db.query("SELECT issues, tamper_score FROM documents WHERE claim_id=?", (cid,), "documents")
        issues = [i for d in docs for i in (d["issues"] or [])]
        return {
            "log_amount_ratio": math.log(max(amt, 1) / med) if med else 0.0,
            "amount_ratio": (amt / med) if med else None,
            "filing_delay_days": (filed - inc).days if filed and inc else 0,
            "cust_claims_365": cust_n,
            "provider_claims_30d": prov_n,
            "doc_issue_count": sum(1 for i in issues if i["severity"] in ("high", "medium")),
            "high_issues": [i for i in issues if i["severity"] == "high"],
            "max_tamper": max([d["tamper_score"] or 0 for d in docs], default=0),
            "amount_to_coverage": amt / pol["coverage_limit"] if pol and pol["coverage_limit"] else 0.0,
            "policy": pol,
            "policy_age_days": (inc - _d(pol["start_date"])).days if pol and inc and _d(pol["start_date"]) else None,
        }

    # ------------------------------------------------------------ rules
    def _duplicates(self, c):
        db, out = self.db, []
        inc = _d(c.get("incident_date"))
        rows = db.query("SELECT claim_no, claim_type, incident_date, description, claim_amount, customer_id, policy_no "
                        "FROM claims WHERE claim_id<>? AND (policy_no=? OR customer_id=?)",
                        (c["claim_id"], c["policy_no"], c["customer_id"]))
        for r in rows:
            d2 = _d(r["incident_date"])
            if r["claim_type"] != c["claim_type"] or not (inc and d2):
                continue
            gap = abs((inc - d2).days)
            sim = SequenceMatcher(None, (c.get("description") or "").lower(), (r["description"] or "").lower()).ratio()
            a1, a2 = float(c["claim_amount"] or 0), float(r["claim_amount"] or 0)
            same_amount = a1 > 0 and abs(a1 - a2) / a1 < 0.05
            if gap <= 3 and sim > 0.6 and same_amount:
                out.append({"claim_no": r["claim_no"], "similarity": round(sim, 2), "kind": "same incident window and amount"})
            elif gap <= 14 and sim > 0.9 and same_amount:
                out.append({"claim_no": r["claim_no"], "similarity": round(sim, 2), "kind": "near-identical resubmission"})
        return out

    def _image_reuse(self, c):
        mine = self.db.query("SELECT phash FROM images WHERE claim_id=?", (c["claim_id"],))
        if not mine:
            return []
        others = self.db.query("SELECT i.phash, c.claim_no, c.customer_id FROM images i JOIN claims c "
                               "ON i.claim_id=c.claim_id WHERE i.claim_id<>?", (c["claim_id"],))
        hits = {}
        for m in mine:
            for o in others:
                if m["phash"] and o["phash"] and vision.hamming(m["phash"], o["phash"]) <= 5:
                    hits[o["claim_no"]] = o["customer_id"] != c["customer_id"]
        return [{"claim_no": k, "other_customer": v} for k, v in hits.items()]

    def evaluate(self, c):
        f = self.features(c)
        ind = []

        def add(code, label, pts, detail=""):
            ind.append({"code": code, "label": label, "points": pts, "detail": detail})

        # --- duplicates
        dups = self._duplicates(c)
        if dups:
            add("DUPLICATE_CLAIM", "Possible duplicate claim", 40,
                "Similar to " + ", ".join(d["claim_no"] for d in dups[:3]))
        reuse = self._image_reuse(c)
        if reuse:
            cross = any(r["other_customer"] for r in reuse)
            add("REUSED_IMAGE", "Damage photo reused from another claim", 45 if cross else 30,
                "Matches photos on " + ", ".join(r["claim_no"] for r in reuse[:3]))
        # --- identity
        if c.get("national_id"):
            names = self.db.query("SELECT DISTINCT customer_name, customer_id FROM claims WHERE national_id=?",
                                  (c["national_id"],))
            if len({n["customer_id"] for n in names}) > 1:
                add("IDENTITY_MISMATCH", "Same national ID used by different customers", 40,
                    ", ".join(sorted({n["customer_name"] for n in names})[:4]))
        shared = self.graph.shared_entities(c["customer_id"])
        for s in shared:
            if s["type"] == "bank":
                add("SHARED_BANK_ACCOUNT", "Payout bank account shared with other customers", 30,
                    f"Also used by {', '.join(s['other_customers'][:3])}")
            elif s["type"] in ("phone", "address"):
                add(f"SHARED_{s['type'].upper()}", f"{s['type'].title()} shared with other customers", 15,
                    ", ".join(s["other_customers"][:3]))
            elif s["type"] == "vehicle":
                add("SHARED_VEHICLE", "Vehicle linked to multiple customers", 20, s["value"])
        ring = self.graph.ring_info(c["customer_id"])
        if ring:
            suspect_prov = (c.get("provider_id") or "").strip().lower() in {p.split(":", 1)[1] for p in self.graph.suspect_providers}
            linked = bool(shared) or suspect_prov           # this very claim touches a ring link, not just the customer
            pts = int(15 + 25 * ring["ring_score"]) if linked else 5
            add("FRAUD_RING", "Member of a suspected fraud ring" if linked else "Customer is in a suspected fraud ring (claim itself not linked)",
                pts, f"{ring['ring_id']}: {ring['size']} customers, {ring['claims']} claims")
        # --- amounts & behaviour
        if f["amount_ratio"] and f["amount_ratio"] > 3:
            add("AMOUNT_OUTLIER", "Claim amount far above typical for this claim type",
                min(25, int(6 * math.log2(f["amount_ratio"]))), f"{f['amount_ratio']:.1f}x the median")
        if f["cust_claims_365"] >= 4:
            add("UNUSUAL_FREQUENCY", "Unusually frequent claims by customer", min(20, 4 * f["cust_claims_365"]),
                f"{f['cust_claims_365']} other claims in the last 12 months")
        if f["policy_age_days"] is not None:
            if f["policy_age_days"] < 0:
                add("OUTSIDE_POLICY_PERIOD", "Incident occurred before policy start", 35,
                    f"{-f['policy_age_days']} days before inception")
            elif f["policy_age_days"] < 30:
                add("NEW_POLICY_CLAIM", "Claim shortly after policy inception", 15, f"{f['policy_age_days']} days after start")
        pol = f["policy"]
        if pol:
            if _d(pol["end_date"]) and _d(c["incident_date"]) and _d(c["incident_date"]) > _d(pol["end_date"]):
                add("POLICY_EXPIRED", "Incident after policy expiry", 35, f"Policy ended {pol['end_date']}")
            if f["amount_to_coverage"] > 1:
                add("EXCEEDS_COVERAGE", "Claim exceeds coverage limit", 15,
                    f"{f['amount_to_coverage']:.1f}x coverage ({pol['coverage_limit']:,.0f})")
            elif f["amount_to_coverage"] > 0.9:
                add("NEAR_COVERAGE_LIMIT", "Claim just under the coverage limit", 10, "Policy abuse pattern")
        if f["filing_delay_days"] > 45:
            add("LATE_FILING", "Claim filed long after the incident", 8, f"{f['filing_delay_days']} days")
        # --- documents / vision
        doc_pts = {"AMOUNT_INFLATED": 25, "NAME_MISMATCH": 25, "ID_MISMATCH": 25, "POLICY_MISMATCH": 20,
                   "DOC_BEFORE_INCIDENT": 20, "TOTAL_INCONSISTENT": 20, "TAMPERING_SUSPECTED": 0}
        for iss in f["high_issues"][:3]:
            pts = doc_pts.get(iss["code"], 12)
            if pts:
                add("DOC_" + iss["code"], "Document inconsistency", pts, iss["message"])
        if f["max_tamper"] >= 0.5:
            add("TAMPERED_DOCUMENT", "Document shows signs of editing", int(15 + 20 * f["max_tamper"]),
                "Image forensics flagged manipulation")
        est = c.get("est_damage_cost")
        if est and c["claim_amount"] and c["claim_amount"] > 3.5 * est and est > 0:
            add("VISION_MISMATCH", "Claimed amount far exceeds AI-estimated damage cost", 20,
                f"Claimed {c['claim_amount']:,.0f} vs estimated {est:,.0f}")
        # --- provider risk
        pstat = self.graph.provider_stats().get((c.get("provider_id") or "").strip().lower())
        if pstat and pstat["claims"] >= 6 and pstat["high_share"] >= 0.4:
            add("SUSPICIOUS_PROVIDER", "Provider has a high share of risky claims", 18,
                f"{pstat['high_share']:.0%} of {pstat['claims']} claims are high risk")
        if f["provider_claims_30d"] >= 15:
            add("PROVIDER_SPIKE", "Provider claim volume spike", 8, f"{f['provider_claims_30d']} claims in 30 days")

        rule_score = min(1.0, sum(i["points"] for i in ind) / 100.0)
        anomaly, anomaly_ind = self._anomaly(f)
        if anomaly_ind:
            ind.extend(anomaly_ind)
        score = rule_score if anomaly is None else min(1.0, 0.85 * rule_score + 0.15 * anomaly)
        ind.sort(key=lambda i: -i["points"])
        public = {k: v for k, v in f.items() if k not in ("policy", "high_issues")}
        return {"score": round(score, 3), "rule_score": round(rule_score, 3),
                "anomaly_score": None if anomaly is None else round(anomaly, 3),
                "level": risk_level(score), "indicators": ind, "features": public,
                "ring": ring, "duplicates": dups}

    # ------------------------------------------------------------ anomaly model
    def _matrix(self, feats):
        return np.array([[f.get(k) or 0.0 for k in FEATURES] for f in feats], dtype=float)

    def _train_if_needed(self):
        n = self.db.scalar("SELECT COUNT(*) FROM claims WHERE features IS NOT NULL") or 0
        if n < 60 or (self._model is not None and n - self._trained_at < 40):
            return
        from sklearn.ensemble import IsolationForest
        rows = self.db.query("SELECT features FROM claims WHERE features IS NOT NULL", (), "claims")
        X = self._matrix([r["features"] for r in rows if isinstance(r["features"], dict)])
        self._mu, self._sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        self._model = IsolationForest(n_estimators=150, contamination=0.08, random_state=42).fit(X)
        self._trained_at = n

    def _anomaly(self, f):
        self._train_if_needed()
        if self._model is None:
            return None, []
        x = self._matrix([f])
        raw = -float(self._model.score_samples(x)[0])           # higher = more anomalous (~0.35..0.75)
        score = float(1 / (1 + math.exp(-12 * (raw - 0.55))))
        z = (x[0] - self._mu) / self._sd
        ind = []
        if score > 0.6:
            for i in np.argsort(-np.abs(z))[:2]:
                if abs(z[i]) > 2:
                    ind.append({"code": "ANOMALY_" + FEATURES[i].upper(), "label": "Statistical anomaly: " +
                                FEATURES[i].replace("_", " "), "points": 0,
                                "detail": f"{z[i]:+.1f} std devs from normal (Isolation Forest)"})
        return score, ind
