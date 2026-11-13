"""Investigation Recommendation Engine + settlement estimator + alert builder + explanation builder."""
from app.config import settings

ACTIONS = ["AUTO_APPROVE", "MANUAL_INVESTIGATION", "HIGH_RISK_INVESTIGATION",
           "ADDITIONAL_DOCUMENTS", "FIELD_INSPECTION", "ESCALATION"]
ACTION_STATUS = {"AUTO_APPROVE": "AUTO_APPROVED", "MANUAL_INVESTIGATION": "INVESTIGATING",
                 "HIGH_RISK_INVESTIGATION": "INVESTIGATING", "ADDITIONAL_DOCUMENTS": "PENDING_DOCUMENTS",
                 "FIELD_INSPECTION": "INVESTIGATING", "ESCALATION": "ESCALATED"}
OPEN_STATUSES = {"RECEIVED", "AUTO_APPROVED", "INVESTIGATING", "PENDING_DOCUMENTS", "ESCALATED"}
FINAL_STATUSES = {"APPROVED", "REJECTED", "SETTLED"}


def required_docs(claim):
    t, amt = claim["claim_type"], float(claim.get("claim_amount") or 0)
    req = {"vehicle": ["repair_estimate"], "health": ["medical_report"],
           "property": ["repair_estimate"], "travel": ["receipt"], "marine": ["invoice"],
           "life": ["medical_report", "identity"]}.get(t, [])
    req = list(req)
    if t in ("vehicle", "property") and amt >= 10000:
        req.append("police_fir")
    return req


def recommend(claim, ev, docs, images):
    codes = {i["code"] for i in ev["indicators"]}
    level, amt, reasons = ev["level"], float(claim.get("claim_amount") or 0), []
    have = {d["doc_type"] for d in docs}
    missing = [d for d in required_docs(claim) if d not in have]
    ring = ev.get("ring")
    priority = round(100 * (0.6 * ev["score"] + 0.4 * min(1.0, amt / settings.escalation_amount)), 1)

    if level == "CRITICAL" or (ring and ring["ring_score"] >= 0.6 and level == "HIGH") or (amt >= settings.escalation_amount and level != "LOW"):
        action = "ESCALATION"
        reasons.append("Critical fraud risk, organised-ring involvement or very large exposure")
    elif level == "HIGH":
        action = "HIGH_RISK_INVESTIGATION"
        reasons.append("High fraud score - assign senior investigator")
    elif (claim["claim_type"] in ("vehicle", "property", "marine") and level != "LOW"
          and (amt >= settings.field_inspection_amount or "VISION_MISMATCH" in codes)):
        action = "FIELD_INSPECTION"
        reasons.append("Physical damage should be verified on site")
    elif missing:
        action = "ADDITIONAL_DOCUMENTS"
        reasons.append("Missing required documents: " + ", ".join(m.replace("_", " ") for m in missing))
    elif level == "MEDIUM":
        action = "MANUAL_INVESTIGATION"
        reasons.append("Moderate fraud indicators need human review")
    elif amt > settings.auto_approve_limit or ev["features"].get("doc_issue_count", 0) > 0:
        action = "MANUAL_INVESTIGATION"
        reasons.append("Low risk but above auto-approval limit or has document findings")
    else:
        action = "AUTO_APPROVE"
        reasons.append("Low risk, documents complete and amount within auto-approval limit")
    return {"action": action, "status": ACTION_STATUS[action], "priority": priority,
            "missing_documents": missing, "reasons": reasons}


def estimate_settlement(claim, docs, policy, est_damage, hold):
    steps, amt = [], float(claim.get("claim_amount") or 0)
    steps.append({"label": "Claimed amount", "amount": amt})
    base = amt
    totals = [d["extracted"].get("total_amount") for d in docs
              if isinstance(d.get("extracted"), dict) and d["extracted"].get("total_amount")
              and d["extracted"].get("doc_type") in ("invoice", "repair_estimate", "medical_report", "receipt")]
    if totals:
        supported = max(totals)
        if supported < base:
            steps.append({"label": "Limited to amount supported by documents", "amount": supported - base})
            base = supported
    elif est_damage:
        ceiling = est_damage * 1.3
        if ceiling < base:
            steps.append({"label": "No documents - limited to AI damage estimate (upper band)", "amount": ceiling - base})
            base = ceiling
    if policy:
        ded = float(policy.get("deductible") or 0)
        if ded:
            steps.append({"label": "Policy deductible", "amount": -min(ded, base)})
            base -= min(ded, base)
        cap = float(policy.get("coverage_limit") or 0)
        if cap and base > cap:
            steps.append({"label": "Capped at coverage limit", "amount": cap - base})
            base = cap
    base = max(0.0, base)
    return {"indicative_amount": round(base, 2), "steps": steps, "currency": settings.currency,
            "payable_now": False if hold else True,
            "note": "Payment on hold pending investigation" if hold else "Eligible for settlement"}


def build_alerts(claim, ev):
    codes = {i["code"] for i in ev["indicators"]}
    out = []
    if ev["level"] in ("HIGH", "CRITICAL"):
        out.append(("HIGH_FRAUD_RISK", "critical" if ev["level"] == "CRITICAL" else "high",
                    f"Fraud score {ev['score']:.0%} ({ev['level']}) on {claim['claim_no']}"))
    if codes & {"DUPLICATE_CLAIM", "REUSED_IMAGE"}:
        out.append(("DUPLICATE_CLAIM", "high", f"{claim['claim_no']} duplicates or reuses evidence from an earlier claim"))
    if "SUSPICIOUS_PROVIDER" in codes:
        kind = "SUSPICIOUS_HOSPITAL_ACTIVITY" if claim.get("provider_type") == "hospital" else "FAKE_REPAIR_SHOP"
        out.append((kind, "high", f"{claim.get('provider_name')} has an unusual share of high-risk claims"))
    if codes & {"IDENTITY_MISMATCH", "DOC_ID_MISMATCH", "DOC_NAME_MISMATCH"}:
        out.append(("IDENTITY_MISMATCH", "high", f"Identity inconsistency detected on {claim['claim_no']}"))
    if "UNUSUAL_FREQUENCY" in codes:
        out.append(("UNUSUAL_FREQUENCY", "medium", f"{claim['customer_name']} filed an unusual number of claims"))
    if "FRAUD_RING" in codes:
        out.append(("FRAUD_RING", "critical", f"{claim['claim_no']} belongs to a suspected fraud ring"))
    return out


def build_explanation(claim, ev, rec, docs, images, settlement):
    top = [i for i in ev["indicators"] if i["points"] > 0][:5]
    factors = "; ".join(f"{i['label']} (+{i['points']})" for i in top) or "no significant fraud indicators"
    text = (f"{claim['claim_no']} received a fraud risk score of {ev['score']:.0%} ({ev['level']}). "
            f"Main factors: {factors}. Recommended action: {rec['action'].replace('_', ' ').title()}. "
            + " ".join(rec["reasons"]))
    doc_issues = [dict(i, document=d["doc_type"]) for d in docs for i in (d.get("issues") or [])]
    vision = next((i["assessment"] for i in images if i.get("assessment")), None)
    return {"summary": text, "top_factors": top, "document_issues": doc_issues, "vision": vision,
            "model": {"rule_score": ev["rule_score"], "anomaly_score": ev["anomaly_score"],
                      "method": "weighted explainable rules (85%) + Isolation Forest anomaly (15%)"},
            "recommendation": rec, "settlement": settlement, "ring": ev.get("ring"),
            "duplicates": ev.get("duplicates")}
