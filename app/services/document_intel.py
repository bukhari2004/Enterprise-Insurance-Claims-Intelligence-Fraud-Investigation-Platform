"""AI Document Intelligence: OCR -> classification -> field extraction -> consistency validation
-> tamper analysis (ELA + metadata). Works on images, PDFs (text layer) and plain text."""
import re
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

from app.config import settings

DOC_TYPES = ["medical_report", "police_fir", "invoice", "repair_estimate",
             "policy", "identity", "receipt", "other"]

KEYWORDS = {
    "medical_report": ["patient", "diagnosis", "hospital", "doctor", "discharge", "treatment"],
    "police_fir": ["fir", "police station", "complainant", "first information"],
    "invoice": ["invoice", "bill to", "invoice no"],
    "repair_estimate": ["estimate", "repair", "workshop", "garage", "labour", "labor"],
    "policy": ["policy schedule", "sum insured", "policy period", "coverage"],
    "identity": ["national identity", "cnic", "passport", "date of birth", "id card"],
    "receipt": ["receipt", "paid", "cash memo"],
}

_ocr_engine = None


# ------------------------------------------------------------------ OCR
def _get_paddle():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_engine


def ocr_image(path):
    """PaddleOCR if OCR_ENGINE=paddle and installed, else Tesseract. Returns '' if neither works."""
    if settings.ocr_engine == "paddle":
        try:
            res = _get_paddle().ocr(str(path), cls=True)
            lines = [ln[1][0] for page in res for ln in (page or [])]
            return "\n".join(lines)
        except Exception:
            pass
    try:
        import pytesseract
        from PIL import Image, ImageOps
        img = ImageOps.autocontrast(Image.open(path).convert("L"))
        if img.width < 1400:                       # small scans OCR far better when upscaled
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""


def read_document(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".txt", ".md"}:
        return p.read_text(errors="ignore")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        except Exception:
            return ""
    return ocr_image(p)


def classify(text):
    t = text.lower()
    best, hits = "other", 0
    for dt, kws in KEYWORDS.items():
        n = sum(k in t for k in kws)
        if n > hits:
            best, hits = dt, n
    return best


# ------------------------------------------------------------ extraction
AMOUNT_RE = re.compile(r"(?:grand\s+total|total\s+amount|total|amount\s+payable|estimate\s+total|net\s+amount)"
                       r"[^\d\n]{0,25}([\d][\d,]*(?:\.\d{1,2})?)", re.I)
POLICY_RE = re.compile(r"\b(POL[-\s]?\d{4,})\b", re.I)
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b")
ID_RE = re.compile(r"\b(\d{5}-\d{7}-\d)\b")
NAME_RE = re.compile(r"(?:patient(?:\s+name)?|name|insured|claimant|complainant|policy\s*holder)\s*[:\-]\s*([A-Za-z][A-Za-z .']{2,40})", re.I)
PROVIDER_RE = re.compile(r"(?:hospital|workshop|garage|clinic|provider)\s*[:\-]\s*([^\n]{3,60})", re.I)
INVOICE_NO_RE = re.compile(r"(?:invoice|bill)\s*(?:no|#|number)?\s*[:\-#]?\s*([A-Z]{2,4}-?\d{3,})", re.I)
FIR_NO_RE = re.compile(r"\bFIR\s*(?:no|number|#)?\s*[:\-#]?\s*(\d{1,6}/\d{2,4}|\d{3,8})", re.I)
LINE_RE = re.compile(r"^\s*[-*\u2022]?\s*([A-Za-z][A-Za-z /&()'.-]{2,50}?)\s*[:\-]?\s+([\d][\d,]*(?:\.\d{1,2})?)\s*$")


def _num(s):
    return float(s.replace(",", ""))


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def extract(text, doc_type=None):
    doc_type = doc_type or classify(text)
    out = {"doc_type": doc_type}
    totals = [_num(m.group(1)) for m in AMOUNT_RE.finditer(text)]
    out["total_amount"] = totals[-1] if totals else None
    m = POLICY_RE.search(text)
    out["policy_no"] = re.sub(r"[\s]", "-", m.group(1).upper()).replace("POL", "POL-", 1).replace("POL--", "POL-") if m else None
    out["dates"] = [d.isoformat() for d in (_parse_date(x) for x in DATE_RE.findall(text)) if d]
    m = ID_RE.search(text)
    out["national_id"] = m.group(1) if m else None
    m = NAME_RE.search(text)
    out["name"] = m.group(1).strip().rstrip(".") if m else None
    m = PROVIDER_RE.search(text)
    out["provider"] = m.group(1).strip() if m else None
    m = INVOICE_NO_RE.search(text)
    out["invoice_no"] = m.group(1) if m else None
    m = FIR_NO_RE.search(text)
    out["fir_no"] = m.group(1) if m else None
    items = []
    for line in text.splitlines():
        lm = LINE_RE.match(line)
        if lm and not re.match(r"\s*(sub\s*total|total|grand|amount|net)", lm.group(1), re.I):
            items.append({"item": lm.group(1).strip(), "amount": _num(lm.group(2))})
    out["line_items"] = items
    return out


# ------------------------------------------------------------ validation
def _issue(code, severity, message):
    return {"code": code, "severity": severity, "message": message}


def _name_sim(a, b):
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).split()
    ta, tb = sorted(norm(a)), sorted(norm(b))
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


def validate(claim, ex):
    """Cross-check extracted document fields against the claim form. Returns list of issues."""
    issues = []
    dt = ex.get("doc_type")
    claimed = float(claim.get("claim_amount") or 0)
    total = ex.get("total_amount")
    if total and dt in {"invoice", "repair_estimate", "medical_report", "receipt"} and claimed:
        ratio = claimed / total
        if ratio > 1.15:
            issues.append(_issue("AMOUNT_INFLATED", "high",
                                 f"Claimed amount {claimed:,.0f} is {ratio:.1f}x the {dt.replace('_', ' ')} total {total:,.0f}"))
        elif ratio < 0.85:
            issues.append(_issue("AMOUNT_BELOW_DOCUMENT", "low",
                                 f"Claimed amount {claimed:,.0f} is lower than document total {total:,.0f}"))
    items = ex.get("line_items") or []
    if total and len(items) >= 2:
        s = sum(i["amount"] for i in items)
        if abs(s - total) > max(1.0, 0.01 * total):
            issues.append(_issue("TOTAL_INCONSISTENT", "high",
                                 f"Line items sum to {s:,.0f} but document total is {total:,.0f}"))
    if ex.get("name") and claim.get("customer_name") and dt != "policy":
        sim = _name_sim(ex["name"], claim["customer_name"])
        if sim < 0.75:
            issues.append(_issue("NAME_MISMATCH", "high",
                                 f"Name on document '{ex['name']}' differs from claimant '{claim['customer_name']}'"))
    if ex.get("policy_no") and claim.get("policy_no") and ex["policy_no"] != claim["policy_no"]:
        issues.append(_issue("POLICY_MISMATCH", "high",
                             f"Policy {ex['policy_no']} on document differs from claim policy {claim['policy_no']}"))
    if ex.get("national_id") and claim.get("national_id") and ex["national_id"] != claim["national_id"]:
        issues.append(_issue("ID_MISMATCH", "high",
                             f"ID {ex['national_id']} on document differs from claimant ID {claim['national_id']}"))
    inc = _parse_date(claim["incident_date"]) if claim.get("incident_date") else None
    if inc and dt in {"invoice", "repair_estimate", "medical_report", "receipt"}:
        for d in ex.get("dates", []):
            dd = date.fromisoformat(d)
            if (inc - dd).days > 1:
                issues.append(_issue("DOC_BEFORE_INCIDENT", "high",
                                     f"Document dated {d} is before the incident date {inc}"))
                break
    if dt == "police_fir" and not ex.get("fir_no"):
        issues.append(_issue("FIR_NUMBER_MISSING", "medium", "Police report has no readable FIR number"))
    if ex.get("provider") and claim.get("provider_name"):
        if _name_sim(ex["provider"], claim["provider_name"]) < 0.6:
            issues.append(_issue("PROVIDER_MISMATCH", "medium",
                                 f"Provider on document '{ex['provider']}' differs from claim provider '{claim['provider_name']}'"))
    return issues


# ------------------------------------------------------------ tamper detection
def tamper_analysis(path):
    """Cheap forensic signals: JPEG Error Level Analysis + editing-software metadata.
    Returns (score 0..1, reasons). Replace/extend with a trained model (see docs) for production."""
    p = Path(path)
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return 0.0, []
    score, reasons = 0.0, []
    try:
        import io
        import numpy as np
        from PIL import Image
        img = Image.open(p)
        software = str(img.info.get("Software", "") or img.getexif().get(305, "")).lower()
        if any(s in software for s in ("photoshop", "gimp", "canva", "paint.net", "pixlr")):
            score += 0.5
            reasons.append(f"Edited with image software ({software.strip()})")
        if p.suffix.lower() in {".jpg", ".jpeg"}:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, "JPEG", quality=90)
            buf.seek(0)
            diff = np.abs(np.asarray(rgb, dtype=np.int16) - np.asarray(Image.open(buf).convert("RGB"), dtype=np.int16))
            ela = diff.mean(axis=2)
            h, w = ela.shape
            blocks = ela[: h // 16 * 16, : w // 16 * 16].reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))
            if blocks.mean() > 0:
                ratio = float(blocks.max() / (blocks.mean() + 1e-6))
                if ratio > 6:
                    score += min(0.5, (ratio - 6) / 20 + 0.25)
                    reasons.append(f"Error-level analysis shows locally inconsistent compression (ratio {ratio:.1f})")
    except Exception:
        pass
    return min(1.0, score), reasons
