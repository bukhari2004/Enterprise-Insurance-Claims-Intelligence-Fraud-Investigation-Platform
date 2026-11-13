"""Computer-vision damage assessment + perceptual hashing for reused-photo detection.

Two engines behind one interface:
  * YOLO (Ultralytics YOLO11) - used automatically when YOLO_WEIGHTS points to a trained model.
  * OpenCV heuristic fallback - colour/edge statistics so the whole pipeline works out-of-the-box
    (no GPU, no weights). It is a *baseline*; see docs/ARCHITECTURE.md for the training plan.
"""
import numpy as np

from app.config import settings

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# base repair/replacement cost per fully-damaged asset (configurable, currency-neutral)
BASE_COST = {"vehicle": 6000, "building": 20000, "flood": 15000, "fire": 25000, "asset": 2500}
CLAIM_TO_DAMAGE = {"vehicle": "vehicle", "property": "building", "marine": "asset",
                   "travel": "asset", "health": "asset", "life": "asset"}
YOLO_CLASS_WEIGHT = {"scratch": 0.15, "dent": 0.35, "crack": 0.4, "broken_glass": 0.5, "lamp_broken": 0.4,
                     "tire_flat": 0.3, "bumper_damage": 0.5, "severe_damage": 1.0, "fire": 1.0, "flood": 0.9}
_yolo = None


def dhash(path, size=8):
    """64-bit difference hash as hex; robust to resize/recompression."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _label(s):
    return "none" if s < 0.1 else "minor" if s < 0.3 else "moderate" if s < 0.6 else "severe" if s < 0.85 else "total_loss"


def _heuristic(img, claim_type):
    h, w = img.shape[:2]
    scale = 480 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    total = H.size
    flame = float(np.mean(((H <= 25) | (H >= 170)) & (S > 140) & (V > 170)))
    char = float(np.mean(V < 40))
    mud = float(np.mean((H >= 8) & (H <= 25) & (S >= 60) & (S <= 200) & (V >= 60) & (V <= 190)))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    edge_density = float(edges.mean() / 255)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c) / total
        if area > 0.01:
            x, y, bw, bh = cv2.boundingRect(c)
            regions.append({"bbox": [x / img.shape[1], y / img.shape[0], bw / img.shape[1], bh / img.shape[0]],
                            "area_ratio": round(float(area), 3)})
    damaged_ratio = min(1.0, sum(r["area_ratio"] for r in regions))
    signals = {
        "fire": min(1.0, 2.0 * (flame + char)),
        "flood": min(1.0, 1.8 * mud) if edge_density < 0.12 else 0.0,
        # edge density above a clean-scene baseline (~0.5%) scales to 0..1 at ~2.5%; blob area adds a small bonus
        "structural": min(1.0, 0.85 * max(0.0, (edge_density - 0.005) / 0.02) + 0.15 * min(1.0, damaged_ratio * 2)),
    }
    default = CLAIM_TO_DAMAGE.get(claim_type, "asset")
    kind = default
    if signals["fire"] > 0.35 and signals["fire"] >= signals["flood"]:
        kind = "fire"
    elif signals["flood"] > 0.35:
        kind = "flood"
    severity = signals["fire"] if kind == "fire" else signals["flood"] if kind == "flood" else signals["structural"]
    return kind, float(min(1.0, severity)), regions, {k: round(v, 3) for k, v in signals.items()}


def _yolo_assess(path):
    global _yolo
    from ultralytics import YOLO
    if _yolo is None:
        _yolo = YOLO(settings.yolo_weights)
    res = _yolo(str(path), verbose=False)[0]
    names = res.names
    dets, sev = [], 0.0
    img_area = res.orig_shape[0] * res.orig_shape[1]
    for b in res.boxes:
        cls = names[int(b.cls)]
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        area = (x2 - x1) * (y2 - y1) / img_area
        dets.append({"class": cls, "confidence": round(float(b.conf), 3),
                     "bbox": [x1, y1, x2, y2], "area_ratio": round(area, 4)})
        sev += YOLO_CLASS_WEIGHT.get(cls, 0.3) * min(1.0, area * 6 + 0.1) * float(b.conf)
    return dets, min(1.0, sev)


def assess(path, claim_type="vehicle"):
    """Return {damage_type, severity, severity_label, estimated_cost{low,expected,high}, method, detections, ...}."""
    if cv2 is None:
        return {"error": "OpenCV not installed"}
    img = cv2.imread(str(path))
    if img is None:
        return {"error": "Unreadable image"}
    method, detections, signals = "opencv-heuristic", [], {}
    if settings.yolo_weights:
        try:
            detections, severity = _yolo_assess(path)
            kind = CLAIM_TO_DAMAGE.get(claim_type, "asset")
            method = "yolo11"
            regions = []
        except Exception as exc:  # fall back rather than fail the claim
            method = f"opencv-heuristic (yolo failed: {exc.__class__.__name__})"
            kind, severity, regions, signals = _heuristic(img, claim_type)
    else:
        kind, severity, regions, signals = _heuristic(img, claim_type)
    base = BASE_COST.get(kind, BASE_COST["asset"])
    expected = base * (severity ** 1.2)
    return {
        "damage_type": kind, "severity": round(severity, 3), "severity_label": _label(severity),
        "estimated_cost": {"low": round(expected * 0.75), "expected": round(expected), "high": round(expected * 1.3)},
        "currency": settings.currency, "method": method, "signals": signals,
        "regions": regions if method.startswith("opencv") else [], "detections": detections,
        "confidence": 0.6 if method.startswith("opencv") else 0.85,
    }
