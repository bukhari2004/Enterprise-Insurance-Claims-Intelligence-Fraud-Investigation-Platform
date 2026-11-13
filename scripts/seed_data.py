"""Generate a realistic synthetic portfolio with injected fraud scenarios, run it through the full
AI pipeline, and (optionally) write sample files you can upload through the dashboard/API.

    python scripts/seed_data.py            # ~400 claims into the configured database
    python scripts/seed_data.py --n 800 --reset
"""
import argparse
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BASE_DIR, settings  # noqa: E402
from app.pipeline import Platform  # noqa: E402

FIRST = ["Ali", "Sara", "Ahmed", "Fatima", "Usman", "Ayesha", "Bilal", "Hina", "Omar", "Zara", "Hamza", "Maryam",
         "Imran", "Nida", "Kamran", "Sana", "Farhan", "Rabia", "Tariq", "Mehwish", "Asad", "Iqra", "Saad", "Noor"]
LAST = ["Khan", "Raza", "Malik", "Sheikh", "Ahmed", "Butt", "Qureshi", "Siddiqui", "Hussain", "Chaudhry", "Baig", "Mirza"]
HOSPITALS = [("H-01", "Shifa Care Hospital"), ("H-02", "Green Valley Hospital"), ("H-03", "Unity Medical Center"),
             ("H-04", "Lakeview Hospital"), ("H-05", "Crescent Clinic"), ("H-06", "Noor General Hospital")]
WORKSHOPS = [("W-01", "AutoPro Garage"), ("W-02", "Motor City Workshop"), ("W-03", "Prime Auto Care"),
             ("W-04", "Zenith Repairs"), ("W-05", "Highway Auto Works"), ("W-06", "Star Motors")]
RING_WORKSHOP = ("W-99", "Speedy Fix Garage")
BAD_HOSPITAL = ("H-99", "City Care Clinic")
DOCTORS = ["Dr. A. Waheed", "Dr. S. Nasir", "Dr. R. Iqbal", "Dr. M. Farooq", "Dr. T. Aslam", "Dr. H. Javed"]
COVER = {"vehicle": 60000, "health": 40000, "property": 150000, "travel": 8000, "marine": 200000, "life": 250000}
AMT = {"vehicle": (7.6, 0.7), "health": (7.9, 0.8), "property": (8.8, 0.8), "travel": (6.5, 0.6),
       "marine": (9.0, 0.8), "life": (9.5, 0.6)}
DESC = {
    "vehicle": ["Rear-end collision at a traffic signal, bumper and boot damaged", "Side-swipe while overtaking on highway, door dented",
                "Windscreen cracked by flying debris on the motorway", "Hit a parked car while reversing, rear lamp broken"],
    "health": ["Hospital admission for appendicitis surgery", "Treatment for dengue fever, 4 days admitted",
               "Fracture of left arm, cast and physiotherapy", "Emergency treatment for food poisoning"],
    "property": ["Water pipe burst causing ceiling and wall damage", "Short circuit fire in kitchen", "Storm damaged roof and windows",
                 "Burglary with forced entry and stolen electronics"],
    "travel": ["Lost luggage on international flight", "Flight cancelled, additional hotel costs", "Medical emergency while abroad"],
    "marine": ["Cargo water damage during shipment", "Container damaged in port handling"],
    "life": ["Hospital expenses for critical illness benefit", "Accidental disability benefit claim"],
}


def make_image(path, kind, severity, seed):
    rng = np.random.default_rng(seed)
    W, H = 480, 320
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (int(rng.integers(150, 200)), int(rng.integers(150, 190)), int(rng.integers(140, 180)))
    img[int(H * 0.6):] = (90, 90, 95)
    x0, y0 = int(rng.integers(40, 100)), int(rng.integers(110, 150))
    cv2.rectangle(img, (x0, y0), (x0 + 330, y0 + 110), tuple(int(v) for v in rng.integers(40, 220, 3)), -1)
    cv2.rectangle(img, (x0 + 60, y0 - 40), (x0 + 250, y0), (200, 210, 220), -1)
    for cx in (x0 + 70, x0 + 260):
        cv2.circle(img, (cx, y0 + 110), 28, (20, 20, 20), -1)
    if kind in ("vehicle", "building", "asset"):
        for _ in range(int(2 + severity * 25)):
            p1 = (int(rng.integers(x0, x0 + 330)), int(rng.integers(y0, y0 + 110)))
            p2 = (p1[0] + int(rng.integers(-60, 60)), p1[1] + int(rng.integers(-30, 30)))
            cv2.line(img, p1, p2, (15, 15, 15), int(rng.integers(1, 4)))
        for _ in range(int(severity * 8)):
            c = (int(rng.integers(x0, x0 + 330)), int(rng.integers(y0, y0 + 110)))
            cv2.ellipse(img, c, (int(rng.integers(10, 40)), int(rng.integers(8, 25))), int(rng.integers(0, 180)), 0, 360, (30, 30, 35), -1)
    elif kind == "fire":
        for _ in range(int(5 + severity * 40)):
            c = (int(rng.integers(0, W)), int(rng.integers(40, H)))
            cv2.circle(img, c, int(rng.integers(15, 55)), (0, int(rng.integers(90, 160)), 255), -1)
        for _ in range(int(severity * 20)):
            cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(20, 60)), (10, 10, 10), -1)
    elif kind == "flood":
        top = int(H * (1 - severity * 0.9))
        img[top:] = (int(rng.integers(40, 60)), int(rng.integers(90, 110)), int(rng.integers(130, 150)))
    noise = rng.normal(0, 6, img.shape)
    cv2.imwrite(str(path), np.clip(img + noise, 0, 255).astype(np.uint8))


def doc_text(kind, c, total, provider, doc_date, name=None, items=None, inconsistent=False):
    name = name or c["customer_name"]
    items = items or [("Labour", round(total * 0.35)), ("Parts", round(total * 0.6)), ("Tax", total - round(total * 0.35) - round(total * 0.6))]
    shown = total + (round(total * 0.2) if inconsistent else 0)
    body = "\n".join(f"{i} {a:.2f}" for i, a in items)
    if kind == "medical_report":
        return (f"Medical Report\nHospital: {provider}\nPatient Name: {name}\nPolicy No: {c['policy_no']}\nDoctor: {c.get('doctor','')}\n"
                f"Diagnosis: {c['description'][:50]}\nDischarge Date: {doc_date}\n{body}\nTotal: {shown:.2f}\n")
    if kind == "invoice":
        return (f"Invoice\nInvoice No: INV-{random.randint(1000,9999)}\nHospital: {provider}\nName: {name}\nPolicy No: {c['policy_no']}\n"
                f"Date: {doc_date}\n{body}\nTotal: {shown:.2f}\n")
    if kind == "police_fir":
        return (f"First Information Report\nFIR No: {random.randint(100,999)}/{date.today().year % 100}\nPolice Station: Central\n"
                f"Complainant: {name}\nDate: {doc_date}\nDetails: {c['description']}\n")
    if kind == "receipt":
        return (f"Receipt\nName: {name}\nPolicy No: {c['policy_no']}\nDate: {doc_date}\nPaid via card\n{body}\nTotal: {shown:.2f}\n")
    return (f"Repair Estimate\nWorkshop: {provider}\nName: {name}\nPolicy No: {c['policy_no']}\nDate: {doc_date}\n{body}\nTotal: {shown:.2f}\n")


def run(n=400, reset=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if reset:
        dbfile = settings.database_url.replace("sqlite:///", "")
        if settings.database_url.startswith("sqlite") and os.path.exists(dbfile):
            os.remove(dbfile)
    pf = Platform()
    imgdir = BASE_DIR / "data" / "seed_images"
    imgdir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    # ---- customers & policies
    customers = []
    for i in range(1, 161):
        name = f"{random.choice(FIRST)} {random.choice(LAST)}"
        cu = {"customer_id": f"C{i:04d}", "customer_name": name,
              "national_id": f"{random.randint(10000,99999)}-{random.randint(1000000,9999999)}-{random.randint(1,9)}",
              "phone": f"03{random.randint(0,4)}{random.randint(10000000,99999999)}",
              "address": f"House {random.randint(1,400)}, Street {random.randint(1,60)}, Sector {random.choice('ABCDEFGH')}-{random.randint(5,16)}",
              "bank_account": f"PK{random.randint(10**10, 10**11-1)}", "policies": []}
        for t in random.sample(list(COVER), random.randint(1, 3)):
            pol = {"policy_no": f"POL-{100000 + len(customers) * 10 + len(cu['policies'])}", "customer_id": cu["customer_id"],
                   "policy_type": t, "coverage_limit": COVER[t], "deductible": round(COVER[t] * 0.01, -1),
                   "start_date": (today - timedelta(days=random.randint(200, 1200))).isoformat(),
                   "end_date": (today + timedelta(days=random.randint(60, 365))).isoformat()}
            pf.create_policy(pol)
            cu["policies"].append(pol)
        customers.append(cu)

    def pick(types=None):
        while True:
            cu = random.choice(customers)
            pols = [p for p in cu["policies"] if not types or p["policy_type"] in types]
            if pols:
                return cu, random.choice(pols)

    specs = []

    def spec(cu, pol, filed_offset, amount, desc=None, provider=None, doctor=None, plate=None, doc_total=None, **kw):
        t = pol["policy_type"]
        image = kw.pop("_image", None)
        filed = today - timedelta(days=filed_offset)
        inc = filed - timedelta(days=kw.pop("delay", random.randint(0, 9)))
        if provider is None:
            provider = random.choice(HOSPITALS) if t in ("health", "life") else random.choice(WORKSHOPS) if t in ("vehicle", "property") else ("", "")
        specs.append(dict(claim_type=t, policy_no=pol["policy_no"], customer_id=cu["customer_id"], customer_name=cu["customer_name"],
                          national_id=cu["national_id"], phone=cu["phone"], address=cu["address"], bank_account=cu["bank_account"],
                          provider_id=provider[0], provider_name=provider[1], provider_type="hospital" if provider[0].startswith("H") else "workshop" if provider[0] else "",
                          doctor=doctor or (random.choice(DOCTORS) if provider[0].startswith("H") else ""),
                          vehicle_plate=plate or (f"LEA-{random.randint(1000,9999)}" if t == "vehicle" else ""),
                          claim_amount=round(amount), incident_date=inc.isoformat(), filed_date=filed.isoformat(),
                          description=desc or random.choice(DESC[t]), _doc_total=doc_total if doc_total is not None else round(amount),
                          _image=image or ("own" if t in ("vehicle", "property") and random.random() < 0.6 else None),
                          **{k: v for k, v in kw.items() if k.startswith("_")}))

    n_normal = n - 60
    for _ in range(n_normal):
        cu, pol = pick()
        mu, sg = AMT[pol["policy_type"]]
        amt = min(float(np.random.lognormal(mu, sg)), pol["coverage_limit"] * 0.8)
        spec(cu, pol, random.randint(1, 160), max(amt, 150))

    # --- scenario 1: organised ring (shared bank/address, one garage, reused photo, inflated amounts)
    ring_customers = random.sample([c for c in customers if any(p["policy_type"] == "vehicle" for p in c["policies"])], 7)
    shared_banks, shared_addr = ["PK99000011111", "PK99000022222"], "Plot 7, Industrial Area, Sector I-9"
    for k, cu in enumerate(ring_customers):
        cu["bank_account"] = shared_banks[k % 2]
        if k < 4:
            cu["address"] = shared_addr
        pol = next(p for p in cu["policies"] if p["policy_type"] == "vehicle")
        spec(cu, pol, random.randint(3, 40), random.randint(24000, 42000), provider=RING_WORKSHOP, doc_total=random.randint(9000, 13000),
             desc="Collision damage to front end and engine bay", _image="ring")
    # --- scenario 2: bad hospital inflating bills
    for _ in range(14):
        cu, pol = pick({"health"})
        spec(cu, pol, random.randint(2, 70), random.randint(22000, 38000), provider=BAD_HOSPITAL, doc_total=random.randint(6000, 11000))
    # --- scenario 3: duplicates (same incident resubmitted)
    for base in random.sample([s for s in specs[:n_normal] if s["claim_type"] in ("vehicle", "health")], 6):
        dup = dict(base)
        dup["filed_date"] = (date.fromisoformat(base["filed_date"]) + timedelta(days=random.randint(1, 6))).isoformat()
        dup["_image"] = None
        specs.append(dup)
    # --- scenario 4: identity fraud (same national ID, different person)
    for _ in range(4):
        victim, _p = pick()
        thief = random.choice(customers)
        pol = random.choice(thief["policies"])
        fake = dict(thief, national_id=victim["national_id"])
        spec(fake, pol, random.randint(2, 50), random.randint(5000, 15000))
    # --- scenario 5: serial claimant
    cu, pol = pick({"vehicle"})
    for k in range(7):
        spec(cu, pol, 5 + k * 12, random.randint(3000, 9000), provider=random.choice(WORKSHOPS))
    # --- scenario 6: document manipulation
    for _ in range(6):
        cu, pol = pick({"vehicle", "health"})
        spec(cu, pol, random.randint(2, 60), random.randint(4000, 12000), _doc_flaw=random.choice(["date", "name", "sum"]))

    specs.sort(key=lambda s: s["filed_date"])
    ring_img = imgdir / "ring_shared.png"
    make_image(ring_img, "vehicle", 0.9, 7)
    done = 0
    for i, s in enumerate(specs):
        s = dict(s)
        doc_total, image, flaw = s.pop("_doc_total"), s.pop("_image", None), s.pop("_doc_flaw", None)
        s["created_at"] = s["filed_date"] + "T09:00:00"
        c = pf.submit_claim(s, actor="seed")
        provider = s["provider_name"] or "Provider"
        incident = s["incident_date"]
        dday = (date.fromisoformat(incident) + timedelta(days=random.randint(0, 3))).isoformat()
        name = s["customer_name"]
        if flaw == "date":
            dday = (date.fromisoformat(incident) - timedelta(days=30)).isoformat()
        if flaw == "name":
            name = f"{random.choice(FIRST)} {random.choice(LAST)}"
        have_docs = random.random() < 0.85 or s["provider_id"] in ("W-99", "H-99")
        if have_docs:
            kind = {"health": "medical_report", "life": "medical_report", "vehicle": "repair_estimate", "property": "repair_estimate",
                    "travel": "receipt", "marine": "invoice"}[s["claim_type"]]
            pf.add_document(c["claim_no"], text=doc_text(kind, s, doc_total, provider, dday, name=name, inconsistent=(flaw == "sum")),
                            doc_type=kind, actor="seed", rescore=False)
            if s["claim_type"] in ("vehicle", "property") and s["claim_amount"] >= 10000:
                pf.add_document(c["claim_no"], text=doc_text("police_fir", s, 0, provider, dday), doc_type="police_fir", actor="seed", rescore=False)
        if image == "ring":
            pf.add_image(c["claim_no"], "ring_shared.png", path=ring_img, actor="seed", rescore=False)
        elif image == "own":
            kind = "fire" if "fire" in s["description"].lower() else "flood" if "water" in s["description"].lower() else \
                "building" if s["claim_type"] == "property" else "vehicle"
            p = imgdir / f"{c['claim_no']}.png"
            make_image(p, kind, float(min(0.9, max(0.1, (s['claim_amount'] / 6000) ** (1 / 1.2)))) if kind == 'vehicle' else random.uniform(0.2, 0.7), 1000 + i)
            pf.add_image(c["claim_no"], p.name, path=p, actor="seed", rescore=False)
        pf.bus.publish("claim.analyzed", {"claim_id": c["claim_id"], "actor": "seed"})
        done += 1
        if done % 100 == 0:
            print(f"  processed {done}/{len(specs)} claims")

    print("  re-scoring suspected ring members:", pf.rescore_ring_members(), "claims")
    # ---- simulate historical human decisions so dashboards have settlement times
    rows = pf.db.query("SELECT claim_id, filed_date, status, risk_level FROM claims")
    for r in rows:
        age = (today - date.fromisoformat(r["filed_date"])).days
        new, days = None, None
        if r["status"] == "AUTO_APPROVED" and age > 7:
            new, days = "SETTLED", random.randint(1, 5)
        elif r["status"] in ("INVESTIGATING", "PENDING_DOCUMENTS") and r["risk_level"] in ("LOW", "MEDIUM") and age > 45:
            new, days = "SETTLED", random.randint(8, 30)
        elif r["status"] in ("ESCALATED", "INVESTIGATING") and r["risk_level"] in ("HIGH", "CRITICAL") and age > 60:
            new, days = "REJECTED", random.randint(15, 40)
        if new:
            dec = datetime.fromisoformat(r["filed_date"] + "T09:00:00") + timedelta(days=days)
            pf.db.update("claims", "claim_id", r["claim_id"], {"status": new, "decided_at": dec.isoformat(timespec="seconds")})
    print(f"Seeded {len(specs)} claims, {pf.db.scalar('SELECT COUNT(*) FROM alerts')} alerts, {len(pf.graph.rings())} suspected rings.")
    return pf


def write_samples():
    """Sample files to try the upload endpoints / dashboard by hand."""
    from PIL import Image, ImageDraw
    out = BASE_DIR / "samples"
    out.mkdir(exist_ok=True)
    for kind, sev, name in [("vehicle", 0.15, "car_minor_damage.png"), ("vehicle", 0.8, "car_severe_damage.png"),
                            ("fire", 0.7, "house_fire.png"), ("flood", 0.6, "flooded_property.png")]:
        make_image(out / name, kind, sev, sum(map(ord, name)) % 1000)
    txt = ("Repair Estimate\nWorkshop: Speedy Fix Garage\nName: Ali Raza\nPolicy No: POL-100000\nDate: 2026-01-10\n"
           "Labour 3000.00\nParts 9000.00\nTax 1000.00\nTotal: 13000.00\n")
    (out / "repair_estimate.txt").write_text(txt)
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            font = ImageFont.load_default()
    img = Image.new("RGB", (1000, 560), "white")
    d = ImageDraw.Draw(img)
    y = 30
    for line in txt.splitlines():
        d.text((40, y), line, fill="black", font=font)
        y += 48
    img.save(out / "repair_estimate_scan.png")
    print("Sample files written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--samples-only", action="store_true")
    a = ap.parse_args()
    if a.samples_only:
        write_samples()
    else:
        run(a.n, a.reset)
        write_samples()
