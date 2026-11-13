"""Run:  python -m unittest discover -s tests -v      (or: pytest -q)"""
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.pipeline import Platform  # noqa: E402
from app.services import document_intel, vision  # noqa: E402

TODAY = date.today()


def mk_platform():
    pf = Platform("sqlite:///:memory:")
    for i in range(1, 12):
        pf.create_policy({"policy_no": f"POL-{1000+i}", "customer_id": f"C{i}", "policy_type": "vehicle",
                          "coverage_limit": 50000, "deductible": 200,
                          "start_date": (TODAY - timedelta(days=400)).isoformat(),
                          "end_date": (TODAY + timedelta(days=200)).isoformat()})
    return pf


def claim(i, amount=3000, **kw):
    base = dict(claim_type="vehicle", policy_no=f"POL-{1000+i}", customer_id=f"C{i}", customer_name=f"Customer {chr(64+i)}son",
                claim_amount=amount, incident_date=(TODAY - timedelta(days=5)).isoformat(),
                description="rear bumper dent in parking lot", bank_account=f"BANK{i}", phone=f"0300000{i:04d}",
                provider_id="W-1", provider_name="AutoPro Garage", provider_type="workshop")
    base.update(kw)
    return base


def estimate(c, total, date_=None, name=None):
    d = date_ or (TODAY - timedelta(days=4)).isoformat()
    return (f"Repair Estimate\nWorkshop: AutoPro Garage\nName: {name or c['customer_name']}\nPolicy No: {c['policy_no']}\n"
            f"Date: {d}\nLabour {total*0.4:.2f}\nParts {total*0.6:.2f}\nTotal: {total:.2f}\n")


class DocumentIntelTests(unittest.TestCase):
    def test_extract_and_inflation_detected(self):
        c = claim(1, amount=9000)
        ex = document_intel.extract(estimate(c, 3000))
        self.assertEqual(ex["doc_type"], "repair_estimate")
        self.assertEqual(ex["total_amount"], 3000.0)
        codes = {i["code"] for i in document_intel.validate(c, ex)}
        self.assertIn("AMOUNT_INFLATED", codes)

    def test_name_mismatch_and_date_before_incident(self):
        c = claim(1)
        txt = estimate(c, 3000, date_=(TODAY - timedelta(days=60)).isoformat(), name="Totally Different")
        codes = {i["code"] for i in document_intel.validate(c, document_intel.extract(txt))}
        self.assertTrue({"NAME_MISMATCH", "DOC_BEFORE_INCIDENT"} <= codes)

    def test_total_inconsistency(self):
        txt = "Invoice\nInvoice No: INV-1234\nHospital: X Hospital\nName: A B\nConsultation 100\nMedicine 200\nTotal: 900\n"
        codes = {i["code"] for i in document_intel.validate({"claim_amount": 900, "customer_name": "A B"}, document_intel.extract(txt))}
        self.assertIn("TOTAL_INCONSISTENT", codes)


class VisionTests(unittest.TestCase):
    def test_severity_orders_correctly_and_hash_matches(self):
        import cv2, numpy as np
        with tempfile.TemporaryDirectory() as d:
            clean = np.full((240, 320, 3), 170, np.uint8)
            damaged = clean.copy()
            rng = np.random.default_rng(1)
            for _ in range(60):
                p = (int(rng.integers(0, 320)), int(rng.integers(0, 240)))
                cv2.line(damaged, p, (p[0] + 40, p[1] + 25), (10, 10, 10), 2)
            a, b = Path(d) / "a.png", Path(d) / "b.png"
            cv2.imwrite(str(a), clean); cv2.imwrite(str(b), damaged)
            ra, rb = vision.assess(a, "vehicle"), vision.assess(b, "vehicle")
            self.assertLess(ra["severity"], rb["severity"])
            self.assertLess(ra["estimated_cost"]["expected"], rb["estimated_cost"]["expected"])
            self.assertEqual(vision.hamming(vision.dhash(b), vision.dhash(b)), 0)


class PipelineTests(unittest.TestCase):
    def test_clean_claim_auto_approved(self):
        pf = mk_platform()
        c = pf.submit_claim(claim(1, 2500))
        pf.add_document(c["claim_no"], text=estimate(claim(1, 2500), 2500))
        d = pf.claim_detail(c["claim_no"])
        self.assertEqual(d["risk_level"], "LOW")
        self.assertEqual(d["recommendation"], "AUTO_APPROVE")

    def test_missing_documents_requested(self):
        pf = mk_platform()
        c = pf.submit_claim(claim(1, 2500))
        self.assertEqual(pf.claim_detail(c["claim_no"])["recommendation"], "ADDITIONAL_DOCUMENTS")

    def test_inflated_claim_flagged_and_explained(self):
        pf = mk_platform()
        cdata = claim(2, 9000)
        c = pf.submit_claim(cdata)
        pf.add_document(c["claim_no"], text=estimate(cdata, 3000))
        d = pf.claim_detail(c["claim_no"])
        self.assertNotEqual(d["recommendation"], "AUTO_APPROVE")
        self.assertTrue(any(i["code"] == "DOC_AMOUNT_INFLATED" for i in d["indicators"]))
        self.assertIn("fraud risk score", d["explanation"]["summary"])

    def test_duplicate_claim_detected(self):
        pf = mk_platform()
        pf.submit_claim(claim(3, 4000))
        c2 = pf.submit_claim(claim(3, 4000))
        d = pf.claim_detail(c2["claim_no"])
        self.assertTrue(any(i["code"] == "DUPLICATE_CLAIM" for i in d["indicators"]))
        self.assertTrue(any(a["alert_type"] == "DUPLICATE_CLAIM" for a in d["alerts"]))

    def test_identity_fraud_detected(self):
        pf = mk_platform()
        pf.submit_claim(claim(4, 3000, national_id="12345-1234567-1"))
        c = pf.submit_claim(claim(5, 3000, national_id="12345-1234567-1"))
        self.assertTrue(any(i["code"] == "IDENTITY_MISMATCH" for i in pf.claim_detail(c["claim_no"])["indicators"]))

    def test_fraud_ring_found_through_shared_bank_account(self):
        pf = mk_platform()
        for i in (6, 7, 8, 9):
            pf.submit_claim(claim(i, 20000, bank_account="SHARED-ACC-1", provider_id="W-99", provider_name="Speedy Fix Garage"))
        rings = pf.graph.rings()
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0]["size"], 4)
        pf.rescore_ring_members()
        self.assertTrue(all(pf.claim_detail(c["claim_no"])["ring_id"] for c in pf.db.query("SELECT claim_no FROM claims")))

    def test_reused_photo_across_customers(self):
        import cv2, numpy as np
        pf = mk_platform()
        with tempfile.TemporaryDirectory() as d:
            img = np.random.default_rng(3).integers(0, 255, (200, 300, 3), dtype=np.uint8)
            p = Path(d) / "x.png"; cv2.imwrite(str(p), img)
            a = pf.submit_claim(claim(1, 5000)); b = pf.submit_claim(claim(2, 5000))
            pf.add_image(a["claim_no"], "x.png", path=p)
            pf.add_image(b["claim_no"], "x.png", path=p)
        self.assertTrue(any(i["code"] == "REUSED_IMAGE" for i in pf.claim_detail(b["claim_no"])["indicators"]))

    def test_human_decision_audited_and_final(self):
        pf = mk_platform()
        c = pf.submit_claim(claim(1, 2500))
        pf.decide(c["claim_no"], "REJECTED", "alice", "insufficient evidence")
        pf.bus.publish("claim.analyzed", {"claim_id": c["claim_id"]})      # re-score must not override a human decision
        self.assertEqual(pf.claim_detail(c["claim_no"])["status"], "REJECTED")
        self.assertTrue(pf.db.query("SELECT 1 FROM audit_log WHERE actor='alice' AND action='claim.rejected'"))

    def test_validation(self):
        pf = mk_platform()
        with self.assertRaises(ValueError):
            pf.submit_claim({"claim_type": "vehicle"})
        with self.assertRaises(ValueError):
            pf.submit_claim(claim(1, -5))


class CopilotTests(unittest.TestCase):
    def test_copilot_intents(self):
        pf = mk_platform()
        cdata = claim(2, 9000)
        c = pf.submit_claim(cdata)
        pf.add_document(c["claim_no"], text=estimate(cdata, 3000))
        for _ in range(5):
            pf.submit_claim(claim(3, 3000, description=f"bumper scratch {_}"))
        no = c["claim_no"]
        self.assertEqual(pf.copilot.ask("why was this flagged?", no)["intent"], "explain")
        self.assertIn("Indicative settlement", pf.copilot.ask("estimate settlement", no)["answer"])
        self.assertEqual(pf.copilot.ask("show fraud indicators", no)["intent"], "indicators")
        self.assertIn("Most similar", pf.copilot.ask("compare with similar claims", no)["answer"])
        self.assertIn(no, pf.copilot.ask(f"summarize {no}")["answer"])
        self.assertIn("couldn't find", pf.copilot.ask("why rejected CLM-999999")["answer"])


@unittest.skipUnless(__import__("importlib").util.find_spec("fastapi"), "fastapi not installed")
class ApiTests(unittest.TestCase):
    def test_api_roundtrip(self):
        from fastapi.testclient import TestClient
        import app.main as m
        with TestClient(m.app) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "ok")
            client.post("/api/policies", json={"policy_no": "POL-9", "customer_id": "C9", "policy_type": "vehicle", "coverage_limit": 50000,
                                               "start_date": "2024-01-01", "end_date": "2030-01-01"})
            r = client.post("/api/claims", json={"claim_type": "vehicle", "policy_no": "POL-9", "customer_id": "C9", "customer_name": "Test User",
                                                 "claim_amount": 1200, "incident_date": TODAY.isoformat()})
            self.assertEqual(r.status_code, 201)
            no = r.json()["claim_no"]
            self.assertIn("answer", client.post("/api/copilot", json={"question": "summarize", "claim_ref": no}).json())
            self.assertGreaterEqual(client.get("/api/dashboard").json()["total_claims"], 1)


if __name__ == "__main__":
    unittest.main()
