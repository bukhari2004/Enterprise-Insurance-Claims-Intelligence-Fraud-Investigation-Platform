"""Fraud Knowledge Graph (NetworkX in-memory, optional Neo4j mirror).

Nodes: Customer, Policy, Claim, Provider(Hospital/Workshop), Doctor, BankAccount, Phone, Address,
       Identity (national id), Vehicle, Investigator.
Fraud rings = groups of customers connected through *strong* shared entities (bank account, phone,
address, identity document, vehicle) and/or through providers that show a suspicious concentration
of high-risk claims.
"""
import hashlib
import logging
import math
from collections import defaultdict

import networkx as nx

log = logging.getLogger("graph")

STRONG = {"bank", "phone", "address", "identity", "vehicle"}
LABELS = {"customer": "Customer", "policy": "Policy", "claim": "Claim", "provider": "Provider",
          "doctor": "Doctor", "bank": "BankAccount", "phone": "Phone", "address": "Address",
          "identity": "Identity", "vehicle": "Vehicle", "investigator": "Investigator"}


def _n(kind, value):
    return f"{kind}:{str(value).strip().lower()}" if value not in (None, "") else None


class FraudGraph:
    def __init__(self, neo4j_uri="", user="", password=""):
        self.g = nx.Graph()
        self.suspect_providers = set()
        self.driver = None
        if neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
                self.driver.verify_connectivity()
            except Exception as exc:  # pragma: no cover
                log.warning("Neo4j unavailable (%s) - using NetworkX only", exc)
                self.driver = None

    # ------------------------------------------------------------ build
    def _add(self, node, kind, **attrs):
        if node is None:
            return None
        if node not in self.g:
            self.g.add_node(node, kind=kind, label=attrs.pop("label", node.split(":", 1)[1]), **attrs)
        else:
            self.g.nodes[node].update({k: v for k, v in attrs.items() if v is not None})
        return node

    def add_claim(self, c):
        cl = self._add(_n("claim", c["claim_no"]), "claim", label=c["claim_no"], amount=c.get("claim_amount"),
                       score=c.get("fraud_score") or 0.0, claim_type=c.get("claim_type"))
        cu = self._add(_n("customer", c["customer_id"]), "customer", label=c.get("customer_name") or c["customer_id"])
        links = [
            (cl, cu, "FILED_BY"),
            (cl, self._add(_n("policy", c.get("policy_no")), "policy"), "UNDER_POLICY"),
            (cu, self._add(_n("bank", c.get("bank_account")), "bank"), "PAYS_TO"),
            (cu, self._add(_n("phone", c.get("phone")), "phone"), "HAS_PHONE"),
            (cu, self._add(_n("address", c.get("address")), "address"), "LIVES_AT"),
            (cu, self._add(_n("identity", c.get("national_id")), "identity"), "IDENTIFIED_BY"),
            (cl, self._add(_n("vehicle", c.get("vehicle_plate")), "vehicle"), "INVOLVES"),
            (cl, self._add(_n("provider", c.get("provider_id")), "provider", label=c.get("provider_name"),
                           provider_type=c.get("provider_type")), "TREATED_OR_REPAIRED_BY"),
            (cl, self._add(_n("doctor", c.get("doctor")), "doctor", label=c.get("doctor")), "ATTENDED_BY"),
            (cl, self._add(_n("investigator", c.get("investigator")), "investigator"), "ASSIGNED_TO"),
        ]
        for a, b, rel in links:
            if a and b:
                self.g.add_edge(a, b, rel=rel)
        if self.driver:
            self._sync_neo4j(c)

    def rebuild(self, claims):
        self.g.clear()
        for c in claims:
            self.add_claim(c)

    def set_suspect_providers(self, provider_ids):
        self.suspect_providers = {_n("provider", p) for p in provider_ids}

    # ------------------------------------------------------------ queries
    def _customers_via(self, entity_node):
        return [n for n in self.g.neighbors(entity_node) if n.startswith("customer:")]

    def shared_entities(self, customer_id):
        """Strong entities of this customer that are also used by other customers."""
        cn = _n("customer", customer_id)
        out = []
        if cn not in self.g:
            return out
        for nb in self.g.neighbors(cn):
            kind = self.g.nodes[nb]["kind"]
            if kind in STRONG:
                others = [x for x in self._customers_via(nb) if x != cn]
                if others:
                    out.append({"type": kind, "value": self.g.nodes[nb]["label"],
                                "other_customers": [self.g.nodes[o]["label"] for o in others][:10],
                                "count": len(others)})
        return out

    def component_of(self, customer_id, cap=300):
        """Customers reachable through strong links / suspect providers (BFS, incremental-friendly)."""
        start = _n("customer", customer_id)
        if start not in self.g:
            return set()
        seen, queue = {start}, [start]
        while queue and len(seen) < cap:
            cur = queue.pop()
            for ent in self.g.neighbors(cur):
                kind = self.g.nodes[ent]["kind"]
                bridge = kind in STRONG
                if kind == "claim":                     # hop through claim -> suspect provider
                    for prov in self.g.neighbors(ent):
                        if prov in self.suspect_providers:
                            for cl in self.g.neighbors(prov):
                                for cu in self.g.neighbors(cl):
                                    if cu.startswith("customer:") and cu not in seen:
                                        seen.add(cu); queue.append(cu)
                if bridge:
                    for cu in self._customers_via(ent):
                        if cu not in seen:
                            seen.add(cu); queue.append(cu)
        return seen

    def ring_info(self, customer_id):
        members = self.component_of(customer_id)
        if len(members) < 3:
            return None
        return self._describe(members)

    def _describe(self, members):
        claims, shared, exposure, scores = [], defaultdict(set), 0.0, []
        for cu in members:
            for cl in self.g.neighbors(cu):
                if self.g.nodes[cl]["kind"] == "claim":
                    claims.append(cl)
                    exposure += float(self.g.nodes[cl].get("amount") or 0)
                    scores.append(float(self.g.nodes[cl].get("score") or 0))
                elif self.g.nodes[cl]["kind"] in STRONG:
                    if len([x for x in self._customers_via(cl) if x in members]) > 1:
                        shared[(self.g.nodes[cl]["kind"], self.g.nodes[cl]["label"])] = None
        provs = defaultdict(int)
        for cl in claims:
            for nb in self.g.neighbors(cl):
                if self.g.nodes[nb]["kind"] == "provider":
                    provs[self.g.nodes[nb]["label"]] += 1
        avg = sum(scores) / len(scores) if scores else 0.0
        ring_score = min(1.0, 0.35 * min(1.0, len(members) / 8) + 0.4 * avg + 0.25 * min(1.0, len(shared) / 3))
        rid = "RING-" + hashlib.md5(",".join(sorted(members)).encode()).hexdigest()[:6].upper()
        return {
            "ring_id": rid, "size": len(members), "claims": len(claims), "exposure": round(exposure, 2),
            "avg_claim_score": round(avg, 3), "ring_score": round(ring_score, 3),
            "members": sorted(self.g.nodes[m]["label"] for m in members),
            "shared_entities": [{"type": k[0], "value": k[1]} for k in shared],
            "top_providers": sorted(({"name": n, "claims": c} for n, c in provs.items()),
                                    key=lambda x: -x["claims"])[:3],
        }

    def rings(self, min_size=3):
        customers = [n for n, d in self.g.nodes(data=True) if d["kind"] == "customer"]
        seen, out = set(), []
        for cu in customers:
            if cu in seen:
                continue
            comp = self.component_of(cu.split(":", 1)[1], cap=500)
            seen |= comp
            if len(comp) >= min_size:
                out.append(self._describe(comp))
        return sorted(out, key=lambda r: -r["ring_score"])

    def provider_stats(self):
        """claims count & avg score per provider (used for suspicious provider detection)."""
        stats = {}
        for n, d in self.g.nodes(data=True):
            if d["kind"] == "provider":
                cl = [x for x in self.g.neighbors(n) if self.g.nodes[x]["kind"] == "claim"]
                sc = [float(self.g.nodes[x].get("score") or 0) for x in cl]
                stats[n.split(":", 1)[1]] = {"claims": len(cl), "avg_score": sum(sc) / len(sc) if sc else 0,
                                             "high_share": sum(s >= 0.5 for s in sc) / len(sc) if sc else 0}
        return stats

    def subgraph(self, claim_no, depth=2):
        start = _n("claim", claim_no)
        if start not in self.g:
            return {"nodes": [], "edges": []}
        nodes = nx.single_source_shortest_path_length(self.g, start, cutoff=depth)
        # avoid exploding through hub providers: cap neighbours of provider/doctor nodes
        keep = set()
        for n, dist in nodes.items():
            k = self.g.nodes[n]["kind"]
            if k in {"provider", "doctor"} and dist == depth:
                continue
            keep.add(n)
        sg = self.g.subgraph(list(keep)[:120])
        return {
            "nodes": [{"id": n, "kind": d["kind"], "label": d["label"], "score": d.get("score", 0)}
                      for n, d in sg.nodes(data=True)],
            "edges": [{"source": a, "target": b, "rel": d.get("rel", "")} for a, b, d in sg.edges(data=True)],
        }

    # ------------------------------------------------------------ neo4j
    def _sync_neo4j(self, c):  # pragma: no cover - requires a Neo4j server
        cy = """
        MERGE (cu:Customer {id:$cid}) SET cu.name=$cname
        MERGE (cl:Claim {no:$no}) SET cl.amount=$amount, cl.type=$ctype
        MERGE (cl)-[:FILED_BY]->(cu)
        FOREACH (_ IN CASE WHEN $bank IS NULL THEN [] ELSE [1] END |
            MERGE (b:BankAccount {id:$bank}) MERGE (cu)-[:PAYS_TO]->(b))
        FOREACH (_ IN CASE WHEN $phone IS NULL THEN [] ELSE [1] END |
            MERGE (p:Phone {id:$phone}) MERGE (cu)-[:HAS_PHONE]->(p))
        FOREACH (_ IN CASE WHEN $prov IS NULL THEN [] ELSE [1] END |
            MERGE (pr:Provider {id:$prov}) SET pr.name=$pname MERGE (cl)-[:TREATED_OR_REPAIRED_BY]->(pr))
        """
        try:
            with self.driver.session() as s:
                s.run(cy, cid=c["customer_id"], cname=c.get("customer_name"), no=c["claim_no"],
                      amount=c.get("claim_amount"), ctype=c.get("claim_type"), bank=c.get("bank_account") or None,
                      phone=c.get("phone") or None, prov=c.get("provider_id") or None, pname=c.get("provider_name"))
        except Exception as exc:
            log.warning("neo4j sync failed: %s", exc)
