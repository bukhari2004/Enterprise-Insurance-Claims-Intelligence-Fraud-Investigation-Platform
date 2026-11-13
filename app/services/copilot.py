"""AI Claims Copilot - multi-agent, grounded in platform data.

Agents: router -> {explain | settlement | indicators | similar | summary | network} -> polish.
Uses LangGraph when installed (graph of agents), otherwise an equivalent plain-Python dispatcher.
Answers are built from structured facts (no hallucinated numbers); an optional LLM only rewords.
"""
import re
from typing import TypedDict

from app.services import llm
from app.services.recommender import estimate_settlement

try:
    from langgraph.graph import END, StateGraph
    HAVE_LANGGRAPH = True
except Exception:
    HAVE_LANGGRAPH = False

INTENTS = [
    ("similar", r"similar|compare|histor|precedent"),
    ("settlement", r"settle|payout|pay out|estimate|how much"),
    ("network", r"ring|network|connected|linked|graph|related"),
    ("explain", r"why|explain|reason|reject|decline|decision|recommend"),
    ("indicators", r"indicator|red flag|fraud|risk|suspicious|flag"),
    ("summary", r"summar|overview|brief|tell me about"),
]


class CopilotState(TypedDict, total=False):
    question: str
    claim_ref: str
    claim: dict
    intent: str
    answer: str
    data: dict


class Copilot:
    def __init__(self, platform):
        self.p = platform
        self.graph = self._build_graph() if HAVE_LANGGRAPH else None

    # ---------------------------------------------------------- agents
    def route(self, s):
        q = s["question"].lower()
        m = re.search(r"CLM-\d+", s["question"], re.I)
        ref = s.get("claim_ref") or (m.group(0).upper() if m else None)
        s["claim_ref"] = ref
        s["intent"] = next((i for i, rx in INTENTS if re.search(rx, q)), "help")
        if ref:
            s["claim"] = self.p.claim_detail(ref)
            if not s["claim"]:
                s["intent"], s["answer"] = "missing", f"I couldn't find claim {ref}."
        elif s["intent"] != "help":
            s["intent"], s["answer"] = "help", "Please tell me which claim (e.g. CLM-000123) or select one in the dashboard."
        return s

    def a_help(self, s):
        s.setdefault("answer", "Ask me things like: 'Why was CLM-000012 flagged?', 'Estimate settlement for CLM-000012', "
                     "'Show fraud indicators', 'Compare with similar claims', 'Is it part of a fraud ring?' or 'Summarize CLM-000012'.")
        return s

    def a_explain(self, s):
        c = s["claim"]
        ex = c.get("explanation") or {}
        s["answer"] = ex.get("summary", "This claim has not been scored yet.")
        if ex.get("document_issues"):
            s["answer"] += "\n\nDocument findings:\n" + "\n".join(f"- {i['message']}" for i in ex["document_issues"][:5])
        s["data"] = {"explanation": ex}
        return s

    def a_indicators(self, s):
        c = s["claim"]
        inds = c.get("indicators") or []
        if not inds:
            s["answer"] = f"No fraud indicators were raised for {c['claim_no']} (score {c.get('fraud_score') or 0:.0%})."
        else:
            lines = [f"- **{i['label']}** (+{i['points']} pts): {i['detail']}" for i in inds[:8]]
            s["answer"] = f"Fraud indicators for {c['claim_no']} (score {c['fraud_score']:.0%}, {c['risk_level']}):\n" + "\n".join(lines)
        s["data"] = {"indicators": inds}
        return s

    def a_settlement(self, s):
        c = s["claim"]
        st = (c.get("explanation") or {}).get("settlement")
        if not st:
            st = estimate_settlement(c, c["documents"], c.get("policy"), c.get("est_damage_cost"), hold=False)
        lines = "\n".join(f"- {x['label']}: {x['amount']:+,.0f}" for x in st["steps"])
        s["answer"] = (f"Indicative settlement for {c['claim_no']}: **{st['indicative_amount']:,.0f} {st['currency']}**. "
                       f"{st['note']}.\n{lines}")
        s["data"] = {"settlement": st}
        return s

    def a_similar(self, s):
        sims = self.p.similar_claims(s["claim"])
        if not sims:
            s["answer"] = "Not enough historical claims of this type to compare."
        else:
            lines = [f"- {x['claim_no']} ({x['claim_amount']:,.0f}, {x['recommendation'] or x['status']}, "
                     f"risk {x['fraud_score'] or 0:.0%}) similarity {x['similarity']:.0%}" for x in sims]
            avg = sum(x["claim_amount"] for x in sims) / len(sims)
            s["answer"] = (f"Most similar historical claims to {s['claim']['claim_no']} (average amount {avg:,.0f}):\n"
                           + "\n".join(lines))
        s["data"] = {"similar": sims}
        return s

    def a_network(self, s):
        c = s["claim"]
        ring = (c.get("explanation") or {}).get("ring")
        shared = self.p.graph.shared_entities(c["customer_id"])
        if ring:
            s["answer"] = (f"Yes - {c['customer_name']} is in **{ring['ring_id']}**: {ring['size']} customers, {ring['claims']} claims, "
                           f"exposure {ring['exposure']:,.0f}. Shared entities: "
                           + ", ".join(f"{e['type']} {e['value']}" for e in ring["shared_entities"][:5]) + ".")
        elif shared:
            s["answer"] = "Linked entities: " + "; ".join(f"{e['type']} {e['value']} also used by {', '.join(e['other_customers'][:3])}" for e in shared)
        else:
            s["answer"] = "No suspicious connections found in the knowledge graph for this customer."
        s["data"] = {"ring": ring, "shared": shared}
        return s

    def a_summary(self, s):
        c = s["claim"]
        ex = c.get("explanation") or {}
        s["answer"] = (f"**{c['claim_no']}** - {c['claim_type'].title()} claim by {c['customer_name']} for {c['claim_amount']:,.0f}, "
                       f"incident {c['incident_date']}, filed {c['filed_date']}. Status: {c['status']}. "
                       f"{len(c['documents'])} document(s), {len(c['images'])} image(s). "
                       f"Risk {c.get('fraud_score') or 0:.0%} ({c.get('risk_level')}). {ex.get('summary', '')}")
        return s

    def polish(self, s):
        facts = s.get("answer", "")
        better = llm.polish(s["question"], facts)
        if better:
            s["answer"] = better
        return s

    # ---------------------------------------------------------- orchestration
    def _build_graph(self):  # pragma: no cover - needs langgraph
        g = StateGraph(CopilotState)
        g.add_node("route", self.route)
        agents = {"explain": self.a_explain, "settlement": self.a_settlement, "indicators": self.a_indicators,
                  "similar": self.a_similar, "summary": self.a_summary, "network": self.a_network,
                  "help": self.a_help, "missing": self.a_help}
        for name, fn in agents.items():
            g.add_node(name, fn)
            g.add_edge(name, "polish")
        g.add_node("polish", self.polish)
        g.set_entry_point("route")
        g.add_conditional_edges("route", lambda s: s["intent"], {k: k for k in agents})
        g.add_edge("polish", END)
        return g.compile()

    def ask(self, question, claim_ref=None):
        state = {"question": question, "claim_ref": claim_ref}
        if self.graph is not None:
            out = self.graph.invoke(state)
        else:
            s = self.route(state)
            fn = {"explain": self.a_explain, "settlement": self.a_settlement, "indicators": self.a_indicators,
                  "similar": self.a_similar, "summary": self.a_summary, "network": self.a_network}.get(s["intent"], self.a_help)
            out = self.polish(fn(s))
        return {"answer": out.get("answer"), "intent": out.get("intent"), "claim": out.get("claim_ref"),
                "data": out.get("data", {}), "engine": "langgraph" if self.graph is not None else "python-dispatch"}
