"""Optional LLM wording layer. The copilot is grounded in structured facts; the LLM may only
rephrase them. Disabled by default (LLM_PROVIDER=none) - everything works without any API key."""
from app.config import settings


def polish(question, facts):
    if settings.llm_provider != "anthropic" or not settings.anthropic_api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.llm_model, max_tokens=600,
            system=("You are an insurance claims copilot. Answer the officer's question using ONLY the facts "
                    "provided. Do not invent numbers, names or reasons. Be concise and structured."),
            messages=[{"role": "user", "content": f"Question: {question}\n\nFacts:\n{facts}"}])
        return "".join(b.text for b in msg.content if getattr(b, "text", None))
    except Exception:
        return None
