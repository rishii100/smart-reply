"""RAG-based email reply generator.

Takes a new incoming email, retrieves k similar past (incoming, reply) pairs
from the retrieval pool, and prompts an LLM to write a suggested reply grounded
in those exemplars.

Trade-off vs. fine-tuning:
  * RAG + few-shot is cheap, transparent, needs no training infra, adapts
    instantly when new data is added, and keeps the grounding explicit.
  * Fine-tuning would internalise the style better for large-scale production,
    but adds cost, training time, and a stale-model risk.  For a demo with ~100
    emails, RAG is the right call.
"""
from __future__ import annotations

from typing import Any

from . import config
from .llm import chat
from .retriever import TFIDFRetriever


_SYSTEM = """\
You are an AI email assistant that drafts professional reply emails.

Rules:
1. Address EVERY question and request in the incoming email.
2. Be specific — give concrete details, dates, next steps, not vague assurances.
3. Match the appropriate tone: warm for casual emails, precise for business.
4. Do NOT invent facts, commitments, or details not grounded in the email or context.
5. Sign off professionally. Do NOT add a subject line.
6. Keep the reply concise — aim for the length that fully answers the email,
   no more.

Below are examples of real emails and their replies from this organisation.
Use them to learn the house style, level of detail, and formatting conventions.
"""


def _build_prompt(
    incoming_email: dict[str, Any],
    exemplars: list[tuple[dict[str, Any], float]],
) -> list[dict[str, str]]:
    """Build the chat messages for the LLM."""
    # System message with style guidance.
    system = _SYSTEM.strip()

    # Exemplar section.
    exemplar_text = []
    for i, (ex, score) in enumerate(exemplars, 1):
        exemplar_text.append(
            f"--- Example {i} (similarity {score:.2f}) ---\n"
            f"Subject: {ex.get('subject', '(no subject)')}\n"
            f"Incoming email:\n{ex['incoming']}\n\n"
            f"Reply sent:\n{ex['reply']}"
        )
    if exemplar_text:
        system += "\n\n" + "\n\n".join(exemplar_text)

    # User message: the new email.
    subject = incoming_email.get("subject", "(no subject)")
    user_msg = (
        f"Now write a reply to this NEW incoming email.\n\n"
        f"Subject: {subject}\n"
        f"Incoming email:\n{incoming_email['incoming']}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def generate_reply(
    incoming_email: dict[str, Any],
    retriever: TFIDFRetriever,
    *,
    k: int | None = None,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Generate a suggested reply for an incoming email.

    Returns a dict with::

        {
            "email_id": ...,
            "incoming": ...,
            "gold_reply": ...,
            "generated_reply": ...,
            "retrieved_ids": [...],
            "retrieved_scores": [...],
        }
    """
    k = k or config.RETRIEVE_K
    exclude_ids = exclude_ids or set()

    # Retrieve similar past emails (excluding the email itself).
    query = f"{incoming_email.get('subject', '')} {incoming_email['incoming']}"
    exemplars = retriever.retrieve(query, k=k, exclude_ids=exclude_ids | {incoming_email["id"]})

    messages = _build_prompt(incoming_email, exemplars)
    reply_text = chat(
        messages,
        model=config.GEN_MODEL,
        temperature=config.GEN_TEMPERATURE,
        max_tokens=1024,
    )

    return {
        "email_id": incoming_email["id"],
        "category": incoming_email.get("category", "unknown"),
        "subject": incoming_email.get("subject", ""),
        "incoming": incoming_email["incoming"],
        "gold_reply": incoming_email.get("reply", ""),
        "generated_reply": reply_text.strip(),
        "retrieved_ids": [e["id"] for e, _ in exemplars],
        "retrieved_scores": [round(s, 4) for _, s in exemplars],
    }
