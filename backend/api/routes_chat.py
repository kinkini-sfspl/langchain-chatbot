
from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..rag.chains import build_chain
from ..rag.retriever import get_retriever

# LangChain conversation history
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


# ---------------- Router ----------------
router = APIRouter(prefix="/api", tags=["chat"])


# ---------------- Models ----------------
class Query(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None, description="Conversation/session id; if omitted, uses 'default'"
    )
    provider: Optional[str] = Field(
        None, description="LLM provider: 'gemini' (default) or 'fireworks'"
    )


# ---------------- Abuse Filter ----------------
# Add/adjust terms for your context; keep it lowercase.
ABUSIVE_WORDS = {
    "idiot",
    "stupid",
    "nonsense",
    "fool",
    "shit",
    "fuck",
    "bastard",
    "moron",
    "asshole",
    "chutiya",
}


def contains_abuse(text: str) -> bool:
    t = text.lower()
    return any(bad in t for bad in ABUSIVE_WORDS)


# ---------------- Simple retriever debug ----------------
@router.post("/chat_debug")
def chat_debug(q: Query):
    """
    Returns the first ~300 chars of retrieved chunks for a given query.
    Useful to confirm embeddings + PGVector wiring.
    """
    try:
        docs = get_retriever().invoke(q.message)  # list[Document]
        return {"chunks": [d.page_content[:300] for d in docs]}
    except Exception as e:
        import traceback

        return {"error": str(e), "trace": traceback.format_exc()}


# ---------------- Chat with memory + RAG ----------------
_histories: Dict[str, InMemoryChatMessageHistory] = {}


def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) an in-memory history bucket per session id."""
    return _histories.setdefault(
        session_id or "default", InMemoryChatMessageHistory()
    )


def _history_from_cfg(cfg) -> InMemoryChatMessageHistory:
    """
    RunnableWithMessageHistory callback: pull session_id from config.
    Expects config={"configurable":{"session_id": "xyz"}}
    """
    sid = "default"
    if isinstance(cfg, dict):
        sid = (cfg.get("configurable") or {}).get("session_id", "default")
    return _get_history(sid)


def _pick_llm(provider: Optional[str]):
    """
    Lazy-import the chosen LLM so dependencies are optional.
    provider: 'gemini' (default) or 'fireworks'
    """
    prov = (provider or "gemini").lower()
    if prov == "fireworks":
        from ..services.llm_fireworks import get_llm as _get
    else:
        from ..services.llm_gemini import get_llm as _get
    return _get()


@router.post("/chat")
def chat(q: Query):
    """
    Main chat endpoint.
    - Abuse guard (short-circuits).
    - LLM provider selection (gemini / fireworks).
    - RAG chain with PGVector retriever.
    - Conversational memory via RunnableWithMessageHistory.
    """
    # 1) Abuse guard (short-circuit before touching the LLM)
    if contains_abuse(q.message):
        return {
            "answer": (
                "I’m here to help with MFI Business Document queries. "
                "I can’t continue when there’s abusive language—please rephrase your question respectfully."
            )
        }

    # 2) Build LLM (based on provider) + retriever + chain
    llm = _pick_llm(q.provider)
    chain = build_chain(llm, get_retriever())  # returns Runnable -> {"answer": str}

    # 3) Wrap with message history
    chat_chain = RunnableWithMessageHistory(
        chain,
        _history_from_cfg,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )

    # 4) Invoke with a stable session id (client decides; defaults to "default")
    cfg = {"configurable": {"session_id": q.session_id or "default"}}
    result = chat_chain.invoke({"question": q.message}, config=cfg)

    # 5) Normalize result
    if isinstance(result, dict) and "answer" in result:
        return {"answer": result["answer"]}
    return {"answer": str(result)}
