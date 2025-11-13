# app.py
import os
import uuid
import time
import requests
import streamlit as st

# ---------- Config ----------
DEFAULT_API = os.getenv("RAG_API_BASE", "http://127.0.0.1:8000")
CHAT_EP = "/api/chat"
DEBUG_EP = "/api/chat_debug"   # optional: to peek retrieval chunks

# Exact refusal text coming from backend (chains.py SCOPE_MSG)
REFUSAL_TEXT = "That topic is outside my scope. I can help with MFI Business Document related information."

st.set_page_config(page_title="RAG Chat UI", page_icon="💬", layout="centered")

# ---------- Sidebar ----------
st.sidebar.title("RAG Chat — Client")

api_base = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_API,
    help="Your FastAPI base URL",
)

# NEW: choose the backend LLM provider for this session
provider = st.sidebar.radio(
    "LLM Provider",
    options=["gemini", "fireworks"],   # 'fireworks' = FireworksAI (e.g., Mixtral)
    index=0,
    help="Pick which LLM the server should use for responses.",
)

# Session id (used by server for history bucketing)
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web-{uuid.uuid4().hex[:8]}"
sid = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
st.session_state.session_id = sid

show_debug = st.sidebar.checkbox("Show retrieved chunks (chat_debug)", value=False)
if st.sidebar.button("New Session / Clear chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: keep the same Session ID to preserve conversation history on the server (if enabled)."
)

# ---------- Header ----------
st.title("💬 RAG Chat")
st.caption("FastAPI + LangChain — Streamlit client")

# ---------- Message store ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        # Render refusal as a callout in history too
        if role == "assistant" and isinstance(content, str) and content.strip().startswith(REFUSAL_TEXT):
            st.warning(f"**Out of scope**\n\n{content}")
        else:
            st.markdown(content)

# ---------- Helpers ----------
def post_json(url: str, payload: dict, timeout=240):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)

# ---------- Input ----------
prompt = st.chat_input("Ask something…")
if prompt:
    # append user msg
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # call main /api/chat
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_thinking…_")

        t0 = time.time()
        data = {
            "message": prompt,
            "session_id": st.session_state.session_id,
            "provider": provider,  # <-- send chosen LLM provider to backend
        }
        resp, err = post_json(api_base + CHAT_EP, data)

        if err:
            placeholder.error(f"Request failed: {err}")
        else:
            answer = resp.get("answer") if isinstance(resp, dict) else resp
            if isinstance(answer, dict):
                answer = str(answer)

            # Render refusal verbatim as a highlighted callout
            if isinstance(answer, str) and answer.strip().startswith(REFUSAL_TEXT):
                placeholder.empty()
                st.warning(f"**Out of scope**\n\n{answer}")
            else:
                placeholder.markdown(answer)

            # store assistant message back into session history
            st.session_state.messages.append(("assistant", answer))

            # optionally call /api/chat_debug to show retrieved chunks
            if show_debug:
                dbg, derr = post_json(api_base + DEBUG_EP, {"message": prompt})
                with st.expander("🔎 Retrieved chunks (chat_debug)"):
                    if derr:
                        st.error(derr)
                    else:
                        chunks = dbg.get("chunks") or []
                        if not chunks:
                            st.write("_No chunks returned._")
                        else:
                            for i, c in enumerate(chunks, 1):
                                st.markdown(f"**Chunk {i}**")
                                st.code(c)

        st.caption(f"Latency: {time.time()-t0:.2f}s")
