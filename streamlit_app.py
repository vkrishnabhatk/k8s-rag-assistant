from __future__ import annotations

import json
import time
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="K8s RAG Assistant",
    page_icon="⎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .source-card {
            background: var(--background-color, #f8f9fa);
            border-left: 3px solid #326ce5;
            padding: 8px 14px;
            margin: 6px 0;
            border-radius: 0 6px 6px 0;
        }
        .source-card a {
            color: #326ce5;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95em;
        }
        .source-card a:hover { text-decoration: underline; }
        .source-score {
            color: #888;
            font-size: 0.82em;
            margin-left: 6px;
        }
        .k8s-header {
            display: flex;
            align-items: center;
            gap: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _render_response_metadata(msg: dict[str, Any]) -> None:
    sources: list[dict[str, Any]] = msg.get("sources", [])
    validation: dict[str, Any] = msg.get("validation", {})
    latency_ms: float | None = msg.get("latency_ms")

    if not sources and not validation:
        return

    guardrail = validation.get("guardrail_triggered") if validation else None

    meta_cols = st.columns([4, 1])
    with meta_cols[1]:
        if latency_ms is not None:
            st.caption(f"⏱ {latency_ms:.0f} ms")
        if validation:
            passed = validation.get("passed", True)
            conf = validation.get("confidence_score", 0.0)
            if guardrail == "general_knowledge":
                st.info("General knowledge", icon="💡")
            elif guardrail:
                st.warning(f"Guardrail: `{guardrail}`", icon="⚠️")
            elif passed:
                st.success(f"Confidence {conf:.0%}", icon="✅")
            else:
                st.error(f"Low confidence {conf:.0%}", icon="⚠️")

    if sources and guardrail != "general_knowledge":
        with st.expander(f"📚 {len(sources)} source{'s' if len(sources) != 1 else ''} retrieved"):
            for src in sources:
                score_pct = int(src.get("score", 0) * 100)
                title = src.get("title") or src.get("url", "")
                url = src.get("url", "#")
                st.markdown(
                    f'<div class="source-card">'
                    f'<a href="{url}" target="_blank">{title}</a>'
                    f'<span class="source-score">· {score_pct}% match</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _stream_query(api_url: str, query: str, top_k: int) -> dict[str, Any]:
    """
    Streams from /v1/query/stream, yields tokens into st.write_stream,
    and returns the final metadata dict (sources, validation, latency_ms).
    """
    collected_tokens: list[str] = []
    metadata: dict[str, Any] = {}
    event_type = "message"
    t0 = time.perf_counter()

    def _token_generator():
        nonlocal event_type, metadata
        with httpx.Client(timeout=60) as client:
            with client.stream(
                "POST",
                f"{api_url}/v1/query/stream",
                json={"query": query, "top_k": top_k},
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    line = raw_line.strip()
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        raw = line.split(":", 1)[1]
                        data = raw[1:] if raw.startswith(" ") else raw
                        if event_type == "done":
                            try:
                                metadata = json.loads(data)
                            except json.JSONDecodeError:
                                pass
                        else:
                            try:
                                token = json.loads(data)
                            except (json.JSONDecodeError, ValueError):
                                token = data
                            collected_tokens.append(token)
                            yield token
                    elif line == "":
                        event_type = "message"

    response_text = st.write_stream(_token_generator())
    metadata["latency_ms"] = (time.perf_counter() - t0) * 1000
    metadata["content"] = response_text or "".join(collected_tokens)
    return metadata


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⎈ K8s RAG Assistant")
    st.caption("Powered by Anthropic Claude + FAISS")
    st.divider()

    api_url = st.text_input("API base URL", value="http://localhost:8000")
    top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=5)

    st.divider()

    check_col, status_col = st.columns([1, 2])
    with check_col:
        check_clicked = st.button("Check API", use_container_width=True)
    with status_col:
        if check_clicked:
            try:
                r = httpx.get(f"{api_url}/ready", timeout=3)
                st.session_state["api_health"] = r.json().get("status", "unknown")
            except Exception:
                st.session_state["api_health"] = "unreachable"

        health = st.session_state.get("api_health", "unknown")
        if health == "ready":
            st.success("Ready", icon="✅")
        elif health == "unreachable":
            st.error("Unreachable", icon="🔴")
        elif health != "unknown":
            st.warning(health, icon="⚠️")
        else:
            st.caption("—")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    with st.expander("Example questions"):
        examples = [
            "What is a Kubernetes Deployment?",
            "How do I scale a StatefulSet?",
            "What are Network Policies used for?",
            "How does a Kubernetes Service route traffic?",
            "What is the difference between a ConfigMap and a Secret?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
                st.session_state["prefill"] = ex
                st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────

st.title("⎈ Kubernetes Documentation Assistant")
st.caption(
    "Ask anything about Kubernetes — concepts, configuration, workloads, networking, storage, and more."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages: list[dict[str, Any]] = st.session_state["messages"]

if not messages:
    st.info(
        "**Get started** — type a question below or pick one from the sidebar examples.",
        icon="💡",
    )

# Render conversation history
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_response_metadata(msg)

# Handle sidebar example prefill
prefill: str = st.session_state.pop("prefill", "") or ""

user_input: str = st.chat_input("Ask about Kubernetes…") or prefill

if user_input:
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            metadata = _stream_query(api_url, user_input, top_k)
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}", icon="🚨")
            metadata = {"content": "", "error": str(exc)}
        except Exception as exc:
            st.error(
                f"Could not reach the API at `{api_url}`. "
                f"Make sure `make serve` is running.\n\n`{exc}`",
                icon="🔌",
            )
            metadata = {"content": "", "error": str(exc)}

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": metadata.get("content", ""),
            "sources": metadata.get("sources", []),
            "validation": metadata.get("validation", {}),
            "latency_ms": metadata.get("latency_ms"),
        }
        _render_response_metadata(assistant_msg)
        messages.append(assistant_msg)
