"""
Web UI (Streamlit):  .venv/bin/streamlit run app.py
Upload PDFs in the sidebar, ask a question, watch the 3 agents work.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import rag
from graph import ask
from main import DEMO_CASES

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent RAG: VoltWay Support Desk")
st.caption("Researcher (searches PDFs) → Analyst (calculates) → Writer (replies & reviews)")

with st.sidebar:
    st.header("📄 Upload PDFs")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if files and st.button("Upload & index", type="primary"):
        for f in files:
            tmp = Path(tempfile.mkdtemp()) / f.name
            tmp.write_bytes(f.getvalue())
            with st.spinner(f"Indexing {f.name}..."):
                n = rag.ingest_pdf(tmp)
            st.success(f"{f.name}: {n} chunks")
    if st.button("Load VoltWay sample PDFs"):
        from make_sample_pdfs import build_all
        for p in build_all():
            rag.ingest_pdf(p)
        st.success("Sample PDFs indexed")

    st.header("Knowledge base")
    docs = rag.list_documents()
    for name, n in docs.items():
        st.write(f"• {name} ({n} chunks)")
    if not docs:
        st.info("Empty: upload a PDF")

example = st.selectbox("Example questions", ["(write your own)"] + DEMO_CASES)
question = st.text_area("Customer question", value="" if example.startswith("(") else example, height=120)

if st.button("Ask the agents", type="primary", disabled=not (question and docs)):
    labels = {"researcher": "🔎 Researcher", "analyst": "🧮 Analyst", "writer": "✍️ Writer"}
    status = st.status("Agents working...", expanded=True)

    def on_event(e: dict) -> None:
        if e.get("status") == "start":
            status.write(f"**{labels[e['agent']]}** is working...")
            return
        if e.get("tools"):
            status.code("\n".join(e["tools"]), language=None)
        status.markdown(e["output"])
        status.divider()

    state = ask(question, on_event=on_event)
    status.update(label="Done", state="complete", expanded=False)
    st.subheader("✅ Final answer")
    st.markdown(state["answer"])
