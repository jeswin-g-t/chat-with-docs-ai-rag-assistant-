import streamlit as st
from rag_pipeline import (
    add_files_to_vectorstore,
    build_self_checking_graph,
    run_self_checking_query,
    extract_page_number,
    explain_page_stream,
    extract_section_number,
    explain_section_stream,
    is_overview_question,
    answer_overview_stream,
    stream_text,
)

st.set_page_config(page_title="Chat With Docs", page_icon="💬", layout="wide")

# --- Light custom styling (subtle color accents) ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #161a2b;
        border-right: 1px solid #2a2f45;
    }
    h1 {
        color: #8ab4ff !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #8ab4ff !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #1a2036;
        border-radius: 10px;
        padding: 6px;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #14201c;
        border-radius: 10px;
        padding: 6px;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #7d84a8 !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session state setup ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: file upload / source management ---
with st.sidebar:
    st.header("📂 Uploaded Files")

    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="uploader"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.sources]
        if new_files:
            try:
                with st.spinner(f"Processing {len(new_files)} file(s)..."):
                    st.session_state.vectorstore, added, new_chunks = add_files_to_vectorstore(
                        new_files, st.session_state.vectorstore
                    )
                    st.session_state.sources.extend(added)
                    st.session_state.all_chunks.extend(new_chunks)
                    st.session_state.rag_graph = build_self_checking_graph(st.session_state.vectorstore)
                st.success(f"Added: {', '.join(added)}")
            except Exception as e:
                st.error(f"Something went wrong while processing: {e}")

    st.divider()

    selected_sources = []
    if st.session_state.sources:
        st.caption(f"{len(st.session_state.sources)} file(s) loaded")
        st.markdown("**Ask about:**")
        select_all = st.checkbox("All files", value=True, key="select_all_sources")

        if select_all:
            selected_sources = st.session_state.sources.copy()
            for name in st.session_state.sources:
                st.markdown(f"📄 {name}")
        else:
            for name in st.session_state.sources:
                if st.checkbox(f"📄 {name}", value=False, key=f"src_{name}"):
                    selected_sources.append(name)
            if not selected_sources:
                st.caption("⚠️ Select at least one file, otherwise all files will be searched.")
    else:
        st.caption("No files uploaded yet.")
        if uploaded_files:
            st.warning(
                "Your session was reset (likely due to inactivity/sleep). "
                "Files shown above will reprocess automatically — just wait a moment, "
                "or remove and re-upload if this persists."
            )

# --- Main area ---
st.title("💬 Chat With Docs")
st.caption("Upload a PDF and ask questions in detail.")

if st.session_state.rag_graph:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            page_num = extract_page_number(query)
            section_match = extract_section_number(query)

            if section_match is not None:
                # "explain program 4", "4th question", etc. — searches for the actual
                # heading in the document instead of relying on semantic similarity
                keyword, number = section_match
                result = explain_section_stream(
                    st.session_state.all_chunks, keyword, number,
                    selected_sources=selected_sources or None
                )
                if isinstance(result, str):
                    st.markdown(result)
                    answer = result
                else:
                    answer = st.write_stream(result)

            elif page_num is not None:
                # Page-specific — single LLM call, streams token by token
                result = explain_page_stream(
                    st.session_state.all_chunks, page_num,
                    selected_sources=selected_sources or None
                )
                if isinstance(result, str):
                    st.markdown(result)
                    answer = result
                else:
                    answer = st.write_stream(result)

            elif is_overview_question(query):
                # Overview/summary — streams (map step runs silently first on long docs)
                with st.spinner("Reading through the document..."):
                    result = answer_overview_stream(
                        st.session_state.all_chunks, query,
                        selected_sources=selected_sources or None
                    )
                if isinstance(result, str):
                    st.markdown(result)
                    answer = result
                else:
                    answer = st.write_stream(result)

            else:
                # Normal question — graph must fully generate + grade before showing anything,
                # so it's validated first, then displayed with a smooth typing effect.
                with st.spinner("Thinking..."):
                    answer, context_docs, was_retried = run_self_checking_query(
                        st.session_state.rag_graph, query,
                        selected_sources=selected_sources or None,
                        all_sources=st.session_state.sources
                    )
                st.write_stream(stream_text(answer))
                if was_retried:
                    st.caption("↻ Re-checked and refined this answer for accuracy.")

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a document in the sidebar to get started.")