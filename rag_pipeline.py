import os
import re
import tempfile
from typing import TypedDict, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

MAX_RETRIES = 1  # how many times the graph is allowed to rewrite + retry retrieval


def get_llm():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

    if not key:
        raise ValueError(
            "Missing Gemini API key. Add GEMINI_API_KEY or GOOGLE_API_KEY to the .env file or Streamlit secrets."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=key
    )


# ---------------------------------------------------------------------------
# File loading / chunking
# ---------------------------------------------------------------------------

def load_and_chunk_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = uploaded_file.name
        if "page" in chunk.metadata:
            chunk.metadata["page_display"] = chunk.metadata["page"] + 1

    os.unlink(tmp_path)
    return chunks


def add_files_to_vectorstore(uploaded_files, vectorstore=None):
    all_chunks = []
    processed_names = []

    for f in uploaded_files:
        chunks = load_and_chunk_file(f)
        all_chunks.extend(chunks)
        processed_names.append(f.name)

    if vectorstore is None:
        vectorstore = Chroma.from_documents(documents=all_chunks, embedding=_embeddings)
    else:
        vectorstore.add_documents(all_chunks)

    return vectorstore, processed_names, all_chunks


# ---------------------------------------------------------------------------
# Shared detailed system prompt
# ---------------------------------------------------------------------------

# Exact refusal text used when the question is unrelated to the uploaded document(s).
# Kept as a constant so grade_node can reliably detect it (see below).
OFF_TOPIC_REFUSAL = "I'm sorry, but the provided document does not contain information to answer that question."

DETAILED_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that explains uploaded documents thoroughly and clearly, "
    "the way a top-tier assistant like ChatGPT or Claude would.\n\n"
    "CRITICAL RULE — read this first:\n"
    f"If the question is NOT related to the provided context (i.e. the context does not address "
    f"what is being asked, even loosely), respond with EXACTLY this sentence and NOTHING else — "
    f"no summary, no 'however here is what the document covers', no extra information:\n"
    f'"{OFF_TOPIC_REFUSAL}"\n'
    "Do not pad this response. Do not describe what the document is about instead. Just that one sentence.\n\n"
    "If the question IS related to the context, follow these guidelines instead:\n"
    "- Give a DETAILED, well-organized answer — not a short summary.\n"
    "- Use markdown formatting: headers (##), bold for key terms, and bullet points for lists.\n"
    "- If the content has multiple topics/sections, break them into clearly labeled sections.\n"
    "- Explain WHAT each part covers, not just that it exists — add specifics found in the context.\n"
    "- End with a short overall takeaway if relevant.\n"
    "- Base your answer ONLY on the provided context. If something isn't in the context, say so — never invent details.\n\n"
    "Context:\n{context}"
)


# ---------------------------------------------------------------------------
# Helpers to build a Chroma metadata filter from selected source filenames
# ---------------------------------------------------------------------------

def _build_source_filter(selected_sources: Optional[List[str]], all_sources: List[str]):
    """Returns a Chroma-compatible filter dict, or None if no filtering is needed
    (i.e. nothing selected, or everything is selected)."""
    if not selected_sources:
        return None
    if set(selected_sources) == set(all_sources):
        return None  # everything selected = no filter needed
    if len(selected_sources) == 1:
        return {"source": selected_sources[0]}
    return {"source": {"$in": selected_sources}}


# ---------------------------------------------------------------------------
# Overview / "what is this about" handling — with map-reduce for long docs
# ---------------------------------------------------------------------------

OVERVIEW_PATTERNS = [
    r"what is this (pdf|document|file) (mainly )?(about|talking about)",
    r"what does this (pdf|document|file) (talk|cover|discuss)",
    r"summarize this",
    r"give (me )?a summary",
    r"main topics",
    r"what.?s (this|it) about",
    r"overview of",
]


def is_overview_question(query: str) -> bool:
    q = query.lower()
    return any(re.search(pattern, q) for pattern in OVERVIEW_PATTERNS)


# How many characters of source text to pack into each map-step batch.
# Kept comfortably under Llama 3.3's context window since each batch also
# needs room for the system prompt and the model's own generation.
MAP_BATCH_CHAR_LIMIT = 12000

# Below this total size, skip map-reduce entirely and just answer directly —
# no need to pay for multiple LLM calls on a short document.
MAP_REDUCE_THRESHOLD = 15000


def _group_chunks_into_batches(chunks, batch_char_limit=MAP_BATCH_CHAR_LIMIT):
    """Groups chunks into text batches under batch_char_limit characters each,
    without splitting any single chunk across two batches."""
    batches = []
    current_batch = ""

    for c in chunks:
        piece = f"\n\n[{c.metadata.get('source','doc')} - page {c.metadata.get('page_display','?')}]\n{c.page_content}"
        if current_batch and len(current_batch) + len(piece) > batch_char_limit:
            batches.append(current_batch)
            current_batch = piece
        else:
            current_batch += piece

    if current_batch:
        batches.append(current_batch)

    return batches


def _map_summarize_batch(batch_text: str, query: str) -> str:
    """Map step: summarizes ONE batch of the document, focused on what's relevant to the query."""
    llm = get_llm()
    prompt = (
        "You are summarizing ONE PART of a larger document. Extract and summarize the key points "
        "from this section in detail — don't skip specifics like names, numbers, features, or steps. "
        f'Keep in mind the reader ultimately wants to know: "{query}"\n\n'
        f"Section content:\n{batch_text}\n\n"
        "Write a thorough summary of this section (not a one-liner)."
    )
    result = llm.invoke(prompt)
    return result.content


def _reduce_summaries(batch_summaries: List[str], query: str) -> str:
    """Reduce step: combines all batch summaries into one final, well-structured answer."""
    combined = "\n\n---\n\n".join(
        f"[Part {i + 1} summary]\n{s}" for i, s in enumerate(batch_summaries)
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", DETAILED_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    result = document_chain.invoke({
        "input": query,
        "context": [Document(page_content=combined, metadata={})]
    })
    return result


def answer_overview(all_chunks, query: str, selected_sources: Optional[List[str]] = None):
    """Non-streaming version — kept for compatibility, use answer_overview_stream in the UI."""
    result = answer_overview_stream(all_chunks, query, selected_sources)
    if isinstance(result, str):
        return result
    return "".join(result)


def answer_overview_stream(all_chunks, query: str, selected_sources: Optional[List[str]] = None):
    """
    Streaming version of answer_overview.

    Short documents: streams the single LLM call directly, token by token.
    Long documents: the map step (per-batch summarization) still runs invisibly
    first, since each batch summary must be complete before the final reduce
    step can combine them — but the final reduce step (the part the user
    actually reads) streams in live.
    """
    chunks_to_use = all_chunks
    if selected_sources:
        chunks_to_use = [c for c in all_chunks if c.metadata.get("source") in selected_sources]
        if not chunks_to_use:
            chunks_to_use = all_chunks  # safety fallback

    total_chars = sum(len(c.page_content) for c in chunks_to_use)

    # --- Short document: stream the direct answer ---
    if total_chars <= MAP_REDUCE_THRESHOLD:
        full_text = "\n\n".join(
            f"[{c.metadata.get('source','doc')} - page {c.metadata.get('page_display','?')}]\n{c.page_content}"
            for c in chunks_to_use
        )
        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", DETAILED_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        document_chain = create_stuff_documents_chain(llm, prompt)

        def _gen_short():
            for chunk in document_chain.stream({
                "input": query,
                "context": [Document(page_content=full_text, metadata={})]
            }):
                if chunk:
                    yield chunk

        return _gen_short()

    # --- Long document: map step runs fully first (not shown), reduce step streams ---
    batches = _group_chunks_into_batches(chunks_to_use)
    batch_summaries = [_map_summarize_batch(batch, query) for batch in batches]

    combined = "\n\n---\n\n".join(
        f"[Part {i + 1} summary]\n{s}" for i, s in enumerate(batch_summaries)
    )

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", DETAILED_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    def _gen_long():
        for chunk in document_chain.stream({
            "input": query,
            "context": [Document(page_content=combined, metadata={})]
        }):
            if chunk:
                yield chunk

    return _gen_long()


# ---------------------------------------------------------------------------
# Page-specific explanation
# ---------------------------------------------------------------------------

def extract_page_number(query: str):
    match = re.search(r"page\s*(?:no\.?|number)?\s*(\d+)", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# Keywords that commonly label numbered sections in academic / lab / assignment PDFs.
SECTION_KEYWORDS = r"(program|question|exercise|problem|experiment|chapter|unit|module|assignment|task)s?"


def extract_section_number(query: str):
    """
    Detects requests like 'explain program 4', 'question no. 4', 'the 4th exercise',
    '4th program'. Returns (keyword, number) e.g. ('program', 4), or None if no match.
    """
    # Pattern A: "program 4", "question no. 4", "exercise number 4"
    match = re.search(
        rf"{SECTION_KEYWORDS}\s*(?:no\.?|number)?\s*(\d+)",
        query, re.IGNORECASE
    )
    if match:
        return match.group(1).lower(), int(match.group(2))

    # Pattern B: "4th program", "4th question"
    match = re.search(
        rf"(\d+)\s*(?:st|nd|rd|th)\s*{SECTION_KEYWORDS}",
        query, re.IGNORECASE
    )
    if match:
        return match.group(2).lower(), int(match.group(1))

    return None


def _find_heading_match(full_text: str, keyword: str, number: int):
    """
    Tries several patterns, from strict to loose, to find a heading like 'Program 2'
    in raw document text. Handles zero-padded numbers ('Program 02'), various
    separators (-, –, —, :, ., #), and number-before-word styles ('2. Program').
    Returns the match object, or None.
    """
    kw = re.escape(keyword)

    patterns = [
        # "Program 2" / "Program No. 2" / "Program-2" / "Program: 2" / "Program #2"
        rf"\b{kw}s?\b[\s:.\-–—#]{{0,10}}(?:no\.?|number)?[\s:.\-–—#]{{0,5}}0*{number}\b",
        # "2. Program" / "2) Program" / "2 - Program"
        rf"\b0*{number}[\s:.\-–—)]{{1,5}}\s*{kw}s?\b",
        # Very loose fallback: keyword and number within a short window of each other,
        # any characters in between (handles unusual PDF text extraction artifacts)
        rf"\b{kw}s?\b[\s\S]{{0,20}}?0*{number}\b",
    ]

    for pat in patterns:
        match = re.search(pat, full_text, re.IGNORECASE)
        if match:
            return match

    return None


def _find_any_heading(full_text: str, keyword: str, start_pos: int):
    """Finds the next occurrence of ANY numbered heading of the same keyword, used to
    bound where the current section ends."""
    kw = re.escape(keyword)
    pattern = rf"\b{kw}s?\b[\s:.\-–—#]{{0,10}}(?:no\.?|number)?[\s:.\-–—#]{{0,5}}(\d+)\b"
    return re.search(pattern, full_text[start_pos:], re.IGNORECASE)


def explain_section_stream(all_chunks, keyword: str, number: int, selected_sources: Optional[List[str]] = None):
    """
    Streaming version: finds a numbered section (e.g. 'Program 4', 'Question 4') by
    searching the actual document text for that heading, rather than relying purely on
    semantic similarity — which avoids confusing adjacent, similarly-worded sections
    (e.g. Program 4 vs Program 5).
    Falls back to a looser keyword+number proximity search if no clean heading is found.
    Returns either a plain string (not found) or a generator of text chunks.
    """
    chunks_to_search = all_chunks
    if selected_sources:
        filtered = [c for c in all_chunks if c.metadata.get("source") in selected_sources]
        if filtered:
            chunks_to_search = filtered

    by_source = {}
    order = []
    for c in chunks_to_search:
        src = c.metadata.get("source", "doc")
        if src not in by_source:
            by_source[src] = []
            order.append(src)
        by_source[src].append(c)

    for src in order:
        src_chunks = by_source[src]
        full_text = "\n\n".join(c.page_content for c in src_chunks)

        match = _find_heading_match(full_text, keyword, number)
        if not match:
            continue

        start = match.start()
        next_match = _find_any_heading(full_text, keyword, match.end())
        end = match.end() + next_match.start() if next_match else min(len(full_text), start + 4000)

        section_text = full_text[start:end].strip()

        llm = get_llm()
        prompt = (
            f"Explain the following '{keyword.title()} {number}' section from the document '{src}' "
            f"in DETAIL, using markdown headers and bullet points where useful. "
            f"Cover every point thoroughly — code, logic, purpose, everything present:\n\n"
            f"{section_text}"
        )

        def _gen():
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content

        return _gen()

    # --- Fallback: no clean heading found anywhere — try semantic retrieval instead
    # of giving up, using the keyword+number as a search query.
    query_hint = f"{keyword} {number}"
    candidates = [
        c for c in chunks_to_search
        if keyword.lower() in c.page_content.lower() and str(number) in c.page_content
    ]

    if not candidates:
        return (
            f"I couldn't find '{keyword.title()} {number}' in the selected document(s). "
            f"It might be labeled differently — try checking the exact heading text in the PDF."
        )

    combined_text = "\n\n".join(c.page_content for c in candidates[:3])
    llm = get_llm()
    prompt = (
        f"The user asked about '{keyword.title()} {number}'. I couldn't find a clean section "
        f"heading, but here is the most relevant content found that mentions both '{keyword}' "
        f"and the number {number}. Explain it in DETAIL if it appears to be the right section, "
        f"using markdown headers and bullet points:\n\n{combined_text}"
    )

    def _gen_fallback():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return _gen_fallback()


def explain_page(all_chunks, page_number: int, selected_sources: Optional[List[str]] = None):
    """Non-streaming version — kept for compatibility, use explain_page_stream in the UI."""
    text_or_gen = explain_page_stream(all_chunks, page_number, selected_sources)
    if isinstance(text_or_gen, str):
        return text_or_gen
    return "".join(text_or_gen)


def explain_page_stream(all_chunks, page_number: int, selected_sources: Optional[List[str]] = None):
    """
    Streaming version: returns either a plain string (for the 'page not found' case,
    nothing to stream) or a generator yielding text chunks as the LLM produces them.
    """
    chunks_to_search = all_chunks
    if selected_sources:
        filtered = [c for c in all_chunks if c.metadata.get("source") in selected_sources]
        if filtered:
            chunks_to_search = filtered

    matching_chunks = [c for c in chunks_to_search if c.metadata.get("page_display") == page_number]

    if not matching_chunks:
        return f"I couldn't find page {page_number} in the selected document(s). Please check the page number and try again."

    combined_text = "\n\n".join(c.page_content for c in matching_chunks)
    source_name = matching_chunks[0].metadata.get("source", "the document")

    llm = get_llm()
    prompt = (
        f"Explain the following content from page {page_number} of '{source_name}' "
        f"in DETAIL, using markdown headers and bullet points where useful. "
        f"Cover every point thoroughly, don't just summarize in one line:\n\n"
        f"{combined_text}"
    )

    def _gen():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return _gen()


# ---------------------------------------------------------------------------
# LangGraph self-checking answer loop
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    original_question: str
    question: str                 # current (possibly rewritten) query used for retrieval
    context: List[Document]
    answer: str
    grounded: bool
    retries: int
    source_filter: Optional[dict]


def _make_retrieve_node(vectorstore):
    def retrieve_node(state: GraphState) -> GraphState:
        search_kwargs = {"k": 8}
        if state.get("source_filter"):
            search_kwargs["filter"] = state["source_filter"]
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        docs = retriever.invoke(state["question"])
        state["context"] = docs
        return state
    return retrieve_node


def generate_node(state: GraphState) -> GraphState:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", DETAILED_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    answer = document_chain.invoke({
        "input": state["original_question"],
        "context": state["context"]
    })
    state["answer"] = answer
    return state


def grade_node(state: GraphState) -> GraphState:
    """Asks the LLM to judge whether the answer is actually supported by the retrieved context.
    An honest 'I don't know' refusal is treated as automatically correct — it should never be
    retried, since there's nothing to retry into."""

    # If the model correctly declined (off-topic question), that's the right answer — don't retry.
    if OFF_TOPIC_REFUSAL.strip().lower() in state["answer"].strip().lower():
        state["grounded"] = True
        return state

    llm = get_llm()

    context_text = "\n\n".join(d.page_content for d in state["context"])[:6000]

    grading_prompt = (
        "You are a strict grader. Given the CONTEXT and the ANSWER below, decide if the answer "
        "is genuinely supported by the context (not guessed, not generic, not hallucinated).\n"
        "Reply with exactly one word: YES or NO.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"ANSWER:\n{state['answer']}\n\n"
        "Is the answer well-supported by the context? Reply YES or NO only."
    )

    result = llm.invoke(grading_prompt)
    verdict = result.content.strip().upper()
    state["grounded"] = verdict.startswith("Y")
    return state


def rewrite_node(state: GraphState) -> GraphState:
    """Rewrites the query to try to retrieve better-matching chunks on the next pass."""
    llm = get_llm()
    rewrite_prompt = (
        "The following question did not retrieve strongly relevant context from a document search. "
        "Rewrite it as a clearer, more specific search query that might retrieve better matching passages. "
        "Reply with ONLY the rewritten query, nothing else.\n\n"
        f"Original question: {state['original_question']}"
    )
    result = llm.invoke(rewrite_prompt)
    state["question"] = result.content.strip()
    state["retries"] += 1
    return state


def _route_after_grade(state: GraphState) -> str:
    if state["grounded"] or state["retries"] >= MAX_RETRIES:
        return "end"
    return "retry"


def build_self_checking_graph(vectorstore):
    """Builds and compiles the LangGraph: retrieve -> generate -> grade -> (retry or end)."""
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", _make_retrieve_node(vectorstore))
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {"end": END, "retry": "rewrite"})
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


def run_self_checking_query(compiled_graph, question: str, selected_sources: Optional[List[str]] = None,
                             all_sources: Optional[List[str]] = None):
    """Runs the graph for a single question, returns (answer, context_docs, was_retried)."""
    source_filter = _build_source_filter(selected_sources, all_sources or [])

    initial_state: GraphState = {
        "original_question": question,
        "question": question,
        "context": [],
        "answer": "",
        "grounded": False,
        "retries": 0,
        "source_filter": source_filter,
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state["answer"], final_state["context"], final_state["retries"] > 0


def stream_text(text: str, chunk_size: int = 4):
    """
    Takes an already-complete answer string and yields it in small pieces,
    to display it with a live typing effect (e.g. via st.write_stream) even
    though the text itself was generated all at once. Used for the
    self-checking graph's answer, since that answer must be fully generated
    and graded BEFORE it's safe to show the user anything — so true
    token-by-token streaming from the LLM isn't possible there without risking
    showing an answer that then gets discarded as ungrounded.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]