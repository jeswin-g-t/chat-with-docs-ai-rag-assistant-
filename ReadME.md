# Chat With Docs — AI RAG Assistant

A multi-document RAG assistant with self-verifying retrieval, built with LangChain, LangGraph, and Streamlit.

## Features
- Multi-file upload with per-file source selection
- Self-checking answer loop (LangGraph): generates, grades for grounding, retries with a rewritten query if weak
- Map-reduce summarization for long documents
- Page-specific and section-specific explanations
- Streaming responses
- 100% free stack — Groq (LLM) + HuggingFace (embeddings)

## Tech Stack
LangChain • LangGraph • Groq (Llama 3.3) • HuggingFace Embeddings • ChromaDB • Streamlit

## Run Locally
\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

Add your free Groq API key to a `.env` file:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`
