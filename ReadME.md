# Chat With Docs — AI RAG Assistant
**[Try it live](https://jntmaetdbhelcx8er3krax.streamlit.app/)**

->upload your documents, ask questions in plain English, and get answers grounded in your own sources — built entirely on free tools.

Overview:
Chat With Docs is a Retrieval-Augmented Generation (RAG) assistant that lets you upload multiple documents into a single notebook and ask questions across all of them at once — no cloud storage, no per-token cost, no OpenAI key required. Every answer is retrieved from your actual uploaded content and structured into clear, readable sentences rather than a raw dump of matched text.

### Features:
- 📁 **Multi-Source Notebooks** — upload several documents into one notebook and query across all of them together, not just one file at a time
- 💬 **Natural-Language Q&A** — ask questions conversationally; answers come back as structured, readable sentences rather than raw retrieved chunks
- 🎯 **Grounded Answers** — every response is retrieved from the uploaded documents, reducing hallucination compared to a plain chatbot
- ⚡ **Fast Inference** — powered by Groq's LPU-backed LLM API for near-instant responses
- 🖥️ **Simple Web UI** — clean Streamlit interface, no local setup needed to try it


### How It Works
1. **Documents are uploaded and split into chunks.**
2. **Each chunk is embedded** using a free Hugging Face sentence-transformer model.
3. **Embeddings are stored** in a local vector database (Store in vector DB (Chroma)).
4. **On each question**, the most relevant chunks are retrieved and passed to the Groq-hosted LLM.
5. **The LLM composes** a grounded, readable answer from only that retrieved context.


Installation:
# Clone the repository
git clone https://github.com/<your-username>/chat-with-docs.git
cd chat-with-docs

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt


Environment Variables:
Create a .env file in the root directory:
GEMINI_API_KEY=your_GEMINI_api_key

Run Locally:
streamlit run app.py
