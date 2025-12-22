# Telecom Research Papers Q&A RAG System

A modern **Retrieval-Augmented Generation (RAG)** application that enables question-answering over a collection of telecom-related research papers stored as PDFs.

The system:
- Ingests PDFs → chunks → embeds with Google Gemini embeddings
- Stores vectors in **Pinecone** (serverless)
- Retrieves relevant chunks and answers questions using **Gemini 1.5 Flash** (fast & cost-effective)
- Provides a clean **Streamlit** chat interface with source citations

Built with best practices: LangChain Expression Language (LCEL), proper metadata handling, relevance thresholding, and streaming responses.

## Features

- PDF ingestion pipeline with progress tracking
- Smart chunking with overlap for better context retention
- Google `embedding-001` embeddings (768-dim)
- Pinecone serverless vector database
- Similarity search with score threshold to avoid irrelevant results
- Conversational chat history support (last 4 messages)
- Streaming responses for better UX
- Source citation display (paper name + page number)

## Project Structure

```
.
├── ingest_papers.py          # Ingestion script: PDFs → Pinecone
├── rag_qa.py                 # Core RAG logic (retriever, chain, prompt)
├── app.py                    # Streamlit frontend
├── .env.example              # Example environment variables
└── papers/                   # Directory for your PDF research papers
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/telecom-rag-qa.git
cd telecom-rag-qa
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

(If no `requirements.txt` exists yet, create one with:)

```txt
streamlit
langchain
langchain-community
langchain-pinecone
langchain-google-genai
pinecone-client
python-dotenv
PyPDF2
tqdm
```

### 4. Set up environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_ai_studio_or_vertex_key
PINECONE_INDEX_NAME=telecom-papers
PINECONE_ENVIRONMENT=us-east-1  # or your preferred region
```

> Get keys from:
> - [Pinecone](https://app.pinecone.io/)
> - [Google AI Studio](https://aistudio.google.com/app/apikey)

## Usage

### Step 1: Ingest Your Research Papers

Place all your telecom research PDFs in a folder (e.g., `./papers/`).

Run the ingestion script:

```bash
python ingest_papers.py ./papers
```

This will:
- Load and parse all PDFs
- Chunk text intelligently
- Generate embeddings
- Upsert into Pinecone (creates index if needed)

You’ll see progress bars and logs. Once complete, your knowledge base is ready.

### Step 2: Launch the Q&A Chat App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

Ask questions like:
- "What are the key findings on 6G channel modeling?"
- "Compare beamforming techniques in mmWave papers."
- "Summarize latency improvements in recent O-RAN papers."

Answers will include **source citations** at the bottom.

## Example Questions

- What challenges are mentioned regarding URLLC in 5G NR?
- How do the papers discuss massive MIMO implementations?
- Explain the differences between split options in O-RAN architecture.

## Customization Tips

- Change `TOP_K_RESULTS` or `RELEVANCE_THRESHOLD` in `rag_qa.py` for stricter/looser retrieval
- Adjust `CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest_papers.py` for different granularity
- Use a different Gemini model (e.g., `gemini-1.5-pro`) by editing `initialize_llm()`

## Notes

- The system uses **relevance thresholding** (`0.78`) to avoid hallucinated answers from weak matches.
- Only the **last 4 messages** are sent as history to stay within token limits.
- Full text chunks are stored in Pinecone metadata for accurate retrieval.

## License

MIT

---

Built with ❤️ using LangChain, Google Gemini, Pinecone, and Streamlit.  
Perfect for researchers, students, or engineers diving deep into telecom literature!