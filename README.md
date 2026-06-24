## 🤖 Enterprise RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)
![FAISS](https://img.shields.io/badge/FAISS-Local%20Vector%20DB-purple)
![Pinecone](https://img.shields.io/badge/Pinecone-AWS%20us--east--1-teal)
![FastAPI](https://img.shields.io/badge/FastAPI-v3.0.0-red)

## Project Overview

Developed a Retrieval-Augmented Generation platform enabling users to search and interact with enterprise documents using natural language queries, combining semantic search, vector
databases, and LLMs. Built on a real 10-K annual report with dual vector database architecture using FAISS (local) and Pinecone (AWS cloud).

## Key Results

| Metric | Value |
|--------|-------|
| Document | 10-K Annual Report Q4 2023 |
| Total Pages | 80 |
| Total Chunks | 358 |
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Embedding Dimensions | 384 |
| Vector DB Local | FAISS — 358 vectors |
| Vector DB Cloud | Pinecone — AWS us-east-1 — 358 vectors |
| LLM | OpenAI gpt-4o-mini |
| API Endpoints | /ask /chat /search /search/pinecone /health |

## Architecture
10-K Annual Report PDF
↓
[Document Ingestion]

PyPDFLoader — 80 pages
↓
[Chunking Pipeline]
RecursiveCharacterTextSplitter
chunk_size=1000 overlap=200
358 chunks created
↓
[Embedding Generation]
sentence-transformers/all-MiniLM-L6-v2
384-dimensional dense vectors
↓
[Dual Vector Indexing]FAISS — local index — fast retrieval
Pinecone — AWS cloud — scalable production
↓
[GPT-4o-mini LLM]
Custom prompt engineering
Document-only answers
Page-level source attribution
Conversational memory
↓
[FastAPI Service v3.0.0]
POST /ask    — GPT Q&A with citations
POST /chat   — Conversational multi-turn
POST /search — FAISS local semantic search
POST /search/pinecone — Pinecone AWS cloud search
GET  /health — System status
## Tech Stack

| Layer | Technology |
|-------|------------|
| Document Loading | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB Local | FAISS |
| Vector DB Cloud | Pinecone Serverless on AWS us-east-1 |
| LLM | OpenAI gpt-4o-mini |
| Orchestration | LangChain RetrievalQA |
| Conversation | ConversationBufferMemory |
| Prompt Engineering | Custom PromptTemplate |
| API Framework | FastAPI 3.0.0 |
| Cloud | AWS (via Pinecone Serverless) |
| Language | Python 3.11 |

## Pinecone AWS Integration

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name="rag-knowledge-base",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

The Pinecone vector database runs on AWS serverless
infrastructure in us-east-1 region, providing scalable
cloud-based semantic search.

## Prompt Engineering

```python
PROMPT = PromptTemplate(
    template='''You are an expert enterprise document analyst.
Answer questions based ONLY on the provided 10-K document context.
If the answer is not in the context say:
This information is not available in the document.
Always cite the page number when possible.

Context: {context}
Question: {question}
Answer:'''
)
```

## FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | System info and vector counts |
| /health | GET | Health check — model and DB status |
| /ask | POST | GPT Q&A with source attribution |
| /chat | POST | Conversational multi-turn Q&A |
| /search | POST | FAISS local semantic search |
| /search/pinecone | POST | Pinecone AWS cloud search |
| /docs | GET | Swagger UI documentation |

## Project Structure
```
RAG-Assistant/
├── notebooks/
│   └── RAG Assistant.ipynb
├── src/
│   └── api.py
├── output/
│   └── rag_final_results.json
├── screenshots/
│   ├── 01_rag_system_summary.png
│   ├── 02_rag_pipeline_architecture.png
│   ├── 03_rag_qa_results.png
│   ├── 04_rag_api_endpoints.png
│   └── 05_rag_chunk_analysis.png
├── requirements.txt
├── .gitignore
└── README.md
```
## Setup and Installation

```bash
git clone https://github.com/navyayalavarthi/rag-assistant.git
cd rag-assistant

python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate

pip install -r requirements.txt

# Add 10-K PDF to docs/ folder
# Set environment variables
export OPENAI_API_KEY=your-openai-key
export PINECONE_API_KEY=your-pinecone-key

# Run notebook to build indexes
jupyter notebook "notebooks/RAG Assistant.ipynb"

# Start FastAPI server
uvicorn src.api:app --reload --port 8006
```

## Skills Used

Python · LangChain · OpenAI GPT · FAISS · Pinecone · FastAPI · AWS  · Sentence Transformers · Vector Databases · Prompt Engineering · Source Attribution · Conversational AI · Semantic Search · Jupyter

## Author

Navya
📧 navya.yalavarthi1@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/navya-yalavarthi-b21297289/

## License

MIT License — for educational and portfolio purposes.
10-K document used for educational demonstration only.
