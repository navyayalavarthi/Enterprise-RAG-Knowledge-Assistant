
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    description="GPT-4o-mini + FAISS + Pinecone powered document Q&A",
    version="3.0.0"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

print("Loading RAG pipeline...")

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(
    "models/faiss_index",
    hf_embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template='''You are an expert enterprise document analyst.
Answer questions based ONLY on the provided 10-K document context.
If the answer is not in the context say:
This information is not available in the document.
Always cite the page number when possible.

Context:
{context}

Question: {question}

Answer:'''
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True
)

if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("rag-knowledge-base")
    print("Pinecone connected.")
else:
    pinecone_index = None
    print("Pinecone not configured.")

print(f"GPT RAG ready. FAISS vectors: {vectorstore.index.ntotal}")

class Question(BaseModel):
    query: str
    k: int = 4

class ConvQuestion(BaseModel):
    question: str

@app.get("/")
def root():
    return {
        "message": "Enterprise RAG Knowledge Assistant",
        "version": "3.0.0",
        "model": "gpt-4o-mini",
        "document": "10-K Annual Report Q4 2023",
        "faiss_vectors": vectorstore.index.ntotal,
        "pinecone_connected": pinecone_index is not None,
        "endpoints": ["/ask", "/chat", "/search",
                      "/search/pinecone", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "gpt-4o-mini",
        "faiss_vectors": vectorstore.index.ntotal,
        "pinecone_connected": pinecone_index is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask")
def ask(q: Question):
    try:
        result = qa_chain.invoke({"query": q.query})
        sources = list(set([
            "Page " + str(doc.metadata.get("page", "?"))
            for doc in result["source_documents"]
        ]))
        return {
            "question": q.query,
            "answer": result["result"],
            "sources": sources,
            "model": "gpt-4o-mini",
            "vector_db": "FAISS",
            "chunks_retrieved": len(result["source_documents"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(q: ConvQuestion):
    try:
        result = conv_chain.invoke({"question": q.question})
        sources = list(set([
            "Page " + str(doc.metadata.get("page", "?"))
            for doc in result["source_documents"]
        ]))
        return {
            "question": q.question,
            "answer": result["answer"],
            "sources": sources,
            "model": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_faiss(q: Question):
    try:
        results = vectorstore.similarity_search_with_score(
            q.query, k=q.k)
        output = []
        for doc, score in results:
            output.append({
                "content": doc.page_content[:400],
                "page": doc.metadata.get("page", "?"),
                "relevance_score": round(float(score), 4),
                "vector_db": "FAISS"
            })
        return {
            "query": q.query,
            "results": output,
            "total_found": len(output),
            "vector_db": "FAISS"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/pinecone")
def search_pinecone(q: Question):
    if not pinecone_index:
        raise HTTPException(
            status_code=503,
            detail="Pinecone not configured"
        )
    try:
        query_embedding = hf_embeddings.embed_query(q.query)
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=q.k,
            include_metadata=True
        )
        output = []
        for match in results['matches']:
            output.append({
                "content": match['metadata'].get('text', '')[:400],
                "page": match['metadata'].get('page', '?'),
                "relevance_score": round(match['score'], 4),
                "vector_db": "Pinecone"
            })
        return {
            "query": q.query,
            "results": output,
            "total_found": len(output),
            "vector_db": "Pinecone"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
