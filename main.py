from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import json
import hashlib

load_dotenv()

# ============ SETUP ============
app = FastAPI(title="Advanced RAG System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only — restrict to your frontend domain before deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM & Embeddings
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # confirmed working on this account's Groq key
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Initialize Qdrant
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 output size

# Create the collection if it doesn't exist yet — Qdrant/langchain_qdrant
# never does this for you, which is what caused the 404 "Collection
# `documents` doesn't exist!" error.
existing_collections = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

# Create vector store
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

# ============ LANGGRAPH STATE DEFINITION ============
class RAGState(TypedDict):
    """State object for LangGraph workflow"""
    question: str
    documents: List[Document]
    web_search_needed: bool
    relevance_score: float
    generation: str
    iteration: int
    max_iterations: int
    rewrites: List[str]

# ============ LANGGRAPH NODES ============

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents from vector store"""
    try:
        # Retrieve top 3 documents
        retrieved = vector_store.similarity_search(state["question"], k=3)
        state["documents"] = retrieved
        state["iteration"] += 1
        return state
    except Exception as e:
        print(f"Retrieval error: {e}")
        state["documents"] = []
        return state

# Grading prompt — grades ALL retrieved documents in a single call instead of
# one call per document. With k=3 this cuts 3 LLM calls down to 1 per query.
batch_grader_prompt = PromptTemplate(
    input_variables=["documents_block", "question"],
    template="""You are grading retrieved documents for relevance to a question.

Question: {question}

Documents:
{documents_block}

For each numbered document, respond with exactly one line in the format:
<number>: relevant
or
<number>: not_relevant

Output only the numbered lines, nothing else."""
)

def grade_documents_node(state: RAGState) -> RAGState:
    """Grade all retrieved documents for relevance in a single batched LLM call"""
    if not state["documents"]:
        state["relevance_score"] = 0
        state["web_search_needed"] = True
        return state

    documents_block = "\n\n".join(
        f"[{i + 1}] {doc.page_content[:500]}"
        for i, doc in enumerate(state["documents"])
    )

    grader = batch_grader_prompt | llm
    result = grader.invoke({
        "documents_block": documents_block,
        "question": state["question"]
    })

    relevant_docs = []
    relevant_count = 0
    for line in result.content.strip().splitlines():
        line = line.strip().lower()
        if ":" not in line:
            continue
        idx_part, verdict = line.split(":", 1)
        idx_part = idx_part.strip()
        verdict = verdict.strip()
        if not idx_part.isdigit():
            continue
        idx = int(idx_part) - 1
        if 0 <= idx < len(state["documents"]) and "relevant" in verdict and "not" not in verdict:
            relevant_docs.append(state["documents"][idx])
            relevant_count += 1

    # Fallback: if parsing produced nothing usable, keep original docs rather
    # than silently dropping everything retrieved.
    if not relevant_docs and not relevant_count:
        relevant_docs = state["documents"]
        relevant_count = len(state["documents"])

    state["documents"] = relevant_docs
    if state["documents"]:
        state["relevance_score"] = relevant_count / len(state["documents"])
    else:
        state["relevance_score"] = 0
    state["web_search_needed"] = state["relevance_score"] < 0.5

    return state

# Query rewrite prompt
question_rewriter_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a question rewriter. Your task is to reformulate the user's question to better retrieve relevant documents.

Original question: {question}

Provide a single reformulated question that is more specific and likely to retrieve better results."""
)

def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite query if documents weren't relevant"""
    if state["iteration"] >= state["max_iterations"]:
        return state
    
    rewriter = question_rewriter_prompt | llm
    rewritten = rewriter.invoke({"question": state["question"]})
    
    state["question"] = rewritten.content
    state["rewrites"].append(rewritten.content)
    
    return state

# Generation prompt
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an assistant for question-answering tasks.
Use only the following pieces of retrieved context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""
)

def generate_node(state: RAGState) -> RAGState:
    """Generate answer from retrieved documents"""
    if not state["documents"]:
        state["generation"] = "No relevant documents found to answer this question."
        return state
    
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    generator = rag_prompt | llm
    answer = generator.invoke({
        "context": context,
        "question": state["question"]
    })
    
    state["generation"] = answer.content
    return state

# Hallucination grader
hallucination_grader_prompt = PromptTemplate(
    input_variables=["generation", "context"],
    template="""You are a hallucination grader. Assess if the generated answer is grounded in the provided context.

Context: {context}
Generated Answer: {generation}

Does the answer contain only facts from the context? Answer with GROUNDED or HALLUCINATED."""
)

def grade_generation_node(state: RAGState) -> RAGState:
    """Grade if generation is grounded in retrieved docs (anti-hallucination)"""
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    grader = hallucination_grader_prompt | llm
    hallucination_check = grader.invoke({
        "generation": state["generation"],
        "context": context
    })
    
    is_hallucinating = "HALLUCINATED" in hallucination_check.content.upper()
    
    if is_hallucinating and state["iteration"] < state["max_iterations"]:
        return rewrite_query_node(state)
    
    return state

# ============ GRAPH ROUTING ============

def should_continue(state: RAGState) -> Literal["retrieve", "generate", END]:
    """Decide whether to continue retrieval, go to generation, or end"""
    if state["relevance_score"] >= 0.5:
        return "generate"
    elif state["iteration"] >= state["max_iterations"]:
        return "generate"
    else:
        return "retrieve"

# ============ BUILD LANGGRAPH ============

workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("grade_generation", grade_generation_node)

# Add edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    should_continue,
    {
        "retrieve": "rewrite_query",
        "generate": "generate",
        END: END
    }
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", "grade_generation")
workflow.add_edge("grade_generation", END)

rag_chain = workflow.compile()

# ============ API MODELS ============

class DocumentInput(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    iterations: int
    rewrites: List[str]
    relevance_score: float

# ============ API ENDPOINTS ============

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "version": "2.0-advanced"}

@app.post("/add-documents")
async def add_documents(doc: DocumentInput):
    """Add documents to vector store with automatic chunking.
    Skips ingestion if this exact content was already indexed (content hash
    check), so re-running test_rag.py or re-uploading the same doc is safe
    and won't create duplicate chunks in Qdrant."""
    try:
        doc_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()

        # Dedup check is fail-safe: if the scroll() call itself errors for any
        # reason (indexing, auth, transient API issue), we log it and proceed
        # with adding the document rather than blocking ingestion entirely.
        try:
            existing = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="metadata.doc_hash", match=MatchValue(value=doc_hash))]
                ),
                limit=1
            )
            if existing[0]:
                return {
                    "status": "skipped",
                    "reason": "identical content already indexed",
                    "chunks_added": 0
                }
        except Exception as dedup_error:
            print(f"[dedup check failed, proceeding with add] {dedup_error}")

        # Split text into chunks
        # chunk_size/chunk_overlap are CHARACTER counts, not tokens
        # 500 chars ≈ 100-130 English tokens
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(doc.content)
        
        # Create Document objects
        documents = [
            Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk": i, "doc_hash": doc_hash}
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # Add to Qdrant
        vector_store.add_documents(documents)
        
        return {
            "status": "success",
            "chunks_added": len(chunks),
            "total_tokens": sum(len(chunk.split()) for chunk in chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system with self-correction"""
    try:
        # Initialize state
        initial_state: RAGState = {
            "question": request.question,
            "documents": [],
            "web_search_needed": False,
            "relevance_score": 0.0,
            "generation": "",
            "iteration": 0,
            "max_iterations": 3,
            "rewrites": []
        }
        
        # Run LangGraph workflow
        final_state = rag_chain.invoke(initial_state)
        
        # Extract sources
        sources = [
            doc.page_content[:300] + "..."
            for doc in final_state["documents"]
        ]
        
        return QueryResponse(
            answer=final_state["generation"],
            sources=sources,
            iterations=final_state["iteration"],
            rewrites=final_state["rewrites"],
            relevance_score=final_state["relevance_score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    """List all collections in Qdrant"""
    try:
        collections = qdrant_client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-documents")
async def clear_documents():
    """Delete and recreate the documents collection (use before re-running test_rag.py)"""
    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)