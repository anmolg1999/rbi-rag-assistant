"""
RAG chain module.
Builds the retrieval-augmented generation chain with source citation.
Supports both RAG-only and hybrid (RAG + Web Search) modes.
"""

import re
import time
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE,
    RETRIEVER_K, RELEVANCE_THRESHOLD, WEB_SEARCH_MAX_RESULTS,
)
from src.prompts import RAG_PROMPT, HYBRID_PROMPT
from src.analytics import log_query
from src.embeddings import get_embeddings
from src.vectorstore import load_vectorstore


def format_documents(docs) -> str:
    """
    Format retrieved documents into a single context string
    with source information for each chunk.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        formatted.append(
            f"[Source {i}: {source}, Page {page + 1 if isinstance(page, int) else page}]\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(formatted)


def get_llm():
    """Create and return the Groq LLM instance."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )


def build_rag_chain(retriever=None):
    """
    Build the RAG chain using LCEL (LangChain Expression Language).

    Pipeline: question → retriever → format docs → prompt → LLM → output

    Args:
        retriever: Optional retriever. If None, loads from saved vector store.

    Returns:
        Tuple of (rag_chain, retriever) for use in the app.
    """
    # Load retriever from saved vector store if not provided
    if retriever is None:
        embeddings = get_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )

    # Build LLM
    llm = get_llm()

    # Build RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def _get_relevance_scores(retriever, question: str) -> list[tuple]:
    """
    Retrieve documents with relevance scores using similarity_search_with_score.

    Returns list of (Document, score) tuples. Lower score = more relevant for FAISS.
    """
    try:
        # Access the underlying vectorstore from the retriever
        vectorstore = retriever.vectorstore
        docs_with_scores = vectorstore.similarity_search_with_score(
            question, k=RETRIEVER_K
        )
        return docs_with_scores
    except Exception:
        # Fallback: retrieve without scores
        docs = retriever.invoke(question)
        return [(doc, 0.0) for doc in docs]


def _is_context_sufficient(docs_with_scores: list[tuple]) -> bool:
    """
    Determine if the retrieved RAG context is good enough,
    or if we should supplement with web search.

    For FAISS L2 distance: lower score = more similar.
    We normalize to a 0-1 similarity scale.
    """
    if not docs_with_scores:
        return False

    # FAISS returns L2 distance; convert to approximate similarity
    # Using: similarity ≈ 1 / (1 + distance)
    best_score = docs_with_scores[0][1]
    similarity = 1.0 / (1.0 + best_score)

    return similarity >= RELEVANCE_THRESHOLD


def query_with_sources(question: str, rag_chain=None, retriever=None,
                       enable_web_search: bool = False):
    """
    Query the RAG chain and return both the answer and source documents.
    Optionally falls back to web search if RAG context is insufficient.

    Args:
        question: User's question string.
        rag_chain: Optional pre-built RAG chain.
        retriever: Optional retriever for fetching source docs.
        enable_web_search: Whether to use web search as fallback.

    Returns:
        Dict with 'answer', 'sources', and 'source_type' keys.
    """
    start_time = time.time()
    
    if rag_chain is None or retriever is None:
        rag_chain, retriever = build_rag_chain(retriever)

    # Get source documents with relevance scores
    docs_with_scores = _get_relevance_scores(retriever, question)
    source_docs = [doc for doc, _ in docs_with_scores]

    # Decide if web search is needed
    web_results = []
    web_context = ""
    source_type = "rag"  # "rag", "web", or "hybrid"

    if enable_web_search:
        context_sufficient = _is_context_sufficient(docs_with_scores)

        if not context_sufficient:
            try:
                from src.web_search import RBIWebSearcher
                searcher = RBIWebSearcher()
                web_results = searcher.search(question, max_results=WEB_SEARCH_MAX_RESULTS)
                web_context = searcher.search_as_context(question, max_results=WEB_SEARCH_MAX_RESULTS)

                if web_context and source_docs:
                    source_type = "hybrid"
                elif web_context:
                    source_type = "web"
            except Exception as e:
                print(f"[WARN] Web search failed: {e}")

    retrieval_time = time.time() - start_time
    llm_start_time = time.time()

    # Generate answer
    if source_type in ("hybrid", "web") and web_context:
        # Use hybrid prompt with both contexts
        llm = get_llm()
        hybrid_chain = HYBRID_PROMPT | llm | StrOutputParser()
        answer = hybrid_chain.invoke({
            "context": format_documents(source_docs),
            "web_context": web_context,
            "question": question,
        })
    else:
        # Use standard RAG chain
        answer = rag_chain.invoke(question)

    # Strip reasoning blocks (<think>...</think>) often generated by models like Qwen or DeepSeek
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    
    llm_time = time.time() - llm_start_time

    # Format sources for display
    sources = []
    for doc in source_docs:
        sources.append({
            "document": doc.metadata.get("document_name", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1 if isinstance(doc.metadata.get("page"), int) else "N/A",
            "filename": doc.metadata.get("filename", "Unknown"),
            "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "type": "indexed",
        })

    # Add web sources
    for wr in web_results:
        sources.append({
            "document": wr.title,
            "page": "Web",
            "filename": wr.url,
            "content_preview": wr.snippet,
            "type": "web",
            "url": wr.url,
        })

    # Estimate Tokens
    # ~1.3 tokens per word for English
    input_tokens = int(len(question.split()) * 1.3)
    if source_type == "rag":
        input_tokens += int(len(format_documents(source_docs).split()) * 1.3)
    elif source_type == "hybrid":
        input_tokens += int(len((format_documents(source_docs) + web_context).split()) * 1.3)
    elif source_type == "web":
        input_tokens += int(len(web_context.split()) * 1.3)
        
    output_tokens = int(len(answer.split()) * 1.3)
    
    # Log to Analytics Database
    try:
        log_query(
            user_query=question,
            answer=answer,
            source_type=source_type,
            retrieval_time_sec=retrieval_time,
            llm_time_sec=llm_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    except Exception as e:
        print(f"[WARN] Failed to log query analytics: {e}")

    return {
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
    }
