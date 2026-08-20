"""
RAG chain module.
Builds the retrieval-augmented generation chain with source citation.
Supports intent-based routing: RAG, general banking, live rates, and meta responses.
"""

import re
import time
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE,
    RETRIEVER_K, RELEVANCE_THRESHOLD,
)
from src.prompts import (
    RAG_PROMPT, GENERAL_BANKING_PROMPT, RATE_ANSWER_PROMPT,
    META_RESPONSE, OUT_OF_SCOPE_RESPONSE,
)
from src.router import classify_intent
from src.rbi_rates import fetch_current_rates, format_rates_for_llm, is_rate_question
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
    Determine if the retrieved RAG context is good enough.

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


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def contextualize_question(question: str, chat_history: list) -> str:
    """
    Given chat history and the latest user question, which might reference context 
    in the chat history, formulate a standalone question.
    """
    if not chat_history:
        return question
        
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # Format chat history for LangChain
    formatted_history = []
    for msg in chat_history[-6:]:  # Only look at the last 3 exchanges (6 messages)
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))
            
    chain = prompt | llm | StrOutputParser()
    standalone_question = chain.invoke({"chat_history": formatted_history, "question": question})
    return standalone_question.strip()


def query_with_sources(question: str, rag_chain=None, retriever=None, chat_history=None, **kwargs):
    """
    Query the assistant with smart intent routing and chat history support.

    Flow:
    1. Contextualize the question using chat history
    2. Classify intent (meta / rag / general_banking / out_of_scope)
    3. Route to appropriate handler
    4. Return answer with sources and source type
    """
    start_time = time.time()
    
    # Contextualize question if we have history (e.g. "explain this in detail" -> "explain non-regularisation of BG in detail")
    standalone_question = contextualize_question(question, chat_history) if chat_history else question

    # ── Step 1: Classify intent ──
    intent = classify_intent(standalone_question)

    # ── Step 2: Handle META questions ──
    if intent == "meta":
        elapsed = time.time() - start_time
        _log_safely(question, META_RESPONSE, "meta", 0, elapsed, 0, 0)
        return {
            "answer": META_RESPONSE,
            "sources": [],
            "source_type": "meta",
        }

    # ── Step 3: Handle OUT OF SCOPE questions ──
    if intent == "out_of_scope":
        elapsed = time.time() - start_time
        _log_safely(question, OUT_OF_SCOPE_RESPONSE, "out_of_scope", 0, elapsed, 0, 0)
        return {
            "answer": OUT_OF_SCOPE_RESPONSE,
            "sources": [],
            "source_type": "out_of_scope",
        }

    # ── Step 4: Handle RATE questions (live scraping) ──
    if intent == "general_banking" and is_rate_question(standalone_question):
        rates = fetch_current_rates()
        if rates:
            retrieval_time = time.time() - start_time
            llm_start = time.time()

            llm = get_llm()
            rate_chain = RATE_ANSWER_PROMPT | llm | StrOutputParser()
            rate_context = format_rates_for_llm(rates)

            answer = rate_chain.invoke({
                "rate_data": rate_context,
                "question": standalone_question,
            })
            answer = _clean_answer(answer)

            llm_time = time.time() - llm_start
            input_tokens = int(len((rate_context + standalone_question).split()) * 1.3)
            output_tokens = int(len(answer.split()) * 1.3)

            _log_safely(standalone_question, answer, "live_rate", retrieval_time, llm_time, input_tokens, output_tokens)

            return {
                "answer": answer,
                "sources": [{
                    "document": "RBI Official Website — Key Rates",
                    "page": rates.get("data_date", "Latest"),
                    "filename": "rbi.org.in",
                    "content_preview": rate_context[:300],
                    "type": "live_rate",
                    "url": "https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4",
                }],
                "source_type": "live_rate",
            }

    # ── Step 5: Ensure RAG chain is ready ──
    if rag_chain is None or retriever is None:
        rag_chain, retriever = build_rag_chain(retriever)

    # ── Step 6: Handle GENERAL BANKING questions ──
    if intent == "general_banking":
        # Retrieve context to supplement the general banking knowledge
        docs_with_scores = _get_relevance_scores(retriever, standalone_question)
        source_docs = [doc for doc, _ in docs_with_scores]
        context_ok = _is_context_sufficient(docs_with_scores)
        
        # Only include context if it's somewhat relevant, otherwise just pass empty
        context = format_documents(source_docs) if context_ok else "No directly relevant indexed documents found."

        retrieval_time = time.time() - start_time
        llm_start = time.time()

        # Always use the general banking prompt for general banking intent!
        llm = get_llm()
        general_chain = GENERAL_BANKING_PROMPT | llm | StrOutputParser()
        answer = general_chain.invoke({
            "context": context,
            "question": standalone_question,
        })
        answer = _clean_answer(answer)
        source_type = "general"

        llm_time = time.time() - llm_start

        # Format sources only if we actually passed them in
        sources = _format_sources(source_docs) if context_ok else []

        input_tokens = int(len((context + standalone_question).split()) * 1.3)
        output_tokens = int(len(answer.split()) * 1.3)

        _log_safely(standalone_question, answer, source_type, retrieval_time, llm_time, input_tokens, output_tokens)

        return {
            "answer": answer,
            "sources": sources,
            "source_type": source_type,
        }

    # ── Step 7: Handle RAG questions (default) ──
    docs_with_scores = _get_relevance_scores(retriever, standalone_question)
    source_docs = [doc for doc, _ in docs_with_scores]

    retrieval_time = time.time() - start_time
    llm_start = time.time()

    answer = rag_chain.invoke(standalone_question)
    answer = _clean_answer(answer)

    llm_time = time.time() - llm_start

    # If the RAG chain couldn't find the answer, AUTOMATICALLY fallback to general banking
    failure_phrases = ["couldn't find the answer", "could not find the answer", "not mention", "no information"]
    if any(phrase in answer.lower() for phrase in failure_phrases):
        
        # We invoke the general banking chain instead, so it can answer with general knowledge
        llm = get_llm()
        general_chain = GENERAL_BANKING_PROMPT | llm | StrOutputParser()
        context_ok = _is_context_sufficient(docs_with_scores)
        context = format_documents(source_docs) if context_ok else "No directly relevant indexed documents found."
        
        fallback_start = time.time()
        answer = general_chain.invoke({
            "context": context,
            "question": standalone_question,
        })
        answer = _clean_answer(answer)
        llm_time += (time.time() - fallback_start)
        
        source_type = "general"
        sources = _format_sources(source_docs) if context_ok else []
    else:
        source_type = "rag"
        sources = _format_sources(source_docs)

    input_tokens = int(len((format_documents(source_docs) + standalone_question).split()) * 1.3)
    output_tokens = int(len(answer.split()) * 1.3)

    _log_safely(standalone_question, answer, source_type, retrieval_time, llm_time, input_tokens, output_tokens)

    return {
        "answer": answer,
        "sources": sources,
        "source_type": source_type,
    }


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _clean_answer(answer: str) -> str:
    """Strip reasoning blocks (<think>...</think>) sometimes generated by models."""
    return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()


def _format_sources(source_docs: list) -> list[dict]:
    """Format source documents for display."""
    sources = []
    for doc in source_docs:
        sources.append({
            "document": doc.metadata.get("document_name", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1 if isinstance(doc.metadata.get("page"), int) else "N/A",
            "filename": doc.metadata.get("filename", "Unknown"),
            "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            "type": "indexed",
        })
    return sources


def _log_safely(question, answer, source_type, retrieval_time, llm_time, input_tokens, output_tokens):
    """Log query analytics, silently catching errors."""
    try:
        log_query(
            user_query=question,
            answer=answer[:500],  # Truncate for DB storage
            source_type=source_type,
            retrieval_time_sec=retrieval_time,
            llm_time_sec=llm_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        print(f"[WARN] Failed to log query analytics: {e}")
