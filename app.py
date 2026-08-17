"""
RBI Regulatory Circular Assistant — Streamlit Chat UI

A RAG-powered chatbot that answers questions about RBI Master Directions
related to credit regulations for commercial banks.
Supports hybrid RAG + Web Search for comprehensive answers.

Usage:
    streamlit run app.py
"""

import streamlit as st
from src.chain import build_rag_chain, query_with_sources
from src.config import VECTORSTORE_DIR


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RBI Regulatory Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.2rem 0 0.3rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .sub-header {
        text-align: center;
        color: #a0aec0;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Source card styling */
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        transition: border-color 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
    }
    .source-card .doc-name {
        color: #667eea;
        font-weight: 600;
    }
    .source-card .page-num {
        color: #a0aec0;
        font-size: 0.8rem;
    }
    .source-card.web-source .doc-name {
        color: #48bb78;
    }

    /* Source type badge */
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-rag {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .badge-web {
        background: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
    .badge-hybrid {
        background: rgba(237, 137, 54, 0.15);
        color: #ed8936;
        border: 1px solid rgba(237, 137, 54, 0.3);
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .sidebar-note {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 14px;
        font-size: 0.88rem;
        color: #cbd5e0;
        line-height: 1.6;
    }
    .sidebar-note strong {
        color: #e2e8f0;
    }

    /* Status indicator */
    .status-ready {
        color: #48bb78;
        font-weight: 600;
    }
    .status-error {
        color: #fc8181;
        font-weight: 600;
    }

    /* Example question buttons */
    .example-btn {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 8px;
        padding: 8px 14px;
        color: #a0aec0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        width: 100%;
    }
    .example-btn:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.5);
        color: #e2e8f0;
    }

    /* Toggle container */
    .toggle-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🏦 RBI RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Status check
    if VECTORSTORE_DIR.exists():
        st.markdown('<span class="status-ready">● Vector Store Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-error">● Vector Store Not Found</span>', unsafe_allow_html=True)
        st.error("Run `python ingest.py` first to build the vector store.")

    st.markdown("---")

    # Descriptive note (replaces the circular list)
    st.markdown("""
    <div class="sidebar-note">
        <strong>What does this assistant do?</strong><br><br>
        Ask any question about <strong>RBI credit regulations</strong> for commercial banks.
        This assistant is powered by <strong>20+ indexed RBI Master Directions</strong>
        covering IRAC norms, credit facilities, PSL, MSME lending, KYC, digital lending,
        capital adequacy, and more.<br><br>
        🔒 All answers are <strong>grounded in official RBI documents</strong> with source citations.
        No hallucinations — if the answer isn't in the data, it will tell you.<br><br>
        🌐 Enable <strong>Web Search</strong> below to also search rbi.org.in for the latest updates.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Web search toggle
    st.markdown("**⚙️ Settings**")
    enable_web_search = st.toggle(
        "🌐 Enable RBI Web Search",
        value=False,
        help="When enabled, the assistant will also search rbi.org.in for the latest circulars and updates if the indexed documents don't fully cover your question."
    )

    st.markdown("---")

    # Update knowledge base button
    if st.button("🔄 Update Knowledge Base", use_container_width=True,
                  help="Detect new or modified PDFs in the data/ folder and update the vector store"):
        with st.spinner("🔄 Checking for updates..."):
            try:
                from update_vectorstore import detect_changes, update_vectorstore as run_update
                changes = detect_changes()
                total = len(changes["new"]) + len(changes["modified"]) + len(changes["deleted"])
                if total == 0:
                    st.success("✅ Knowledge base is up to date!")
                else:
                    with st.spinner(f"📥 Updating... ({total} changes detected)"):
                        run_update()
                    st.success(f"✅ Updated! {len(changes['new'])} new, {len(changes['modified'])} modified, {len(changes['deleted'])} removed.")
                    st.cache_resource.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Update failed: {str(e)}")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **RBI Regulatory Circular Assistant**

        A RAG (Retrieval-Augmented Generation) chatbot that answers questions
        about RBI Master Directions related to credit regulations.

        **Tech Stack:**
        - 🧠 LLM: Llama 3.1 8B (via Groq)
        - 📊 Embeddings: all-MiniLM-L6-v2
        - 🗃️ Vector Store: FAISS
        - 🔗 Framework: LangChain
        - 🌐 Web Search: DuckDuckGo (rbi.org.in)
        - 🎨 UI: Streamlit

        **Anti-Hallucination:**
        The assistant answers ONLY from the indexed RBI circulars
        and cites sources for every response.

        **Features:**
        - 📄 20+ indexed RBI Master Directions
        - 🌐 Optional web search restricted to rbi.org.in
        - 🔄 Incremental knowledge base updates
        - 📑 Source citations with page numbers
        """)


# ──────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────
st.markdown('<div class="main-header">🏦 RBI Regulatory Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about RBI Master Directions on Credit Regulations</div>',
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.retriever = None


@st.cache_resource(show_spinner="🔄 Loading RAG pipeline...")
def load_rag_pipeline():
    """Load and cache the RAG chain and retriever."""
    chain, retriever = build_rag_chain()
    return chain, retriever


# Load RAG pipeline
if VECTORSTORE_DIR.exists():
    try:
        rag_chain, retriever = load_rag_pipeline()
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
    except Exception as e:
        st.error(f"❌ Error loading RAG pipeline: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Vector store not found. Please run `python ingest.py` first.")
    st.stop()


# ──────────────────────────────────────────────
# Example Questions (shown only when chat is empty)
# ──────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What are the IRAC norms for asset classification?",
    "What are the Priority Sector Lending targets for banks?",
    "Explain the KYC requirements for opening a bank account",
    "What is the LTV ratio for housing loans?",
    "What are the guidelines for digital lending?",
    "How are wilful defaulters classified by RBI?",
]

if not st.session_state.messages:
    st.markdown("##### 💡 Try asking:")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(f"→ {question}", key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()


# ──────────────────────────────────────────────
# Display chat history
# ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Show source type badge for assistant messages
        if message["role"] == "assistant" and "source_type" in message:
            source_type = message["source_type"]
            if source_type == "rag":
                st.markdown('<span class="source-badge badge-rag">📄 From Indexed Documents</span>', unsafe_allow_html=True)
            elif source_type == "web":
                st.markdown('<span class="source-badge badge-web">🌐 From RBI Website</span>', unsafe_allow_html=True)
            elif source_type == "hybrid":
                st.markdown('<span class="source-badge badge-hybrid">📄🌐 Indexed + Web Results</span>', unsafe_allow_html=True)

        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 Sources", expanded=False):
                for src in message["sources"]:
                    is_web = src.get("type") == "web"
                    card_class = "source-card web-source" if is_web else "source-card"
                    icon = "🌐" if is_web else "📄"

                    url_line = ""
                    if is_web and src.get("url"):
                        url_line = f'<a href="{src["url"]}" target="_blank" style="color:#48bb78;font-size:0.8rem;">🔗 Open on RBI website</a><br>'

                    st.markdown(
                        f'<div class="{card_class}">'
                        f'<span class="doc-name">{icon} {src["document"]}</span><br>'
                        f'<span class="page-num">{"URL" if is_web else "Page"} {src["page"]}</span><br>'
                        f'{url_line}'
                        f'<small>{src["content_preview"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────
if prompt := st.chat_input("Ask about RBI credit regulations..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Generate response if the last message is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        search_label = "🔍 Searching RBI circulars & web..." if enable_web_search else "🔍 Searching RBI circulars..."
        with st.spinner(search_label):
            try:
                result = query_with_sources(
                    question=prompt,
                    rag_chain=st.session_state.rag_chain,
                    retriever=st.session_state.retriever,
                    enable_web_search=enable_web_search,
                )

                # Show source type badge
                source_type = result.get("source_type", "rag")
                if source_type == "rag":
                    st.markdown('<span class="source-badge badge-rag">📄 From Indexed Documents</span>', unsafe_allow_html=True)
                elif source_type == "web":
                    st.markdown('<span class="source-badge badge-web">🌐 From RBI Website</span>', unsafe_allow_html=True)
                elif source_type == "hybrid":
                    st.markdown('<span class="source-badge badge-hybrid">📄🌐 Indexed + Web Results</span>', unsafe_allow_html=True)

                # Display answer
                st.markdown(result["answer"])

                # Display sources
                with st.expander("📄 Sources", expanded=False):
                    for src in result["sources"]:
                        is_web = src.get("type") == "web"
                        card_class = "source-card web-source" if is_web else "source-card"
                        icon = "🌐" if is_web else "📄"

                        url_line = ""
                        if is_web and src.get("url"):
                            url_line = f'<a href="{src["url"]}" target="_blank" style="color:#48bb78;font-size:0.8rem;">🔗 Open on RBI website</a><br>'

                        st.markdown(
                            f'<div class="{card_class}">'
                            f'<span class="doc-name">{icon} {src["document"]}</span><br>'
                            f'<span class="page-num">{"URL" if is_web else "Page"} {src["page"]}</span><br>'
                            f'{url_line}'
                            f'<small>{src["content_preview"]}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "source_type": source_type,
                })

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

