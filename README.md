# 🏦 RBI Regulatory Circular Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about public RBI Master Directions related to **credit regulations** for commercial banks. Features hybrid RAG + Web Search, incremental knowledge base updates, and source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Groq](https://img.shields.io/badge/LLM-Llama_3.1_8B-orange?logo=meta)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-blue)

## 🏗️ Architecture

```
RBI Circular PDFs (20 documents)
       │
       ▼
Document Loader + Chunking (PyPDFLoader + RecursiveCharacterTextSplitter)
       │
       ▼
Embeddings (HuggingFace sentence-transformers/all-MiniLM-L6-v2, local)
       │
       ▼
Vector Store (FAISS, local)
       │
       ▼
Retriever ──► Relevance Check ──► Anti-Hallucination Prompt ──► Llama 3.1 8B (via Groq)
       │              │
       │        Low relevance?
       │              │
       │              ▼
       │     Web Search (rbi.org.in only)
       │              │
       │              ▼
       │     Hybrid Prompt (RAG + Web)
       │
       ▼
Answer + Source Citation (Document Name + Page Number + Web URLs)
       │
       ▼
Streamlit Chat UI
```

## ✨ Features

- **📄 20+ Indexed RBI Master Directions** — Comprehensive credit regulation coverage
- **🌐 RBI Web Search** — Optional live search on rbi.org.in for latest updates
- **🔄 Incremental Updates** — Smart updater detects new/modified PDFs and updates only what's changed
- **🛡️ Anti-Hallucination** — Answers ONLY from official RBI documents with source citations
- **🏷️ Source Badges** — Clear indicators showing if answers come from indexed docs, web, or both
- **💡 Example Questions** — Quick-start buttons for common queries
- **🎨 Premium UI** — Dark theme with gradient accents and smooth interactions

## 📚 Indexed Documents

| # | Document | Coverage |
|---|----------|----------|
| 1 | IRAC Norms | Income Recognition, Asset Classification & Provisioning |
| 2 | Credit Facilities | Lending norms, statutory restrictions |
| 3 | Credit Risk Management | Risk framework for commercial banks |
| 4 | Credit Information Reporting | Credit bureau reporting norms |
| 5 | Credit Cards & Debit Cards | Issuance and conduct guidelines |
| 6 | Capital Charge for Credit Risk | Standardised approach |
| 7 | Resolution of Stressed Assets | Stressed asset resolution framework |
| 8 | Wilful Defaulters | Treatment of wilful and large defaulters |
| 9 | Transfer of Credit Risk | Credit risk distribution norms |
| 10 | Concentration Risk | Exposure limit management |
| 11 | Capital Adequacy (Basel III) | Basel III prudential norms |
| 12 | Interest Rates on Advances | Lending rate regulations |
| 13 | Securitisation Transactions | Securitisation framework |
| 14 | **Priority Sector Lending** | PSL targets, agriculture, MSME allocation |
| 15 | **MSME Lending** | MSME credit guidelines, Udyam registration |
| 16 | **KYC Direction** | KYC/AML norms for credit operations |
| 17 | **Loans & Advances Restrictions** | Statutory lending restrictions |
| 18 | **Housing Finance** | Housing credit norms, LTV ratios |
| 19 | **Gold Loans** | Gold loan LTV norms, lending guidelines |
| 20 | **Digital Lending** | LSP framework, FLDG, digital credit rules |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file:
```env
GROQ_API_KEY="your_groq_api_key"
HF_TOKEN="your_huggingface_token"
```

### 3. Download Additional Circulars (Optional)

```bash
python download_circulars.py
```

This downloads 7 additional credit-related RBI Master Directions. You can also manually download PDFs from [rbi.org.in](https://rbi.org.in) and place them in the `data/` folder.

### 4. Ingest Documents

```bash
python ingest.py
```

This loads all PDFs, chunks them, creates embeddings, and saves the FAISS index.

### 5. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 🔄 Updating the Knowledge Base

When new circulars are available or existing ones are updated:

1. Place new/updated PDFs in the `data/` folder
2. Run the incremental updater:
   ```bash
   python update_vectorstore.py
   ```
   Or click the **🔄 Update Knowledge Base** button in the sidebar.

The updater uses file hashes to detect changes — only new/modified files are re-processed, not the entire corpus.

## 🌐 Web Search Feature

The assistant can optionally search **rbi.org.in** for the latest circulars and notifications:

1. Toggle **"🌐 Enable RBI Web Search"** in the sidebar
2. When enabled, if the indexed documents don't fully answer your question, the assistant automatically searches the RBI website
3. Results from web search are clearly marked with 🌐 badges
4. All web results link back to the official RBI page

## 🛡️ Anti-Hallucination Design

This assistant is designed for a **regulated industry context**:

- **Grounded responses**: Answers ONLY from retrieved RBI circular text
- **Source citation**: Every answer includes document name and page number
- **Graceful fallback**: Explicitly states when a question isn't covered
- **Zero temperature**: Deterministic LLM responses for consistency
- **Source badges**: Clear indication of source type (indexed / web / hybrid)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Llama 3.1 8B Instant (via Groq API) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Vector Store | FAISS (local) |
| Framework | LangChain (LCEL) |
| Web Search | DuckDuckGo (rbi.org.in restricted) |
| UI | Streamlit |
| Documents | 20 RBI Master Directions (PDF) |

## 📁 Project Structure

```
RBI-RAG-Assistant/
├── .env                        # API keys (not committed)
├── .gitignore
├── .streamlit/
│   ├── config.toml             # Streamlit theme config
│   └── secrets.toml.example    # Secrets template
├── requirements.txt
├── packages.txt                # System deps for Streamlit Cloud
├── README.md
├── download_circulars.py       # Download additional RBI PDFs
├── ingest.py                   # PDF → FAISS full ingestion
├── update_vectorstore.py       # Incremental vector store updater
├── app.py                      # Streamlit chat UI
├── src/
│   ├── __init__.py
│   ├── config.py               # Environment & constants
│   ├── document_loader.py      # PDF loading + chunking
│   ├── embeddings.py           # HuggingFace embeddings
│   ├── vectorstore.py          # FAISS operations + manifest
│   ├── prompts.py              # Anti-hallucination prompts
│   ├── chain.py                # RAG chain with hybrid search
│   └── web_search.py           # RBI-restricted web search
├── vectorstore/                # FAISS index (generated)
└── ../data/                    # RBI circular PDFs
```

## 🚀 Deploy to Streamlit Cloud

1. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RBI RAG Assistant"
   git remote add origin https://github.com/anmolg1999/RBI-RAG-Assistant.git
   git push -u origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io/)

3. Connect your GitHub account and select the repository

4. Add your secrets in the Streamlit Cloud dashboard:
   - `GROQ_API_KEY`
   - `HF_TOKEN`

5. Deploy! Your app will be live at `your-app-name.streamlit.app`

> **Note:** The `data/` folder with PDFs and `vectorstore/` must be committed for cloud deployment, or you'll need to set up a build step to generate them.

## 📝 License

Personal project — built for learning and portfolio purposes.
