"""
Configuration module for the RBI RAG Assistant.
Loads environment variables and defines project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"  # ../data (where PDFs are stored)
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# ──────────────────────────────────────────────
# Document Processing
# ──────────────────────────────────────────────
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
LLM_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0        # Deterministic answers for regulatory queries

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
RETRIEVER_K = 4            # Number of chunks to retrieve

# ──────────────────────────────────────────────
# Web Search
# ──────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 5
RELEVANCE_THRESHOLD = 0.55     # Minimum similarity score to skip web search

# ──────────────────────────────────────────────
# PDF filename → human-readable document name mapping
# ──────────────────────────────────────────────
DOCUMENT_NAMES = {
    # ── Original 13 documents ──
    "164MD.pdf": "IRAC Norms (Income Recognition, Asset Classification & Provisioning)",
    "154MD.pdf": "Credit Facilities",
    "157MD.pdf": "Credit Risk Management",
    "156MD.pdf": "Credit Information Reporting",
    "155MD.pdf": "Credit Cards & Debit Cards: Issuance and Conduct",
    "397MD7DCC951F9F014A738C60E0FBC1A73D70.pdf": "Capital Charge for Credit Risk – Standardised Approach",
    "165MD.pdf": "Resolution of Stressed Assets",
    "166MD.pdf": "Treatment of Wilful Defaulters and Large Defaulters",
    "159MD.pdf": "Transfer and Distribution of Credit Risk",
    "158MD.pdf": "Concentration Risk Management",
    "151MD.pdf": "Prudential Norms on Capital Adequacy (Basel III)",
    "161MD.pdf": "Interest Rates on Advances",
    "160MD.pdf": "Securitisation Transactions",
    # ── Additional credit-related documents ──
    "MD_PSL.pdf": "Priority Sector Lending – Targets and Classification",
    "MD_MSME.pdf": "Lending to Micro, Small & Medium Enterprises (MSME) Sector",
    "MD_KYC.pdf": "Know Your Customer (KYC) Direction",
    "MD_CRR_SLR.pdf": "Cash Reserve Ratio and Statutory Liquidity Ratio",
    "MD_HousingFinance.pdf": "Housing Finance Companies",
    "MD_InterestRates.pdf": "Interest Rates on Advances",
    "MD_DigitalPayments.pdf": "Digital Payment Security Controls",
}


def get_document_name(filename: str) -> str:
    """Return human-readable name for a PDF filename."""
    return DOCUMENT_NAMES.get(filename, filename)
