"""
Document loader and chunking module.
Loads RBI circular PDFs and splits them into chunks for embedding.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, get_document_name


def load_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load all PDF files from the data directory.
    
    Each page is loaded as a separate Document with metadata including
    the source filename and page number.
    
    Args:
        data_dir: Path to directory containing PDF files.
        
    Returns:
        List of Document objects with page content and metadata.
    """
    all_documents = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Please place your RBI circular PDFs in the data directory."
        )
    
    print(f"[INFO] Found {len(pdf_files)} PDF files in {data_dir}")
    
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Enrich metadata with human-readable document name
        for page in pages:
            page.metadata["document_name"] = get_document_name(pdf_path.name)
            page.metadata["filename"] = pdf_path.name
        
        all_documents.extend(pages)
        print(f"     -> {len(pages)} pages loaded")
    
    print(f"\n[OK] Total pages loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraph breaks, then sentences, then words - preserving
    semantic coherence within each chunk.
    
    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[OK] Split {len(documents)} pages into {len(chunks)} chunks")
    print(f"   (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


def load_and_chunk(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Convenience function: load PDFs and chunk them in one step.
    
    Args:
        data_dir: Path to directory containing PDF files.
        
    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_pdfs(data_dir)
    chunks = chunk_documents(documents)
    return chunks
