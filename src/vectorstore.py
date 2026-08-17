"""
Vector store module.
Handles FAISS index creation, saving, loading, and incremental updates.
"""

import hashlib
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR


def create_vectorstore(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Create a FAISS vector store from documents.

    Args:
        documents: List of chunked Document objects.
        embeddings: Embedding model to use for vectorization.

    Returns:
        FAISS vector store instance.
    """
    print(f"[INFO] Creating FAISS index from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"[OK] FAISS index created with {len(documents)} vectors")
    return vectorstore


def save_vectorstore(
    vectorstore: FAISS,
    save_path: Path = VECTORSTORE_DIR,
) -> None:
    """
    Save FAISS vector store to disk.

    Args:
        vectorstore: FAISS vector store to save.
        save_path: Directory path to save the index.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    print(f"[OK] Vector store saved to: {save_path}")


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    load_path: Path = VECTORSTORE_DIR,
) -> FAISS:
    """
    Load FAISS vector store from disk.

    Args:
        embeddings: Embedding model (must be the same used during creation).
        load_path: Directory path to load the index from.

    Returns:
        FAISS vector store instance.

    Raises:
        FileNotFoundError: If vector store directory doesn't exist.
    """
    if not load_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {load_path}. "
            "Run 'python ingest.py' first to create it."
        )

    print(f"[INFO] Loading vector store from: {load_path}")
    vectorstore = FAISS.load_local(
        str(load_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[OK] Vector store loaded successfully")
    return vectorstore


def merge_vectorstore(
    existing: FAISS,
    new_documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Merge new documents into an existing FAISS vector store.

    Creates a temporary FAISS index from the new documents, then
    merges it into the existing one.

    Args:
        existing: Existing FAISS vector store.
        new_documents: New chunked documents to add.
        embeddings: Embedding model.

    Returns:
        Updated FAISS vector store with new documents merged in.
    """
    if not new_documents:
        print("[INFO] No new documents to merge.")
        return existing

    print(f"[INFO] Merging {len(new_documents)} new chunks into existing index...")
    new_vectorstore = FAISS.from_documents(new_documents, embeddings)
    existing.merge_from(new_vectorstore)
    print(f"[OK] Merge complete. Index now contains merged vectors.")
    return existing


def remove_documents_by_filename(
    vectorstore: FAISS,
    filename: str,
) -> int:
    """
    Remove all document chunks associated with a specific filename.

    Note: FAISS doesn't natively support deletion by metadata.
    This rebuilds the index without the matching documents, which
    is only practical for small-scale updates.

    Args:
        vectorstore: FAISS vector store.
        filename: The filename to remove (e.g., "154MD.pdf").

    Returns:
        Number of documents removed.
    """
    # Get all documents from the vectorstore
    all_docs = vectorstore.docstore._dict
    ids_to_remove = []

    for doc_id, doc in all_docs.items():
        if doc.metadata.get("filename") == filename:
            ids_to_remove.append(doc_id)

    if ids_to_remove:
        # FAISS delete by IDs
        vectorstore.delete(ids_to_remove)
        print(f"[OK] Removed {len(ids_to_remove)} chunks for '{filename}'")
    else:
        print(f"[INFO] No chunks found for '{filename}'")

    return len(ids_to_remove)


# ──────────────────────────────────────────────
# Manifest Management (for incremental updates)
# ──────────────────────────────────────────────
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_manifest() -> dict:
    """
    Load the ingestion manifest from disk.

    Returns:
        Dict mapping filename → {"hash": str, "chunks": int}
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """
    Save the ingestion manifest to disk.

    Args:
        manifest: Dict mapping filename → {"hash": str, "chunks": int}
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest saved: {len(manifest)} files tracked")
