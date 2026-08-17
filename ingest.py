"""
Ingestion script for the RBI RAG Assistant.
Loads PDFs, chunks them, creates embeddings, and saves the FAISS index.
Also generates a manifest for tracking ingested files.

Usage:
    python ingest.py
"""

import time
from src.document_loader import load_and_chunk
from src.embeddings import get_embeddings
from src.vectorstore import (
    create_vectorstore, save_vectorstore,
    compute_file_hash, save_manifest,
)
from src.config import DATA_DIR, VECTORSTORE_DIR


def main():
    print("=" * 60)
    print("RBI RAG Assistant - Document Ingestion Pipeline")
    print("=" * 60)
    print()

    start_time = time.time()

    # Step 1: Load and chunk PDFs
    print("[Step 1] Loading and chunking PDFs...")
    print("-" * 40)
    chunks = load_and_chunk(DATA_DIR)
    print()

    # Step 2: Create embeddings
    print("[Step 2] Creating embeddings...")
    print("-" * 40)
    embeddings = get_embeddings()
    print()

    # Step 3: Build FAISS index
    print("[Step 3] Building FAISS vector store...")
    print("-" * 40)
    vectorstore = create_vectorstore(chunks, embeddings)
    print()

    # Step 4: Save to disk
    print("[Step 4] Saving vector store...")
    print("-" * 40)
    save_vectorstore(vectorstore, VECTORSTORE_DIR)
    print()

    # Step 5: Generate manifest
    print("[Step 5] Generating file manifest...")
    print("-" * 40)
    manifest = {}
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        file_hash = compute_file_hash(pdf_path)
        # Count chunks belonging to this file
        chunk_count = sum(
            1 for c in chunks if c.metadata.get("filename") == pdf_path.name
        )
        manifest[pdf_path.name] = {
            "hash": file_hash,
            "chunks": chunk_count,
            "size_bytes": pdf_path.stat().st_size,
        }
        print(f"  {pdf_path.name}: {chunk_count} chunks, hash={file_hash[:12]}...")
    save_manifest(manifest)
    print()

    elapsed = time.time() - start_time

    # Summary
    print("=" * 60)
    print("[DONE] Ingestion Complete!")
    print(f"   PDFs processed: {len(pdf_files)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Index saved to: {VECTORSTORE_DIR}")
    print(f"   Manifest saved: {VECTORSTORE_DIR / 'manifest.json'}")
    print(f"   Time taken: {elapsed:.1f}s")
    print("=" * 60)
    print()
    print("You can now run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
