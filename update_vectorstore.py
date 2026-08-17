"""
Incremental vector store updater for the RBI RAG Assistant.

Scans the data/ directory, detects new or modified PDFs by comparing
file hashes against the stored manifest, and incrementally updates
the FAISS index without full rebuilds.

Usage:
    python update_vectorstore.py
"""

import time
from pathlib import Path

from src.config import DATA_DIR, VECTORSTORE_DIR
from src.document_loader import load_pdfs, chunk_documents
from src.embeddings import get_embeddings
from src.vectorstore import (
    load_vectorstore, save_vectorstore,
    merge_vectorstore, remove_documents_by_filename,
    compute_file_hash, load_manifest, save_manifest,
    create_vectorstore,
)


def detect_changes(data_dir: Path = DATA_DIR) -> dict:
    """
    Compare current PDFs against the manifest to find changes.

    Returns:
        Dict with keys: "new", "modified", "deleted" —
        each mapping to a list of filenames.
    """
    manifest = load_manifest()
    current_files = {f.name: f for f in sorted(data_dir.glob("*.pdf"))}

    changes = {
        "new": [],
        "modified": [],
        "deleted": [],
    }

    # Check for new and modified files
    for filename, filepath in current_files.items():
        if filename not in manifest:
            changes["new"].append(filename)
        else:
            current_hash = compute_file_hash(filepath)
            if current_hash != manifest[filename].get("hash"):
                changes["modified"].append(filename)

    # Check for deleted files
    for filename in manifest:
        if filename not in current_files:
            changes["deleted"].append(filename)

    return changes


def update_vectorstore(data_dir: Path = DATA_DIR):
    """
    Perform an incremental update of the vector store.

    1. Detect new/modified/deleted PDFs
    2. Remove chunks for modified/deleted files
    3. Add chunks for new/modified files
    4. Save the updated index and manifest
    """
    print("=" * 60)
    print("RBI RAG Assistant — Incremental Vector Store Update")
    print("=" * 60)
    print()

    start_time = time.time()

    # Step 1: Detect changes
    print("[Step 1] Detecting changes...")
    print("-" * 40)
    changes = detect_changes(data_dir)

    total_changes = len(changes["new"]) + len(changes["modified"]) + len(changes["deleted"])

    if total_changes == 0:
        print("  ✅ No changes detected. Vector store is up to date.")
        return

    if changes["new"]:
        print(f"  🆕 New files ({len(changes['new'])}):")
        for f in changes["new"]:
            print(f"     + {f}")
    if changes["modified"]:
        print(f"  📝 Modified files ({len(changes['modified'])}):")
        for f in changes["modified"]:
            print(f"     ~ {f}")
    if changes["deleted"]:
        print(f"  🗑️  Deleted files ({len(changes['deleted'])}):")
        for f in changes["deleted"]:
            print(f"     - {f}")
    print()

    # Step 2: Load embeddings
    print("[Step 2] Loading embedding model...")
    print("-" * 40)
    embeddings = get_embeddings()
    print()

    # Step 3: Load or create vector store
    print("[Step 3] Loading existing vector store...")
    print("-" * 40)
    manifest = load_manifest()

    if VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists():
        vectorstore = load_vectorstore(embeddings)
    else:
        print("  ⚠️ No existing vector store. Will create from scratch.")
        vectorstore = None

    print()

    # Step 4: Handle deletions and modifications (remove old chunks)
    files_to_remove = changes["deleted"] + changes["modified"]
    if files_to_remove and vectorstore is not None:
        print("[Step 4] Removing outdated chunks...")
        print("-" * 40)
        for filename in files_to_remove:
            remove_documents_by_filename(vectorstore, filename)
            if filename in manifest:
                del manifest[filename]
        print()
    else:
        print("[Step 4] No chunks to remove.")
        print()

    # Step 5: Process new and modified files
    files_to_add = changes["new"] + changes["modified"]
    if files_to_add:
        print("[Step 5] Processing new/modified files...")
        print("-" * 40)

        # Load only the specific PDFs
        new_docs = []
        for filename in files_to_add:
            filepath = data_dir / filename
            if filepath.exists():
                from langchain_community.document_loaders import PyPDFLoader
                from src.config import get_document_name

                print(f"  Loading: {filename}")
                loader = PyPDFLoader(str(filepath))
                pages = loader.load()

                for page in pages:
                    page.metadata["document_name"] = get_document_name(filename)
                    page.metadata["filename"] = filename

                new_docs.extend(pages)
                print(f"     -> {len(pages)} pages loaded")

        # Chunk the new documents
        new_chunks = chunk_documents(new_docs)
        print(f"  Total new chunks: {len(new_chunks)}")

        # Merge into existing or create new
        if vectorstore is not None:
            vectorstore = merge_vectorstore(vectorstore, new_chunks, embeddings)
        else:
            vectorstore = create_vectorstore(new_chunks, embeddings)

        # Update manifest for new/modified files
        for filename in files_to_add:
            filepath = data_dir / filename
            if filepath.exists():
                file_hash = compute_file_hash(filepath)
                chunk_count = sum(
                    1 for c in new_chunks if c.metadata.get("filename") == filename
                )
                manifest[filename] = {
                    "hash": file_hash,
                    "chunks": chunk_count,
                    "size_bytes": filepath.stat().st_size,
                }
        print()

    # Step 6: Save everything
    print("[Step 6] Saving updated vector store and manifest...")
    print("-" * 40)
    save_vectorstore(vectorstore, VECTORSTORE_DIR)
    save_manifest(manifest)
    print()

    elapsed = time.time() - start_time

    # Summary
    print("=" * 60)
    print("[DONE] Incremental Update Complete!")
    print(f"   New files added: {len(changes['new'])}")
    print(f"   Files updated: {len(changes['modified'])}")
    print(f"   Files removed: {len(changes['deleted'])}")
    print(f"   Total files in manifest: {len(manifest)}")
    print(f"   Time taken: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    update_vectorstore()
