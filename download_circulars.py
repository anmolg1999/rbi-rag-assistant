"""
Download additional RBI Master Direction PDFs for the RAG Assistant.

Downloads credit-related Master Directions from the official RBI website
and saves them to the data/ directory.

Usage:
    python download_circulars.py
"""

import os
import time
import hashlib
import requests
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Additional credit-related RBI Master Directions to download
# Format: (filename, URL, human-readable name)
ADDITIONAL_CIRCULARS = [
    {
        "filename": "MD_PSL.pdf",
        "name": "Priority Sector Lending – Targets and Classification",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/128MD66C4DDCB167C4DC9A5BD913570CB3D47.PDF",
        "description": "Covers PSL targets, sub-targets, agriculture, MSME, housing credit allocation",
    },
    {
        "filename": "MD_MSME.pdf",
        "name": "Lending to Micro, Small & Medium Enterprises (MSME) Sector",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/56MD24072017E50D0ED63F9B4414AA756FF0FC72FB66.PDF",
        "description": "MSME credit guidelines, Udyam registration, credit guarantee schemes",
    },
    {
        "filename": "MD_KYC.pdf",
        "name": "Know Your Customer (KYC) Direction",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/169MD.PDF",
        "description": "KYC/AML norms – prerequisite for extending credit",
    },
    {
        "filename": "MD_CRR_SLR.pdf",
        "name": "Cash Reserve Ratio and Statutory Liquidity Ratio",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/150MD.PDF",
        "description": "Statutory restrictions on lending and reserve ratios.",
    },
    {
        "filename": "MD_HousingFinance.pdf",
        "name": "Housing Finance Companies",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/365MD8214102C853241B98E06B2123AB42657.PDF",
        "description": "Housing credit norms, LTV ratios, risk weights for home loans",
    },
    {
        "filename": "MD_InterestRates.pdf",
        "name": "Interest Rates on Advances",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/161MD.PDF",
        "description": "Interest rate regulations on advances and loans.",
    },
    {
        "filename": "MD_DigitalPayments.pdf",
        "name": "Digital Payment Security Controls",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/411MD1918998D1AD74F159D57664FB10E053F.PDF",
        "description": "Digital payment and digital lending security controls.",
    },
]


def download_file(url: str, filepath: Path, retries: int = 3) -> bool:
    """
    Download a file from URL with retry logic.

    Args:
        url: Download URL.
        filepath: Local path to save the file.
        retries: Number of retry attempts.

    Returns:
        True if download succeeded, False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries}...")
            response = requests.get(
                url,
                timeout=60,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            # Verify it's a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
                print(f"  ⚠️  Warning: Response may not be a PDF (Content-Type: {content_type})")

            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(response.content)

            size_kb = len(response.content) / 1024
            print(f"  ✅ Downloaded: {filepath.name} ({size_kb:.0f} KB)")
            return True

        except requests.RequestException as e:
            print(f"  ❌ Attempt {attempt} failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    return False


def get_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def main():
    print("=" * 60)
    print("RBI RAG Assistant — Download Additional Circulars")
    print("=" * 60)
    print()
    print(f"Data directory: {DATA_DIR}")
    print(f"Circulars to download: {len(ADDITIONAL_CIRCULARS)}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for circular in ADDITIONAL_CIRCULARS:
        filepath = DATA_DIR / circular["filename"]
        print(f"\n📄 {circular['name']}")
        print(f"   File: {circular['filename']}")

        # Skip if already downloaded
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"   ⏭️  Already exists ({size_kb:.0f} KB) — skipping")
            skip_count += 1
            continue

        print(f"   URL: {circular['url']}")
        if download_file(circular["url"], filepath):
            success_count += 1
        else:
            fail_count += 1
            print(f"   ❌ FAILED to download: {circular['name']}")

    # Summary
    print()
    print("=" * 60)
    print("[DONE] Download Summary")
    print(f"   ✅ Downloaded: {success_count}")
    print(f"   ⏭️  Skipped (already exist): {skip_count}")
    print(f"   ❌ Failed: {fail_count}")
    print("=" * 60)

    if fail_count > 0:
        print()
        print("⚠️  Some downloads failed. You can:")
        print("   1. Re-run this script to retry failed downloads")
        print("   2. Download manually from https://rbi.org.in and place in data/")
        print()
        print("   Note: RBI may change PDF URLs over time.")
        print("   If URLs are broken, visit https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx")
        print("   to find the latest PDF links.")

    if success_count > 0 or skip_count > 0:
        total_available = len(list(DATA_DIR.glob("*.pdf")))
        print()
        print(f"📚 Total PDFs in data/: {total_available}")
        print("   Next step: Run 'python ingest.py' to rebuild the vector store.")


if __name__ == "__main__":
    main()
