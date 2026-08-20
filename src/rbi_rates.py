"""
Live RBI rate scraper.

Fetches current key policy rates (Repo Rate, CRR, SLR, etc.) directly
from the official RBI National Summary Data Page (NSDP).
Uses caching to avoid repeated requests — rates change infrequently.
"""

import re
import time
import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────
_cache = {
    "rates": None,
    "last_fetched": 0,
}
CACHE_TTL = 3600  # 1 hour

# The rates we care about and their display names
RATE_KEYS = [
    "Policy Repo Rate",
    "Fixed Reverse Repo Rate",
    "Standing Deposit Facility (SDF) Rate",
    "Marginal Standing Facility (MSF) Rate",
    "Bank Rate",
    "Cash Reserve Ratio",
    "Statutory Liquidity Ratio",
    "Base Rate",
    "MCLR (Overnight)",
    "Savings Deposit Rate",
]

RBI_NSDP_URL = "https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4"


def fetch_current_rates() -> dict | None:
    """
    Fetch current RBI key rates from the official NSDP page.

    Returns:
        Dict mapping rate name → latest value (as string, e.g. "5.25"),
        plus a "data_date" key with the page's publication date.
        Returns None if scraping fails.
    """
    now = time.time()

    # Return cached data if fresh
    if _cache["rates"] and (now - _cache["last_fetched"]) < CACHE_TTL:
        return _cache["rates"]

    try:
        response = requests.get(
            RBI_NSDP_URL,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
        return _parse_rates_html(response.text)

    except Exception as e:
        print(f"[WARN] Failed to fetch RBI rates: {e}")
        # Return stale cache if available
        if _cache["rates"]:
            return _cache["rates"]
        return None


def _parse_rates_html(html: str) -> dict | None:
    """
    Parse the RBI NSDP HTML page to extract key rates.

    The page embeds rate data inside a large HTML table within a hidden
    __VIEWSTATE field (base64-encoded) AND as rendered HTML. We parse
    whichever is accessible.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Try to extract from rendered tables first
        rates = {}

        # Look for the data date
        data_date = ""
        bold_tags = soup.find_all("b")
        for b in bold_tags:
            text = b.get_text(strip=True)
            if "Date" in text:
                data_date = text.replace("Date :", "").replace("Date:", "").strip()
                break

        # Find all tables and look for rate data
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    # Get the LAST column value (most recent data)
                    value = cells[-1].get_text(strip=True)

                    for rate_key in RATE_KEYS:
                        if rate_key.lower() in label.lower():
                            if value and value.replace(".", "").replace("/", "").isdigit():
                                rates[rate_key] = value
                            break

        # If we didn't get rates from rendered HTML, try parsing __VIEWSTATE
        if len(rates) < 3:
            viewstate = soup.find("input", {"id": "__VIEWSTATE"})
            if viewstate:
                vs_value = viewstate.get("value", "")
                rates = _parse_rates_from_viewstate(vs_value, rates)

        if not rates:
            return None

        rates["data_date"] = data_date or "Latest available"
        rates["source"] = "Reserve Bank of India (rbi.org.in)"

        # Update cache
        _cache["rates"] = rates
        _cache["last_fetched"] = time.time()

        print(f"[OK] Fetched {len(rates) - 2} RBI rates (as of {data_date})")
        return rates

    except Exception as e:
        print(f"[WARN] Failed to parse RBI rates HTML: {e}")
        return None


def _parse_rates_from_viewstate(vs_value: str, existing_rates: dict) -> dict:
    """
    Parse rates from the __VIEWSTATE field which contains embedded HTML.
    The VIEWSTATE contains the table HTML as a long encoded string.
    """
    rates = existing_rates.copy()

    try:
        # The viewstate contains HTML table data as text
        # Extract rate values using regex patterns on the raw string
        rate_patterns = {
            "Policy Repo Rate": r"Policy Repo Rate</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Cash Reserve Ratio": r"Cash Reserve Ratio</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Statutory Liquidity Ratio": r"Statutory Liquidity Ratio</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Fixed Reverse Repo Rate": r"Fixed Reverse Repo Rate</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Standing Deposit Facility (SDF) Rate": r"Standing Deposit Facility \(SDF\) Rate[^<]*</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Marginal Standing Facility (MSF) Rate": r"Marginal Standing Facility \(MSF\) Rate</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Bank Rate": r"Bank Rate</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
            "Base Rate": r"Base Rate</td>\s*(?:<td[^>]*>[\d./]+</td>\s*)*<td[^>]*>([\d./]+)</td>",
            "MCLR (Overnight)": r"MCLR \(Overnight\)</td>\s*(?:<td[^>]*>[\d./]+</td>\s*)*<td[^>]*>([\d./]+)</td>",
            "Savings Deposit Rate": r"Savings Deposit Rate</td>\s*(?:<td[^>]*>[\d.]+</td>\s*)*<td[^>]*>([\d.]+)</td>",
        }

        for rate_name, pattern in rate_patterns.items():
            if rate_name not in rates:
                match = re.search(pattern, vs_value, re.IGNORECASE)
                if match:
                    rates[rate_name] = match.group(1)

    except Exception as e:
        print(f"[WARN] Failed to parse viewstate: {e}")

    return rates


def format_rates_for_llm(rates: dict) -> str:
    """
    Format the scraped rates into a clean context string for the LLM.
    """
    if not rates:
        return ""

    lines = [
        f"Data as of: {rates.get('data_date', 'Latest available')}",
        f"Source: {rates.get('source', 'RBI')}",
        "",
        "Current RBI Key Rates and Ratios (in per cent):",
    ]

    for key in RATE_KEYS:
        if key in rates:
            lines.append(f"  • {key}: {rates[key]}%")

    return "\n".join(lines)


def is_rate_question(question: str) -> bool:
    """
    Check if the user's question is about current RBI rates.
    Uses keyword matching for speed.
    """
    q = question.lower()

    rate_keywords = [
        "repo rate", "reverse repo", "crr", "slr",
        "cash reserve ratio", "statutory liquidity ratio",
        "bank rate", "msf rate", "marginal standing",
        "standing deposit", "sdf rate", "base rate",
        "mclr", "policy rate", "interest rate",
        "current rate", "latest rate", "rbi rate",
        "savings deposit rate", "savings rate",
    ]

    # Must also have an indicator that they want the *current* value
    current_keywords = [
        "current", "latest", "today", "now", "present",
        "what is", "what are", "how much", "tell me",
        "rate", "ratio",
    ]

    has_rate = any(kw in q for kw in rate_keywords)
    has_current = any(kw in q for kw in current_keywords)

    return has_rate and has_current
