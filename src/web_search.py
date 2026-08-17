"""
RBI-restricted web search module.

Uses DuckDuckGo Search (ddgs) to query only rbi.org.in for the latest
circulars, notifications, and regulatory updates.
"""

from dataclasses import dataclass
from ddgs import DDGS


@dataclass
class WebSearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str


class RBIWebSearcher:
    """
    Search engine restricted to the RBI website (rbi.org.in).

    Uses DuckDuckGo Search (free, no API key required) with
    site: restriction to ensure only official RBI content is returned.
    """

    DOMAIN = "rbi.org.in"
    MAX_RESULTS = 5

    def search(self, query: str, max_results: int = None) -> list[WebSearchResult]:
        """
        Search rbi.org.in for the given query.

        Args:
            query: The user's search query.
            max_results: Maximum number of results to return.

        Returns:
            List of WebSearchResult objects.
        """
        max_results = max_results or self.MAX_RESULTS
        restricted_query = f"site:{self.DOMAIN} {query}"

        try:
            ddgs = DDGS()
            raw_results = list(
                ddgs.text(restricted_query, max_results=max_results)
            )

            results = []
            for r in raw_results:
                results.append(
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", r.get("link", "")),
                        snippet=r.get("body", r.get("snippet", "")),
                    )
                )

            return results

        except Exception as e:
            print(f"[WARN] Web search failed: {e}")
            return []

    def search_as_context(self, query: str, max_results: int = None) -> str:
        """
        Search and format results as a context string for the LLM.

        Args:
            query: The user's search query.
            max_results: Maximum number of results.

        Returns:
            Formatted string of web search results, or empty string if none.
        """
        results = self.search(query, max_results)

        if not results:
            return ""

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[Web Source {i}: {result.title}]\n"
                f"URL: {result.url}\n"
                f"{result.snippet}"
            )

        return "\n\n---\n\n".join(formatted)
