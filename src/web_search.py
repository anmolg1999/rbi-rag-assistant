# This module has been deprecated.
# Web search (DuckDuckGo/Tavily) was removed because:
# - DuckDuckGo gets rate-limited on Streamlit Cloud shared IPs
# - Tavily requires an API key from an inaccessible signup page
#
# Replaced by:
# - src/rbi_rates.py — Live rate scraping from rbi.org.in
# - src/router.py — Intent classification for general banking knowledge
# - LLM general knowledge fallback via updated prompts
