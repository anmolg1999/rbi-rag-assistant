import re

path = r'C:\Users\Anmol Gupta\.gemini\antigravity-ide\brain\28021ffa-9f79-40de-bc82-9ee84f21e20a\.system_generated\steps\134\content.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

links = re.findall(r'<a[^>]*href=[\'\"](https://rbidocs\.rbi\.org\.in/rdocs/notification/PDFs/[^\'\"]+\.PDF)[\'\"][^>]*>.*?alt=[\'\"]PDF - (.*?)[\'\"]', content, re.IGNORECASE)

print(f'Found {len(links)} PDF links.')
for url, title in links:
    if any(kw in title.lower() for kw in ['statutory', 'gold', 'digital lending', 'digital payment']):
        print(f'- {title}\n  {url}')
