from ddgs import DDGS

queries = [
    '"Loans and Advances" "Statutory and Other Restrictions" site:rbidocs.rbi.org.in/rdocs/notification/PDFs/ filetype:pdf',
    '"Housing Finance" Master Direction site:rbidocs.rbi.org.in/rdocs/notification/PDFs/ filetype:pdf',
    '"Gold Monetisation" OR "Gold Loan" Master Direction site:rbidocs.rbi.org.in/rdocs/notification/PDFs/ filetype:pdf',
    '"Digital Lending" Guidelines site:rbidocs.rbi.org.in/rdocs/notification/PDFs/ filetype:pdf'
]

with DDGS() as ddgs:
    for q in queries:
        print(f'\nSearch: {q}')
        results = list(ddgs.text(q, max_results=2))
        for r in results:
            print(f"- {r['title']}\n  {r['href']}")
