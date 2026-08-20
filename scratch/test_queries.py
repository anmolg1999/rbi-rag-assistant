import os
from dotenv import load_dotenv
load_dotenv()

from src.chain import query_with_sources

test_queries = [
    "What is devolvement of LC?",
    "Explain invocation of Bank Guarantee in detail",
    "What can you do?",
    "What is the current repo rate?",
    "Tell me a joke",
    "What are the IRAC norms for asset classification?"
]

for q in test_queries:
    print(f"\n--- QUERY: {q} ---")
    res = query_with_sources(q, chat_history=[])
    print(f"SOURCE TYPE: {res['source_type']}")
    print(f"ANSWER: {res['answer'][:150]}...")
