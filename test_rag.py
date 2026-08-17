import time
from src.chain import build_rag_chain, _get_relevance_scores, format_documents
from src.prompts import RAG_PROMPT
from src.config import LLM_MODEL_NAME

def run_tests():
    print(f"Starting Evaluation using model: {LLM_MODEL_NAME}")
    rag_chain, retriever = build_rag_chain()
    # Extract the LLM from the chain
    llm = rag_chain.steps[2] # The ChatGroq instance
    
    test_questions = [
        "What are the Priority Sector Lending (PSL) targets for commercial banks?",
        "Explain the KYC requirements for opening a new bank account.",
        "What is the maximum LTV ratio for housing loans?"
    ]
    
    results = []
    
    for q in test_questions:
        print(f"\nEvaluating: {q}")
        start_time = time.time()
        
        # Retrieve context
        docs_with_scores = _get_relevance_scores(retriever, q)
        source_docs = [doc for doc, _ in docs_with_scores]
        context = format_documents(source_docs)
        
        # Build prompt
        messages = RAG_PROMPT.format_messages(context=context, question=q)
        
        # Invoke LLM
        response = llm.invoke(messages)
        end_time = time.time()
        
        # Extract data
        answer = response.content
        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        token_usage = response.response_metadata.get('token_usage', {})
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        res = {
            "question": q,
            "answer": answer,
            "latency": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "sources": [doc.metadata.get('document_name', 'Unknown') for doc in source_docs]
        }
        results.append(res)
        print(f"  Tokens: {total_tokens} (In: {prompt_tokens}, Out: {completion_tokens})")
        print(f"  Latency: {res['latency']:.2f}s")

    # Generate markdown report
    with open("test_results.md", "w", encoding="utf-8") as f:
        f.write("# RAG System Evaluation Results\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Test Case {i}\n")
            f.write(f"**Question:** {r['question']}\n\n")
            f.write(f"**Answer:**\n{r['answer']}\n\n")
            f.write(f"**Sources Retrieved:** {', '.join(set(r['sources']))}\n\n")
            f.write("### Metrics\n")
            f.write(f"- **Latency:** {r['latency']:.2f} seconds\n")
            f.write(f"- **Prompt Tokens:** {r['prompt_tokens']}\n")
            f.write(f"- **Completion Tokens:** {r['completion_tokens']}\n")
            f.write(f"- **Total Tokens:** {r['total_tokens']}\n")
            f.write("---\n\n")
            
    print("\nResults saved to test_results.md")

if __name__ == '__main__':
    run_tests()
