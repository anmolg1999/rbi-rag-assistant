"""
Intent router for the RBI RAG Assistant.

Classifies user queries into three intents:
- "meta": Questions about the chatbot itself (capabilities, features, etc.)
- "rag": RBI-specific regulatory questions (best answered from indexed documents)
- "general_banking": General banking/finance questions (LLM knowledge + optional rate scraping)

Uses fast keyword matching first, falls back to LLM classification for ambiguous cases.
"""

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import GROQ_API_KEY, LLM_MODEL_NAME
from src.rbi_rates import is_rate_question


# ──────────────────────────────────────────────
# Keyword-based fast classification
# ──────────────────────────────────────────────

META_KEYWORDS = [
    "what can you do", "what do you do", "what does you do", "who are you", 
    "your features", "about you", "your capabilities", "what are you", 
    "who built you", "who made you", "how do you work", "what is this app", 
    "what is this bot", "what is this chatbot", "your purpose", "help me", 
    "how to use", "what questions", "what topics", "introduce yourself",
    "tell me about yourself", "describe yourself", "your functionality",
    "what is the functionality of you", "what kind of questions", "what can i ask",
]

# Keywords that strongly indicate an out-of-scope / non-banking question
OUT_OF_SCOPE_KEYWORDS = [
    "weather", "recipe", "movie", "sport", "cricket", "football",
    "song", "music", "joke", "poem", "story", "game",
    "travel", "flight", "hotel", "restaurant",
]

GENERAL_BANKING_KEYWORDS = [
    "what is repo rate", "what is crr", "what is slr",
    "what is npa", "what is bank rate", "what is msf",
    "what is mclr", "what is base rate", "what is rbi",
    "what is reserve bank", "what is monetary policy",
    "what is fiscal policy", "what is inflation",
    "what is gdp", "what is priority sector",
    "what is nbfc", "what is banking", "what is credit",
    "explain repo", "explain crr", "explain slr",
    "explain npa", "define repo", "define crr", "define slr",
    "meaning of repo", "meaning of crr", "meaning of slr",
    "difference between", "how does rbi", "why does rbi",
    "role of rbi", "functions of rbi",
    "current repo", "current crr", "current slr", "latest rate",
    "current rate", "today rate", "jaiib", "caiib", "iibf",
    "credit related", "general banking"
]


def classify_intent(question: str) -> str:
    """
    Classify the user's question intent.
    Uses LLM classification for robust intent detection.
    """
    q = question.lower().strip()
    
    # 1. Fast path for obvious meta questions
    for kw in META_KEYWORDS:
        if kw in q:
            return "meta"
            
    # 2. Fast path for obvious rate questions
    if is_rate_question(question):
        return "general_banking"

    # 3. Use LLM to classify the intent accurately
    return classify_intent_with_llm(question)


def classify_intent_with_llm(question: str) -> str:
    """
    Use LLM to classify ambiguous questions.
    """
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier for an RBI banking regulation chatbot.
Classify the user's question into exactly ONE of these categories:

- "meta": The user is asking about the chatbot itself (e.g. "what do you do", "what is your functionality", "who made you", "what can I ask you")
- "rag": The user has a specific question about RBI regulations, Master Directions, or circulars that would be best answered from official indexed documents
- "general_banking": The user has a general banking/finance question, a credit-related question, or a topic from the IIBF JAIIB/CAIIB syllabus. This category doesn't need specific circular text and can rely on broad banking knowledge.
- "out_of_scope": The question is completely unrelated to banking, finance, or RBI

Respond with ONLY the category name in lowercase, nothing else."""),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0,
        max_tokens=10,
    )

    chain = classifier_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().lower().strip('"\'')

    valid_intents = {"meta", "rag", "general_banking", "out_of_scope"}
    return result if result in valid_intents else "rag"
