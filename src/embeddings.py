"""
Embeddings module.
Configures HuggingFace sentence-transformer embeddings for the project.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFace embedding model instance.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 by default:
    - 384-dimensional embeddings
    - Fast inference (runs locally on CPU)
    - No API cost
    - Good quality for semantic search
    
    Args:
        model_name: HuggingFace model name for embeddings.
        
    Returns:
        Configured HuggingFaceEmbeddings instance.
    """
    print(f"[INFO] Loading embedding model: {model_name}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print(f"[OK] Embedding model loaded successfully")
    return embeddings
