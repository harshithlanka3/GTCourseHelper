"""
Shared utilities for embedding model access and query encoding with caching.

This module provides cached access to the SentenceTransformer model to avoid
repeated model loading across multiple function calls. It also caches query
embeddings to avoid recomputing embeddings for identical queries.
"""
import functools
import logging
from typing import Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging for cache monitoring
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """
    Get or create the SentenceTransformer model instance.
    
    This function is cached with maxsize=1 to ensure only one model instance
    is created per process. The model is thread-safe under FastAPI's default
    single-process deployment.
    
    Returns:
        SentenceTransformer: The cached model instance
    """
    logger.debug("Loading SentenceTransformer model (first call or cache miss)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.debug(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    return model


def normalize_query(text: str) -> str:
    """
    Normalize query text for caching purposes.
    
    Args:
        text: The query text to normalize
        
    Returns:
        Normalized query string (lowercase, stripped whitespace)
    """
    return text.lower().strip()


@functools.lru_cache(maxsize=512)
def _encode_query_normalized(normalized_text: str) -> Tuple:
    """
    Internal function to encode normalized query text into an embedding vector.
    
    This function caches embeddings for up to 512 unique normalized queries.
    The cache key is the normalized text itself (lowercase, stripped).
    
    Args:
        normalized_text: The normalized query text to encode (already lowercased and stripped)
        
    Returns:
        Tuple representation of the embedding vector (for hashability)
    """
    logger.debug(f"Encoding query (cache lookup for: '{normalized_text[:50]}...')")
    
    model = _get_embedding_model()
    # Encode the normalized text (sentence transformers handle case differences well)
    embedding = model.encode(normalized_text)
    
    # Convert numpy array to tuple for hashability in cache
    embedding_tuple = tuple(embedding.tolist())
    
    logger.debug(f"Query encoded: {len(embedding_tuple)} dimensions")
    return embedding_tuple


def get_query_embedding(text: str) -> np.ndarray:
    """
    Get query embedding as a numpy array, using cached encoding when possible.
    
    This is the main public function for encoding queries. It normalizes the input
    text (lowercase, stripped) for cache key purposes, then retrieves or computes
    the embedding. This ensures that semantically equivalent queries (differing
    only in case/whitespace) share the same cached embedding.
    
    Args:
        text: The query text to encode (will be normalized for caching)
        
    Returns:
        numpy.ndarray: The embedding vector, reshaped to (1, -1) for similarity calculations
    """
    normalized = normalize_query(text)
    embedding_tuple = _encode_query_normalized(normalized)
    embedding = np.array(embedding_tuple).reshape(1, -1)
    return embedding


def get_model() -> SentenceTransformer:
    """
    Get the cached embedding model instance.
    
    This is a convenience function for direct model access when needed.
    
    Returns:
        SentenceTransformer: The cached model instance
    """
    return _get_embedding_model()


def clear_cache():
    """
    Clear all embedding caches.
    
    Useful for testing or when you need to force a fresh model load.
    """
    _get_embedding_model.cache_clear()
    _encode_query_normalized.cache_clear()
    logger.debug("Embedding caches cleared")


def get_cache_info() -> dict:
    """
    Get cache statistics for monitoring.
    
    Returns:
        dict: Cache hit/miss statistics for model and query caches
    """
    model_info = _get_embedding_model.cache_info()
    query_info = _encode_query_normalized.cache_info()
    
    return {
        "model_cache": {
            "hits": model_info.hits,
            "misses": model_info.misses,
            "current_size": model_info.currsize,
            "max_size": model_info.maxsize
        },
        "query_cache": {
            "hits": query_info.hits,
            "misses": query_info.misses,
            "current_size": query_info.currsize,
            "max_size": query_info.maxsize
        }
    }

