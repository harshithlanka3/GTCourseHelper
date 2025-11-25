import functools
import logging
from typing import Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:

    logger.debug("Loading SentenceTransformer model (first call or cache miss)")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.debug(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")
    return model


def normalize_query(text: str) -> str:
   
    return text.lower().strip()


@functools.lru_cache(maxsize=512)
def _encode_query_normalized(normalized_text: str) -> Tuple:
    
    logger.debug(f"Encoding query (cache lookup for: '{normalized_text[:50]}...')")
    
    model = _get_embedding_model()
    embedding = model.encode(normalized_text)
    
    embedding_tuple = tuple(embedding.tolist())
    
    logger.debug(f"Query encoded: {len(embedding_tuple)} dimensions")
    return embedding_tuple


def get_query_embedding(text: str) -> np.ndarray:

    normalized = normalize_query(text)
    embedding_tuple = _encode_query_normalized(normalized)
    embedding = np.array(embedding_tuple).reshape(1, -1)
    return embedding


def get_model() -> SentenceTransformer:

    return _get_embedding_model()


def clear_cache():

    _get_embedding_model.cache_clear()
    _encode_query_normalized.cache_clear()
    logger.debug("Embedding caches cleared")


def get_cache_info() -> dict:

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

