"""
embedding_engine.py - Course Embedding and Similarity Search Module

This module handles the conversion of text (course descriptions) into numerical vectors
called "embeddings". Think of embeddings as a way to represent the "meaning" of text
as numbers that computers can compare.

Analogy: If courses were people, embeddings would be like their GPS coordinates.
Just as you can find people near you by comparing GPS coordinates, we can find
similar courses by comparing their embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseEmbeddingEngine:
    """
    Manages the creation and searching of course embeddings.
    
    Key concepts:
    1. Embeddings: Numerical representations of text (like coordinates for meaning)
    2. Sentence Transformers: AI models that create these embeddings
    3. FAISS: Facebook's library for fast similarity search
    4. Cosine Similarity: How we measure "closeness" between embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Which sentence transformer model to use.
                       'all-MiniLM-L6-v2' is fast and good for semantic search
                       'all-mpnet-base-v2' is slower but more accurate
                       
        The model downloads automatically from HuggingFace on first use (~80MB).
        """
        logger.info(f"Initializing embedding engine with model: {model_name}")
        
        # Load the sentence transformer model
        # This model converts text -> embeddings (numbers)
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension (usually 384 for MiniLM, 768 for MPNet)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # These will store our embeddings and search index
        self.embeddings = None  # numpy array of all course embeddings
        self.index = None       # FAISS index for fast search
        self.course_ids = []    # List to map index position to course ID
        
    def generate_embeddings(self, courses_df: pd.DataFrame, 
                          text_column: str = 'searchable_text',
                          batch_size: int = 32,
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate embeddings for all courses in the DataFrame.
        
        This is the core function that converts course text into numerical vectors.
        It processes courses in batches for efficiency.
        
        Args:
            courses_df: DataFrame with course data
            text_column: Which column contains the text to embed
            batch_size: How many courses to process at once (higher = faster but more memory)
            save_path: Optional path to save embeddings for reuse
            
        Returns:
            numpy array of embeddings, shape (n_courses, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(courses_df)} courses...")
        
        # Prepare the texts for embedding
        texts = []
        for idx, row in courses_df.iterrows():
            # Create a rich text representation for better embeddings
            # The more context we give, the better the embedding quality
            text_parts = []
            
            # Add course number and name (weighted heavily)
            if pd.notna(row.get('course_number')):
                text_parts.append(f"Course: {row['course_number']}")
            if pd.notna(row.get('course_name')):
                text_parts.append(f"Title: {row['course_name']}")
            
            # Add description (main content)
            if pd.notna(row.get('description')):
                text_parts.append(f"Description: {row['description']}")
            
            # Add professor (helps with professor-specific searches)
            if pd.notna(row.get('professor')):
                text_parts.append(f"Instructor: {row['professor']}")
            
            # Add prerequisites (important for understanding course level)
            if 'prerequisites' in row and row['prerequisites']:
                prereqs = row['prerequisites']
                if isinstance(prereqs, list) and prereqs:
                    text_parts.append(f"Prerequisites: {', '.join(prereqs)}")
            
            # Add level indicator
            if pd.notna(row.get('level')):
                level_desc = {
                    1000: "Introductory undergraduate",
                    2000: "Intermediate undergraduate", 
                    3000: "Advanced undergraduate",
                    4000: "Senior undergraduate",
                    6000: "Graduate level",
                    7000: "Advanced graduate",
                    8000: "Doctoral level"
                }
                text_parts.append(level_desc.get(row['level'], f"Level {row['level']}"))
            
            # Combine all parts into final text
            full_text = " | ".join(text_parts)
            texts.append(full_text)
        
        # Generate embeddings in batches with progress bar
        logger.info("Encoding texts into embeddings...")
        
        # Show progress bar if tqdm is available
        try:
            from tqdm import tqdm
            texts_with_progress = tqdm(texts, desc="Generating embeddings")
        except ImportError:
            texts_with_progress = texts
            logger.info("Install tqdm for progress bars: pip install tqdm")
        
        # Convert texts to embeddings
        # This is where the AI magic happens - text becomes numbers!
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        # This makes all vectors have length 1, so dot product = cosine similarity
        self.embeddings = self._normalize_embeddings(self.embeddings)
        
        # Store course IDs for mapping results back
        self.course_ids = courses_df.index.tolist()
        
        # Build the search index
        self._build_faiss_index()
        
        # Optionally save embeddings for later use
        if save_path:
            self.save_embeddings(save_path)
            
        logger.info(f"Generated {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
        return self.embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length for cosine similarity.
        
        Math explanation: When vectors have length 1, their dot product equals
        the cosine of the angle between them, which measures similarity.
        
        Args:
            embeddings: Raw embeddings from the model
            
        Returns:
            Normalized embeddings with unit length
        """
        # Calculate the length (norm) of each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        # Divide each embedding by its norm
        return embeddings / norms
    
    def _build_faiss_index(self):
        """
        Build a FAISS index for fast similarity search.
        
        FAISS (Facebook AI Similarity Search) is a library that makes
        searching through millions of embeddings super fast.
        
        Think of it like building a smart phone book that can instantly
        find similar entries instead of exact matches.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to index! Run generate_embeddings first.")
        
        logger.info("Building FAISS index for fast search...")
        
        # Create index for Inner Product (equivalent to cosine similarity when normalized)
        # IndexFlatIP = "Index Flat Inner Product"
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add all embeddings to the index
        self.index.add(self.embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search_similar_courses(self, 
                              query_text: Optional[str] = None,
                              query_embedding: Optional[np.ndarray] = None,
                              k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Find the k most similar courses to a query.
        
        You can search by providing either:
        1. query_text: A string that will be converted to an embedding
        2. query_embedding: A pre-computed embedding vector
        
        Args:
            query_text: Text to search for (e.g., "machine learning with Python")
            query_embedding: Pre-computed embedding vector
            k: Number of similar courses to return
            
        Returns:
            Tuple of (course_indices, similarity_scores)
            - course_indices: List of indices into the original DataFrame
            - similarity_scores: List of similarity scores (0-1, higher is better)
        """
        if query_text is None and query_embedding is None:
            raise ValueError("Provide either query_text or query_embedding")
        
        if self.index is None:
            raise ValueError("No index built! Run generate_embeddings first.")
        
        # Convert text to embedding if needed
        if query_text is not None:
            logger.debug(f"Searching for: {query_text[:100]}...")
            query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        
        # Ensure query embedding is the right shape and normalized
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search the index
        # Returns distances (similarity scores) and indices of nearest neighbors
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert FAISS indices back to course IDs
        course_indices = [self.course_ids[idx] for idx in indices[0]]
        similarity_scores = distances[0].tolist()
        
        return course_indices, similarity_scores
    
    def generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate an embedding for a search query.
        
        This is useful when you want to generate the embedding once
        and use it multiple times or modify it before searching.
        
        Args:
            query_text: The search query
            
        Returns:
            Normalized embedding vector
        """
        embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        return self._normalize_embeddings(embedding).squeeze()
    
    def find_similar_between_courses(self, course_idx: int, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Find courses similar to a specific course (useful for "related courses" features).
        
        Args:
            course_idx: Index of the course to find similar courses for
            k: Number of similar courses to return (will return k+1 including itself)
            
        Returns:
            Tuple of (course_indices, similarity_scores)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings generated!")
        
        # Get the embedding of the target course
        course_embedding = self.embeddings[course_idx].reshape(1, -1)
        
        # Search for similar courses (will include itself)
        distances, indices = self.index.search(course_embedding, k + 1)
        
        # Remove the course itself from results (it will be the first result)
        course_indices = [self.course_ids[idx] for idx in indices[0][1:]]
        similarity_scores = distances[0][1:].tolist()
        
        return course_indices, similarity_scores
    
    def save_embeddings(self, save_path: str):
        """
        Save embeddings and related data to disk for later use.
        
        This avoids having to regenerate embeddings every time you restart the system.
        
        Args:
            save_path: Where to save the embeddings (e.g., "data/embeddings/course_embeddings.pkl")
        """
        logger.info(f"Saving embeddings to {save_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save all necessary data
        save_data = {
            'embeddings': self.embeddings,
            'course_ids': self.course_ids,
            'embedding_dim': self.embedding_dim,
            'model_name': self.encoder.get_sentence_embedding_dimension()  # For verification
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save the FAISS index separately for faster loading
        index_path = save_path.replace('.pkl', '.faiss')
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        
        logger.info(f"Saved {len(self.course_ids)} embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str):
        """
        Load previously saved embeddings from disk.
        
        Args:
            load_path: Path to the saved embeddings file
        """
        logger.info(f"Loading embeddings from {load_path}...")
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.embeddings = save_data['embeddings']
        self.course_ids = save_data['course_ids']
        self.embedding_dim = save_data['embedding_dim']
        
        # Try to load FAISS index
        index_path = load_path.replace('.pkl', '.faiss')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Rebuild index if not found
            self._build_faiss_index()
        
        logger.info(f"Loaded {len(self.course_ids)} embeddings")
    
    def compute_similarity_matrix(self, course_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute pairwise similarity between courses (useful for analysis).
        
        Warning: This creates an n×n matrix, so it can use lots of memory for many courses!
        
        Args:
            course_indices: Specific courses to compare (None = all courses)
            
        Returns:
            Similarity matrix where element [i,j] is similarity between course i and j
        """
        if self.embeddings is None:
            raise ValueError("No embeddings generated!")
        
        if course_indices is None:
            embeddings_subset = self.embeddings
        else:
            embeddings_subset = self.embeddings[course_indices]
        
        # Compute cosine similarity matrix
        # Since embeddings are normalized, this is just the dot product
        similarity_matrix = np.dot(embeddings_subset, embeddings_subset.T)
        
        return similarity_matrix


# Example usage and testing
if __name__ == "__main__":
    # This runs when you execute the file directly
    # Useful for testing the embedding engine independently
    
    from data_processor import CourseDataProcessor
    
    # Load some course data
    print("Loading course data...")
    processor = CourseDataProcessor('data/raw/gt_courses.json')
    courses_df = processor.courses_df
    
    # Initialize embedding engine
    print("\nInitializing embedding engine...")
    engine = CourseEmbeddingEngine(model_name='all-MiniLM-L6-v2')
    
    # Generate embeddings for all courses
    print("\nGenerating embeddings...")
    embeddings = engine.generate_embeddings(
        courses_df, 
        save_path='data/embeddings/course_embeddings.pkl'
    )
    
    # Test search functionality
    test_queries = [
        "introduction to machine learning",
        "web development with JavaScript",
        "calculus and differential equations",
        "Professor Smith morning classes"
    ]
    
    print("\n=== Search Tests ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        course_indices, scores = engine.search_similar_courses(query, k=3)
        
        for idx, score in zip(course_indices, scores):
            course = courses_df.loc[idx]
            print(f"  [{score:.3f}] {course['course_number']}: {course['course_name']}")
    
    # Find similar courses to a specific course
    print("\n=== Related Courses Test ===")
    test_course_idx = 0
    test_course = courses_df.iloc[test_course_idx]
    print(f"Finding courses similar to: {test_course['course_number']} - {test_course['course_name']}")
    
    similar_indices, similar_scores = engine.find_similar_between_courses(test_course_idx, k=3)
    for idx, score in zip(similar_indices, similar_scores):
        course = courses_df.loc[idx]
        print(f"  [{score:.3f}] {course['course_number']}: {course['course_name']}")
