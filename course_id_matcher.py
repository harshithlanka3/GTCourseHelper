"""
Course ID Matching Module

Extracts course IDs from user queries and performs exact/fuzzy matching
to enhance semantic search results.
"""
import re
from typing import List, Tuple, Set
from difflib import SequenceMatcher
import pandas as pd


def extract_course_ids(query: str) -> List[str]:
    """
    Extract course IDs from user query using regex patterns.
    
    Supports formats:
    - "CS 1332" (space)
    - "CS1332" (no space)
    - "CS-1332" (dash)
    - "cs 1332" (case insensitive)
    
    Args:
        query (str): User query string
        
    Returns:
        List[str]: List of extracted course IDs in format "PREFIX NUMBER"
    """
    # Pattern: 2-4 letter prefix + optional space/dash + 4 digits
    pattern = r'\b([A-Z]{2,4})[\s-]?(\d{4})\b'
    matches = re.findall(pattern, query.upper())
    course_ids = [f"{prefix} {num}" for prefix, num in matches]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for course_id in course_ids:
        if course_id not in seen:
            seen.add(course_id)
            unique_ids.append(course_id)
    
    return unique_ids


def fuzzy_match_course_id(
    query_id: str, 
    available_ids: List[str], 
    threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """
    Fuzzy match a course ID against available courses.
    
    Args:
        query_id (str): Course ID to match (e.g., "CS 1332")
        available_ids (List[str]): List of available course IDs
        threshold (float): Minimum similarity score (0-1)
        
    Returns:
        List[Tuple[str, float]]: List of (course_id, similarity_score) tuples,
                                 sorted by similarity (highest first)
    """
    matches = []
    query_upper = query_id.upper()
    
    for course_id in available_ids:
        course_upper = course_id.upper()
        
        # Exact match gets highest score
        if query_upper == course_upper:
            matches.append((course_id, 1.0))
            continue
        
        # Calculate similarity
        similarity = SequenceMatcher(None, query_upper, course_upper).ratio()
        
        if similarity >= threshold:
            matches.append((course_id, similarity))
    
    # Sort by similarity (highest first)
    return sorted(matches, key=lambda x: x[1], reverse=True)


def hybrid_search(
    user_query: str, 
    df: pd.DataFrame, 
    top_k: int = 50,
    boost_factor: float = 1.5,
    fuzzy_threshold: float = 0.8
) -> pd.DataFrame:
    """
    Combine exact/fuzzy course ID matching with semantic search.
    
    Strategy:
    1. Extract course IDs from query
    2. Find exact matches in dataframe
    3. Perform semantic search (caller should handle this)
    4. Boost scores for mentioned courses
    5. Combine results with exact matches prioritized
    
    Args:
        user_query (str): User's search query
        df (pd.DataFrame): Course dataframe
        top_k (int): Number of results to return
        boost_factor (float): Multiplier for mentioned course scores
        fuzzy_threshold (float): Minimum similarity for fuzzy matching
        
    Returns:
        pd.DataFrame: Combined search results with similarity scores
    """
    # Extract course IDs from query
    mentioned_ids = extract_course_ids(user_query)
    
    if not mentioned_ids:
        # No course IDs found, return empty dataframe
        # (semantic search will be handled by caller)
        return pd.DataFrame()
    
    # Find exact matches
    exact_matches = df[df['course_id'].isin(mentioned_ids)].copy()
    
    # Find fuzzy matches for IDs not found exactly
    found_exact = set(exact_matches['course_id'].tolist())
    all_course_ids = df['course_id'].unique().tolist()
    
    fuzzy_matches = []
    for query_id in mentioned_ids:
        if query_id not in found_exact:
            # Try fuzzy matching
            matches = fuzzy_match_course_id(query_id, all_course_ids, fuzzy_threshold)
            if matches:
                # Take the best match
                best_match_id, similarity = matches[0]
                if similarity >= fuzzy_threshold:
                    fuzzy_matches.append(best_match_id)
    
    # Combine exact and fuzzy matches
    all_matched_ids = set(mentioned_ids) | set(fuzzy_matches)
    matched_courses = df[df['course_id'].isin(all_matched_ids)].copy()
    
    # Add match type indicator
    matched_courses['match_type'] = matched_courses['course_id'].apply(
        lambda x: 'exact' if x in mentioned_ids else 'fuzzy'
    )
    
    return matched_courses


def enhance_search_results_with_ids(
    user_query: str,
    semantic_results: pd.DataFrame,
    df: pd.DataFrame,
    boost_factor: float = 1.5
) -> pd.DataFrame:
    """
    Enhance semantic search results by boosting courses mentioned by ID.
    
    Args:
        user_query (str): Original user query
        semantic_results (pd.DataFrame): Results from semantic search
        df (pd.DataFrame): Full course dataframe
        boost_factor (float): Score multiplier for mentioned courses
        
    Returns:
        pd.DataFrame: Enhanced results with boosted scores
    """
    mentioned_ids = extract_course_ids(user_query)
    
    if not mentioned_ids or len(semantic_results) == 0:
        return semantic_results
    
    # Create a copy to avoid modifying original
    enhanced = semantic_results.copy()
    
    # Boost scores for mentioned courses
    for idx in enhanced.index:
        course_id = enhanced.loc[idx, 'course_id']
        if course_id in mentioned_ids:
            # Boost the similarity score
            enhanced.loc[idx, 'similarity_score'] *= boost_factor
            # Mark as mentioned
            if 'match_type' not in enhanced.columns:
                enhanced['match_type'] = 'semantic'
            enhanced.loc[idx, 'match_type'] = 'mentioned'
    
    # Re-sort by boosted scores
    enhanced = enhanced.sort_values('similarity_score', ascending=False)
    
    return enhanced


def combine_id_and_semantic_results(
    id_results: pd.DataFrame,
    semantic_results: pd.DataFrame,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Combine course ID matches with semantic search results.
    
    Strategy:
    - Exact ID matches first (highest priority)
    - Fuzzy ID matches next
    - Semantic results (excluding already matched courses)
    - Limit to top_k total results
    
    Args:
        id_results (pd.DataFrame): Results from ID matching
        semantic_results (pd.DataFrame): Results from semantic search
        top_k (int): Total number of results to return
        
    Returns:
        pd.DataFrame: Combined and deduplicated results
    """
    if len(id_results) == 0:
        return semantic_results.head(top_k)
    
    if len(semantic_results) == 0:
        return id_results.head(top_k)
    
    # Get course IDs already matched by ID
    matched_ids = set(id_results['course_id'].tolist())
    
    # Filter semantic results to exclude already matched courses
    semantic_filtered = semantic_results[
        ~semantic_results['course_id'].isin(matched_ids)
    ].copy()
    
    # Combine: ID matches first, then semantic
    combined = pd.concat([id_results, semantic_filtered], ignore_index=True)
    
    # Remove duplicates (keep first occurrence, which will be ID matches)
    combined = combined.drop_duplicates(subset=['course_id'], keep='first')
    
    # Sort: exact matches first, then by similarity score
    if 'match_type' in combined.columns:
        # Create sort key: exact > fuzzy > mentioned > semantic
        type_order = {'exact': 0, 'fuzzy': 1, 'mentioned': 2, 'semantic': 3}
        combined['sort_key'] = combined['match_type'].map(type_order).fillna(3)
        combined = combined.sort_values(['sort_key', 'similarity_score'], ascending=[True, False])
        combined = combined.drop('sort_key', axis=1)
    else:
        combined = combined.sort_values('similarity_score', ascending=False)
    
    return combined.head(top_k)

