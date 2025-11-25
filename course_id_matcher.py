import re
from typing import List, Tuple, Set
from difflib import SequenceMatcher
import pandas as pd


def extract_course_ids(query: str) -> List[str]:
    """
    Args:
        query (str): User query string
        
    Returns:
        List[str]: List of extracted course IDs in format "PREFIX NUMBER"
    """
    pattern = r'\b([A-Z]{2,4})[\s-]?(\d{4})\b'
    matches = re.findall(pattern, query.upper())
    course_ids = [f"{prefix} {num}" for prefix, num in matches]
    
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
    matches = []
    query_upper = query_id.upper()
    
    for course_id in available_ids:
        course_upper = course_id.upper()
        
        if query_upper == course_upper:
            matches.append((course_id, 1.0))
            continue
        
        similarity = SequenceMatcher(None, query_upper, course_upper).ratio()
        
        if similarity >= threshold:
            matches.append((course_id, similarity))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)


def hybrid_search(
    user_query: str, 
    df: pd.DataFrame, 
    top_k: int = 50,
    boost_factor: float = 1.5,
    fuzzy_threshold: float = 0.8
) -> pd.DataFrame:
    mentioned_ids = extract_course_ids(user_query)
    
    if not mentioned_ids:

        return pd.DataFrame()
    
    exact_matches = df[df['course_id'].isin(mentioned_ids)].copy()
    
    found_exact = set(exact_matches['course_id'].tolist())
    all_course_ids = df['course_id'].unique().tolist()
    
    fuzzy_matches = []
    for query_id in mentioned_ids:
        if query_id not in found_exact:
            matches = fuzzy_match_course_id(query_id, all_course_ids, fuzzy_threshold)
            if matches:
                best_match_id, similarity = matches[0]
                if similarity >= fuzzy_threshold:
                    fuzzy_matches.append(best_match_id)
    
    all_matched_ids = set(mentioned_ids) | set(fuzzy_matches)
    matched_courses = df[df['course_id'].isin(all_matched_ids)].copy()
    
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
    mentioned_ids = extract_course_ids(user_query)
    
    if not mentioned_ids or len(semantic_results) == 0:
        return semantic_results
    
    enhanced = semantic_results.copy()
    
    for idx in enhanced.index:
        course_id = enhanced.loc[idx, 'course_id']
        if course_id in mentioned_ids:
            enhanced.loc[idx, 'similarity_score'] *= boost_factor
            if 'match_type' not in enhanced.columns:
                enhanced['match_type'] = 'semantic'
            enhanced.loc[idx, 'match_type'] = 'mentioned'
    
    enhanced = enhanced.sort_values('similarity_score', ascending=False)
    
    return enhanced


def combine_id_and_semantic_results(
    id_results: pd.DataFrame,
    semantic_results: pd.DataFrame,
    top_k: int = 50
) -> pd.DataFrame:

    if len(id_results) == 0:
        return semantic_results.head(top_k)
    
    if len(semantic_results) == 0:
        return id_results.head(top_k)
    
    matched_ids = set(id_results['course_id'].tolist())
    
    semantic_filtered = semantic_results[
        ~semantic_results['course_id'].isin(matched_ids)
    ].copy()
    
    combined = pd.concat([id_results, semantic_filtered], ignore_index=True)
    
    combined = combined.drop_duplicates(subset=['course_id'], keep='first')
    
    if 'match_type' in combined.columns:
        type_order = {'exact': 0, 'fuzzy': 1, 'mentioned': 2, 'semantic': 3}
        combined['sort_key'] = combined['match_type'].map(type_order).fillna(3)
        combined = combined.sort_values(['sort_key', 'similarity_score'], ascending=[True, False])
        combined = combined.drop('sort_key', axis=1)
    else:
        combined = combined.sort_values('similarity_score', ascending=False)
    
    return combined.head(top_k)

