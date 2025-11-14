"""
Comparison and metrics module for pipeline permutation experiments.

Provides functions to:
- Calculate quantitative metrics (Jaccard similarity, Spearman correlation, etc.)
- Display side-by-side comparison of results
- Generate comparison reports
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        float: Jaccard similarity (0-1), or 0 if both sets are empty
    """
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def calculate_rank_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[float, int]:
    """
    Calculate Spearman rank correlation for courses appearing in both result sets.
    
    Args:
        df1: First result DataFrame (must have 'course_id' and 'similarity_score' columns)
        df2: Second result DataFrame (must have 'course_id' and 'similarity_score' columns)
        
    Returns:
        tuple: (correlation coefficient, number of common courses)
               Returns (0.0, 0) if no common courses or insufficient data
    """
    if len(df1) == 0 or len(df2) == 0:
        return (0.0, 0)
    
    # Get course IDs and their ranks in each DataFrame
    df1_ranks = {course_id: rank for rank, course_id in enumerate(df1['course_id'].tolist(), 1)}
    df2_ranks = {course_id: rank for rank, course_id in enumerate(df2['course_id'].tolist(), 1)}
    
    # Find common courses
    common_courses = set(df1_ranks.keys()) & set(df2_ranks.keys())
    
    if len(common_courses) < 2:
        return (0.0, len(common_courses))
    
    # Get ranks for common courses
    ranks1 = [df1_ranks[course_id] for course_id in common_courses]
    ranks2 = [df2_ranks[course_id] for course_id in common_courses]
    
    # Calculate Spearman correlation
    correlation, _ = spearmanr(ranks1, ranks2)
    
    # Handle NaN (can occur if ranks are identical)
    if np.isnan(correlation):
        correlation = 1.0 if ranks1 == ranks2 else 0.0
    
    return (correlation, len(common_courses))


def calculate_coverage(expected_courses: Optional[List[str]], results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate how many expected courses appear in results.
    
    Args:
        expected_courses: List of expected course IDs (can be None)
        results: Result DataFrame with 'course_id' column
        
    Returns:
        dict: Dictionary with 'found_count', 'total_expected', and 'coverage_ratio'
    """
    if expected_courses is None or len(expected_courses) == 0:
        return {
            "found_count": 0,
            "total_expected": 0,
            "coverage_ratio": 0.0
        }
    
    if len(results) == 0:
        return {
            "found_count": 0,
            "total_expected": len(expected_courses),
            "coverage_ratio": 0.0
        }
    
    result_course_ids = set(results['course_id'].tolist())
    expected_set = set(expected_courses)
    found_count = len(result_course_ids & expected_set)
    
    return {
        "found_count": found_count,
        "total_expected": len(expected_courses),
        "coverage_ratio": found_count / len(expected_courses) if expected_courses else 0.0
    }


def calculate_metrics(
    embedding_results: pd.DataFrame,
    id_results: pd.DataFrame,
    combined_results: pd.DataFrame,
    expected_courses: Optional[List[str]] = None
) -> Dict:
    """
    Calculate all metrics comparing the three search methods.
    
    Args:
        embedding_results: Results from embedding-only search
        id_results: Results from ID-matching-only search
        combined_results: Results from combined search
        expected_courses: Optional list of expected course IDs
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    embedding_ids = set(embedding_results['course_id'].tolist()) if len(embedding_results) > 0 else set()
    id_ids = set(id_results['course_id'].tolist()) if len(id_results) > 0 else set()
    combined_ids = set(combined_results['course_id'].tolist()) if len(combined_results) > 0 else set()
    
    metrics = {
        "result_counts": {
            "embedding": len(embedding_results),
            "id_matching": len(id_results),
            "combined": len(combined_results)
        },
        "overlap": {
            "embedding_vs_combined": jaccard_similarity(embedding_ids, combined_ids),
            "id_vs_combined": jaccard_similarity(id_ids, combined_ids),
            "embedding_vs_id": jaccard_similarity(embedding_ids, id_ids)
        },
        "rank_correlations": {
            "embedding_vs_combined": calculate_rank_correlation(embedding_results, combined_results),
            "id_vs_combined": calculate_rank_correlation(id_results, combined_results),
            "embedding_vs_id": calculate_rank_correlation(embedding_results, id_results)
        }
    }
    
    # Add coverage metrics if expected courses provided
    if expected_courses:
        metrics["coverage"] = {
            "embedding": calculate_coverage(expected_courses, embedding_results),
            "id_matching": calculate_coverage(expected_courses, id_results),
            "combined": calculate_coverage(expected_courses, combined_results)
        }
    
    return metrics


def format_course_result(course_id: str, title: str, score: float, max_title_len: int = 40) -> str:
    """
    Format a single course result for display.
    
    Args:
        course_id: Course ID
        title: Course title
        score: Similarity score
        max_title_len: Maximum length for title truncation
        
    Returns:
        str: Formatted string
    """
    title_truncated = title[:max_title_len] + "..." if len(title) > max_title_len else title
    return f"{course_id}: {title_truncated} ({score:.3f})"


def display_side_by_side(
    embedding_results: pd.DataFrame,
    id_results: pd.DataFrame,
    combined_results: pd.DataFrame,
    query: str,
    top_n: int = 10
) -> str:
    """
    Generate a side-by-side comparison display of results from all three methods.
    
    Args:
        embedding_results: Results from embedding-only search
        id_results: Results from ID-matching-only search
        combined_results: Results from combined search
        query: The original query
        top_n: Number of top results to display
        
    Returns:
        str: Formatted side-by-side comparison string
    """
    lines = []
    lines.append(f"\nQuery: \"{query}\"")
    lines.append("=" * 100)
    
    # Prepare data for each column
    embedding_list = []
    id_list = []
    combined_list = []
    
    for i in range(min(top_n, len(embedding_results))):
        row = embedding_results.iloc[i]
        embedding_list.append(
            f"{i+1}. {format_course_result(row['course_id'], row['title'], row.get('similarity_score', 0.0))}"
        )
    
    for i in range(min(top_n, len(id_results))):
        row = id_results.iloc[i]
        id_list.append(
            f"{i+1}. {format_course_result(row['course_id'], row['title'], row.get('similarity_score', 0.0))}"
        )
    
    for i in range(min(top_n, len(combined_results))):
        row = combined_results.iloc[i]
        combined_list.append(
            f"{i+1}. {format_course_result(row['course_id'], row['title'], row.get('similarity_score', 0.0))}"
        )
    
    # Handle empty results
    if len(embedding_list) == 0:
        embedding_list = ["(no results)"]
    if len(id_list) == 0:
        id_list = ["(no results)"]
    if len(combined_list) == 0:
        combined_list = ["(no results)"]
    
    # Calculate column widths
    max_len = max(
        max(len(line) for line in embedding_list),
        max(len(line) for line in id_list),
        max(len(line) for line in combined_list),
        20  # Minimum width
    )
    
    # Create header
    header = f"│ {'Embedding Only':<{max_len}} │ {'ID Matching Only':<{max_len}} │ {'Combined':<{max_len}} │"
    separator = f"├{'-' * (max_len + 2)}┼{'-' * (max_len + 2)}┼{'-' * (max_len + 2)}┤"
    top_border = f"┌{'-' * (max_len + 2)}┬{'-' * (max_len + 2)}┬{'-' * (max_len + 2)}┐"
    bottom_border = f"└{'-' * (max_len + 2)}┴{'-' * (max_len + 2)}┴{'-' * (max_len + 2)}┘"
    
    lines.append(top_border)
    lines.append(header)
    lines.append(separator)
    
    # Fill rows
    max_rows = max(len(embedding_list), len(id_list), len(combined_list))
    for i in range(max_rows):
        emb_line = embedding_list[i] if i < len(embedding_list) else ""
        id_line = id_list[i] if i < len(id_list) else ""
        comb_line = combined_list[i] if i < len(combined_list) else ""
        
        row = f"│ {emb_line:<{max_len}} │ {id_line:<{max_len}} │ {comb_line:<{max_len}} │"
        lines.append(row)
    
    lines.append(bottom_border)
    
    return "\n".join(lines)


def format_metrics_summary(metrics: Dict, execution_times: Dict) -> str:
    """
    Format metrics summary as a readable string.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        execution_times: Dictionary with 'embedding', 'id_matching', 'combined' keys
        
    Returns:
        str: Formatted metrics summary
    """
    lines = []
    lines.append("\nMetrics Summary:")
    lines.append("-" * 80)
    
    # Result counts
    lines.append("\nResult Counts:")
    counts = metrics["result_counts"]
    lines.append(f"  Embedding Only: {counts['embedding']}")
    lines.append(f"  ID Matching Only: {counts['id_matching']}")
    lines.append(f"  Combined: {counts['combined']}")
    
    # Overlap (Jaccard similarity)
    lines.append("\nOverlap (Jaccard Similarity):")
    overlap = metrics["overlap"]
    lines.append(f"  Embedding vs Combined: {overlap['embedding_vs_combined']:.3f}")
    lines.append(f"  ID Matching vs Combined: {overlap['id_vs_combined']:.3f}")
    lines.append(f"  Embedding vs ID Matching: {overlap['embedding_vs_id']:.3f}")
    
    # Rank correlations
    lines.append("\nRank Correlations (Spearman):")
    rank_corr = metrics["rank_correlations"]
    emb_comb_corr, emb_comb_n = rank_corr["embedding_vs_combined"]
    id_comb_corr, id_comb_n = rank_corr["id_vs_combined"]
    emb_id_corr, emb_id_n = rank_corr["embedding_vs_id"]
    
    lines.append(f"  Embedding vs Combined: {emb_comb_corr:.3f} (n={emb_comb_n} common courses)")
    lines.append(f"  ID Matching vs Combined: {id_comb_corr:.3f} (n={id_comb_n} common courses)")
    lines.append(f"  Embedding vs ID Matching: {emb_id_corr:.3f} (n={emb_id_n} common courses)")
    
    # Execution times
    lines.append("\nExecution Times:")
    lines.append(f"  Embedding Only: {execution_times.get('embedding', 0):.3f}s")
    lines.append(f"  ID Matching Only: {execution_times.get('id_matching', 0):.3f}s")
    lines.append(f"  Combined: {execution_times.get('combined', 0):.3f}s")
    
    # Coverage (if available)
    if "coverage" in metrics:
        lines.append("\nCoverage (Expected Courses Found):")
        coverage = metrics["coverage"]
        for method in ["embedding", "id_matching", "combined"]:
            cov = coverage[method]
            if cov["total_expected"] > 0:
                lines.append(
                    f"  {method.capitalize()}: {cov['found_count']}/{cov['total_expected']} "
                    f"({cov['coverage_ratio']:.1%})"
                )
    
    return "\n".join(lines)

