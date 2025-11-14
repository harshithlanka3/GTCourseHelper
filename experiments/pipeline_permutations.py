"""
Pipeline Permutation Experiments

Main script to test three search pipeline configurations:
1. Embedding-only: Pure semantic similarity search
2. ID-matching-only: Exact/fuzzy course ID matching
3. Combined: Hybrid approach (both methods together)
"""

import os
import sys
import time
import json
import pandas as pd
from pathlib import Path

# Add parent directory to path to import search_courses
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import search_courses - handle API key requirement gracefully
try:
    from search_courses import search_courses
    SEARCH_COURSES_AVAILABLE = True
except ValueError as e:
    if "No API key configured" in str(e):
        print("Warning: API key not configured. Embedding and combined searches will require API keys.")
        SEARCH_COURSES_AVAILABLE = False
        search_courses = None
    else:
        raise

from course_id_matcher import extract_course_ids, hybrid_search
from experiments.test_queries import TEST_QUERIES, get_queries_by_category
from experiments.comparison import (
    calculate_metrics,
    display_side_by_side,
    format_metrics_summary
)


def search_embedding_only(user_query: str, df: pd.DataFrame, top_k: int = 50, use_gpt: bool = True) -> pd.DataFrame:
    """
    Search using only embedding-based semantic similarity (no ID matching).
    
    Args:
        user_query: Natural language query
        df: Course DataFrame (already loaded)
        top_k: Number of results to return
        use_gpt: Whether to use GPT for query refinement
        
    Returns:
        DataFrame with search results
    """
    if not SEARCH_COURSES_AVAILABLE:
        raise ValueError("API key required for embedding search. Please set POE_API_KEY or OPENAI_API_KEY.")
    
    # Temporarily save df to a temp pickle for search_courses to load
    # (search_courses loads from file, so we need to work around this)
    import tempfile
    import pickle
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        df.to_pickle(tmp_path)
    
    try:
        # Suppress print statements from search_courses
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            results = search_courses(
                user_query=user_query,
                df_path=tmp_path,
                top_k=top_k,
                use_gpt=use_gpt,
                use_id_matching=False  # Disable ID matching
            )
        return results
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def search_id_matching_only(user_query: str, df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """
    Search using only course ID matching (exact/fuzzy), no embeddings.
    
    Args:
        user_query: Natural language query
        df: Course DataFrame
        top_k: Number of results to return
        
    Returns:
        DataFrame with search results, or empty DataFrame if no IDs found
    """
    mentioned_ids = extract_course_ids(user_query)
    if not mentioned_ids:
        return pd.DataFrame()
    
    results = hybrid_search(user_query, df, top_k=top_k)
    
    # Ensure similarity_score column exists
    if len(results) > 0 and 'similarity_score' not in results.columns:
        results['similarity_score'] = 0.95  # High score for exact matches
    
    return results


def search_combined(user_query: str, df: pd.DataFrame, top_k: int = 50, use_gpt: bool = True) -> pd.DataFrame:
    """
    Search using combined approach (both ID matching and embeddings).
    
    Args:
        user_query: Natural language query
        df: Course DataFrame (already loaded)
        top_k: Number of results to return
        use_gpt: Whether to use GPT for query refinement
        
    Returns:
        DataFrame with search results
    """
    if not SEARCH_COURSES_AVAILABLE:
        raise ValueError("API key required for combined search. Please set POE_API_KEY or OPENAI_API_KEY.")
    
    # Temporarily save df to a temp pickle for search_courses to load
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        df.to_pickle(tmp_path)
    
    try:
        # Suppress print statements from search_courses
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            results = search_courses(
                user_query=user_query,
                df_path=tmp_path,
                top_k=top_k,
                use_gpt=use_gpt,
                use_id_matching=True  # Enable ID matching
            )
        return results
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_experiment(
    query_dict: dict,
    df: pd.DataFrame,
    top_k: int = 50,
    use_gpt: bool = True,
    verbose: bool = True
) -> dict:
    """
    Run all three search permutations on a single query.
    
    Args:
        query_dict: Dictionary with 'query', 'category', and optional 'expected_courses'
        df: Course DataFrame
        top_k: Number of results to return
        use_gpt: Whether to use GPT for query refinement
        verbose: Whether to print progress
        
    Returns:
        dict: Dictionary containing results, metrics, and execution times
    """
    query = query_dict["query"]
    expected_courses = query_dict.get("expected_courses")
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running experiment for query: \"{query}\"")
        print(f"Category: {query_dict.get('category', 'unknown')}")
        print(f"{'='*80}")
    
    execution_times = {}
    results = {}
    
    # Run embedding-only search
    if verbose:
        print("\n[1/3] Running embedding-only search...")
    try:
        start_time = time.time()
        embedding_results = search_embedding_only(query, df, top_k=top_k, use_gpt=use_gpt)
        execution_times["embedding"] = time.time() - start_time
        results["embedding"] = embedding_results
        if verbose:
            print(f"  Found {len(embedding_results)} results in {execution_times['embedding']:.3f}s")
    except ValueError as e:
        if "API key required" in str(e):
            if verbose:
                print(f"  Skipped (API key required)")
            embedding_results = pd.DataFrame()
            execution_times["embedding"] = 0.0
            results["embedding"] = embedding_results
        else:
            raise
    
    # Run ID-matching-only search
    if verbose:
        print("\n[2/3] Running ID-matching-only search...")
    start_time = time.time()
    id_results = search_id_matching_only(query, df, top_k=top_k)
    execution_times["id_matching"] = time.time() - start_time
    results["id_matching"] = id_results
    if verbose:
        print(f"  Found {len(id_results)} results in {execution_times['id_matching']:.3f}s")
    
    # Run combined search
    if verbose:
        print("\n[3/3] Running combined search...")
    try:
        start_time = time.time()
        combined_results = search_combined(query, df, top_k=top_k, use_gpt=use_gpt)
        execution_times["combined"] = time.time() - start_time
        results["combined"] = combined_results
        if verbose:
            print(f"  Found {len(combined_results)} results in {execution_times['combined']:.3f}s")
    except ValueError as e:
        if "API key required" in str(e):
            if verbose:
                print(f"  Skipped (API key required)")
            combined_results = pd.DataFrame()
            execution_times["combined"] = 0.0
            results["combined"] = combined_results
        else:
            raise
    
    # Calculate metrics
    if verbose:
        print("\nCalculating metrics...")
    metrics = calculate_metrics(
        embedding_results,
        id_results,
        combined_results,
        expected_courses
    )
    
    # Generate comparison display
    comparison_display = display_side_by_side(
        embedding_results,
        id_results,
        combined_results,
        query,
        top_n=10
    )
    
    metrics_summary = format_metrics_summary(metrics, execution_times)
    
    return {
        "query": query,
        "category": query_dict.get("category"),
        "expected_courses": expected_courses,
        "results": {
            "embedding": embedding_results.to_dict('records') if len(embedding_results) > 0 else [],
            "id_matching": id_results.to_dict('records') if len(id_results) > 0 else [],
            "combined": combined_results.to_dict('records') if len(combined_results) > 0 else []
        },
        "metrics": metrics,
        "execution_times": execution_times,
        "comparison_display": comparison_display,
        "metrics_summary": metrics_summary
    }


def save_experiment_result(experiment_result: dict, output_dir: Path):
    """
    Save individual experiment result to JSON file.
    
    Args:
        experiment_result: Result dictionary from run_experiment()
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename from query
    query_safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in experiment_result["query"])
    query_safe = query_safe[:50]  # Limit length
    filename = f"query_{hash(experiment_result['query']) % 100000}.json"
    filepath = output_dir / filename
    
    # Convert to JSON-serializable format
    json_result = {
        "query": experiment_result["query"],
        "category": experiment_result["category"],
        "expected_courses": experiment_result.get("expected_courses"),
        "execution_times": experiment_result["execution_times"],
        "metrics": experiment_result["metrics"],
        "result_counts": {
            "embedding": len(experiment_result["results"]["embedding"]),
            "id_matching": len(experiment_result["results"]["id_matching"]),
            "combined": len(experiment_result["results"]["combined"])
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(json_result, f, indent=2)
    
    return filepath


def run_all_experiments(
    df_path: str = 'data/202508_processed.pkl',
    queries: list = None,
    category: str = None,
    top_k: int = 50,
    use_gpt: bool = True,
    save_results: bool = True,
    output_dir: str = 'experiments/results'
) -> pd.DataFrame:
    """
    Run experiments on all test queries (or a filtered subset).
    
    Args:
        df_path: Path to processed courses pickle file
        queries: List of query dicts to test (if None, uses all TEST_QUERIES)
        category: Filter queries by category (if provided)
        top_k: Number of results to return per query
        use_gpt: Whether to use GPT for query refinement
        save_results: Whether to save individual JSON results
        output_dir: Directory to save results
        
    Returns:
        DataFrame with summary metrics for all experiments
    """
    # Load course data
    print(f"Loading courses from {df_path}...")
    df = pd.read_pickle(df_path)
    print(f"Loaded {len(df)} courses")
    
    # Get queries to test
    if queries is None:
        from experiments.test_queries import get_queries_by_category
        queries = get_queries_by_category(category)
    
    print(f"\nRunning experiments on {len(queries)} queries...")
    
    all_results = []
    output_path = Path(output_dir)
    
    for i, query_dict in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(queries)}")
        print(f"{'='*80}")
        
        try:
            result = run_experiment(query_dict, df, top_k=top_k, use_gpt=use_gpt, verbose=True)
            
            # Save individual result
            if save_results:
                save_experiment_result(result, output_path)
            
            # Print comparison
            print(result["comparison_display"])
            print(result["metrics_summary"])
            
            # Add to summary
            summary_row = {
                "query": result["query"],
                "category": result["category"],
                "embedding_count": len(result["results"]["embedding"]),
                "id_matching_count": len(result["results"]["id_matching"]),
                "combined_count": len(result["results"]["combined"]),
                "embedding_time": result["execution_times"]["embedding"],
                "id_matching_time": result["execution_times"]["id_matching"],
                "combined_time": result["execution_times"]["combined"],
                "overlap_embedding_combined": result["metrics"]["overlap"]["embedding_vs_combined"],
                "overlap_id_combined": result["metrics"]["overlap"]["id_vs_combined"],
                "overlap_embedding_id": result["metrics"]["overlap"]["embedding_vs_id"],
                "rank_corr_embedding_combined": result["metrics"]["rank_correlations"]["embedding_vs_combined"][0],
                "rank_corr_id_combined": result["metrics"]["rank_correlations"]["id_vs_combined"][0],
                "rank_corr_embedding_id": result["metrics"]["rank_correlations"]["embedding_vs_id"][0]
            }
            
            # Add coverage if available
            if "coverage" in result["metrics"]:
                for method in ["embedding", "id_matching", "combined"]:
                    cov = result["metrics"]["coverage"][method]
                    summary_row[f"{method}_coverage"] = cov["coverage_ratio"]
            
            all_results.append(summary_row)
            
        except Exception as e:
            print(f"ERROR: Failed to run experiment for query: {query_dict['query']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # Save summary CSV
    if save_results:
        summary_path = output_path / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}")
    
    return summary_df


def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline permutation experiments")
    parser.add_argument("--df-path", type=str, default="data/202508_processed.pkl",
                        help="Path to processed courses pickle file")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter queries by category (semantic-only, id-only, mixed, etc.)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of results to return per query")
    parser.add_argument("--no-gpt", action="store_true",
                        help="Disable GPT query refinement")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to files")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    summary_df = run_all_experiments(
        df_path=args.df_path,
        category=args.category,
        top_k=args.top_k,
        use_gpt=not args.no_gpt,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

