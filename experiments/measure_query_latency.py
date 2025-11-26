import sys
import time
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_courses import generate_idealized_course_description
from course_id_matcher import extract_course_ids, hybrid_search, enhance_search_results_with_ids, combine_id_and_semantic_results
from embedding_utils import get_query_embedding
from experiments.test_queries import TEST_QUERIES


def search_courses_with_timing(user_query, df, df_path=None, top_k=50, use_gpt=True, use_id_matching=True):
    timings = {}
    if df_path:
        start = time.time()
        df = pd.read_pickle(df_path)
        timings['data_loading'] = time.time() - start
    else:
        timings['data_loading'] = 0.0
    start = time.time()
    if 'department' not in df.columns:
        df['department'] = df['course_id'].str.split().str[0]
    if 'is_graduate_level' not in df.columns:
        numeric_part = df['course_id'].str.extract(r'(\d{4})')[0].astype(float)
        df['is_graduate_level'] = numeric_part.fillna(0) > 4000
        df['is_graduate_level'] = df['is_graduate_level'].fillna(False)
    timings['data_preprocessing'] = time.time() - start
    start = time.time()
    mentioned_ids = extract_course_ids(user_query) if use_id_matching else []
    timings['id_extraction'] = time.time() - start
    if use_gpt:
        start = time.time()
        query_for_search = generate_idealized_course_description(user_query)
        timings['gpt_refinement'] = time.time() - start
    else:
        query_for_search = user_query
        timings['gpt_refinement'] = 0.0
    id_results = pd.DataFrame()
    if use_id_matching and mentioned_ids:
        start = time.time()
        id_results = hybrid_search(user_query, df, top_k=top_k)
        if len(id_results) > 0:
            if 'similarity_score' not in id_results.columns:
                id_results['similarity_score'] = 0.95
        timings['id_matching'] = time.time() - start
    else:
        timings['id_matching'] = 0.0
    start = time.time()
    query_embedding = get_query_embedding(query_for_search)
    timings['query_embedding'] = time.time() - start
    start = time.time()
    from sklearn.metrics.pairwise import cosine_similarity
    course_embeddings = np.array(df['embedding'].tolist())
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    semantic_results = df.iloc[top_indices].copy()
    semantic_results['similarity_score'] = similarities[top_indices]
    timings['semantic_search'] = time.time() - start
    if use_id_matching and mentioned_ids:
        start = time.time()
        semantic_results = enhance_search_results_with_ids(user_query, semantic_results, df, boost_factor=1.5)
        timings['result_enhancement'] = time.time() - start
    else:
        timings['result_enhancement'] = 0.0
    start = time.time()
    semantic_results = semantic_results[['course_id', 'title', 'description', 'prerequisites', 'meeting_times', 'department', 'is_graduate_level', 'similarity_score', 'embedding']]
    if use_id_matching and len(id_results) > 0:
        id_results = id_results[['course_id', 'title', 'description', 'prerequisites', 'meeting_times', 'department', 'is_graduate_level', 'similarity_score', 'embedding']]
        results = combine_id_and_semantic_results(id_results, semantic_results, top_k=top_k)
    else:
        results = semantic_results
    timings['result_combination'] = time.time() - start
    timings['total'] = sum([v for k, v in timings.items() if k != 'total'])
    return results, timings


def measure_query_batch(query, df, num_runs, use_gpt=True, use_id_matching=True, top_k=50):
    all_timings = []
    for run in range(num_runs):
        _, timings = search_courses_with_timing(query, df, df_path=None, top_k=top_k, use_gpt=use_gpt, use_id_matching=use_id_matching)
        all_timings.append(timings)
    if not all_timings:
        return None
    component_stats = {}
    all_components = set()
    for timing in all_timings:
        all_components.update(timing.keys())
    for component in all_components:
        values = [t.get(component, 0) for t in all_timings]
        if values:
            component_stats[component] = {
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "count": len(values)
            }
    return {"num_runs": num_runs, "query": query, "use_gpt": use_gpt, "use_id_matching": use_id_matching, "component_stats": component_stats}


def run_latency_experiments(df_path='data/202508_processed.pkl', queries=None, num_runs_list=[1, 5, 10], output_dir='experiments/results/latency'):
    df = pd.read_pickle(df_path)
    if queries is None:
        queries = TEST_QUERIES
    total_experiments = len(queries) * 4 * len(num_runs_list)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_results = []
    experiment_count = 0
    configurations = [
        {"use_gpt": True, "use_id_matching": True, "name": "full_system"},
        {"use_gpt": False, "use_id_matching": True, "name": "no_gpt"},
        {"use_gpt": True, "use_id_matching": False, "name": "no_id_matching"},
        {"use_gpt": False, "use_id_matching": False, "name": "minimal"},
    ]
    
    for config in configurations:
        config_name = config["name"]
        total_queries = len(queries)
        for query_idx, query_dict in enumerate(queries, 1):
            query = query_dict["query"]
            for num_runs in num_runs_list:
                result = measure_query_batch(
                    query, df, num_runs=num_runs,
                    use_gpt=config["use_gpt"],
                    use_id_matching=config["use_id_matching"]
                )
                if result and result["component_stats"]:
                    stats = result["component_stats"]
                    total_stats = stats.get("total", {})
                    row = {
                        "query": query,
                        "category": query_dict.get("category", "unknown"),
                        "configuration": config_name,
                        "num_runs": num_runs,
                        "use_gpt": config["use_gpt"],
                        "use_id_matching": config["use_id_matching"],
                        "total_min": total_stats.get("min", 0),
                        "total_max": total_stats.get("max", 0),
                        "total_avg": total_stats.get("avg", 0),
                        "total_median": total_stats.get("median", 0),
                        "total_std": total_stats.get("std", 0),
                        "gpt_refinement_min": stats.get("gpt_refinement", {}).get("min", 0),
                        "gpt_refinement_max": stats.get("gpt_refinement", {}).get("max", 0),
                        "gpt_refinement_avg": stats.get("gpt_refinement", {}).get("avg", 0),
                        "id_matching_min": stats.get("id_matching", {}).get("min", 0),
                        "id_matching_max": stats.get("id_matching", {}).get("max", 0),
                        "id_matching_avg": stats.get("id_matching", {}).get("avg", 0),
                        "query_embedding_min": stats.get("query_embedding", {}).get("min", 0),
                        "query_embedding_max": stats.get("query_embedding", {}).get("max", 0),
                        "query_embedding_avg": stats.get("query_embedding", {}).get("avg", 0),
                        "semantic_search_min": stats.get("semantic_search", {}).get("min", 0),
                        "semantic_search_max": stats.get("semantic_search", {}).get("max", 0),
                        "semantic_search_avg": stats.get("semantic_search", {}).get("avg", 0),
                    }
                    all_results.append(row)
    results_df = pd.DataFrame(all_results)
    csv_path = output_path / "latency_results.csv"
    results_df.to_csv(csv_path, index=False)
    return results_df


def generate_plots(results_df, output_dir='experiments/results/latency'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Query Latency Analysis', fontsize=16, fontweight='bold')
    ax1 = axes[0, 0]
    config_avg = results_df.groupby(['configuration', 'num_runs'])['total_avg'].mean().unstack()
    config_avg.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Average Total Query Time by Configuration')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend(title='Number of Runs', labels=['1 query', '5 queries', '10 queries'])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax2 = axes[0, 1]
    single_query = results_df[results_df['num_runs'] == 1]
    config_stats = single_query.groupby('configuration').agg({
        'total_min': 'mean',
        'total_avg': 'mean',
        'total_max': 'mean'
    })
    x = np.arange(len(config_stats))
    width = 0.25
    ax2.bar(x - width, config_stats['total_min'], width, label='Min', alpha=0.8)
    ax2.bar(x, config_stats['total_avg'], width, label='Avg', alpha=0.8)
    ax2.bar(x + width, config_stats['total_max'], width, label='Max', alpha=0.8)
    ax2.set_title('Single Query: Min/Max/Avg by Configuration')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_stats.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax3 = axes[1, 0]
    full_system = results_df[
        (results_df['configuration'] == 'full_system') & 
        (results_df['num_runs'] == 1)
    ]
    components = ['gpt_refinement_avg', 'id_matching_avg', 'query_embedding_avg', 'semantic_search_avg']
    component_names = ['GPT Refinement', 'ID Matching', 'Query Embedding', 'Semantic Search']
    component_means = [full_system[comp].mean() for comp in components]
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    bars = ax3.bar(component_names, component_means, color=colors, alpha=0.8)
    ax3.set_title('Component Timing Breakdown (Full System, Single Query)')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticklabels(component_names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
    ax4 = axes[1, 1]
    run_comparison = results_df.groupby(['num_runs', 'configuration']).agg({
        'total_min': 'mean',
        'total_avg': 'mean',
        'total_max': 'mean'
    }).reset_index()
    
    for config in run_comparison['configuration'].unique():
        config_data = run_comparison[run_comparison['configuration'] == config]
        ax4.plot(config_data['num_runs'], config_data['total_avg'], 
                marker='o', label=config, linewidth=2, markersize=8)
    
    ax4.set_title('Average Query Time vs Number of Runs')
    ax4.set_xlabel('Number of Runs')
    ax4.set_ylabel('Average Time (seconds)')
    ax4.set_xticks([1, 5, 10])
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'query_latency_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Component-Level Timing Analysis', fontsize=16, fontweight='bold')
    ax = axes2[0, 0]
    single_query = results_df[results_df['num_runs'] == 1]
    components_to_plot = {
        'gpt_refinement_avg': 'GPT Refinement',
        'id_matching_avg': 'ID Matching',
        'query_embedding_avg': 'Query Embedding',
        'semantic_search_avg': 'Semantic Search'
    }
    
    configs = single_query['configuration'].unique()
    x = np.arange(len(configs))
    width = 0.2
    multiplier = 0
    
    for comp_key, comp_name in components_to_plot.items():
        offset = width * multiplier
        values = [single_query[single_query['configuration'] == cfg][comp_key].mean() 
                 for cfg in configs]
        bars = ax.bar(x + offset, values, width, label=comp_name, alpha=0.8)
        multiplier += 1
    
    ax.set_title('Component Timing by Configuration (Single Query)')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax = axes2[0, 1]
    full_system = results_df[results_df['configuration'] == 'full_system']
    run_stats = full_system.groupby('num_runs').agg({
        'total_min': 'mean',
        'total_avg': 'mean',
        'total_max': 'mean'
    })
    
    x = np.arange(len(run_stats))
    width = 0.25
    ax.bar(x - width, run_stats['total_min'], width, label='Min', alpha=0.8)
    ax.bar(x, run_stats['total_avg'], width, label='Avg', alpha=0.8)
    ax.bar(x + width, run_stats['total_max'], width, label='Max', alpha=0.8)
    ax.set_title('Full System: Min/Max/Avg by Number of Runs')
    ax.set_xlabel('Number of Runs')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(run_stats.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax = axes2[1, 0]
    single_query = results_df[results_df['num_runs'] == 1]
    data_to_plot = [single_query[single_query['configuration'] == cfg]['total_avg'].values 
                    for cfg in single_query['configuration'].unique()]
    bp = ax.boxplot(data_to_plot, labels=single_query['configuration'].unique(), patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_title('Distribution of Query Times by Configuration')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax = axes2[1, 1]
    ax.axis('tight')
    ax.axis('off')
    summary_data = []
    for config in results_df['configuration'].unique():
        config_data = results_df[results_df['configuration'] == config]
        for num_runs in [1, 5, 10]:
            run_data = config_data[config_data['num_runs'] == num_runs]
            if len(run_data) > 0:
                summary_data.append([
                    config,
                    num_runs,
                    f"{run_data['total_min'].mean():.3f}",
                    f"{run_data['total_avg'].mean():.3f}",
                    f"{run_data['total_max'].mean():.3f}"
                ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Config', 'Runs', 'Min (s)', 'Avg (s)', 'Max (s)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path2 = output_path / 'component_timing_analysis.png'
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure query latency with component-level timing")
    parser.add_argument("--df-path", type=str, default="data/202508_processed.pkl",
                      help="Path to processed courses pickle file")
    parser.add_argument("--num-queries", type=int, nargs='+', default=[1, 5, 10],
                      help="Number of queries to test (e.g., 1 5 10)")
    parser.add_argument("--output-dir", type=str, default="experiments/results/latency",
                      help="Directory to save results")
    parser.add_argument("--skip-experiments", action="store_true",
                      help="Skip running experiments, only generate plots")
    parser.add_argument("--skip-plots", action="store_true",
                      help="Skip generating plots")
    
    args = parser.parse_args()
    
    if not args.skip_experiments:
        results_df = run_latency_experiments(
            df_path=args.df_path,
            num_runs_list=args.num_queries,
            output_dir=args.output_dir
        )
    else:
        csv_path = Path(args.output_dir) / "latency_results.csv"
        if csv_path.exists():
            results_df = pd.read_csv(csv_path)
        else:
            return
    if not args.skip_plots:
        generate_plots(results_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

