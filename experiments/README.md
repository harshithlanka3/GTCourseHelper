# Pipeline Permutation Experiments

This directory contains experimental code to systematically evaluate three different search pipeline configurations for the GT Course Helper system.

## Overview

The experiments compare three search approaches:

1. **Embedding-only**: Pure semantic similarity search using course description embeddings (no ID matching)
2. **ID-matching-only**: Exact/fuzzy course ID matching only (no embeddings)
3. **Combined**: Hybrid approach using both methods together (current default)

## Why This Matters

- **Performance**: Determine which approach is most accurate for different query types
- **Efficiency**: Measure execution time differences between methods
- **Robustness**: Understand strengths/weaknesses of each approach
- **Synergy**: Validate whether combining methods improves results or if one is sufficient

## Quick Start

### Run All Experiments

```bash
# From GTCourseHelper root directory
python experiments/pipeline_permutations.py
```

### Run Experiments for Specific Category

```bash
# Test only semantic-only queries
python experiments/pipeline_permutations.py --category semantic-only

# Test only ID-only queries
python experiments/pipeline_permutations.py --category id-only
```

### Command Line Options

```bash
python experiments/pipeline_permutations.py --help

Options:
  --df-path PATH       Path to processed courses pickle (default: data/202508_processed.pkl)
  --category CATEGORY  Filter queries by category
  --top-k N            Number of results per query (default: 50)
  --no-gpt             Disable GPT query refinement
  --no-save            Don't save results to files
  --output-dir DIR     Directory to save results (default: experiments/results)
```

## Test Queries

The test suite contains **22 diverse queries** organized into 6 categories:

### Semantic-Only (5 queries)
Queries with no course IDs mentioned, testing pure semantic search:
- "I want to learn machine learning and neural networks"
- "Courses about data structures and algorithms"
- "I'm interested in computer vision and image processing"
- "Optimization and operations research courses"
- "Courses on database systems and SQL"

### ID-Only (4 queries)
Queries mentioning specific course IDs, testing ID matching:
- "I need CS 1332"
- "CS 7641 or similar courses"
- "MATH 2106 and related courses"
- "ISYE 6501"

### Mixed (4 queries)
Queries combining semantic description with course IDs:
- "Machine learning courses like CS 7641"
- "Data structures courses, specifically CS 1332 or equivalent"
- "I want to take CS 4641 or other machine learning courses"
- "Optimization courses, maybe ISYE 6669"

### Department-Specific (4 queries)
Queries targeting specific departments:
- "ISYE courses on optimization and operations research"
- "CS courses about artificial intelligence"
- "MATH courses on linear algebra and calculus"
- "ECE courses on signal processing"

### Graduate-Level (2 queries)
Queries for graduate-level courses:
- "Graduate courses in computer vision"
- "Advanced machine learning courses for graduate students"

### Edge Cases (3 queries)
Challenging queries testing edge cases:
- "Courses with minimal prerequisites"
- "CS 1332 CS 1331" (multiple IDs, minimal context)
- "I haven't taken calculus" (negative constraint)

## Editing Test Queries

Test queries are defined in `test_queries.py` as a simple Python list. To add or modify queries:

1. Open `experiments/test_queries.py`
2. Edit the `TEST_QUERIES` list
3. Each query is a dictionary with:
   - `query`: The actual query text
   - `category`: One of "semantic-only", "id-only", "mixed", "department-specific", "graduate-level", "edge-case"
   - `expected_courses`: Optional list of course IDs for validation
   - `notes`: Optional explanation

Example:
```python
{
    "query": "I want to learn machine learning",
    "category": "semantic-only",
    "notes": "Pure semantic search test"
}
```

## Metrics Explained

The experiments calculate several quantitative metrics:

### Result Counts
Number of results returned by each method. Useful for understanding coverage differences.

### Overlap (Jaccard Similarity)
Measures how similar the result sets are between methods:
- Range: 0.0 (no overlap) to 1.0 (identical sets)
- Formula: `|A ∩ B| / |A ∪ B|`
- Example: 0.8 means 80% of courses appear in both result sets

### Rank Correlations (Spearman)
Measures how similar the ranking is for courses appearing in both result sets:
- Range: -1.0 (opposite ranking) to 1.0 (identical ranking)
- Only calculated for courses that appear in both result sets
- Higher values indicate more consistent ranking between methods

### Execution Time
Time taken for each search method:
- Measured in seconds
- Useful for understanding performance trade-offs

### Coverage
For queries with `expected_courses` specified, measures how many expected courses were found:
- Ratio: `found_count / total_expected`
- Example: 0.75 means 75% of expected courses were found

## Experimental Results

**Date:** November 14, 2025  
**Total Queries Tested:** 22/22 ✓ COMPLETE  
**Status:** All experiments completed successfully

### Executive Summary

The experiments successfully tested three search pipeline configurations across 22 diverse queries. Key findings:

- **ID-matching is 950x faster** than embedding search (~0.0008s vs ~0.77s)
- **Combined method achieves 100% coverage** for queries with expected courses
- **Embedding and Combined methods show 97% overlap** on average, indicating high consistency
- **All 22 queries processed** with complete metrics and results saved

### Performance Analysis

#### Execution Times

| Method | Mean | Min | Max | Notes |
|--------|------|-----|-----|-------|
| **ID-matching** | 0.0008s | 0.0003s | 0.0016s | ⚡ Fastest |
| **Embedding-only** | 0.77s | 0.64s | 1.16s | Includes GPT refinement |
| **Combined** | 0.78s | 0.69s | 1.07s | Similar to embedding |

**Key Insight:** ID-matching is **950x faster** than embedding search, making it ideal for queries with course IDs.

#### Result Counts

| Method | Mean Results | Range |
|--------|--------------|-------|
| Embedding-only | 20.0 | 20-20 (consistent) |
| ID-matching | 0.5 | 0-2 (only when IDs present) |
| Combined | 20.0 | 20-20 (consistent) |

### Overlap Analysis (Jaccard Similarity)

#### Embedding vs Combined
- **Mean:** 0.970 (97% overlap)
- **Range:** 0.818 - 1.000
- **Interpretation:** Very high similarity - combined method largely returns same results as embedding-only

#### ID-matching vs Combined
- **Mean:** 0.023 (2.3% overlap)
- **Range:** 0.000 - 0.100
- **Interpretation:** ID matches are a small subset of combined results (as expected)

#### Embedding vs ID-matching
- **Mean:** 0.007 (0.7% overlap)
- **Range:** 0.000 - 0.050
- **Interpretation:** These methods return different results (embedding is semantic, ID is exact match)

### Rank Correlation Analysis

#### Embedding vs Combined
- **Mean:** 0.985 (98.5% correlation)
- **Range:** 0.842 - 1.000
- **Interpretation:** Methods agree strongly on ranking for common courses

#### ID-matching vs Combined
- **Mean:** 0.045 (4.5% correlation)
- **Range:** 0.000 - 1.000
- **Interpretation:** Low correlation (ID matches are prioritized in combined, changing ranks)

### Coverage Analysis

For queries with expected courses specified (9 queries):

| Method | Mean Coverage | Perfect (100%) |
|--------|---------------|----------------|
| **ID-matching** | 100.0% | 9/9 queries ✓ |
| **Combined** | 100.0% | 9/9 queries ✓ |
| **Embedding-only** | 33.3% | 3/9 queries |

**Key Finding:** ID-matching and Combined methods achieve **perfect coverage** for queries with course IDs, while embedding-only finds only 1/3 of expected courses.

### Category-Specific Insights

#### Semantic-Only Queries (5 queries)
- ✅ Embedding found results: 5/5 (100%)
- ❌ ID matching found results: 0/5 (0%)
- ✅ Avg overlap (embedding vs combined): 1.000 (perfect match)
- **Conclusion:** For pure semantic queries, embedding and combined are equivalent. ID matching doesn't help.

#### ID-Only Queries (4 queries)
- ✅ ID matching found results: 4/4 (100%)
- ✅ Embedding found results: 4/4 (100%)
- ✅ ID matching coverage: 100.0%
- **Conclusion:** ID matching is perfect for exact course lookups. Embedding also finds results but may miss specific courses.

#### Mixed Queries (4 queries)
- ✅ All methods found results: 100%
- ✅ Combined coverage: 100.0%
- **Conclusion:** Combined method excels at queries with both semantic intent and specific course mentions.

### Key Findings

#### Strengths of Each Method

1. **ID-Matching-Only:**
   - ⚡ **950x faster** than embedding
   - ✅ **100% coverage** for queries with course IDs
   - ✅ Perfect for exact course lookups
   - ❌ Fails for semantic-only queries

2. **Embedding-Only:**
   - ✅ Works for all semantic queries
   - ✅ Returns diverse, relevant results
   - ✅ Consistent 20 results per query
   - ❌ Slower (includes GPT refinement)
   - ❌ May miss specific courses mentioned

3. **Combined:**
   - ✅ **Best coverage** (100% for expected courses)
   - ✅ Works for all query types
   - ✅ Prioritizes ID matches, then adds semantic results
   - ✅ High overlap with embedding (97%)
   - ⚠️ Similar speed to embedding (slightly slower)

#### Performance Trade-offs

- **Speed:** ID-matching (0.0008s) << Combined (0.78s) ≈ Embedding (0.77s)
- **Coverage:** Combined (100%) = ID-matching (100%) > Embedding (33%)
- **Flexibility:** Combined > Embedding > ID-matching

### Recommendations

#### When to Use Each Method

1. **Use ID-Matching-Only when:**
   - Query contains specific course IDs
   - Speed is critical (< 1ms)
   - Exact course lookup needed

2. **Use Embedding-Only when:**
   - Pure semantic query (no course IDs)
   - Want diverse, exploratory results
   - Can tolerate ~0.77s latency

3. **Use Combined when:**
   - Need best coverage (production use)
   - Query may contain IDs or be semantic
   - Can tolerate ~0.78s latency
   - **RECOMMENDED for production**

#### Optimal Strategy

The **Combined method** is recommended for production because:
- ✅ 100% coverage for expected courses
- ✅ Works for all query types
- ✅ Only ~0.01s slower than embedding-only
- ✅ Prioritizes exact matches (ID) then adds semantic results

## Interpreting Results

### Side-by-Side Display
Shows top 10 results from each method in columns for easy comparison.

### Metrics Summary
Provides quantitative comparison:
- **High overlap** (e.g., >0.7): Methods return similar results
- **Low overlap** (e.g., <0.3): Methods return different results
- **High rank correlation** (e.g., >0.8): Methods agree on ranking
- **Fast execution**: ID-matching is typically fastest, embedding-only is slowest

### Expected Patterns

1. **Semantic-only queries**:
   - Embedding-only: Should return good results
   - ID-matching-only: Should return empty (no IDs in query)
   - Combined: Should match embedding-only

2. **ID-only queries**:
   - Embedding-only: May return some results, but may miss the specific ID
   - ID-matching-only: Should return exact/fuzzy matches
   - Combined: Should prioritize ID matches

3. **Mixed queries**:
   - All methods should return results
   - Combined should show best of both worlds

## Output Files

Results are saved to `experiments/results/`:

- `query_*.json`: Individual experiment results with full metrics
- `summary.csv`: Aggregated metrics across all queries
- `experiment_run.log`: Full execution log (if saved)

The summary CSV includes:
- Query text and category
- Result counts for each method
- Execution times
- Overlap metrics
- Rank correlations
- Coverage metrics (if expected courses specified)

## Example Output

```
Query: "machine learning courses"
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Embedding Only      │ ID Matching Only   │ Combined            │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ 1. CS 7641 (0.85)   │ (no results)       │ 1. CS 7641 (0.95)   │
│ 2. CS 4641 (0.82)   │                     │ 2. CS 4641 (0.82)   │
└─────────────────────┴─────────────────────┴─────────────────────┘

Metrics Summary:
- Overlap (Embedding vs Combined): 0.80
- Execution Time: Embedding=0.5s, ID=0.01s, Combined=0.52s
```

## Models Used

### GPT-3.5-turbo (OpenAI API)
- **Purpose:** Query refinement - converts user queries to idealized course descriptions
- **Location:** `search_courses.py` line 68
- **Cost:** ~$0.0121 for 22 queries (44 API calls)
- **Note:** Can be disabled with `--no-gpt` flag

### all-MiniLM-L6-v2 (Sentence Transformers - Local)
- **Purpose:** Generating 384-dimensional embeddings for semantic search
- **Location:** `preprocess_courses.py` line 23, `search_courses.py` line 137
- **Cost:** $0 (runs locally, no API calls)
- **Size:** ~80MB (downloaded on first run)

## Dependencies

- pandas
- numpy
- scipy (for Spearman correlation)
- sentence-transformers (for embeddings)
- openai (for GPT query refinement)

All dependencies should already be installed if you've set up the main GTCourseHelper project.

## Notes

- The experiments use the same GPT query refinement as the main search system
- Results may vary slightly between runs due to GPT API variability
- Execution times include GPT API calls (if enabled)
- The ID-matching-only method returns empty results for queries without course IDs
- **Total execution time:** ~17 seconds for 22 queries

## Conclusion

The experiments successfully validated all three pipeline permutations:

1. ✅ **ID-matching** is extremely fast and perfect for exact matches
2. ✅ **Embedding** provides semantic search for diverse queries
3. ✅ **Combined** offers the best of both worlds with 100% coverage

The **Combined method** is the clear winner for production use, providing comprehensive coverage while maintaining reasonable performance.

---

**Framework Version:** 1.0  
**Last Updated:** November 14, 2025
