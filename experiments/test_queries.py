"""
Easily editable test query suite for pipeline permutation experiments.

To add/edit queries, simply modify the TEST_QUERIES list below.
Each query is a dictionary with:
- query: The actual query text
- category: One of "semantic-only", "id-only", "mixed", "department-specific", "graduate-level", "edge-case"
- expected_courses: Optional list of course IDs for validation
- notes: Optional explanation of why this query is useful
"""

TEST_QUERIES = [
    # === SEMANTIC-ONLY QUERIES ===
    {
        "query": "I want to learn machine learning and neural networks",
        "category": "semantic-only",
        "notes": "Pure semantic search, no course IDs"
    },
    {
        "query": "Courses about data structures and algorithms",
        "category": "semantic-only"
    },
    {
        "query": "I'm interested in computer vision and image processing",
        "category": "semantic-only"
    },
    {
        "query": "Optimization and operations research courses",
        "category": "semantic-only"
    },
    {
        "query": "Courses on database systems and SQL",
        "category": "semantic-only"
    },
    
    # === ID-ONLY QUERIES ===
    {
        "query": "I need CS 1332",
        "category": "id-only",
        "expected_courses": ["CS 1332"],
        "notes": "Single exact course ID"
    },
    {
        "query": "CS 7641 or similar courses",
        "category": "id-only",
        "expected_courses": ["CS 7641"]
    },
    {
        "query": "MATH 2106 and related courses",
        "category": "id-only",
        "expected_courses": ["MATH 2106"]
    },
    {
        "query": "ISYE 6501",
        "category": "id-only",
        "expected_courses": ["ISYE 6501"]
    },
    
    # === MIXED QUERIES ===
    {
        "query": "Machine learning courses like CS 7641",
        "category": "mixed",
        "expected_courses": ["CS 7641"],
        "notes": "Semantic + specific ID"
    },
    {
        "query": "Data structures courses, specifically CS 1332 or equivalent",
        "category": "mixed",
        "expected_courses": ["CS 1332"]
    },
    {
        "query": "I want to take CS 4641 or other machine learning courses",
        "category": "mixed",
        "expected_courses": ["CS 4641"]
    },
    {
        "query": "Optimization courses, maybe ISYE 6669",
        "category": "mixed",
        "expected_courses": ["ISYE 6669"]
    },
    
    # === DEPARTMENT-SPECIFIC QUERIES ===
    {
        "query": "ISYE courses on optimization and operations research",
        "category": "department-specific"
    },
    {
        "query": "CS courses about artificial intelligence",
        "category": "department-specific"
    },
    {
        "query": "MATH courses on linear algebra and calculus",
        "category": "department-specific"
    },
    {
        "query": "ECE courses on signal processing",
        "category": "department-specific"
    },
    
    # === GRADUATE-LEVEL QUERIES ===
    {
        "query": "Graduate courses in computer vision",
        "category": "graduate-level"
    },
    {
        "query": "Advanced machine learning courses for graduate students",
        "category": "graduate-level"
    },
    
    # === EDGE CASES ===
    {
        "query": "Courses with minimal prerequisites",
        "category": "edge-case",
        "notes": "Tests semantic understanding of constraints"
    },
    {
        "query": "CS 1332 CS 1331",
        "category": "edge-case",
        "expected_courses": ["CS 1332", "CS 1331"],
        "notes": "Multiple IDs, minimal context"
    },
    {
        "query": "I haven't taken calculus",
        "category": "edge-case",
        "notes": "Negative constraint, tests filtering"
    }
]


def get_queries_by_category(category=None):
    """
    Filter queries by category.
    
    Args:
        category (str, optional): Category to filter by. If None, returns all queries.
        
    Returns:
        list: List of query dictionaries matching the category
    """
    if category is None:
        return TEST_QUERIES
    return [q for q in TEST_QUERIES if q.get("category") == category]


def get_all_categories():
    """
    Get list of all unique categories in the test query suite.
    
    Returns:
        list: Sorted list of unique category strings
    """
    return sorted(list(set(q.get("category") for q in TEST_QUERIES)))


def get_query_by_index(index):
    """
    Get a specific query by its index in TEST_QUERIES.
    
    Args:
        index (int): Zero-based index of the query
        
    Returns:
        dict: Query dictionary, or None if index is out of range
    """
    if 0 <= index < len(TEST_QUERIES):
        return TEST_QUERIES[index]
    return None

