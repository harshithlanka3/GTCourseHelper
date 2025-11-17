import os
import sys
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
import pickle
from course_id_matcher import (
    extract_course_ids,
    hybrid_search,
    enhance_search_results_with_ids,
    combine_id_and_semantic_results
)
from embedding_utils import get_query_embedding


dotenv.load_dotenv()

# Initialize OpenAI client for query generation
poe_api_key = os.getenv("POE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if poe_api_key:
    client = OpenAI(
        api_key=poe_api_key,
        base_url="https://api.poe.com/v1"
    )
    print("Using Poe API for query generation")
elif openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("Using OpenAI API for query generation")
else:
    print("ERROR: No API key found!")
    print("Please set one of the following environment variables:")
    print("  - POE_API_KEY: Get your key at https://poe.com/api_key")
    print("  - OPENAI_API_KEY: Get your key at https://platform.openai.com")
    raise ValueError("No API key configured. Please set POE_API_KEY or OPENAI_API_KEY.")


def generate_idealized_course_description(user_query):
    """
    Use GPT-3.5-turbo to transform a user's natural language query into an
    idealized course description that better matches the academic structure
    typical of course catalogs.
    
    Args:
        user_query (str): The user's natural language description of their interests
        
    Returns:
        str: An idealized course description optimized for embedding search
    """
    prompt = f"""You will be given a request from a student at Georgia Institute of Technology to provide quality course recommendations. \
Generate a concise, content-focused course description that best matches their academic interests. Provide a list of topics and a general description suitable for embedding search.

CRITICAL:
- Do NOT include any scheduling, days, times, availability, or section-related constraints even if mentioned by the student. Ignore all time/schedule preferences.
- Do NOT include any prerequisite-related constraints or the student's background/eligibility (e.g., "I haven't taken X", "minimal prerequisites"). Ignore prerequisite preferences entirely.
- Focus ONLY on subject matter, skills, and learning objectives.
- Keep under 200 words.

Student Request:
{user_query}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0
        )
        
        idealized_description = response.choices[0].message.content.strip()
        return idealized_description
    except Exception as e:
        print(f"Error generating idealized description: {e}")
        return user_query  # Fallback to original query


def search_courses(user_query, df_path='data/202508_processed.pkl', top_k=50, use_gpt=True, use_id_matching=True):
    """
    Search for courses matching a user's query using hybrid approach:
    - Course ID matching (exact/fuzzy) if IDs are mentioned
    - Semantic similarity search
    
    Args:
        user_query (str): Natural language description of course interests
        df_path (str): Path to the processed courses pickle file
        top_k (int): Number of top results to return
        use_gpt (bool): Whether to use GPT to refine the query first
        use_id_matching (bool): Whether to enable course ID matching
        
    Returns:
        pandas.DataFrame: DataFrame containing the top_k matching courses
    """
    # Load the processed courses
    print(f"Loading courses from {df_path}...")
    df = pd.read_pickle(df_path)
    print(f"Loaded {len(df)} courses")
    
    # Ensure new metadata columns are present even for legacy pickles
    if 'department' not in df.columns:
        df['department'] = df['course_id'].str.split().str[0]
    
    if 'is_graduate_level' not in df.columns:
        numeric_part = df['course_id'].str.extract(r'(\d{4})')[0].astype(float)
        df['is_graduate_level'] = numeric_part.fillna(0) > 4000
        df['is_graduate_level'] = df['is_graduate_level'].fillna(False)
    
    # Check for course IDs in query
    mentioned_ids = extract_course_ids(user_query) if use_id_matching else []
    if mentioned_ids:
        print(f"Found course IDs in query: {', '.join(mentioned_ids)}")
    
    # Generate idealized query if requested
    if use_gpt:
        print(f"\nOriginal query: '{user_query}'")
        query_for_search = generate_idealized_course_description(user_query)
        print(f"Idealized query: '{query_for_search}'")
    else:
        query_for_search = user_query
    
    # Perform course ID matching if enabled and IDs found
    id_results = pd.DataFrame()
    if use_id_matching and mentioned_ids:
        id_results = hybrid_search(user_query, df, top_k=top_k)
        if len(id_results) > 0:
            # Add similarity scores for ID matches (set to high value)
            if 'similarity_score' not in id_results.columns:
                id_results['similarity_score'] = 0.95  # High score for exact matches
            print(f"Found {len(id_results)} courses matching mentioned IDs")
    
    # Generate embedding for the query (using cached model and query encoding)
    query_embedding = get_query_embedding(query_for_search)
    
    # Get all course embeddings
    course_embeddings = np.array(df['embedding'].tolist())
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]
    
    # Get top k matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Create semantic results DataFrame
    semantic_results = df.iloc[top_indices].copy()
    semantic_results['similarity_score'] = similarities[top_indices]
    
    # Enhance semantic results by boosting mentioned courses
    if use_id_matching and mentioned_ids:
        semantic_results = enhance_search_results_with_ids(
            user_query, semantic_results, df, boost_factor=1.5
        )
    
    # Reorder columns for better readability
    semantic_results = semantic_results[
        [
            'course_id',
            'title',
            'description',
            'prerequisites',
            'meeting_times',
            'department',
            'is_graduate_level',
            'similarity_score',
            'embedding'
        ]
    ]
    
    # Combine ID matches with semantic results
    if use_id_matching and len(id_results) > 0:
        # Ensure id_results has same columns
        id_results = id_results[
            [
                'course_id',
                'title',
                'description',
                'prerequisites',
                'meeting_times',
                'department',
                'is_graduate_level',
                'similarity_score',
                'embedding'
            ]
        ]
        results = combine_id_and_semantic_results(id_results, semantic_results, top_k=top_k)
    else:
        results = semantic_results
    
    return results


def print_course_results(results_df):
    """Pretty print the search results."""
    print(f"\n{'='*80}")
    print(f"Found {len(results_df)} matching courses:")
    print(f"{'='*80}\n")
    
    for idx, row in results_df.iterrows():
        print(f"Course ID: {row['course_id']}")
        print(f"Title: {row['title']}")
        print(f"Similarity: {row['similarity_score']:.3f}")
        print(f"Prerequisites: {row['prerequisites']}")
        
        # Print meeting times
        if row['meeting_times']:
            print("Meeting Times:")
            for section, times in row['meeting_times'].items():
                print(f"  Section {section}: {', '.join(times)}")
        
        # Print description preview
        desc = row['description'][:200] if row['description'] else "No description available"
        print(f"Description: {desc}...")
        print(f"{'-'*80}\n")


def main():

    num_results = int(os.getenv('NUM_RESULTS', 10))
    
    # Check if query was provided as cli arg
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        # Interactive mode: prompt for query
        user_query = input("Enter your course search query: ").strip()
        if not user_query:
            print("No query provided. Exiting.")
            return
    
    # Perform the search
    results = search_courses(
        user_query=user_query,
        df_path='data/202508_processed.pkl',
        top_k=num_results,
        use_gpt=True
    )
    
    # Print the results
    print_course_results(results)


if __name__ == "__main__":
    main()