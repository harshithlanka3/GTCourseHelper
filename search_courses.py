import os
import sys
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
import pickle
from sentence_transformers import SentenceTransformer


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
    prompt = f"""Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a general description of the course. Limit the description to be less than 200 words.

Student Request:
{user_query}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating academic course descriptions for university catalogs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        idealized_description = response.choices[0].message.content.strip()
        return idealized_description
    except Exception as e:
        print(f"Error generating idealized description: {e}")
        return user_query  # Fallback to original query


def search_courses(user_query, df_path='data/202508_processed.pkl', top_k=50, use_gpt=True):
    """
    Search for courses matching a user's query using semantic similarity.
    
    Args:
        user_query (str): Natural language description of course interests
        df_path (str): Path to the processed courses pickle file
        top_k (int): Number of top results to return
        use_gpt (bool): Whether to use GPT to refine the query first
        
    Returns:
        pandas.DataFrame: DataFrame containing the top_k matching courses
    """
    # Load the processed courses
    print(f"Loading courses from {df_path}...")
    df = pd.read_pickle(df_path)
    print(f"Loaded {len(df)} courses")
    
    # Generate idealized query if requested
    if use_gpt:
        print(f"\nOriginal query: '{user_query}'")
        query_for_search = generate_idealized_course_description(user_query)
        print(f"Idealized query: '{query_for_search}'")
    else:
        query_for_search = user_query
    
    # Generate embedding for the query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_for_search).reshape(1, -1)
    
    # Get all course embeddings
    course_embeddings = np.array(df['embedding'].tolist())
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]
    
    # Get top k matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Create results DataFrame
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    # Reorder columns for better readability
    results = results[['course_id', 'title', 'description', 'prerequisites', 
                       'meeting_times', 'similarity_score', 'embedding']]
    
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