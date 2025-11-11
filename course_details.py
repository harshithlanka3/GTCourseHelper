"""
Course Details Module

Provides functions to retrieve detailed information about specific courses.
"""
import pandas as pd
import os
import sys
import re

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Set default data path relative to project root
DEFAULT_DF_PATH = os.path.join(project_root, 'data', '202508_processed.pkl')


def get_course_details(course_id: str, df_path: str = DEFAULT_DF_PATH) -> dict:
    """
    Get detailed information about a specific course.
    
    Args:
        course_id (str): Course ID in format "PREFIX NUMBER" (e.g., "CS 1332")
        df_path (str): Path to the processed courses pickle file
        
    Returns:
        dict: Course details with keys:
            - course_id: Course identifier
            - title: Course title
            - description: Course description
            - prerequisites: Prerequisite requirements
            - meeting_times: Dictionary of section meeting times
            - found: Boolean indicating if course was found
    """
    try:
        # Load the processed courses
        df = pd.read_pickle(df_path)
        
        # Normalize course_id format (ensure space between prefix and number)
        course_id_normalized = course_id.upper().strip()
        if ' ' not in course_id_normalized and len(course_id_normalized) > 4:
            # Try to insert space: "CS1332" -> "CS 1332"
            match = re.match(r'([A-Z]{2,4})(\d{4})', course_id_normalized)
            if match:
                course_id_normalized = f"{match.group(1)} {match.group(2)}"
        
        # Search for exact match
        course_row = df[df['course_id'] == course_id_normalized]
        
        if len(course_row) == 0:
            # Try case-insensitive search
            course_row = df[df['course_id'].str.upper() == course_id_normalized]
        
        if len(course_row) == 0:
            return {
                'course_id': course_id,
                'title': None,
                'description': None,
                'prerequisites': None,
                'meeting_times': None,
                'found': False
            }
        
        # Get the first (and should be only) match
        row = course_row.iloc[0]
        
        return {
            'course_id': row['course_id'],
            'title': row['title'],
            'description': row.get('description', ''),
            'prerequisites': row.get('prerequisites', ''),
            'meeting_times': row.get('meeting_times', {}),
            'found': True
        }
    except Exception as e:
        print(f"Error getting course details for {course_id}: {e}")
        return {
            'course_id': course_id,
            'title': None,
            'description': None,
            'prerequisites': None,
            'meeting_times': None,
            'found': False
        }


def format_course_details(course_details: dict) -> str:
    """
    Format course details into a readable string response.
    
    Args:
        course_details (dict): Course details dictionary from get_course_details()
        
    Returns:
        str: Formatted course information
    """
    if not course_details.get('found', False):
        return f"Sorry, I couldn't find information about {course_details['course_id']}. Please check the course ID and try again."
    
    parts = []
    
    # Course ID and Title
    parts.append(f"**{course_details['course_id']}: {course_details['title']}**")
    parts.append("")
    
    # Description
    if course_details.get('description'):
        parts.append(f"**Description:**\n{course_details['description']}")
        parts.append("")
    
    # Prerequisites
    prereqs = course_details.get('prerequisites', '')
    if prereqs:
        parts.append(f"**Prerequisites:** {prereqs}")
    else:
        parts.append("**Prerequisites:** None explicitly stated")
    parts.append("")
    
    # Meeting Times
    meeting_times = course_details.get('meeting_times', {})
    if meeting_times:
        parts.append("**Meeting Times:**")
        for section, times in meeting_times.items():
            times_str = ", ".join(times) if times else "TBA"
            parts.append(f"  Section {section}: {times_str}")
    else:
        parts.append("**Meeting Times:** None listed")
    
    return "\n".join(parts)

