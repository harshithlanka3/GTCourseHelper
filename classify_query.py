"""
Query Classification Module

Classifies user queries into different types to determine appropriate response behavior.
"""
import re
from typing import List
from course_id_matcher import extract_course_ids


def classify_query(message: str) -> str:
    """
    Classify a user query to determine the appropriate response behavior.
    
    Args:
        message (str): User's message/query
        
    Returns:
        str: One of:
            - "conversation": Non-course related chat
            - "single_course": Exactly one valid course ID detected
            - "compare": Multiple course IDs detected
            - "general_recommendation": Course-related but needs full recommendation pipeline
    """
    # Extract course IDs from the message
    course_ids = extract_course_ids(message)
    
    # Check for multiple course IDs (comparison)
    if len(course_ids) > 1:
        return "compare"
    
    # Check for exactly one course ID (single course lookup)
    if len(course_ids) == 1:
        return "single_course"
    
    # Check if message contains course-related keywords
    course_keywords = [
        'course', 'class', 'elective', 'prerequisite', 'prereq',
        'recommend', 'recommendation', 'suggest', 'suggestion',
        'choose', 'select', 'pick', 'take', 'enroll',
        'best', 'good', 'fit', 'match', 'suitable',
        'learn', 'study', 'teach', 'cover', 'focus',
        'major', 'minor', 'degree', 'program', 'curriculum',
        'semester', 'quarter', 'credit', 'hours', 'units'
    ]
    
    message_lower = message.lower()
    has_course_keywords = any(keyword in message_lower for keyword in course_keywords)
    
    # Check for guidance/recommendation keywords specifically
    guidance_keywords = [
        'recommend', 'recommendation', 'suggest', 'suggestion',
        'choose', 'select', 'pick', 'what should', 'what to take',
        'best', 'good for', 'fit', 'match', 'suitable'
    ]
    has_guidance_keywords = any(keyword in message_lower for keyword in guidance_keywords)
    
    # If no course-related content at all, treat as conversation
    if not has_course_keywords:
        return "conversation"
    
    # If has guidance keywords, use recommendation pipeline
    if has_guidance_keywords:
        return "general_recommendation"
    
    # If has course keywords but no specific guidance, still use recommendation
    # (user might be asking about courses in general)
    return "general_recommendation"

