import os
import re
import sys
import argparse
from io import StringIO
import pandas as pd
import numpy as np
from openai import OpenAI
import dotenv
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional
from search_courses import search_courses
from recommend_courses import get_client, deduplicate_courses, build_prompt, call_gpt_recommendations

dotenv.load_dotenv()


class CourseChatbot:
    """A chatbot for course scheduling and information queries."""
    
    def __init__(self, df_path: str = "data/202508_processed.pkl"):
        self.df_path = df_path
        self.df = pd.read_pickle(df_path)
        self.client = get_client()
        print(f"Loaded {len(self.df)} courses")
    
    def classify_intent(self, query: str) -> str:
        """
        Classify the user's query into one of several intents.
        Returns: 'prerequisite', 'course_id', 'time_section', 'recommendation', or 'general'
        """
        query_lower = query.lower()
        
        # Check for prerequisite/scheduling question keywords
        prereq_keywords = ['haven\'t taken', 'havent taken', 'haven\'t', 'havent', 'without', 
                          'can i take', 'can i still take', 'prerequisite', 'prereq', 
                          'need to take', 'required', 'before taking', 'before i take']
        course_id_pattern = r'\b([A-Z]{2,4})\s*(\d{4})\b'
        course_ids = re.findall(course_id_pattern, query, re.IGNORECASE)
        
        # If there are multiple course IDs and prerequisite keywords, it's a prerequisite question
        if len(course_ids) >= 2 and any(keyword in query_lower for keyword in prereq_keywords):
            return 'prerequisite'
        
        # Check for course ID patterns (e.g., "CS 3510", "MATH2106", "what is CS 1332")
        if re.search(course_id_pattern, query, re.IGNORECASE):
            return 'course_id'
        
        # Check for time/section related keywords
        time_keywords = ['time', 'section', 'schedule', 'available', 'monday', 'tuesday', 
                        'wednesday', 'thursday', 'friday', 'mwf', 'tr', 'morning', 
                        'afternoon', 'evening', 'when', 'what times', 'when does']
        if any(keyword in query_lower for keyword in time_keywords):
            # But not if it's clearly a course ID query
            if not re.search(course_id_pattern, query, re.IGNORECASE):
                return 'time_section'
        
        # Check for recommendation keywords
        rec_keywords = ['recommend', 'suggest', 'interested', 'want to learn', 
                       'looking for', 'find courses', 'courses about', 'what courses']
        if any(keyword in query_lower for keyword in rec_keywords):
            return 'recommendation'
        
        # Default to general
        return 'general'
    
    def extract_course_id(self, query: str) -> Optional[str]:
        """
        Extract first course ID from query using regex.
        Returns normalized course ID (e.g., "CS 3510") or None.
        """
        # Pattern: 2-4 letters, optional space, 4 digits
        pattern = r'\b([A-Z]{2,4})\s*(\d{4})\b'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            dept = match.group(1).upper()
            num = match.group(2)
            return f"{dept} {num}"
        return None
    
    def extract_all_course_ids(self, query: str) -> List[str]:
        """
        Extract all course IDs from query using regex.
        Returns list of normalized course IDs (e.g., ["CS 3510", "CS 2050"]).
        """
        pattern = r'\b([A-Z]{2,4})\s*(\d{4})\b'
        matches = re.findall(pattern, query, re.IGNORECASE)
        course_ids = []
        for match in matches:
            dept = match[0].upper()
            num = match[1]
            course_id = f"{dept} {num}"
            if course_id not in course_ids:  # Avoid duplicates
                course_ids.append(course_id)
        return course_ids
    
    def fuzzy_match_course_id(self, query_id: str, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Fuzzy match a course ID against all course IDs in the database.
        Returns list of (course_id, similarity_score) tuples.
        """
        matches = []
        query_normalized = query_id.replace(' ', '').upper()
        
        for course_id in self.df['course_id'].unique():
            course_normalized = course_id.replace(' ', '').upper()
            similarity = SequenceMatcher(None, query_normalized, course_normalized).ratio()
            
            # Also check if query is substring
            if query_normalized in course_normalized or course_normalized in query_normalized:
                similarity = max(similarity, 0.8)
            
            if similarity >= threshold:
                matches.append((course_id, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def lookup_course_by_id(self, course_id: str) -> Optional[pd.DataFrame]:
        """Look up a course by exact course ID."""
        result = self.df[self.df['course_id'] == course_id]
        return result if len(result) > 0 else None
    
    def format_course_info(self, row: pd.Series) -> str:
        """Format a course row into a readable string."""
        lines = [
            f"**{row['course_id']}: {row['title']}**",
            f"Description: {row['description'] or 'No description available'}",
        ]
        
        # Format sections
        if row.get('meeting_times'):
            sections = []
            for sec, times in row['meeting_times'].items():
                times_str = ", ".join(times) if times else "TBA"
                sections.append(f"Section {sec}: {times_str}")
            lines.append(f"Sections: {'; '.join(sections)}")
        else:
            lines.append("Sections: None listed")
        
        # Format prerequisites
        prereqs = row.get('prerequisites') or ""
        lines.append(f"Prerequisites: {prereqs or 'None explicitly stated'}")
        
        return "\n".join(lines)
    
    def handle_course_id_query(self, query: str) -> str:
        """
        Handle queries asking about a specific course ID.
        Supports exact match, regex match, and fuzzy matching.
        If multiple course IDs are found, returns info for all of them.
        """
        # Extract all course IDs
        course_ids = self.extract_all_course_ids(query)
        
        if not course_ids:
            # Try fuzzy matching on the first extracted ID
            extracted_id = self.extract_course_id(query)
            if extracted_id:
                fuzzy_matches = self.fuzzy_match_course_id(extracted_id, threshold=0.5)
                if fuzzy_matches:
                    response = f"Did you mean one of these courses?\n\n"
                    for course_id, score in fuzzy_matches[:5]:  # Top 5 matches
                        course_df = self.lookup_course_by_id(course_id)
                        if course_df is not None and len(course_df) > 0:
                            response += f"**{course_id}** (similarity: {score:.2f})\n"
                    response += "\nPlease specify which course you'd like information about."
                    return response
            return "I couldn't find a matching course ID in your query. Please try again with a format like 'CS 3510' or 'MATH 2106'."
        
        # Look up all found course IDs
        response = ""
        found_any = False
        
        for course_id in course_ids:
            exact_match = self.lookup_course_by_id(course_id)
            if exact_match is not None and len(exact_match) > 0:
                found_any = True
                if len(course_ids) > 1:
                    response += f"**{course_id}:**\n\n"
                else:
                    response += "Found course:\n\n"
                for _, row in exact_match.iterrows():
                    response += self.format_course_info(row) + "\n\n"
        
        if not found_any:
            return "I couldn't find any of the mentioned courses in the database. Please check the course IDs and try again."
        
        return response
    
    def handle_prerequisite_query(self, query: str) -> str:
        """
        Handle queries about prerequisites and whether courses can be taken.
        Examples: "Can I take CS 3510 if I haven't taken CS 2050?"
        """
        course_ids = self.extract_all_course_ids(query)
        query_lower = query.lower()
        
        if len(course_ids) < 2:
            # Not enough course IDs, fall back to general handler
            return self.handle_general_query(query)
        
        # Determine which course the user wants to take and which they haven't taken
        target_course = None
        missing_course = None
        
        # Find course mentioned with "haven't taken" or "without" (this is what they're missing)
        if "haven't taken" in query_lower or "havent taken" in query_lower:
            missing_idx = query_lower.find("haven't taken")
            if missing_idx == -1:
                missing_idx = query_lower.find("havent taken")
            if missing_idx != -1:
                remaining_query = query[missing_idx:]
                missing_ids = self.extract_all_course_ids(remaining_query)
                if missing_ids:
                    missing_course = missing_ids[0]
        
        if "without" in query_lower:
            # Find course after "without"
            without_idx = query_lower.find("without")
            if without_idx != -1:
                remaining_query = query[without_idx:]
                missing_ids = self.extract_all_course_ids(remaining_query)
                if missing_ids:
                    missing_course = missing_ids[0]
        
        # Look for patterns like "take X" or "can i take X" (this is what they want to take)
        # Search for "can i still take" first, then "can i take", to avoid matching "take" in "haven't taken"
        if "can i still take" in query_lower:
            take_idx = query_lower.find("can i still take")
            if take_idx != -1:
                # Look for course ID after "can i still take"
                remaining_query = query[take_idx + len("can i still take"):]
                target_ids = self.extract_all_course_ids(remaining_query)
                if target_ids:
                    target_course = target_ids[0]
        elif "can i take" in query_lower:
            take_idx = query_lower.find("can i take")
            if take_idx != -1:
                # Look for course ID after "can i take"
                remaining_query = query[take_idx + len("can i take"):]
                target_ids = self.extract_all_course_ids(remaining_query)
                if target_ids:
                    target_course = target_ids[0]
        elif "take" in query_lower:
            # Fallback: find "take" but make sure it's not part of "haven't taken"
            # Look for "take" that comes after "haven't taken" if that phrase exists
            if "haven't taken" in query_lower or "havent taken" in query_lower:
                missing_end_idx = query_lower.find("haven't taken")
                if missing_end_idx == -1:
                    missing_end_idx = query_lower.find("havent taken")
                if missing_end_idx != -1:
                    missing_end_idx += len("haven't taken") if "haven't taken" in query_lower else len("havent taken")
                    # Look for "take" after this point
                    take_idx = query_lower.find("take", missing_end_idx)
                    if take_idx != -1:
                        remaining_query = query[take_idx:]
                        target_ids = self.extract_all_course_ids(remaining_query)
                        if target_ids:
                            target_course = target_ids[0]
            else:
                # No "haven't taken", just find any "take"
                take_idx = query_lower.find("take")
                if take_idx != -1:
                    remaining_query = query[take_idx:]
                    target_ids = self.extract_all_course_ids(remaining_query)
                    if target_ids:
                        target_course = target_ids[0]
        
        # If we couldn't determine, use heuristics based on course ID positions
        # Usually the course they want to take comes last, and the one they haven't taken comes first
        if not target_course:
            target_course = course_ids[-1]  # Usually the last mentioned is what they want to take
        if not missing_course:
            # If we found a target, the missing one is likely the other course
            if target_course and len(course_ids) >= 2:
                missing_course = course_ids[0] if course_ids[0] != target_course else course_ids[1]
            else:
                missing_course = course_ids[0]  # First mentioned is what they haven't taken
        
        # Safety check: ensure target and missing are different
        if target_course == missing_course and len(course_ids) >= 2:
            # If they're the same, use the other course
            if course_ids[0] == target_course:
                missing_course = course_ids[1]
            else:
                target_course = course_ids[-1]
                missing_course = course_ids[0]
        
        # Look up both courses
        target_df = self.lookup_course_by_id(target_course)
        missing_df = self.lookup_course_by_id(missing_course)
        
        if target_df is None or len(target_df) == 0:
            return f"I couldn't find information about {target_course}. Please check the course ID and try again."
        
        if missing_df is None or len(missing_df) == 0:
            return f"I couldn't find information about {missing_course}. Please check the course ID and try again."
        
        target_row = target_df.iloc[0]
        missing_row = missing_df.iloc[0]
        
        # Get prerequisites
        prereqs = target_row.get('prerequisites') or ""
        prereqs_upper = prereqs.upper()
        
        # Check if the missing course is in the prerequisites
        missing_course_upper = missing_course.upper().replace(' ', '')
        prereqs_no_spaces = prereqs_upper.replace(' ', '')
        
        has_prereq = missing_course_upper in prereqs_no_spaces
        
        # Build response
        response = f"**{target_course}: {target_row['title']}**\n\n"
        response += f"Prerequisites: {prereqs or 'None explicitly stated'}\n\n"
        
        if has_prereq:
            response += f"❌ **No, you cannot take {target_course} without {missing_course}.**\n\n"
            response += f"{target_course} requires {missing_course} as a prerequisite. "
            response += f"You would need to complete {missing_course} first.\n\n"
        else:
            if prereqs:
                response += f"✅ **Yes, you can take {target_course} without {missing_course}.**\n\n"
                response += f"{target_course} does not require {missing_course} as a prerequisite. "
                response += f"However, it does have other prerequisites: {prereqs}\n\n"
            else:
                response += f"✅ **Yes, you can take {target_course} without {missing_course}.**\n\n"
                response += f"{target_course} has no prerequisites listed, so you can take it without {missing_course}.\n\n"
        
        response += f"\n**Course Information:**\n\n"
        response += f"**{target_course}: {target_row['title']}**\n"
        response += f"Description: {target_row['description'][:200] if target_row['description'] else 'No description'}...\n\n"
        
        response += f"**{missing_course}: {missing_row['title']}**\n"
        response += f"Description: {missing_row['description'][:200] if missing_row['description'] else 'No description'}...\n"
        
        return response
    
    def parse_time_constraints(self, query: str) -> Dict[str, any]:
        """
        Parse time constraints from query.
        Returns dict with keys: days, time_range, keywords
        """
        query_lower = query.lower()
        constraints = {
            'days': [],
            'time_range': None,
            'keywords': []
        }
        
        # Days
        day_map = {
            'monday': 'M', 'tuesday': 'T', 'wednesday': 'W', 
            'thursday': 'R', 'friday': 'F', 'saturday': 'S', 'sunday': 'U'
        }
        for day_name, day_code in day_map.items():
            if day_name in query_lower:
                constraints['days'].append(day_code)
        
        # Common day combinations
        if 'mwf' in query_lower or 'monday, wednesday, friday' in query_lower:
            constraints['days'] = ['M', 'W', 'F']
        if 'tr' in query_lower or 'tuesday, thursday' in query_lower:
            constraints['days'] = ['T', 'R']
        
        # Time ranges
        time_patterns = [
            (r'morning', 'morning'),
            (r'afternoon', 'afternoon'),
            (r'evening', 'evening'),
            (r'(\d{1,2}):?(\d{2})?\s*(am|pm)', 'specific_time'),
        ]
        
        for pattern, label in time_patterns:
            if re.search(pattern, query_lower):
                constraints['keywords'].append(label)
        
        return constraints
    
    def filter_courses_by_time(self, constraints: Dict[str, any], top_k: int = 20) -> pd.DataFrame:
        """
        Filter courses that match time constraints.
        """
        matching_courses = []
        
        for _, row in self.df.iterrows():
            meeting_times = row.get('meeting_times') or {}
            if not meeting_times:
                continue
            
            matches = False
            for section, times in meeting_times.items():
                for time_str in times:
                    # Check day matches
                    if constraints['days']:
                        time_upper = time_str.upper()
                        if any(day in time_upper for day in constraints['days']):
                            matches = True
                            break
                    else:
                        # If no specific days, include all
                        matches = True
                    
                    # Check time range keywords
                    if constraints['keywords']:
                        time_lower = time_str.lower()
                        if any(keyword in time_lower for keyword in constraints['keywords']):
                            matches = True
                            break
                
                if matches:
                    break
            
            if matches:
                matching_courses.append(row)
        
        if matching_courses:
            result_df = pd.DataFrame(matching_courses)
            return result_df.head(top_k)
        return pd.DataFrame()
    
    def handle_time_section_query(self, query: str) -> str:
        """
        Handle queries about courses available during specific times/sections.
        """
        constraints = self.parse_time_constraints(query)
        
        # Also try to extract course ID if present
        course_id = self.extract_course_id(query)
        
        if course_id:
            # User asking about specific course's times
            course_df = self.lookup_course_by_id(course_id)
            if course_df is not None and len(course_df) > 0:
                response = f"Here are the available sections for {course_id}:\n\n"
                for _, row in course_df.iterrows():
                    response += self.format_course_info(row) + "\n\n"
                return response
        
        # Filter courses by time constraints
        filtered = self.filter_courses_by_time(constraints, top_k=20)
        
        if len(filtered) == 0:
            return "I couldn't find any courses matching those time constraints. Try being more specific or checking different times."
        
        response = f"Found {len(filtered)} courses matching your time constraints:\n\n"
        
        for idx, (_, row) in enumerate(filtered.iterrows(), 1):
            response += f"{idx}. {self.format_course_info(row)}\n\n"
            if idx >= 10:  # Limit to top 10
                response += f"... and {len(filtered) - 10} more courses.\n"
                break
        
        return response
    
    def handle_recommendation_query(self, query: str) -> str:
        """
        Handle course recommendation queries using existing recommendation system.
        """
        # Suppress print statements during search
        # Temporarily redirect stdout to suppress search_courses print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Use existing semantic search
            candidates = search_courses(
                user_query=query,
                df_path=self.df_path,
                top_k=50,
                use_gpt=True,
            )
        finally:
            sys.stdout = old_stdout
        
        candidates = deduplicate_courses(candidates)
        
        if len(candidates) == 0:
            return "I couldn't find any courses matching your query. Try rephrasing or being more specific."
        
        # Use GPT to generate recommendations
        prompt = build_prompt(query, candidates)
        recommendations = call_gpt_recommendations(self.client, prompt)
        
        return recommendations
    
    def handle_general_query(self, query: str) -> str:
        """
        Handle general questions using GPT with course data context.
        """
        # Suppress print statements during search
        # Temporarily redirect stdout to suppress search_courses print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Get relevant courses for context
            candidates = search_courses(
                user_query=query,
                df_path=self.df_path,
                top_k=10,
                use_gpt=True,
            )
        finally:
            sys.stdout = old_stdout
        
        # Build context from top courses
        context_lines = []
        for idx, (_, row) in enumerate(candidates.head(5).iterrows(), 1):
            context_lines.append(
                f"{row['course_id']}: {row['title']}\n"
                f"Description: {row['description'][:200] if row['description'] else 'No description'}"
            )
        
        context = "\n\n".join(context_lines)
        
        prompt = f"""You are a helpful academic advisor at Georgia Tech. Answer the student's question about courses and scheduling.

Available Course Information (for context):
{context}

Student Question: {query}

Provide a helpful, accurate answer. If the question is about specific courses, use the course information provided. If you need more specific information, suggest the student ask about a particular course ID or time frame.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I encountered an error processing your question. Please try rephrasing it. Error: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """
        Main entry point to process a user query.
        Routes to appropriate handler based on intent.
        """
        intent = self.classify_intent(query)
        
        if intent == 'prerequisite':
            return self.handle_prerequisite_query(query)
        elif intent == 'course_id':
            return self.handle_course_id_query(query)
        elif intent == 'time_section':
            return self.handle_time_section_query(query)
        elif intent == 'recommendation':
            return self.handle_recommendation_query(query)
        else:
            return self.handle_general_query(query)
    
    def run_interactive(self):
        """Run an interactive chat session."""
        print("\n" + "="*80)
        print("Georgia Tech Course Helper Chatbot")
        print("="*80)
        print("\nI can help you with:")
        print("  - Looking up specific courses (e.g., 'What is CS 3510?')")
        print("  - Finding courses available at specific times (e.g., 'What courses are available on MWF mornings?')")
        print("  - Getting course recommendations (e.g., 'I want to learn about machine learning')")
        print("  - General scheduling questions")
        print("\nType 'quit' or 'exit' to end the session.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! Good luck with your course planning!")
                    break
                
                if not query:
                    continue
                
                print("\nBot: ", end="")
                response = self.process_query(query)
                print(response)
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! Good luck with your course planning!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or rephrase your question.\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive chatbot for GT course scheduling")
    parser.add_argument("--df-path", type=str, default="data/202508_processed.pkl",
                       help="Path to processed courses pickle")
    parser.add_argument("--query", type=str, default=None,
                       help="Single query to process (non-interactive mode)")
    args = parser.parse_args()
    
    chatbot = CourseChatbot(df_path=args.df_path)
    
    if args.query:
        # Non-interactive mode
        response = chatbot.process_query(args.query)
        print(response)
    else:
        # Interactive mode
        chatbot.run_interactive()


if __name__ == "__main__":
    main()

