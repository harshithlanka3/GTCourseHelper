import openai
from typing import Optional

class QueryEnhancer:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    def generate_ideal_course_description(self, user_query: str, context: Dict = None) -> str:
        """
        Generate an 'ideal' course description based on user's natural language query
        This bridges the semantic gap between casual queries and formal course descriptions
        """
        prompt = f"""
        Given a student's query about courses they want to take, generate a detailed, 
        formal course description that would match their interests. Include relevant 
        topics, skills, and academic terminology.
        
        Student Query: {user_query}
        
        Additional Context:
        - Student's completed courses: {context.get('completed_courses', [])}
        - Student's major: {context.get('major', 'Unknown')}
        - Semester: {context.get('semester', 'Unknown')}
        
        Generate a formal course description (2-3 sentences) that captures what the 
        student is looking for:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an academic course description generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def extract_query_constraints(self, user_query: str) -> Dict:
        """
        Extract structured constraints from natural language query
        """
        prompt = f"""
        Extract specific constraints from this course query. Return a JSON object with:
        - time_preferences: [] (e.g., ["morning", "afternoon", "evening"])
        - days_preferences: [] (e.g., ["MW", "TR"])
        - professor_names: []
        - course_numbers: []
        - level_constraints: [] (e.g., [3000, 4000])
        - prerequisite_check: bool (whether to check prerequisites)
        
        Query: {user_query}
        
        Return only valid JSON:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a query parser. Extract constraints as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)