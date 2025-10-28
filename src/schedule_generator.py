from datetime import datetime, time
import itertools

class ScheduleGenerator:
    def __init__(self, courses_df: pd.DataFrame):
        self.courses_df = courses_df
        
    def generate_schedule(self, selected_course_ids: List[str], 
                         preferences: Dict) -> List[Dict]:
        """
        Generate valid schedules from selected courses
        """
        courses = self.courses_df[
            self.courses_df['course_number'].isin(selected_course_ids)
        ]
        
        # Group courses by their sections
        course_sections = {}
        for course_num in selected_course_ids:
            sections = courses[courses['course_number'] == course_num]
            course_sections[course_num] = sections.to_dict('records')
        
        # Generate all possible combinations
        all_combinations = list(itertools.product(*course_sections.values()))
        
        # Filter valid schedules (no time conflicts)
        valid_schedules = []
        for combination in all_combinations:
            if self.is_valid_schedule(combination):
                schedule = {
                    'courses': combination,
                    'score': self.score_schedule(combination, preferences)
                }
                valid_schedules.append(schedule)
        
        # Sort by score
        valid_schedules.sort(key=lambda x: x['score'], reverse=True)
        
        return valid_schedules[:5]  # Return top 5 schedules
    
    def is_valid_schedule(self, courses: List[Dict]) -> bool:
        """Check if courses have time conflicts"""
        for i, course1 in enumerate(courses):
            for course2 in courses[i+1:]:
                if self.has_time_conflict(course1, course2):
                    return False
        return True
    
    def has_time_conflict(self, course1: Dict, course2: Dict) -> bool:
        """Check if two courses have overlapping times"""
        # Check day overlap
        days1 = set(course1['days'])
        days2 = set(course2['days'])
        
        if not days1.intersection(days2):
            return False
        
        # Check time overlap
        start1, end1 = course1['start_time'], course1['end_time']
        start2, end2 = course2['start_time'], course2['end_time']
        
        return not (end1 <= start2 or end2 <= start1)
    
    def score_schedule(self, courses: List[Dict], preferences: Dict) -> float:
        """Score schedule based on user preferences"""
        score = 0.0
        
        # Prefer certain professors
        if 'preferred_professors' in preferences:
            for course in courses:
                if course['professor'] in preferences['preferred_professors']:
                    score += 10
        
        # Prefer certain times
        if 'preferred_times' in preferences:
            for course in courses:
                if self.matches_time_preference(course, preferences['preferred_times']):
                    score += 5
        
        # Minimize gaps between classes
        score -= self.calculate_gap_penalty(courses)
        
        return score