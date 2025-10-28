class CourseAdvisorRAG:
    def __init__(self, retriever: HybridRetriever, schedule_gen: ScheduleGenerator):
        self.retriever = retriever
        self.schedule_gen = schedule_gen
        self.conversation_history = []
        
    def process_query(self, user_query: str, user_context: Dict) -> str:
        """
        Main entry point for processing user queries
        """
        # Classify query type
        query_type = self.classify_query(user_query)
        
        if query_type == 'course_search':
            return self.handle_course_search(user_query, user_context)
        elif query_type == 'schedule_generation':
            return self.handle_schedule_generation(user_query, user_context)
        elif query_type == 'prerequisite_check':
            return self.handle_prerequisite_check(user_query, user_context)
        elif query_type == 'general_question':
            return self.handle_general_question(user_query, user_context)
        else:
            return self.handle_conversational(user_query, user_context)
    
    def classify_query(self, query: str) -> str:
        """Classify the type of query using LLM"""
        prompt = f"""
        Classify this student query into one of these categories:
        - course_search: Looking for specific courses or topics
        - schedule_generation: Wants a complete schedule created
        - prerequisite_check: Asking about prerequisites or requirements
        - general_question: General questions about courses, requirements, etc.
        - conversational: Follow-up or clarification
        
        Query: {query}
        
        Return only the category name:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def handle_course_search(self, query: str, context: Dict) -> str:
        """Handle course search queries"""
        # Retrieve relevant courses
        courses = self.retriever.retrieve(query, context, top_k=10)
        
        if courses.empty:
            return "I couldn't find any courses matching your criteria. Could you provide more details?"
        
        # Format response
        response = "Based on your query, here are the most relevant courses:\n\n"
        
        for idx, row in courses.iterrows():
            response += f"**{row['course_number']}: {row['course_name']}**\n"
            response += f"Professor: {row['professor']}\n"
            response += f"Time: {row['days']} {row['start_time']}-{row['end_time']}\n"
            response += f"Description: {row['description'][:150]}...\n"
            response += f"Prerequisites: {', '.join(row['prerequisites']) if row['prerequisites'] else 'None'}\n\n"
        
        # Add conversational follow-up
        response += "\nWould you like more details about any of these courses, or should I help you create a schedule with some of them?"
        
        return response
    
    def handle_schedule_generation(self, query: str, context: Dict) -> str:
        """Generate complete schedules"""
        # First, extract desired courses from query
        prompt = f"""
        Extract the course numbers or topics the student wants in their schedule.
        
        Query: {query}
        Context: {context}
        
        Return a list of course numbers or topic areas:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Get courses for each topic/number
        all_courses = []
        topics = response.choices[0].message.content.split('\n')
        
        for topic in topics:
            courses = self.retriever.retrieve(topic, context, top_k=3)
            all_courses.extend(courses['course_number'].tolist())
        
        # Generate schedules
        schedules = self.schedule_gen.generate_schedule(
            all_courses, 
            context.get('preferences', {})
        )
        
        if not schedules:
            return "I couldn't create a valid schedule with those courses. There might be time conflicts."
        
        # Format response
        response = "Here are the top schedule options for you:\n\n"
        
        for i, schedule in enumerate(schedules[:3], 1):
            response += f"**Option {i}:**\n"
            for course in schedule['courses']:
                response += f"- {course['course_number']}: {course['course_name']} "
                response += f"({course['days']} {course['start_time']}-{course['end_time']})\n"
            response += f"Schedule Score: {schedule['score']:.2f}\n\n"
        
        return response