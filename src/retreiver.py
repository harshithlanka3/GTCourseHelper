class HybridRetriever:
    def __init__(self, courses_df: pd.DataFrame, embedding_engine: CourseEmbeddingEngine, 
                 query_enhancer: QueryEnhancer):
        self.courses_df = courses_df
        self.embedding_engine = embedding_engine
        self.query_enhancer = query_enhancer
        
    def retrieve(self, user_query: str, user_context: Dict, top_k: int = 20) -> pd.DataFrame:
        """
        Multi-stage retrieval process:
        1. Extract constraints from query
        2. Generate ideal course description
        3. Semantic search with embeddings
        4. Apply hard filters
        5. Re-rank with LLM
        """
        
        # Stage 1: Extract constraints
        constraints = self.query_enhancer.extract_query_constraints(user_query)
        
        # Stage 2: Generate ideal description
        ideal_description = self.query_enhancer.generate_ideal_course_description(
            user_query, user_context
        )
        
        # Stage 3: Embedding-based retrieval
        query_embedding = self.embedding_engine.encoder.encode([ideal_description])
        indices, scores = self.embedding_engine.search_similar_courses(
            query_embedding[0], k=top_k * 2  # Retrieve more for filtering
        )
        
        # Get candidate courses
        candidate_courses = self.courses_df.iloc[indices].copy()
        candidate_courses['similarity_score'] = scores
        
        # Stage 4: Apply hard constraints
        filtered_courses = self.apply_constraints(candidate_courses, constraints, user_context)
        
        # Stage 5: Re-rank with LLM (optional but recommended)
        if len(filtered_courses) > 0:
            filtered_courses = self.rerank_with_llm(
                filtered_courses, user_query, user_context
            )
        
        return filtered_courses.head(top_k)
    
    def apply_constraints(self, courses: pd.DataFrame, constraints: Dict, 
                         user_context: Dict) -> pd.DataFrame:
        """Apply hard filtering based on extracted constraints"""
        
        filtered = courses.copy()
        
        # Time constraints
        if constraints.get('time_preferences'):
            # Filter by time slots
            pass
        
        # Day constraints  
        if constraints.get('days_preferences'):
            filtered = filtered[filtered['days'].isin(constraints['days_preferences'])]
        
        # Level constraints
        if constraints.get('level_constraints'):
            filtered = filtered[filtered['level'].isin(constraints['level_constraints'])]
        
        # Prerequisites check
        if constraints.get('prerequisite_check'):
            completed = set(user_context.get('completed_courses', []))
            filtered = filtered[
                filtered['prerequisites'].apply(
                    lambda x: all(p in completed for p in x) if x else True
                )
            ]
        
        # Specific course numbers
        if constraints.get('course_numbers'):
            course_filter = filtered['course_number'].str.contains(
                '|'.join(constraints['course_numbers']), case=False
            )
            filtered = filtered[course_filter]
        
        return filtered
    
    def rerank_with_llm(self, courses: pd.DataFrame, query: str, 
                       context: Dict) -> pd.DataFrame:
        """Use LLM to re-rank courses based on relevance"""
        
        # Prepare course descriptions for ranking
        course_summaries = []
        for _, row in courses.iterrows():
            summary = f"{row['course_number']}: {row['course_name']} - {row['description'][:100]}"
            course_summaries.append(summary)
        
        prompt = f"""
        Rank these courses from most to least relevant for this student query.
        
        Query: {query}
        Student Context: {context}
        
        Courses:
        {chr(10).join([f"{i+1}. {s}" for i, s in enumerate(course_summaries)])}
        
        Return just the numbers in order of relevance (e.g., "3,1,5,2,4"):
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Parse ranking and reorder dataframe
        ranking = [int(x)-1 for x in response.choices[0].message.content.split(',')]
        courses = courses.iloc[ranking]
        
        return courses