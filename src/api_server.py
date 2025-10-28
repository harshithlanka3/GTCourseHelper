from flask import Flask, request, jsonify
import asyncio

class GTCourseAdvisorAPI:
    def __init__(self, config: Dict):
        # Initialize all components
        self.processor = CourseDataProcessor(config['data_file'])
        self.embedding_engine = CourseEmbeddingEngine()
        self.embedding_engine.generate_embeddings(self.processor.courses_df)
        
        self.query_enhancer = QueryEnhancer(config['openai_api_key'])
        self.retriever = HybridRetriever(
            self.processor.courses_df,
            self.embedding_engine,
            self.query_enhancer
        )
        
        self.schedule_gen = ScheduleGenerator(self.processor.courses_df)
        self.advisor = CourseAdvisorRAG(self.retriever, self.schedule_gen)
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            data = request.json
            user_query = data.get('query')
            user_context = data.get('context', {})
            
            try:
                response = self.advisor.process_query(user_query, user_context)
                return jsonify({
                    'status': 'success',
                    'response': response
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/embed_course', methods=['POST'])
        def embed_course():
            """Endpoint to add new courses to the system"""
            course_data = request.json
            # Add to dataframe and regenerate embeddings
            # Implementation here
            pass
    
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)

# Main execution
if __name__ == '__main__':
    config = {
        'data_file': 'gt_courses.json',
        'openai_api_key': 'your-api-key',
        'embedding_model': 'all-MiniLM-L6-v2'
    }
    
    api = GTCourseAdvisorAPI(config)
    api.run()