"""
FastAPI backend for GTCourseHelper
Provides REST API endpoints for course search, recommendations, and chat
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set default data path relative to project root
DEFAULT_DF_PATH = os.path.join(project_root, 'data', '202508_processed.pkl')

from search_courses import search_courses
from recommend_courses import build_prompt, call_gpt_recommendations, get_client, deduplicate_courses

app = FastAPI(title="GT Course Helper API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_gpt: Optional[bool] = True
    use_id_matching: Optional[bool] = True


class CourseResult(BaseModel):
    course_id: str
    title: str
    description: str
    prerequisites: str
    meeting_times: Dict
    similarity_score: float


class SearchResponse(BaseModel):
    courses: List[CourseResult]
    total: int


class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 50
    use_gpt_query: Optional[bool] = True


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str
    session_id: str
    recommendations: Optional[str] = None


# API Endpoints
@app.get("/")
async def root():
    return {"message": "GT Course Helper API", "version": "1.0.0"}


@app.post("/api/search", response_model=SearchResponse)
async def search_courses_endpoint(request: SearchRequest):
    """
    Search for courses using semantic search and course ID matching
    """
    try:
        results = search_courses(
            user_query=request.query,
            df_path=DEFAULT_DF_PATH,
            top_k=request.top_k,
            use_gpt=request.use_gpt,
            use_id_matching=request.use_id_matching
        )
        
        # Convert to response format
        courses = []
        for _, row in results.iterrows():
            courses.append(CourseResult(
                course_id=row['course_id'],
                title=row['title'],
                description=row['description'] or "",
                prerequisites=row['prerequisites'] or "",
                meeting_times=row['meeting_times'] or {},
                similarity_score=float(row['similarity_score'])
            ))
        
        return SearchResponse(courses=courses, total=len(courses))
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error in chat endpoint: {error_detail}")  # Print to server logs
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
async def recommend_courses_endpoint(request: RecommendRequest):
    """
    Get GPT-powered course recommendations
    """
    try:
        # Get semantic search candidates
        candidates = search_courses(
            user_query=request.query,
            df_path=DEFAULT_DF_PATH,
            top_k=request.top_k,
            use_gpt=request.use_gpt_query,
            use_id_matching=True
        )
        
        candidates = deduplicate_courses(candidates)
        
        # Build prompt and get GPT recommendations
        prompt = build_prompt(request.query, candidates)
        client = get_client()
        recommendations = call_gpt_recommendations(client, prompt)
        
        return {"recommendations": recommendations}
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error in chat endpoint: {error_detail}")  # Print to server logs
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat interface with conversation context
    For now, this is a simple wrapper around recommendations
    TODO: Add proper conversation management
    """
    try:
        print(f"Received chat request: {request.message[:100]}...")  # Debug log
        
        # Build context from conversation history
        context_parts = []
        if request.conversation_history:
            # Include last 3 messages for context
            recent_messages = request.conversation_history[-3:]
            for msg in recent_messages:
                context_parts.append(f"{msg.role}: {msg.content}")
        
        # Combine context with current message
        if context_parts:
            enhanced_query = "\n".join(context_parts) + f"\n\nLatest query: {request.message}"
        else:
            enhanced_query = request.message
        
        print(f"Enhanced query: {enhanced_query[:200]}...")  # Debug log
        
        # Get recommendations
        print("Calling search_courses...")
        candidates = search_courses(
            user_query=enhanced_query,
            df_path=DEFAULT_DF_PATH,
            top_k=50,
            use_gpt=True,
            use_id_matching=True
        )
        
        print(f"Found {len(candidates)} candidates")
        
        candidates = deduplicate_courses(candidates)
        print(f"After deduplication: {len(candidates)} candidates")
        
        # Build prompt with context
        print("Building prompt...")
        prompt = build_prompt(enhanced_query, candidates)
        
        print("Getting GPT client...")
        client = get_client()
        
        print("Calling GPT for recommendations...")
        recommendations = call_gpt_recommendations(client, prompt)
        print("Got recommendations!")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{hash(enhanced_query) % 1000000}"
        
        return ChatResponse(
            response=recommendations,
            session_id=session_id,
            recommendations=recommendations
        )
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error in chat endpoint: {error_detail}")  # Print to server logs
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

