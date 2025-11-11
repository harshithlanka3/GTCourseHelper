"""
FastAPI backend for GTCourseHelper
Provides REST API endpoints for course search, recommendations, and chat
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Set default data path relative to project root
DEFAULT_DF_PATH = os.path.join(project_root, 'data', '202508_processed.pkl')

from search_courses import search_courses
from recommend_courses import build_prompt, call_gpt_recommendations, get_client, deduplicate_courses

SESSION_TTL_MINUTES = int(os.getenv("CHAT_SESSION_TTL_MINUTES", "30"))
SESSION_TTL = timedelta(minutes=SESSION_TTL_MINUTES)
MAX_CONTEXT_MESSAGES = 12
MAX_SESSION_MESSAGES = 120


@dataclass
class SessionState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    candidates: Optional[pd.DataFrame] = None
    last_enhanced_query: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


session_store: Dict[str, SessionState] = {}


def cleanup_sessions() -> None:
    """Remove expired chat sessions from in-memory store."""
    if not session_store:
        return

    now = datetime.utcnow()
    expired = [
        session_id
        for session_id, state in session_store.items()
        if now - state.updated_at > SESSION_TTL
    ]

    for session_id in expired:
        session_store.pop(session_id, None)


def create_session(session_id: Optional[str], seed_history: Optional[List["ChatMessage"]]) -> str:
    """Fetch or initialize a session, optionally seeding with provided history."""
    if session_id and session_id in session_store:
        return session_id

    new_session_id = session_id or f"session_{uuid4().hex}"
    seeded_messages: List[Dict[str, str]] = []

    if seed_history:
        for msg in seed_history:
            seeded_messages.append({"role": msg.role, "content": msg.content})

    session_store[new_session_id] = SessionState(messages=seeded_messages)
    return new_session_id


def build_enhanced_query(messages: List[Dict[str, str]]) -> str:
    """Construct an enhanced query string from recent conversation turns."""
    if not messages:
        return ""

    recent_messages = messages[-MAX_CONTEXT_MESSAGES:]
    if not recent_messages:
        return ""

    latest_message = recent_messages[-1]
    prior_messages = recent_messages[:-1]

    context_lines = [f"{msg['role']}: {msg['content']}" for msg in prior_messages]

    if context_lines:
        return "\n".join(context_lines) + f"\n\nLatest query: {latest_message['content']}"

    return latest_message['content']


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
    conversation_history: Optional[List[ChatMessage]] = None


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
    Conversational course recommendations with session-scoped history.
    """
    try:
        cleanup_sessions()

        session_id = create_session(request.session_id, request.conversation_history)
        session_state = session_store[session_id]

        # Append the latest user message to the in-memory session
        session_state.messages.append({"role": "user", "content": request.message})
        if len(session_state.messages) > MAX_SESSION_MESSAGES:
            session_state.messages = session_state.messages[-MAX_SESSION_MESSAGES:]

        enhanced_query = build_enhanced_query(session_state.messages)
        if not enhanced_query:
            enhanced_query = request.message

        if session_state.candidates is None:
            candidates = search_courses(
                user_query=enhanced_query,
                df_path=DEFAULT_DF_PATH,
                top_k=50,
                use_gpt=True,
                use_id_matching=True
            )

            candidates = deduplicate_courses(candidates)
            session_state.candidates = candidates
        else:
            candidates = session_state.candidates

        session_state.last_enhanced_query = enhanced_query

        prompt = build_prompt(enhanced_query, candidates)
        print(f"[chat] Prompt sent for session {session_id}:\n{prompt}\n{'-'*80}")
        
        client = get_client()
        
        recommendations = call_gpt_recommendations(client, prompt)

        # Persist assistant response and update timestamp
        session_state.messages.append({"role": "assistant", "content": recommendations})
        if len(session_state.messages) > MAX_SESSION_MESSAGES:
            session_state.messages = session_state.messages[-MAX_SESSION_MESSAGES:]
        session_state.updated_at = datetime.utcnow()

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

