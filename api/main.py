"""
FastAPI backend for GTCourseHelper
Provides REST API endpoints for course search, recommendations, and chat
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from conversation_store import conversation_store
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
from classify_query import classify_query
from course_details import get_course_details, format_course_details
from course_id_matcher import extract_course_ids

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


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None


class SessionRenameRequest(BaseModel):
    title: str


class SessionInfo(BaseModel):
    session_id: str
    title: str
    updated_at: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


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
    try:
        # Assign or reuse session
        session_id = request.session_id or f"session_{hash(request.message) % 1000000}"

        # Add user message to conversation memory
        conversation_store.add_message(session_id, "user", request.message)

        # Classify the query to determine behavior
        query_type = classify_query(request.message)
        print(f"DEBUG: Classified query as: {query_type}")

        # Route to appropriate behavior based on query type
        if query_type == "conversation":
            # Simple conversational response - no search, no recommendations
            response_text = "I'm here and ready to help! What courses are you thinking about?"
            
        elif query_type == "single_course":
            # Get details for a single course
            course_ids = extract_course_ids(request.message)
            if course_ids:
                course_id = course_ids[0]
                course_details = get_course_details(course_id, DEFAULT_DF_PATH)
                response_text = format_course_details(course_details)
            else:
                # Fallback if course ID extraction failed
                response_text = "I couldn't identify a specific course. Please provide a course ID like 'CS 1332' or 'MATH 1554'."
                
        elif query_type == "compare":
            # Compare multiple courses
            course_ids = extract_course_ids(request.message)
            if len(course_ids) >= 2:
                # Get details for each course
                course_details_list = []
                for cid in course_ids[:5]:  # Limit to 5 courses max
                    details = get_course_details(cid, DEFAULT_DF_PATH)
                    if details.get('found'):
                        course_details_list.append(details)
                
                if course_details_list:
                    # Format comparison
                    parts = [f"**Comparison of {len(course_details_list)} courses:**\n"]
                    for details in course_details_list:
                        parts.append(f"\n**{details['course_id']}: {details['title']}**")
                        if details.get('description'):
                            desc = details['description'][:200] + "..." if len(details.get('description', '')) > 200 else details.get('description', '')
                            parts.append(f"Description: {desc}")
                        if details.get('prerequisites'):
                            parts.append(f"Prerequisites: {details['prerequisites']}")
                        parts.append("")
                    response_text = "\n".join(parts)
                else:
                    response_text = f"Sorry, I couldn't find details for the courses you mentioned: {', '.join(course_ids)}"
            else:
                response_text = "Please provide at least two course IDs to compare (e.g., 'CS 1332 vs CS 2110')."
                
        else:  # query_type == "general_recommendation"
            # Use existing recommendation pipeline
            # Build context string from last 3 interactions
            history = conversation_store.get_history(session_id)
            context_parts = [f"{m['role']}: {m['content']}" for m in history[-3:]]
            enhanced_query = "\n".join(context_parts) + f"\n\nUser: {request.message}"

            # Get course recommendations
            candidates = search_courses(
                user_query=enhanced_query,
                df_path=DEFAULT_DF_PATH,
                top_k=50,
                use_gpt=True,
                use_id_matching=True
            )

            candidates = deduplicate_courses(candidates)
            prompt = build_prompt(enhanced_query, candidates)
            client = get_client()
            response_text = call_gpt_recommendations(client, prompt)

        # Store assistant response
        conversation_store.add_message(session_id, "assistant", response_text)
        
        # Debug: Verify messages are stored
        stored_history = conversation_store.get_history(session_id)
        print(f"DEBUG: Session {session_id} now has {len(stored_history)} messages")
        print(f"DEBUG: Message roles: {[m['role'] for m in stored_history]}")

        return ChatResponse(
            response=response_text,
            session_id=session_id,
        )

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(f"Error in chat endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


# Session Management Endpoints
@app.post("/api/session")
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    try:
        session_id = conversation_store.create(title=request.title)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all chat sessions"""
    try:
        sessions = conversation_store.list()
        return [SessionInfo(**s) for s in sessions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
async def get_session(session_id: str):
    """Get conversation history for a session"""
    try:
        if not conversation_store.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = conversation_store.get_history(session_id)
        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in history]
        print(f"DEBUG: Returning {len(messages)} messages for session {session_id}")
        print(f"DEBUG: Message roles in response: {[m.role for m in messages]}")
        return SessionHistoryResponse(session_id=session_id, messages=messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/session/{session_id}")
async def rename_session(session_id: str, request: SessionRenameRequest):
    """Rename a session"""
    try:
        if not conversation_store.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        conversation_store.rename(session_id, request.title)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        if not conversation_store.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        conversation_store.clear(session_id)
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

