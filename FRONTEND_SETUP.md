# Frontend Setup Guide

## Quick Start

### 1. Backend Setup

First, set up the FastAPI backend:

```bash

cd api
pip install -r requirements.txt


cd ..
pip install -r requirements.txt


export POE_API_KEY="your-key-here"  


cd api
python main.py

```

The backend will run on `http://localhost:8000`

### 2. Frontend Setup

In a new terminal:

```bash

cd frontend


npm install


npm run dev
```

The frontend will run on `http://localhost:3000`

### 3. Test the Application

1. Open `http://localhost:3000` in your browser
2. Try queries like:
   - "I want to learn about machine learning"
   - "Show me CS 1332 and MATH 2106"
   - "What courses are good for data science?"

## API Endpoints

The backend provides these endpoints:

- `POST /api/chat` - Chat interface with conversation context
- `POST /api/search` - Direct course search
- `POST /api/recommend` - Get GPT recommendations
- `GET /api/health` - Health check

## Course ID Matching

The system now supports course ID matching! Try queries like:
- "I want CS 1332"
- "Show me CS 1332 and MATH 2106"
- "What about CS 1332? Also interested in machine learning"

The system will:
1. Extract course IDs from your query
2. Find exact matches
3. Perform fuzzy matching for typos
4. Combine with semantic search results
5. Boost scores for mentioned courses
