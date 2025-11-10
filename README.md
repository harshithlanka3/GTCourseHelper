# GTCourseHelper

A tool for processing and analyzing Georgia Tech course data with embedding support for semantic search.

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

The preprocessing script uses **sentence-transformers** for local embedding generation - no API keys required! 

**Note:** The first run will download the model (~80MB), which may take a few minutes.

### Configure API Key for Semantic Search

The search functionality uses GPT-3.5-turbo to refine queries, which requires an API key:

**Option 1: Poe API (Recommended)**
```bash
export POE_API_KEY="your-poe-api-key-here"
```

**Option 2: OpenAI API**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

Get your Poe API key at [poe.com/api_key](https://poe.com/api_key) or OpenAI key at [platform.openai.com](https://platform.openai.com).

## Usage

Run the preprocessing script:
```bash
# With default input file (data/202508.json)
python preprocess_courses.py

# With custom input file
python preprocess_courses.py data/202408.json
```

This will:
- Parse course data from the specified JSON file (or `data/202508.json` by default)
- Generate 384-dimensional embeddings for course descriptions using the local `all-MiniLM-L6-v2` model
- Create a pandas DataFrame with all course information
- Save the processed data to a `.pkl` file with the same name as input (e.g., `data/202508_processed.pkl`)

**Note:** The output file is automatically named based on the input file (input: `202508.json` → output: `202508_processed.pkl`)

## Semantic Search

Search for courses using natural language queries:

```bash
# Interactive search
python search_courses.py

# Direct search from command line
python search_courses.py "I want to learn about machine learning and neural networks"

# Specify number of results (default is 10)
NUM_RESULTS=5 python search_courses.py "data structures and algorithms"
```

The search system uses a **two-stage approach**:

1. **Query Refinement**: GPT-3.5-turbo transforms your natural language query into an idealized course description that matches academic catalog language
2. **Semantic Matching**: The refined query is converted to an embedding vector and matched against all course descriptions using cosine similarity

This approach significantly improves search accuracy by translating student language into the formal academic language used in course catalogs.

## Course Recommendation Chatbot

The chatbot provides an interactive interface for course scheduling and information queries. It automatically classifies your questions and routes them to the appropriate handler.

### Interactive Mode

```bash
# Start interactive chatbot session
python chatbot.py

# With custom data file
python chatbot.py --df-path data/202408_processed.pkl
```

### Single Query Mode

```bash
# Process a single query
python chatbot.py --query "What is CS 3510?"
python chatbot.py --query "What courses are available on MWF mornings?"
python chatbot.py --query "I want to learn about machine learning"
```

### Chatbot Capabilities

The chatbot can handle several types of queries:

1. **Course ID Lookup**: Direct queries about specific courses
   - "What is CS 3510?"
   - "Tell me about MATH 2106"
   - "CS1332" (handles spacing variations with fuzzy matching)

2. **Time/Section Queries**: Questions about course availability during specific times
   - "What courses are available on Monday, Wednesday, Friday mornings?"
   - "Show me courses on TR afternoons"
   - "What sections does CS 3510 have?"

3. **Course Recommendations**: Requests for course suggestions based on interests
   - "I want to learn about neural networks"
   - "Recommend courses for data science"
   - "What courses cover machine learning?"

4. **General Questions**: Other scheduling and course-related questions
   - "Can I take CS 3510 and CS 1332 at the same time?"
   - "What are the prerequisites for CS 7641?"

### How It Works

The chatbot uses **intent classification** to automatically detect the type of question:

- **Regex matching** for course IDs (e.g., "CS 3510", "MATH2106")
- **Fuzzy matching** for typos and variations in course IDs
- **Keyword detection** for time/section queries (days, times, scheduling keywords)
- **Semantic search integration** for recommendations and general queries
- **GPT-powered responses** for complex questions with course context

## Features

- Course information extraction
- Meeting times parsing  
- Prerequisites parsing (AND/OR trees)
- Semantic embeddings for course descriptions (384-dimensional vectors)
- Intelligent semantic search with GPT-3.5-turbo query refinement
- Interactive chatbot for course scheduling and information queries
- Course ID lookup with regex and fuzzy matching
- Time/section-based course filtering
- Automatic pickle file generation for easy data persistence

## DataFrame Structure

The processed DataFrame contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `course_id` | str | Course identifier (e.g., "CS 3510") |
| `title` | str | Full course title |
| `description` | str | Course description |
| `meeting_times` | dict | Dictionary mapping section IDs to their meeting times (e.g., `{"A": ["MWF 0900-0950"]}`) |
| `prerequisites` | str | Human-readable prerequisite requirements (AND/OR logic) |
| `embedding` | list | 384-dimensional semantic embedding vector |