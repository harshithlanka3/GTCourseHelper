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

## Features

- Course information extraction
- Meeting times parsing  
- Prerequisites parsing (AND/OR trees)
- Semantic embeddings for course descriptions (384-dimensional vectors)
- Department/school classification derived from course prefixes
- Graduate-level flag (courses above the 4000-level are marked as graduate)
- Intelligent semantic search with GPT-3.5-turbo query refinement
- Automatic pickle file generation for easy data persistence

## Analytics & Visualization

Explore the embedding space with interactive 3D visualizations:

### Install Visualization Dependencies

```bash
pip install -r requirements-viz.txt
```

### 3D Embedding Visualization

Create an interactive 3D scatter plot of course embeddings:

```bash
# Basic usage (default: colored by department)
python visualizations/plot_embeddings_3d.py

# Color by graduate level
python visualizations/plot_embeddings_3d.py --color-by graduate

# Use UMAP instead of PCA (requires umap-learn)
python visualizations/plot_embeddings_3d.py --method umap

# Limit to 500 courses for faster processing
python visualizations/plot_embeddings_3d.py --limit 500

# Custom data path and output
python visualizations/plot_embeddings_3d.py --data-path data/202408_processed.pkl --output visualizations/my_viz.html

# Save as static image (PNG) in addition to HTML
python visualizations/plot_embeddings_3d.py --img

# Save only as image, skip HTML
python visualizations/plot_embeddings_3d.py --img-only

# Save as PDF instead of PNG
python visualizations/plot_embeddings_3d.py --img-only --img-format pdf
```

The script will:
- Load processed course data from the pickle file
- Reduce embeddings to 3D using PCA, UMAP (default), or t-SNE
- Create an interactive Plotly visualization with hover tooltips
- Save an HTML file that you can open in any browser (unless `--img-only` is used)
- Optionally save a static image (PNG, PDF, SVG, or JPG)

**Features:**
- Hover over points to see course ID, title, department, and description
- Rotate, zoom, and pan the 3D view (in HTML)
- Color courses by department or graduate level
- Export-ready HTML file (no server required)
- Static image export for presentations/reports (PNG, PDF, SVG, JPG)

The visualization helps you:
- Identify clusters of similar courses
- Validate that embeddings separate subjects as expected
- Explore the semantic structure of the course catalog
- Discover relationships between courses across departments

## DataFrame Structure

The processed DataFrame contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `course_id` | str | Course identifier (e.g., "CS 3510") |
| `title` | str | Full course title |
| `description` | str | Course description |
| `meeting_times` | dict | Dictionary mapping section IDs to their meeting times (e.g., `{"A": ["MWF 0900-0950"]}`) |
| `prerequisites` | str | Human-readable prerequisite requirements (AND/OR logic) |
| `department` | str | Department or school associated with the course prefix (e.g., "Computer Science") |
| `is_graduate_level` | bool | `True` if the course number is above 4000, otherwise `False` |
| `embedding` | list | 384-dimensional semantic embedding vector |