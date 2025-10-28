# GTCourseHelper

A tool for processing and analyzing Georgia Tech course data with embedding support for semantic search.

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

The script uses **sentence-transformers** for local embedding generation - no API keys required! 

**Note:** The first run will download the model (~80MB), which may take a few minutes.

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

## Features

- Course information extraction
- Meeting times parsing
- Prerequisites parsing (AND/OR trees)
- Semantic embeddings for course descriptions (384-dimensional vectors)
- Automatic pickle file generation for easy data persistence