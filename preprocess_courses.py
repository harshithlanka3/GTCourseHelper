import json
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Parse command line arguments
if len(sys.argv) > 1:
    TERM_FILE = sys.argv[1]
else:
    TERM_FILE = "data/202508.json"

# Derive output filename from input filename
input_basename = os.path.splitext(os.path.basename(TERM_FILE))[0]
input_dir = os.path.dirname(TERM_FILE) if os.path.dirname(TERM_FILE) else ""
OUTPUT_FILE = os.path.join(input_dir, f"{input_basename}_processed.pkl") if input_dir else f"{input_basename}_processed.pkl"

# Initialize the local embedding model
print(f"Processing: {TERM_FILE}")
print(f"Output will be saved to: {OUTPUT_FILE}")
print("\nLoading embedding model (this may take a moment on first run)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

with open(TERM_FILE, "r") as f:
    raw = json.load(f)

courses = raw["courses"]
caches  = raw["caches"] # contains the period look-up table


def get_embedding(text):
    """Generate embedding using sentence-transformers local model."""
    if not text:
        return None
    try:
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def prereq_tree_to_text(expr):
    """Convert the AND/OR prefix tree into a readable infix string."""
    if not expr:
        return ""
    if isinstance(expr, dict):                        # leaf
        return f"{expr.get('id','')} ≥ {expr.get('grade','')}".strip()
    op, *rest = expr                                  # ["and", expr1, expr2, ...]
    joined = f" {op.upper()} ".join(prereq_tree_to_text(r) for r in rest)
    return f"({joined})"


def meeting_tuple_to_text(m):
    """
    Convert one Meeting tuple into "MWF 0900-0950"
    (index 0 = period idx → "0900 - 0950", index 1 = days string).
    """
    period = caches["periods"][m[0]]
    days   = m[1] or "TBA"
    return f"{days} {period}"


rows = []

for course_id, (title, sections, prereqs, description) in courses.items():
    section_ids = list(sections.keys())

    # map section → ["MWF 0900-0950", …]
    section_meetings = {
        sid: [meeting_tuple_to_text(m) for m in sec_data[1]]
        for sid, sec_data in sections.items()
    }

    rows.append(
        {
            "course_id": course_id,
            "title": title,
            "description": description or "",
            "meeting_times": section_meetings,
            "prerequisites": prereq_tree_to_text(prereqs),
        }
    )

df = pd.DataFrame(rows)

# Generate embeddings for course descriptions
print("Generating embeddings for course descriptions...")
embeddings = []
for desc in tqdm(df['description'], desc="Embedding descriptions"):
    embeddings.append(get_embedding(desc))

df['embedding'] = embeddings

print("Embeddings generated successfully!")
print(f"DataFrame shape: {df.shape}")
if len(df) > 0 and df['embedding'].iloc[0] is not None:
    print(f"Embedding dimension: {len(df['embedding'].iloc[0])}")

# Save the dataframe to a pickle file
print(f"\nSaving processed courses to {OUTPUT_FILE}...")
df.to_pickle(OUTPUT_FILE)
print("Done! Processed courses saved successfully.")

