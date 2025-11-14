import json
import os
import sys
import pandas as pd
from tqdm import tqdm
from embedding_utils import get_model

# Parse command line arguments
if len(sys.argv) > 1:
    TERM_FILE = sys.argv[1]
else:
    TERM_FILE = "data/202508.json"

# Derive output filename from input filename
input_basename = os.path.splitext(os.path.basename(TERM_FILE))[0]
input_dir = os.path.dirname(TERM_FILE) if os.path.dirname(TERM_FILE) else ""
OUTPUT_FILE = os.path.join(input_dir, f"{input_basename}_processed.pkl") if input_dir else f"{input_basename}_processed.pkl"

# Initialize the local embedding model (using cached model from embedding_utils)
print(f"Processing: {TERM_FILE}")
print(f"Output will be saved to: {OUTPUT_FILE}")
print("\nLoading embedding model (this may take a moment on first run)...")
model = get_model()
print(f"Model loaded: {model.get_sentence_embedding_dimension()} dimensions")

with open(TERM_FILE, "r") as f:
    raw = json.load(f)

courses = raw["courses"]
caches  = raw["caches"] # contains the period look-up table

DEPARTMENT_MAP = {
    "ACCT": "Accounting",
    "AE": "Aerospace Engineering",
    "AS": "Air Force Aerospace Studies",
    "APPH": "Applied Physiology",
    "ASE": "Applied Systems Engineering",
    "ARBC": "Arabic",
    "ARCH": "Architecture",
    "BIOS": "Biological Sciences",
    "BIOL": "Biology",
    "BMEJ": "Biomed Engr/Joint Emory PKU",
    "BMED": "Biomedical Engineering",
    "BMEM": "Biomedical Engr/Joint Emory",
    "BC": "Building Construction",
    "BCP": "Building Construction - Professional",
    "CETL": "Center Enhancement-Teach/Learn",
    "CHBE": "Chemical & Biomolecular Engr",
    "CHEM": "Chemistry",
    "CHIN": "Chinese",
    "CP": "City Planning",
    "CEE": "Civil and Environmental Engr",
    "COA": "College of Architecture",
    "COE": "College of Engineering",
    "COS": "College of Sciences",
    "CX": "Computational Mod, Sim, & Data",
    "CSE": "Computational Science & Engr",
    "CS": "Computer Science",
    "COOP": "Cooperative Work Assignment",
    "UCGA": "Cross Enrollment",
    "EAS": "Earth and Atmospheric Sciences",
    "ECON": "Economics",
    "ECEP": "Elect & Comp Engr-Professional",
    "ECE": "Electrical & Computer Engr",
    "ENGL": "English",
    "FS": "Foreign Studies",
    "FREE": "Free Elective",
    "FREN": "French",
    "GT": "Georgia Tech",
    "GTL": "Georgia Tech Lorraine",
    "GRMN": "German",
    "HS": "Health Systems",
    "HEBW": "Hebrew",
    "HIN": "Hindi",
    "HIST": "History",
    "HTS": "History, Technology & Society",
    "HUM": "Humanities Elective",
    "ID": "Industrial Design",
    "ISYE": "Industrial & Systems Engr",
    "INTA": "International Affairs",
    "IL": "International Logistics",
    "INTN": "Internship",
    "IMBA": "Intl Executive MBA",
    "IAC": "Ivan Allen College",
    "JAPN": "Japanese",
    "KOR": "Korean",
    "LATN": "Latin",
    "LS": "Learning Support",
    "LING": "Linguistics",
    "LMC": "Literature, Media & Comm",
    "MGT": "Management",
    "MOT": "Management of Technology",
    "MLDR": "Manufacturing Leadership",
    "MSE": "Materials Science & Engr",
    "MATH": "Mathematics",
    "ME": "Mechanical Engineering",
    "MP": "Medical Physics",
    "MSL": "Military Science & Leadership",
    "ML": "Modern Languages",
    "MUSI": "Music",
    "NS": "Naval Science",
    "NEUR": "Neuroscience",
    "NRE": "Nuclear & Radiological Engr",
    "PERS": "Persian",
    "PHIL": "Philosophy",
    "PHYS": "Physics",
    "POL": "Political Science",
    "PTFE": "Polymer, Textile and Fiber Eng",
    "PORT": "Portuguese",
    "DOPP": "Professional Practive",
    "PSYC": "Psychology",
    "PUBJ": "Public Policy/Joint GSU PhD",
    "PUBP": "Public Policy",
    "RUSS": "Russian",
    "SCI": "Science",
    "SLS": "Serve Learn Sustain",
    "SS": "Social Science Elective",
    "SOC": "Sociology",
    "SPAN": "Spanish",
    "SWAH": "Swahili",
    "VIP": "Vertically Integrated Project",
    "WOLO": "Wolof",
}


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
    prefix = course_id.split()[0] if course_id else ""
    department = DEPARTMENT_MAP.get(prefix, prefix)

    course_number = None
    if course_id:
        parts = course_id.split()
        if len(parts) > 1 and parts[1].isdigit():
            course_number = int(parts[1])

    rows.append(
        {
            "course_id": course_id,
            "title": title,
            "description": description or "",
            "meeting_times": section_meetings,
            "prerequisites": prereq_tree_to_text(prereqs),
            "department": department,
            "is_graduate_level": bool(course_number and course_number > 4000),
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

