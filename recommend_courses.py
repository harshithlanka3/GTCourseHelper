import os
import sys
import argparse
import dotenv
import pandas as pd
from openai import OpenAI

from search_courses import search_courses


dotenv.load_dotenv()


def get_client() -> OpenAI:
    poe_api_key = os.getenv("POE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if poe_api_key:
        return OpenAI(api_key=poe_api_key, base_url="https://api.poe.com/v1")
    if openai_api_key:
        return OpenAI(api_key=openai_api_key)
    raise ValueError("No API key configured. Please set POE_API_KEY or OPENAI_API_KEY.")


def deduplicate_courses(df: pd.DataFrame) -> pd.DataFrame:
    # Remove exact duplicates by title+description (keeps first occurrence)
    key = df["title"].fillna("").str.strip() + "\n" + df["description"].fillna("").str.strip()
    deduped = df.loc[~key.duplicated()].copy()
    return deduped


def _format_sections(meeting_times: dict) -> str:
    if not meeting_times:
        return "None listed"
    parts = []
    for section, times in meeting_times.items():
        times_str = ", ".join(times) if times else "TBA"
        parts.append(f"Section {section}: {times_str}")
    return "; ".join(parts)


def save_course_options_to_file(df: pd.DataFrame, output_file: str) -> None:
    """Save formatted course options to a file for inspection."""
    course_lines = []
    for _, row in df.iterrows():
        sections_str = _format_sections(row.get("meeting_times"))
        prereqs_str = row.get("prerequisites") or ""
        course_lines.append(
            f"{row['course_id']}: {row['title']}\n"
            f"Description: {row['description']}\n"
            f"Prerequisites: {prereqs_str}\n"
            f"Sections: {sections_str}"
        )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total courses: {len(df)}\n\n")
        f.write("\n\n".join(course_lines))
    
    print(f"Course options saved to: {output_file}")


def build_prompt(query: str, df: pd.DataFrame) -> str:
    course_lines = []
    valid_course_ids = []
    for _, row in df.iterrows():
        course_id = row['course_id']
        valid_course_ids.append(course_id)
        sections_str = _format_sections(row.get("meeting_times"))
        prereqs_str = row.get("prerequisites") or ""
        course_lines.append(
            f"{course_id}: {row['title']}\n"
            f"Description: {row['description']}\n"
            f"Prerequisites: {prereqs_str}\n"
            f"Sections: {sections_str}"
        )
    course_string = "\n\n".join(course_lines)
    valid_ids_list = ", ".join(valid_course_ids)

    system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})

VALID COURSE IDs (YOU CAN ONLY RECOMMEND FROM THIS LIST):
{valid_ids_list}

Course Options: 
{course_string}

CRITICAL: YOU CAN ONLY RECOMMEND COURSES FROM THE "VALID COURSE IDs" LIST ABOVE
- The course_id MUST appear in the "VALID COURSE IDs" list at the top of this prompt
- If a course is mentioned in a prerequisite field but does NOT appear in the VALID COURSE IDs list, you CANNOT recommend it
- You must verify each recommended course_id exists in the VALID COURSE IDs list before recommending it

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- ALL recommended courses MUST have their course_id listed in the VALID COURSE IDs list above
- Respect explicit scheduling constraints in the student profile (days/times). When schedule is specified, prioritize courses whose listed Sections match the requested window.
- Respect prerequisite preferences stated in the student profile:
  - If the student requests minimal prerequisites or entry-friendly options, prefer courses with no/low prerequisites.
  - If the student lacks a specific prerequisite (e.g., says they have not taken X), avoid recommending courses that explicitly require that prerequisite.
  - If the student requests specific prerequisites (e.g., needs a course that requires calculus), favor courses whose prerequisites reflect that requirement.
- Recommend ONLY courses whose course_id appears in the Course Options list above - NO EXCEPTIONS
- For each recommendation include:
  1. Course number
  2. Course name
  3. Two-sentence explanation focused on the student's specific profile/goals, explicitly referencing schedule/prerequisite fit when relevant
  4. Confidence level (High/Medium/Low)
  5. Available sections (from provided data)
  6. Prerequisites (as provided)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit; you may mention prerequisites, schedule, or other courses in your explanation, but ONLY recommend courses from the VALID COURSE IDs list]
Sections: [COPY EXACTLY as shown in Course Options above - e.g., "Section A: MW 0930 - 1045; Section B: MW 1400 - 1515" or "None listed" if that's what was provided]
Prerequisites: [COPY EXACTLY as shown in Course Options above, or "None explicitly stated" if empty]
Confidence: [Level]

2. [Next course...]

CRITICAL OUTPUT RULES:
- For Sections and Prerequisites, you MUST copy the exact text from the "Course Options" section above. Do NOT say "Not listed in provided data" or similar - use the actual text provided.
- If the Course Options show sections, copy them exactly. If they say "None listed", use "None listed".
- If the Course Options show prerequisites, copy them exactly. If the field is empty, say "None explicitly stated".

CONSTRAINTS:
- YOU CAN ONLY RECOMMEND courses whose course_id appears in the VALID COURSE IDs list at the top of this prompt
- You CAN mention prerequisites or other courses in your Rationale (e.g., "This course requires CS 1332 as a prerequisite"), but you CANNOT recommend those courses unless they are also in the VALID COURSE IDs list
- If you see a course_id mentioned in a prerequisite field (e.g., "CS 1332" in prerequisites), you CAN mention it in your explanation but you CANNOT recommend it as one of your 10 recommendations unless "CS 1332" also appears in the VALID COURSE IDs list
- NO mention of being an AI or advisor
- **If multiple courses have identical titles and descriptions (cross-listed), recommend only ONE of them**

VALIDATION: Before outputting each recommendation, verify the course_id appears in the VALID COURSE IDs list at the top of this prompt. If it doesn't, do not recommend it."""
    return system_rec_message


def call_gpt_recommendations(client: OpenAI, prompt: str) -> str:
    # Prefer gpt-4o, fallback to gpt-3.5-turbo if needed
    for model in ("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            continue
    raise RuntimeError("All model attempts failed (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend top 10 GT courses using GPT on top of semantic search.")
    parser.add_argument("query", type=str, help="Natural language query / student profile")
    parser.add_argument("--df-path", type=str, default="data/202508_processed.pkl", help="Path to processed courses pickle")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates to retrieve before GPT ranking (default: 50)")
    parser.add_argument("--no-gpt-query", action="store_true", help="Disable GPT query enhancement inside semantic search")
    parser.add_argument("--save-options", type=str, default=None, help="Save formatted course options to this file for inspection (optional)")
    args = parser.parse_args()

    print(f"Searching courses (k={args.top_k})...")
    candidates = search_courses(
        user_query=args.query,
        df_path=args.df_path,
        top_k=args.top_k,
        use_gpt=not args.no_gpt_query,
    )

    candidates = deduplicate_courses(candidates)

    # Save course options to file if requested
    if args.save_options:
        save_course_options_to_file(candidates, args.save_options)

    prompt = build_prompt(args.query, candidates)
    client = get_client()
    print("\nGenerating recommendations with GPT...\n")
    recommendations_md = call_gpt_recommendations(client, prompt)

    print(recommendations_md)


if __name__ == "__main__":
    main()


