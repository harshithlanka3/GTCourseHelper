import pandas as pd, re

df = pd.read_pickle("/Users/harshithlanka/GaTech/cs6220/GTCourseHelper/data/202508_processed.pkl")

cand = df[df["course_id"] == "MATH 2106"]
print(cand[["course_id","title"]].to_string(index=False))

for _, row in cand.iterrows():
    print(f"\n{row['course_id']}: {row['title']}")
    mt = row["meeting_times"] or {}
    if not mt:
        print("  Sections: None listed")
    else:
        print("  Sections:")
        for sec, times in mt.items():
            print(f"    {sec}: {', '.join(times)}")
    print(f"  Prerequisites: {row['prerequisites'] or 'None explicitly stated'}")