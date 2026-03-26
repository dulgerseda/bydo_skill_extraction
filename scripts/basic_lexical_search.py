import json
import re

# Step 1: Load the ESCO JSON file
with open("data/processed/esco_skill_dictionary.json", "r", encoding="utf-8") as f:
    esco_data = json.load(f)

# Print the number of records
print(f"Loaded {len(esco_data)} ESCO skill records.")
# Loaded 13939 ESCO skill records.


# Step 2 (updated): Prepare records with separate fields

records = []

for label, info in esco_data.items():
    alt_names = info.get("alternative_names", [])
    description = info.get("description", "")

    search_text = " ".join(
        [label] + alt_names + [description]
    ).lower()

    records.append({
        "label": label,
        "alternative_names": alt_names,
        "description": description,
        "search_text": search_text
    })

print(f"Prepared {len(records)} searchable records.")

# Quick check
# Step 2 check: inspect full record structure

for i in range(3):
    print(f"\nLabel: {records[i]['label']}")
    print(f"Alternative names: {records[i]['alternative_names'][:3]}")  # first 3
    print(f"Description: {records[i]['description'][:200]}")  # first 200 chars
    print(f"Search text: {records[i]['search_text'][:200]}")
    print("-" * 60)  

"""--- Sample search_text examples ---
Label: manage musical staff
Alternative names: ['manage music staff', 'coordinate duties of musical staff', 'direct musical staff']
Description: Assign and manage staff tasks in areas such as scoring, arranging, copying music and vocal coaching.
Search text: manage musical staff manage music staff coordinate duties of musical staff direct musical staff manage staff of music assign and manage staff tasks in areas such as scoring, arranging, copying music a
------------------------------------------------------------
Label: supervise correctional procedures
Alternative names: ['manage prison procedures', 'monitor correctional procedures', 'oversee prison procedures']
Description: Supervise the operations of a correctional facility or other correctional procedures, ensuring that they are compliant with legal regulations, and ensure that the staff complies with regulations, and 
Search text: supervise correctional procedures manage prison procedures monitor correctional procedures oversee prison procedures oversee correctional procedures manage correctional procedures monitor prison proce
------------------------------------------------------------"""    

# Step 3: Improved lexical search with weighted fields
# This function takes a query (extracted skill) and finds the most relevant ESCO skills

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9+/.-]+", text.lower())

def search_esco(query, top_k=5):
    query = query.lower().strip()
    query_words = tokenize(query)

    results = []

    for rec in records:
        label = rec["label"].lower()
        alt_names = [alt.lower() for alt in rec["alternative_names"]]
        description = rec["description"].lower()
        search_text = rec["search_text"]

        label_tokens = tokenize(label)
        alt_tokens = [tokenize(alt) for alt in alt_names]
        desc_tokens = tokenize(description)

        score = 0

        # exact/substring checks
        if query == label:
            score += 15
        elif query in label:
            score += 8

        for alt in alt_names:
            if query == alt:
                score += 12
            elif query in alt:
                score += 6

        # word-level checks
        for word in query_words:
            if word in label_tokens:
                score += 5
            elif any(word in alt_tok for alt_tok in alt_tokens):
                score += 3
            elif word in desc_tokens:
                score += 1

        # extra bonus: if whole query words all exist somewhere in search_text
        if all(word in search_text for word in query_words):
            score += 3

        if score > 0:
            results.append((score, rec["label"]))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]



# Step 4: Test the lexical search function
# We will test it with a few sample queries that are relevant to the skills we expect to find in ESCO.

queries = [
    "Python scientific stack",
    "signal processing",
    "CI/CD environment"
]

for q in queries:
    print(f"\nQuery: {q}")
    matches = search_esco(q, top_k=5)

    if not matches:
        print("  No match found.")
        continue

    for score, label in matches:
        print(f"  score={score:.1f} | label={label}")


# Step 5: Analyze results and refine
# Let's analyze the results for the "Python scientific stack" query and see if we can identify
query = "Python scientific stack"
query_words = tokenize(query)

python_related = []

for rec in records:
    label = rec["label"].lower()
    alt_names = [alt.lower() for alt in rec["alternative_names"]]
    description = rec["description"].lower()
    search_text = rec["search_text"]

    label_tokens = tokenize(label)
    alt_tokens = [tokenize(alt) for alt in alt_names]
    desc_tokens = tokenize(description)

    score = 0
    matched_words = []

    if query == label:
        score += 15
    elif query in label:
        score += 8

    for alt in alt_names:
        if query == alt:
            score += 12
        elif query in alt:
            score += 6

    for word in query_words:
        if word in label_tokens:
            score += 5
            matched_words.append((word, "label", 5))
        elif any(word in alt_tok for alt_tok in alt_tokens):
            score += 3
            matched_words.append((word, "alt", 3))
        elif word in desc_tokens:
            score += 1
            matched_words.append((word, "desc", 1))

    if all(word in search_text for word in query_words):
        score += 3

    # sadece python geçenleri incele
    if "python" in search_text:
        python_related.append({
            "label": rec["label"],
            "score": score,
            "matched_words": matched_words
        })

python_related = sorted(python_related, key=lambda x: x["score"], reverse=True)

for row in python_related[:30]:
    print(row["score"], "|", row["label"], "|", row["matched_words"])



# Step 6: pool results for all queries

queries = [
    "Python scientific stack",
    "signal processing",
    "CI/CD environment"
]

for q in queries:
    print(f"\nQuery: {q}")
    matches = search_esco(q, top_k=len(records))   # top 5 yerine tüm havuz

    if not matches:
        print("  No match found.")
        continue

    print(f"  Total matches: {len(matches)}")

    for score, label in matches:
        print(f"  score={score:.1f} | label={label}")    