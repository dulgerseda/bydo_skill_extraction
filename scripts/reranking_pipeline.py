"""
=============================================================
RERANKING PIPELINE — BYDO Project
=============================================================
Reads answers.json (with domain wrapper + skill_type + domain)
Runs semantic search + cross-encoder reranking per skill
Adds: esco_match, match_type, confidence, decision, gap
Saves to results/output.json
=============================================================
"""


import json
import math
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR        = Path("data/processed")
EMBEDDINGS_PATH = DATA_DIR / "esco_embeddings.npy"
LABELS_PATH     = DATA_DIR / "esco_labels.json"
ESCO_JSON       = DATA_DIR / "esco_skill_dictionary.json"
ANSWERS_PATH = Path("answers/normalized_answers.json")
OUTPUT_PATH     = Path("results/output.json")

BI_ENCODER_MODEL    = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(ESCO_JSON, "r", encoding="utf-8") as f:
        esco_data = json.load(f)
    return embeddings, labels, esco_data

def semantic_search(term, embeddings, labels, bi_encoder, top_k=10):
    query_vec = bi_encoder.encode([term])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), labels[i]) for i in top_indices]

def rerank(term, esco_form, candidates, esco_data, cross_encoder):
    query = f"{term}. {esco_form}"
    pairs = []
    for _, label in candidates:
        info = esco_data.get(label, {})
        description = info.get("description", "")
        pairs.append([query, f"{label}. {description}"])
    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        zip(scores, [c[1] for c in candidates]),
        key=lambda x: x[0], reverse=True
    )
    return reranked

def normalize_score(score):
    """Cross-encoder raw skor normalised 0-1 with sigmoid"""
    return round(1 / (1 + math.exp(-score * 0.5)), 4)

def make_match_type(confidence):
    """Hocanın Neo4j şemasına uygun match tipi."""
    if confidence >= 0.85:
        return "exactMatch"
    elif confidence >= 0.65:
        return "closeMatch"
    else:
        return "emerging"

def make_decision(confidence):
    """Hocanın threshold mantığına göre karar ver."""
    if confidence >= 0.8:
        return "accept"
    elif confidence >= 0.5:
        return "review"
    else:
        return "emerging"

if __name__ == "__main__":

    print("Loading index and models...")
    embeddings, labels, esco_data = load_index()
    bi_encoder    = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"ESCO index loaded: {embeddings.shape[0]} records\n")

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers = json.load(f)

    output = {}

    for job_id, job_data in answers.items():
        if isinstance(job_data, list):
            skills     = job_data
            job_domain = "Unknown"
        else:
            skills     = job_data["skills"]
            job_domain = job_data.get("domain", "Unknown")

        print(f"\n{'='*60}")
        print(f"  {job_id}  |  domain: {job_domain}  |  {len(skills)} skills")
        print(f"{'='*60}")

        enriched_skills = []

        for skill in skills:
            noun      = skill["noun"]
            esco_form = skill.get("esco_form", noun)

            print(f"  -> {noun}")

            candidates = semantic_search(noun, embeddings, labels, bi_encoder, top_k=10)
            reranked   = rerank(noun, esco_form, candidates, esco_data, cross_encoder)

            top_score, top_label = reranked[0]
            confidence = normalize_score(top_score)
            match_type = make_match_type(confidence)
            decision   = make_decision(confidence)
            gap        = decision == "emerging"

            enriched = {
                **skill,
                "esco_match": top_label if not gap else None,
                "match_type": match_type,
                "confidence": confidence,
                "decision":   decision,
                "gap":        gap
            }
            enriched_skills.append(enriched)

            print(f"     confidence: {confidence}  match_type: {match_type}  decision: {decision}  gap: {gap}")

        output[job_id] = {
            "domain": job_domain,
            "skills": enriched_skills
        }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nDone! -> {OUTPUT_PATH}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for job_id, job_data in output.items():
        skills   = job_data["skills"]
        accept   = sum(1 for s in skills if s["decision"] == "accept")
        review   = sum(1 for s in skills if s["decision"] == "review")
        emerging = sum(1 for s in skills if s["decision"] == "emerging")
        exact    = sum(1 for s in skills if s["match_type"] == "exactMatch")
        close    = sum(1 for s in skills if s["match_type"] == "closeMatch")
        print(f"  {job_id[:35]:<35} accept:{accept}  review:{review}  emerging:{emerging}  |  exact:{exact}  close:{close}")

# streamlit run app.py         