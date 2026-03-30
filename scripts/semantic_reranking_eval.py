"""
=============================================================
SEMANTIC SEARCH + CROSS-ENCODER RERANKING — BYDO Project (Week 5)
=============================================================

PIPELINE OVERVIEW:
------------------
For each extracted skill term (+ its source sentence), the system
runs a two-stage ESCO matching process inspired by the LELA
architecture (Haffoudhi et al., 2026):

  Stage 1 — Candidate Retrieval (Semantic Search)
      The extracted term is encoded into a vector using a
      bi-encoder (all-MiniLM-L6-v2). Cosine similarity is
      computed against all 13,939 pre-encoded ESCO vectors.
      The top-10 most similar ESCO candidates are returned.

  Stage 2 — Reranking (Cross-Encoder)
      Each of the top-10 candidates is re-scored by a
      cross-encoder (ms-marco-MiniLM-L-6-v2). Unlike the
      bi-encoder, the cross-encoder reads the skill term,
      its source sentence, and the ESCO candidate together
      in a single forward pass — capturing fine-grained
      contextual interactions. The candidates are re-ranked
      by this new score.

  Output
      For each term: semantic top-5 vs reranked top-5,
      shown side by side. A summary table reports match
      quality (strong / partial / weak) per job.

NOTE ON MODEL CHOICE:
      The original LELA paper uses Qwen3-Reranker-4B, which
      requires GPU resources. This study uses the lightweight
      cross-encoder/ms-marco-MiniLM-L-6-v2 as an equivalent
      alternative suitable for CPU-based local development.
      LELA's own ablation results confirm that performance is
      robust across different reranker choices (Section 4.6).
=============================================================
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# --- Paths ---
DATA_DIR = Path("data/processed")
EMBEDDINGS_PATH = DATA_DIR / "esco_embeddings.npy"
LABELS_PATH = DATA_DIR / "esco_labels.json"
ESCO_JSON = DATA_DIR / "esco_skill_dictionary.json"

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# =============================================================
# JOB DATA — 3 job descriptions with terms + source sentences
# =============================================================

JOB_1 = {
    "title": "Job 1 — Structural Health Monitoring / Data Science",
    "terms": [
        ("Structural Life Estimation",        "Build predictive models to estimate the remaining useful life of structures and detect early-stage structural degradation."),
        ("Structural Degradation Detection",  "Build predictive models to estimate the remaining useful life of structures and detect early-stage structural degradation."),
        ("Time-Series Health Monitoring",     "Join us as a Data Scientist with a solid background on time-series analysis for health monitoring applications."),
        ("Automated Feature Extraction",      "Develop and refine algorithms for automated feature extraction and anomaly detection from high-frequency time-series data."),
        ("Anomaly Detection Algorithms",      "Engineer robust anomaly detection algorithms to filter noise and handle non-stationary environmental effects in real-world sensor data."),
        ("Data Validation & Quality",         "Implement automated data validation and quality layers to identify sensor malfunctions, drifts, or data gaps."),
        ("Complex Health Metrics Visualization", "Visualise complex health metrics to provide clear, data-driven insights for stakeholders and decision-makers."),
        ("Structural Engineering Collaboration", "Collaborate with Structural Engineers to close the loop between data-driven insights and physical inspections."),
        ("Strong Organization",               "Independent, with meticulous attention to detail, and strong organisational skills."),
        ("Collaborative Team Player",         "A collaborative team player who thrives in a dynamic and inclusive work environment."),
        ("Professional English Proficiency",  "Proficient in English, both written and verbal."),
        ("Python Scientific Stack",           "Deep proficiency in Python and the scientific stack (NumPy, SciPy, Pandas, Scikit-learn, PyTorch/TensorFlow)."),
        ("Signal Processing",                 "Expertise in Signal Processing (FFT, Filtering, Wavelets) and its application to time-history data."),
        ("Structural Health Monitoring",      "Experience with SHM and predictive maintenance methodologies is a significant advantage."),
        ("Predictive Maintenance Methodologies", "Experience with SHM and predictive maintenance methodologies is a significant advantage."),
        ("SQL & NoSQL Databases",             "Solid understanding of SQL/NoSQL and experience working with large datasets in cloud environments."),
        ("Version Control Git",               "Familiarity with version control (git) and collaborative development in a CI/CD environment."),
        ("Continuous Integration & Deployment", "Familiarity with version control (git) and collaborative development in a CI/CD environment."),
    ]
}

JOB_2 = {
    "title": "Job 2 — Chemistry / Chemical Engineering",
    "terms": [
        ("Pilot-scale organic synthesis",             "Pilot-scale organic synthesis of base oil components (up to 100 L) in an ATEX-certified environment."),
        ("Lubricants formulation and polymer synthesis", "Laboratory-scale lubricants formulation and polymer synthesis."),
        ("Raw material and reagent ordering",         "Raw material and reagent ordering."),
        ("In-process and quality control",            "In-process controls (IPC) and quality control (QC)."),
        ("Product purification",                      "Product purification."),
        ("Production equipment maintenance",          "Operation and routine maintenance of production equipment."),
        ("SOP development and documentation",         "Development, adaptation, and documentation of SOPs for synthesis and equipment maintenance."),
        ("Pilot plant operations support",            "Experience operating and supporting pilot plant activities in the chemical industry."),
        ("Collaborative working mindset",             "Strong collaborative mindset with the ability to work autonomously and take initiative."),
        ("Technical communication skills",            "Excellent communication skills with both technical and non-technical stakeholders."),
        ("Structured working style",                  "Structured, rigorous, and well-organized working style."),
        ("Proactive solution-oriented mindset",       "Proactive and solution-oriented mindset."),
        ("Chemical Engineering or Chemistry",         "Degree in Chemical Engineering or Chemistry, with a strong background in organic synthesis and industrial processes."),
        ("Organic synthesis",                         "Degree in Chemical Engineering or Chemistry, with a strong background in organic synthesis and industrial processes."),
        ("Short-path distillation processes",         "Knowledge of short-path distillation processes is an advantage."),
        ("Enterprise Resource Planning systems",      "Experience with ERP systems and proficiency in MS Office."),
        ("Microsoft Office proficiency",              "Experience with ERP systems and proficiency in MS Office."),
    ]
}

JOB_3 = {
    "title": "Job 3 — ML / AI Engineering",
    "terms": [
        ("Production-grade Machine Learning Models",    "Design, train, and ship production-grade ML models including deep learning, NLP, and computer vision systems."),
        ("Natural Language Processing Systems",         "Design, train, and ship production-grade ML models including deep learning, NLP, and computer vision systems."),
        ("Computer Vision Systems",                     "Design, train, and ship production-grade ML models including deep learning, NLP, and computer vision systems."),
        ("Foundation Model Fine-Tuning",                "Apply advanced fine-tuning strategies (e.g., PEFT, LoRA) to adapt state-of-the-art foundation models to specific domain tasks."),
        ("Scalable Machine Learning Pipelines",         "Architect scalable ML pipelines for data processing, feature engineering, training, and evaluation."),
        ("Machine Learning Performance Optimization",   "Optimize model performance for latency, throughput, and resource utilization."),
        ("Machine Learning Model Deployment Automation","Champion MLOps excellence by automating deployment workflows and implementing CI/CD for ML."),
        ("Model Drift Monitoring",                      "Champion MLOps excellence by establishing robust monitoring for model drift and health."),
        ("Distributed Training Systems",                "Expertise in PyTorch or JAX, including experience with distributed training (DDP, FSDP)."),
        ("End-to-End Machine Learning Delivery",        "Track record of end-to-end ML delivery, from exploratory data analysis to deploying models in production."),
        ("High-Throughput Inference Systems",           "Experience designing resilient architectures that handle vast datasets and high-throughput inference requests."),
        ("Exploratory Data Analysis",                   "Conduct deep exploratory research on massive datasets to uncover novel patterns in user behavior."),
        ("Feature Engineering",                         "Architect scalable ML pipelines for data processing, feature engineering, training, and evaluation."),
        ("Cross-Functional Collaboration",              "Collaborate cross-functionally with data engineers, product managers, and software engineers."),
        ("Machine Learning Theory",                     "Strong foundation in ML theory and statistics, including hypothesis testing and optimization techniques."),
        ("Statistical Modeling Techniques",             "Strong foundation in ML theory and statistics, including regression, classification, and optimization techniques."),
        ("Python Programming",                          "You are comfortable writing production-level Python and have a deep understanding of data structures and algorithms."),
        ("Distributed System Design",                   "You have a deep understanding of data structures, algorithms, and distributed system design."),
        ("Modern Machine Learning Stack",               "Deep proficiency in Python and the modern ML stack, with hands-on experience using PyTorch, TensorFlow."),
        ("Transformer Architectures",                   "Evaluating novel algorithms and techniques such as Transformer architectures and quantization to drive innovation."),
    ]
}

JOBS = [JOB_1, JOB_2, JOB_3]

# =============================================================
# LOAD INDEX
# =============================================================

def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(ESCO_JSON, "r", encoding="utf-8") as f:
        esco_data = json.load(f)
    return embeddings, labels, esco_data

# =============================================================
# STAGE 1 — SEMANTIC SEARCH
# =============================================================

def semantic_search(term, embeddings, labels, bi_encoder, top_k=10):
    query_vec = bi_encoder.encode([term])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], labels[i]) for i in top_indices]

# =============================================================
# STAGE 2 — CROSS-ENCODER RERANKING
# =============================================================

def rerank(term, source_sentence, candidates, esco_data, cross_encoder):
    """
    The cross-encoder reads [term + source sentence] against each
    ESCO candidate (label + description) and assigns a relevance score.
    Candidates are re-sorted by this score.
    """
    query = f"{term}. {source_sentence}"
    pairs = []
    for _, label in candidates:
        info = esco_data.get(label, {})
        description = info.get("description", "")
        candidate_text = f"{label}. {description}"
        pairs.append([query, candidate_text])

    scores = cross_encoder.predict(pairs)
    reranked = sorted(
        zip(scores, [c[1] for c in candidates]),
        key=lambda x: x[0],
        reverse=True
    )
    return reranked

# =============================================================
# CLASSIFY MATCH QUALITY
# =============================================================

def classify(score, stage="semantic"):
    if stage == "semantic":
        if score >= 0.69: return "STRONG"
        if score >= 0.50: return "PARTIAL"
        return "WEAK"
    else:
        if score >= 1.0:  return "STRONG"
        if score >= -3.0: return "PARTIAL"
        return "WEAK"

# =============================================================
# MAIN EVALUATION
# =============================================================

if __name__ == "__main__":

    print("Loading index and models...")
    embeddings, labels, esco_data = load_index()
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"Index loaded: {embeddings.shape[0]} ESCO records\n")

    summary = []

    for job in JOBS:
        print(f"\n{'='*65}")
        print(f"  {job['title']}")
        print(f"{'='*65}")

        job_results = {"title": job["title"], "strong": 0, "partial": 0, "weak": 0}

        for term, source_sentence in job["terms"]:
            print(f"\n  Term: {term}")
            print(f"  Source: {source_sentence[:80]}...")

            # Stage 1
            sem_candidates = semantic_search(term, embeddings, labels, bi_encoder, top_k=10)

            # Stage 2
            reranked = rerank(term, source_sentence, sem_candidates, esco_data, cross_encoder)

            # Display top-5
            print(f"\n  {'SEMANTIC (top-5)':<35}  {'RERANKED (top-5)'}")
            print(f"  {'-'*33}  {'-'*33}")
            for i in range(5):
                s_score, s_label = sem_candidates[i]
                r_score, r_label = reranked[i]
                print(f"  {s_score:.3f}  {s_label[:28]:<28}  {r_score:6.2f}  {r_label[:28]}")

            # Track match quality based on reranked top-1
            top_score = reranked[0][0]
            quality = classify(top_score, stage="reranked")
            job_results[quality.lower()] += 1

        summary.append(job_results)

    # Summary table
    print(f"\n\n{'='*65}")
    print("  SUMMARY — Match quality per job (based on reranked top-1)")
    print(f"{'='*65}")
    print(f"  {'Job':<40} {'Strong':>8} {'Partial':>8} {'Weak':>8}")
    print(f"  {'-'*62}")
    for r in summary:
        print(f"  {r['title'][:38]:<40} {r['strong']:>8} {r['partial']:>8} {r['weak']:>8}")
