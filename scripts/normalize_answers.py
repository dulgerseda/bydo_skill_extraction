import json
from pathlib import Path
from copy import deepcopy


INPUT_PATH = Path("answers/answers.json")
OUTPUT_PATH = Path("answers/normalized_answers.json")


ACTION_KEYWORDS = [
    "develop",
    "build",
    "implement",
    "apply",
    "optimize",
    "process",
    "analyze",
    "analyse",
    "detect",
    "monitor",
    "design",
    "deploy",
    "evaluate",
    "filter",
    "handle",
    "manage",
    "estimate",
    "investigate",
    "create",
    "validate",
    "visualise",
    "visualize",
    "communicate",
    "collaborate",
]


def normalize_skill_type(skill: dict) -> str:
    """
    Safely normalizes skill categories without losing metadata like 'gap'.
    """
    # 1. Sanitize inputs
    noun = str(skill.get("noun", "")).strip().lower()
    verb = str(skill.get("verb", "")).strip().lower()
    category = str(skill.get("category", "")).strip()
    
    # 2. Ensure 'gap' key exists to prevent Streamlit KeyError
    if "gap" not in skill:
        skill["gap"] = False

    # 3. Aggressive Overrides for Core Technologies
    core_techs = ["python", "sql", "nosql", "java", "r language", "c++", "git", "excel"]
    if any(tech in noun for tech in core_techs):
        # Force generic terms to Knowledge (K)
        if len(noun.split()) <= 2 or any(w in noun for w in ["programming", "scripting", "language"]):
            category = "K"
            skill["category"] = "K"

    # 4. Return mappings based on category
    if category == "SF":
        return "Soft Skill"
        
    if category == "K":
        tool_keywords = core_techs + ["docker", "aws", "database", "spark", "pytorch"]
        if skill.get("tool") or any(t in noun for t in tool_keywords):
            return "Tool"
        return "Knowledge"

    if category == "IPS":
        if any(term in noun for term in ["development", "design", "architecture"]):
            return "Methodology"
        return "Tool"

    # Default mapping for OS and others
    manual_overrides = {
        "data visualization": "Methodology",
        "machine learning": "Methodology",
        "ai llm model development": "Methodology"
    }
    
    if noun in manual_overrides:
        return manual_overrides[noun]

    return "Methodology"



def normalize_answers(data: dict) -> dict:
    normalized = deepcopy(data)

    for jd_key, jd_value in normalized.items():
        skills = jd_value.get("skills", [])
        for skill in skills:
            skill["skill_type"] = normalize_skill_type(skill)

    return normalized


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized_data = normalize_answers(raw_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(normalized_data, f, indent=2, ensure_ascii=False)

    print(f"Normalized answers written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()