import json
from pathlib import Path
from copy import deepcopy


INPUT_PATH = Path("answers/answers.json")
OUTPUT_PATH = Path("answers/normalized_answers_updated.json")


def normalize_skill_type(skill: dict) -> str:
    """
    Safety-net normalizer — prompt_v10 already handles correct categorization.
    This function only corrects malformed or edge-case outputs.

    Rules (aligned with prompt_v10):
    - SF  → Soft Skill
    - IPS → Tool (only if tool + action both present, else demote to OS)
    - K   → Tool (if noun signals a named platform/language) else Knowledge
    - OS  → Methodology
    """
    noun = str(skill.get("noun", "")).strip().lower()
    category = str(skill.get("category", "")).strip()
    tool = skill.get("tool")
    action = skill.get("action")

    # Ensure 'gap' key exists to prevent Streamlit KeyError
    if "gap" not in skill:
        skill["gap"] = False

    # ── SF → Soft Skill ───────────────────────────────────────────────────────
    if category == "SF":
        return "Soft Skill"

    # ── IPS → valid only when both tool and action are present ────────────────
    if category == "IPS":
        if tool and action:
            return "Tool"
        # Malformed IPS (prompt missed tool or action) → demote to OS
        skill["category"] = "OS"
        category = "OS"

    # ── K → Tool if noun signals a named platform/language, else Knowledge ────
    if category == "K":
        tool_signals = [
            "programming", "database", "stack", "framework", "library",
            "platform", "cloud", "containerization", "version control",
            "monitoring", "tool", "engine", "sdk", "api"
        ]
        if any(sig in noun for sig in tool_signals):
            return "Tool"
        # If the model already set skill_type to Tool, respect it
        if skill.get("skill_type") == "Tool":
            return "Tool"
        return "Knowledge"

    # ── OS → Methodology ──────────────────────────────────────────────────────
    if category == "OS":
        return "Methodology"

    # ── Fallback ──────────────────────────────────────────────────────────────
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
