import json
from pathlib import Path
from copy import deepcopy

INPUT_DIR  = Path("answers")
OUTPUT_DIR = Path("normalized_answers")


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
    noun     = str(skill.get("noun", "")).strip().lower()
    category = str(skill.get("category", "")).strip()
    tool     = skill.get("tool")
    action   = skill.get("action")

    if "gap" not in skill:
        skill["gap"] = False

    if category == "SF":
        return "Soft Skill"

    if category == "IPS":
        if tool and action:
            return "Tool"
        skill["category"] = "OS"
        category = "OS"

    if category == "K":
        tool_signals = [
            "programming", "database", "stack", "framework", "library",
            "platform", "cloud", "containerization", "version control",
            "monitoring", "tool", "engine", "sdk", "api"
        ]
        if any(sig in noun for sig in tool_signals):
            return "Tool"
        if skill.get("skill_type") == "Tool":
            return "Tool"
        return "Knowledge"

    if category == "OS":
        return "Methodology"

    return "Methodology"


def normalize_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized = deepcopy(raw_data)
    for jd_key, jd_value in normalized.items():
        for skill in jd_value.get("skills", []):
            skill["skill_type"] = normalize_skill_type(skill)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(f"✓ {input_path.name} → {output_path}")


def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")

    for input_path in json_files:
        output_path = OUTPUT_DIR / input_path.name
        normalize_file(input_path, output_path)

    print(f"\nDone! {len(json_files)} file(s) normalized.")


if __name__ == "__main__":
    main()