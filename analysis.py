import json
import re
from pathlib import Path

# Path to your outputs.jsonl
OUTPUT_JSONL_PATH = Path("outputs.jsonl")
EXTRACTED_JSON_PATH = Path("extracted_reasoning_outputs.jsonl")

# Regular expression to extract JSON block from text
JSON_PATTERN = re.compile(r'{.*}', re.DOTALL)

def extract_json_from_text(text):
    """Extract the first JSON-like block from text."""
    match = JSON_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("⚠️ Failed to decode JSON block.")
            return None
    return None

def main():
    extracted_data = []

    with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                data = json.loads(line)
                raw_output = data.get("output", "")
                parsed_json = extract_json_from_text(raw_output)
                if parsed_json:
                    extracted_data.append(parsed_json)
            except json.JSONDecodeError:
                print("⚠️ Skipping invalid JSON line.")
                continue

    # Save extracted JSONs
    with open(EXTRACTED_JSON_PATH, "w", encoding="utf-8") as outfile:
        for item in extracted_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Extracted {len(extracted_data)} JSON objects to {EXTRACTED_JSON_PATH}")

if __name__ == "__main__":
    main()
