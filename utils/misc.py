import re

def truncate_after_boxed(raw_text: str) -> str:
    match = re.search(r"\\boxed\{.*?\}", raw_text) # Match \boxed{...} including the prefix
    if match:
        return raw_text[:match.end()]
    else:
        return raw_text
