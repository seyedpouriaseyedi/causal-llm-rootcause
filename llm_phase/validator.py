# llm_phase/validator.py
import json
import re
from typing import Any, List, Set, Tuple


def _collect_all_strings(x: Any) -> List[str]:
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, list):
        for i in x:
            out.extend(_collect_all_strings(i))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_collect_all_strings(v))
    return out


def validate_llm_json(text: str, allowed_vars: Set[str], allowed_edges: Set[Tuple[str, str]]) -> dict:
    """
    Validates:
      1) Output is JSON
      2) Required keys exist (minimal schema check)
      3) top_cause.variable must be allowed
      4) top_cause.causal_chain must use only allowed edges
      5) No forbidden variable names appear anywhere in the JSON text fields
    """
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")

    # ---- minimal required keys ----
    required_keys = ["target", "incident_index", "top_cause", "alternatives", "recommended_actions", "validation_tests", "limitations"]
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["top_cause"], dict):
        raise ValueError("top_cause must be an object/dict")

    # ---- top cause variable must be allowed ----
    top_var = obj["top_cause"].get("variable")
    if top_var not in allowed_vars:
        raise ValueError(f"Top cause variable not allowed: {top_var}")

    # ---- causal_chain edges must be allowed ----
    chain = obj["top_cause"].get("causal_chain", [])
    if not isinstance(chain, list) or len(chain) < 2:
        raise ValueError("top_cause.causal_chain must be a list of >=2 variables")

    for a, b in zip(chain[:-1], chain[1:]):
        if (a, b) not in allowed_edges:
            raise ValueError(f"Causal chain uses forbidden edge: {a} -> {b}")

    # ---- forbid mentioning variables not in allowed_vars anywhere ----
    # We scan all strings and look for exact variable name occurrences as whole tokens.
    all_strings = _collect_all_strings(obj)
    combined_text = "\n".join(all_strings)

    # Build regex for forbidden variables (only those that appear in text)
    # (We can't enumerate all possible words; we block exact matches for known column names.)
    for var in sorted(list(allowed_vars)):
        pass  # just to keep intent clear

    # Find any variable-like tokens in text and compare to allowed vars
    # This is conservative: it catches explicit mentions of dataset variables.
    # It won't catch subtle paraphrases, which is fine.
    # Tokenization: split on non-word plus allow spaces by scanning for exact var substring boundaries.
    for maybe_var in _extract_possible_var_mentions(combined_text):
        if maybe_var not in allowed_vars:
            raise ValueError(f"Output mentions forbidden variable: '{maybe_var}'")

    return obj


def _extract_possible_var_mentions(text: str) -> Set[str]:
    """
    Extracts phrases that look like dataset variables by matching patterns like:
    - words with spaces (e.g., "Air temperature")
    - words with underscores (e.g., "Tool_wear")
    We do this by scanning for sequences of letters/numbers/space/underscore.
    """
    candidates = set()
    # pick up sequences that look like variable names (length >= 3)
    for m in re.finditer(r"[A-Za-z0-9_][A-Za-z0-9_ ]{2,}", text):
        s = m.group(0).strip()
        # remove trailing punctuation-like spaces
        s = re.sub(r"\s+", " ", s)
        candidates.add(s)
    return candidates
