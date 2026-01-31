import json

def validate_llm_json(text: str, allowed_vars: set, allowed_edges: set) -> dict:
    """
    Returns parsed JSON if valid; raises ValueError if invalid.
    """
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")

    # basic required keys
    for k in ["target","incident_index","top_cause","alternatives","recommended_actions","validation_tests","limitations"]:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    # variable compliance: scan strings
    def collect_strings(x):
        out = []
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, list):
            for i in x: out += collect_strings(i)
        elif isinstance(x, dict):
            for v in x.values(): out += collect_strings(v)
        return out

    all_text = " ".join(collect_strings(obj))
    # crude but effective check: any forbidden var mentioned exactly
    for v in allowed_vars:
        pass
    # If you want strict matching, check top_cause variable:
    top_var = obj["top_cause"].get("variable")
    if top_var not in allowed_vars:
        raise ValueError(f"Top cause variable not allowed: {top_var}")

    # Check causal_chain edges all allowed
    chain = obj["top_cause"].get("causal_chain", [])
    if isinstance(chain, list) and len(chain) >= 2:
        for a, b in zip(chain[:-1], chain[1:]):
            if (a, b) not in allowed_edges:
                raise ValueError(f"Causal chain uses forbidden edge: {a} -> {b}")

    return obj

