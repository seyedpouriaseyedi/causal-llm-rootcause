import json
from typing import Any, Dict, List, Set, Tuple


REQUIRED_KEYS = [
    "target",
    "incident_index",
    "top_cause",
    "alternatives",
    "recommended_actions",
    "validation_tests",
    "limitations",
]


def _ensure_list_of_str(x: Any, field: str) -> List[str]:
    if x is None:
        return []
    if not isinstance(x, list) or any(not isinstance(i, str) for i in x):
        raise ValueError(f"{field} must be a list of strings")
    return x


def _validate_cause_obj(
    cause: Any,
    target: str,
    allowed_vars: Set[str],
    allowed_edges: Set[Tuple[str, str]],
    field_name: str,
) -> Dict[str, Any]:
    if not isinstance(cause, dict):
        raise ValueError(f"{field_name} must be an object/dict")

    if "variable" not in cause or "causal_chain" not in cause:
        raise ValueError(f"{field_name} must contain keys: variable, causal_chain")

    var = cause.get("variable")
    chain = cause.get("causal_chain")

    if not isinstance(var, str):
        raise ValueError(f"{field_name}.variable must be a string")

    if var not in allowed_vars:
        raise ValueError(f"{field_name}.variable not allowed: {var}")

    if not isinstance(chain, list) or len(chain) < 2 or any(not isinstance(x, str) for x in chain):
        raise ValueError(f"{field_name}.causal_chain must be a list of >=2 strings")

    # Must start at the stated variable and end at the report target
    if chain[0] != var:
        raise ValueError(f"{field_name}.causal_chain must start with variable '{var}' (got '{chain[0]}')")

    if chain[-1] != target:
        raise ValueError(f"{field_name}.causal_chain must end with target '{target}' (got '{chain[-1]}')")

    # Variables must be allowed
    for node in chain:
        if node not in allowed_vars:
            raise ValueError(f"{field_name}.causal_chain uses forbidden variable: {node}")

    # Edges must be allowed
    for a, b in zip(chain[:-1], chain[1:]):
        if (a, b) not in allowed_edges:
            raise ValueError(f"{field_name}.causal_chain uses forbidden edge: {a} -> {b}")

    return {"variable": var, "causal_chain": chain}


def validate_llm_json(text: str, allowed_vars: Set[str], allowed_edges: Set[Tuple[str, str]]) -> dict:
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"LLM output is not valid JSON: {e}")

    for k in REQUIRED_KEYS:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(obj["target"], str):
        raise ValueError("target must be a string")
    if obj["target"] not in allowed_vars:
        raise ValueError(f"target not allowed: {obj['target']}")

    if not isinstance(obj["incident_index"], int):
        raise ValueError("incident_index must be an integer")

    target = obj["target"]

    obj["top_cause"] = _validate_cause_obj(
        obj["top_cause"], target=target, allowed_vars=allowed_vars, allowed_edges=allowed_edges, field_name="top_cause"
    )

    if not isinstance(obj["alternatives"], list):
        raise ValueError("alternatives must be a list")

    cleaned_alts = []
    for i, alt in enumerate(obj["alternatives"]):
        cleaned_alts.append(
            _validate_cause_obj(
                alt, target=target, allowed_vars=allowed_vars, allowed_edges=allowed_edges, field_name=f"alternatives[{i}]"
            )
        )
    obj["alternatives"] = cleaned_alts

    obj["recommended_actions"] = _ensure_list_of_str(obj.get("recommended_actions"), "recommended_actions")
    obj["validation_tests"] = _ensure_list_of_str(obj.get("validation_tests"), "validation_tests")
    obj["limitations"] = _ensure_list_of_str(obj.get("limitations"), "limitations")

    return obj
