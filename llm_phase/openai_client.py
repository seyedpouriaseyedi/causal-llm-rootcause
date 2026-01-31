# llm_phase/openai_client.py
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


def _get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets if present, otherwise env var.
    # (We import streamlit lazily so this module still works outside Streamlit.)
    try:
        import streamlit as st  # type: ignore
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def _client() -> OpenAI:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env var OPENAI_API_KEY or Streamlit secret OPENAI_API_KEY."
        )
    return OpenAI(api_key=api_key)


# -----------------------------
# Structured output schema
# -----------------------------
class RootCauseReport(BaseModel):
    """
    Matches your validator expectation (top-level keys).
    Keep this strict to avoid random extra keys.
    """
    model_config = ConfigDict(extra="forbid")

    target: str
    incident_index: int
    top_cause: str
    alternatives: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    validation_tests: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


def generate_root_cause_report_json(
    prompt: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """
    Calls OpenAI with Structured Outputs so the model MUST return a JSON object
    matching RootCauseReport. This avoids your users pasting the wrong JSON.
    """
    client = _client()

    # Responses API + Pydantic structured outputs:
    # OpenAI docs: client.responses.parse(..., text_format=YourPydanticModel)
    # :contentReference[oaicite:2]{index=2}
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return ONLY the JSON object that matches the schema."},
            {"role": "user", "content": prompt},
        ],
        text_format=RootCauseReport,
    )

    # Parsed Pydantic object:
    report_obj: RootCauseReport = resp.output_parsed
    return report_obj.model_dump()


def chat_grounded(
    messages: List[Dict[str, str]],
    model: str = "gpt-5-mini",
    verbosity: str = "medium",
) -> str:
    """
    Normal chat call (no schema). Uses Responses API.
    """
    client = _client()

    resp = client.responses.create(
        model=model,
        input=messages,
        text={"verbosity": verbosity},
        reasoning={"effort": "none"},  # fast + cheap for chat
    )

    # The SDK provides helpers, but safest is to extract text from outputs.
    # We'll use output_text if present, else fallback.
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback extraction
    out = []
    for item in getattr(resp, "output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    out.append(c["text"])
    return "\n".join(out).strip()
