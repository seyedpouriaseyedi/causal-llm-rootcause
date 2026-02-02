import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


def _get_api_key() -> Optional[str]:
    # Prefer Streamlit secrets if present, otherwise env var.
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
class Cause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variable: str
    causal_chain: List[str] = Field(min_length=2)


class RootCauseReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    incident_index: int
    top_cause: Cause
    alternatives: List[Cause] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    validation_tests: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


def generate_root_cause_report_json(
    prompt: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """
    Calls OpenAI with Structured Outputs so the model MUST return a JSON object
    matching RootCauseReport.
    """
    client = _client()

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return ONLY valid JSON that matches the schema. No extra keys."},
            {"role": "user", "content": prompt},
        ],
        text_format=RootCauseReport,
    )

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
        reasoning={"effort": "none"},
    )

    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    return "No text returned by the model."
