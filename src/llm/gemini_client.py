from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _build_function_schema() -> Dict[str, Any]:
    """gemini function declaration for structured output"""
    return {
        "name": "trading_decisions",
        "description": "Return trading decisions for a portfolio of stocks",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": "List of trading decisions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Stock ticker symbol"},
                            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"], "description": "Trading action"},
                            "target_weight": {"type": "number", "description": "Target portfolio weight (0.0-1.0)"},
                            "reason": {"type": "string", "description": "Brief rationale for decision"},
                        },
                        "required": ["symbol", "action", "target_weight", "reason"],
                    },
                },
            },
            "required": ["decisions"],
        },
    }


def call_gemini_json(prompt: str, model_name: str = "gemini-1.5-pro-latest", temperature: float = 0.4, use_structured: bool = True) -> Dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"error": "missing_api_key", "decisions": []}
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        
        # try structured mode first
        if use_structured:
            try:
                schema = _build_function_schema()
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature},
                    tools=[{"function_declarations": [schema]}],
                )
                # parse function call response
                if hasattr(resp, "candidates") and resp.candidates:
                    for part in resp.candidates[0].content.parts:
                        if hasattr(part, "function_call"):
                            args = dict(part.function_call.args)
                            return args
            except Exception as e:
                # fallback to text parsing if structured fails
                pass
        
        # text parsing fallback
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        txt = resp.text or ""
        # attempt to parse json from response
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(txt[start : end + 1])
            return payload
        # fallback: try to parse a code block
        if "```" in txt:
            blk = txt.split("```")
            for seg in blk:
                seg = seg.strip()
                if seg.startswith("json"):
                    seg = seg[4:].strip()
                if seg.startswith("{") and seg.endswith("}"):
                    return json.loads(seg)
        return {"error": "no_json", "raw": txt, "decisions": []}
    except Exception as e:
        return {"error": str(e), "decisions": []}


def build_prompt(portfolio: Dict[str, Any], summary_rows: List[str], rules: Dict[str, Any], allowed_symbols: List[str]) -> str:
    header = (
        "You are an autonomous trading agent in a head-to-head competition. "
        "Decide BUY/SELL/HOLD and target weights for up to N tickers. "
        "Universe is strictly limited to the provided allowed_symbols list. Do not invent symbols. "
        "Use ONLY the provided DATA_CSV and PORTFOLIO; do not fetch external data. "
        "Respond ONLY with strict JSON matching the schema."
    )
    schema = {
        "decisions": [
            {
                "symbol": "string",
                "action": "BUY|SELL|HOLD",
                "target_weight": 0.0,
                "reason": "short text",
            }
        ]
    }
    rules_txt = json.dumps(rules)
    pf_txt = json.dumps(portfolio)
    table = "symbol,last_px,r1d,r5d,rsi\n" + "\n".join(summary_rows)
    allow = json.dumps(sorted(list(set(allowed_symbols))))
    return (
        f"{header}\n\nALLOWED_SYMBOLS:{allow}\n\nRULES:{rules_txt}\n\nPORTFOLIO:{pf_txt}\n\nDATA_CSV:\n{table}\n\n"
        f"Return JSON only with keys: {list(schema.keys())}."
    )
