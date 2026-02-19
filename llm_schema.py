"""
Column meaning summarizer: produces a 1-sentence description per column.
Supports OpenAI (if OPENAI_API_KEY) or deterministic heuristic fallback for offline use.
Sends at most 8 sample values per column, each truncated to 80 chars; never the full dataset.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

# Optional OpenAI: wrap import so missing package does not break offline mode
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None

BATCH_SIZE = 10


def truncate_value(v, max_len: int = 80) -> str:
    """Truncate a value to max_len chars for safe display/sending to LLM."""
    s = str(v).strip() if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _heuristic_description(col_name: str, dtype: str, nunique: int, missing_pct: float, sample: list) -> str:
    """
    Rule-based 1-sentence guess for column meaning.
    Used when no API key is present. Produces a deterministic best guess.
    """
    name_lower = col_name.lower()
    sample_str = str(sample)[:120] if sample else ""

    # ID-like patterns
    if any(x in name_lower for x in ("id", "guid", "uuid", "key", "number")) and nunique > 50:
        return f"Identifier or unique reference column ({nunique} unique values)."
    if "quote" in name_lower and "number" in name_lower:
        return "Quote or transaction identifier."
    if "account" in name_lower and "id" in name_lower:
        return "Account identifier or foreign key."

    # Date/time
    if "date" in name_lower or "time" in name_lower:
        if "close" in name_lower:
            return "Closing or completion date for the record."
        if "start" in name_lower:
            return "Contract or period start date."
        if "end" in name_lower:
            return "Contract or period end date."
        return "Date or timestamp field."

    # Financial / pricing
    if any(x in name_lower for x in ("price", "amount", "total", "arr", "revenue", "billed", "invoice")):
        if "unit" in name_lower and "price" in name_lower:
            return "Per-unit price or list price for the item."
        if "total" in name_lower or "amount" in name_lower:
            return "Total monetary amount or sum."
        if "arr" in name_lower:
            return "Annual recurring revenue metric."
        return "Monetary or financial value."

    # Product / opportunity
    if "product" in name_lower:
        if "name" in name_lower:
            return "Name or title of the product."
        if "family" in name_lower:
            return "Product family or category grouping."
    if "opportunity" in name_lower:
        if "owner" in name_lower:
            return "Sales or opportunity owner name."
        if "type" in name_lower:
            return "Type of opportunity (e.g. new business, renewal)."
    if "account" in name_lower and "segment" in name_lower:
        return "Account segment or customer tier."

    # Quantity / numeric
    if "quantity" in name_lower or "qty" in name_lower:
        return "Quantity or count of units."
    if "contract" in name_lower and "term" in name_lower:
        return "Contract term length (e.g. years or fraction)."
    if "description" in name_lower:
        return "Free-text or structured description of the record."
    if "close" in name_lower and "year" in name_lower:
        return "Year when the record was closed."

    # Numeric by dtype
    if "number" in str(dtype).lower() or "float" in str(dtype).lower():
        if nunique <= 5:
            return f"Low-cardinality numeric field with {nunique} distinct values."
        return "Numeric measure or quantity."
    if "object" in str(dtype).lower() or "string" in str(dtype).lower():
        if nunique < 50:
            return f"Categorical or coded field with {nunique} categories."
        return "Text or categorical field."
    if "datetime" in str(dtype).lower():
        return "Date or datetime field."
    return f"Column '{col_name}' with {nunique} unique values and {missing_pct:.0f}% missing."


def _openai_batch_descriptions(
    batch: list[tuple[str, str, int, float, list]],
) -> list[tuple[str, str]]:
    """
    Call OpenAI for a batch of column descriptions. Returns list of (description, confidence).
    Each output is exactly 1 sentence, no bullets.
    """
    if not _OPENAI_AVAILABLE:
        return [
            (_heuristic_description(c, d, n, m, s), "low")
            for c, d, n, m, s in batch
        ]
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [
            (_heuristic_description(c, d, n, m, s), "low")
            for c, d, n, m, s in batch
        ]

    items = []
    for col_name, dtype, nunique, missing_pct, sample in batch:
        sample_str = " | ".join(truncate_value(v, 80) for v in (sample or [])[:8])
        items.append({
            "column": col_name,
            "dtype": dtype,
            "nunique": nunique,
            "missing_pct": round(missing_pct, 1),
            "sample_values": sample_str,
        })

    sys_prompt = (
        "You are a data analyst. For each column provided, return exactly one sentence "
        "describing what the column likely represents. No bullet points, no lists, no numbering. "
        "Return valid JSON: {\"descriptions\": [\"sentence1.\", \"sentence2.\", ...]} with one sentence per column in order."
    )
    user_content = "Columns:\n" + json.dumps(items, indent=2)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=500,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        descs = parsed.get("descriptions", [])
        out = []
        for i, (c, d, n, m, s) in enumerate(batch):
            if i < len(descs) and descs[i]:
                sent = str(descs[i]).strip()
                if not sent.endswith("."):
                    sent = sent + "." if sent else ""
                sent = sent[:500]
                out.append((sent or _heuristic_description(c, d, n, m, s), "high"))
            else:
                out.append((_heuristic_description(c, d, n, m, s), "low"))
        return out
    except Exception:
        return [
            (_heuristic_description(c, d, n, m, s), "low")
            for c, d, n, m, s in batch
        ]


def summarize_columns(df: pd.DataFrame, sample_n: int = 8, use_openai: Optional[bool] = None) -> pd.DataFrame:
    """
    Produce a table with one row per column: column, dtype, missing_pct, nunique,
    sample_values, description, confidence (low/med/high), notes.
    If OPENAI_API_KEY exists and use_openai is True, calls OpenAI in batches of 10; else uses heuristic.
    Never sends more than sample_n values per column (default 8), each truncated to 80 chars.
    """
    if use_openai is None:
        use_openai = False
    use_openai = use_openai and _OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY"))

    # Build column infos first
    infos = []
    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        missing_pct = ser.isna().mean() * 100
        nunique = int(ser.nunique(dropna=True))
        sample_vals = [truncate_value(v, 80) for v in ser.dropna().head(sample_n).tolist()]
        sample_str = " | ".join(sample_vals) if sample_vals else ""
        infos.append((col, dtype, missing_pct, nunique, sample_vals, sample_str, ser))

    if use_openai:
        batch_results = []
        for i in range(0, len(infos), BATCH_SIZE):
            chunk = [
                (col, dtype, nunique, missing_pct, sample_vals)
                for col, dtype, missing_pct, nunique, sample_vals, _s, _ in infos[i : i + BATCH_SIZE]
            ]
            batch_results.extend(_openai_batch_descriptions(chunk))
    else:
        batch_results = [
            (_heuristic_description(col, dtype, nunique, missing_pct, sample_vals), "low")
            for col, dtype, missing_pct, nunique, sample_vals, _s, _ in infos
        ]

    rows = []
    for i, (col, dtype, missing_pct, nunique, sample_vals, sample_str, ser) in enumerate(infos):
        desc, conf = batch_results[i] if i < len(batch_results) else (
            _heuristic_description(col, dtype, nunique, missing_pct, sample_vals), "low"
        )
        notes = ""
        if missing_pct > 40:
            notes = "high_missing"
        elif nunique <= 2:
            notes = "near_constant"
        elif nunique > 200 and ser.dtype == "object":
            notes = "high_cardinality"

        rows.append({
            "column": col,
            "dtype": dtype,
            "missing_pct": round(missing_pct, 1),
            "nunique": nunique,
            "sample_values": sample_str,
            "description": desc,
            "confidence": conf,
            "notes": notes,
        })

    return pd.DataFrame(rows)


def self_check() -> bool:
    """Run a minimal self-check without API key. Returns True if module works offline."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = summarize_columns(df, sample_n=3, use_openai=False)
    return len(out) == 2 and "description" in out.columns and all(out["confidence"].isin(["low", "med", "high"]))
