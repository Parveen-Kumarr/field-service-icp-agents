"""Make a model's structured answer safe to use.

Local models sometimes run out of output tokens mid-answer (truncated JSON) or
leave out required fields. Instead of crashing an agent with a KeyError, we
repair the JSON where possible and fill any missing required field with a
neutral default taken from the schema ("unknown", 0, [], ...), and log what
was filled so it's visible.
"""
from __future__ import annotations

import json
from typing import Any

NEUTRAL_ENUM = ("unknown", "none", "low", "other")


def default_for(schema: dict) -> Any:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if "enum" in schema:
        return next((v for v in NEUTRAL_ENUM if v in schema["enum"]), schema["enum"][0])
    if t == "string":
        return "not provided"
    if t in ("integer", "number"):
        return schema.get("minimum", 0)
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return fill_required({}, schema)[0]
    return None


def conform(value: Any, schema: dict) -> Any:
    """Bend a value into the shape its schema expects, where the intent is clear:
    a JSON string where an object/array is expected is parsed; a plain string inside an array of objects
    becomes {first_property: string}; a single item where an array is expected is wrapped; numbers sent as
    strings are converted. Anything unclear is left alone."""
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "object":
        if isinstance(value, str):
            value = repair_json(value) or value
        if isinstance(value, dict):
            props = schema.get("properties", {})
            return {k: conform(v, props[k]) if k in props else v for k, v in value.items()}
        return value
    if t == "array":
        if isinstance(value, str):
            parsed = None
            if value.strip().startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
            value = parsed if isinstance(parsed, list) else [value]
        elif isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            items = schema.get("items", {})
            if items.get("type") == "object" and items.get("properties"):
                first = next(iter(items["properties"]))
                value = [{first: v} if isinstance(v, str) else v for v in value]
                value = [v for v in value if isinstance(v, dict)]
            return [conform(v, items) for v in value] if items else value
        return value
    if t in ("integer", "number") and isinstance(value, str):
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return int(float(m.group())) if t == "integer" else float(m.group())
    return value


def fill_required(output: Any, schema: dict, path: str = "") -> tuple[dict, list[str]]:
    """Return (output with every required field present, list of fields that were filled).
    The output is first bent into the schema's shape (see conform)."""
    if isinstance(output, dict) and not path:
        output = conform(output, schema)
    if not isinstance(output, dict):
        output = {}
    filled: list[str] = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        sub = props.get(key, {})
        if key not in output or output[key] is None:
            output[key] = default_for(sub)
            filled.append(path + key)
        elif sub.get("type") == "object" and isinstance(output[key], dict):
            output[key], inner = fill_required(output[key], sub, f"{path}{key}.")
            filled += inner
        elif "enum" in sub and output[key] not in sub["enum"]:
            output[key] = default_for(sub)
            filled.append(path + key)
    return output, filled


def repair_json(text: str) -> dict | None:
    """Parse JSON that may be wrapped in prose or cut off mid-way (closes open strings/brackets)."""
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # walk the text, tracking open containers; remember the last point where a value ended cleanly
    stack: list[str] = []
    in_str = esc = False
    last_good = -1
    last_good_stack: list[str] = []
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:  # complete top-level object followed by trailing text
                try:
                    return json.loads(body[:i + 1])
                except json.JSONDecodeError:
                    return None
            last_good, last_good_stack = i, list(stack)
        elif ch == ",":
            last_good, last_good_stack = i - 1, list(stack)
    if last_good < 0:
        return None
    candidate = body[:last_good + 1].rstrip().rstrip(",") + "".join(reversed(last_good_stack))
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
