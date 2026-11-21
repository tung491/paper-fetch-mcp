"""The single response envelope every MCP tool returns through (spec §7 "Output envelope").

``respond()`` serialises a tool's result to YAML (existing clients and skills
parse YAML), adds a top-level ``meta`` block, and guarantees the returned
string never exceeds ``hard_cap`` characters — 74 responses overflowed the
client cap in the month mined for the refactor design doc, and that must not
happen again.

This module is a pure function library: no I/O, no global state, and
``respond()`` is guaranteed to never raise.

Moved here from ``paper_fetch_mcp/output.py`` per the refactor design spec
§5 "Target architecture" (docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

DEFAULT_BUDGET = 20_000
HARD_CAP = 40_000

_TRUNC_MARKER = "\n…[truncated]"


def respond(
    payload: Any,
    *,
    budget: int = DEFAULT_BUDGET,
    hard_cap: int = HARD_CAP,
    list_key: Optional[str] = None,
    next_hint: Optional[Union[List[str], str]] = None,
) -> str:
    """Serialise payload to a YAML envelope, truncated to fit under hard_cap.

    Never raises. Never returns a string longer than hard_cap.
    """
    try:
        text = _build_envelope(
            payload, budget=budget, hard_cap=hard_cap, list_key=list_key, next_hint=next_hint
        )
    except Exception:
        text = None

    if not isinstance(text, str):
        text = _fallback_envelope(budget)

    if len(text) > hard_cap:
        text = _clip_raw(text, hard_cap)

    return text


def error(message: str, *, kind: str = "error", **extra) -> str:
    """Build an error envelope. Routes through respond() so it shares the same guarantees."""
    payload: Dict[str, Any] = {"error": message, "kind": kind}
    payload.update(extra)
    return respond(payload)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _dump(data: Dict[str, Any]) -> str:
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _fallback_envelope(budget: int) -> str:
    try:
        return _dump(
            {
                "error": "respond_failed",
                "meta": {"returned": 0, "total": 0, "truncated": True, "budget_chars": budget},
            }
        )
    except Exception:
        return "error: respond_failed\n"


def _clip_raw(text: str, hard_cap: int) -> str:
    if hard_cap <= 0:
        return ""
    if hard_cap <= len(_TRUNC_MARKER):
        return text[:hard_cap]
    return text[: hard_cap - len(_TRUNC_MARKER)] + _TRUNC_MARKER


def _find_main_list(data: Dict[str, Any], list_key: Optional[str]) -> Tuple[Optional[str], Optional[list]]:
    """Locate the payload's main truncatable list.

    If list_key is given, use payload[list_key] when it is a list.
    Otherwise use the first value in the dict that is a list of dicts
    (an empty list also counts, vacuously).
    """
    if list_key is not None:
        val = data.get(list_key)
        if isinstance(val, list):
            return list_key, val
        return None, None

    for key, val in data.items():
        if isinstance(val, list) and all(isinstance(item, dict) for item in val):
            return key, val

    return None, None


def _make_meta(
    returned: int,
    total: int,
    truncated: bool,
    budget: int,
    next_hint: Optional[Union[List[str], str]],
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "returned": returned,
        "total": total,
        "truncated": truncated,
        "budget_chars": budget,
    }
    if next_hint is not None:
        meta["next"] = next_hint
    elif truncated:
        meta["next"] = f"re-query with offset={returned}"
    return meta


def _iter_string_leaves(obj: Any):
    """Yield (container, key_or_index, string_value) for every string leaf, recursively."""
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return

    for key, val in items:
        if isinstance(val, str):
            yield obj, key, val
        elif isinstance(val, (dict, list)):
            yield from _iter_string_leaves(val)


def _trim_to_budget(
    data: Dict[str, Any],
    key: str,
    full_list: list,
    budget: int,
    next_hint: Optional[Union[List[str], str]],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Binary-search the largest prefix of full_list that fits under budget.

    Items are kept from the head and dropped from the tail, per spec.
    """
    total = len(full_list)

    def render(k: int) -> Tuple[Dict[str, Any], str]:
        truncated = k < total
        data[key] = full_list[:k]
        meta = _make_meta(k, total, truncated, budget, next_hint if truncated else None)
        wrapper = data.copy()
        wrapper["meta"] = meta
        return meta, _dump(wrapper)

    # Fast path: everything fits.
    full_meta, full_candidate = render(total)
    if len(full_candidate) <= budget or total == 0:
        return data, full_meta, full_candidate

    lo, hi = 0, total - 1
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _, candidate = render(mid)
        if len(candidate) <= budget:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # render() mutates data[key] as a side effect on every call, so the last
    # probe in the loop above may not have been the winning one. Re-render
    # the winner last so data/meta/candidate agree with each other.
    best_meta, best_candidate = render(best_k)
    return data, best_meta, best_candidate


def _hard_truncate_strings(
    data: Dict[str, Any], meta: Dict[str, Any], hard_cap: int
) -> Tuple[Dict[str, Any], str]:
    """Last resort before the raw-string clip: shrink the largest string leaves."""
    for _ in range(100):
        wrapper = data.copy()
        wrapper["meta"] = meta
        candidate = _dump(wrapper)
        if len(candidate) <= hard_cap:
            return meta, candidate

        leaves = list(_iter_string_leaves(data))
        if not leaves:
            return meta, candidate

        container, leaf_key, value = max(leaves, key=lambda item: len(item[2]))
        if len(value) < 100:
            # Nothing large enough left to usefully cut.
            return meta, candidate

        overflow = len(candidate) - hard_cap
        cut = min(len(value), max(overflow + len(_TRUNC_MARKER) + 32, len(value) // 2))
        new_value = value[: max(0, len(value) - cut)].rstrip() + _TRUNC_MARKER
        container[leaf_key] = new_value
        meta["truncated"] = True

    wrapper = data.copy()
    wrapper["meta"] = meta
    return meta, _dump(wrapper)


def _build_envelope(
    payload: Any,
    *,
    budget: int,
    hard_cap: int,
    list_key: Optional[str],
    next_hint: Optional[Union[List[str], str]],
) -> str:
    if isinstance(payload, dict):
        # Deep-copy: the truncation/hard-cap passes below mutate nested
        # containers in place, and must never mutate the caller's payload.
        data: Dict[str, Any] = copy.deepcopy(payload)
    else:
        data = {"content": payload}

    key, main_list = _find_main_list(data, list_key)

    if main_list is not None and key is not None:
        data, meta, candidate = _trim_to_budget(data, key, list(main_list), budget, next_hint)
    else:
        meta = _make_meta(1, 1, False, budget, next_hint)
        wrapper = data.copy()
        wrapper["meta"] = meta
        candidate = _dump(wrapper)

    if len(candidate) > hard_cap:
        meta, candidate = _hard_truncate_strings(data, meta, hard_cap)

    return candidate
