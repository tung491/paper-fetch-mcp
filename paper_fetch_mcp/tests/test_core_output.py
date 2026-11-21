"""Tests for paper_fetch_mcp.core.output: the shared YAML response envelope.

Formerly an equivalence suite that compared core.output against the
original paper_fetch_mcp/output.py (byte-for-byte, to prove Wave 1a's move
was behaviour-preserving). Now that output.py has been deleted -- its job
done -- these assert directly against core.output, absorbing every
assertion that used to live in the now-deleted tests/test_output.py (the
suite this file's payload shapes were mirroring) plus this file's own
extra shapes and private-helper checks. No assertion from either source
file was dropped in the conversion.
"""

import threading

import yaml

from paper_fetch_mcp.core import output as core_output


def test_hard_cap_invariant_on_enormous_payload():
    # 200 results x 30 KB of text -- comfortably larger than both budgets.
    results = [{"citation_key": f"paper{i}", "content": "x" * 30_000} for i in range(200)]
    payload = {"results": results}

    text = core_output.respond(payload)

    assert isinstance(text, str)
    assert len(text) <= core_output.HARD_CAP

    parsed = yaml.safe_load(text)
    assert parsed["meta"]["truncated"] is True
    assert parsed["meta"]["total"] == 200
    assert parsed["meta"]["returned"] == len(parsed["results"])
    assert parsed["meta"]["returned"] < 200


def test_meta_not_truncated_when_payload_fits():
    payload = {"results": [{"a": 1}, {"a": 2}]}
    text = core_output.respond(payload)
    parsed = yaml.safe_load(text)

    assert parsed["meta"]["truncated"] is False
    assert parsed["meta"]["returned"] == 2
    assert parsed["meta"]["total"] == 2
    assert "next" not in parsed["meta"]


def test_bare_string_payload():
    text = core_output.respond("hello world")
    parsed = yaml.safe_load(text)

    assert parsed["content"] == "hello world"
    assert parsed["meta"]["returned"] == 1
    assert parsed["meta"]["total"] == 1
    assert parsed["meta"]["truncated"] is False


def test_payload_with_no_list_still_gets_clipped():
    payload = {"detail": "y" * 100_000, "id": 7}
    text = core_output.respond(payload)

    assert len(text) <= core_output.HARD_CAP
    parsed = yaml.safe_load(text)
    assert parsed["id"] == 7
    assert parsed["meta"]["truncated"] is True
    assert "…[truncated]" in parsed["detail"]


def test_single_string_leaf_over_hard_cap_no_list():
    # Exercises the hard-truncate-strings path with no main list at all.
    payload = {"blob": "z" * 200_000}
    text = core_output.respond(payload)

    assert len(text) <= core_output.HARD_CAP
    parsed = yaml.safe_load(text)
    assert parsed["meta"]["returned"] == 1
    assert parsed["meta"]["total"] == 1
    assert parsed["meta"]["truncated"] is True
    assert parsed["blob"].endswith(core_output._TRUNC_MARKER)


def test_nested_structure_with_deep_string_leaf():
    # A huge nested leaf forces the main list down to zero items even though
    # the list itself was small and would otherwise fit.
    payload = {"outer": {"inner": {"leaf": "w" * 50_000}}, "items": [{"i": i} for i in range(5)]}
    text = core_output.respond(payload)
    parsed = yaml.safe_load(text)

    assert len(text) <= core_output.HARD_CAP
    assert parsed["items"] == []
    assert parsed["meta"]["total"] == 5
    assert parsed["meta"]["truncated"] is True
    assert parsed["outer"]["inner"]["leaf"].endswith(core_output._TRUNC_MARKER)


def test_default_next_hint_format_when_truncated():
    payload = {"results": [{"i": i, "blob": "z" * 4000} for i in range(30)]}
    text = core_output.respond(payload, budget=6000)
    parsed = yaml.safe_load(text)

    assert parsed["meta"]["truncated"] is True
    expected = f"re-query with offset={parsed['meta']['returned']}"
    assert parsed["meta"]["next"] == expected


def test_explicit_next_hint_is_used_verbatim():
    payload = {"job_id": "abc123", "status": "running"}
    text = core_output.respond(payload, next_hint="poll job_status(job_id='abc123')")
    parsed = yaml.safe_load(text)

    assert parsed["meta"]["next"] == "poll job_status(job_id='abc123')"
    # A hint may be supplied even when nothing was truncated.
    assert parsed["meta"]["truncated"] is False


def test_list_key_present_but_not_selected_by_default_scan():
    # No list_key given: the first list-of-dicts found ("items") is treated
    # as the main list, and "other_list" (a list of ints) is left untouched.
    payload = {"other_list": [1, 2, 3], "items": [{"i": i, "blob": "v" * 3000} for i in range(40)]}
    text = core_output.respond(payload)
    parsed = yaml.safe_load(text)

    assert parsed["other_list"] == [1, 2, 3]
    assert parsed["meta"]["total"] == 40
    assert parsed["meta"]["truncated"] is True
    assert len(parsed["items"]) == parsed["meta"]["returned"] < 40


def test_list_key_selects_the_named_list():
    payload = {
        "other_list": [1, 2, 3],
        "items": [{"i": i, "blob": "w" * 3000} for i in range(40)],
    }
    text = core_output.respond(payload, list_key="items", budget=5000)
    parsed = yaml.safe_load(text)

    assert parsed["other_list"] == [1, 2, 3]
    assert len(parsed["items"]) < 40
    assert parsed["meta"]["total"] == 40


def test_empty_payload():
    text = core_output.respond({})
    parsed = yaml.safe_load(text)

    assert parsed["meta"] == {"returned": 1, "total": 1, "truncated": False, "budget_chars": core_output.DEFAULT_BUDGET}


def test_payload_with_already_empty_list():
    text = core_output.respond({"results": []})
    parsed = yaml.safe_load(text)

    assert parsed["results"] == []
    assert parsed["meta"]["returned"] == 0
    assert parsed["meta"]["total"] == 0
    assert parsed["meta"]["truncated"] is False


def test_error_helper_routes_through_respond():
    text = core_output.error("boom", kind="timeout", extra_field=1)
    parsed = yaml.safe_load(text)

    assert parsed["error"] == "boom"
    assert parsed["kind"] == "timeout"
    assert parsed["extra_field"] == 1
    assert "meta" in parsed


def test_error_default_kind():
    text = core_output.error("something went wrong")
    parsed = yaml.safe_load(text)

    assert parsed["error"] == "something went wrong"
    assert parsed["kind"] == "error"


def test_respond_never_raises_on_unserialisable_payload():
    # A thread lock can't be deep-copied or YAML-dumped -- both raise
    # TypeError. respond() must swallow that and still return a string.
    text = core_output.respond({"thing": threading.Lock()})

    assert isinstance(text, str)
    assert len(text) <= core_output.HARD_CAP


def test_respond_does_not_mutate_caller_payload():
    original_items = [{"blob": "a" * 30_000} for _ in range(50)]
    payload = {"results": original_items}

    core_output.respond(payload, budget=1000)

    assert len(payload["results"]) == 50
    assert payload["results"][0]["blob"] == "a" * 30_000
    assert payload["results"] is original_items


def test_respond_result_type_and_length_hold_across_budgets():
    payload = {"results": [{"i": i, "text": "q" * 2000} for i in range(100)]}
    for budget in (100, 1_000, 10_000, 50_000):
        text = core_output.respond(payload, budget=budget, hard_cap=core_output.HARD_CAP)
        assert isinstance(text, str)
        assert len(text) <= core_output.HARD_CAP


def test_respond_holds_across_custom_budget_and_hard_cap_combinations():
    payload = {"results": [{"i": i, "blob": "z" * 4000} for i in range(30)]}
    for budget, hard_cap in [(100, core_output.HARD_CAP), (1_000, 5_000), (6_000, core_output.HARD_CAP)]:
        text = core_output.respond(payload, budget=budget, hard_cap=hard_cap)
        assert isinstance(text, str)
        assert len(text) <= hard_cap


def test_public_constants():
    assert core_output.DEFAULT_BUDGET == 20_000
    assert core_output.HARD_CAP == 40_000


def test_private_helpers_behaviour():
    # _dump round-trips a dict through YAML faithfully.
    data = {"results": [{"i": i} for i in range(5)], "meta": {}}
    assert yaml.safe_load(core_output._dump(data)) == data

    # _fallback_envelope produces the documented error shape.
    assert yaml.safe_load(core_output._fallback_envelope(20_000)) == {
        "error": "respond_failed",
        "meta": {"returned": 0, "total": 0, "truncated": True, "budget_chars": 20_000},
    }

    # _clip_raw hard-clips to exactly hard_cap chars, ending in the marker.
    clipped = core_output._clip_raw("x" * 100, 40)
    assert len(clipped) == 40
    assert clipped.endswith(core_output._TRUNC_MARKER)

    # _find_main_list picks out the payload's list-of-dicts by identity.
    payload = {"results": [{"i": i} for i in range(10)]}
    key, val = core_output._find_main_list(payload, None)
    assert key == "results"
    assert val is payload["results"]

    # _make_meta builds the expected meta block, including the default hint.
    assert core_output._make_meta(3, 10, True, 5000, None) == {
        "returned": 3,
        "total": 10,
        "truncated": True,
        "budget_chars": 5000,
        "next": "re-query with offset=3",
    }

    # _iter_string_leaves recurses through dicts/lists yielding every string leaf.
    nested = {"a": "x" * 10, "b": [{"c": "y" * 20}]}
    leaves = [(key, value) for _container, key, value in core_output._iter_string_leaves(nested)]
    assert leaves == [("a", "x" * 10), ("c", "y" * 20)]
