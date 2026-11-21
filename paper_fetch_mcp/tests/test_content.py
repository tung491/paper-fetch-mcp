"""Tests for content retrieval helpers.

Phase 1 wave 5a note: `paper_fetch_mcp/tools/content.py` was replaced with a
thin `@mcp.tool` wrapper over `paper_fetch_mcp.core.papers` (design spec
`docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md`, §5).
There is no `_get_paper_content` in the new `tools/content.py` to import --
the four tests this file used to hold (all-sections, specific-section,
abstract-only, not-found) are already covered, byte-for-byte identical in
fixture and assertions, by `tests/test_core_papers.py`'s `paper_setup`
fixture and:

  - test_get_paper_content_all_sections_includes_abstract
  - test_get_paper_content_specific_section_excludes_abstract
  - test_get_paper_content_abstract_only
  - test_get_paper_content_not_found

(all exercising `paper_fetch_mcp.core.papers.get_paper_content`, the
function this old file's `_get_paper_content` was merged into). Per the
wave 5a task brief ("only drop a case if tests/test_core_papers.py already
covers it"), all four cases were dropped here rather than duplicated.
"""
