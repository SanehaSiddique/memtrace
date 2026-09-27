"""Agent1 — the naive baseline (docs/IMPLEMENTATION.md §5).

The same user message, the same shared LLM/tool clients, but the deliberately
unsophisticated choices: a flat Postgres facts table with no dedup or staleness
logic, every registered tool schema in every prompt, and the tool's raw JSON
pasted verbatim into the next context window. It is the control group, not a
straw man: it is exactly what a straightforward tool-using agent does.
"""
