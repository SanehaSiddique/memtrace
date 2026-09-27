"""Shared core layer for the agent-comparison demo (docs/IMPLEMENTATION.md §4).

Agent1 (naive baseline) and Agent2 (graph-memory + JEV) differ only in what
they store as long-term memory, how they choose a tool, and what they do with a
tool's raw result before it re-enters context. Everything else — the LLM
client, the JEV client, the graph8 MCP client, tracing, and cost math — lives
here once and is injected into both agents, which is what makes the comparison
fair and auditable.
"""
