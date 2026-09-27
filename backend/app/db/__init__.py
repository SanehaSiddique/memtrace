"""Databases for the comparison demo (docs/IMPLEMENTATION.md §3, §5.1, §6.1).

Real Postgres for Agent1's flat facts table, real Neo4j for Agent2's memory
graph. Both modules expose the same shape: a real store, an explicitly-labeled
in-process fallback for when the database is not provisioned, and an
``open_*`` factory that reports which one you got.
"""
