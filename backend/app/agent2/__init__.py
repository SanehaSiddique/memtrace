"""Agent2 — graph memory + JEV (docs/IMPLEMENTATION.md §6).

Same user message, same shared LLM/tool clients as Agent1, but three
deliberate deltas:

  1. long-term memory is a Neo4j fact graph with explicit staleness
     (`SUPERSEDED_BY` instead of contradictions piling up),
  2. JEV `Choice` decides which tool (if any) needs to be in the prompt at all,
  3. JEV `Score` filters the tool's raw payload per chunk before it is allowed
     to enter the context window.

Everything else — STM handling, final-answer prompting, fact extraction — is
identical to Agent1, which is what keeps the comparison fair.
"""
