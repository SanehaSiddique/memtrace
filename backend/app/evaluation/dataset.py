"""Evaluation dataset: 10 cases against the Project Alpha demo story
(app/demo/seed.py), one per required category. Each case's expectations are
substring checks against the synthesized answer — simple, but real: nothing
here is a hand-picked "the model said so" score."""

from typing import List

from pydantic import BaseModel


class EvalCase(BaseModel):
    name: str
    category: str
    query: str
    expect_contains: List[str] = []
    expect_excludes: List[str] = []


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        name="direct_current_fact",
        category="direct_current_fact",
        query="What database are we currently using?",
        expect_contains=["PostgreSQL"],
        expect_excludes=["MongoDB"],
    ),
    EvalCase(
        name="historical_fact",
        category="historical_fact",
        query="What database did we use before PostgreSQL?",
        expect_contains=["MongoDB"],
    ),
    EvalCase(
        name="contradictory_fact",
        category="contradictory_fact",
        query="Did the team ever consider MySQL?",
        expect_contains=["MySQL"],
    ),
    EvalCase(
        name="multi_hop_relationship",
        category="multi_hop_relationship",
        query="What replaced MongoDB?",
        expect_contains=["PostgreSQL"],
    ),
    EvalCase(
        name="temporal_update",
        category="temporal_update",
        query="What deployment platform are we using now?",
        expect_contains=["Railway"],
    ),
    EvalCase(
        name="irrelevant_memory",
        category="irrelevant_memory",
        query="What's the weather like today?",
        expect_contains=[],
        expect_excludes=["PostgreSQL", "MongoDB", "Railway", "JWT"],
    ),
    EvalCase(
        name="stale_memory",
        category="stale_memory",
        query="Are we still deployed on AWS?",
        expect_contains=["Railway"],
        expect_excludes=["AWS"],
    ),
    EvalCase(
        name="decision_and_reason",
        category="decision_and_reason",
        query="Why did we move to PostgreSQL?",
        expect_contains=["PostgreSQL", "relational querying"],
    ),
    EvalCase(
        name="entity_relationship",
        category="entity_relationship",
        query="What does Project Alpha depend on?",
        expect_contains=["Redis"],
    ),
    EvalCase(
        name="memory_conflict",
        category="memory_conflict",
        query="What authentication method are we using?",
        expect_contains=["JWT"],
        expect_excludes=["session"],
    ),
]
