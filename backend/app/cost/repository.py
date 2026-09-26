"""Persists RunRecord rows — one per answered query — and MemoryCostImpact
rows — one per individual memory decision's estimated dollar effect — so the
executive dashboard has real recorded activity to aggregate instead of
hardcoded numbers."""

from datetime import datetime
from typing import List, Optional

import aiosqlite

from app.cost.models import MemoryCostImpact, RunRecord

_COLUMNS = """
    id, agent_id, conversation_id, query, model, baseline_tokens, optimized_tokens,
    output_tokens, baseline_cost, optimized_cost, savings, leak_category,
    langsmith_run_id, created_at
"""

_IMPACT_COLUMNS = "id, agent_id, memory_id, operation, run_id, cost_avoided, reason, created_at"


class SQLiteCostRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS run_costs (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    model TEXT NOT NULL,
                    baseline_tokens INTEGER NOT NULL,
                    optimized_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    baseline_cost REAL NOT NULL,
                    optimized_cost REAL NOT NULL,
                    savings REAL NOT NULL,
                    leak_category TEXT NOT NULL,
                    langsmith_run_id TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_run_costs_agent ON run_costs(agent_id);")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_cost_impacts (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    run_id TEXT,
                    cost_avoided REAL NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memory_impacts_agent ON memory_cost_impacts(agent_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_memory_impacts_memory ON memory_cost_impacts(memory_id);")
            await db.commit()

    def _row_to_record(self, row: tuple) -> RunRecord:
        return RunRecord(
            id=row[0],
            agent_id=row[1],
            conversation_id=row[2],
            query=row[3],
            model=row[4],
            baseline_tokens=row[5],
            optimized_tokens=row[6],
            output_tokens=row[7],
            baseline_cost=row[8],
            optimized_cost=row[9],
            savings=row[10],
            leak_category=row[11],
            langsmith_run_id=row[12],
            created_at=datetime.fromisoformat(row[13]),
        )

    async def record(self, run: RunRecord) -> RunRecord:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO run_costs ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.agent_id,
                    run.conversation_id,
                    run.query,
                    run.model,
                    run.baseline_tokens,
                    run.optimized_tokens,
                    run.output_tokens,
                    run.baseline_cost,
                    run.optimized_cost,
                    run.savings,
                    run.leak_category,
                    run.langsmith_run_id,
                    run.created_at.isoformat(),
                ),
            )
            await db.commit()
        return run

    async def list_all(self, agent_id: str, limit: int = 100000) -> List[RunRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {_COLUMNS} FROM run_costs WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?",
                (agent_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_record(r) for r in rows]

    async def list_recent(self, agent_id: str, limit: int = 20) -> List[RunRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {_COLUMNS} FROM run_costs WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
                (agent_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_record(r) for r in rows]

    # -- memory cost impacts --------------------------------------------------

    def _row_to_impact(self, row: tuple) -> MemoryCostImpact:
        return MemoryCostImpact(
            id=row[0],
            agent_id=row[1],
            memory_id=row[2],
            operation=row[3],
            run_id=row[4],
            cost_avoided=row[5],
            reason=row[6],
            created_at=datetime.fromisoformat(row[7]),
        )

    async def record_impact(self, impact: MemoryCostImpact) -> MemoryCostImpact:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT INTO memory_cost_impacts ({_IMPACT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    impact.id,
                    impact.agent_id,
                    impact.memory_id,
                    impact.operation,
                    impact.run_id,
                    impact.cost_avoided,
                    impact.reason,
                    impact.created_at.isoformat(),
                ),
            )
            await db.commit()
        return impact

    async def list_impacts(self, agent_id: str, limit: int = 100000) -> List[MemoryCostImpact]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {_IMPACT_COLUMNS} FROM memory_cost_impacts WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?",
                (agent_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_impact(r) for r in rows]

    async def list_impacts_for_memory(self, memory_id: str) -> List[MemoryCostImpact]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {_IMPACT_COLUMNS} FROM memory_cost_impacts WHERE memory_id = ? ORDER BY created_at ASC",
                (memory_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_impact(r) for r in rows]
