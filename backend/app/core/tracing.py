"""LangSmith run tagging for the comparison demo (docs/IMPLEMENTATION.md §4.4).

One project, one run *tree per agent per turn*, joined by a shared
``run_group_id`` (a UUID generated once per user message and passed to both
agents) plus an ``agent_id`` tag so the two agents' runs can be filtered and
compared in the LangSmith UI. Taggable by call site to satisfy §4.2's
per-call-site JEV breakdown ("tool_routing" | "result_filtering" |
"staleness_check").

Every LLM call, JEV call, and tool call goes through `child_trace` — the doc is
explicit that silent/untraced calls are not allowed, because the metrics
dashboard would then have gaps. When tracing is disabled (no key), langsmith's
run tree becomes a local no-op and nothing is uploaded; the existing
`configure_langsmith()` handles that switch.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from langsmith.run_helpers import get_current_run_tree
from langsmith.run_helpers import trace as _ls_trace


def turn_tags(run_group_id: str, agent_id: str) -> List[str]:
    return [f"run_group_id:{run_group_id}", f"agent_id:{agent_id}"]


@contextmanager
def agent_turn_trace(
    agent_id: str,
    run_group_id: str,
    session_id: str,
    model: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Wrap one agent's whole turn in a single traced chain run.

    Both agents get the *same* `run_group_id`, which is how §4.4 joins them.
    """
    metadata: Dict[str, Any] = {"run_group_id": run_group_id, "agent_id": agent_id, "session_id": session_id}
    if model:
        metadata["model"] = model
    if extra_metadata:
        metadata.update(extra_metadata)
    with _ls_trace(
        name=f"{agent_id}.turn",
        run_type="chain",
        tags=turn_tags(run_group_id, agent_id),
        metadata=metadata,
    ) as run:
        yield run


@contextmanager
def child_trace(
    name: str,
    run_type: str,
    agent_id: str,
    run_group_id: str,
    component: str,
    call_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """One traced child run for a single LLM / JEV / tool call.

    `component` is "llm" | "jev" | "tool" | "memory"; `call_type` refines it
    ("reasoning" | "tool_selection" | "final_answer" | "fact_extraction" for LLM
    calls, and the JEV call site for JEV calls).
    """
    tags = turn_tags(run_group_id, agent_id) + [f"component:{component}"]
    if call_type:
        tags.append(f"call_type:{call_type}")
    payload: Dict[str, Any] = {"run_group_id": run_group_id, "agent_id": agent_id, "component": component}
    if call_type:
        payload["call_type"] = call_type
    if metadata:
        payload.update(metadata)
    with _ls_trace(name=name, run_type=run_type, tags=tags, metadata=payload) as run:
        yield run


def current_run_id() -> Optional[str]:
    """The id of the run currently being traced (for the engineering view)."""
    run_tree = get_current_run_tree()
    return str(run_tree.id) if run_tree is not None else None
