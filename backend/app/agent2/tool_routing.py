"""Agent2 JEV Tool Routing (docs/IMPLEMENTATION.md §6.3).

Instead of injecting every registered tool schema into every turn's reasoning
prompt (the naive Agent1 baseline), Agent2 asks Jev to route the query:
  * "none" (high confidence) -> zero tool schemas injected (huge token savings!).
  * a specific tool -> only that tool's single schema is injected.
  * low confidence / JEV error -> safe fallback to full tool registry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.graph8_client import Graph8MCPClient
from app.core.jev_client import JEVClient
from app.tracing.langsmith import traced

# §6.3 recommended threshold: below this, fall back to giving LLM all tools
# as a safety net so an uncertain JEV call cannot silently break tool access.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

_TOOL_DESCRIPTIONS = {
    "g8_search_contacts": "Search for people, leads, or contacts by name, job title, email, or company.",
    "g8_search_companies": "Search for companies, organizations, or accounts by name, industry, or domain.",
    "g8_lookup_company": "Fetch full detailed profile of a specific company by exact company ID or domain.",
    "none": "No CRM tool needed. The query is conversational, asks about prior session facts, or needs general reasoning.",
}


@dataclass
class ToolRoutingDecision:
    """The outcome of JEV tool routing for this turn."""

    routed_choice: str  # tool name or "none" or "all" (fallback)
    confidence: float = 0.0
    schemas_to_inject: List[Dict[str, Any]] = field(default_factory=list)
    tools_considered: int = 0
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0
    notes: List[str] = field(default_factory=list)


@traced(name="agent2.jev_route_tools")
async def route_tools(
    jev: JEVClient,
    tools: Graph8MCPClient,
    user_message: str,
    recent_stm: List[dict],
    run_group_id: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ToolRoutingDecision:
    """Run JEV Choice to select zero, one, or fallback tool schemas (§6.3)."""
    all_schemas = tools.tool_schemas()
    available_tools = [s["function"]["name"] for s in all_schemas]
    options = list(available_tools) + ["none"]
    criteria = {opt: _TOOL_DESCRIPTIONS.get(opt, opt) for opt in options}

    # Context given to Jev: the current message plus the last STM turn if any
    context = {
        "user_message": user_message,
        "recent_conversation": [f"{m.get('role')}: {m.get('content')}" for m in recent_stm[-2:]],
    }

    result = await jev.jev_choice(
        question="Which tool, if any, is needed to answer this user query?",
        options=options,
        context=context,
        call_site="tool_routing",
        option_criteria=criteria,
        instructions=(
            "Evaluate whether the query requires looking up live CRM contacts or companies, "
            "or if it is conversational/memory-based. Choose the single most appropriate tool, or 'none'."
        ),
        agent_id="agent2",
        run_group_id=run_group_id,
    )

    # Fallback branch 1: JEV endpoint failed (e.g. gateway 403 or network error)
    if not result.ok:
        return ToolRoutingDecision(
            routed_choice="all",
            confidence=0.0,
            schemas_to_inject=all_schemas,
            tools_considered=len(all_schemas),
            used_fallback=True,
            fallback_reason=f"jev_unavailable: {result.error}",
            latency_ms=result.latency_ms,
            notes=[f"jev_routing_fallback_due_to_error: {result.error}"],
        )

    # Fallback branch 2: JEV answered, but confidence is below safety net threshold
    if result.confidence < confidence_threshold:
        return ToolRoutingDecision(
            routed_choice=result.choice,
            confidence=result.confidence,
            schemas_to_inject=all_schemas,
            tools_considered=len(all_schemas),
            used_fallback=True,
            fallback_reason=f"low_confidence ({result.confidence:.2f} < {confidence_threshold:.2f})",
            latency_ms=result.latency_ms,
            notes=[f"jev_routing_fallback_low_confidence: {result.choice} @ {result.confidence:.2f}"],
        )

    # High-confidence JEV decisions:
    if result.choice == "none":
        # Zero schemas in prompt! Fixed per-tool token overhead eliminated.
        return ToolRoutingDecision(
            routed_choice="none",
            confidence=result.confidence,
            schemas_to_inject=[],
            tools_considered=0,
            used_fallback=False,
            latency_ms=result.latency_ms,
            notes=[f"jev_routed_to_none: confidence={result.confidence:.2f}"],
        )

    # Single targeted tool schema
    targeted = [s for s in all_schemas if s["function"]["name"] == result.choice]
    if targeted:
        return ToolRoutingDecision(
            routed_choice=result.choice,
            confidence=result.confidence,
            schemas_to_inject=targeted,
            tools_considered=1,
            used_fallback=False,
            latency_ms=result.latency_ms,
            notes=[f"jev_routed_to_single_tool: {result.choice} @ {result.confidence:.2f}"],
        )

    # Fallback if choice returned wasn't in schema list
    return ToolRoutingDecision(
        routed_choice="all",
        confidence=result.confidence,
        schemas_to_inject=all_schemas,
        tools_considered=len(all_schemas),
        used_fallback=True,
        fallback_reason=f"unknown_choice_{result.choice}",
        latency_ms=result.latency_ms,
        notes=[f"jev_routing_unknown_choice: {result.choice}"],
    )
