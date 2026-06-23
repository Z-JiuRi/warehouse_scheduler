"""LangGraph builder: constructs and compiles the planning workflow graph.

Usage:
    from app.orchestration.graph_builder import build_graph

    graph = build_graph(
        parser=task_parser_agent,
        replanning_agent=replanning_agent,
        replanning_policy=replanning_policy,
    )
    compiled = graph.compile()
    result = compiled.invoke(initial_state)
"""

from langgraph.graph import StateGraph, START, END
from app.domain.graph_state import GraphState
from app.domain.planning_state import BatchStatus

from app.orchestration.graph_nodes import (
    make_parse_instruction,
    validate_and_resolve_goals,
    build_obstacles,
    initial_plan,
    conflict_check,
    make_replan_decide,
    make_apply_replan,
    partial_execution,
    validate_final,
    compute_metrics,
)

# ── Replanning loop limit ──────────────────────────────────────────────────
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional routing functions
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_conflict_check(state: GraphState) -> str:
    """Decide where to go after conflict_check.

    Returns one of: "replan", "partial", "validate", "infeasible"
    """
    current_status = state.get("status")

    # Hard infeasible → stop
    if current_status == BatchStatus.INFEASIBLE:
        return "infeasible"

    # Check for planning failures and conflicts
    current_paths = state.get("current_paths", {})
    current_conflicts = state.get("current_conflicts", [])
    retry_count = state.get("retry_count", 0)

    # Count successful paths
    failed = [rid for rid, rp in current_paths.items() if not rp.success]

    # No conflicts and all paths succeeded → validate
    if not current_conflicts and not failed:
        return "validate"

    # All paths failed → partial execution
    if failed and len(failed) == len(current_paths):
        return "partial"

    # Conflicts exist and we still have retries → replan
    if current_conflicts and retry_count < MAX_RETRIES:
        return "replan"

    # Conflicts exist but no retries left, or some failed → partial
    return "partial"


def route_after_partial_execution(state: GraphState) -> str:
    """Decide after partial_execution: validate or stop."""
    status = state.get("status")
    if status == BatchStatus.PARTIALLY_SUCCEEDED:
        return "validate"
    return "infeasible"


def route_after_validate(state: GraphState) -> str:
    """Decide after validate_final: compute metrics or stop."""
    status = state.get("status")
    if status in (BatchStatus.SUCCEEDED, BatchStatus.PARTIALLY_SUCCEEDED):
        return "compute"
    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(
    parser,                 # TaskParserAgent instance
    replanning_agent,       # ReplanningAgent instance
    replanning_policy=None, # ReplanningPolicy instance (optional)
):
    """Build and return a compiled LangGraph StateGraph.

    Args:
        parser: TaskParserAgent (provides .parse(instruction) → TaskBatch)
        replanning_agent: ReplanningAgent (provides .decide(...) → ReplanDecision)
        replanning_policy: ReplanningPolicy (provides .apply(...)); if None,
                           the apply_replan node uses inline logic.

    Returns:
        Compiled LangGraph graph ready for .invoke(state).
    """
    # Create node factories (bind external agents)
    include_parse = parser is not None
    replan_decide_node = make_replan_decide(replanning_agent)

    if replanning_policy is not None:
        apply_replan_node = make_apply_replan(replanning_policy)
    else:
        # Fallback: use inline apply_replan (already bound to nothing special)
        # The inline version works standalone
        apply_replan_node = make_apply_replan(None)

    # Build the graph
    builder = StateGraph(GraphState)

    # ── Add nodes ──────────────────────────────────────────────────────
    if include_parse:
        parse_node = make_parse_instruction(parser)
        builder.add_node("parse_instruction", parse_node)
    builder.add_node("validate_and_resolve_goals", validate_and_resolve_goals)
    builder.add_node("build_obstacles", build_obstacles)
    builder.add_node("initial_plan", initial_plan)
    builder.add_node("conflict_check", conflict_check)
    builder.add_node("replan_decide", replan_decide_node)
    builder.add_node("apply_replan", apply_replan_node)
    builder.add_node("partial_execution", partial_execution)
    builder.add_node("validate_final", validate_final)
    builder.add_node("compute_metrics", compute_metrics)

    # ── Add edges ──────────────────────────────────────────────────────
    # Happy path: start → parse (if included) → validate
    if include_parse:
        builder.add_edge(START, "parse_instruction")
        builder.add_edge("parse_instruction", "validate_and_resolve_goals")
    else:
        builder.add_edge(START, "validate_and_resolve_goals")
    builder.add_edge("validate_and_resolve_goals", "build_obstacles")
    builder.add_edge("build_obstacles", "initial_plan")
    builder.add_edge("initial_plan", "conflict_check")

    # Conditional branch after conflict_check:
    builder.add_conditional_edges(
        "conflict_check",
        route_after_conflict_check,
        {
            "validate": "validate_final",
            "replan": "replan_decide",
            "partial": "partial_execution",
            "infeasible": "compute_metrics",
        },
    )

    # Replanning loop: replan_decide → apply_replan → conflict_check
    builder.add_edge("replan_decide", "apply_replan")
    builder.add_edge("apply_replan", "conflict_check")

    # Partial execution branch
    builder.add_conditional_edges(
        "partial_execution",
        route_after_partial_execution,
        {
            "validate": "validate_final",
            "infeasible": "compute_metrics",
        },
    )

    # Final validation
    builder.add_conditional_edges(
        "validate_final",
        route_after_validate,
        {
            "compute": "compute_metrics",
            "end": "compute_metrics",
        },
    )

    # Metrics → END
    builder.add_edge("compute_metrics", END)

    # ── Compile ────────────────────────────────────────────────────────
    return builder.compile()
