"""LangGraph-compatible state schema for the planning workflow.

This TypedDict defines the state that flows through every graph node.
Fields annotated with operator.add are APPEND-ONLY (reducer = add);
all other fields use the default overwrite (operator.setitem).
"""

from typing import TypedDict, Annotated, Optional, Any
import operator

from .planning_state import (
    BatchStatus,
    ReplanDecision,
    SingleTaskResult,
    PlanningMetrics,
)
from .conflict_models import Conflict
from .path_models import PathPlanResult
from .task_models import TaskBatch


class GraphState(TypedDict, total=False):
    """The unified state dictionary that flows through the LangGraph pipeline.

    Accumulating fields (Annotated with operator.add):
      - replan_history, warnings, errors, task_results
    Overwrite fields (default reducer):
      - everything else
    """

    # ── Input / identification ────────────────────────────────────────────
    request_id: str
    original_instruction: str

    # ── Task & goal resolution ────────────────────────────────────────────
    task_batch: Optional[TaskBatch]
    priority_order: list[str]

    # ── Obstacle sets (built from map + blockages) ────────────────────────
    static_obstacles: set[tuple[int, int]]         # (x, y) static obstacles
    dynamic_obstacles: set[tuple[int, int, int]]   # (x, y, t) dynamic obstacles
    blockages: list[Any]                           # List[DynamicBlockage]

    # ── Graph plumbing ────────────────────────────────────────────────────
    max_timestep: int
    warehouse_map: Optional[Any]                   # WarehouseMap

    # ── Initial planning results ──────────────────────────────────────────
    initial_paths: dict[str, PathPlanResult]

    # ── Current working-set paths (mutated by replanning) ─────────────────
    current_paths: dict[str, PathPlanResult]

    # ── Conflict state ────────────────────────────────────────────────────
    initial_conflicts: list[Conflict]
    current_conflicts: list[Conflict]

    # ── Replanning loop control ───────────────────────────────────────────
    retry_count: int
    replan_history: Annotated[list[ReplanDecision], operator.add]

    # ── Outcome ───────────────────────────────────────────────────────────
    status: BatchStatus
    failure_reason: Optional[str]

    # ── Metrics (set once by the compute_metrics node) ────────────────────
    total_planning_time_ms: float
    metrics: Optional[PlanningMetrics]

    # ── Per-task results (accumulated) ────────────────────────────────────
    task_results: Annotated[list[SingleTaskResult], operator.add]

    # ── Human-readable diagnostics (accumulated) ──────────────────────────
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
