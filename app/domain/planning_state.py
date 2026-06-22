"""Domain models for planning state, replan decisions, and metrics."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from .path_models import TimedPosition, PathPlanResult
from .conflict_models import Conflict
from .task_models import RobotTask


class BatchStatus(str, Enum):
    RECEIVED = "received"
    PARSED = "parsed"
    VALIDATED = "validated"
    INITIAL_PLANNED = "initial_planned"
    CONFLICT_CHECKED = "conflict_checked"
    REPLANNING = "replanning"
    SUCCEEDED = "succeeded"
    PARTIALLY_SUCCEEDED = "partially_succeeded"
    INFEASIBLE = "infeasible"


@dataclass
class ReplanDecision:
    failure_category: str
    affected_robots: List[str] = field(default_factory=list)
    robot_to_replan: List[str] = field(default_factory=list)
    action: str = ""  # "fix_high_priority", "add_time_window", "adjust_priority"
    priority_changes: Dict[str, int] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""


@dataclass
class SingleTaskResult:
    robot_id: str
    task: Optional[RobotTask] = None
    path: List[TimedPosition] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None
    replanned: bool = False


@dataclass
class PlanningMetrics:
    total_task_count: int = 0
    planned_task_count: int = 0
    planning_failed_task_count: int = 0
    planning_success_rate: float = 0.0
    total_planning_time_ms: float = 0.0
    parsing_time_ms: float = 0.0
    initial_planning_time_ms: float = 0.0
    replanning_time_ms: float = 0.0
    average_planning_time_per_task_ms: float = 0.0
    initial_conflict_count: int = 0
    final_conflict_count: int = 0
    replanning_triggered: bool = False
    retry_count: int = 0
    astar_call_count: int = 0
    total_expanded_nodes: int = 0


@dataclass
class PlanningState:
    request_id: str
    original_instruction: str
    task_batch: Optional[Any] = None  # TaskBatch
    priority_order: List[str] = field(default_factory=list)
    initial_paths: Dict[str, PathPlanResult] = field(default_factory=dict)
    current_paths: Dict[str, PathPlanResult] = field(default_factory=dict)
    initial_conflicts: List[Conflict] = field(default_factory=list)
    current_conflicts: List[Conflict] = field(default_factory=list)
    retry_count: int = 0
    replan_history: List[ReplanDecision] = field(default_factory=list)
    status: BatchStatus = BatchStatus.RECEIVED
    failure_reason: Optional[str] = None
    total_planning_time_ms: float = 0.0
    metrics: Optional[PlanningMetrics] = None
    task_results: List[SingleTaskResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
