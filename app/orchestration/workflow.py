"""Workflow: top-level orchestration powered by a compiled LangGraph graph.

The Workflow class now wraps a compiled StateGraph instead of directly
invoking SchedulerAgent. The external API (run / run_structured → PlanningState)
remains backward-compatible.
"""

import json
import os
import uuid
import time
from typing import Optional

from app.domain.graph_state import GraphState
from app.domain.map_models import WarehouseMap
from app.domain.task_models import RobotTask, TaskBatch
from app.domain.planning_state import (
    PlanningState,
    BatchStatus,
    ReplanDecision,
    SingleTaskResult,
    PlanningMetrics,
)
from app.domain.runtime_models import DynamicBlockage
from app.services.map_loader import MapLoader
from app.services.robot_registry import RobotRegistry
from app.services.location_resolver import LocationResolver
from app.agents.task_parser_agent import TaskParserAgent
from app.agents.replanning_agent import ReplanningAgent
from app.orchestration.replanning_policy import ReplanningPolicy
from app.orchestration.graph_builder import build_graph


class Workflow:
    """End-to-end planning workflow, now backed by a LangGraph StateGraph."""

    def __init__(
        self,
        map_path: str = None,
        runtime_path: str = None,
        api_config_path: str = None,
        max_timestep: int = 200,
    ):
        # Resolve default paths
        if map_path is None:
            map_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs",
                "warehouse_map.json",
            )
        if runtime_path is None:
            runtime_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs",
                "warehouse_runtime.json",
            )

        self.map_path = map_path
        self.runtime_path = runtime_path
        self.api_config_path = api_config_path
        self.max_timestep = max_timestep

        # Load map
        self.map_loader = MapLoader(map_path)
        self.warehouse_map, self.map_errors = self.map_loader.load()

        # Load runtime
        self.robot_registry = RobotRegistry(runtime_path)
        self.runtime_errors = self.robot_registry.load()

        # Agents & policy (created lazily if map available)
        self.parser = None
        self.replanning_agent = None
        self.replanning_policy = None
        self._compiled_graph = None
        self._compiled_graph_no_parse = None

        if self.warehouse_map is not None:
            self.parser = TaskParserAgent(
                self.warehouse_map,
                self.robot_registry,
                api_config_path=api_config_path,
            )
            self.replanning_agent = ReplanningAgent()
            self.replanning_policy = ReplanningPolicy(max_retries=3)

            # Build the full graph (with parsing)
            self._compiled_graph = build_graph(
                parser=self.parser,
                replanning_agent=self.replanning_agent,
                replanning_policy=self.replanning_policy,
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, instruction: str) -> PlanningState:
        """Execute the full workflow for a natural language instruction.

        Flow:  instruction  →  compiled LangGraph  →  PlanningState
        """
        t0 = time.time()
        request_id = str(uuid.uuid4())[:8]

        # Pre-validation: map/runtime errors
        if self.warehouse_map is None:
            return PlanningState(
                request_id=request_id,
                original_instruction=instruction,
                status=BatchStatus.INFEASIBLE,
                failure_reason="Map loading failed",
                errors=list(self.map_errors),
            )

        if self._compiled_graph is None:
            return PlanningState(
                request_id=request_id,
                original_instruction=instruction,
                status=BatchStatus.INFEASIBLE,
                failure_reason="Graph not compiled",
            )

        # Build initial GraphState
        blockages = list(self.robot_registry.get_blockages())
        initial_state: GraphState = {
            "request_id": request_id,
            "original_instruction": instruction,
            "blockages": blockages,
            "max_timestep": self.max_timestep,
            "warehouse_map": self.warehouse_map,
            # Accumulating fields — start empty
            "replan_history": [],
            "task_results": [],
            "warnings": [],
            "errors": [],
        }

        # Run the graph
        result = self._compiled_graph.invoke(initial_state)
        total_time_ms = (time.time() - t0) * 1000

        # Convert graph result dict → PlanningState
        return self._result_to_planning_state(result, total_time_ms)

    def run_structured(self, tasks_json: dict) -> PlanningState:
        """Run with pre-structured tasks (bypass LLM parsing).

        Builds a TaskBatch from the JSON, then starts the graph from
        the validate step (skipping parse_instruction).
        """
        t0 = time.time()
        request_id = str(uuid.uuid4())[:8]

        if self.warehouse_map is None:
            return PlanningState(
                request_id=request_id,
                original_instruction="",
                status=BatchStatus.INFEASIBLE,
                failure_reason="Map loading failed",
            )

        # Build TaskBatch from JSON
        task_batch = TaskBatch()
        for t_raw in tasks_json.get("tasks", []):
            task = RobotTask(
                robot_id=t_raw["robot_id"],
                start=tuple(t_raw["start"]),
                goal_location_id=t_raw["goal_location_id"],
                priority=t_raw.get("priority", 1),
            )
            resolver = LocationResolver(self.warehouse_map)
            loc = resolver.resolve(task.goal_location_id)
            if loc:
                task.candidate_goals = list(loc.entry_cells)
                task.selected_goal = loc.entry_cells[0]
            task_batch.tasks.append(task)

        # Collect blockages
        blockages = list(self.robot_registry.get_blockages())
        for c_raw in tasks_json.get("runtime_constraints", []):
            if c_raw.get("constraint_type") == "closed_corridor":
                corridor = self.warehouse_map.find_corridor(
                    c_raw.get("target_id", "")
                )
                if corridor:
                    blockages.append(
                        DynamicBlockage(
                            blockage_id=f"closed_{c_raw.get('target_id')}",
                            target_type="corridor",
                            target_id=c_raw.get("target_id"),
                            cells=list(corridor.cells),
                            start_time=c_raw.get("start_time", 0),
                            end_time=c_raw.get("end_time"),
                            source="user_instruction",
                        )
                    )

        # For structured mode, we have two options:
        # 1) Use a separate graph that starts at validate_and_resolve_goals
        # 2) Invoke the full graph with a pre-built task_batch — the
        #    parse_instruction node will be a no-op when task_batch exists.
        #
        # We go with option 2 for simplicity: the parse_instruction node
        # built by make_parse_instruction always runs LLM parsing, so we
        # use a "skip parse" graph. Build it lazily.

        if self._compiled_graph_no_parse is None:
            self._compiled_graph_no_parse = build_graph(
                parser=None,  # Signal: no parser → skip parse node
                replanning_agent=self.replanning_agent,
                replanning_policy=self.replanning_policy,
            )

        initial_state: GraphState = {
            "request_id": request_id,
            "original_instruction": "",
            "task_batch": task_batch,
            "blockages": blockages,
            "max_timestep": self.max_timestep,
            "warehouse_map": self.warehouse_map,
            "replan_history": [],
            "task_results": [],
            "warnings": [],
            "errors": [],
        }

        result = self._compiled_graph_no_parse.invoke(initial_state)
        total_time_ms = (time.time() - t0) * 1000

        return self._result_to_planning_state(result, total_time_ms)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _result_to_planning_state(
        self,
        result: dict,
        total_time_ms: float,
    ) -> PlanningState:
        """Convert a GraphState dict (from graph invocation) back to a
        PlanningState dataclass for backward compatibility.
        """
        metrics = result.get("metrics")
        if metrics is not None:
            # Propagate the workflow-level total time into the metrics object
            metrics.total_planning_time_ms = total_time_ms

        state = PlanningState(
            request_id=result.get("request_id", ""),
            original_instruction=result.get("original_instruction", ""),
            task_batch=result.get("task_batch"),
            priority_order=result.get("priority_order", []),
            initial_paths=result.get("initial_paths", {}),
            current_paths=result.get("current_paths", {}),
            initial_conflicts=result.get("initial_conflicts", []),
            current_conflicts=result.get("current_conflicts", []),
            retry_count=result.get("retry_count", 0),
            replan_history=result.get("replan_history", []),
            status=result.get("status", BatchStatus.RECEIVED),
            failure_reason=result.get("failure_reason"),
            total_planning_time_ms=total_time_ms,
            metrics=metrics,
            task_results=result.get("task_results", []),
            warnings=result.get("warnings", []),
            errors=result.get("errors", []),
        )
        return state
