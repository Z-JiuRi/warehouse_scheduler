"""Workflow: top-level orchestration tying together agents, tools, and services."""

import json
import os
from typing import Tuple, List, Optional
from app.domain.map_models import WarehouseMap
from app.domain.task_models import TaskBatch
from app.domain.planning_state import PlanningState, BatchStatus
from app.domain.runtime_models import DynamicBlockage
from app.services.map_loader import MapLoader
from app.services.robot_registry import RobotRegistry
from app.services.metrics_collector import MetricsCollector
from app.agents.task_parser_agent import TaskParserAgent
from app.agents.scheduler_agent import SchedulerAgent


class Workflow:
    """End-to-end planning workflow."""

    def __init__(
        self,
        map_path: str = None,
        runtime_path: str = None,
        api_config_path: str = None,
        max_timestep: int = 200,
    ):
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

        self.metrics = MetricsCollector()
        self.scheduler = None
        self.parser = None

        if self.warehouse_map is not None:
            self.scheduler = SchedulerAgent(
                self.warehouse_map,
                self.metrics,
                max_timestep=self.max_timestep,
            )
            self.parser = TaskParserAgent(
                self.warehouse_map,
                self.robot_registry,
                api_config_path=api_config_path,
            )

    def run(self, instruction: str) -> PlanningState:
        """Execute the full workflow for a natural language instruction."""
        errors = []
        if self.map_errors:
            errors.extend(self.map_errors)
        if self.runtime_errors:
            errors.extend(self.runtime_errors)

        if self.warehouse_map is None:
            state = PlanningState(
                request_id="error",
                original_instruction=instruction,
                status=BatchStatus.INFEASIBLE,
                failure_reason="Map loading failed",
                errors=errors,
            )
            return state

        # Parse instruction
        self.metrics.start_parsing()
        task_batch = self.parser.parse(instruction)
        self.metrics.end_parsing()

        if not task_batch.is_valid:
            state = PlanningState(
                request_id="parse_error",
                original_instruction=instruction,
                task_batch=task_batch,
                status=BatchStatus.INFEASIBLE,
                failure_reason="Parsing failed",
                errors=task_batch.parse_errors,
            )
            state.metrics = self.metrics.build_metrics(0, 0, 0)
            return state

        # Get dynamic blockages
        blockages = list(self.robot_registry.get_blockages())

        # Resolve corridor blockages from constraints
        for constraint in task_batch.runtime_constraints:
            if constraint.get("constraint_type") == "closed_corridor":
                target_id = constraint.get("target_id", "")
                corridor = self.warehouse_map.find_corridor(target_id)
                if corridor:
                    blockages.append(
                        DynamicBlockage(
                            blockage_id=f"user_closed_{target_id}",
                            target_type="corridor",
                            target_id=target_id,
                            cells=list(corridor.cells),
                            start_time=constraint.get("start_time", 0),
                            end_time=constraint.get("end_time"),
                            reason=constraint.get("reason", "User instruction"),
                            source="user_instruction",
                        )
                    )

        # Run scheduler
        state = self.scheduler.run(instruction, task_batch, blockages)
        return state

    def run_structured(self, tasks_json: dict) -> PlanningState:
        """Run with pre-structured tasks (bypass LLM parsing)."""
        from app.domain.task_models import RobotTask

        if self.warehouse_map is None:
            return PlanningState(
                request_id="error",
                original_instruction="",
                status=BatchStatus.INFEASIBLE,
                failure_reason="Map loading failed",
            )

        task_batch = TaskBatch()
        for t_raw in tasks_json.get("tasks", []):
            task = RobotTask(
                robot_id=t_raw["robot_id"],
                start=tuple(t_raw["start"]),
                goal_location_id=t_raw["goal_location_id"],
                priority=t_raw.get("priority", 1),
            )
            # Resolve goal
            from app.services.location_resolver import LocationResolver
            resolver = LocationResolver(self.warehouse_map)
            loc = resolver.resolve(task.goal_location_id)
            if loc:
                task.candidate_goals = list(loc.entry_cells)
                task.selected_goal = loc.entry_cells[0]
            task_batch.tasks.append(task)

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

        self.metrics.start()
        state = self.scheduler.run(
            task_batch.original_instruction
            if hasattr(task_batch, "original_instruction")
            else "",
            task_batch,
            blockages,
        )
        return state
