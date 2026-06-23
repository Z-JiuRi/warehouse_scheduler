"""Scheduler agent: orchestrates the end-to-end planning workflow.

DEPRECATED as of LangGraph migration (2025-06).
This module has been replaced by:
  - app/orchestration/graph_nodes.py  (individual node functions)
  - app/orchestration/graph_builder.py (StateGraph construction)
  - app/orchestration/workflow.py     (LangGraph-powered Workflow class)

Kept for reference; no active code imports it.
"""

import time
import uuid
from typing import Dict, List, Tuple, Optional, Set
from app.domain.task_models import RobotTask, TaskBatch
from app.domain.path_models import PathPlanResult, TimedPosition
from app.domain.conflict_models import Conflict
from app.domain.planning_state import (
    PlanningState,
    BatchStatus,
    ReplanDecision,
    SingleTaskResult,
)
from app.domain.map_models import WarehouseMap
from app.domain.runtime_models import DynamicBlockage
from app.services.metrics_collector import MetricsCollector
from app.services.location_resolver import LocationResolver
from app.tools.astar_planner import AStarPlanner
from app.tools.reservation_table import ReservationTable
from app.tools.conflict_detector import ConflictDetector
from app.tools.path_validator import PathValidator
from app.agents.replanning_agent import ReplanningAgent
from app.orchestration.replanning_policy import ReplanningPolicy


class SchedulerAgent:
    """Main scheduling orchestrator."""

    def __init__(
        self,
        warehouse_map: WarehouseMap,
        metrics: MetricsCollector,
        max_timestep: int = 200,
    ):
        self.map = warehouse_map
        self.metrics = metrics
        self.max_timestep = max_timestep
        self.location_resolver = LocationResolver(warehouse_map)
        self.validator = PathValidator(warehouse_map)
        self.conflict_detector = ConflictDetector()
        self.replanning_agent = ReplanningAgent()
        self.replanning_policy = ReplanningPolicy(max_retries=3)

    def run(
        self,
        instruction: str,
        task_batch: TaskBatch,
        dynamic_blockages: List[DynamicBlockage] = None,
    ) -> PlanningState:
        """Execute the full planning workflow."""
        request_id = str(uuid.uuid4())[:8]
        state = PlanningState(
            request_id=request_id,
            original_instruction=instruction,
            task_batch=task_batch,
            status=BatchStatus.PARSED,
        )

        self.metrics.start()

        # Validate
        state.status = BatchStatus.VALIDATED
        if not task_batch.is_valid:
            state.status = BatchStatus.INFEASIBLE
            state.failure_reason = "Task batch has parse errors"
            state.errors = task_batch.parse_errors
            return self._finalize(state)

        # Resolve goals for each task
        occupied_goals: Set[Tuple[int, int]] = set()
        for task in task_batch.tasks:
            goal = self.location_resolver.select_best_goal(
                task.goal_location_id,
                task.start,
                occupied_goals,
            )
            if goal is None:
                state.errors.append(
                    f"Robot {task.robot_id}: no free entry for {task.goal_location_id}"
                )
                continue
            task.selected_goal = goal
            occupied_goals.add(goal)

        if state.errors:
            state.status = BatchStatus.INFEASIBLE
            state.failure_reason = "Some tasks have no valid goal"
            return self._finalize(state)

        # Set priority order
        task_batch.tasks.sort(key=lambda t: t.priority)
        state.priority_order = [t.robot_id for t in task_batch.tasks]

        # Build obstacle set
        static_obs = self._build_static_obstacle_set()
        dynamic_obs = self._build_dynamic_obstacle_set(dynamic_blockages or [])

        # Initial independent planning
        self.metrics.start_initial_planning()
        state.initial_paths = self._initial_plan(
            task_batch, static_obs, dynamic_obs
        )
        self.metrics.end_initial_planning()
        state.current_paths = dict(state.initial_paths)
        state.status = BatchStatus.INITIAL_PLANNED

        # Check for initial planning failures
        failed_initial = [
            rid
            for rid, rp in state.initial_paths.items()
            if not rp.success
        ]
        if failed_initial:
            state.errors.extend(
                f"Robot {rid}: initial planning failed ({state.initial_paths[rid].failure_reason.value})"
                for rid in failed_initial
            )

        # If ALL initial planning failed, go directly to infeasible
        if failed_initial and len(failed_initial) == len(state.initial_paths):
            state.status = BatchStatus.INFEASIBLE
            state.failure_reason = "All tasks failed initial planning"
            return self._finalize(state)

        # Conflict detection on initial paths
        initial_conflicts = self.conflict_detector.detect(state.initial_paths)
        state.initial_conflicts = initial_conflicts
        state.current_conflicts = list(initial_conflicts)
        self.metrics.set_initial_conflicts(len(initial_conflicts))
        state.status = BatchStatus.CONFLICT_CHECKED

        if initial_conflicts:
            # Enter replanning loop
            state = self._replan_loop(state, static_obs, dynamic_obs)
        else:
            # No conflicts. Check if any paths failed.
            failed_now = [
                rid for rid, rp in state.current_paths.items() if not rp.success
            ]
            if failed_now:
                state.status = BatchStatus.PARTIALLY_SUCCEEDED
                state.failure_reason = (
                    f"Some tasks failed initial planning: {failed_now}"
                )
            else:
                validation_errors = self.validator.validate_multi_robot(
                    state.current_paths
                )
                if validation_errors:
                    state.errors.extend(validation_errors)
                    state.status = BatchStatus.INFEASIBLE
                    state.failure_reason = "Validation failed"
                else:
                    state.status = BatchStatus.SUCCEEDED

        # If still has conflicts or failures after retries, try partial execution
        if state.status not in (
            BatchStatus.SUCCEEDED,
            BatchStatus.PARTIALLY_SUCCEEDED,
            BatchStatus.INFEASIBLE,
        ):
            state = self._partial_execution(state, static_obs, dynamic_obs)

        # Validate final result
        if state.status == BatchStatus.SUCCEEDED:
            validation_errors = self.validator.validate_multi_robot(
                state.current_paths
            )
            if validation_errors:
                state.status = BatchStatus.INFEASIBLE
                state.failure_reason = (
                    "Final validation failed: " + "; ".join(validation_errors)
                )

        return self._finalize(state)

    def _initial_plan(
        self,
        task_batch: TaskBatch,
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int, int]],
    ) -> Dict[str, PathPlanResult]:
        """Run independent A* for each task (no inter-robot reservations)."""
        results = {}
        for task in task_batch.tasks:
            planner = AStarPlanner(
                width=self.map.width,
                height=self.map.height,
                static_obstacles=static_obs,
                max_timestep=self.max_timestep,
                move_cost=self.map.movement.move_cost,
                wait_cost=self.map.movement.wait_cost,
            )
            planner.set_dynamic_obstacles(dynamic_obs)
            result = planner.plan(
                start=task.start,
                goal=task.selected_goal,
                robot_id=task.robot_id,
            )
            results[task.robot_id] = result
            self.metrics.increment_astar_calls()
            self.metrics.add_expanded_nodes(result.expanded_nodes)
        return results

    def _replan_loop(
        self,
        state: PlanningState,
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int, int]],
    ) -> PlanningState:
        """Execute up to 3 replanning attempts."""
        while (
            state.retry_count < self.replanning_policy.max_retries
            and state.current_conflicts
        ):
            state.status = BatchStatus.REPLANNING

            # Diagnose and decide
            decision = self.replanning_agent.decide(
                state.current_conflicts,
                state.current_paths,
                state.priority_order,
                state.retry_count,
            )
            state.replan_history.append(decision)

            # Apply decision
            state = self.replanning_policy.apply(
                state,
                decision,
                static_obs,
                dynamic_obs,
                self.map,
                self.metrics,
                self.max_timestep,
            )

            # Re-detect conflicts
            new_conflicts = self.conflict_detector.detect(state.current_paths)
            state.current_conflicts = new_conflicts
            self.metrics.set_retry_count(state.retry_count)
            self.metrics.set_final_conflicts(len(new_conflicts))

            if not new_conflicts:
                break

        if state.current_conflicts:
            state.failure_reason = (
                f"Conflicts remain after {state.retry_count} retries"
            )
        else:
            # Check if any paths still failed
            failed_after = [
                rid
                for rid, rp in state.current_paths.items()
                if not rp.success
            ]
            if failed_after and len(failed_after) < len(state.current_paths):
                state.status = BatchStatus.PARTIALLY_SUCCEEDED
                state.failure_reason = f"Some tasks failed: {failed_after}"
            elif failed_after:
                state.failure_reason = "All tasks failed during replanning"
                # Status remains as is (will be set to INFEASIBLE by caller)
            else:
                state.status = BatchStatus.SUCCEEDED

        return state

    def _partial_execution(
        self,
        state: PlanningState,
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int, int]],
    ) -> PlanningState:
        """Search for a feasible subset of tasks."""
        all_tasks = state.task_batch.tasks
        n = len(all_tasks)
        if n <= 1:
            state.status = BatchStatus.INFEASIBLE
            state.failure_reason = "Only 1 task but still infeasible"
            return state

        # Get currently failed/unresolved robots
        failed_robots = set()
        for rid, rp in state.current_paths.items():
            if not rp.success:
                failed_robots.add(rid)
        # Also robots involved in remaining conflicts
        for c in state.current_conflicts:
            failed_robots.update(c.robot_ids)

        # Try removing failed robots one by one, starting with lowest priority
        # Sort by priority (higher number = lower priority)
        sorted_tasks = sorted(all_tasks, key=lambda t: -t.priority)
        removed_robots = set()

        for task in sorted_tasks:
            if task.robot_id not in failed_robots:
                continue
            removed_robots.add(task.robot_id)
            # Try subset without this robot
            subset_tasks = [
                t for t in all_tasks if t.robot_id not in removed_robots
            ]
            if len(subset_tasks) < 1:
                break

            # Keep removed robots occupying their start positions
            occupied_starts = {
                t.robot_id: t.start
                for t in all_tasks
                if t.robot_id in removed_robots
            }

            # Re-plan subset
            result = self._plan_subset(
                subset_tasks,
                static_obs,
                dynamic_obs,
                occupied_starts,
            )
            if result is not None:
                state.current_paths = result
                state.warnings.append(
                    f"Partial execution: removed robots {list(removed_robots)}"
                )
                state.status = BatchStatus.PARTIALLY_SUCCEEDED
                return state

        state.status = BatchStatus.INFEASIBLE
        state.failure_reason = "No feasible subset found"
        return state

    def _plan_subset(
        self,
        subset_tasks: List[RobotTask],
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int, int]],
        occupied_starts: Dict[str, Tuple[int, int]],
    ) -> Optional[Dict[str, PathPlanResult]]:
        """Plan a subset of tasks with occupied starts as dynamic obstacles."""
        # Build fresh dynamic obstacles including occupied starts
        combined_dynamic = set(dynamic_obs)
        for rid, pos in occupied_starts.items():
            # Block the start position for all time
            for t in range(self.max_timestep + 1):
                combined_dynamic.add((pos[0], pos[1], t))

        # Priority-ordered planning
        sorted_tasks = sorted(subset_tasks, key=lambda t: t.priority)
        reservation = ReservationTable()
        results: Dict[str, PathPlanResult] = {}

        for task in sorted_tasks:
            planner = AStarPlanner(
                width=self.map.width,
                height=self.map.height,
                static_obstacles=static_obs,
                max_timestep=self.max_timestep,
                move_cost=self.map.movement.move_cost,
                wait_cost=self.map.movement.wait_cost,
            )
            planner.set_dynamic_obstacles(combined_dynamic)
            planner.set_reservation_table(reservation)

            result = planner.plan(
                start=task.start,
                goal=task.selected_goal,
                robot_id=task.robot_id,
            )
            self.metrics.increment_astar_calls()
            self.metrics.add_expanded_nodes(result.expanded_nodes)

            results[task.robot_id] = result
            if result.success:
                reservation.reserve_path(
                    result.path,
                    task.robot_id,
                    self.max_timestep,
                )
            else:
                return None  # This subset is infeasible

        # Verify no conflicts with each other
        conflicts = self.conflict_detector.detect(results)
        if conflicts:
            return None

        # Also verify no conflicts with occupied starts
        for rid, rp in results.items():
            if not rp.success:
                continue
            for node in rp.path:
                if (node.x, node.y, node.time) in combined_dynamic:
                    return None

        return results

    def _build_static_obstacle_set(self) -> Set[Tuple[int, int]]:
        obs = set()
        for so in self.map.static_obstacles:
            obs.update(tuple(c) for c in so.cells)
        return obs

    def _build_dynamic_obstacle_set(
        self,
        blockages: List[DynamicBlockage],
    ) -> Set[Tuple[int, int, int]]:
        obs = set()
        for b in blockages:
            cells = b.cells or []
            for t in range(self.max_timestep + 1):
                if b.is_active_at(t):
                    for cx, cy in cells:
                        obs.add((cx, cy, t))
        return obs

    def _finalize(self, state: PlanningState) -> PlanningState:
        """Compute final metrics and task results."""
        successful = sum(
            1 for rp in state.current_paths.values() if rp.success
        )
        failed = state.task_batch.task_count - successful

        state.metrics = self.metrics.build_metrics(
            total_task_count=state.task_batch.task_count,
            planned_task_count=successful,
            planning_failed_task_count=failed,
        )
        state.total_planning_time_ms = state.metrics.total_planning_time_ms

        # Build per-task results
        for task in state.task_batch.tasks:
            rp = state.current_paths.get(task.robot_id)
            state.task_results.append(
                SingleTaskResult(
                    robot_id=task.robot_id,
                    task=task,
                    path=rp.path if rp and rp.success else [],
                    success=rp.success if rp else False,
                    failure_reason=rp.failure_reason.value if rp and rp.failure_reason else None,
                    replanned=(
                        task.robot_id
                        in [
                            r
                            for dec in state.replan_history
                            for r in dec.robot_to_replan
                        ]
                    ),
                )
            )

        return state
