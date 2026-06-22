"""Replanning policy: applies replan decisions to the current state."""

from typing import Dict, Set, Tuple
from app.domain.planning_state import PlanningState, ReplanDecision
from app.domain.path_models import PathPlanResult
from app.domain.map_models import WarehouseMap
from app.services.metrics_collector import MetricsCollector
from app.tools.astar_planner import AStarPlanner
from app.tools.reservation_table import ReservationTable
from app.tools.conflict_detector import ConflictDetector


class ReplanningPolicy:
    """Executes replan decisions with up to 3 retries."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def apply(
        self,
        state: PlanningState,
        decision: ReplanDecision,
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int, int]],
        warehouse_map: WarehouseMap,
        metrics: MetricsCollector,
        max_timestep: int,
    ) -> PlanningState:
        """
        Apply a replan decision to the state.
        Modifies state.current_paths for the robots being replanned.
        """
        import time

        t0 = time.time()

        # Build reservation table from high-priority (non-replanned) robots
        reservation = ReservationTable()
        replan_set = set(decision.robot_to_replan)

        # First, reserve paths of robots NOT being replanned
        for rid, rp in state.current_paths.items():
            if rid not in replan_set and rp.success:
                reservation.reserve_path(rp.path, rid, max_timestep)

        # Apply priority changes if any
        if decision.priority_changes:
            for rid, new_prio in decision.priority_changes.items():
                for task in state.task_batch.tasks:
                    if task.robot_id == rid:
                        task.priority = new_prio
            # Re-sort priority order
            state.task_batch.tasks.sort(key=lambda t: t.priority)
            state.priority_order = [t.robot_id for t in state.task_batch.tasks]

        # Build combined dynamic obstacles
        combined_dynamic = set(dynamic_obs)
        # Add transition constraints from decision
        for constraint in decision.constraints:
            if constraint["type"] == "vertex_window":
                px, py = constraint["position"]
                for t in range(constraint["time_start"], constraint["time_end"] + 1):
                    combined_dynamic.add((px, py, t))

        # Replan affected robots in priority order
        replan_order = [
            rid for rid in state.priority_order if rid in replan_set
        ]

        for rid in replan_order:
            task = next(
                (t for t in state.task_batch.tasks if t.robot_id == rid), None
            )
            if task is None:
                continue

            planner = AStarPlanner(
                width=warehouse_map.width,
                height=warehouse_map.height,
                static_obstacles=static_obs,
                max_timestep=max_timestep,
                move_cost=warehouse_map.movement.move_cost,
                wait_cost=warehouse_map.movement.wait_cost,
            )
            planner.set_dynamic_obstacles(combined_dynamic)
            planner.set_reservation_table(reservation)

            result = planner.plan(
                start=task.start,
                goal=task.selected_goal,
                robot_id=rid,
            )
            metrics.increment_astar_calls()
            metrics.add_expanded_nodes(result.expanded_nodes)

            state.current_paths[rid] = result
            if result.success:
                reservation.reserve_path(result.path, rid, max_timestep)

        state.retry_count += 1
        elapsed = (time.time() - t0) * 1000
        metrics.add_replanning_time(elapsed)

        return state
