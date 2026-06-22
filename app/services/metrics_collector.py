"""Metrics collector: collects and computes planning metrics."""

import time
from typing import List
from app.domain.planning_state import PlanningMetrics
from app.domain.path_models import PathPlanResult


class MetricsCollector:
    def __init__(self):
        self._start_time: float = 0.0
        self._parsing_start: float = 0.0
        self._parsing_time: float = 0.0
        self._initial_planning_start: float = 0.0
        self._initial_planning_time: float = 0.0
        self._replanning_time: float = 0.0
        self._astar_call_count: int = 0
        self._total_expanded_nodes: int = 0
        self._initial_conflict_count: int = 0
        self._final_conflict_count: int = 0
        self._retry_count: int = 0
        self._replanning_triggered: bool = False

    def start(self):
        self._start_time = time.time()

    def start_parsing(self):
        self._parsing_start = time.time()

    def end_parsing(self):
        self._parsing_time = (time.time() - self._parsing_start) * 1000

    def start_initial_planning(self):
        self._initial_planning_start = time.time()

    def end_initial_planning(self):
        self._initial_planning_time = (time.time() - self._initial_planning_start) * 1000

    def add_replanning_time(self, ms: float):
        self._replanning_time += ms

    def increment_astar_calls(self, count: int = 1):
        self._astar_call_count += count

    def add_expanded_nodes(self, count: int):
        self._total_expanded_nodes += count

    def set_initial_conflicts(self, count: int):
        self._initial_conflict_count = count

    def set_final_conflicts(self, count: int):
        self._final_conflict_count = count

    def set_retry_count(self, count: int):
        self._retry_count = count
        if count > 0:
            self._replanning_triggered = True

    def build_metrics(
        self,
        total_task_count: int,
        planned_task_count: int,
        planning_failed_task_count: int,
    ) -> PlanningMetrics:
        total_time = (time.time() - self._start_time) * 1000
        success_rate = planned_task_count / max(total_task_count, 1)

        avg_per_task = 0.0
        if total_task_count > 0:
            avg_per_task = total_time / total_task_count

        return PlanningMetrics(
            total_task_count=total_task_count,
            planned_task_count=planned_task_count,
            planning_failed_task_count=planning_failed_task_count,
            planning_success_rate=success_rate,
            total_planning_time_ms=total_time,
            parsing_time_ms=self._parsing_time,
            initial_planning_time_ms=self._initial_planning_time,
            replanning_time_ms=self._replanning_time,
            average_planning_time_per_task_ms=avg_per_task,
            initial_conflict_count=self._initial_conflict_count,
            final_conflict_count=self._final_conflict_count,
            replanning_triggered=self._replanning_triggered,
            retry_count=self._retry_count,
            astar_call_count=self._astar_call_count,
            total_expanded_nodes=self._total_expanded_nodes,
        )
