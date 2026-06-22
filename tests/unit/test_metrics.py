"""Unit tests for metrics collector."""

import pytest
import time
from app.services.metrics_collector import MetricsCollector


class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        c = MetricsCollector()
        c.start()
        return c

    def test_initial_state(self, collector):
        metrics = collector.build_metrics(0, 0, 0)
        assert metrics.total_task_count == 0
        assert metrics.planned_task_count == 0
        assert metrics.planning_success_rate == 0.0

    def test_success_rate(self, collector):
        metrics = collector.build_metrics(
            total_task_count=5,
            planned_task_count=3,
            planning_failed_task_count=2,
        )
        assert metrics.planning_success_rate == 0.6

    def test_success_rate_all_success(self, collector):
        metrics = collector.build_metrics(
            total_task_count=3,
            planned_task_count=3,
            planning_failed_task_count=0,
        )
        assert metrics.planning_success_rate == 1.0

    def test_success_rate_all_fail(self, collector):
        metrics = collector.build_metrics(
            total_task_count=3,
            planned_task_count=0,
            planning_failed_task_count=3,
        )
        assert metrics.planning_success_rate == 0.0

    def test_zero_tasks(self, collector):
        metrics = collector.build_metrics(0, 0, 0)
        assert metrics.planning_success_rate == 0.0

    def test_parsing_time(self, collector):
        collector.start_parsing()
        time.sleep(0.01)
        collector.end_parsing()
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.parsing_time_ms > 0

    def test_initial_planning_time(self, collector):
        collector.start_initial_planning()
        time.sleep(0.01)
        collector.end_initial_planning()
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.initial_planning_time_ms > 0

    def test_replanning_time(self, collector):
        collector.add_replanning_time(100.0)
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.replanning_time_ms == 100.0

    def test_astar_call_count(self, collector):
        collector.increment_astar_calls(3)
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.astar_call_count == 3

    def test_expanded_nodes(self, collector):
        collector.add_expanded_nodes(42)
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.total_expanded_nodes == 42

    def test_retry_tracking(self, collector):
        collector.set_retry_count(2)
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.retry_count == 2
        assert metrics.replanning_triggered is True

    def test_no_retry_not_triggered(self, collector):
        collector.set_retry_count(0)
        metrics = collector.build_metrics(1, 1, 0)
        assert metrics.retry_count == 0
        assert metrics.replanning_triggered is False

    def test_average_time_per_task(self, collector):
        metrics = collector.build_metrics(4, 4, 0)
        assert metrics.average_planning_time_per_task_ms > 0

    def test_conflict_counts(self, collector):
        collector.set_initial_conflicts(5)
        collector.set_final_conflicts(2)
        metrics = collector.build_metrics(3, 3, 0)
        assert metrics.initial_conflict_count == 5
        assert metrics.final_conflict_count == 2
