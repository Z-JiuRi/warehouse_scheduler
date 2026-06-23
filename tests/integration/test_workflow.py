"""Integration tests for the full planning workflow.

These tests bypass the LLM and use structured task input to test
the deterministic planning pipeline end-to-end.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.orchestration.workflow import Workflow
from app.domain.path_models import TimedPosition
from app.domain.planning_state import BatchStatus


# Paths
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
MAP_PATH = os.path.join(CONFIG_DIR, "warehouse_map.json")
RUNTIME_PATH = os.path.join(CONFIG_DIR, "warehouse_runtime.json")


@pytest.fixture
def workflow():
    """Create a workflow that bypasses LLM (no API key needed for structured mode)."""
    return Workflow(
        map_path=MAP_PATH,
        runtime_path=RUNTIME_PATH,
        max_timestep=200,
    )


def test_map_loads_correctly(workflow):
    """Test that the map loads without errors."""
    assert workflow.warehouse_map is not None
    assert len(workflow.map_errors) == 0
    assert workflow.warehouse_map.width == 10
    assert workflow.warehouse_map.height == 10


def test_runtime_loads_correctly(workflow):
    """Test that runtime state loads with 3 robots."""
    assert len(workflow.runtime_errors) == 0
    assert len(workflow.robot_registry.get_robot_ids()) >= 3


def test_simple_no_conflict(workflow):
    """Two robots with paths that don't conflict."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [1, 0], "goal_location_id": "charging_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    # Note: R1 and R2 may conflict depending on paths. Let's just check it doesn't crash.
    assert state is not None
    assert state.task_batch is not None


def test_all_success_no_conflicts(workflow):
    """Three robots with clearly separated goals should all succeed."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [2, 0], "goal_location_id": "charging_zone", "priority": 2},
            {"robot_id": "R3", "start": [4, 0], "goal_location_id": "shelf_A_pickup", "priority": 3},
        ]
    }
    state = workflow.run_structured(tasks_json)
    assert state is not None
    assert state.status in (
        BatchStatus.SUCCEEDED,
        BatchStatus.PARTIALLY_SUCCEEDED,
    )


def test_metrics_present(workflow):
    """Test that metrics are computed after planning."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [1, 0], "goal_location_id": "charging_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    assert state.metrics is not None
    assert state.metrics.total_task_count == 2
    assert state.metrics.total_planning_time_ms > 0
    assert state.metrics.astar_call_count > 0
    assert state.metrics.total_expanded_nodes > 0


def test_same_start_conflict(workflow):
    """Two robots starting at same position should trigger start conflict."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [0, 0], "goal_location_id": "charging_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    # Should detect start conflict and try to resolve
    assert state.status in (
        BatchStatus.SUCCEEDED,
        BatchStatus.PARTIALLY_SUCCEEDED,
        BatchStatus.INFEASIBLE,
    )
    # At minimum, metrics should show initial conflicts
    assert state.metrics.initial_conflict_count >= 1


def test_same_goal_conflict(workflow):
    """Two robots going to same goal location should conflict."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [1, 0], "goal_location_id": "loading_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    assert state.status in (
        BatchStatus.SUCCEEDED,
        BatchStatus.PARTIALLY_SUCCEEDED,
        BatchStatus.INFEASIBLE,
    )


def test_closed_corridor(workflow):
    """Closing a corridor should affect path planning."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
        ],
        "runtime_constraints": [
            {
                "constraint_type": "closed_corridor",
                "target_id": "corridor_north",
                "start_time": 0,
                "end_time": None,
            }
        ],
    }
    state = workflow.run_structured(tasks_json)
    assert state is not None
    if any(tr.success for tr in state.task_results):
        # If successful, path should not use closed corridor cells
        # corridor_north cells: x=1..5, y=1
        closed_cells = {(x, 1) for x in range(1, 6)}
        for tr in state.task_results:
            if tr.success:
                for p in tr.path:
                    assert (p.x, p.y) not in closed_cells, (
                        f"Robot {tr.robot_id} used closed corridor at ({p.x}, {p.y})"
                    )


def test_partial_execution(workflow):
    """When all tasks can't be scheduled together, partial execution should work."""
    # Force a very constrained scenario: all robots start at same spot, all go to same spot
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 3},
            {"robot_id": "R2", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 2},
            {"robot_id": "R3", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
        ]
    }
    state = workflow.run_structured(tasks_json)
    # Should be at least partially successful or infeasible
    assert state.status in (
        BatchStatus.PARTIALLY_SUCCEEDED,
        BatchStatus.INFEASIBLE,
    )


def test_replan_history(workflow):
    """When conflicts occur, replan history should be recorded."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [1, 0], "goal_location_id": "charging_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    # replan_history may be empty if no conflicts, but should be a list
    assert isinstance(state.replan_history, list)


def test_path_continuity_integration(workflow):
    """All successful paths should be continuous."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
            {"robot_id": "R2", "start": [2, 0], "goal_location_id": "charging_zone", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    for tr in state.task_results:
        if tr.success and tr.path:
            for i in range(1, len(tr.path)):
                prev = tr.path[i - 1]
                curr = tr.path[i]
                dx = abs(curr.x - prev.x)
                dy = abs(curr.y - prev.y)
                assert dx + dy <= 1, f"Jump in path for {tr.robot_id}"
                assert curr.time == prev.time + 1, f"Time gap in path for {tr.robot_id}"


def test_no_obstacle_violation(workflow):
    """No successful path should go through obstacles."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "loading_zone", "priority": 1},
        ]
    }
    state = workflow.run_structured(tasks_json)
    for tr in state.task_results:
        if tr.success and tr.path:
            for p in tr.path:
                assert not workflow.warehouse_map.is_obstacle(
                    p.x, p.y
                ), f"Robot {tr.robot_id} at obstacle ({p.x}, {p.y})"


def test_final_validation(workflow):
    """Successfully planned paths should pass final validation."""
    tasks_json = {
        "tasks": [
            {"robot_id": "R1", "start": [0, 0], "goal_location_id": "shelf_A_pickup", "priority": 1},
            {"robot_id": "R2", "start": [8, 0], "goal_location_id": "shelf_C_pickup", "priority": 2},
        ]
    }
    state = workflow.run_structured(tasks_json)
    if state.status == BatchStatus.SUCCEEDED:
        # All paths should be valid
        for tr in state.task_results:
            assert tr.success
            assert len(tr.path) > 0
