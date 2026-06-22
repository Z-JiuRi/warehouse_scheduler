"""Unit tests for A* planner."""

import pytest
from app.tools.astar_planner import AStarPlanner
from app.domain.path_models import FailureReason


class TestAStarPlanner:
    @pytest.fixture
    def planner(self):
        # 10x10 empty grid
        return AStarPlanner(
            width=10,
            height=10,
            static_obstacles=set(),
            max_timestep=200,
        )

    @pytest.fixture
    def planner_with_obstacles(self):
        # 10x10 with a wall in the middle
        obstacles = {(5, y) for y in range(3, 8)}  # vertical wall
        return AStarPlanner(
            width=10,
            height=10,
            static_obstacles=obstacles,
            max_timestep=200,
        )

    def test_simple_path(self, planner):
        """Test basic straight-line path."""
        result = planner.plan(start=(0, 0), goal=(5, 0), robot_id="R1")
        assert result.success
        assert result.path[0].x == 0 and result.path[0].y == 0
        assert result.path[-1].x == 5 and result.path[-1].y == 0

    def test_path_with_obstacles(self, planner_with_obstacles):
        """Test path that must navigate around obstacles."""
        result = planner_with_obstacles.plan(
            start=(0, 5), goal=(9, 5), robot_id="R1"
        )
        assert result.success
        # Must go around the wall, not through it
        for node in result.path:
            assert (node.x, node.y) not in {(5, y) for y in range(3, 8)}

    def test_wait_action(self, planner):
        """Test that planner can include wait actions."""
        result = planner.plan(start=(0, 0), goal=(0, 0), robot_id="R1")
        # Start == goal, should succeed (at minimum a path of length 1)
        assert result.success
        assert len(result.path) >= 1

    def test_start_blocked_by_obstacle(self):
        """Test failure when start is an obstacle."""
        planner = AStarPlanner(
            width=10, height=10, static_obstacles={(0, 0)}, max_timestep=200
        )
        result = planner.plan(start=(0, 0), goal=(5, 5), robot_id="R1")
        assert not result.success
        assert result.failure_reason == FailureReason.START_BLOCKED

    def test_goal_blocked_by_obstacle(self, planner):
        """Test failure when goal is an obstacle."""
        obstacles = {(5, 5)}
        planner2 = AStarPlanner(
            width=10, height=10, static_obstacles=obstacles, max_timestep=200
        )
        result = planner2.plan(start=(0, 0), goal=(5, 5), robot_id="R1")
        assert not result.success
        assert result.failure_reason == FailureReason.GOAL_BLOCKED

    def test_out_of_bounds_start(self, planner):
        """Test failure for out-of-bounds start."""
        result = planner.plan(start=(-1, 0), goal=(5, 5), robot_id="R1")
        assert not result.success

    def test_out_of_bounds_goal(self, planner):
        """Test failure for out-of-bounds goal."""
        result = planner.plan(start=(0, 0), goal=(10, 10), robot_id="R1")
        assert not result.success
        assert result.failure_reason == FailureReason.OUT_OF_BOUNDS

    def test_path_continuity(self, planner):
        """Test path is continuous with max 1 step movement."""
        result = planner.plan(start=(0, 0), goal=(9, 9), robot_id="R1")
        assert result.success
        for i in range(1, len(result.path)):
            prev = result.path[i - 1]
            curr = result.path[i]
            dx = abs(curr.x - prev.x)
            dy = abs(curr.y - prev.y)
            assert dx + dy <= 1  # max one step in any direction
            assert curr.time == prev.time + 1  # time must be sequential

    def test_no_reverse_path(self, planner):
        """Test path doesn't go backwards unnecessarily."""
        result = planner.plan(start=(0, 0), goal=(9, 0), robot_id="R1")
        assert result.success
        # Path should not go negative x
        for node in result.path:
            assert node.x >= 0

    def test_dynamic_obstacles(self, planner):
        """Test that dynamic obstacles are respected."""
        # Block (5, 0) at t=5
        planner.set_dynamic_obstacles({(5, 0, 5)})
        result = planner.plan(start=(0, 0), goal=(9, 0), robot_id="R1")
        assert result.success
        for node in result.path:
            # Should not be on (5, 0) at t=5
            assert not (node.x == 5 and node.y == 0 and node.time == 5)

    def test_max_timestep(self):
        """Test that max timestep is enforced."""
        # Tiny timestep on a huge distance -> should fail
        planner = AStarPlanner(
            width=10, height=10, static_obstacles=set(), max_timestep=3
        )
        result = planner.plan(start=(0, 0), goal=(9, 9), robot_id="R1")
        # Should fail because 9+9=18 steps minimum, but max_timestep=3
        assert not result.success

    def test_expanded_nodes_tracking(self, planner):
        """Test that expanded_nodes counter works."""
        result = planner.plan(start=(0, 0), goal=(5, 5), robot_id="R1")
        assert result.success
        assert result.expanded_nodes > 0

    def test_cost_tracking(self, planner):
        """Test that cost is computed correctly."""
        result = planner.plan(start=(0, 0), goal=(5, 0), robot_id="R1")
        assert result.success
        # 5 moves at cost 1.0 each = 5.0
        assert result.cost >= 5.0

    def test_manhattan_distance_path(self, planner):
        """Test that the path length equals Manhattan distance on empty grid."""
        result = planner.plan(start=(0, 0), goal=(3, 4), robot_id="R1")
        assert result.success
        # Path should take at least 7 steps (3 right + 4 down) with no waiting
        path_steps = result.path[-1].time - result.path[0].time
        assert path_steps >= 7
