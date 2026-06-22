"""Unit tests for path validator."""

import pytest
from app.tools.path_validator import PathValidator
from app.domain.path_models import TimedPosition, PathPlanResult
from app.domain.map_models import WarehouseMap, StaticObstacle


def make_map(width=10, height=10, obstacles=None):
    """Create a minimal test map."""
    obs = []
    if obstacles:
        for i, cells in enumerate(obstacles):
            obs.append(
                StaticObstacle(
                    obstacle_id=f"obs_{i}",
                    type="test",
                    cells=[tuple(c) if isinstance(c, list) else c for c in cells],
                )
            )
    wmap = WarehouseMap(
        map_id="test_map",
        name="Test Map",
        width=width,
        height=height,
        static_obstacles=obs,
    )
    wmap.build_indices()
    return wmap


def make_result(path_tuples):
    return PathPlanResult(
        success=True,
        path=[TimedPosition(x=p[0], y=p[1], time=p[2]) for p in path_tuples],
    )


class TestPathValidator:
    def test_valid_single_path(self):
        """A correct path should validate with no errors."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1), TimedPosition(2, 0, 2)]
        errors = validator.validate_single_path(path, (0, 0), (2, 0))
        assert len(errors) == 0

    def test_wrong_start(self):
        """Path starting at wrong position should error."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(1, 0, 0), TimedPosition(2, 0, 1)]
        errors = validator.validate_single_path(path, (0, 0), (2, 0))
        assert any("starts at" in e for e in errors)

    def test_wrong_goal(self):
        """Path ending at wrong position should error."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1)]
        errors = validator.validate_single_path(path, (0, 0), (2, 0))
        assert any("ends at" in e for e in errors)

    def test_obstacle_violation(self):
        """Path through obstacle should error."""
        wmap = make_map(obstacles=[[(1, 0)]])
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1), TimedPosition(2, 0, 2)]
        errors = validator.validate_single_path(path, (0, 0), (2, 0))
        assert any("obstacle" in e.lower() for e in errors)

    def test_out_of_bounds(self):
        """Path out of bounds should error."""
        wmap = make_map(width=5, height=5)
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(6, 0, 1)]
        errors = validator.validate_single_path(path, (0, 0), (6, 0))
        assert any("bounds" in e.lower() for e in errors)

    def test_non_monotonic_time(self):
        """Non-sequential time should error."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 3)]  # time jump
        errors = validator.validate_single_path(path, (0, 0), (1, 0))
        assert any("monotonic" in e.lower() for e in errors)

    def test_invalid_move(self):
        """Moving more than 1 cell should error."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(2, 0, 1)]  # 2-step jump
        errors = validator.validate_single_path(path, (0, 0), (2, 0))
        assert any("invalid move" in e.lower() for e in errors)

    def test_dynamic_obstacle_violation(self):
        """Path through dynamic obstacle should error."""
        wmap = make_map()
        validator = PathValidator(wmap)
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1)]
        errors = validator.validate_single_path(
            path, (0, 0), (1, 0), dynamic_obstacles={(1, 0, 1)}
        )
        assert any("dynamic" in e.lower() or "blocked" in e.lower() for e in errors)

    def test_multi_robot_vertex_conflict(self):
        """Multi-robot validation should catch vertex conflicts."""
        wmap = make_map()
        validator = PathValidator(wmap)
        p1 = make_result([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_result([(3, 0, 0), (2, 0, 1), (1, 0, 2)])
        # Both at (2, 0) at t=2? No: p1[2]=(2,0,2), p2[1]=(2,0,1)
        # Let's make them actually conflict
        p1 = make_result([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_result([(4, 0, 0), (3, 0, 1), (2, 0, 2)])
        errors = validator.validate_multi_robot({"R1": p1, "R2": p2})
        assert len(errors) >= 1
        assert any("conflict" in e.lower() for e in errors)

    def test_multi_robot_swap_conflict(self):
        """Multi-robot validation should catch swap conflicts."""
        wmap = make_map()
        validator = PathValidator(wmap)
        p1 = make_result([(0, 0, 0), (1, 0, 1)])
        p2 = make_result([(1, 0, 0), (0, 0, 1)])
        errors = validator.validate_multi_robot({"R1": p1, "R2": p2})
        assert len(errors) >= 1

    def test_multi_robot_no_conflict(self):
        """Separate paths should validate cleanly."""
        wmap = make_map()
        validator = PathValidator(wmap)
        p1 = make_result([(0, 0, 0), (1, 0, 1)])
        p2 = make_result([(5, 5, 0), (5, 4, 1)])
        errors = validator.validate_multi_robot({"R1": p1, "R2": p2})
        assert len(errors) == 0

    def test_single_robot_multi_validation(self):
        """Single robot should pass multi validation."""
        wmap = make_map()
        validator = PathValidator(wmap)
        p1 = make_result([(0, 0, 0), (1, 0, 1)])
        errors = validator.validate_multi_robot({"R1": p1})
        assert len(errors) == 0

    def test_endpoint_conflict_multi(self):
        """Endpoint padding should catch late-arriving robots."""
        wmap = make_map()
        validator = PathValidator(wmap)
        p1 = make_result([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_result([(5, 5, 0), (4, 5, 1), (3, 5, 2), (2, 0, 3)])
        errors = validator.validate_multi_robot({"R1": p1, "R2": p2})
        assert len(errors) >= 1
