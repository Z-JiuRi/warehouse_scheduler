"""Unit tests for conflict detector."""

import pytest
from app.tools.conflict_detector import ConflictDetector
from app.domain.path_models import PathPlanResult, TimedPosition
from app.domain.conflict_models import ConflictType


def make_path(path_tuples):
    """Helper to create a PathPlanResult from list of (x, y, t) tuples."""
    return PathPlanResult(
        success=True,
        path=[TimedPosition(x=p[0], y=p[1], time=p[2]) for p in path_tuples],
    )


class TestConflictDetector:
    @pytest.fixture
    def detector(self):
        return ConflictDetector()

    def test_no_conflict_separate_paths(self, detector):
        """Two robots on completely separate paths."""
        p1 = make_path([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_path([(5, 5, 0), (5, 4, 1), (5, 3, 2)])
        conflicts = detector.detect({"R1": p1, "R2": p2})
        assert len(conflicts) == 0

    def test_vertex_conflict(self, detector):
        """Two robots occupy same cell at same time."""
        p1 = make_path([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_path([(3, 0, 0), (2, 0, 1), (1, 0, 2)])
        # Both at (2, 0, 2)? No: p[2] for p1 = (2,0,2), for p2 p2[2] = (1,0,2)
        # Actually let me think: p1 goes (0,0)->(1,0)->(2,0), p2 goes (3,0)->(2,0)->(1,0)
        # At t=1: p1=(1,0), p2=(2,0) -> no conflict
        # At t=2: p1=(2,0), p2=(1,0) -> no conflict
        # Let's fix: make them meet at same cell
        p1 = make_path([(0, 0, 0), (1, 0, 1), (2, 0, 2), (2, 0, 3)])
        p2 = make_path([(4, 0, 0), (3, 0, 1), (2, 0, 2), (1, 0, 3)])
        # Both at (2, 0) at t=2
        conflicts = detector.detect({"R1": p1, "R2": p2})
        vertex = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        assert len(vertex) >= 1
        assert vertex[0].position == (2, 0)
        assert vertex[0].time == 2

    def test_swap_conflict(self, detector):
        """Two robots swap adjacent positions."""
        p1 = make_path([(0, 0, 0), (1, 0, 1)])
        p2 = make_path([(1, 0, 0), (0, 0, 1)])
        conflicts = detector.detect({"R1": p1, "R2": p2})
        swap = [c for c in conflicts if c.conflict_type == ConflictType.SWAP]
        assert len(swap) >= 1

    def test_start_conflict(self, detector):
        """Two robots start at same position."""
        p1 = make_path([(0, 0, 0), (1, 0, 1)])
        p2 = make_path([(0, 0, 0), (0, 1, 1)])
        conflicts = detector.detect({"R1": p1, "R2": p2})
        start = [c for c in conflicts if c.conflict_type == ConflictType.START]
        assert len(start) >= 1

    def test_goal_conflict(self, detector):
        """Two robots have same final destination."""
        p1 = make_path([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_path([(0, 1, 0), (1, 1, 1), (2, 0, 2)])
        conflicts = detector.detect({"R1": p1, "R2": p2})
        goal = [c for c in conflicts if c.conflict_type == ConflictType.GOAL]
        assert len(goal) >= 1

    def test_endpoint_occupancy_padding(self, detector):
        """Shorter path robot continues occupying endpoint."""
        p1 = make_path([(0, 0, 0), (1, 0, 1), (2, 0, 2)])
        p2 = make_path([(5, 5, 0), (4, 5, 1), (3, 5, 2), (2, 0, 3)])
        # p1 stops at (2,0) at t=2, p2 arrives at (2,0) at t=3 -> conflict!
        conflicts = detector.detect({"R1": p1, "R2": p2})
        vertex = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        assert len(vertex) >= 1
        # Should detect at t=3 where p1 padded occupies (2,0) and p2 is at (2,0)

    def test_single_robot_no_conflict(self, detector):
        """Single robot path should have no conflicts."""
        p1 = make_path([(0, 0, 0), (1, 0, 1)])
        conflicts = detector.detect({"R1": p1})
        assert len(conflicts) == 0

    def test_empty_paths(self, detector):
        """Empty paths should not cause crashes."""
        p1 = PathPlanResult(success=False)
        p2 = make_path([(0, 0, 0), (1, 0, 1)])
        conflicts = detector.detect({"R1": p1, "R2": p2})
        # Failed path means no path to conflict on
        assert len(conflicts) == 0
