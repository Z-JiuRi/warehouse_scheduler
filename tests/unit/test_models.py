"""Unit tests for domain models."""

import pytest
from app.domain.map_models import WarehouseMap, Location, StaticObstacle, Corridor
from app.domain.task_models import RobotTask, TaskBatch
from app.domain.path_models import TimedPosition, PathPlanResult, FailureReason
from app.domain.conflict_models import Conflict, ConflictType


class TestWarehouseMap:
    @pytest.fixture
    def sample_map(self):
        wmap = WarehouseMap(
            map_id="test",
            name="Test Map",
            width=10,
            height=10,
            static_obstacles=[
                StaticObstacle(
                    obstacle_id="wall",
                    type="wall",
                    cells=[(5, 5)],
                )
            ],
            locations=[
                Location(
                    location_id="zone_a",
                    name="区域A",
                    aliases=["A区", "Zone A"],
                    type="loading_zone",
                    entry_cells=[(1, 1)],
                    capacity=1,
                )
            ],
            corridors=[
                Corridor(
                    corridor_id="corr_1",
                    name="北通道",
                    cells=[(0, 0), (1, 0)],
                )
            ],
        )
        wmap.build_indices()
        return wmap

    def test_is_obstacle(self, sample_map):
        assert sample_map.is_obstacle(5, 5)
        assert not sample_map.is_obstacle(0, 0)

    def test_in_bounds(self, sample_map):
        assert sample_map.in_bounds(0, 0)
        assert sample_map.in_bounds(9, 9)
        assert not sample_map.in_bounds(-1, 0)
        assert not sample_map.in_bounds(10, 10)

    def test_is_walkable(self, sample_map):
        assert sample_map.is_walkable(0, 0)
        assert not sample_map.is_walkable(5, 5)  # obstacle
        assert not sample_map.is_walkable(10, 10)  # out of bounds

    def test_find_location_by_id(self, sample_map):
        loc = sample_map.find_location("zone_a")
        assert loc is not None
        assert loc.name == "区域A"

    def test_find_location_by_alias(self, sample_map):
        loc = sample_map.find_location("A区")
        assert loc is not None
        assert loc.location_id == "zone_a"

    def test_find_location_case_insensitive(self, sample_map):
        loc = sample_map.find_location("zone a")
        assert loc is not None

    def test_find_location_not_found(self, sample_map):
        assert sample_map.find_location("nonexistent") is None

    def test_find_corridor_by_id(self, sample_map):
        corr = sample_map.find_corridor("corr_1")
        assert corr is not None
        assert corr.name == "北通道"

    def test_find_corridor_by_name(self, sample_map):
        corr = sample_map.find_corridor("北通道")
        assert corr is not None
        assert corr.corridor_id == "corr_1"

    def test_find_corridor_not_found(self, sample_map):
        assert sample_map.find_corridor("nonexistent") is None


class TestRobotTask:
    def test_task_creation(self):
        task = RobotTask(
            robot_id="R1",
            start=(0, 0),
            goal_location_id="zone_a",
            candidate_goals=[(1, 1), (2, 2)],
            priority=2,
        )
        assert task.robot_id == "R1"
        assert task.start == (0, 0)
        assert task.candidate_goals == [(1, 1), (2, 2)]
        assert task.priority == 2
        assert task.selected_goal is None

    def test_task_list_conversion(self):
        task = RobotTask(
            robot_id="R1",
            start=[0, 0],
            goal_location_id="zone_a",
            candidate_goals=[[1, 1]],
            selected_goal=[1, 1],
        )
        assert isinstance(task.start, tuple)
        assert isinstance(task.candidate_goals[0], tuple)
        assert isinstance(task.selected_goal, tuple)


class TestTaskBatch:
    def test_empty_batch(self):
        batch = TaskBatch()
        assert batch.task_count == 0
        assert not batch.is_valid

    def test_valid_batch(self):
        task = RobotTask(robot_id="R1", start=(0, 0), goal_location_id="zone_a")
        batch = TaskBatch(tasks=[task])
        assert batch.task_count == 1
        assert batch.is_valid

    def test_batch_with_errors(self):
        task = RobotTask(robot_id="R1", start=(0, 0), goal_location_id="zone_a")
        batch = TaskBatch(
            tasks=[task], parse_errors=["Some error"]
        )
        assert not batch.is_valid


class TestTimedPosition:
    def test_as_tuple(self):
        tp = TimedPosition(1, 2, 3)
        assert tp.as_tuple() == (1, 2, 3)

    def test_as_pos_tuple(self):
        tp = TimedPosition(1, 2, 3)
        assert tp.as_pos_tuple() == (1, 2)


class TestPathPlanResult:
    def test_success_path(self):
        path = [TimedPosition(0, 0, 0), TimedPosition(1, 0, 1)]
        result = PathPlanResult(success=True, path=path, cost=1.0, expanded_nodes=5)
        assert result.success
        assert result.length == 2
        assert result.makespan == 1

    def test_failure(self):
        result = PathPlanResult(
            success=False, failure_reason=FailureReason.NO_PATH
        )
        assert not result.success
        assert result.length == 0
        assert result.makespan == 0


class TestConflict:
    def test_conflict_str(self):
        c = Conflict(
            conflict_type=ConflictType.VERTEX,
            robot_ids=["R1", "R2"],
            time=5,
            position=(3, 3),
        )
        assert "vertex" in str(c)
        assert "R1" in str(c)
        assert "R2" in str(c)
