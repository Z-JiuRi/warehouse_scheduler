"""Domain models for path planning results."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class FailureReason(str, Enum):
    NO_PATH = "no_path"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    MAX_TIMESTEP_EXCEEDED = "max_timestep_exceeded"
    UNKNOWN_ROBOT = "unknown_robot"
    UNKNOWN_LOCATION = "unknown_location"
    MISSING_START = "missing_start"
    MISSING_GOAL = "missing_goal"
    INVALID_PRIORITY = "invalid_priority"
    OUT_OF_BOUNDS = "out_of_bounds"


@dataclass
class TimedPosition:
    x: int
    y: int
    time: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.time)

    def as_pos_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class PathPlanResult:
    success: bool
    path: List[TimedPosition] = field(default_factory=list)
    cost: float = 0.0
    expanded_nodes: int = 0
    planning_time_ms: float = 0.0
    failure_reason: Optional[FailureReason] = None

    @property
    def length(self) -> int:
        return len(self.path)

    @property
    def makespan(self) -> int:
        if not self.path:
            return 0
        return max(p.time for p in self.path)

    def get_positions(self) -> List[Tuple[int, int]]:
        return [(p.x, p.y) for p in self.path]


def path_to_tuples(path: List[TimedPosition]) -> List[Tuple[int, int, int]]:
    return [p.as_tuple() for p in path]
