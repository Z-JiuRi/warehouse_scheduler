"""Domain models for structured robot tasks."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RobotTask:
    robot_id: str
    start: Tuple[int, int]
    goal_location_id: str
    candidate_goals: List[Tuple[int, int]] = field(default_factory=list)
    selected_goal: Optional[Tuple[int, int]] = None
    priority: int = 0

    def __post_init__(self):
        if isinstance(self.start, list):
            self.start = tuple(self.start)
        self.candidate_goals = [
            tuple(g) if isinstance(g, list) else g for g in self.candidate_goals
        ]
        if isinstance(self.selected_goal, list):
            self.selected_goal = tuple(self.selected_goal)


@dataclass
class TaskBatch:
    tasks: List[RobotTask] = field(default_factory=list)
    runtime_constraints: List = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def is_valid(self) -> bool:
        return len(self.parse_errors) == 0 and len(self.tasks) > 0
