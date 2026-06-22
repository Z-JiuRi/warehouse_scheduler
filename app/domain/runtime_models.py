"""Domain models for robot runtime state and dynamic blockages."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class RobotState:
    robot_id: str
    position: Tuple[int, int]
    status: str = "idle"
    enabled: bool = True


@dataclass
class DynamicBlockage:
    blockage_id: str
    target_type: str  # "corridor", "cells"
    target_id: Optional[str] = None
    cells: Optional[List[Tuple[int, int]]] = None
    start_time: int = 0
    end_time: Optional[int] = None
    reason: str = ""
    source: str = "runtime"  # "runtime" or "user_instruction"

    def is_active_at(self, time: int) -> bool:
        if self.end_time is None:
            return time >= self.start_time
        return self.start_time <= time <= self.end_time
