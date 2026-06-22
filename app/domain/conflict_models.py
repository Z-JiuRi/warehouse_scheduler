"""Domain models for conflicts."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ConflictType(str, Enum):
    VERTEX = "vertex"
    EDGE = "edge"
    SWAP = "swap"
    START = "start"
    GOAL = "goal"


@dataclass
class Conflict:
    conflict_type: ConflictType
    robot_ids: List[str]
    time: int
    position: Optional[Tuple[int, int]] = None
    edge: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    description: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.conflict_type.value}] t={self.time} "
            f"robots={self.robot_ids} pos={self.position} edge={self.edge}"
        )
