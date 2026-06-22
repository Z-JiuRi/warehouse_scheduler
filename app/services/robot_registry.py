"""Robot registry: manages robot runtime positions and states."""

import json
import os
from typing import Dict, Tuple, Optional, List
from app.domain.runtime_models import RobotState, DynamicBlockage


class RobotRegistry:
    def __init__(self, runtime_path: str = None):
        if runtime_path is None:
            runtime_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs",
                "warehouse_runtime.json",
            )
        self.runtime_path = runtime_path
        self._robots: Dict[str, RobotState] = {}
        self._blockages: List[DynamicBlockage] = []

    def load(self) -> List[str]:
        """Load runtime state. Returns errors."""
        errors = []
        try:
            with open(self.runtime_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            errors.append(f"Runtime file not found: {self.runtime_path}")
            return errors
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in runtime file: {e}")
            return errors

        self._robots.clear()
        robot_ids = set()
        for r in raw.get("robots", []):
            rid = r.get("robot_id", "")
            if rid in robot_ids:
                errors.append(f"Duplicate robot_id: {rid}")
                continue
            robot_ids.add(rid)
            pos = r.get("position", [0, 0])
            if isinstance(pos, list) and len(pos) == 2:
                pos = tuple(pos)
            self._robots[rid] = RobotState(
                robot_id=rid,
                position=pos,
                status=r.get("status", "idle"),
                enabled=r.get("enabled", True),
            )

        self._blockages.clear()
        for b in raw.get("active_blockages", []):
            cells = None
            if "cells" in b and b["cells"]:
                cells = [tuple(c) if isinstance(c, list) else c for c in b["cells"]]
            self._blockages.append(
                DynamicBlockage(
                    blockage_id=b.get("blockage_id", ""),
                    target_type=b.get("target_type", "cells"),
                    target_id=b.get("target_id"),
                    cells=cells,
                    start_time=b.get("start_time", 0),
                    end_time=b.get("end_time"),
                    reason=b.get("reason", ""),
                    source=b.get("source", "runtime"),
                )
            )

        return errors

    def get_position(self, robot_id: str) -> Optional[Tuple[int, int]]:
        """Get the current position of a robot."""
        r = self._robots.get(robot_id)
        return r.position if r and r.enabled else None

    def is_enabled(self, robot_id: str) -> bool:
        r = self._robots.get(robot_id)
        return r is not None and r.enabled

    def get_robot_ids(self) -> List[str]:
        return [r.robot_id for r in self._robots.values() if r.enabled]

    def get_all_positions(self) -> Dict[str, Tuple[int, int]]:
        return {rid: r.position for rid, r in self._robots.items() if r.enabled}

    def get_blockages(self) -> List[DynamicBlockage]:
        return list(self._blockages)

    def set_position(self, robot_id: str, position: Tuple[int, int]):
        """Update robot position. Only for planning purposes; does NOT persist."""
        if robot_id in self._robots:
            self._robots[robot_id].position = position
