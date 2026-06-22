"""Path validator: validates path continuity, obstacle avoidance, and multi-robot safety."""

from typing import List, Dict, Set, Tuple
from app.domain.path_models import TimedPosition, PathPlanResult
from app.domain.map_models import WarehouseMap


class PathValidator:
    """Validates individual and multi-robot paths."""

    def __init__(self, warehouse_map: WarehouseMap):
        self.map = warehouse_map
        self.width = warehouse_map.width
        self.height = warehouse_map.height

    def validate_single_path(
        self,
        path: List[TimedPosition],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        dynamic_obstacles: Set[Tuple[int, int, int]] = None,
    ) -> List[str]:
        """
        Validate a single path for continuity, bounds, obstacles, and endpoints.

        Returns list of error messages (empty = valid).
        """
        errors = []
        if not path:
            errors.append("Path is empty")
            return errors

        dynamic_obs = dynamic_obstacles or set()

        # Check start
        first = path[0]
        if (first.x, first.y) != start:
            errors.append(
                f"Path starts at ({first.x},{first.y}), expected {start}"
            )

        # Check goal
        last = path[-1]
        if (last.x, last.y) != goal:
            errors.append(
                f"Path ends at ({last.x},{last.y}), expected goal {goal}"
            )

        # Check continuity and validity
        for i in range(len(path)):
            node = path[i]
            x, y, t = node.x, node.y, node.time

            # Bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                errors.append(f"Position ({x},{y}) at t={t} out of bounds")

            # Static obstacles
            if self.map.is_obstacle(x, y):
                errors.append(f"Position ({x},{y}) at t={t} is a static obstacle")

            # Dynamic obstacles
            if (x, y, t) in dynamic_obs:
                errors.append(
                    f"Position ({x},{y}) at t={t} is blocked by dynamic obstacle"
                )

            # Time monotonicity
            if i > 0:
                prev = path[i - 1]
                if t != prev.time + 1:
                    errors.append(
                        f"Time not monotonic: t={prev.time} -> t={t}"
                    )

                # Movement validity: max 1 step in any direction
                dx = abs(x - prev.x)
                dy = abs(y - prev.y)
                if dx + dy > 1:
                    errors.append(
                        f"Invalid move from ({prev.x},{prev.y}) to ({x},{y}) at t={t}"
                    )

        return errors

    def validate_multi_robot(
        self,
        paths: Dict[str, PathPlanResult],
    ) -> List[str]:
        """
        Validate multi-robot safety: no vertex or swap conflicts.

        Returns list of error messages (empty = all safe).
        """
        errors = []
        robot_ids = list(paths.keys())

        if len(robot_ids) < 2:
            return errors

        # Compute max time
        max_t = max(
            (rp.path[-1].time for rp in paths.values() if rp.success and rp.path),
            default=0,
        )
        max_t += 10

        # Build occupancy map: (x, y, t) -> robot_id
        occupancy: Dict[Tuple[int, int, int], str] = {}
        for rid, rp in paths.items():
            if not rp.success or not rp.path:
                continue
            last_pos = (rp.path[-1].x, rp.path[-1].y)
            last_t = rp.path[-1].time
            for node in rp.path:
                key = (node.x, node.y, node.time)
                if key in occupancy:
                    errors.append(
                        f"Vertex conflict: {rid} and {occupancy[key]} "
                        f"both at ({node.x},{node.y}), t={node.time}"
                    )
                occupancy[key] = rid
            # Pad endpoint
            for t in range(last_t + 1, max_t + 1):
                key = (last_pos[0], last_pos[1], t)
                if key in occupancy:
                    errors.append(
                        f"Endpoint conflict: {rid} and {occupancy[key]} "
                        f"both occupy ({last_pos[0]},{last_pos[1]}) at t={t}"
                    )
                occupancy[key] = rid

        # Check swap conflicts
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                rid_a = robot_ids[i]
                rid_b = robot_ids[j]
                rp_a = paths.get(rid_a)
                rp_b = paths.get(rid_b)
                if not rp_a or not rp_b or not rp_a.success or not rp_b.success:
                    continue
                pa = rp_a.path
                pb = rp_b.path
                for k in range(min(len(pa) - 1, len(pb) - 1)):
                    a_cur = (pa[k].x, pa[k].y)
                    a_next = (pa[k + 1].x, pa[k + 1].y)
                    b_cur = (pb[k].x, pb[k].y)
                    b_next = (pb[k + 1].x, pb[k + 1].y)
                    if a_cur == b_next and a_next == b_cur:
                        errors.append(
                            f"Swap conflict: {rid_a} and {rid_b} swap positions "
                            f"between t={pa[k].time} and t={pa[k+1].time}"
                        )

        return errors
