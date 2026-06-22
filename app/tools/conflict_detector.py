"""Conflict detector: detects vertex, swap, start, and goal conflicts."""

from typing import List, Dict, Tuple
from app.domain.conflict_models import Conflict, ConflictType
from app.domain.path_models import TimedPosition, PathPlanResult


class ConflictDetector:
    """Detects conflicts between multiple robot paths."""

    def detect(
        self,
        paths: Dict[str, PathPlanResult],
        max_t: int = None,
    ) -> List[Conflict]:
        """
        Detect all conflicts among the given paths.

        Args:
            paths: dict of robot_id -> PathPlanResult
            max_t: maximum time to check (for endpoint occupancy padding)

        Returns:
            List of Conflict objects
        """
        conflicts: List[Conflict] = []

        robot_ids = list(paths.keys())
        if len(robot_ids) < 2:
            return conflicts

        # Compute actual max_t from paths + padding for endpoint occupancy
        if max_t is None:
            max_t = 0
            for rp in paths.values():
                if rp.success and rp.path:
                    max_t = max(max_t, rp.path[-1].time)
            max_t += 10  # padding for endpoint occupancy

        # Pad paths: robots occupying end position for all remaining time
        padded: Dict[str, Dict[int, Tuple[int, int]]] = {}
        for rid, rp in paths.items():
            if not rp.success or not rp.path:
                padded[rid] = {}
                continue
            padded[rid] = {}
            last_pos = (rp.path[-1].x, rp.path[-1].y)
            last_t = rp.path[-1].time
            for node in rp.path:
                padded[rid][node.time] = (node.x, node.y)
            # Pad endpoint occupancy
            for t in range(last_t + 1, max_t + 1):
                padded[rid][t] = last_pos

        # Check all pairs
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                rid_a = robot_ids[i]
                rid_b = robot_ids[j]
                if not padded[rid_a] or not padded[rid_b]:
                    continue

                # Check each time step
                all_times = sorted(
                    set(list(padded[rid_a].keys()) + list(padded[rid_b].keys()))
                )

                prev_pos_a = None
                prev_pos_b = None

                for t in all_times:
                    pos_a = padded[rid_a].get(t)
                    pos_b = padded[rid_b].get(t)

                    # Vertex conflict
                    if pos_a is not None and pos_b is not None and pos_a == pos_b:
                        conflicts.append(
                            Conflict(
                                conflict_type=ConflictType.VERTEX,
                                robot_ids=[rid_a, rid_b],
                                time=t,
                                position=pos_a,
                                description=f"Vertex conflict at {pos_a}, t={t}",
                            )
                        )

                    # Swap/edge conflict: A and B swap positions between t-1 and t
                    if (
                        pos_a is not None
                        and pos_b is not None
                        and prev_pos_a is not None
                        and prev_pos_b is not None
                    ):
                        if prev_pos_a == pos_b and prev_pos_b == pos_a:
                            conflicts.append(
                                Conflict(
                                    conflict_type=ConflictType.SWAP,
                                    robot_ids=[rid_a, rid_b],
                                    time=t,
                                    position=None,
                                    edge=(prev_pos_a, pos_a),
                                    description=f"Swap conflict between {rid_a} and {rid_b} at t={t}",
                                )
                            )

                    prev_pos_a = pos_a
                    prev_pos_b = pos_b

        # Start conflicts (t=0 same position)
        start_positions = {}
        for rid, rp in paths.items():
            if rp.success and rp.path:
                p0 = (rp.path[0].x, rp.path[0].y)
                if p0 in start_positions:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.START,
                            robot_ids=[start_positions[p0], rid],
                            time=0,
                            position=p0,
                            description=f"Start conflict at {p0}",
                        )
                    )
                else:
                    start_positions[p0] = rid

        # Goal conflicts (same final destination)
        goal_positions: Dict[Tuple[int, int], str] = {}
        for rid, rp in paths.items():
            if rp.success and rp.path:
                g_pos = (rp.path[-1].x, rp.path[-1].y)
                if g_pos in goal_positions:
                    conflicts.append(
                        Conflict(
                            conflict_type=ConflictType.GOAL,
                            robot_ids=[goal_positions[g_pos], rid],
                            time=-1,
                            position=g_pos,
                            description=f"Goal conflict at {g_pos}",
                        )
                    )
                else:
                    goal_positions[g_pos] = rid

        # Deduplicate conflicts
        return self._deduplicate(conflicts)

    def _deduplicate(self, conflicts: List[Conflict]) -> List[Conflict]:
        """Remove duplicate conflicts."""
        seen = set()
        result = []
        for c in conflicts:
            key = (
                c.conflict_type.value,
                tuple(sorted(c.robot_ids)),
                c.time,
                c.position,
            )
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result
