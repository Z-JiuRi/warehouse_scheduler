"""Spatiotemporal A* planner for single-robot path finding."""

import heapq
import time
from typing import List, Tuple, Set, Optional, Callable
from app.domain.path_models import TimedPosition, PathPlanResult, FailureReason


# Movement directions: up, down, left, right, wait
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]


class AStarPlanner:
    def __init__(
        self,
        width: int,
        height: int,
        static_obstacles: Set[Tuple[int, int]],
        max_timestep: int = 200,
        move_cost: float = 1.0,
        wait_cost: float = 1.0,
    ):
        self.width = width
        self.height = height
        self.static_obstacles = static_obstacles
        self.max_timestep = max_timestep
        self.move_cost = move_cost
        self.wait_cost = wait_cost

        # Dynamic obstacles: set of (x, y, t) blocked cells
        self._dynamic_obstacles: Set[Tuple[int, int, int]] = set()
        self._dynamic_edges: Set[
            Tuple[int, int, int, int, int]
        ] = set()  # (x1,y1,x2,y2,t)
        self._reservation_table = None

    def set_dynamic_obstacles(
        self,
        blocked_cells: Set[Tuple[int, int, int]],
    ):
        """Set time-specific blocked cells (from closures or reservations)."""
        self._dynamic_obstacles = blocked_cells

    def set_reservation_table(self, table):
        """Attach a reservation table for conflict-aware planning."""
        self._reservation_table = table

    def is_free(self, x: int, y: int, t: int, robot_id: str = "") -> bool:
        """Check if a cell is free at a given time."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if (x, y) in self.static_obstacles:
            return False
        if (x, y, t) in self._dynamic_obstacles:
            return False
        if self._reservation_table is not None:
            if self._reservation_table.is_vertex_reserved(x, y, t):
                owner = self._reservation_table.get_vertex_owner(x, y, t)
                if owner != robot_id:
                    return False
        return True

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        robot_id: str = "",
        start_time: int = 0,
    ) -> PathPlanResult:
        """
        Find a spacetime path from start to goal using A*.

        State: (x, y, t)
        g-cost: cumulative cost from start
        h-cost: Manhattan distance to goal
        """
        t0 = time.time()
        expanded = 0

        sx, sy = start
        gx, gy = goal

        # Check start validity
        if not self.is_free(sx, sy, start_time, robot_id):
            return PathPlanResult(
                success=False,
                failure_reason=FailureReason.START_BLOCKED,
                expanded_nodes=0,
                planning_time_ms=(time.time() - t0) * 1000,
            )

        # Check goal validity (at some time)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return PathPlanResult(
                success=False,
                failure_reason=FailureReason.OUT_OF_BOUNDS,
                expanded_nodes=0,
                planning_time_ms=(time.time() - t0) * 1000,
            )
        if (gx, gy) in self.static_obstacles:
            return PathPlanResult(
                success=False,
                failure_reason=FailureReason.GOAL_BLOCKED,
                expanded_nodes=0,
                planning_time_ms=(time.time() - t0) * 1000,
            )

        # A* search
        # heap: (f, g, x, y, t, parent_hash)
        open_set = []
        start_h = abs(sx - gx) + abs(sy - gy)
        # Use a unique entry counter to avoid tie-breaking issues
        counter = 0
        heapq.heappush(
            open_set,
            (start_h, 0.0, counter, sx, sy, start_time, None),
        )
        counter += 1

        # closed: (x, y, t) -> (parent, g)
        closed: dict = {}
        best_state = None

        while open_set:
            f, g, _, x, y, t, parent = heapq.heappop(open_set)

            if (x, y, t) in closed and closed[(x, y, t)][1] <= g:
                continue

            closed[(x, y, t)] = (parent, g)
            expanded += 1

            # Goal check: reached goal position
            if (x, y) == (gx, gy):
                best_state = (x, y, t, parent, g)
                break

            # Time limit
            if t >= self.max_timestep:
                continue

            # Try all moves
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                nt = t + 1

                # Edge case: wait
                if dx == 0 and dy == 0:
                    if not self.is_free(nx, ny, nt, robot_id):
                        continue
                    if self._reservation_table is not None:
                        if not self._reservation_table.is_wait_safe(
                            x, y, t, robot_id
                        ):
                            continue
                    step_cost = self.wait_cost
                else:
                    # Movement
                    if not self.is_free(nx, ny, nt, robot_id):
                        continue
                    # Check edge safety
                    if self._reservation_table is not None:
                        if not self._reservation_table.is_move_safe(
                            x, y, nx, ny, t, robot_id
                        ):
                            continue
                    step_cost = self.move_cost

                ng = g + step_cost
                if (nx, ny, nt) in closed and closed[(nx, ny, nt)][1] <= ng:
                    continue

                nh = abs(nx - gx) + abs(ny - gy)
                nf = ng + nh
                heapq.heappush(
                    open_set,
                    (nf, ng, counter, nx, ny, nt, (x, y, t)),
                )
                counter += 1

        elapsed_ms = (time.time() - t0) * 1000

        if best_state is None:
            return PathPlanResult(
                success=False,
                failure_reason=FailureReason.NO_PATH
                if t < self.max_timestep
                else FailureReason.MAX_TIMESTEP_EXCEEDED,
                expanded_nodes=expanded,
                planning_time_ms=elapsed_ms,
            )

        # Reconstruct path
        path = self._reconstruct(closed, best_state)
        return PathPlanResult(
            success=True,
            path=path,
            cost=best_state[4],
            expanded_nodes=expanded,
            planning_time_ms=elapsed_ms,
        )

    def _reconstruct(self, closed: dict, best_state) -> List[TimedPosition]:
        """Reconstruct path from closed set."""
        x, y, t, parent, g = best_state
        nodes = [(x, y, t)]
        while parent is not None:
            px, py, pt = parent
            nodes.append((px, py, pt))
            parent, _ = closed[(px, py, pt)]
        nodes.reverse()
        return [TimedPosition(x=n[0], y=n[1], time=n[2]) for n in nodes]
