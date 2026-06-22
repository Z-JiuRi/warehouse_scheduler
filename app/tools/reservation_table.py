"""Reservation table: tracks vertex and edge reservations in spacetime."""

from typing import Set, Tuple, Dict, Optional


class ReservationTable:
    """Tracks which cells and edges are reserved at which times."""

    def __init__(self):
        # vertex_reservations: (x, y, t) -> robot_id
        self._vertex: Dict[Tuple[int, int, int], str] = {}
        # edge_reservations: (x1, y1, x2, y2, t) -> robot_id  (moving from pos1 to pos2 at time t)
        self._edge: Dict[Tuple[int, int, int, int, int], str] = {}

    def reserve_vertex(self, x: int, y: int, t: int, robot_id: str):
        self._vertex[(x, y, t)] = robot_id

    def reserve_edge(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        t: int,
        robot_id: str,
    ):
        """Reserve the edge at time t (the move happens between t and t+1)."""
        self._edge[(from_x, from_y, to_x, to_y, t)] = robot_id

    def is_vertex_reserved(self, x: int, y: int, t: int) -> bool:
        return (x, y, t) in self._vertex

    def is_edge_reserved(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        t: int,
    ) -> bool:
        return (from_x, from_y, to_x, to_y, t) in self._edge

    def get_vertex_owner(self, x: int, y: int, t: int) -> Optional[str]:
        return self._vertex.get((x, y, t))

    def reserve_path(
        self,
        path: list,
        robot_id: str,
        max_t: int = None,
    ):
        """
        Reserve all vertices and edges for a path.
        Path is a list of TimedPosition or (x, y, t) tuples.
        For the final position, reserve all future times up to max_t.
        """
        if not path:
            return

        # Reserve vertices
        for node in path:
            if hasattr(node, 'as_tuple'):
                x, y, t = node.as_tuple()
            else:
                x, y, t = node
            self.reserve_vertex(x, y, t, robot_id)

        # Reserve edges (between consecutive positions)
        for i in range(len(path) - 1):
            if hasattr(path[i], 'as_tuple'):
                x1, y1, t1 = path[i].as_tuple()
                x2, y2, t2 = path[i + 1].as_tuple()
            else:
                x1, y1, t1 = path[i]
                x2, y2, t2 = path[i + 1]
            # Moving from (x1,y1) to (x2,y2): reserve edge at time t1
            if (x1, y1) != (x2, y2):
                self.reserve_edge(x1, y1, x2, y2, t1, robot_id)

        # Reserve endpoint occupancy for all future times up to max_t
        if max_t is not None:
            last = path[-1]
            if hasattr(last, 'as_tuple'):
                lx, ly, lt = last.as_tuple()
            else:
                lx, ly, lt = last
            for t in range(lt + 1, max_t + 1):
                self.reserve_vertex(lx, ly, t, robot_id)

    def is_move_safe(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        from_t: int,
        robot_id: str,
    ) -> bool:
        """
        Check if moving from (from_x,from_y) at time from_t to (to_x,to_y) at time from_t+1
        is safe (no vertex or edge conflicts).
        """
        # Destination vertex
        if self.is_vertex_reserved(to_x, to_y, from_t + 1):
            owner = self.get_vertex_owner(to_x, to_y, from_t + 1)
            if owner != robot_id:
                return False

        # Edge: moving A->B while B->A at same time = swap conflict
        if self.is_edge_reserved(to_x, to_y, from_x, from_y, from_t):
            return False

        # Edge: this move's edge
        if (from_x, from_y) != (to_x, to_y):
            if self.is_edge_reserved(from_x, from_y, to_x, to_y, from_t):
                return False

        return True

    def is_wait_safe(self, x: int, y: int, t: int, robot_id: str) -> bool:
        """Check if waiting at (x,y) at time t (i.e. occupying same cell at t and t+1) is safe."""
        owner = self.get_vertex_owner(x, y, t + 1)
        return owner is None or owner == robot_id

    def clear(self):
        self._vertex.clear()
        self._edge.clear()
