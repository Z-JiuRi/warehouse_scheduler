"""Location resolver: resolves semantic location names/aliases to entry cells."""

from typing import List, Tuple, Optional
from app.domain.map_models import WarehouseMap, Location


class LocationResolver:
    def __init__(self, warehouse_map: WarehouseMap):
        self.map = warehouse_map

    def resolve(self, identifier: str) -> Optional[Location]:
        """Find a location by ID or alias."""
        return self.map.find_location(identifier)

    def get_candidate_goals(self, identifier: str) -> List[Tuple[int, int]]:
        """Get all candidate entry cells for a location."""
        loc = self.resolve(identifier)
        if loc is None:
            return []
        return list(loc.entry_cells)

    def get_free_entry_cells(
        self,
        identifier: str,
        occupied: set,
    ) -> List[Tuple[int, int]]:
        """Get entry cells that are not currently occupied."""
        candidates = self.get_candidate_goals(identifier)
        return [c for c in candidates if c not in occupied]

    def select_best_goal(
        self,
        identifier: str,
        start: Tuple[int, int],
        occupied: set,
        is_walkable_fn=None,
    ) -> Optional[Tuple[int, int]]:
        """
        Select the best entry cell: free, reachable via Manhattan heuristic.
        """
        free = self.get_free_entry_cells(identifier, occupied)
        if not free:
            return None
        # Prefer closer cell by Manhattan distance
        free.sort(key=lambda g: abs(g[0] - start[0]) + abs(g[1] - start[1]))
        return free[0]
