"""Map loader: loads and validates the warehouse map JSON."""

import json
import os
from typing import Tuple, List
from app.domain.map_models import (
    WarehouseMap,
    CoordinateSystem,
    MovementRules,
    Location,
    StaticObstacle,
    Corridor,
)


class MapLoader:
    def __init__(self, map_path: str = None):
        if map_path is None:
            map_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs",
                "warehouse_map.json",
            )
        self.map_path = map_path

    def load(self) -> Tuple[WarehouseMap, List[str]]:
        """Load and validate the map. Returns (map, errors)."""
        errors = []

        try:
            with open(self.map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            errors.append(f"Map file not found: {self.map_path}")
            return None, errors
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in map file: {e}")
            return None, errors

        errors.extend(self._validate_schema(raw))
        if errors:
            return None, errors

        try:
            wmap = self._parse(raw)
        except Exception as e:
            errors.append(f"Failed to parse map: {e}")
            return None, errors

        wmap.build_indices()
        errors.extend(self._validate_semantics(wmap))
        return wmap, errors

    def _validate_schema(self, raw: dict) -> List[str]:
        errors = []
        if "map" not in raw:
            errors.append("Missing 'map' key")
            return errors
        m = raw["map"]
        if "map_id" not in m:
            errors.append("Missing map.map_id")
        if "width" not in m or not isinstance(m.get("width"), int) or m["width"] < 1:
            errors.append("map.width must be a positive integer")
        if "height" not in m or not isinstance(m.get("height"), int) or m["height"] < 1:
            errors.append("map.height must be a positive integer")
        return errors

    def _parse(self, raw: dict) -> WarehouseMap:
        m = raw["map"]
        coord = raw.get("coordinate_system", {})
        mov = raw.get("movement", {})

        wmap = WarehouseMap(
            map_id=m["map_id"],
            name=m.get("name", ""),
            width=m["width"],
            height=m["height"],
            coordinate_system=CoordinateSystem(
                format=coord.get("format", "[x, y]"),
                origin=coord.get("origin", "top_left"),
                x_direction=coord.get("x_direction", "right"),
                y_direction=coord.get("y_direction", "down"),
            ),
            movement=MovementRules(
                allow_up=mov.get("allow_up", True),
                allow_down=mov.get("allow_down", True),
                allow_left=mov.get("allow_left", True),
                allow_right=mov.get("allow_right", True),
                allow_wait=mov.get("allow_wait", True),
                allow_diagonal=mov.get("allow_diagonal", False),
                move_cost=mov.get("move_cost", 1.0),
                wait_cost=mov.get("wait_cost", 1.0),
            ),
        )

        for obs in raw.get("static_obstacles", []):
            cells = [tuple(c) for c in obs.get("cells", [])]
            wmap.static_obstacles.append(
                StaticObstacle(
                    obstacle_id=obs.get("obstacle_id", ""),
                    type=obs.get("type", ""),
                    cells=cells,
                )
            )

        location_ids = set()
        all_aliases = set()
        for loc in raw.get("locations", []):
            lid = loc.get("location_id", "")
            if lid in location_ids:
                raise ValueError(f"Duplicate location_id: {lid}")
            location_ids.add(lid)
            for alias in loc.get("aliases", []):
                al = alias.lower()
                if al in all_aliases:
                    raise ValueError(f"Duplicate alias: {alias}")
                all_aliases.add(al)

            wmap.locations.append(
                Location(
                    location_id=lid,
                    name=loc.get("name", ""),
                    aliases=loc.get("aliases", []),
                    type=loc.get("type", ""),
                    facility_cells=[tuple(c) for c in loc.get("facility_cells", [])],
                    entry_cells=[tuple(c) for c in loc.get("entry_cells", [])],
                    capacity=loc.get("capacity", 1),
                )
            )

        for corr in raw.get("corridors", []):
            wmap.corridors.append(
                Corridor(
                    corridor_id=corr.get("corridor_id", ""),
                    name=corr.get("name", ""),
                    cells=[tuple(c) for c in corr.get("cells", [])],
                    direction=corr.get("direction"),
                    capacity=corr.get("capacity"),
                )
            )

        return wmap

    def _validate_semantics(self, wmap: WarehouseMap) -> List[str]:
        errors = []
        w, h = wmap.width, wmap.height

        # Check obstacles are in bounds
        for obs in wmap.static_obstacles:
            for cx, cy in obs.cells:
                if not (0 <= cx < w and 0 <= cy < h):
                    errors.append(
                        f"Obstacle {obs.obstacle_id} cell [{cx},{cy}] out of bounds"
                    )

        # Check locations
        for loc in wmap.locations:
            for cx, cy in loc.facility_cells + loc.entry_cells:
                if not (0 <= cx < w and 0 <= cy < h):
                    errors.append(
                        f"Location {loc.location_id} cell [{cx},{cy}] out of bounds"
                    )
            for ex, ey in loc.entry_cells:
                if wmap.is_obstacle(ex, ey):
                    errors.append(
                        f"Location {loc.location_id} entry [{ex},{ey}] is an obstacle"
                    )
            if loc.capacity > len(loc.entry_cells):
                errors.append(
                    f"Location {loc.location_id} capacity {loc.capacity} exceeds entry count {len(loc.entry_cells)}"
                )

        # Check corridors
        for corr in wmap.corridors:
            for cx, cy in corr.cells:
                if not (0 <= cx < w and 0 <= cy < h):
                    errors.append(
                        f"Corridor {corr.corridor_id} cell [{cx},{cy}] out of bounds"
                    )
                if wmap.is_obstacle(cx, cy):
                    errors.append(
                        f"Corridor {corr.corridor_id} cell [{cx},{cy}] is an obstacle"
                    )

        return errors
