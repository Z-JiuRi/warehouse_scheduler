"""Replanning agent: diagnoses conflicts and produces structured replan decisions."""

from typing import List, Dict, Optional
from app.domain.conflict_models import Conflict, ConflictType
from app.domain.path_models import PathPlanResult
from app.domain.planning_state import ReplanDecision


class ReplanningAgent:
    """Analyzes conflicts and decides on replanning actions."""

    def decide(
        self,
        conflicts: List[Conflict],
        paths: Dict[str, PathPlanResult],
        priority_order: List[str],
        retry_count: int,
    ) -> ReplanDecision:
        """
        Diagnose conflicts and produce a structured replan decision.

        The three-level strategy:
        1. retry_count == 0: fix_high_priority — keep high-priority paths, replan low
        2. retry_count == 1: add_time_window — add wait constraints around conflict points
        3. retry_count == 2: adjust_priority — swap priorities of conflicting robots
        """
        if not conflicts:
            return ReplanDecision(
                failure_category="no_conflict",
                action="none",
                explanation="No conflicts to resolve",
            )

        # Categorize conflicts
        vertex_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.VERTEX]
        swap_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.SWAP]
        goal_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.GOAL]
        start_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.START]

        # Collect all affected robots
        affected = set()
        for c in conflicts:
            affected.update(c.robot_ids)
        affected_robots = list(affected)

        if retry_count == 0:
            return self._strategy_fix_high_priority(
                conflicts, paths, priority_order, affected_robots
            )
        elif retry_count == 1:
            return self._strategy_add_time_window(
                conflicts, paths, priority_order, affected_robots
            )
        else:  # retry_count >= 2
            return self._strategy_adjust_priority(
                conflicts, paths, priority_order, affected_robots
            )

    def _strategy_fix_high_priority(
        self,
        conflicts: List[Conflict],
        paths: Dict[str, PathPlanResult],
        priority_order: List[str],
        affected_robots: List[str],
    ) -> ReplanDecision:
        """Keep high-priority robot paths fixed, replan low-priority robots."""
        # Find the lowest-priority robot among conflicting pairs
        robots_to_replan = []
        for c in conflicts:
            low_prio = None
            lowest_rank = -1
            for rid in c.robot_ids:
                try:
                    rank = priority_order.index(rid)
                except ValueError:
                    rank = len(priority_order)
                if rank > lowest_rank:
                    lowest_rank = rank
                    low_prio = rid
            if low_prio and low_prio not in robots_to_replan:
                robots_to_replan.append(low_prio)

        # If a robot appears in multiple conflict pairs as low-priority, still replan it
        # If no clear low-priority robot, pick all affected except the highest
        if not robots_to_replan:
            highest = priority_order[0] if priority_order else affected_robots[0]
            robots_to_replan = [r for r in affected_robots if r != highest]

        return ReplanDecision(
            failure_category="priority_conflict",
            affected_robots=affected_robots,
            robot_to_replan=robots_to_replan,
            action="fix_high_priority",
            explanation=f"Fixing high-priority paths, replanning {robots_to_replan}",
        )

    def _strategy_add_time_window(
        self,
        conflicts: List[Conflict],
        paths: Dict[str, PathPlanResult],
        priority_order: List[str],
        affected_robots: List[str],
    ) -> ReplanDecision:
        """Add time-window constraints around conflict points to force waiting/rerouting."""
        constraints = []
        robots_to_replan = []

        for c in conflicts:
            if c.position is not None:
                # Block the conflict cell for a time window around the conflict
                constraints.append({
                    "type": "vertex_window",
                    "position": list(c.position),
                    "time_start": max(0, c.time - 2),
                    "time_end": c.time + 3,
                })

            # Find which robots to replan (lower priority ones)
            for rid in c.robot_ids:
                if rid not in robots_to_replan:
                    try:
                        rank = priority_order.index(rid)
                    except ValueError:
                        rank = len(priority_order)
                    # Replan robots that aren't the absolute top priority
                    if rank > 0 or len(c.robot_ids) > 1:
                        robots_to_replan.append(rid)

        return ReplanDecision(
            failure_category="spatial_conflict",
            affected_robots=affected_robots,
            robot_to_replan=list(set(robots_to_replan)),
            action="add_time_window",
            constraints=constraints,
            explanation=f"Adding time-window constraints to force waiting/rerouting",
        )

    def _strategy_adjust_priority(
        self,
        conflicts: List[Conflict],
        paths: Dict[str, PathPlanResult],
        priority_order: List[str],
        affected_robots: List[str],
    ) -> ReplanDecision:
        """Swap local priorities of conflicting robots."""
        priority_changes = {}
        robots_to_replan = []

        for c in conflicts:
            if len(c.robot_ids) == 2:
                a, b = c.robot_ids
                try:
                    rank_a = priority_order.index(a)
                except ValueError:
                    rank_a = len(priority_order)
                try:
                    rank_b = priority_order.index(b)
                except ValueError:
                    rank_b = len(priority_order)
                # Swap: give higher priority to the one that was lower
                if rank_a > rank_b:
                    priority_changes[a] = rank_b
                    priority_changes[b] = rank_a
                else:
                    priority_changes[b] = rank_a
                    priority_changes[a] = rank_b
                robots_to_replan.extend(c.robot_ids)

        return ReplanDecision(
            failure_category="swap_priority",
            affected_robots=affected_robots,
            robot_to_replan=list(set(robots_to_replan)),
            action="adjust_priority",
            priority_changes=priority_changes,
            explanation=f"Swapping priorities for conflicting robots",
        )
