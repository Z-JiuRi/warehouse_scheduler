"""Visualization: renders warehouse map and robot paths using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Set
from app.domain.map_models import WarehouseMap
from app.domain.path_models import TimedPosition, PathPlanResult
from app.domain.planning_state import PlanningState, BatchStatus


def _setup_cjk_font():
    """Configure matplotlib to use a CJK-compatible font if available."""
    import matplotlib.font_manager as fm
    # Try macOS/iOS fonts first, then common Linux CJK fonts
    candidates = [
        "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei",
        "Microsoft YaHei", "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # Fallback: suppress glyph warnings
    import warnings
    warnings.filterwarnings("ignore", message="Glyph.*missing from font")


_setup_cjk_font()


# Distinct colors for robots
ROBOT_COLORS = [
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
]

ROBOT_COLORS_LIGHT = [
    "#f5b7b1",
    "#a9dfbf",
    "#aed6f1",
    "#f9e79f",
    "#d7bde2",
]


def render_static_map(
    warehouse_map: WarehouseMap,
    ax: plt.Axes = None,
    title: str = "Warehouse Map",
):
    """Render the static warehouse map with obstacles, locations, and corridors."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    w, h = warehouse_map.width, warehouse_map.height

    # Draw grid
    for x in range(w + 1):
        ax.axvline(x, color="gray", linewidth=0.5, alpha=0.5)
    for y in range(h + 1):
        ax.axhline(y, color="gray", linewidth=0.5, alpha=0.5)

    # Draw obstacles
    for obs in warehouse_map.static_obstacles:
        for cx, cy in obs.cells:
            rect = patches.Rectangle(
                (cx, cy), 1, 1, linewidth=0, facecolor="#34495e", alpha=0.8
            )
            ax.add_patch(rect)

    # Draw locations
    for loc in warehouse_map.locations:
        # Facility cells
        for cx, cy in loc.facility_cells:
            rect = patches.Rectangle(
                (cx, cy), 1, 1, linewidth=1, facecolor="#bdc3c7", alpha=0.6
            )
            ax.add_patch(rect)
            ax.text(
                cx + 0.5,
                cy + 0.5,
                loc.name,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        # Entry cells
        for cx, cy in loc.entry_cells:
            rect = patches.Rectangle(
                (cx, cy), 1, 1, linewidth=1, edgecolor="#27ae60", facecolor="#a9dfbf", alpha=0.5, linestyle="--"
            )
            ax.add_patch(rect)

    # Draw corridors
    for corr in warehouse_map.corridors:
        for cx, cy in corr.cells:
            rect = patches.Rectangle(
                (cx, cy), 1, 1, linewidth=0.5, edgecolor="#2980b9",
                facecolor="#d4e6f1", alpha=0.7
            )
            ax.add_patch(rect)
        # Label corridor at its midpoint
        if corr.cells:
            mid_idx = len(corr.cells) // 2
            mx, my = corr.cells[mid_idx]
            ax.text(
                mx + 0.5, my + 0.5, corr.name,
                ha="center", va="center", fontsize=6,
                color="#2980b9", fontstyle="italic", alpha=0.9,
            )

    ax.set_xlim(-0.5, w + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # top-left origin

    return ax


def render_paths(
    warehouse_map: WarehouseMap,
    paths: Dict[str, PathPlanResult],
    title: str = "Robot Paths",
    block: bool = True,
):
    """Render robot paths on the warehouse map."""
    fig, ax = plt.subplots(figsize=(10, 10))
    render_static_map(warehouse_map, ax, title)

    for i, (rid, rp) in enumerate(paths.items()):
        if not rp.success or not rp.path:
            continue
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        xs = [p.x + 0.5 for p in rp.path]
        ys = [p.y + 0.5 for p in rp.path]

        # Draw path line
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.8, label=rid)

        # Draw start marker
        if rp.path:
            ax.scatter(
                rp.path[0].x + 0.5,
                rp.path[0].y + 0.5,
                color=color,
                marker="o",
                s=100,
                edgecolors="white",
                linewidth=1,
                zorder=5,
            )
            # Draw goal marker
            ax.scatter(
                rp.path[-1].x + 0.5,
                rp.path[-1].y + 0.5,
                color=color,
                marker="s",
                s=100,
                edgecolors="white",
                linewidth=1,
                zorder=5,
            )

            # Label start/time
            ax.annotate(
                f"{rid} start",
                (rp.path[0].x + 0.5, rp.path[0].y + 0.5),
                textcoords="offset points",
                xytext=(5, 10),
                fontsize=8,
                color=color,
            )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show(block=block)
    return fig, ax


def render_step_by_step(
    warehouse_map: WarehouseMap,
    paths: Dict[str, PathPlanResult],
    title: str = "Step-by-Step Animation",
    interval: int = 500,
    save_path: str = None,
):
    """Create an animated step-by-step visualization of robot paths."""
    fig, ax = plt.subplots(figsize=(10, 10))
    render_static_map(warehouse_map, ax, title)

    # Determine max time
    max_t = 0
    for rp in paths.values():
        if rp.success and rp.path:
            max_t = max(max_t, rp.path[-1].time)

    # Build position lookup: time -> {robot_id: (x, y)}
    time_positions: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for rid, rp in paths.items():
        if not rp.success or not rp.path:
            continue
        last_pos = (rp.path[-1].x, rp.path[-1].y)
        last_t = rp.path[-1].time
        for node in rp.path:
            if node.time not in time_positions:
                time_positions[node.time] = {}
            time_positions[node.time][rid] = (node.x, node.y)
        # Pad
        for t in range(last_t + 1, max_t + 1):
            if t not in time_positions:
                time_positions[t] = {}
            time_positions[t][rid] = last_pos

    # Create robot scatter artists
    robot_dots = {}
    robot_trails: Dict[str, List[Tuple[int, int]]] = {rid: [] for rid in paths}
    for i, rid in enumerate(paths):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        (dot,) = ax.plot([], [], "o", color=color, markersize=15, zorder=10)
        robot_dots[rid] = dot
        # Add text label
        ax.text(0, 0, "", fontsize=8, color=color, fontweight="bold")

    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    def init():
        for dot in robot_dots.values():
            dot.set_data([], [])
        time_text.set_text("")
        return list(robot_dots.values()) + [time_text]

    def update(frame):
        if frame not in time_positions:
            return list(robot_dots.values()) + [time_text]

        positions = time_positions[frame]
        for rid, dot in robot_dots.items():
            if rid in positions:
                x, y = positions[rid]
                dot.set_data([x + 0.5], [y + 0.5])
                robot_trails[rid].append((x, y))
            else:
                dot.set_data([], [])

        time_text.set_text(f"Time step: {frame}")
        return list(robot_dots.values()) + [time_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(max_t + 1),
        init_func=init,
        interval=interval,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()

    if save_path:
        ani.save(save_path, writer="pillow", fps=2)
        print(f"Animation saved to {save_path}")

    plt.show()
    return fig, ani


def render_summary(state: PlanningState):
    """Print a text summary of the planning result."""
    print("\n" + "=" * 60)
    print(f"  Planning Result: {state.request_id}")
    print("=" * 60)
    print(f"  Status: {state.status.value}")
    print(f"  Instruction: {state.original_instruction[:80]}...")
    print("-" * 60)

    for tr in state.task_results:
        status_icon = "✅" if tr.success else "❌"
        print(f"  {status_icon} {tr.robot_id}: ", end="")
        if tr.success and tr.path:
            print(
                f"path_len={len(tr.path)}, "
                f"makespan={tr.path[-1].time if tr.path else 'N/A'}"
            )
        else:
            print(f"FAILED ({tr.failure_reason})")

    if state.metrics:
        m = state.metrics
        print("-" * 60)
        print(f"  Success Rate: {m.planning_success_rate:.1%}")
        print(f"  Total Time: {m.total_planning_time_ms:.1f}ms")
        print(f"  Retries: {m.retry_count}")
        print(f"  A* Calls: {m.astar_call_count}")
        print(f"  Initial Conflicts: {m.initial_conflict_count}")
        print(f"  Final Conflicts: {m.final_conflict_count}")

    if state.warnings:
        print("-" * 60)
        for w in state.warnings:
            print(f"  ⚠️  {w}")

    if state.errors:
        print("-" * 60)
        for e in state.errors:
            print(f"  ❌ {e}")

    print("=" * 60 + "\n")
