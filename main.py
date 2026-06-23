#!/usr/bin/env python3
"""Main entry point for the Warehouse Robot Scheduling System."""

import sys
import os
import json
import argparse

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

# Add local vendored packages (langgraph etc.)
_venv_packages = os.path.join(_project_root, ".venv_packages")
if os.path.isdir(_venv_packages):
    sys.path.insert(0, _venv_packages)

from app.orchestration.workflow import Workflow
from app.visualization.renderer import render_paths, render_step_by_step, render_summary, render_static_map


def main():
    parser = argparse.ArgumentParser(
        description="智能仓储机器人调度系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --instruct "R1从左上角前往装卸区，R2前往货架B，R3前往充电区"
  python main.py --instruct "关闭北侧通道，R1前往货架A，R2前往充电区"
  python main.py --instruct "R1去装卸区，R2去货架A，R3去货架B，R4去充电区"
  python main.py --structured tasks.json
  python main.py --show-map
  python main.py --show-map --map my_map.json
        """,
    )
    parser.add_argument(
        "--instruct", "-i", type=str, help="Natural language instruction"
    )
    parser.add_argument(
        "--structured", "-s", type=str, help="Path to structured task JSON file"
    )
    parser.add_argument(
        "--map", "-m", type=str, default=None, help="Path to warehouse map JSON"
    )
    parser.add_argument(
        "--runtime", "-r", type=str, default=None, help="Path to runtime state JSON"
    )
    parser.add_argument(
        "--api-config", "-a", type=str, default=None, help="Path to API config JSON"
    )
    parser.add_argument(
        "--max-timestep", "-t", type=int, default=200, help="Maximum planning timestep"
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization"
    )
    parser.add_argument(
        "--animate", action="store_true", help="Show step-by-step animation"
    )
    parser.add_argument(
        "--save-animation", type=str, default=None, help="Save animation to file (GIF)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "--show-map", action="store_true", help="Only display the warehouse map and exit"
    )

    args = parser.parse_args()

    # --show-map mode: load map and display it, then exit
    if args.show_map:
        from app.services.map_loader import MapLoader
        loader = MapLoader(args.map) if args.map else MapLoader()
        wmap, errors = loader.load()
        if wmap is None:
            print("ERROR: Failed to load map")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        import matplotlib.pyplot as plt
        render_static_map(wmap, title=f"{wmap.name} ({wmap.width}×{wmap.height})")
        plt.show()
        return

    if not args.instruct and not args.structured:
        parser.print_help()
        print("\nError: --instruct or --structured is required")
        sys.exit(1)

    # Initialize workflow
    print("Loading map and runtime state...")
    workflow = Workflow(
        map_path=args.map,
        runtime_path=args.runtime,
        api_config_path=args.api_config,
        max_timestep=args.max_timestep,
    )

    if workflow.warehouse_map is None:
        print("ERROR: Failed to load warehouse map")
        for err in workflow.map_errors:
            print(f"  - {err}")
        sys.exit(1)

    if workflow.runtime_errors:
        print("WARNING: Runtime state errors:")
        for err in workflow.runtime_errors:
            print(f"  - {err}")

    # Run planning
    if args.instruct:
        print(f"\nProcessing instruction: {args.instruct}")
        state = workflow.run(args.instruct)
    else:
        with open(args.structured, "r", encoding="utf-8") as f:
            tasks_json = json.load(f)
        print(f"\nProcessing structured tasks from: {args.structured}")
        state = workflow.run_structured(tasks_json)

    # Print summary
    render_summary(state)

    # Render visualization
    if state.current_paths:
        if args.animate:
            render_step_by_step(
                workflow.warehouse_map,
                state.current_paths,
                title=f"Robot Paths - {state.status.value}",
                save_path=args.save_animation,
            )
        elif not args.no_viz:
            render_paths(
                workflow.warehouse_map,
                state.current_paths,
                title=f"Robot Paths - {state.status.value}",
            )

    # Save output
    if args.output:
        output_data = {
            "request_id": state.request_id,
            "batch_status": state.status.value,
            "original_instruction": state.original_instruction,
            "tasks": [
                {
                    "robot_id": tr.robot_id,
                    "success": tr.success,
                    "path": [
                        {"x": p.x, "y": p.y, "time": p.time} for p in tr.path
                    ],
                    "failure_reason": tr.failure_reason,
                    "replanned": tr.replanned,
                }
                for tr in state.task_results
            ],
            "warnings": state.warnings,
            "errors": state.errors,
        }
        if state.metrics:
            output_data["metrics"] = {
                "total_task_count": state.metrics.total_task_count,
                "planned_task_count": state.metrics.planned_task_count,
                "planning_failed_task_count": state.metrics.planning_failed_task_count,
                "planning_success_rate": state.metrics.planning_success_rate,
                "total_planning_time_ms": state.metrics.total_planning_time_ms,
                "replanning_triggered": state.metrics.replanning_triggered,
                "retry_count": state.metrics.retry_count,
                "astar_call_count": state.metrics.astar_call_count,
            }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
