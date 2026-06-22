"""Task parser agent: converts natural language to structured task batch using DeepSeek API."""

import json
import os
from typing import Tuple, List, Optional
from openai import OpenAI
from app.domain.task_models import RobotTask, TaskBatch
from app.domain.map_models import WarehouseMap
from app.services.location_resolver import LocationResolver
from app.services.robot_registry import RobotRegistry


PARSE_SYSTEM_PROMPT = """你是一个仓储机器人任务解析器。你的任务是将用户自然语言指令转换为结构化的机器人任务和临时封闭约束。

## 输出格式
你必须返回纯JSON，格式如下：
```json
{
  "tasks": [
    {
      "robot_id": "R1",
      "start": [0, 0],
      "goal_location_id": "loading_zone",
      "priority": 1
    }
  ],
  "constraints": [
    {
      "constraint_type": "closed_corridor",
      "target_id": "corridor_north",
      "start_time": 0,
      "end_time": null
    }
  ],
  "parse_warnings": [],
  "parse_errors": []
}
```

## 规则
1. **robot_id** 必须使用用户指定的ID（如R1, R2, R3），如果用户没指定，按出现顺序分配R1, R2...
2. **start** 坐标：如果用户明确指定了起点坐标（如"左上角"对应[0,0]），请填写。如果用户没有指定，填 null，系统会用机器人当前位置补全。
3. **goal_location_id** 必须使用地图中已定义的位置ID。根据用户提到的目标（如"装卸区"、"充电区"、"货架A"），匹配对应的location_id。
4. **priority** 数字越小优先级越高。如果用户指定了优先级就按用户说的，否则按指令中出现顺序（1, 2, 3...）。
5. **临时封闭**：如果用户提到"关闭某通道"或"封闭某区域"，生成对应的constraint。target_id填通道ID。
6. 如果用户的指令中有无法理解的内容，放到parse_warnings。如果有严重错误（如不存在的机器人），放到parse_errors。

## 可用位置
{locations_info}

## 可用通道
{corridors_info}

## 可用机器人
{robots_info}
"""


class TaskParserAgent:
    """Uses DeepSeek LLM to parse natural language into structured tasks."""

    def __init__(
        self,
        warehouse_map: WarehouseMap,
        robot_registry: RobotRegistry,
        api_config_path: str = None,
    ):
        self.map = warehouse_map
        self.registry = robot_registry
        self.location_resolver = LocationResolver(warehouse_map)

        if api_config_path is None:
            api_config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "configs",
                "api_config.json",
            )

        with open(api_config_path, "r", encoding="utf-8") as f:
            self.api_config = json.load(f)

        self.client = OpenAI(
            api_key=self.api_config["deepseek_api_key"],
            base_url=self.api_config.get(
                "deepseek_base_url", "https://api.deepseek.com/v1"
            ),
        )
        self.model = self.api_config.get("model", "deepseek-chat")
        self.temperature = self.api_config.get("temperature", 0.1)
        self.max_tokens = self.api_config.get("max_tokens", 2000)

    def parse(self, instruction: str) -> TaskBatch:
        """Parse natural language instruction into a TaskBatch."""
        # Build the system prompt with actual map data
        locations_info = self._build_locations_info()
        corridors_info = self._build_corridors_info()
        robots_info = self._build_robots_info()

        system_prompt = PARSE_SYSTEM_PROMPT.replace(
            "{locations_info}", locations_info
        ).replace(
            "{corridors_info}", corridors_info
        ).replace(
            "{robots_info}", robots_info
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            return TaskBatch(
                tasks=[],
                parse_errors=[f"LLM API call failed: {str(e)}"],
            )

        # Parse JSON from response
        parsed = self._extract_json(raw_output)
        if parsed is None:
            return TaskBatch(
                tasks=[],
                parse_errors=[f"Failed to parse LLM output as JSON: {raw_output[:200]}"],
            )

        # Build TaskBatch from parsed JSON
        return self._build_batch(parsed)

    def _build_locations_info(self) -> str:
        lines = []
        for loc in self.map.locations:
            lines.append(
                f"- location_id: {loc.location_id}, name: {loc.name}, "
                f"aliases: {loc.aliases}, entry_cells: {loc.entry_cells}"
            )
        return "\n".join(lines)

    def _build_corridors_info(self) -> str:
        lines = []
        for corr in self.map.corridors:
            lines.append(
                f"- corridor_id: {corr.corridor_id}, name: {corr.name}"
            )
        return "\n".join(lines)

    def _build_robots_info(self) -> str:
        lines = []
        for rid in self.registry.get_robot_ids():
            pos = self.registry.get_position(rid)
            lines.append(f"- robot_id: {rid}, current_position: {list(pos)}")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON object from text (may be wrapped in markdown code blocks)."""
        text = text.strip()
        # Remove markdown code block markers
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _build_batch(self, parsed: dict) -> TaskBatch:
        """Convert parsed JSON into a validated TaskBatch."""
        batch = TaskBatch()
        errors = parsed.get("parse_errors", [])
        warnings = parsed.get("parse_warnings", [])
        batch.parse_errors = list(errors)
        batch.parse_warnings = list(warnings)

        # Process tasks
        seen_priorities = set()
        for i, t_raw in enumerate(parsed.get("tasks", [])):
            robot_id = t_raw.get("robot_id", f"R{i+1}")

            # Get start position
            start_raw = t_raw.get("start")
            if start_raw is None or start_raw == [None, None]:
                # Use runtime position
                pos = self.registry.get_position(robot_id)
                if pos is None:
                    batch.parse_errors.append(
                        f"Robot {robot_id}: unknown and no start specified"
                    )
                    continue
                start = pos
            else:
                if isinstance(start_raw, list) and len(start_raw) == 2:
                    start = tuple(start_raw)
                else:
                    batch.parse_errors.append(
                        f"Robot {robot_id}: invalid start format"
                    )
                    continue

            # Validate start in bounds
            if not (0 <= start[0] < self.map.width and 0 <= start[1] < self.map.height):
                batch.parse_errors.append(
                    f"Robot {robot_id}: start {list(start)} out of bounds"
                )
                continue

            # Get goal
            goal_id = t_raw.get("goal_location_id", "")
            loc = self.location_resolver.resolve(goal_id)
            if loc is None:
                # Try to match by name
                for l in self.map.locations:
                    if l.name == goal_id or goal_id in l.aliases:
                        loc = l
                        goal_id = l.location_id
                        break
            if loc is None:
                batch.parse_errors.append(
                    f"Robot {robot_id}: unknown location '{goal_id}'"
                )
                continue

            # Get priority
            priority = t_raw.get("priority", i + 1)
            if not isinstance(priority, int) or priority < 1:
                priority = i + 1
            if priority in seen_priorities:
                priority = max(seen_priorities) + 1
            seen_priorities.add(priority)

            task = RobotTask(
                robot_id=robot_id,
                start=start,
                goal_location_id=goal_id,
                candidate_goals=list(loc.entry_cells),
                priority=priority,
            )
            batch.tasks.append(task)

        # Process constraints
        for c_raw in parsed.get("constraints", []):
            batch.runtime_constraints.append(c_raw)

        return batch
