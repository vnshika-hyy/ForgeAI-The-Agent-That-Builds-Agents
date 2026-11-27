from datetime import date

from .registry import ToolResult


class CurrentDateTool:
    def execute(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(success=True, output=date.today().isoformat())
