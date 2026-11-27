from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: str
    error: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    use_cases: list[str] = []
    input_schema: dict[str, object] = {}
    external_network_required: bool = False


class ToolRegistry:
    def __init__(
        self,
        tools: dict[str, ToolDefinition] | None = None,
        executors: dict[str, object] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else {
            "csv_lookup": ToolDefinition(
                name="csv_lookup",
                description="Looks up values and rows from supplied CSV/tabular data.",
                use_cases=["Find matching rows", "Summarize tabular records"],
            ),
            "calculator": ToolDefinition(
                name="calculator",
                description="Performs deterministic arithmetic calculations.",
                use_cases=["Calculate totals", "Perform unit conversions"],
            ),
            "current_date": ToolDefinition(
                name="current_date",
                description="Returns the current date or time.",
                use_cases=["Date-sensitive answers", "Determine relative dates"],
            ),
            "weather": ToolDefinition(
                name="weather",
                description="Retrieves current or forecast weather information.",
                use_cases=["Look up weather conditions"],
                external_network_required=True,
            ),
        }
        if executors is None and tools is None:
            from .calculator import CalculatorTool
            from .csv_lookup import CSVLookupTool
            from .current_date import CurrentDateTool
            from .weather import WeatherTool

            executors = {
                "csv_lookup": CSVLookupTool(),
                "calculator": CalculatorTool(),
                "current_date": CurrentDateTool(),
                "weather": WeatherTool(),
            }
        self._executors = executors or {}

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def descriptions(self) -> str:
        return "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values()
        )

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        allowed_tools: list[str],
        *,
        csv_files: dict[str, str] | None = None,
    ) -> ToolResult:
        if tool_name not in allowed_tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' is not enabled for this agent.",
            )
        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' is not registered.",
            )
        executor = self._executors.get(tool_name)
        if executor is None or not hasattr(executor, "execute"):
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' has no runtime implementation.",
            )
        if tool_name == "csv_lookup":
            return executor.execute(arguments, csv_files or {})
        return executor.execute(arguments)


DEFAULT_TOOL_REGISTRY = ToolRegistry()
