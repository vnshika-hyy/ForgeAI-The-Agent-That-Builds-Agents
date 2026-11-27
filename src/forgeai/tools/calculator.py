import ast
import math
import operator

from .registry import ToolResult


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_ABS_RESULT = 1e100


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > _MAX_ABS_RESULT:
            raise ValueError("Numeric value is outside the supported range.")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude must not exceed 100.")
        result = _BINARY_OPERATORS[type(node.op)](left, right)
        if type(result) not in (int, float):
            raise ValueError("Calculation must produce a real number.")
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Calculation result is not finite.")
        if abs(result) > _MAX_ABS_RESULT:
            raise ValueError("Calculation result is outside the supported range.")
        return result
    raise ValueError("Only numeric arithmetic expressions are allowed.")


class CalculatorTool:
    def execute(self, arguments: dict[str, object]) -> ToolResult:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult(
                success=False,
                output="",
                error="Provide a non-empty arithmetic expression.",
            )
        if len(expression) > 256:
            return ToolResult(
                success=False,
                output="",
                error="Expression exceeds the 256-character limit.",
            )
        try:
            tree = ast.parse(expression, mode="eval")
            result = _evaluate(tree)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(success=False, output="", error=str(exc))
        return ToolResult(success=True, output=str(result))
