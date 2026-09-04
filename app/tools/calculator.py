from __future__ import annotations

import ast
import operator
from typing import Any

from pydantic import Field

from app.tools.registry import (
    ToolArguments,
    ToolDefinition,
    ToolPermission,
)


# Arguments được phép truyền vào calculator.
class CalculatorArguments(ToolArguments):
    expression: str = Field(min_length=1, max_length=200)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


# Nhận một node AST đã parse; trả kết quả số sau khi chỉ cho phép toán tử an toàn.
def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
        if abs(value) > 1_000_000_000_000_000:
            raise ValueError("Result is too large")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Expression contains an unsupported operation")


# Nhận biểu thức số dạng chuỗi; trả biểu thức gốc và kết quả tính toán chính xác.
def calculate(expression: str) -> dict[str, Any]:
    tree = ast.parse(expression.replace("×", "*").replace("÷", "/"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expression is too complex")
    return {"expression": expression, "result": _evaluate(tree)}


# Nhận thời gian chờ tối đa; trả định nghĩa calculator để đăng ký vào ToolRegistry.
def build_calculator_tool(timeout_seconds: float) -> ToolDefinition:
    return ToolDefinition(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression accurately. Use this tool whenever "
            "the user asks for arithmetic; do not calculate mentally. Only numeric "
            "operators are accepted, never code or variable names."
        ),
        input_model=CalculatorArguments,
        handler=calculate,
        timeout_seconds=timeout_seconds,
        permission=ToolPermission.COMPUTE,
    )
