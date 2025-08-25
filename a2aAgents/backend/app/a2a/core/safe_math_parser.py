"""
Safe Mathematical Expression Parser
Provides secure evaluation of mathematical expressions without using eval()
"""

import ast
import operator
import math
from typing import Dict, Any, Optional, Union
import re


class SafeMathParser:
    """
    Safe parser for mathematical expressions that prevents code injection
    """

    # Define allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Define allowed functions
    FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'round': round,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'pow': pow,
        'floor': math.floor,
        'ceil': math.ceil,
        'degrees': math.degrees,
        'radians': math.radians,
    }

    # Define allowed constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }

    def __init__(self, allowed_names: Optional[Dict[str, Any]] = None):
        """
        Initialize the safe math parser

        Args:
            allowed_names: Dictionary of allowed variable names and their values
        """
        self.allowed_names = allowed_names or {}
        # Combine allowed names with constants and functions
        self.namespace = {
            **self.CONSTANTS,
            **self.FUNCTIONS,
            **self.allowed_names
        }

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Result of the evaluation

        Raises:
            ValueError: If the expression contains unsafe operations
            SyntaxError: If the expression is malformed
        """
        # Sanitize expression
        expression = self._sanitize_expression(expression)

        # Parse the expression into an AST
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression: {e}")

        # Validate the AST for safety
        self._validate_ast(tree)

        # Evaluate the expression
        return self._eval_node(tree.body)

    def _sanitize_expression(self, expression: str) -> str:
        """
        Sanitize the expression to prevent injection attacks

        Args:
            expression: Raw expression string

        Returns:
            Sanitized expression
        """
        # Remove any potentially dangerous characters
        # Allow only alphanumeric, math operators, parentheses, dots, and spaces
        allowed_chars = re.compile(r'^[a-zA-Z0-9\s\+\-\*\/\^\%\(\)\.\,]+$')

        if not allowed_chars.match(expression):
            raise ValueError(f"Expression contains invalid characters")

        # Replace ^ with ** for exponentiation (common math notation)
        expression = expression.replace('^', '**')

        return expression.strip()

    def _validate_ast(self, tree: ast.AST) -> None:
        """
        Validate that the AST only contains safe operations

        Args:
            tree: AST to validate

        Raises:
            ValueError: If unsafe operations are detected
        """
        for node in ast.walk(tree):
            # Check for forbidden node types
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Exec,
                               ast.FunctionDef, ast.ClassDef, ast.With,
                               ast.Raise, ast.Try, ast.Assert, ast.Delete,
                               ast.Global, ast.Nonlocal, ast.Pass, ast.Break,
                               ast.Continue, ast.Lambda, ast.Yield, ast.YieldFrom,
                               ast.Await, ast.AsyncFunctionDef, ast.AsyncWith,
                               ast.AsyncFor)):
                raise ValueError(f"Forbidden operation: {type(node).__name__}")

            # Check for attribute access (prevents things like __builtins__.eval)
            if isinstance(node, ast.Attribute):
                raise ValueError("Attribute access is not allowed")

            # Check for list/dict comprehensions
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                raise ValueError("Comprehensions are not allowed")

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate an AST node

        Args:
            node: AST node to evaluate

        Returns:
            Result of evaluation

        Raises:
            ValueError: If the node type is not supported
        """
        if isinstance(node, ast.Constant):
            # Python 3.8+ uses ast.Constant
            return node.value

        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility
            return node.n

        elif isinstance(node, ast.Name):
            # Variable lookup
            if node.id not in self.namespace:
                raise ValueError(f"Undefined variable: {node.id}")
            return self.namespace[node.id]

        elif isinstance(node, ast.BinOp):
            # Binary operation
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)

            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            # Prevent division by zero
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ValueError("Division by zero")

            return self.OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            operand = self._eval_node(node.operand)
            op_type = type(node.op)

            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

            return self.OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            # Function call
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")

            func_name = node.func.id
            if func_name not in self.namespace:
                raise ValueError(f"Unknown function: {func_name}")

            # Evaluate arguments
            args = [self._eval_node(arg) for arg in node.args]

            # No keyword arguments allowed for simplicity
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")

            # Call the function
            func = self.namespace[func_name]
            try:
                return func(*args)
            except Exception as e:
                raise ValueError(f"Error calling {func_name}: {e}")

        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")


def safe_eval_math(expression: str, variables: Optional[Dict[str, Any]] = None) -> Union[int, float]:
    """
    Convenience function to safely evaluate a mathematical expression

    Args:
        expression: Mathematical expression to evaluate
        variables: Optional dictionary of variables

    Returns:
        Result of the evaluation
    """
    parser = SafeMathParser(allowed_names=variables)
    return parser.evaluate(expression)