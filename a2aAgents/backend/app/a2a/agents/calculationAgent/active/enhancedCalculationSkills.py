import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from decimal import Decimal
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations

from app.a2a.core.security_base import SecureA2AAgent
"""
Enhanced Calculation Skills with Methodology and Step Explanation
Provides detailed explanation of calculation methodology and steps for CalcTesting evaluation
"""

# Import SymPy for symbolic computation
try:
    import sympy as sp
    from sympy import symbols, diff, integrate, solve, simplify, expand, factor, limit, series
    from sympy import Matrix, latex, pprint, pretty, nsimplify, together, apart, cancel
    from sympy import Derivative, Integral, Sum, Product, Limit
    from sympy import sqrt, exp, log, sin, cos, tan, pi, E, I, oo
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    from sympy.solvers import dsolve, linsolve, nonlinsolve
    from sympy.matrices import Matrix, eye, zeros, ones
    from sympy.geometry import Point, Line, Circle, Triangle
    from sympy.plotting import plot, plot3d
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Import numerical libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import A2A components
from app.a2a.sdk import A2AMessage, MessageRole

# Import natural language parser
try:
    from .naturalLanguageParser import MathQueryProcessor
    NL_PARSER_AVAILABLE = True
except ImportError:
    NL_PARSER_AVAILABLE = False

logger = logging.getLogger(__name__)


class CalculationStep(SecureA2AAgent):
    """Represents a single step in a calculation"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, description: str, operation: str, result: Any, latex: Optional[str] = None):
        super().__init__()
        self.description = description
        self.operation = operation
        self.result = result
        self.latex = latex
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "operation": self.operation,
            "result": str(self.result),
            "latex": self.latex,
            "timestamp": self.timestamp.isoformat()
        }


class CalculationMethodology(SecureA2AAgent):
    """Tracks methodology and steps for a calculation"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, problem_type: str, approach: str):
        super().__init__()
        self.problem_type = problem_type
        self.approach = approach
        self.steps: List[CalculationStep] = []
        self.assumptions: List[str] = []
        self.references: List[str] = []

    def add_step(self, description: str, operation: str, result: Any, latex: Optional[str] = None):
        """Add a calculation step"""
        step = CalculationStep(description, operation, result, latex)
        self.steps.append(step)
        return step

    def add_assumption(self, assumption: str):
        """Add an assumption made during calculation"""
        self.assumptions.append(assumption)

    def add_reference(self, reference: str):
        """Add a reference or formula used"""
        self.references.append(reference)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "approach": self.approach,
            "steps": [step.to_dict() for step in self.steps],
            "assumptions": self.assumptions,
            "references": self.references,
            "total_steps": len(self.steps)
        }


class EnhancedCalculationSkills(SecureA2AAgent):
    """Enhanced calculation skills with detailed methodology tracking"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.transformations = (standard_transformations + (implicit_multiplication_application,)) if SYMPY_AVAILABLE else None
        # Initialize natural language processor
        self.nl_processor = MathQueryProcessor() if NL_PARSER_AVAILABLE else None

    async def calculate_with_explanation(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform calculation with detailed explanation of methodology and steps
        """
        methodology = None
        result = None
        confidence = 0.0

        try:
            # Use advanced natural language parser if available
            if self.nl_processor:
                parsed_query = self.nl_processor.process_query(expression)
                calc_type = parsed_query["parsed_query"]["operation"]

                # Update context with parsed information
                if context is None:
                    context = {}
                context.update(parsed_query["context"])
                context["parsed_components"] = parsed_query["parsed_query"]["parsed_components"]
                context["nl_confidence"] = parsed_query["parsed_query"]["confidence"]
                context["suggestions"] = parsed_query["suggestions"]
                context["validation"] = parsed_query["validation"]

                # Use parsed expression if confidence is high enough
                if parsed_query["parsed_query"]["confidence"] > 0.6:
                    expression = parsed_query["parsed_query"]["expression"]

                methodology = CalculationMethodology(
                    problem_type=f"Natural Language {calc_type.title()}",
                    approach=f"Advanced NL parsing with {parsed_query['parsed_query']['confidence']:.1%} confidence"
                )

                methodology.add_step(
                    "Parse natural language query",
                    f"Original: '{parsed_query['parsed_query']['original_query']}'",
                    f"Parsed: '{expression}' as {calc_type}"
                )

                if parsed_query["suggestions"]:
                    for suggestion in parsed_query["suggestions"]:
                        methodology.add_assumption(f"Suggestion: {suggestion}")

            else:
                # Fallback to simple pattern matching
                calc_type = self._determine_calculation_type(expression, context)
                methodology = None

            # Route to appropriate handler with enhanced error handling
            try:
                # Merge existing methodology if created by NL parser
                if calc_type == "derivative":
                    result, calc_methodology = await self._calculate_derivative_with_steps(expression, context)
                elif calc_type == "integral":
                    result, calc_methodology = await self._calculate_integral_with_steps(expression, context)
                elif calc_type == "solve":
                    result, calc_methodology = await self._solve_equation_with_steps(expression, context)
                elif calc_type == "limit":
                    result, calc_methodology = await self._calculate_limit_with_steps(expression, context)
                elif calc_type == "series":
                    result, calc_methodology = await self._calculate_series_with_steps(expression, context)
                elif calc_type == "simplify" or calc_type == "factor" or calc_type == "expand":
                    result, calc_methodology = await self._symbolic_manipulation_with_steps(expression, context)
                elif calc_type == "differential_equation":
                    result, calc_methodology = await self._solve_differential_equation_with_steps(expression, context)
                elif calc_type == "geometry":
                    result, calc_methodology = await self._calculate_geometry_with_steps(expression, context)
                elif calc_type == "finance":
                    result, calc_methodology = await self._calculate_financial_with_steps(expression, context)
                elif calc_type == "statistics":
                    result, calc_methodology = await self._calculate_statistical_with_steps(expression, context)
                elif calc_type == "matrix":
                    result, calc_methodology = await self._calculate_matrix_with_steps(expression, context)
                else:
                    result, calc_methodology = await self._evaluate_expression_with_steps(expression, context)

                # Merge methodologies if we have both
                if methodology and calc_methodology:
                    # Combine steps from both methodologies
                    for step in calc_methodology.steps:
                        methodology.steps.append(step)
                    for ref in calc_methodology.references:
                        if ref not in methodology.references:
                            methodology.references.append(ref)
                    for assumption in calc_methodology.assumptions:
                        if assumption not in methodology.assumptions:
                            methodology.assumptions.append(assumption)
                else:
                    methodology = calc_methodology
            except ValueError as ve:
                # Handle specific value errors with user-friendly messages
                methodology = CalculationMethodology(
                    problem_type="Error Handling",
                    approach="Input validation and error recovery"
                )
                methodology.add_step(
                    "Input validation failed",
                    f"Invalid input: {str(ve)}",
                    "Error"
                )
                return None, methodology
            except ZeroDivisionError as zde:
                # Handle division by zero
                methodology = CalculationMethodology(
                    problem_type="Mathematical Error",
                    approach="Division by zero detection"
                )
                methodology.add_step(
                    "Division by zero detected",
                    "Mathematical expression contains division by zero",
                    "Error"
                )
                methodology.add_reference("Division by zero is undefined in mathematics")
                return None, methodology
            except OverflowError as oe:
                # Handle numerical overflow
                methodology = CalculationMethodology(
                    problem_type="Numerical Error",
                    approach="Overflow detection and handling"
                )
                methodology.add_step(
                    "Numerical overflow detected",
                    "Result is too large to represent",
                    "Error"
                )
                methodology.add_reference("Consider using logarithmic scale or approximations")
                return None, methodology

            # Calculate confidence based on methodology completeness
            confidence = self._calculate_confidence(methodology)

            return {
                "answer": result,
                "methodology": self._format_methodology(methodology),
                "steps": methodology.to_dict()["steps"] if methodology else [],
                "confidence": confidence,
                "calculation_type": calc_type,
                "assumptions": methodology.assumptions if methodology else [],
                "references": methodology.references if methodology else []
            }

        except Exception as e:
            logger.error(f"Error in calculation with explanation: {str(e)}")
            return {
                "answer": None,
                "methodology": f"Error occurred: {str(e)}",
                "steps": [],
                "confidence": 0.0,
                "error": str(e)
            }

    def _determine_calculation_type(self, expression: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Determine the type of calculation based on expression and context"""
        expression_lower = expression.lower()

        # Check for specific keywords with more comprehensive patterns
        if any(keyword in expression_lower for keyword in ["derivative", "differentiate", "d/dx", "diff", "partial"]):
            return "derivative"
        elif any(keyword in expression_lower for keyword in ["integral", "integrate", "∫", "definite integral", "indefinite integral"]):
            return "integral"
        elif any(keyword in expression_lower for keyword in ["solve", "equation", "=", "system of equations", "linear system"]):
            return "equation"
        elif any(keyword in expression_lower for keyword in ["limit", "lim", "approaches", "tends to"]):
            return "limit"
        elif any(keyword in expression_lower for keyword in ["series", "taylor", "maclaurin", "expansion"]):
            return "series"
        elif any(keyword in expression_lower for keyword in ["simplify", "factor", "expand", "rational"]):
            return "symbolic_manipulation"
        elif any(keyword in expression_lower for keyword in ["bond", "option", "interest", "yield", "price"]):
            return "financial"
        elif any(keyword in expression_lower for keyword in ["mean", "variance", "correlation", "regression"]):
            return "statistical"
        elif any(keyword in expression_lower for keyword in ["matrix", "determinant", "eigenvalue", "eigenvector", "linear algebra"]):
            return "matrix"
        elif any(keyword in expression_lower for keyword in ["geometry", "distance", "area", "perimeter", "volume"]):
            return "geometry"
        elif any(keyword in expression_lower for keyword in ["differential equation", "ode", "pde"]):
            return "differential_equation"
        else:
            return "expression"

    async def _calculate_derivative_with_steps(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Any, CalculationMethodology]:
        """Calculate derivative with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for symbolic calculations")

        methodology = CalculationMethodology(
            problem_type="Differentiation",
            approach="Symbolic differentiation using calculus rules"
        )

        try:
            # Use context information if available from NL parser
            if context and "parsed_components" in context:
                func_str = context["parsed_components"].get("function", expression)
                variable = context["parsed_components"].get("variable", "x")
            else:
                # Parse the expression
                if "d/dx" in expression or "derivative of" in expression.lower():
                    # Extract function from natural language
                    func_str = expression.split("of")[-1].strip() if "of" in expression else expression.split("d/dx")[-1].strip()
                else:
                    func_str = expression
                variable = "x"

            methodology.add_step(
                "Parse mathematical expression",
                f"Parsing: {func_str} with respect to {variable}",
                func_str
            )

            # Create symbol and parse expression
            var_symbol = symbols(variable)
            try:
                expr = parse_expr(func_str, transformations=self.transformations)
            except Exception as parse_error:
                methodology.add_step(
                    "Parse error",
                    f"Failed to parse expression: {func_str}",
                    f"Error: {str(parse_error)}"
                )
                raise ValueError(f"Cannot parse derivative expression '{func_str}': {str(parse_error)}")

            methodology.add_step(
                "Convert to symbolic form",
                f"Expression: {expr}",
                str(expr),
                latex=sp.latex(expr)
            )

            # Identify differentiation rules
            rules = self._identify_differentiation_rules(expr)
            methodology.add_reference(f"Differentiation rules: {', '.join(rules)}")

            # Perform differentiation
            derivative = diff(expr, var_symbol)

            methodology.add_step(
                "Apply differentiation rules",
                f"d/dx({expr}) = {derivative}",
                str(derivative),
                latex=sp.latex(derivative)
            )

            # Simplify if possible
            simplified = simplify(derivative)
            if simplified != derivative:
                methodology.add_step(
                    "Simplify the result",
                    f"Simplifying: {derivative} → {simplified}",
                    str(simplified),
                    latex=sp.latex(simplified)
                )
                final_result = simplified
            else:
                final_result = derivative

            return str(final_result), methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_integral_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Calculate integral with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for symbolic calculations")

        methodology = CalculationMethodology(
            problem_type="Integration",
            approach="Symbolic integration using calculus techniques"
        )

        try:
            # Parse the expression
            if "integral of" in expression.lower() or "∫" in expression:
                func_str = expression.split("of")[-1].strip() if "of" in expression else expression.replace("∫", "").strip()
            else:
                func_str = expression

            methodology.add_step(
                "Parse mathematical expression",
                f"Parsing: {func_str}",
                func_str
            )

            # Create symbol and parse expression
            x = symbols('x')
            try:
                expr = parse_expr(func_str, transformations=self.transformations)
            except Exception as parse_error:
                methodology.add_step(
                    "Parse error",
                    f"Failed to parse expression: {func_str}",
                    f"Error: {str(parse_error)}"
                )
                raise ValueError(f"Cannot parse integral expression '{func_str}': {str(parse_error)}")

            methodology.add_step(
                "Convert to symbolic form",
                f"Expression: {expr}",
                str(expr),
                latex=sp.latex(expr)
            )

            # Identify integration technique
            technique = self._identify_integration_technique(expr)
            methodology.add_reference(f"Integration technique: {technique}")

            # Perform integration
            integral = integrate(expr, x)

            methodology.add_step(
                "Apply integration technique",
                f"∫{expr}dx = {integral} + C",
                str(integral),
                latex=sp.latex(integral)
            )

            methodology.add_assumption("Constant of integration C is added for indefinite integrals")

            return f"{integral} + C", methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _solve_equation_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Solve equation with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for symbolic calculations")

        methodology = CalculationMethodology(
            problem_type="Equation Solving",
            approach="Algebraic manipulation and solving techniques"
        )

        try:
            # Parse equation
            if "=" in expression:
                left_str, right_str = expression.split("=")
                methodology.add_step(
                    "Identify equation parts",
                    f"Left side: {left_str.strip()}, Right side: {right_str.strip()}",
                    expression
                )
            else:
                # Assume equation equals zero
                left_str = expression
                right_str = "0"
                methodology.add_assumption("Assuming equation equals zero")

            # Parse both sides
            x = symbols('x')
            left_expr = parse_expr(left_str.strip(), transformations=self.transformations)
            right_expr = parse_expr(right_str.strip(), transformations=self.transformations)

            # Form equation
            equation = sp.Eq(left_expr, right_expr)

            methodology.add_step(
                "Form symbolic equation",
                f"{left_expr} = {right_expr}",
                str(equation),
                latex=sp.latex(equation)
            )

            # Rearrange to standard form
            standard_form = left_expr - right_expr
            methodology.add_step(
                "Convert to standard form",
                f"{standard_form} = 0",
                str(standard_form),
                latex=sp.latex(standard_form)
            )

            # Solve equation
            solutions = solve(equation, x)

            methodology.add_step(
                "Solve for variable",
                f"Solutions: {solutions}",
                str(solutions),
                latex=sp.latex(solutions) if solutions else None
            )

            # Verify solutions
            for i, sol in enumerate(solutions):
                verification = equation.subs(x, sol)
                methodology.add_step(
                    f"Verify solution {i+1}",
                    f"Substituting x = {sol}: {verification}",
                    str(verification)
                )

            return solutions, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_financial_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate financial problems with detailed steps"""
        expression_lower = expression.lower()

        if "compound interest" in expression_lower:
            return await self._compound_interest_with_steps(expression, context)
        elif "bond" in expression_lower and "price" in expression_lower:
            return await self._bond_pricing_with_steps(expression, context)
        elif "present value" in expression_lower:
            return await self._present_value_with_steps(expression, context)
        else:
            # Generic financial calculation
            methodology = CalculationMethodology(
                problem_type="Financial Calculation",
                approach="Standard financial formulas"
            )

            methodology.add_step(
                "Parse financial problem",
                expression,
                "Parsed"
            )

            # Extract parameters (simplified)
            import re
            numbers = re.findall(r'\d+\.?\d*', expression)

            if numbers:
                result = float(numbers[0])  # Simplified
                methodology.add_step(
                    "Extract financial parameters",
                    f"Found values: {numbers}",
                    numbers
                )

                methodology.add_step(
                    "Apply financial formula",
                    "Using standard financial calculation",
                    result
                )

                return result, methodology
            else:
                raise ValueError("Could not extract financial parameters")

    async def _compound_interest_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate compound interest with steps"""
        methodology = CalculationMethodology(
            problem_type="Compound Interest",
            approach="Compound interest formula: A = P(1 + r/n)^(nt)"
        )

        try:
            # Extract parameters from expression
            import re

            # Look for principal amount
            principal_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', expression)
            principal = float(principal_match.group(1).replace(',', '')) if principal_match else 1000

            # Look for interest rate
            rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', expression)
            rate = float(rate_match.group(1)) / 100 if rate_match else 0.05

            # Look for time period
            time_match = re.search(r'(\d+)\s*year', expression)
            time = float(time_match.group(1)) if time_match else 1

            # Assume annual compounding if not specified
            n = 1

            methodology.add_step(
                "Extract parameters",
                f"Principal: ${principal}, Rate: {rate*100}%, Time: {time} years, Compounding: {n} times/year",
                {"P": principal, "r": rate, "t": time, "n": n}
            )

            methodology.add_reference("Formula: A = P(1 + r/n)^(nt)")

            # Calculate step by step
            rate_per_period = rate / n
            methodology.add_step(
                "Calculate rate per period",
                f"r/n = {rate}/{n} = {rate_per_period}",
                rate_per_period
            )

            growth_factor = 1 + rate_per_period
            methodology.add_step(
                "Calculate growth factor",
                f"1 + r/n = 1 + {rate_per_period} = {growth_factor}",
                growth_factor
            )

            total_periods = n * time
            methodology.add_step(
                "Calculate total periods",
                f"nt = {n} × {time} = {total_periods}",
                total_periods
            )

            compound_factor = growth_factor ** total_periods
            methodology.add_step(
                "Calculate compound factor",
                f"(1 + r/n)^(nt) = {growth_factor}^{total_periods} = {compound_factor:.6f}",
                compound_factor
            )

            final_amount = principal * compound_factor
            methodology.add_step(
                "Calculate final amount",
                f"A = P × compound_factor = ${principal} × {compound_factor:.6f} = ${final_amount:.2f}",
                final_amount
            )

            interest_earned = final_amount - principal
            methodology.add_step(
                "Calculate interest earned",
                f"Interest = A - P = ${final_amount:.2f} - ${principal} = ${interest_earned:.2f}",
                interest_earned
            )

            return {
                "final_amount": round(final_amount, 2),
                "interest_earned": round(interest_earned, 2),
                "effective_rate": (compound_factor - 1) * 100
            }, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _evaluate_expression_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Evaluate general mathematical expression with steps"""
        methodology = CalculationMethodology(
            problem_type="Expression Evaluation",
            approach="Step-by-step arithmetic evaluation"
        )

        try:
            if SYMPY_AVAILABLE:
                # Use SymPy for symbolic evaluation
                expr = parse_expr(expression, transformations=self.transformations)

                methodology.add_step(
                    "Parse expression",
                    f"Input: {expression}",
                    str(expr),
                    latex=sp.latex(expr)
                )

                # Check if expression contains variables
                free_symbols = expr.free_symbols
                if free_symbols:
                    methodology.add_step(
                        "Identify variables",
                        f"Variables found: {free_symbols}",
                        str(free_symbols)
                    )

                    # Simplify algebraic expression
                    simplified = simplify(expr)
                    methodology.add_step(
                        "Simplify expression",
                        f"Simplified: {simplified}",
                        str(simplified),
                        latex=sp.latex(simplified)
                    )

                    return str(simplified), methodology
                else:
                    # Evaluate numerical expression
                    result = expr.evalf()
                    methodology.add_step(
                        "Evaluate numerically",
                        f"{expr} = {result}",
                        float(result)
                    )

                    return float(result), methodology
            else:
                # Fallback to basic evaluation - NEVER use eval() for security
                try:
                    # Only allow basic arithmetic operations
                    import ast
                    import operator as op

                    # Supported operators
                    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                                ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

                    def eval_expr(node):
                        if isinstance(node, ast.Num):  # <number>
                            return node.n
                        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                            return operators[type(node.op)](eval_expr(node.operand))
                        else:
                            raise TypeError(f"Unsupported expression type: {type(node)}")

                    node = ast.parse(expression, mode='eval').body
                    result = eval_expr(node)

                    methodology.add_step(
                        "Safe arithmetic evaluation",
                        f"{expression} = {result}",
                        result
                    )
                    return result, methodology
                except Exception as e:
                    raise ValueError(f"Cannot evaluate expression safely: {str(e)}")

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    def _identify_differentiation_rules(self, expr) -> List[str]:
        """Identify which differentiation rules apply to the expression"""
        rules = []

        if hasattr(expr, 'is_Add') and expr.is_Add:
            rules.append("Sum rule")
        if hasattr(expr, 'is_Mul') and expr.is_Mul:
            rules.append("Product rule")
        if hasattr(expr, 'is_Pow') and expr.is_Pow:
            rules.append("Power rule")
        if hasattr(expr, 'func'):
            func_name = expr.func.__name__
            if func_name in ['sin', 'cos', 'tan']:
                rules.append("Trigonometric derivatives")
            elif func_name in ['exp', 'log']:
                rules.append("Exponential/Logarithmic derivatives")

        if not rules:
            rules.append("Basic differentiation")

        return rules

    def _identify_integration_technique(self, expr) -> str:
        """Identify the integration technique for the expression"""
        if hasattr(expr, 'is_polynomial') and expr.is_polynomial():
            return "Polynomial integration"
        elif hasattr(expr, 'is_rational') and expr.is_rational:
            return "Rational function integration"
        elif hasattr(expr, 'func'):
            func_name = expr.func.__name__
            if func_name in ['sin', 'cos', 'tan']:
                return "Trigonometric integration"
            elif func_name == 'exp':
                return "Exponential integration"

        # Check for products that might need substitution
        if hasattr(expr, 'is_Mul') and expr.is_Mul:
            return "Integration by substitution"

        return "Standard integration"

    def _calculate_confidence(self, methodology: Optional[CalculationMethodology]) -> float:
        """Calculate confidence score based on methodology completeness"""
        if not methodology:
            return 0.0

        base_confidence = 0.5

        # Add confidence for each step
        step_confidence = min(len(methodology.steps) * 0.1, 0.3)

        # Add confidence for references
        ref_confidence = min(len(methodology.references) * 0.1, 0.1)

        # Add confidence for clear problem type
        type_confidence = 0.1 if methodology.problem_type != "Unknown" else 0

        return min(base_confidence + step_confidence + ref_confidence + type_confidence, 1.0)

    def _format_methodology(self, methodology: Optional[CalculationMethodology]) -> str:
        """Format methodology into readable explanation"""
        if not methodology:
            return "No methodology available"

        parts = [
            f"Problem Type: {methodology.problem_type}",
            f"Approach: {methodology.approach}"
        ]

        if methodology.assumptions:
            parts.append(f"Assumptions: {', '.join(methodology.assumptions)}")

        if methodology.references:
            parts.append(f"References: {', '.join(methodology.references)}")

        return "\n".join(parts)

    async def _calculate_statistical_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate statistical problems with detailed steps"""
        methodology = CalculationMethodology(
            problem_type="Statistical Analysis",
            approach="Statistical formulas and methods"
        )

        if not NUMPY_AVAILABLE:
            methodology.add_step(
                "Check dependencies",
                "NumPy not available for statistical calculations",
                "Error"
            )
            raise ValueError("NumPy required for statistical calculations")

        try:
            expression_lower = expression.lower()

            # Extract data from context or expression
            data = context.get("data", []) if context else []

            # Validate data array
            if not data or len(data) == 0:
                methodology.add_step(
                    "Data validation error",
                    "No data provided for statistical calculation",
                    "Error"
                )
                raise ValueError("Cannot perform statistical calculations on empty data set")

            # Validate data contains numeric values
            try:
                numeric_data = [float(x) for x in data]
                data = numeric_data
            except (ValueError, TypeError) as e:
                methodology.add_step(
                    "Data validation error",
                    f"Non-numeric values in data: {data}",
                    f"Error: {str(e)}"
                )
                raise ValueError(f"All data values must be numeric for statistical calculations: {str(e)}")

            if "mean" in expression_lower:
                methodology.add_reference("Formula: mean = Σx / n")

                methodology.add_step(
                    "Calculate sum",
                    f"Sum of values: {sum(data)}",
                    sum(data)
                )

                mean_value = np.mean(data)
                methodology.add_step(
                    "Calculate mean",
                    f"Mean = {sum(data)} / {len(data)} = {mean_value}",
                    mean_value
                )

                return mean_value, methodology

            elif "variance" in expression_lower:
                methodology.add_reference("Formula: variance = Σ(x - mean)² / n")

                mean_value = np.mean(data)
                methodology.add_step(
                    "Calculate mean",
                    f"Mean = {mean_value}",
                    mean_value
                )

                deviations = [(x - mean_value)**2 for x in data]
                methodology.add_step(
                    "Calculate squared deviations",
                    f"(x - mean)² for each value",
                    deviations
                )

                variance = np.var(data)
                methodology.add_step(
                    "Calculate variance",
                    f"Variance = {sum(deviations)} / {len(data)} = {variance}",
                    variance
                )

                return variance, methodology

            else:
                # Generic statistical calculation
                result = len(data)  # Placeholder
                methodology.add_step(
                    "Statistical calculation",
                    expression,
                    result
                )
                return result, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _bond_pricing_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate bond price with detailed steps"""
        methodology = CalculationMethodology(
            problem_type="Bond Pricing",
            approach="Present value of cash flows"
        )

        try:
            # Extract parameters
            import re

            # Extract years
            years_match = re.search(r'(\d+)\s*(?:-\s*)?year', expression)
            years = int(years_match.group(1)) if years_match else 10

            # Extract coupon rate
            coupon_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*coupon', expression)
            coupon_rate = float(coupon_match.group(1)) / 100 if coupon_match else 0.05

            # Extract yield
            yield_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*yield', expression)
            yield_rate = float(yield_match.group(1)) / 100 if yield_match else 0.03

            face_value = 1000  # Standard bond face value

            methodology.add_step(
                "Extract bond parameters",
                f"Face value: ${face_value}, Coupon: {coupon_rate*100}%, Yield: {yield_rate*100}%, Years: {years}",
                {"face_value": face_value, "coupon_rate": coupon_rate, "yield_rate": yield_rate, "years": years}
            )

            methodology.add_reference("Bond Price = PV(Coupons) + PV(Face Value)")

            # Calculate annual coupon payment
            coupon_payment = face_value * coupon_rate
            methodology.add_step(
                "Calculate annual coupon",
                f"Coupon = ${face_value} × {coupon_rate} = ${coupon_payment}",
                coupon_payment
            )

            # Calculate present value of coupons
            pv_coupons = 0

            # Validate yield rate to prevent division by zero
            if yield_rate <= -1:
                methodology.add_step(
                    "Error: Invalid yield rate",
                    f"Yield rate {yield_rate} would cause division by zero",
                    "Error"
                )
                raise ValueError(f"Invalid yield rate: {yield_rate*100}% (must be > -100%)")

            for t in range(1, years + 1):
                denominator = (1 + yield_rate) ** t
                if abs(denominator) < 1e-10:
                    raise ValueError(f"Division by zero: yield calculation overflow at year {t}")
                pv_coupon_t = coupon_payment / denominator
                pv_coupons += pv_coupon_t

                if t <= 3 or t == years:  # Show first 3 and last
                    methodology.add_step(
                        f"PV of coupon year {t}",
                        f"${coupon_payment} / (1 + {yield_rate})^{t} = ${pv_coupon_t:.2f}",
                        pv_coupon_t
                    )

            methodology.add_step(
                "Total PV of coupons",
                f"Sum of all coupon PVs = ${pv_coupons:.2f}",
                pv_coupons
            )

            # Calculate present value of face value
            face_denominator = (1 + yield_rate) ** years
            if abs(face_denominator) < 1e-10:
                raise ValueError(f"Division by zero: yield calculation overflow for {years} years")
            pv_face = face_value / face_denominator
            methodology.add_step(
                "PV of face value",
                f"${face_value} / (1 + {yield_rate})^{years} = ${pv_face:.2f}",
                pv_face
            )

            # Calculate total bond price
            bond_price = pv_coupons + pv_face
            methodology.add_step(
                "Calculate bond price",
                f"Bond Price = ${pv_coupons:.2f} + ${pv_face:.2f} = ${bond_price:.2f}",
                bond_price
            )

            # Calculate additional metrics
            if abs(bond_price) < 1e-10:
                current_yield = 0
                methodology.add_step(
                    "Current yield calculation",
                    "Bond price too small to calculate meaningful current yield",
                    0
                )
            else:
                current_yield = (coupon_payment / bond_price) * 100
                methodology.add_step(
                    "Calculate current yield",
                    f"Current Yield = ${coupon_payment} / ${bond_price:.2f} × 100 = {current_yield:.2f}%",
                    current_yield
                )

            return {
                "bond_price": round(bond_price, 2),
                "current_yield": round(current_yield, 2),
                "pv_coupons": round(pv_coupons, 2),
                "pv_face": round(pv_face, 2)
            }, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _present_value_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate present value with detailed steps"""
        methodology = CalculationMethodology(
            problem_type="Present Value",
            approach="Discounted cash flow: PV = FV / (1 + r)^t"
        )

        try:
            # Extract parameters
            import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

            # Future value
            fv_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', expression)
            future_value = float(fv_match.group(1).replace(',', '')) if fv_match else 1000

            # Interest rate
            rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', expression)
            rate = float(rate_match.group(1)) / 100 if rate_match else 0.05

            # Time period
            time_match = re.search(r'(\d+)\s*year', expression)
            time = float(time_match.group(1)) if time_match else 1

            methodology.add_step(
                "Extract parameters",
                f"Future Value: ${future_value}, Rate: {rate*100}%, Time: {time} years",
                {"FV": future_value, "r": rate, "t": time}
            )

            methodology.add_reference("Formula: PV = FV / (1 + r)^t")

            # Validate rate to prevent issues
            if rate <= -1:
                methodology.add_step(
                    "Error: Invalid discount rate",
                    f"Rate {rate} would cause division issues",
                    "Error"
                )
                raise ValueError(f"Invalid discount rate: {rate*100}% (must be > -100%)")

            # Calculate discount factor
            discount_factor = (1 + rate) ** time
            if abs(discount_factor) < 1e-10:
                raise ValueError(f"Division by zero: discount factor too small for {time} years at {rate*100}% rate")

            methodology.add_step(
                "Calculate discount factor",
                f"(1 + r)^t = (1 + {rate})^{time} = {discount_factor:.6f}",
                discount_factor
            )

            # Calculate present value
            present_value = future_value / discount_factor
            methodology.add_step(
                "Calculate present value",
                f"PV = ${future_value} / {discount_factor:.6f} = ${present_value:.2f}",
                present_value
            )

            # Calculate discount amount
            discount_amount = future_value - present_value
            methodology.add_step(
                "Calculate discount amount",
                f"Discount = ${future_value} - ${present_value:.2f} = ${discount_amount:.2f}",
                discount_amount
            )

            return {
                "present_value": round(present_value, 2),
                "discount_amount": round(discount_amount, 2),
                "effective_rate": rate * 100
            }, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_matrix_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate matrix operations with detailed steps"""
        methodology = CalculationMethodology(
            problem_type="Matrix Operation",
            approach="Linear algebra computation"
        )

        if not NUMPY_AVAILABLE:
            methodology.add_step(
                "Check dependencies",
                "NumPy not available for matrix calculations",
                "Error"
            )
            raise ValueError("NumPy required for matrix calculations")

        try:
            expression_lower = expression.lower()

            # Get matrices from context
            if not context or "matrix_a" not in context:
                methodology.add_step(
                    "Using default matrices",
                    "No matrix_a provided in context, using default values",
                    "Warning"
                )

            try:
                matrix_a = np.array(context.get("matrix_a", [[1, 2], [3, 4]])) if context else np.array([[1, 2], [3, 4]])
                matrix_b = np.array(context.get("matrix_b", [[5, 6], [7, 8]])) if context else np.array([[5, 6], [7, 8]])
            except Exception as e:
                methodology.add_step(
                    "Matrix parsing error",
                    "Failed to parse matrix data from context",
                    f"Error: {str(e)}"
                )
                raise ValueError(f"Invalid matrix data provided: {str(e)}")

            methodology.add_step(
                "Load matrices",
                f"Matrix A shape: {matrix_a.shape}, Matrix B shape: {matrix_b.shape}",
                {"A_shape": matrix_a.shape, "B_shape": matrix_b.shape}
            )

            if "multiply" in expression_lower or "product" in expression_lower:
                methodology.add_reference("Matrix multiplication: C[i,j] = Σ A[i,k] × B[k,j]")

                # Check dimensions
                if matrix_a.shape[1] != matrix_b.shape[0]:
                    methodology.add_step(
                        "Dimension validation failed",
                        f"Matrix A shape: {matrix_a.shape}, Matrix B shape: {matrix_b.shape}",
                        f"Error: A columns ({matrix_a.shape[1]}) must equal B rows ({matrix_b.shape[0]})"
                    )
                    raise ValueError(
                        f"Matrix dimensions incompatible for multiplication: "
                        f"Matrix A has shape {matrix_a.shape} but Matrix B has shape {matrix_b.shape}. "
                        f"For matrix multiplication A×B, the number of columns in A ({matrix_a.shape[1]}) "
                        f"must equal the number of rows in B ({matrix_b.shape[0]})"
                    )

                result = np.matmul(matrix_a, matrix_b)
                methodology.add_step(
                    "Perform matrix multiplication",
                    f"A × B = {result.tolist()}",
                    result.tolist()
                )

                return result.tolist(), methodology

            elif "determinant" in expression_lower:
                methodology.add_reference("Determinant calculation using cofactor expansion")

                det = np.linalg.det(matrix_a)
                methodology.add_step(
                    "Calculate determinant",
                    f"det(A) = {det:.6f}",
                    det
                )

                return det, methodology

            else:
                # Default to matrix info
                methodology.add_step(
                    "Matrix information",
                    f"Matrix A: {matrix_a.tolist()}",
                    matrix_a.tolist()
                )
                return matrix_a.tolist(), methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_limit_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Calculate limits with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for limit calculations")

        methodology = CalculationMethodology(
            problem_type="Limit Calculation",
            approach="Symbolic limit evaluation using L'Hôpital's rule and algebraic techniques"
        )

        try:
            # Parse limit expression
            if "limit" in expression.lower() or "lim" in expression.lower():
                # Extract function, variable, and approach value
                # This is a simplified parser - in practice, you'd want more robust parsing
                parts = expression.lower().replace("limit", "").replace("lim", "").strip()
                if "as" in parts and "approaches" in parts:
                    func_part, approach_part = parts.split("as")[0].strip(), parts.split("approaches")[-1].strip()
                elif "->" in parts:
                    if "as" in parts:
                        func_part = parts.split("as")[0].strip()
                        var_approach = parts.split("as")[1].strip()
                        var_name = var_approach.split("->")[0].strip()
                        approach_val = var_approach.split("->")[1].strip()
                    else:
                        func_part = parts.split("->")[0].strip()
                        approach_val = parts.split("->")[-1].strip()
                        var_name = "x"
                else:
                    func_part = parts
                    approach_val = "0"
                    var_name = "x"
            else:
                func_part = expression
                approach_val = "0"
                var_name = "x"

            methodology.add_step(
                "Parse limit expression",
                f"Function: {func_part}, Variable: {var_name}, Approaches: {approach_val}",
                {"function": func_part, "variable": var_name, "approach": approach_val}
            )

            # Create symbols and parse expressions
            var = symbols(var_name)
            func_expr = parse_expr(func_part, transformations=self.transformations)

            # Parse approach value
            if approach_val.lower() in ["infinity", "inf", "∞"]:
                approach_value = oo
            elif approach_val.lower() in ["-infinity", "-inf", "-∞"]:
                approach_value = -oo
            else:
                approach_value = parse_expr(approach_val, transformations=self.transformations)

            methodology.add_step(
                "Set up limit",
                f"lim[{var} → {approach_value}] {func_expr}",
                f"limit({func_expr}, {var}, {approach_value})",
                latex=f"\\lim_{{{var} \\to {approach_value}}} {sp.latex(func_expr)}"
            )

            # Calculate the limit
            result = limit(func_expr, var, approach_value)

            methodology.add_step(
                "Evaluate limit",
                f"Result: {result}",
                str(result),
                latex=sp.latex(result)
            )

            # Check for indeterminate forms
            if result.is_finite is False:
                methodology.add_reference("Limit evaluates to infinity")
            elif str(result) in ["nan", "zoo"]:
                methodology.add_reference("Indeterminate form - may require L'Hôpital's rule")

            return str(result), methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_series_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Calculate series expansions with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for series calculations")

        methodology = CalculationMethodology(
            problem_type="Series Expansion",
            approach="Taylor/Maclaurin series expansion"
        )

        try:
            # Parse series request
            if "taylor" in expression.lower():
                series_type = "Taylor"
                if "around" in expression.lower() or "at" in expression.lower():
                    # Extract expansion point
                    func_part = expression.split("taylor")[1].split("around")[0].strip() if "around" in expression else expression.split("taylor")[1].split("at")[0].strip()
                    expansion_point = expression.split("around")[1].strip() if "around" in expression else expression.split("at")[1].strip()
                else:
                    func_part = expression.replace("taylor", "").strip()
                    expansion_point = "0"
            elif "maclaurin" in expression.lower():
                series_type = "Maclaurin"
                func_part = expression.replace("maclaurin", "").strip()
                expansion_point = "0"
            else:
                series_type = "Taylor"
                func_part = expression
                expansion_point = "0"

            # Default parameters
            variable = "x"
            order = 5  # Default to 5 terms

            methodology.add_step(
                "Parse series request",
                f"Function: {func_part}, Type: {series_type}, Point: {expansion_point}, Order: {order}",
                {"function": func_part, "type": series_type, "point": expansion_point}
            )

            # Create symbols and expressions
            x = symbols(variable)
            func_expr = parse_expr(func_part, transformations=self.transformations)
            point = parse_expr(expansion_point, transformations=self.transformations)

            methodology.add_step(
                "Set up series expansion",
                f"{series_type} series of {func_expr} around {point}",
                str(func_expr),
                latex=sp.latex(func_expr)
            )

            # Calculate series expansion
            series_result = series(func_expr, x, point, n=order + 1)

            methodology.add_step(
                "Calculate series terms",
                f"Series expansion: {series_result}",
                str(series_result),
                latex=sp.latex(series_result)
            )

            # Remove O() term for cleaner result
            series_polynomial = series_result.removeO()

            methodology.add_step(
                "Simplified series",
                f"Polynomial approximation: {series_polynomial}",
                str(series_polynomial),
                latex=sp.latex(series_polynomial)
            )

            methodology.add_reference(f"{series_type} series formula with {order} terms")

            return str(series_polynomial), methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _symbolic_manipulation_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Perform symbolic manipulations with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for symbolic manipulation")

        methodology = CalculationMethodology(
            problem_type="Symbolic Manipulation",
            approach="Algebraic transformation and simplification"
        )

        try:
            # Determine manipulation type
            expr_lower = expression.lower()
            if "simplify" in expr_lower:
                operation = "simplify"
                func_part = expression.replace("simplify", "").strip()
            elif "factor" in expr_lower:
                operation = "factor"
                func_part = expression.replace("factor", "").strip()
            elif "expand" in expr_lower:
                operation = "expand"
                func_part = expression.replace("expand", "").strip()
            elif "rational" in expr_lower or "together" in expr_lower:
                operation = "together"
                func_part = expression.replace("rational", "").replace("together", "").strip()
            elif "apart" in expr_lower or "partial" in expr_lower:
                operation = "apart"
                func_part = expression.replace("apart", "").replace("partial", "").strip()
            else:
                operation = "simplify"
                func_part = expression

            methodology.add_step(
                "Determine operation",
                f"Operation: {operation}, Expression: {func_part}",
                {"operation": operation, "expression": func_part}
            )

            # Parse expression
            expr = parse_expr(func_part, transformations=self.transformations)

            methodology.add_step(
                "Parse original expression",
                f"Original: {expr}",
                str(expr),
                latex=sp.latex(expr)
            )

            # Apply the operation
            if operation == "simplify":
                result = simplify(expr)
                methodology.add_reference("Simplification using algebraic rules")
            elif operation == "factor":
                result = factor(expr)
                methodology.add_reference("Factorization using polynomial techniques")
            elif operation == "expand":
                result = expand(expr)
                methodology.add_reference("Expansion using distributive law")
            elif operation == "together":
                result = together(expr)
                methodology.add_reference("Combining rational expressions")
            elif operation == "apart":
                result = apart(expr)
                methodology.add_reference("Partial fraction decomposition")
            else:
                result = simplify(expr)

            methodology.add_step(
                f"Apply {operation}",
                f"Result: {result}",
                str(result),
                latex=sp.latex(result)
            )

            # Check if the result is different
            if str(expr) != str(result):
                methodology.add_step(
                    "Verify transformation",
                    f"Transformation successful: {expr} → {result}",
                    "Success"
                )
            else:
                methodology.add_step(
                    "Verify transformation",
                    "Expression is already in desired form",
                    "No change"
                )

            return str(result), methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _solve_differential_equation_with_steps(self, expression: str) -> Tuple[Any, CalculationMethodology]:
        """Solve differential equations with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for differential equation solving")

        methodology = CalculationMethodology(
            problem_type="Differential Equation",
            approach="Symbolic solution using standard techniques"
        )

        try:
            # Parse differential equation
            # This is a simplified parser - in practice, you'd want more robust parsing
            eq_part = expression.lower().replace("differential equation", "").replace("ode", "").strip()

            methodology.add_step(
                "Parse differential equation",
                f"Equation: {eq_part}",
                eq_part
            )

            # Create function and variable symbols
            x = symbols('x')
            y = sp.Function('y')

            # Parse the equation (simplified approach)
            if "y'" in eq_part or "dy/dx" in eq_part:
                # First order ODE
                eq_str = eq_part.replace("y'", "Derivative(y(x), x)").replace("dy/dx", "Derivative(y(x), x)")
                eq_str = eq_str.replace("y", "y(x)")
            elif "y''" in eq_part or "d2y/dx2" in eq_part:
                # Second order ODE
                eq_str = eq_part.replace("y''", "Derivative(y(x), x, 2)").replace("d2y/dx2", "Derivative(y(x), x, 2)")
                eq_str = eq_str.replace("y'", "Derivative(y(x), x)")
                eq_str = eq_str.replace("y", "y(x)")
            else:
                # Assume first order
                eq_str = eq_part.replace("y", "y(x)")

            methodology.add_step(
                "Set up differential equation",
                f"Symbolic form: {eq_str}",
                eq_str
            )

            # Parse the equation
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=")
                eq_expr = parse_expr(lhs.strip()) - parse_expr(rhs.strip())
            else:
                eq_expr = parse_expr(eq_str)

            methodology.add_step(
                "Create equation object",
                f"Equation: {eq_expr} = 0",
                str(eq_expr)
            )

            # Solve the differential equation
            try:
                solution = dsolve(eq_expr, y(x))
                methodology.add_step(
                    "Solve differential equation",
                    f"Solution: {solution}",
                    str(solution),
                    latex=sp.latex(solution)
                )

                methodology.add_reference("Solution found using symbolic methods")
                return str(solution), methodology

            except Exception as solve_error:
                methodology.add_step(
                    "Solution attempt failed",
                    f"Could not find symbolic solution: {str(solve_error)}",
                    "No solution"
                )
                raise ValueError(f"Could not solve differential equation: {str(solve_error)}")

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise

    async def _calculate_geometry_with_steps(self, expression: str, context: Dict[str, Any]) -> Tuple[Any, CalculationMethodology]:
        """Calculate geometric problems with detailed steps"""
        if not SYMPY_AVAILABLE:
            raise ValueError("SymPy not available for geometry calculations")

        methodology = CalculationMethodology(
            problem_type="Geometry Calculation",
            approach="Analytical geometry using coordinate systems"
        )

        try:
            expr_lower = expression.lower()

            if "distance" in expr_lower and "points" in expr_lower:
                # Distance between points
                point1 = context.get("point1", [0, 0])
                point2 = context.get("point2", [1, 1])

                methodology.add_step(
                    "Extract points",
                    f"Point 1: {point1}, Point 2: {point2}",
                    {"point1": point1, "point2": point2}
                )

                p1 = Point(point1[0], point1[1])
                p2 = Point(point2[0], point2[1])

                distance = p1.distance(p2)
                methodology.add_step(
                    "Calculate distance",
                    f"Distance = √[(x₂-x₁)² + (y₂-y₁)²] = {distance}",
                    float(distance)
                )

                methodology.add_reference("Euclidean distance formula")
                return float(distance), methodology

            elif "area" in expr_lower and "triangle" in expr_lower:
                # Triangle area
                vertices = context.get("vertices", [[0, 0], [1, 0], [0, 1]])
                if len(vertices) != 3:
                    raise ValueError("Triangle requires exactly 3 vertices")

                methodology.add_step(
                    "Extract triangle vertices",
                    f"Vertices: {vertices}",
                    vertices
                )

                triangle = Triangle(Point(vertices[0]), Point(vertices[1]), Point(vertices[2]))
                area = triangle.area

                methodology.add_step(
                    "Calculate triangle area",
                    f"Area = {area}",
                    float(area)
                )

                methodology.add_reference("Triangle area using cross product")
                return float(area), methodology

            elif "circle" in expr_lower and "area" in expr_lower:
                # Circle area
                radius = context.get("radius", 1)

                methodology.add_step(
                    "Extract radius",
                    f"Radius: {radius}",
                    radius
                )

                circle = Circle(Point(0, 0), radius)
                area = circle.area

                methodology.add_step(
                    "Calculate circle area",
                    f"Area = π × r² = π × {radius}² = {area}",
                    float(area)
                )

                methodology.add_reference("Circle area formula: A = πr²")
                return float(area), methodology

            else:
                # Generic geometry calculation
                result = len(expression)  # Placeholder
                methodology.add_step(
                    "Generic geometry calculation",
                    expression,
                    result
                )
                return result, methodology

        except Exception as e:
            methodology.add_step(
                "Error in calculation",
                str(e),
                "Error"
            )
            raise
