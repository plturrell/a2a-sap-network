"""
Advanced Natural Language Parser for Mathematical Expressions
Provides sophisticated parsing of natural language mathematical queries
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Import SymPy for mathematical parsing validation
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class MathOperation(Enum):
    """Mathematical operation types"""
    EVALUATE = "evaluate"
    DERIVATIVE = "derivative"
    INTEGRAL = "integral"
    SOLVE = "solve"
    LIMIT = "limit"
    SERIES = "series"
    SIMPLIFY = "simplify"
    FACTOR = "factor"
    EXPAND = "expand"
    MATRIX = "matrix"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    FINANCE = "finance"
    OPTIMIZATION = "optimization"
    DIFFERENTIAL_EQUATION = "differential_equation"

@dataclass
class ParsedMathQuery:
    """Structured representation of a parsed mathematical query"""
    operation: MathOperation
    expression: str
    variables: List[str]
    parameters: Dict[str, Any]
    context: Dict[str, Any]
    confidence: float
    original_query: str
    parsed_components: Dict[str, str]

class MathematicalNLParser:
    """Advanced natural language parser for mathematical expressions"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.math_keywords = self._initialize_math_keywords()
        self.variable_patterns = self._initialize_variable_patterns()
        self.function_mappings = self._initialize_function_mappings()
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for different mathematical operations"""
        return {
            "derivative": [
                r"(?:find|calculate|compute|take|get)\s+(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?",
                r"differentiate\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?",
                r"d/d(\w+)\s*\(\s*(.+?)\s*\)",
                r"∂/∂(\w+)\s*\(\s*(.+?)\s*\)",
                r"what\s+is\s+(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?",
                r"(.+?)\s+differentiated\s+(?:with\s+respect\s+to\s+)?(\w+)"
            ],
            "integral": [
                r"(?:find|calculate|compute|integrate)\s+(?:the\s+)?integral\s+of\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?",
                r"integrate\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?",
                r"∫\s*(.+?)(?:\s+d(\w+))?(?:\s+from\s+(.+?)\s+to\s+(.+?))?",
                r"what\s+is\s+(?:the\s+)?integral\s+of\s+(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?",
                r"area\s+under\s+(?:the\s+curve\s+)?(.+?)(?:\s+from\s+(.+?)\s+to\s+(.+?))?"
            ],
            "limit": [
                r"(?:find|calculate|compute)\s+(?:the\s+)?limit\s+of\s+(.+?)\s+as\s+(\w+)\s+approaches\s+(.+)",
                r"lim\s*(?:\(\s*(\w+)\s*→\s*(.+?)\s*\))?\s*(.+)",
                r"limit\s+(.+?)\s+as\s+(\w+)\s+→\s+(.+)",
                r"what\s+happens\s+to\s+(.+?)\s+as\s+(\w+)\s+approaches\s+(.+)",
                r"(?:the\s+)?limit\s+(.+?)\s+when\s+(\w+)\s+goes\s+to\s+(.+)"
            ],
            "series": [
                r"(?:find|calculate|compute)\s+(?:the\s+)?(?:taylor|maclaurin)\s+series\s+(?:expansion\s+)?(?:of\s+)?(.+?)(?:\s+(?:around|at|about)\s+(.+?))?",
                r"expand\s+(.+?)\s+(?:as\s+a\s+)?(?:taylor|maclaurin)\s+series(?:\s+(?:around|at|about)\s+(.+?))?",
                r"taylor\s+series\s+(?:of\s+)?(.+?)(?:\s+(?:around|at|about)\s+(.+?))?",
                r"maclaurin\s+series\s+(?:of\s+)?(.+?)",
                r"series\s+expansion\s+(?:of\s+)?(.+?)(?:\s+(?:around|at|about)\s+(.+?))?"
            ],
            "solve": [
                r"solve\s+(?:the\s+equation\s+)?(.+?)(?:\s+for\s+(\w+))?",
                r"(?:find|calculate)\s+(?:the\s+)?(?:solution|roots?)\s+(?:of|to)\s+(.+?)(?:\s+for\s+(\w+))?",
                r"what\s+(?:is|are)\s+(?:the\s+)?(?:solution|roots?)\s+(?:of|to)\s+(.+?)(?:\s+for\s+(\w+))?",
                r"(.+?)\s*=\s*(.+?)(?:\s+solve\s+for\s+(\w+))?",
                r"when\s+(?:is\s+)?(.+?)\s*=\s*(.+?)",
                r"find\s+(\w+)\s+(?:such\s+that\s+|where\s+)?(.+)"
            ],
            "simplify": [
                r"simplify\s+(.+)",
                r"(?:make\s+)?(?:simpler|reduce)\s+(.+)",
                r"what\s+is\s+(.+?)\s+simplified",
                r"can\s+you\s+simplify\s+(.+)",
                r"reduce\s+(?:the\s+expression\s+)?(.+)"
            ],
            "factor": [
                r"factor\s+(.+)",
                r"factorize\s+(.+)",
                r"what\s+are\s+(?:the\s+)?factors\s+of\s+(.+)",
                r"factor\s+(?:the\s+expression\s+)?(.+)",
                r"write\s+(.+?)\s+in\s+factored\s+form"
            ],
            "expand": [
                r"expand\s+(.+)",
                r"multiply\s+out\s+(.+)",
                r"what\s+is\s+(.+?)\s+expanded",
                r"distribute\s+(.+)",
                r"expand\s+(?:the\s+expression\s+)?(.+)"
            ],
            "evaluate": [
                r"(?:calculate|compute|evaluate|find)\s+(?:the\s+value\s+of\s+)?(.+)",
                r"what\s+(?:is|does)\s+(.+?)\s+(?:equal|evaluate\s+to)?",
                r"(.+?)\s*=\s*\?",
                r"solve\s+(.+?)(?:\s+numerically)?",
                r"give\s+me\s+(?:the\s+value\s+of\s+)?(.+)"
            ]
        }
    
    def _initialize_math_keywords(self) -> Dict[str, List[str]]:
        """Initialize mathematical keywords and their synonyms"""
        return {
            "functions": [
                "sin", "cos", "tan", "sec", "csc", "cot",
                "arcsin", "arccos", "arctan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
                "exp", "log", "ln", "log10", "sqrt", "abs", "floor", "ceil"
            ],
            "constants": [
                "pi", "π", "e", "euler", "infinity", "inf", "∞", "i", "j"
            ],
            "operators": [
                "+", "-", "*", "×", "·", "/", "÷", "^", "**", "√", "∛"
            ],
            "variables": [
                "x", "y", "z", "t", "u", "v", "w", "a", "b", "c", "n", "m", "k"
            ],
            "greek_letters": [
                "alpha", "β", "gamma", "δ", "epsilon", "ζ", "eta", "θ", "iota",
                "kappa", "λ", "mu", "ν", "xi", "omicron", "π", "rho", "σ",
                "tau", "υ", "phi", "χ", "psi", "ω"
            ]
        }
    
    def _initialize_variable_patterns(self) -> List[str]:
        """Initialize patterns for variable detection"""
        return [
            r"\b([a-zA-Z])\b(?!\s*\()",  # Single letters not followed by parentheses
            r"\b([a-zA-Z]+\d*)\b",       # Variables with optional numbers
            r"([α-ωΑ-Ω])",              # Greek letters
        ]
    
    def _initialize_function_mappings(self) -> Dict[str, str]:
        """Initialize natural language to mathematical function mappings"""
        return {
            "sine": "sin",
            "cosine": "cos", 
            "tangent": "tan",
            "natural log": "log",
            "natural logarithm": "log",
            "logarithm": "log10",
            "square root": "sqrt",
            "cube root": "cbrt",
            "absolute value": "abs",
            "exponential": "exp",
            "factorial": "factorial",
            "binomial coefficient": "binomial"
        }
    
    def parse_mathematical_query(self, query: str) -> ParsedMathQuery:
        """
        Parse a natural language mathematical query into structured components
        """
        query = query.strip()
        original_query = query
        
        # Step 1: Detect operation type
        operation, confidence = self._detect_operation(query)
        
        # Step 2: Extract mathematical expression and parameters
        expression, parameters, parsed_components = self._extract_expression_and_parameters(query, operation)
        
        # Step 3: Normalize mathematical notation
        normalized_expression = self._normalize_mathematical_notation(expression)
        
        # Step 4: Extract variables
        variables = self._extract_variables(normalized_expression)
        
        # Step 5: Build context
        context = self._build_context(query, operation, parameters)
        
        # Step 6: Validate and adjust confidence
        final_confidence = self._validate_and_adjust_confidence(
            normalized_expression, operation, confidence
        )
        
        return ParsedMathQuery(
            operation=operation,
            expression=normalized_expression,
            variables=variables,
            parameters=parameters,
            context=context,
            confidence=final_confidence,
            original_query=original_query,
            parsed_components=parsed_components
        )
    
    def _detect_operation(self, query: str) -> Tuple[MathOperation, float]:
        """Detect the mathematical operation from natural language"""
        query_lower = query.lower()
        
        # Check each operation pattern
        for operation_name, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    try:
                        operation = MathOperation(operation_name)
                        confidence = 0.8  # Base confidence for pattern match
                        
                        # Boost confidence for exact keyword matches
                        if operation_name in query_lower:
                            confidence += 0.1
                        
                        return operation, min(confidence, 1.0)
                    except ValueError:
                        continue
        
        # Default to evaluation if no specific operation detected
        return MathOperation.EVALUATE, 0.5
    
    def _extract_expression_and_parameters(self, query: str, operation: MathOperation) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        """Extract mathematical expression and operation-specific parameters"""
        query_lower = query.lower()
        parameters = {}
        parsed_components = {}
        
        if operation in self.patterns:
            for pattern in self.patterns[operation.value]:
                match = re.search(pattern, query_lower)
                if match:
                    groups = match.groups()
                    
                    if operation == MathOperation.DERIVATIVE:
                        expression = groups[0] if groups[0] else ""
                        parameters["variable"] = groups[1] if len(groups) > 1 and groups[1] else "x"
                        parsed_components = {"function": expression, "variable": parameters["variable"]}
                        
                    elif operation == MathOperation.INTEGRAL:
                        expression = groups[0] if groups[0] else ""
                        if len(groups) >= 3 and groups[1] and groups[2]:
                            parameters["limits"] = [groups[1], groups[2]]
                            parameters["definite"] = True
                        else:
                            parameters["definite"] = False
                        parsed_components = {"function": expression, "limits": parameters.get("limits")}
                        
                    elif operation == MathOperation.LIMIT:
                        if len(groups) >= 3:
                            expression = groups[2] if groups[2] else groups[0]
                            parameters["variable"] = groups[1] if groups[1] else groups[0]
                            parameters["approach"] = groups[2] if len(groups) > 2 else groups[1]
                        else:
                            expression = groups[0] if groups[0] else ""
                            parameters["variable"] = "x"
                            parameters["approach"] = "0"
                        parsed_components = {"function": expression, "variable": parameters["variable"], "approach": parameters["approach"]}
                        
                    elif operation == MathOperation.SERIES:
                        expression = groups[0] if groups[0] else ""
                        parameters["point"] = groups[1] if len(groups) > 1 and groups[1] else "0"
                        parameters["order"] = 5  # Default order
                        if "taylor" in query_lower:
                            parameters["series_type"] = "taylor"
                        elif "maclaurin" in query_lower:
                            parameters["series_type"] = "maclaurin"
                        else:
                            parameters["series_type"] = "taylor"
                        parsed_components = {"function": expression, "point": parameters["point"], "type": parameters["series_type"]}
                        
                    elif operation == MathOperation.SOLVE:
                        if len(groups) >= 2 and "=" in query:
                            # Handle equation format
                            equation_parts = query.split("=")
                            expression = f"{equation_parts[0].strip()} = {equation_parts[1].strip()}"
                            parameters["variable"] = groups[-1] if groups[-1] else self._guess_primary_variable(expression)
                        else:
                            expression = groups[0] if groups[0] else ""
                            parameters["variable"] = groups[1] if len(groups) > 1 and groups[1] else self._guess_primary_variable(expression)
                        parsed_components = {"equation": expression, "variable": parameters["variable"]}
                        
                    else:
                        # Default extraction for other operations
                        expression = groups[0] if groups else ""
                        parsed_components = {"expression": expression}
                    
                    return expression.strip(), parameters, parsed_components
        
        # Fallback: treat entire query as expression
        expression = self._extract_mathematical_expression(query)
        parsed_components = {"expression": expression}
        return expression, parameters, parsed_components
    
    def _normalize_mathematical_notation(self, expression: str) -> str:
        """Convert natural language mathematical notation to symbolic form"""
        if not expression:
            return ""
        
        normalized = expression
        
        # Replace natural language functions with symbolic equivalents
        for nl_func, sym_func in self.function_mappings.items():
            normalized = re.sub(rf"\b{re.escape(nl_func)}\b", sym_func, normalized, flags=re.IGNORECASE)
        
        # Replace common mathematical phrases
        replacements = [
            (r"\bsquared\b", "^2"),
            (r"\bcubed\b", "^3"),
            (r"\bto\s+the\s+power\s+of\s+(\d+)", r"^\\1"),
            (r"\bto\s+the\s+(\d+)(?:st|nd|rd|th)\s+power", r"^\\1"),
            (r"\bsquare\s+root\s+of\b", "sqrt"),
            (r"\bcube\s+root\s+of\b", "cbrt"),
            (r"\babsolute\s+value\s+of\b", "abs"),
            (r"\bnatural\s+log\s+of\b", "log"),
            (r"\bln\s+of\b", "log"),
            (r"\blog\s+of\b", "log10"),
            (r"\bexponential\s+of\b", "exp"),
            (r"\be\s+to\s+the\s+power\s+of\b", "exp"),
            (r"\bpi\b", "π"),
            (r"\beuler'?s\s+number\b", "e"),
            (r"\binfinity\b", "∞"),
            (r"\btimes\b", "*"),
            (r"\bmultiplied\s+by\b", "*"),
            (r"\bdivided\s+by\b", "/"),
            (r"\bover\b", "/"),
            (r"\bplus\b", "+"),
            (r"\bminus\b", "-"),
            (r"\bpow\(([^,]+),\s*([^)]+)\)", r"(\\1)^(\\2)"),
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Handle implicit multiplication
        normalized = re.sub(r"(\d+)([a-zA-Z])", r"\\1*\\2", normalized)  # 2x -> 2*x
        normalized = re.sub(r"([a-zA-Z])(\d+)", r"\\1*\\2", normalized)  # x2 -> x*2
        normalized = re.sub(r"\)(\w)", r")*\\1", normalized)             # )x -> )*x
        normalized = re.sub(r"(\w)\(", r"\\1*(", normalized)             # x( -> x*(
        
        # Clean up extra spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names from mathematical expression"""
        variables = set()
        
        for pattern in self.variable_patterns:
            matches = re.findall(pattern, expression)
            for match in matches:
                if isinstance(match, tuple):
                    variables.update(match)
                else:
                    variables.add(match)
        
        # Filter out mathematical functions and constants
        filtered_variables = []
        for var in variables:
            if (var not in self.math_keywords["functions"] and 
                var not in self.math_keywords["constants"] and
                len(var) <= 3):  # Reasonable variable name length
                filtered_variables.append(var)
        
        return sorted(list(set(filtered_variables)))
    
    def _build_context(self, query: str, operation: MathOperation, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build context information for the mathematical query"""
        context = {
            "operation_type": operation.value,
            "query_length": len(query),
            "has_numbers": bool(re.search(r"\d", query)),
            "has_fractions": bool(re.search(r"\d+/\d+", query)),
            "has_decimals": bool(re.search(r"\d+\.\d+", query)),
            "has_parentheses": "(" in query and ")" in query,
            "has_equals": "=" in query,
            "word_count": len(query.split()),
            "complexity_score": self._calculate_complexity_score(query)
        }
        
        # Add operation-specific context
        context.update(parameters)
        
        return context
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate a complexity score for the mathematical query"""
        score = 0.0
        
        # Base complexity from length
        score += min(len(query) / 100, 0.3)
        
        # Add complexity for mathematical functions
        for func in self.math_keywords["functions"]:
            if func in query.lower():
                score += 0.1
        
        # Add complexity for operators
        operator_count = sum(1 for op in self.math_keywords["operators"] if op in query)
        score += min(operator_count * 0.05, 0.2)
        
        # Add complexity for parentheses nesting
        nesting_depth = 0
        max_depth = 0
        for char in query:
            if char == "(":
                nesting_depth += 1
                max_depth = max(max_depth, nesting_depth)
            elif char == ")":
                nesting_depth -= 1
        score += min(max_depth * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _validate_and_adjust_confidence(self, expression: str, operation: MathOperation, base_confidence: float) -> float:
        """Validate the parsed expression and adjust confidence accordingly"""
        if not expression.strip():
            return 0.1  # Very low confidence for empty expressions
        
        confidence = base_confidence
        
        # Try to parse with SymPy if available
        if SYMPY_AVAILABLE:
            try:
                parsed = parse_expr(expression, transformations=standard_transformations + (implicit_multiplication_application,))
                confidence += 0.1  # Boost for valid SymPy expression
            except Exception:
                confidence -= 0.2  # Penalty for invalid expression
        
        # Check for balanced parentheses
        if expression.count("(") != expression.count(")"):
            confidence -= 0.15
        
        # Check for valid variable names
        variables = self._extract_variables(expression)
        if variables and all(len(var) <= 2 and var.isalpha() for var in variables):
            confidence += 0.05
        
        # Operation-specific validation
        if operation == MathOperation.DERIVATIVE and "d" in expression:
            confidence += 0.05
        elif operation == MathOperation.INTEGRAL and ("∫" in expression or "integral" in expression.lower()):
            confidence += 0.05
        elif operation == MathOperation.SOLVE and "=" in expression:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_mathematical_expression(self, query: str) -> str:
        """Extract the core mathematical expression from a natural language query"""
        # Remove common question words and phrases
        stopwords = [
            "what", "is", "the", "of", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "with", "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "can", "you", "please", "find", "calculate", "compute", "evaluate",
            "give", "me", "show", "tell", "help", "solve", "determine"
        ]
        
        words = query.split()
        filtered_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\d\.\-\+\*/\^\(\)=]', '', word.lower())
            if clean_word and clean_word not in stopwords:
                filtered_words.append(word)  # Keep original case for mathematical expressions
        
        return " ".join(filtered_words)
    
    def _guess_primary_variable(self, expression: str) -> str:
        """Guess the primary variable in an expression"""
        variables = self._extract_variables(expression)
        if not variables:
            return "x"  # Default
        
        # Prefer common variable names
        preferred_order = ["x", "y", "z", "t", "n", "a", "b", "c"]
        for var in preferred_order:
            if var in variables:
                return var
        
        # Return first alphabetically
        return sorted(variables)[0]

class MathQueryProcessor:
    """Processes parsed mathematical queries and provides enhanced context"""
    
    def __init__(self):
        self.parser = MathematicalNLParser()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a mathematical query and return structured information"""
        parsed_query = self.parser.parse_mathematical_query(query)
        
        return {
            "parsed_query": {
                "operation": parsed_query.operation.value,
                "expression": parsed_query.expression,
                "variables": parsed_query.variables,
                "parameters": parsed_query.parameters,
                "confidence": parsed_query.confidence,
                "original_query": parsed_query.original_query,
                "parsed_components": parsed_query.parsed_components
            },
            "context": parsed_query.context,
            "suggestions": self._generate_suggestions(parsed_query),
            "validation": self._validate_query(parsed_query)
        }
    
    def _generate_suggestions(self, parsed_query: ParsedMathQuery) -> List[str]:
        """Generate helpful suggestions for the user"""
        suggestions = []
        
        if parsed_query.confidence < 0.6:
            suggestions.append("Consider rephrasing your question for better accuracy")
            
        if not parsed_query.variables and parsed_query.operation in [MathOperation.DERIVATIVE, MathOperation.SOLVE]:
            suggestions.append("Specify the variable you want to solve for or differentiate with respect to")
            
        if parsed_query.operation == MathOperation.INTEGRAL and "limits" not in parsed_query.parameters:
            suggestions.append("Add integration limits for a definite integral (e.g., 'from 0 to 1')")
            
        if "=" not in parsed_query.expression and parsed_query.operation == MathOperation.SOLVE:
            suggestions.append("Include an equation with '=' to solve")
            
        return suggestions
    
    def _validate_query(self, parsed_query: ParsedMathQuery) -> Dict[str, Any]:
        """Validate the parsed query and provide feedback"""
        validation = {
            "is_valid": True,
            "issues": [],
            "confidence_level": "high" if parsed_query.confidence > 0.8 else "medium" if parsed_query.confidence > 0.5 else "low"
        }
        
        if not parsed_query.expression.strip():
            validation["is_valid"] = False
            validation["issues"].append("No mathematical expression found")
            
        if parsed_query.expression.count("(") != parsed_query.expression.count(")"):
            validation["is_valid"] = False
            validation["issues"].append("Unbalanced parentheses")
            
        if parsed_query.confidence < 0.3:
            validation["is_valid"] = False
            validation["issues"].append("Low confidence in query interpretation")
            
        return validation