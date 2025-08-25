import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from app.clients.grokClient import GrokClient, GrokConfig

from app.a2a.core.security_base import SecureA2AAgent
"""
Intelligent Dispatch Skill for Calculation Agent
Uses GrokClient to analyze natural language calculation requests and dispatch to appropriate skills
"""

logger = logging.getLogger(__name__)


class IntelligentDispatchSkill(SecureA2AAgent):
    """Analyzes calculation requests and dispatches to appropriate calculation skills"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, grok_client: Optional[GrokClient] = None):
        super().__init__()
        self.grok_client = grok_client or GrokClient()

        # Skill mapping with descriptions and parameter schemas
        self.skill_registry = {
            "evaluate_calculation": {
                "description": "Basic mathematical expression evaluation",
                "keywords": ["evaluate", "calculate", "compute", "solve expression", "what is"],
                "parameters": {
                    "expression": "string",
                    "variables": "dict (optional)"
                }
            },
            "differentiate_calculation": {
                "description": "Differentiate mathematical expressions (calculus)",
                "keywords": ["differentiate", "derivative", "d/dx", "rate of change", "calculus"],
                "parameters": {
                    "expression": "string",
                    "variable": "string (default: x)",
                    "order": "int (default: 1)"
                }
            },
            "integrate_calculation": {
                "description": "Integrate mathematical expressions",
                "keywords": ["integrate", "integral", "antiderivative", "area under curve"],
                "parameters": {
                    "expression": "string",
                    "variable": "string (default: x)",
                    "limits": "list[float] (optional)"
                }
            },
            "solve_calculation": {
                "description": "Solve equations and systems of equations",
                "keywords": ["solve equation", "find x", "solve for", "equation", "system"],
                "parameters": {
                    "equations": "list[string]",
                    "variables": "list[string]"
                }
            },
            "simplify_calculation": {
                "description": "Simplify mathematical expressions",
                "keywords": ["simplify", "reduce", "factor", "expand"],
                "parameters": {
                    "expression": "string",
                    "method": "string (simplify/expand/factor)"
                }
            },
            "quantlib_bond_pricing": {
                "description": "Price bonds using QuantLib",
                "keywords": ["bond", "bond price", "fixed income", "coupon", "yield"],
                "parameters": {
                    "face_value": "float",
                    "coupon_rate": "float",
                    "maturity_date": "string",
                    "yield_rate": "float"
                }
            },
            "quantlib_option_pricing": {
                "description": "Price options using various models",
                "keywords": ["option", "call", "put", "black-scholes", "option price"],
                "parameters": {
                    "option_type": "string (european/american)",
                    "call_put": "string (call/put)",
                    "spot_price": "float",
                    "strike_price": "float",
                    "volatility": "float",
                    "risk_free_rate": "float",
                    "maturity_days": "int"
                }
            },
            "financial_calculation": {
                "description": "Various financial calculations",
                "keywords": ["compound interest", "present value", "future value", "loan", "payment"],
                "parameters": {
                    "calculation_type": "string",
                    "principal": "float",
                    "rate": "float",
                    "time": "float"
                }
            },
            "networkx_graph_analysis": {
                "description": "Analyze graph structures",
                "keywords": ["graph", "network", "nodes", "edges", "centrality", "path"],
                "parameters": {
                    "nodes": "list",
                    "edges": "list",
                    "analysis_type": "string"
                }
            },
            "networkx_shortest_path": {
                "description": "Find shortest paths in graphs",
                "keywords": ["shortest path", "dijkstra", "route", "path finding"],
                "parameters": {
                    "nodes": "list",
                    "edges": "list",
                    "source": "string/int",
                    "target": "string/int"
                }
            }
        }

    async def analyze_and_dispatch(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a natural language request and dispatch to appropriate skill"""
        try:
            # Step 1: Use Grok to analyze the request
            analysis = await self._analyze_request_with_grok(request, context)

            # Step 2: Determine the best skill based on analysis
            skill_name, confidence = self._determine_best_skill(analysis)

            # Step 3: Extract parameters for the chosen skill
            parameters = await self._extract_parameters_with_grok(
                request,
                skill_name,
                self.skill_registry[skill_name]["parameters"],
                context
            )

            # Step 4: Validate and prepare the dispatch
            dispatch_info = {
                "skill": skill_name,
                "confidence": confidence,
                "parameters": parameters,
                "original_request": request,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Step 5: Add any additional context or preprocessing
            dispatch_info = self._preprocess_dispatch(dispatch_info, context)

            return {
                "success": True,
                "dispatch": dispatch_info,
                "ready_to_execute": True
            }

        except Exception as e:
            logger.error(f"Failed to analyze and dispatch request: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_skill": "evaluate_calculation",
                "original_request": request
            }

    async def _analyze_request_with_grok(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use Grok to analyze the calculation request"""

        system_prompt = """You are an expert at understanding mathematical and financial calculation requests.
        Analyze the user's request and identify:
        1. The type of calculation needed
        2. Key mathematical/financial concepts mentioned
        3. Any specific values, variables, or parameters
        4. The expected output format

        Respond in JSON format."""

        user_prompt = f"""
        Analyze this calculation request:

        Request: {request}
        Context: {json.dumps(context or {}, indent=2)}

        Provide analysis as JSON with these fields:
        - calculation_type: main type of calculation
        - concepts: list of mathematical/financial concepts
        - extracted_values: any numbers, variables, or parameters mentioned
        - output_expectation: what the user expects as output
        - complexity: simple/moderate/complex
        """

        try:
            response = self.grok_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=500
            )

            # Parse the JSON response
            content = response.content

            # Try to extract JSON from the response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            return json.loads(json_str)

        except Exception as e:
            logger.error(f"Grok analysis failed: {e}")
            # Fallback analysis
            return {
                "calculation_type": "unknown",
                "concepts": [],
                "extracted_values": {},
                "output_expectation": "numerical result",
                "complexity": "simple"
            }

    def _determine_best_skill(self, analysis: Dict[str, Any]) -> Tuple[str, float]:
        """Determine the best skill based on analysis"""

        calc_type = analysis.get("calculation_type", "").lower()
        concepts = [c.lower() for c in analysis.get("concepts", [])]

        skill_scores = {}

        # Score each skill based on keyword matches
        for skill_name, skill_info in self.skill_registry.items():
            score = 0.0

            # Check calculation type
            for keyword in skill_info["keywords"]:
                if keyword in calc_type:
                    score += 2.0
                if any(keyword in concept for concept in concepts):
                    score += 1.5

            # Check for specific skill indicators
            if skill_name == "quantlib_bond_pricing" and any(word in calc_type + " ".join(concepts)
                                                            for word in ["bond", "coupon", "yield"]):
                score += 3.0

            if skill_name == "quantlib_option_pricing" and any(word in calc_type + " ".join(concepts)
                                                              for word in ["option", "call", "put"]):
                score += 3.0

            if skill_name == "networkx_graph_analysis" and any(word in calc_type + " ".join(concepts)
                                                              for word in ["graph", "network", "nodes"]):
                score += 3.0

            skill_scores[skill_name] = score

        # Get the best skill
        if not skill_scores or max(skill_scores.values()) == 0:
            return "evaluate_calculation", 0.5  # Default fallback

        best_skill = max(skill_scores, key=skill_scores.get)
        max_score = skill_scores[best_skill]

        # Calculate confidence (normalize to 0-1)
        confidence = min(max_score / 5.0, 1.0)

        return best_skill, confidence

    async def _extract_parameters_with_grok(
        self,
        request: str,
        skill_name: str,
        param_schema: Dict[str, str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract parameters for the specific skill using Grok"""

        system_prompt = f"""You are extracting parameters for a {skill_name} calculation.
        Extract values that match the parameter schema and return them as JSON.
        Be precise with numerical values and mathematical expressions."""

        user_prompt = f"""
        Extract parameters from this request for {skill_name}:

        Request: {request}
        Context: {json.dumps(context or {}, indent=2)}

        Expected parameters:
        {json.dumps(param_schema, indent=2)}

        Return a JSON object with the extracted parameters.
        For expressions, preserve them as strings.
        For missing optional parameters, omit them.
        For missing required parameters, use reasonable defaults.
        """

        try:
            response = self.grok_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Very low temperature for parameter extraction
                max_tokens=300
            )

            # Parse the JSON response
            content = response.content

            # Extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            parameters = json.loads(json_str)

            # Validate and clean parameters
            return self._validate_parameters(parameters, param_schema, skill_name)

        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            # Return minimal default parameters
            return self._get_default_parameters(skill_name)

    def _validate_parameters(self, parameters: Dict[str, Any], schema: Dict[str, str], skill_name: str) -> Dict[str, Any]:
        """Validate and clean extracted parameters"""

        validated = {}

        for param_name, param_type in schema.items():
            if param_name in parameters:
                value = parameters[param_name]

                # Type conversion and validation
                if "float" in param_type:
                    try:
                        validated[param_name] = float(value)
                    except:
                        logger.warning(f"Failed to convert {param_name} to float: {value}")

                elif "int" in param_type:
                    try:
                        validated[param_name] = int(value)
                    except:
                        logger.warning(f"Failed to convert {param_name} to int: {value}")

                elif "list" in param_type:
                    if isinstance(value, list):
                        validated[param_name] = value
                    elif isinstance(value, str):
                        # Try to parse as list
                        try:
                            validated[param_name] = json.loads(value)
                        except:
                            validated[param_name] = [value]

                else:
                    validated[param_name] = value

        # Add defaults for missing required parameters
        if skill_name == "evaluate_calculation" and "expression" not in validated:
            validated["expression"] = "0"  # Safe default

        return validated

    def _get_default_parameters(self, skill_name: str) -> Dict[str, Any]:
        """Get default parameters for a skill"""

        defaults = {
            "evaluate_calculation": {"expression": "1+1"},
            "differentiate_calculation": {"expression": "x^2", "variable": "x"},
            "integrate_calculation": {"expression": "x", "variable": "x"},
            "solve_calculation": {"equations": ["x-1=0"], "variables": ["x"]},
            "simplify_calculation": {"expression": "x+x", "method": "simplify"},
            "financial_calculation": {"calculation_type": "compound_interest", "principal": 1000, "rate": 5, "time": 1}
        }

        return defaults.get(skill_name, {})

    def _preprocess_dispatch(self, dispatch_info: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Preprocess dispatch information before execution"""

        # Add context to parameters if needed
        if context and "user_preferences" in context:
            dispatch_info["user_preferences"] = context["user_preferences"]

        # Add metadata
        dispatch_info["metadata"] = {
            "dispatch_version": "1.0",
            "preprocessing_applied": True,
            "context_provided": context is not None
        }

        # Skill-specific preprocessing
        skill = dispatch_info["skill"]

        if skill == "financial_calculation":
            # Convert percentage rates if needed
            if "rate" in dispatch_info["parameters"]:
                rate = dispatch_info["parameters"]["rate"]
                if rate > 1:  # Likely given as percentage
                    dispatch_info["parameters"]["rate"] = rate / 100

        elif skill in ["quantlib_bond_pricing", "quantlib_option_pricing"]:
            # Ensure dates are properly formatted
            for date_field in ["maturity_date", "issue_date", "settlement_date"]:
                if date_field in dispatch_info["parameters"]:
                    # Ensure ISO format
                    date_str = dispatch_info["parameters"][date_field]
                    if "/" in date_str or "-" not in date_str:
                        # Convert to ISO format
                        dispatch_info["parameters"][date_field] = self._convert_to_iso_date(date_str)

        return dispatch_info

    def _convert_to_iso_date(self, date_str: str) -> str:
        """Convert various date formats to ISO format"""
        from datetime import datetime

        # Try common formats
        formats = [
            "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%B %d, %Y", "%d %B %Y"
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except:
                continue

        # If all fail, return as is
        return date_str