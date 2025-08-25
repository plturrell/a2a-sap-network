import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from app.clients.grokClient import GrokClient, GrokConfig

from app.a2a.core.security_base import SecureA2AAgent
"""
Intelligent Dispatcher Skill for Calculation Agent
Uses GrokClient to analyze natural language calculation requests and dispatch to appropriate skills
"""

logger = logging.getLogger(__name__)


class IntelligentDispatcherSkill(SecureA2AAgent):
    """Analyzes calculation requests and dispatches to appropriate calculation skills"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self):
        super().__init__()
        # Initialize GrokClient
        self.grok_client = GrokClient()
        
        # Map of calculation types to skills
        self.skill_mapping = {
            "evaluate": {
                "skill": "evaluate_calculation",
                "keywords": ["evaluate", "calculate", "compute", "solve expression", "what is", "equals"],
                "patterns": [r"\d+\s*[\+\-\*\/\^]\s*\d+", r"[a-zA-Z]\s*=", r"f\([xy]\)"]
            },
            "differentiate": {
                "skill": "differentiate_calculation",
                "keywords": ["derivative", "differentiate", "d/dx", "gradient", "rate of change"],
                "patterns": [r"d.*?/d[a-zA-Z]", r"derivative", r"∂"]
            },
            "integrate": {
                "skill": "integrate_calculation",
                "keywords": ["integral", "integrate", "area under", "antiderivative"],
                "patterns": [r"∫", r"integral", r"integrate.*?d[a-zA-Z]"]
            },
            "solve": {
                "skill": "solve_calculation",
                "keywords": ["solve for", "find x", "equation", "system of equations", "roots"],
                "patterns": [r"solve.*?for", r"find.*?[xyz]", r".*?=.*?[xyz]"]
            },
            "simplify": {
                "skill": "simplify_calculation",
                "keywords": ["simplify", "reduce", "factor", "expand", "combine"],
                "patterns": [r"simplify", r"factor", r"expand"]
            },
            "financial_bond": {
                "skill": "quantlib_bond_pricing",
                "keywords": ["bond", "yield", "coupon", "duration", "convexity", "fixed income"],
                "patterns": [r"bond.*?price", r"yield.*?maturity", r"duration"]
            },
            "financial_option": {
                "skill": "quantlib_option_pricing",
                "keywords": ["option", "call", "put", "strike", "volatility", "black-scholes", "greeks"],
                "patterns": [r"option.*?price", r"call.*?put", r"delta.*?gamma"]
            },
            "graph_analysis": {
                "skill": "networkx_graph_analysis",
                "keywords": ["graph", "network", "nodes", "edges", "centrality", "path", "connected"],
                "patterns": [r"graph.*?analysis", r"network.*?structure", r"centrality"]
            },
            "graph_path": {
                "skill": "networkx_shortest_path",
                "keywords": ["shortest path", "pathfinding", "route", "distance between", "dijkstra"],
                "patterns": [r"shortest.*?path", r"path.*?from.*?to", r"distance.*?between"]
            },
            "financial_general": {
                "skill": "financial_calculation",
                "keywords": ["compound interest", "present value", "future value", "loan", "payment"],
                "patterns": [r"compound.*?interest", r"present.*?value", r"loan.*?payment"]
            }
        }
        
        # Prompt template for Grok analysis
        self.analysis_prompt = """You are an expert mathematical and financial calculation assistant.
        Analyze the following calculation request and determine:
        1. The type of calculation needed
        2. The required parameters
        3. Any specific constraints or requirements
        
        Request: {request}
        
        Respond in JSON format with:
        {{
            "calculation_type": "type of calculation",
            "confidence": 0.0-1.0,
            "parameters": {{extracted parameters}},
            "alternative_types": ["other possible calculation types"],
            "clarification_needed": "what needs clarification if anything"
        }}
        
        Available calculation types: {calculation_types}
        """
    
    async def analyze_and_dispatch(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a natural language calculation request and dispatch to appropriate skill"""
        try:
            # First, try pattern matching for quick dispatch
            quick_match = self._quick_pattern_match(request)
            if quick_match and quick_match["confidence"] > 0.8:
                logger.info(f"Quick pattern match found: {quick_match['skill']} with confidence {quick_match['confidence']}")
                return {
                    "success": True,
                    "dispatch_method": "pattern_matching",
                    "skill": quick_match["skill"],
                    "parameters": self._extract_basic_parameters(request, quick_match["type"]),
                    "confidence": quick_match["confidence"]
                }
            
            # Use Grok for more complex analysis
            grok_analysis = await self._analyze_with_grok(request, context)
            
            if grok_analysis["success"]:
                skill_name = self._map_type_to_skill(grok_analysis["calculation_type"])
                
                return {
                    "success": True,
                    "dispatch_method": "ai_analysis",
                    "skill": skill_name,
                    "parameters": grok_analysis["parameters"],
                    "confidence": grok_analysis["confidence"],
                    "alternative_skills": [
                        self._map_type_to_skill(alt_type) 
                        for alt_type in grok_analysis.get("alternative_types", [])
                    ],
                    "clarification_needed": grok_analysis.get("clarification_needed"),
                    "analysis": grok_analysis
                }
            else:
                # Fallback to keyword matching
                keyword_match = self._keyword_based_dispatch(request)
                return {
                    "success": True,
                    "dispatch_method": "keyword_matching",
                    "skill": keyword_match["skill"],
                    "parameters": self._extract_basic_parameters(request, keyword_match["type"]),
                    "confidence": keyword_match["confidence"]
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze and dispatch request: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_skill": "evaluate_calculation",
                "parameters": {"expression": request}
            }
    
    def _quick_pattern_match(self, request: str) -> Optional[Dict[str, Any]]:
        """Quick pattern matching for common calculation types"""
        request_lower = request.lower()
        
        for calc_type, config in self.skill_mapping.items():
            # Check patterns
            for pattern in config.get("patterns", []):
                if re.search(pattern, request_lower):
                    return {
                        "type": calc_type,
                        "skill": config["skill"],
                        "confidence": 0.9
                    }
            
            # Check keywords with high confidence
            keyword_count = sum(1 for keyword in config["keywords"] if keyword in request_lower)
            if keyword_count >= 2:
                return {
                    "type": calc_type,
                    "skill": config["skill"],
                    "confidence": 0.85
                }
        
        return None
    
    async def _analyze_with_grok(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use Grok to analyze the calculation request"""
        try:
            # Prepare the prompt
            calculation_types = ", ".join(self.skill_mapping.keys())
            prompt = self.analysis_prompt.format(
                request=request,
                calculation_types=calculation_types
            )
            
            # Add context if provided
            if context:
                prompt += f"\n\nAdditional context: {json.dumps(context, indent=2)}"
            
            # Call Grok
            messages = [
                {"role": "system", "content": "You are a mathematical calculation analyzer. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.grok_client.chat_completion(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent analysis
                max_tokens=500
            )
            
            # Parse response
            try:
                analysis = json.loads(response.content)
                analysis["success"] = True
                return analysis
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    analysis["success"] = True
                    return analysis
                else:
                    logger.error(f"Failed to parse Grok response as JSON: {response.content}")
                    return {"success": False, "error": "Invalid JSON response from Grok"}
                    
        except Exception as e:
            logger.error(f"Grok analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _map_type_to_skill(self, calculation_type: str) -> str:
        """Map calculation type to skill name"""
        # Direct mapping
        if calculation_type in self.skill_mapping:
            return self.skill_mapping[calculation_type]["skill"]
        
        # Fuzzy matching
        calc_type_lower = calculation_type.lower()
        for mapped_type, config in self.skill_mapping.items():
            if mapped_type in calc_type_lower or calc_type_lower in mapped_type:
                return config["skill"]
            
            # Check keywords
            for keyword in config["keywords"]:
                if keyword in calc_type_lower:
                    return config["skill"]
        
        # Default to evaluate
        return "evaluate_calculation"
    
    def _keyword_based_dispatch(self, request: str) -> Dict[str, Any]:
        """Fallback keyword-based dispatching"""
        request_lower = request.lower()
        best_match = None
        best_score = 0
        
        for calc_type, config in self.skill_mapping.items():
            score = sum(1 for keyword in config["keywords"] if keyword in request_lower)
            
            if score > best_score:
                best_score = score
                best_match = {
                    "type": calc_type,
                    "skill": config["skill"],
                    "confidence": min(score * 0.2, 0.7)  # Max 0.7 confidence for keyword matching
                }
        
        if best_match:
            return best_match
        
        # Default fallback
        return {
            "type": "evaluate",
            "skill": "evaluate_calculation",
            "confidence": 0.3
        }
    
    def _extract_basic_parameters(self, request: str, calc_type: str) -> Dict[str, Any]:
        """Extract basic parameters from the request based on calculation type"""
        parameters = {}
        
        if calc_type == "evaluate":
            # Extract mathematical expression
            # Remove question words
            expression = re.sub(r'^(what is|calculate|compute|evaluate)\s+', '', request.lower())
            parameters["expression"] = expression.strip()
            
        elif calc_type == "differentiate":
            # Extract function and variable
            var_match = re.search(r'd/d([a-zA-Z])', request) or re.search(r'respect to ([a-zA-Z])', request)
            if var_match:
                parameters["variable"] = var_match.group(1)
            
            # Extract expression
            expr_match = re.search(r'of\s+(.+?)(?:\s+with|\s+respect|$)', request)
            if expr_match:
                parameters["expression"] = expr_match.group(1).strip()
            
        elif calc_type == "solve":
            # Extract equations
            equations = []
            # Look for equations with = sign
            eq_matches = re.findall(r'([^,;]+=[^,;]+)', request)
            if eq_matches:
                equations = [eq.strip() for eq in eq_matches]
            parameters["equations"] = equations
            
            # Extract variables to solve for
            var_match = re.search(r'solve for ([a-zA-Z,\s]+)', request)
            if var_match:
                variables = [v.strip() for v in var_match.group(1).split(',')]
                parameters["variables"] = variables
        
        elif calc_type in ["financial_bond", "financial_option"]:
            # Extract numerical values
            numbers = re.findall(r'\d+\.?\d*', request)
            
            # Try to map common financial terms
            if "bond" in request:
                if len(numbers) >= 2:
                    parameters["face_value"] = float(numbers[0])
                    parameters["coupon_rate"] = float(numbers[1]) / 100 if float(numbers[1]) > 1 else float(numbers[1])
                    
            elif "option" in request:
                if len(numbers) >= 2:
                    parameters["spot_price"] = float(numbers[0])
                    parameters["strike_price"] = float(numbers[1])
                
                parameters["option_type"] = "call" if "call" in request.lower() else "put"
        
        elif calc_type == "graph_analysis":
            # Extract graph structure hints
            node_match = re.search(r'(\d+)\s*nodes?', request)
            edge_match = re.search(r'(\d+)\s*edges?', request)
            
            if node_match:
                parameters["num_nodes"] = int(node_match.group(1))
            if edge_match:
                parameters["num_edges"] = int(edge_match.group(1))
            
            parameters["directed"] = "directed" in request.lower()
            parameters["weighted"] = "weight" in request.lower()
        
        return parameters
    
    async def suggest_calculation_approach(self, request: str) -> Dict[str, Any]:
        """Suggest the best approach for a calculation request"""
        try:
            # Use Grok to suggest approach
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in computational methods. Suggest the best approach for calculations."
                },
                {
                    "role": "user",
                    "content": f"""For this calculation request: "{request}"
                    
                    Suggest:
                    1. The best computational approach
                    2. Required libraries or methods
                    3. Potential challenges
                    4. Alternative approaches
                    5. Estimated complexity
                    
                    Format as a structured analysis."""
                }
            ]
            
            response = self.grok_client.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=600
            )
            
            return {
                "success": True,
                "request": request,
                "suggestion": response.content,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to suggest calculation approach: {e}")
            return {
                "success": False,
                "error": str(e)
            }