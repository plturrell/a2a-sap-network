import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

from app.a2a.core.security_base import SecureA2AAgent
"""
Enhanced Intelligent Dispatch Skill with Advanced Natural Language Processing
Provides sophisticated routing and analysis of calculation requests
"""

# Import natural language processor
try:
    from .naturalLanguageParser import MathQueryProcessor, MathOperation


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    NL_PARSER_AVAILABLE = True
except ImportError:
    NL_PARSER_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedIntelligentDispatchSkill(SecureA2AAgent):
    """Enhanced intelligent dispatch with natural language understanding"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, grok_client=None):
        super().__init__()
        self.grok_client = grok_client
        self.nl_processor = MathQueryProcessor() if NL_PARSER_AVAILABLE else None
        self.skill_mappings = self._initialize_skill_mappings()
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
    def _initialize_skill_mappings(self) -> Dict[str, str]:
        """Map mathematical operations to agent skills"""
        return {
            "evaluate": "evaluate_calculation",
            "derivative": "differentiate_calculation", 
            "integral": "integrate_calculation",
            "solve": "solve_calculation",
            "limit": "intelligent_dispatch",  # Handle via enhanced skills
            "series": "intelligent_dispatch",  # Handle via enhanced skills  
            "simplify": "simplify_calculation",
            "factor": "simplify_calculation",
            "expand": "simplify_calculation",
            "optimization": "mathematical_optimization",
            "matrix": "networkx_graph_analysis",  # For matrix operations
            "geometry": "intelligent_dispatch",
            "statistics": "intelligent_dispatch", 
            "finance": "financial_calculation",
            "differential_equation": "intelligent_dispatch"
        }
    
    async def analyze_and_dispatch(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze natural language request and dispatch to appropriate skill
        """
        try:
            # Step 1: Use advanced NL parser if available
            if self.nl_processor:
                analysis_result = await self._analyze_with_nl_parser(request, context)
            else:
                # Fallback to simple pattern matching
                analysis_result = await self._analyze_with_patterns(request, context)
            
            # Step 2: Enhance with AI if available
            if self.grok_client and analysis_result["confidence"] < self.confidence_thresholds["high"]:
                ai_enhancement = await self._enhance_with_ai(request, analysis_result, context)
                analysis_result = self._merge_analysis_results(analysis_result, ai_enhancement)
            
            # Step 3: Determine final dispatch decision
            dispatch_decision = self._make_dispatch_decision(analysis_result)
            
            return {
                "success": True,
                "skill": dispatch_decision["skill"],
                "parameters": dispatch_decision["parameters"],
                "confidence": dispatch_decision["confidence"],
                "analysis": analysis_result,
                "method": "enhanced_nl_dispatch"
            }
            
        except Exception as e:
            logger.error(f"Enhanced dispatch analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_skill": "evaluate_calculation",
                "method": "error_fallback"
            }
    
    async def _analyze_with_nl_parser(self, request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze request using advanced natural language parser"""
        processed_query = self.nl_processor.process_query(request)
        
        parsed_query = processed_query["parsed_query"]
        operation = parsed_query["operation"]
        
        # Build comprehensive analysis
        analysis = {
            "operation_detected": operation,
            "confidence": parsed_query["confidence"],
            "parsed_expression": parsed_query["expression"],
            "variables": parsed_query["variables"],
            "parameters": parsed_query["parameters"],
            "original_query": parsed_query["original_query"],
            "parsed_components": parsed_query["parsed_components"],
            "context_info": processed_query["context"],
            "suggestions": processed_query["suggestions"],
            "validation": processed_query["validation"],
            "complexity_score": processed_query["context"].get("complexity_score", 0.5),
            "analysis_method": "advanced_nl_parser"
        }
        
        # Add confidence adjustments based on validation
        if not processed_query["validation"]["is_valid"]:
            analysis["confidence"] *= 0.7
            analysis["validation_issues"] = processed_query["validation"]["issues"]
        
        # Boost confidence for high-quality parses
        if (parsed_query["confidence"] > 0.8 and 
            processed_query["validation"]["confidence_level"] == "high"):
            analysis["confidence"] = min(1.0, analysis["confidence"] + 0.1)
        
        return analysis
    
    async def _analyze_with_patterns(self, request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback pattern-based analysis"""
        request_lower = request.lower()
        
        # Simple pattern matching for operation detection
        operation_patterns = {
            "derivative": ["derivative", "differentiate", "d/dx", "diff"],
            "integral": ["integral", "integrate", "∫", "area under"],
            "solve": ["solve", "equation", "=", "find x", "roots"],
            "limit": ["limit", "lim", "approaches", "tends to"],
            "series": ["series", "taylor", "maclaurin", "expansion"],
            "simplify": ["simplify", "factor", "expand", "reduce"],
            "evaluate": ["calculate", "compute", "evaluate", "what is"]
        }
        
        detected_operation = "evaluate"  # Default
        confidence = 0.3  # Low confidence for pattern matching
        
        for operation, patterns in operation_patterns.items():
            for pattern in patterns:
                if pattern in request_lower:
                    detected_operation = operation
                    confidence = 0.6
                    break
            if confidence > 0.3:
                break
        
        return {
            "operation_detected": detected_operation,
            "confidence": confidence,
            "parsed_expression": request,
            "variables": self._extract_variables_simple(request),
            "parameters": {},
            "analysis_method": "simple_patterns",
            "complexity_score": min(len(request) / 100, 1.0)
        }
    
    async def _enhance_with_ai(self, request: str, initial_analysis: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance analysis using AI (Grok) assistance"""
        try:
            # Prepare prompt for AI analysis
            prompt = f"""
            Analyze this mathematical request and provide structured information:
            
            Request: "{request}"
            Initial Analysis: {json.dumps(initial_analysis, indent=2)}
            Context: {json.dumps(context or {}, indent=2)}
            
            Please provide:
            1. Mathematical operation type (derivative, integral, solve, evaluate, etc.)
            2. Main mathematical expression
            3. Variables involved
            4. Confidence level (0.0-1.0)
            5. Any additional parameters needed
            6. Suggestions for improvement
            
            Respond in JSON format.
            """
            
            ai_response = await self.grok_client.generate_completion(
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            # Parse AI response
            try:
                ai_analysis = json.loads(ai_response)
                return {
                    "ai_operation": ai_analysis.get("operation_type", initial_analysis["operation_detected"]),
                    "ai_confidence": float(ai_analysis.get("confidence", 0.5)),
                    "ai_expression": ai_analysis.get("expression", initial_analysis["parsed_expression"]),
                    "ai_variables": ai_analysis.get("variables", []),
                    "ai_parameters": ai_analysis.get("parameters", {}),
                    "ai_suggestions": ai_analysis.get("suggestions", []),
                    "analysis_method": "ai_enhanced"
                }
            except json.JSONDecodeError:
                # Fallback if AI doesn't return valid JSON
                return {
                    "ai_analysis_text": ai_response,
                    "analysis_method": "ai_text_only",
                    "ai_confidence": 0.4
                }
                
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return {
                "ai_enhancement_error": str(e),
                "analysis_method": "ai_failed"
            }
    
    def _merge_analysis_results(self, nl_analysis: Dict[str, Any], ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Merge natural language and AI analysis results"""
        merged = nl_analysis.copy()
        
        # Weighted confidence combination
        nl_weight = 0.7
        ai_weight = 0.3
        
        if "ai_confidence" in ai_analysis:
            combined_confidence = (
                nl_analysis["confidence"] * nl_weight + 
                ai_analysis["ai_confidence"] * ai_weight
            )
            merged["confidence"] = combined_confidence
        
        # Override operation if AI has high confidence and disagrees
        if ("ai_operation" in ai_analysis and 
            ai_analysis.get("ai_confidence", 0) > 0.8 and
            ai_analysis["ai_operation"] != nl_analysis["operation_detected"]):
            merged["operation_detected"] = ai_analysis["ai_operation"]
            merged["operation_source"] = "ai_override"
        
        # Merge suggestions
        merged["ai_suggestions"] = ai_analysis.get("ai_suggestions", [])
        all_suggestions = merged.get("suggestions", []) + merged["ai_suggestions"]
        merged["suggestions"] = list(set(all_suggestions))  # Remove duplicates
        
        # Add AI analysis details
        merged["ai_analysis"] = ai_analysis
        
        return merged
    
    def _make_dispatch_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make final dispatch decision based on analysis"""
        operation = analysis["operation_detected"]
        confidence = analysis["confidence"]
        
        # Map operation to skill
        skill_name = self.skill_mappings.get(operation, "evaluate_calculation")
        
        # Prepare parameters for the skill
        parameters = {
            "expression": analysis["parsed_expression"],
            "type": operation
        }
        
        # Add operation-specific parameters
        if operation == "derivative":
            parameters["variable"] = analysis.get("variables", ["x"])[0] if analysis.get("variables") else "x"
            
        elif operation == "integral":
            if "parameters" in analysis and "limits" in analysis["parameters"]:
                parameters["limits"] = analysis["parameters"]["limits"]
                
        elif operation == "solve":
            parameters["variables"] = analysis.get("variables", ["x"])
            if "=" in analysis["parsed_expression"]:
                parameters["equations"] = [analysis["parsed_expression"]]
            
        elif operation == "limit":
            if "parameters" in analysis:
                parameters.update(analysis["parameters"])
                
        elif operation == "series":
            if "parameters" in analysis:
                parameters.update(analysis["parameters"])
        
        # Add context information
        if "context_info" in analysis:
            parameters["context"] = analysis["context_info"]
        
        # Adjust confidence based on parameter completeness
        if len(parameters) > 2:  # More than just expression and type
            confidence = min(1.0, confidence + 0.05)
        
        return {
            "skill": skill_name,
            "parameters": parameters,
            "confidence": confidence,
            "dispatch_reasoning": f"Operation '{operation}' mapped to skill '{skill_name}' with {confidence:.1%} confidence"
        }
    
    def _extract_variables_simple(self, text: str) -> List[str]:
        """Simple variable extraction for fallback"""
        variables = set()
        
        # Look for single letters that might be variables
        pattern = r'\b([a-zA-Z])\b(?!\s*\()'
        matches = re.findall(pattern, text)
        
        common_vars = ['x', 'y', 'z', 't', 'a', 'b', 'c', 'n', 'm']
        for match in matches:
            if match.lower() in common_vars:
                variables.add(match.lower())
        
        return sorted(list(variables))
    
    async def suggest_calculation_approach(self, request: str) -> Dict[str, Any]:
        """Suggest the best approach for a calculation without executing"""
        try:
            # Analyze the request
            if self.nl_processor:
                processed_query = self.nl_processor.process_query(request)
                
                return {
                    "success": True,
                    "suggested_operation": processed_query["parsed_query"]["operation"],
                    "confidence": processed_query["parsed_query"]["confidence"],
                    "parsed_expression": processed_query["parsed_query"]["expression"],
                    "suggestions": processed_query["suggestions"],
                    "methodology_hints": self._get_methodology_hints(processed_query["parsed_query"]["operation"]),
                    "complexity_assessment": self._assess_complexity(processed_query),
                    "required_tools": self._get_required_tools(processed_query["parsed_query"]["operation"])
                }
            else:
                return {
                    "success": False,
                    "error": "Advanced analysis not available",
                    "fallback": "Try rephrasing with more specific mathematical terms"
                }
                
        except Exception as e:
            logger.error(f"Approach suggestion failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_methodology_hints(self, operation: str) -> List[str]:
        """Get methodology hints for different operations"""
        hints = {
            "derivative": [
                "Use chain rule for composite functions",
                "Apply product rule for products of functions", 
                "Use quotient rule for ratios of functions",
                "Consider implicit differentiation for implicit functions"
            ],
            "integral": [
                "Try substitution method for complex expressions",
                "Use integration by parts for products",
                "Consider partial fractions for rational functions",
                "Check for standard integral forms"
            ],
            "solve": [
                "Factor expressions when possible",
                "Use quadratic formula for second-degree equations",
                "Consider substitution for complex equations",
                "Graph functions to understand behavior"
            ],
            "limit": [
                "Check for direct substitution first",
                "Use L'Hôpital's rule for indeterminate forms",
                "Consider algebraic manipulation",
                "Analyze behavior near the limit point"
            ],
            "series": [
                "Identify the function type for appropriate series",
                "Determine convergence radius",
                "Choose appropriate expansion point",
                "Consider truncation error"
            ]
        }
        
        return hints.get(operation, ["Analyze the problem structure", "Break down into simpler components"])
    
    def _assess_complexity(self, processed_query: Dict[str, Any]) -> Dict[str, str]:
        """Assess the complexity of the mathematical problem"""
        complexity_score = processed_query["context"].get("complexity_score", 0.5)
        
        if complexity_score < 0.3:
            level = "Low"
            description = "Simple arithmetic or basic algebraic operations"
        elif complexity_score < 0.6:
            level = "Medium" 
            description = "Intermediate calculus or algebraic manipulation"
        elif complexity_score < 0.8:
            level = "High"
            description = "Advanced mathematical concepts or multi-step solutions"
        else:
            level = "Very High"
            description = "Complex mathematical analysis or specialized techniques required"
        
        return {
            "level": level,
            "score": f"{complexity_score:.2f}",
            "description": description
        }
    
    def _get_required_tools(self, operation: str) -> List[str]:
        """Get list of mathematical tools/libraries required"""
        tool_requirements = {
            "derivative": ["SymPy", "Calculus knowledge"],
            "integral": ["SymPy", "Integration techniques"],
            "solve": ["SymPy", "Algebraic manipulation"],
            "limit": ["SymPy", "Limit analysis"],
            "series": ["SymPy", "Series analysis"],
            "optimization": ["SciPy", "Optimization algorithms"],
            "matrix": ["NumPy", "Linear algebra"],
            "statistics": ["NumPy", "SciPy", "Statistical methods"],
            "finance": ["QuantLib", "Financial mathematics"],
            "geometry": ["SymPy Geometry", "Analytical geometry"]
        }
        
        return tool_requirements.get(operation, ["Basic mathematical computation"])