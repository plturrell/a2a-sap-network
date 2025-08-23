"""
import time
Grok Mathematical Client - Enhanced AI for Mathematical Understanding
Extends the base Grok client with specialized mathematical capabilities
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import re

from .grokClient import GrokClient, GrokConfig, GrokResponse

logger = logging.getLogger(__name__)

class GrokMathematicalClient(GrokClient):
    """Extended Grok client with mathematical specialization"""
    
    def __init__(self, config: Optional[GrokConfig] = None):
        super().__init__(config)
        self.math_system_prompt = """You are a world-class mathematician and AI assistant specializing in understanding natural language mathematical queries.
You excel at:
1. Parsing complex mathematical expressions from natural language
2. Identifying mathematical operations and their parameters
3. Providing step-by-step solutions
4. Explaining mathematical concepts clearly
5. Validating mathematical results

Always respond with structured, precise information that can be processed by computational systems."""
    
    async def analyze_mathematical_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a mathematical query with enhanced AI understanding"""
        
        prompt = f"""Analyze this mathematical query and provide a comprehensive structured response.

Query: "{query}"
Context: {json.dumps(context or {}, indent=2)}

Provide a JSON response with the following structure:
{{
    "operation_type": "derivative|integral|solve|limit|series|evaluate|optimize|matrix|statistics|geometry|finance",
    "mathematical_expression": "the core mathematical expression extracted and normalized",
    "variables": ["list", "of", "variables"],
    "parameters": {{
        "any_specific_parameters": "like limits, points, etc"
    }},
    "confidence": 0.95,
    "explanation": "Brief explanation of what the user wants",
    "mathematical_notation": "expression in proper mathematical notation",
    "suggested_approach": "How to solve this problem step by step",
    "potential_difficulties": ["list of challenges"],
    "clarification_needed": ["any ambiguities that need clarification"],
    "alternative_interpretations": ["other possible interpretations of the query"]
}}

Be extremely precise in extracting the mathematical intent."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.async_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            # Parse JSON from response
            content = response.content
            
            # Try to extract JSON even if wrapped in text
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    fixed_json = json_match.group()
                    fixed_json = re.sub(r',\s*}', '}', fixed_json)  # Remove trailing commas
                    fixed_json = re.sub(r',\s*]', ']', fixed_json)
                    try:
                        return json.loads(fixed_json)
                    except:
                        pass
            
            # Fallback parsing
            return self._parse_mathematical_response(content)
            
        except Exception as e:
            logger.error(f"Mathematical query analysis failed: {e}")
            return {
                "operation_type": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "raw_query": query
            }
    
    async def generate_step_by_step_solution(self, 
                                           query: str, 
                                           analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed step-by-step solution for a mathematical problem"""
        
        prompt = f"""Create a detailed step-by-step solution for this mathematical problem.

Problem: "{query}"
Analysis: {json.dumps(analysis, indent=2)}

Provide a JSON response with:
{{
    "problem_statement": "Clear statement of the problem",
    "solution_strategy": "Overall approach to solving",
    "prerequisites": ["mathematical concepts needed"],
    "steps": [
        {{
            "step_number": 1,
            "description": "What to do in this step",
            "mathematical_operation": "The specific calculation",
            "formula_used": "Any formula applied",
            "result": "Result of this step",
            "explanation": "Why this step is necessary",
            "common_mistakes": ["potential errors to avoid"]
        }}
    ],
    "final_answer": "The complete solution",
    "verification": {{
        "method": "How to verify the answer",
        "check_calculation": "Verification calculation",
        "is_verified": true
    }},
    "alternative_methods": ["other ways to solve this"],
    "visualization_suggestion": "How to visualize this problem"
}}

Be thorough and educational in your explanation."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.async_chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"steps": [{"description": content}], "raw_response": content}
            
            return {"solution_text": content}
            
        except Exception as e:
            logger.error(f"Step-by-step solution generation failed: {e}")
            return {"error": str(e), "query": query}
    
    async def validate_mathematical_result(self, 
                                         query: str, 
                                         calculated_result: Any,
                                         calculation_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a mathematical calculation result"""
        
        prompt = f"""Validate this mathematical calculation and verify the result.

Original Problem: "{query}"
Calculated Result: {calculated_result}
Calculation Steps: {json.dumps(calculation_steps, indent=2)}

Analyze and provide:
{{
    "is_correct": true/false,
    "confidence": 0.95,
    "verification_method": "How you verified this",
    "mathematical_proof": "Brief proof or verification",
    "edge_cases_considered": ["list of edge cases checked"],
    "numerical_accuracy": "Assessment of numerical precision",
    "alternative_verification": "Another way to check this",
    "potential_errors": ["possible mistakes in the calculation"],
    "suggestions": ["improvements to the solution method"]
}}

Be rigorous in your validation."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.async_chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for validation
                max_tokens=1000
            )
            
            # Parse response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # Fallback
            return {
                "is_correct": "uncertain",
                "confidence": 0.5,
                "validation_text": content
            }
            
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            return {"is_correct": "error", "error": str(e)}
    
    async def explain_mathematical_concept(self, 
                                         concept: str, 
                                         level: str = "intermediate",
                                         include_examples: bool = True) -> str:
        """Explain a mathematical concept at the specified level"""
        
        level_descriptions = {
            "beginner": "Explain using simple terms, avoiding technical jargon, with visual analogies",
            "intermediate": "Use standard mathematical terminology with clear explanations",
            "advanced": "Provide rigorous mathematical treatment with formal definitions"
        }
        
        prompt = f"""Explain the mathematical concept: {concept}

Level: {level} - {level_descriptions.get(level, level_descriptions['intermediate'])}

Include:
1. Clear definition
2. Why this concept is important
3. When it's used
4. Key properties and theorems
{"5. 2-3 practical examples with solutions" if include_examples else ""}
6. Common misconceptions and how to avoid them
7. Connections to other mathematical concepts
8. Visual or intuitive understanding

Make it engaging and educational."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.async_chat_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=1500
        )
        
        return response.content
    
    async def suggest_practice_problems(self, 
                                      topic: str, 
                                      difficulty: str,
                                      num_problems: int = 5) -> List[Dict[str, str]]:
        """Generate practice problems for a mathematical topic"""
        
        prompt = f"""Create {num_problems} practice problems for the topic: {topic}
Difficulty level: {difficulty} (easy/medium/hard)

For each problem, provide:
{{
    "problem_number": 1,
    "problem_statement": "The complete problem",
    "difficulty": "{difficulty}",
    "concepts_tested": ["list of concepts"],
    "hint": "A helpful hint without giving away the answer",
    "solution_approach": "Brief approach to solve",
    "estimated_time": "5-10 minutes",
    "similar_to": "Reference to similar standard problems"
}}

Return a JSON array of problems. Make them progressively challenging."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.async_chat_completion(
                messages=messages,
                temperature=0.6,
                max_tokens=1500
            )
            
            # Parse response
            content = response.content
            
            # Try to extract JSON array
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # Fallback: create simple problems
            return [{
                "problem_number": i + 1,
                "problem_statement": f"Practice problem {i + 1} for {topic}",
                "difficulty": difficulty,
                "hint": "Think about the fundamental concepts"
            } for i in range(num_problems)]
            
        except Exception as e:
            logger.error(f"Practice problem generation failed: {e}")
            return []
    
    async def provide_calculation_feedback(self, 
                                         user_solution: str,
                                         correct_solution: str,
                                         problem: str) -> Dict[str, Any]:
        """Provide detailed feedback on a user's solution"""
        
        prompt = f"""Compare the user's solution with the correct solution and provide educational feedback.

Problem: "{problem}"
User's Solution: {user_solution}
Correct Solution: {correct_solution}

Provide feedback including:
{{
    "correctness": "fully_correct|partially_correct|incorrect",
    "score": 85,
    "strengths": ["what the user did well"],
    "errors": [
        {{
            "type": "conceptual|computational|notation",
            "description": "What went wrong",
            "location": "Where in the solution",
            "correction": "How to fix it"
        }}
    ],
    "misconceptions": ["underlying misunderstandings"],
    "suggestions": ["how to improve"],
    "encouragement": "Positive, motivating message",
    "next_steps": ["what to study or practice next"]
}}

Be constructive and educational in your feedback."""
        
        messages = [
            {"role": "system", "content": self.math_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.async_chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            return {"feedback_text": content}
            
        except Exception as e:
            logger.error(f"Feedback generation failed: {e}")
            return {"error": str(e)}
    
    def _parse_mathematical_response(self, content: str) -> Dict[str, Any]:
        """Fallback parser for mathematical responses"""
        
        # Try to extract key information using patterns
        result = {
            "operation_type": "unknown",
            "confidence": 0.5,
            "raw_response": content
        }
        
        # Operation type detection
        operations = {
            "derivative": ["derivative", "differentiate", "d/dx", "differentiation"],
            "integral": ["integral", "integrate", "integration", "antiderivative"],
            "solve": ["solve", "equation", "find", "solution"],
            "limit": ["limit", "approaches", "tends to"],
            "series": ["series", "taylor", "maclaurin", "expansion"],
            "evaluate": ["calculate", "compute", "evaluate", "simplify"]
        }
        
        content_lower = content.lower()
        for op_type, keywords in operations.items():
            if any(keyword in content_lower for keyword in keywords):
                result["operation_type"] = op_type
                result["confidence"] = 0.7
                break
        
        # Try to extract expression
        expr_match = re.search(r'expression[:\s]+([^\n]+)', content, re.IGNORECASE)
        if expr_match:
            result["mathematical_expression"] = expr_match.group(1).strip()
        
        # Extract confidence if mentioned
        conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', content, re.IGNORECASE)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))
        
        return result

class GrokMathematicalAssistant:
    """High-level mathematical assistant using enhanced Grok client"""
    
    def __init__(self, client: Optional[GrokMathematicalClient] = None):
        self.client = client or GrokMathematicalClient()
        self.conversation_history = []
        self.problem_solving_context = {}
    
    async def interactive_problem_solving(self, user_input: str) -> Dict[str, Any]:
        """Interactive problem-solving session with the user"""
        
        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Analyze the query
        analysis = await self.client.analyze_mathematical_query(user_input, self.problem_solving_context)
        
        # Generate solution if confidence is high
        if analysis.get("confidence", 0) > 0.7:
            solution = await self.client.generate_step_by_step_solution(user_input, analysis)
            
            response = {
                "status": "ready_to_solve",
                "analysis": analysis,
                "solution": solution,
                "requires_computation": True
            }
        else:
            # Need clarification
            response = {
                "status": "needs_clarification",
                "analysis": analysis,
                "clarification_questions": analysis.get("clarification_needed", []),
                "suggestions": self._generate_clarification_suggestions(analysis)
            }
        
        # Update conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return response
    
    def _generate_clarification_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions for clarification"""
        
        suggestions = []
        
        if analysis.get("operation_type") == "unknown":
            suggestions.append("Please specify what mathematical operation you want to perform")
        
        if not analysis.get("variables"):
            suggestions.append("Please indicate which variables are involved")
        
        if "alternative_interpretations" in analysis:
            suggestions.append("Your query could mean several things. Please be more specific.")
        
        return suggestions
    
    async def teach_concept_interactively(self, concept: str, user_level: str = "intermediate") -> Dict[str, Any]:
        """Teach a mathematical concept interactively"""
        
        # Get explanation
        explanation = await self.client.explain_mathematical_concept(
            concept=concept,
            level=user_level,
            include_examples=True
        )
        
        # Generate practice problems
        practice_problems = await self.client.suggest_practice_problems(
            topic=concept,
            difficulty="easy" if user_level == "beginner" else "medium",
            num_problems=3
        )
        
        return {
            "concept": concept,
            "explanation": explanation,
            "practice_problems": practice_problems,
            "next_concepts": self._suggest_related_concepts(concept),
            "resources": self._get_learning_resources(concept)
        }
    
    def _suggest_related_concepts(self, concept: str) -> List[str]:
        """Suggest related mathematical concepts"""
        
        # Simple concept graph
        concept_relations = {
            "derivative": ["integral", "limit", "chain rule", "optimization"],
            "integral": ["derivative", "area", "fundamental theorem", "substitution"],
            "limit": ["continuity", "derivative", "infinite series", "l'hopital"],
            "matrix": ["determinant", "eigenvalues", "linear systems", "transformations"],
            "probability": ["statistics", "distributions", "bayes theorem", "expectation"]
        }
        
        return concept_relations.get(concept.lower(), ["advanced topics in " + concept])
    
    def _get_learning_resources(self, concept: str) -> List[Dict[str, str]]:
        """Get learning resources for a concept"""
        
        return [
            {
                "type": "video",
                "title": f"Visual Introduction to {concept}",
                "description": "Animated explanation with examples"
            },
            {
                "type": "practice",
                "title": f"{concept} Practice Problems",
                "description": "Interactive exercises with instant feedback"
            },
            {
                "type": "reference",
                "title": f"{concept} Quick Reference",
                "description": "Formulas and key concepts"
            }
        ]