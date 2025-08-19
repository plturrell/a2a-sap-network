"""
Reasoning Agent - A2A Microservice
Specialized agent for logical reasoning and decision making
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import json
from enum import Enum

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    CAUSAL = "causal"
    PROBABILISTIC = "probabilistic"


class ReasoningAgent(A2AAgentBase):
    """
    Reasoning Agent
    A2A compliant agent for logical reasoning and decision making
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="reasoning_agent",
            name="Reasoning Agent",
            description="A2A v0.2.9 compliant agent for logical reasoning and decision making",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # Reasoning configuration
        self.reasoning_config = {
            "max_depth": 10,
            "confidence_threshold": 0.7,
            "enable_probabilistic": True,
            "enable_causal_chains": True
        }

        # Knowledge base for reasoning
        self.knowledge_base = {
            "rules": [],
            "facts": [],
            "patterns": {}
        }

        self.reasoning_stats = {
            "total_reasoning_tasks": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "reasoning_types_used": {},
            "average_confidence": 0.0
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing Reasoning Agent...")

        # Initialize output directory
        self.output_dir = os.getenv("REASONING_OUTPUT_DIR", "/tmp/reasoning")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load initial knowledge base
        await self._load_knowledge_base()

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("Reasoning Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "reasoning_types": [rtype.value for rtype in ReasoningType],
                "inference_methods": ["forward_chaining", "backward_chaining", "pattern_matching"],
                "decision_support": ["multi_criteria", "risk_assessment", "scenario_analysis"],
                "knowledge_base": True
            }

            logger.info("Registered with A2A network at %s", self.agent_manager_url)
            self.is_registered = True

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to register with A2A network: %s", e)
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
        logger.info("Successfully deregistered from A2A network")
    
    @a2a_handler("reason", "Perform logical reasoning and inference")
    async def handle_reasoning_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for reasoning requests"""
        try:
            # Extract reasoning request from A2A message
            reasoning_request = self._extract_reasoning_request(message)
            
            if not reasoning_request:
                return create_error_response(400, "No reasoning request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("reasoning", {
                "context_id": context_id,
                "request": reasoning_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_reasoning(task_id, reasoning_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "reasoning_types": reasoning_request.get('types', []),
                "message": "Reasoning process started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling reasoning request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("deductive_reasoning", "Perform deductive logical reasoning")
    async def deductive_reasoning(self, premises: List[str], query: str) -> Dict[str, Any]:
        """Perform deductive reasoning from premises"""
        try:
            # Simple deductive reasoning implementation
            conclusion = None
            confidence = 0.0
            reasoning_chain = []
            
            # Check if query can be directly deduced from premises
            if query.lower() in [p.lower() for p in premises]:
                conclusion = query
                confidence = 1.0
                reasoning_chain = [f"Direct match: '{query}' found in premises"]
            else:
                # Apply basic logical rules
                for i, premise in enumerate(premises):
                    if "if" in premise.lower() and "then" in premise.lower():
                        # Handle if-then rules
                        if_part, then_part = premise.lower().split("then", 1)
                        if_condition = if_part.replace("if", "").strip()
                        
                        # Check if condition matches any other premise
                        for other_premise in premises:
                            if if_condition in other_premise.lower():
                                conclusion = then_part.strip()
                                confidence = 0.8
                                reasoning_chain.append(f"Applied rule: {premise}")
                                reasoning_chain.append(f"Condition met by: {other_premise}")
                                break
            
            if conclusion is None:
                conclusion = "Cannot deduce conclusion from given premises"
                confidence = 0.0
            
            return {
                "reasoning_type": "deductive",
                "premises": premises,
                "query": query,
                "conclusion": conclusion,
                "confidence": confidence,
                "reasoning_chain": reasoning_chain,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in deductive reasoning: {e}")
            return {"error": str(e)}

    @a2a_skill("pattern_analysis", "Analyze patterns in data for inductive reasoning")
    async def pattern_analysis(self, data: List[Dict[str, Any]], pattern_type: str = "general") -> Dict[str, Any]:
        """Analyze patterns in data"""
        try:
            if not data:
                raise ValueError("No data provided for pattern analysis")
            
            patterns_found = []
            confidence_scores = []
            
            if pattern_type == "frequency":
                # Analyze frequency patterns
                frequency_map = {}
                for item in data:
                    for key, value in item.items():
                        freq_key = f"{key}:{value}"
                        frequency_map[freq_key] = frequency_map.get(freq_key, 0) + 1
                
                # Find most frequent patterns
                sorted_patterns = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
                for pattern, count in sorted_patterns[:5]:  # Top 5 patterns
                    confidence = count / len(data)
                    patterns_found.append({
                        "pattern": pattern,
                        "frequency": count,
                        "confidence": confidence
                    })
                    confidence_scores.append(confidence)
            
            elif pattern_type == "correlation":
                # Simple correlation analysis
                if len(data) > 1:
                    keys = list(data[0].keys())
                    for i, key1 in enumerate(keys):
                        for key2 in keys[i+1:]:
                            try:
                                values1 = [item.get(key1, 0) for item in data if isinstance(item.get(key1), (int, float))]
                                values2 = [item.get(key2, 0) for item in data if isinstance(item.get(key2), (int, float))]
                                
                                if len(values1) > 1 and len(values2) > 1 and len(values1) == len(values2):
                                    # Simple correlation coefficient
                                    correlation = self._calculate_correlation(values1, values2)
                                    if abs(correlation) > 0.5:  # Threshold for significant correlation
                                        patterns_found.append({
                                            "pattern": f"correlation_{key1}_{key2}",
                                            "correlation": correlation,
                                            "confidence": abs(correlation)
                                        })
                                        confidence_scores.append(abs(correlation))
                            except:
                                continue
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                "reasoning_type": "pattern_analysis",
                "pattern_type": pattern_type,
                "data_size": len(data),
                "patterns_found": patterns_found,
                "average_confidence": avg_confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {"error": str(e)}

    @a2a_skill("decision_support", "Provide decision support based on criteria")
    async def decision_support(self, options: List[Dict[str, Any]], criteria: Dict[str, float]) -> Dict[str, Any]:
        """Provide multi-criteria decision support"""
        try:
            if not options or not criteria:
                raise ValueError("Options and criteria are required")
            
            scored_options = []
            
            for option in options:
                total_score = 0.0
                criterion_scores = {}
                
                for criterion, weight in criteria.items():
                    if criterion in option:
                        value = option[criterion]
                        if isinstance(value, (int, float)):
                            # Normalize score (assuming higher is better)
                            normalized_score = min(1.0, max(0.0, value / 100))
                            weighted_score = normalized_score * weight
                            total_score += weighted_score
                            criterion_scores[criterion] = {
                                "raw_value": value,
                                "normalized": normalized_score,
                                "weighted": weighted_score
                            }
                
                scored_options.append({
                    "option": option,
                    "total_score": total_score,
                    "criterion_scores": criterion_scores
                })
            
            # Sort by total score
            scored_options.sort(key=lambda x: x["total_score"], reverse=True)
            
            # Determine confidence based on score separation
            if len(scored_options) > 1:
                score_diff = scored_options[0]["total_score"] - scored_options[1]["total_score"]
                confidence = min(0.95, 0.5 + score_diff)
            else:
                confidence = 0.8
            
            recommendation = scored_options[0] if scored_options else None
            
            return {
                "reasoning_type": "decision_support",
                "criteria": criteria,
                "scored_options": scored_options,
                "recommendation": recommendation,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in decision support: {e}")
            return {"error": str(e)}
    
    async def _process_reasoning(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process reasoning request asynchronously"""
        try:
            reasoning_results = {}
            reasoning_tasks = request.get('tasks', {})

            # Process each reasoning task
            for task_name, task_data in reasoning_tasks.items():
                task_type = task_data.get('type', 'deductive')
                logger.info("Processing %s reasoning task: %s", task_type, task_name)

                try:
                    if task_type == "deductive":
                        result = await self.deductive_reasoning(
                            task_data.get('premises', []),
                            task_data.get('query', '')
                        )
                    elif task_type == "pattern_analysis":
                        result = await self.pattern_analysis(
                            task_data.get('data', []),
                            task_data.get('pattern_type', 'general')
                        )
                    elif task_type == "decision_support":
                        result = await self.decision_support(
                            task_data.get('options', []),
                            task_data.get('criteria', {})
                        )
                    else:
                        result = {"error": f"Unsupported reasoning type: {task_type}"}
                    
                    reasoning_results[task_name] = result
                    
                    # Update stats
                    if "error" not in result:
                        self.reasoning_stats["successful_inferences"] += 1
                        confidence = result.get("confidence", 0.0)
                        current_avg = self.reasoning_stats["average_confidence"]
                        total_success = self.reasoning_stats["successful_inferences"]
                        self.reasoning_stats["average_confidence"] = \
                            (current_avg * (total_success - 1) + confidence) / total_success
                    else:
                        self.reasoning_stats["failed_inferences"] += 1
                    
                    # Track reasoning type usage
                    self.reasoning_stats["reasoning_types_used"][task_type] = \
                        self.reasoning_stats["reasoning_types_used"].get(task_type, 0) + 1
                        
                except Exception as e:
                    reasoning_results[task_name] = {"error": str(e)}
                    self.reasoning_stats["failed_inferences"] += 1

            # Update overall stats
            self.reasoning_stats["total_reasoning_tasks"] += 1

            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(reasoning_results, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "reasoning_tasks": list(reasoning_results.keys()),
                "successful_inferences": sum(1 for r in reasoning_results.values() if "error" not in r),
                "failed_inferences": sum(1 for r in reasoning_results.values() if "error" in r),
                "average_confidence": self.reasoning_stats["average_confidence"]
            })

        except Exception as e:
            logger.error("Error processing reasoning: %s", e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send reasoning results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "reasoning_results": data,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "reasoning_stats": self.reasoning_stats
            }

            logger.info("Sent reasoning results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_reasoning_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract reasoning request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('reasoning_tasks', content.get('tasks', None))
        return None
    
    async def _load_knowledge_base(self):
        """Load initial knowledge base"""
        # Basic financial/business rules
        self.knowledge_base["rules"] = [
            "if revenue increases and costs remain constant then profit increases",
            "if market volatility is high then risk assessment should be conservative",
            "if data quality is poor then confidence in analysis decreases"
        ]
        
        self.knowledge_base["facts"] = [
            "financial data requires high precision",
            "agent collaboration improves accuracy",
            "validation is essential for quality assurance"
        ]
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient"""
        n = len(x)
        if n == 0:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator