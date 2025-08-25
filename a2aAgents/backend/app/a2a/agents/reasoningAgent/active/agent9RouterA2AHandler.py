"""
A2A-Compliant Message Handler for Agent 9 - Reasoning
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .comprehensiveReasoningAgentSdk import ComprehensiveReasoningAgentSdk

logger = logging.getLogger(__name__)


class Agent9RouterA2AHandler(SecureA2AAgent):
    """
    A2A-compliant handler for Agent 9 - Reasoning
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: ComprehensiveReasoningAgentSdk):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="logical_reasoning_agent",
            agent_name="Logical Reasoning Agent",
            agent_version="2.0.0",
            allowed_operations={
                "get_agent_card",
                # Registry capabilities
                "logical_reasoning",
                "inference_generation",
                "decision_making",
                "knowledge_synthesis",
                "problem_solving",
                # Enhanced operations
                "perform_logical_reasoning",
                "generate_inferences_enhanced",
                "make_decisions_enhanced",
                "synthesize_knowledge_enhanced",
                "solve_problems_enhanced",
                "json_rpc",
                "create_reasoning_task",
                "list_reasoning_tasks",
                "start_reasoning",
                "validate_conclusion",
                "explain_reasoning",
                "add_knowledge",
                "validate_knowledge_base",
                "generate_inferences",
                "make_decision",
                "solve_problem",
                "get_dashboard_data",
                "get_reasoning_options",
                "health_check"
            },
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=100,
            rate_limit_window=60
        )

        super().__init__(config)

        self.agent_sdk = agent_sdk

        # Initialize A2A blockchain client
        self.a2a_client = A2ANetworkClient(
            agent_id=config.agent_id,
            private_key=os.getenv('A2A_PRIVATE_KEY'),
            rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
        )

        # Register message handlers
        self._register_handlers()

        logger.info(f"A2A-compliant handler initialized for {config.agent_name}")

    def _register_handlers(self):
        """Register A2A message handlers"""

        @self.secure_handler("get_agent_card")
        async def handle_get_agent_card(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get agent card information"""
            try:
                agent_card = await self.agent_sdk.get_agent_card()
                result = agent_card

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_agent_card",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_agent_card: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("json_rpc")
        async def handle_json_rpc(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle json_rpc operation"""
            try:
                # Process JSON-RPC request through agent SDK
                result = await self.agent_sdk.handle_json_rpc(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="json_rpc",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to json_rpc: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("create_reasoning_task")
        async def handle_create_reasoning_task(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle create_reasoning_task operation"""
            try:
                # Create a new reasoning task
                task_data = {
                    "task_id": data.get("task_id", str(uuid.uuid4())),
                    "task_type": data.get("task_type", "logical_reasoning"),
                    "query": data.get("query", ""),
                    "premises": data.get("premises", []),
                    "domain": data.get("domain", "general"),
                    "priority": data.get("priority", "medium"),
                    "metadata": data.get("metadata", {})
                }
                
                # Use agent SDK to create reasoning task
                result = await self.agent_sdk.create_reasoning_chain({
                    "chain_id": task_data["task_id"],
                    "reasoning_type": task_data["task_type"],
                    "initial_query": task_data["query"],
                    "domain": task_data["domain"],
                    "premises": task_data["premises"]
                })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="create_reasoning_task",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to create_reasoning_task: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("list_reasoning_tasks")
        async def handle_list_reasoning_tasks(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle list_reasoning_tasks operation"""
            try:
                # List active reasoning tasks
                active_chains = self.agent_sdk.get_active_chains()
                tasks = []
                
                for chain_id, chain_data in active_chains.items():
                    tasks.append({
                        "task_id": chain_id,
                        "task_type": chain_data.get("reasoning_type", "unknown"),
                        "status": "active" if chain_data.get("active", True) else "completed",
                        "created_at": chain_data.get("created_at", ""),
                        "steps_completed": len(chain_data.get("reasoning_steps", [])),
                        "confidence": chain_data.get("confidence_scores", {}).get("overall", 0.0)
                    })
                
                # Sort by creation time (newest first)
                tasks.sort(key=lambda x: x["created_at"], reverse=True)
                
                result = {
                    "tasks": tasks,
                    "total_count": len(tasks),
                    "active_count": sum(1 for t in tasks if t["status"] == "active")
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="list_reasoning_tasks",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to list_reasoning_tasks: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("start_reasoning")
        async def handle_start_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle start_reasoning operation"""
            try:
                # Start reasoning process using logical reasoning skill
                reasoning_query = data.get("query", "")
                reasoning_type = data.get("reasoning_type", "deductive")
                premises = data.get("premises", [])
                domain = data.get("domain", "general")

                result = await self.agent_sdk.logical_reasoning({
                    "query": reasoning_query,
                    "reasoning_type": reasoning_type,
                    "domain": domain,
                    "premises": premises,
                    "context": data.get("context", {})
                })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="start_reasoning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to start_reasoning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_conclusion")
        async def handle_validate_conclusion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validate_conclusion operation"""
            try:
                # Validate conclusion using confidence assessment
                conclusion = data.get("conclusion", "")
                evidence = data.get("evidence", [])
                reasoning_type = data.get("reasoning_type", "deductive")
                chain_id = data.get("chain_id")

                result = await self.agent_sdk.confidence_assessment({
                    "chain_id": chain_id,
                    "conclusion": conclusion,
                    "evidence": evidence,
                    "reasoning_type": reasoning_type
                })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_conclusion",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_conclusion: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("explain_reasoning")
        async def handle_explain_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle explain_reasoning operation"""
            try:
                # Explain reasoning process
                chain_id = data.get("chain_id") or data.get("task_id")
                step_index = data.get("step_index", -1)  # -1 for full explanation
                
                if not chain_id:
                    return self.create_secure_response(
                        "No chain_id or task_id provided",
                        status="error"
                    )
                
                # Get reasoning chain
                chain_data = self.agent_sdk.reasoning_chains.get(chain_id)
                if not chain_data:
                    return self.create_secure_response(
                        f"Reasoning chain {chain_id} not found",
                        status="error"
                    )
                
                # Build explanation
                explanation = {
                    "chain_id": chain_id,
                    "reasoning_type": chain_data.get("reasoning_type", "unknown"),
                    "initial_query": chain_data.get("initial_query", ""),
                    "premises": chain_data.get("premises", []),
                    "reasoning_steps": []
                }
                
                # Add step-by-step reasoning
                steps = chain_data.get("reasoning_steps", [])
                if step_index >= 0 and step_index < len(steps):
                    # Explain specific step
                    step = steps[step_index]
                    explanation["reasoning_steps"] = [{
                        "step_index": step_index,
                        "type": step.get("type", ""),
                        "input": step.get("input", ""),
                        "reasoning": step.get("reasoning", ""),
                        "output": step.get("output", ""),
                        "confidence": step.get("confidence", 0.0),
                        "evidence": step.get("evidence", [])
                    }]
                else:
                    # Explain all steps
                    for idx, step in enumerate(steps):
                        explanation["reasoning_steps"].append({
                            "step_index": idx,
                            "type": step.get("type", ""),
                            "reasoning": step.get("reasoning", ""),
                            "confidence": step.get("confidence", 0.0)
                        })
                
                # Add conclusion if available
                if chain_data.get("conclusion"):
                    explanation["conclusion"] = {
                        "statement": chain_data["conclusion"],
                        "confidence": chain_data.get("confidence_scores", {}).get("overall", 0.0),
                        "supporting_evidence": chain_data.get("supporting_evidence", [])
                    }
                
                result = explanation

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="explain_reasoning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to explain_reasoning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("add_knowledge")
        async def handle_add_knowledge(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle add_knowledge operation"""
            try:
                # Add knowledge to the reasoning agent's knowledge base
                knowledge_item = {
                    "id": data.get("id", str(uuid.uuid4())),
                    "type": data.get("type", "fact"),  # fact, rule, constraint, axiom
                    "domain": data.get("domain", "general"),
                    "content": data.get("content", ""),
                    "confidence": data.get("confidence", 1.0),
                    "source": data.get("source", "user_provided"),
                    "metadata": data.get("metadata", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Validate knowledge item
                if not knowledge_item["content"]:
                    return self.create_secure_response(
                        "Knowledge content cannot be empty",
                        status="error"
                    )
                
                # Add to knowledge graph
                domain_knowledge = self.agent_sdk.knowledge_graph.setdefault(
                    knowledge_item["domain"], []
                )
                domain_knowledge.append(knowledge_item)
                
                # Update embeddings if available
                if hasattr(self.agent_sdk, 'knowledge_embeddings'):
                    embedding = self.agent_sdk._generate_embedding(knowledge_item["content"])
                    self.agent_sdk.knowledge_embeddings[knowledge_item["id"]] = embedding
                
                result = {
                    "status": "success",
                    "knowledge_id": knowledge_item["id"],
                    "domain": knowledge_item["domain"],
                    "type": knowledge_item["type"],
                    "message": f"Knowledge added to {knowledge_item['domain']} domain"
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="add_knowledge",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to add_knowledge: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("validate_knowledge_base")
        async def handle_validate_knowledge_base(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle validate_knowledge_base operation"""
            try:
                # Validate knowledge base consistency and integrity
                domain = data.get("domain", "all")
                validation_type = data.get("validation_type", "full")  # full, consistency, completeness
                
                validation_results = {
                    "domain": domain,
                    "validation_type": validation_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "issues": [],
                    "statistics": {},
                    "recommendations": []
                }
                
                # Get domains to validate
                domains_to_check = [domain] if domain != "all" else list(self.agent_sdk.knowledge_graph.keys())
                
                total_items = 0
                inconsistencies = 0
                
                for check_domain in domains_to_check:
                    domain_knowledge = self.agent_sdk.knowledge_graph.get(check_domain, [])
                    total_items += len(domain_knowledge)
                    
                    # Check for contradictions
                    for i, item1 in enumerate(domain_knowledge):
                        for j, item2 in enumerate(domain_knowledge[i+1:], i+1):
                            if self._check_contradiction(item1, item2):
                                inconsistencies += 1
                                validation_results["issues"].append({
                                    "type": "contradiction",
                                    "severity": "high",
                                    "domain": check_domain,
                                    "items": [item1["id"], item2["id"]],
                                    "description": f"Potential contradiction between items"
                                })
                    
                    # Check for incomplete rules
                    rules = [item for item in domain_knowledge if item["type"] == "rule"]
                    for rule in rules:
                        if not self._validate_rule_completeness(rule):
                            validation_results["issues"].append({
                                "type": "incomplete_rule",
                                "severity": "medium",
                                "domain": check_domain,
                                "item_id": rule["id"],
                                "description": "Rule missing required components"
                            })
                
                # Calculate statistics
                validation_results["statistics"] = {
                    "total_items": total_items,
                    "domains_checked": len(domains_to_check),
                    "inconsistencies_found": inconsistencies,
                    "issues_count": len(validation_results["issues"]),
                    "health_score": max(0, 1 - (len(validation_results["issues"]) / max(total_items, 1)))
                }
                
                # Generate recommendations
                if inconsistencies > 0:
                    validation_results["recommendations"].append(
                        "Review and resolve contradictions in the knowledge base"
                    )
                if total_items < 10:
                    validation_results["recommendations"].append(
                        "Consider adding more knowledge items for better reasoning"
                    )
                
                result = validation_results

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="validate_knowledge_base",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to validate_knowledge_base: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("generate_inferences")
        async def handle_generate_inferences(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle generate_inferences operation"""
            try:
                # Generate inferences using pattern analysis
                inference_data = data.get("data", [])
                pattern_type = data.get("pattern_type", "logical")
                analysis_depth = data.get("analysis_depth", "comprehensive")

                result = await self.agent_sdk.pattern_analysis({
                    "data": inference_data,
                    "pattern_type": pattern_type,
                    "analysis_depth": analysis_depth
                })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="generate_inferences",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to generate_inferences: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("make_decision")
        async def handle_make_decision(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle make_decision operation"""
            try:
                # Make decision using multi-criteria analysis
                decision_query = data.get("query", "")
                options = data.get("options", [])
                criteria = data.get("criteria", [])
                constraints = data.get("constraints", [])
                decision_type = data.get("decision_type", "multi_criteria")
                
                # Use the enhanced decision making capability
                result = await self.agent_sdk.make_decisions_enhanced({
                    "query": decision_query,
                    "options": options,
                    "criteria": criteria,
                    "constraints": constraints,
                    "decision_type": decision_type,
                    "require_explanation": data.get("require_explanation", True)
                })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="make_decision",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to make_decision: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("solve_problem")
        async def handle_solve_problem(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle solve_problem operation"""
            try:
                # Solve problem using collaborative reasoning if needed
                problem_query = data.get("query", data.get("problem", ""))
                participants = data.get("participant_agents", [])
                strategy = data.get("strategy", "consensus")
                domain = data.get("domain", "general")

                if participants:
                    # Use collaborative reasoning for complex problems
                    result = await self.agent_sdk.collaborative_reasoning({
                        "participant_agents": participants,
                        "query": problem_query,
                        "strategy": strategy,
                        "domain": domain
                    })
                else:
                    # Use logical reasoning for individual problem solving
                    result = await self.agent_sdk.logical_reasoning({
                        "query": problem_query,
                        "reasoning_type": data.get("reasoning_type", "abductive"),
                        "domain": domain,
                        "premises": data.get("premises", [])
                    })

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="solve_problem",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to solve_problem: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_dashboard_data")
        async def handle_get_dashboard_data(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_dashboard_data operation"""
            try:
                # Get dashboard data for reasoning agent
                time_range = data.get("time_range", "24h")
                metrics = data.get("metrics", ["all"])
                
                dashboard_data = {
                    "agent_info": {
                        "name": self.config.agent_name,
                        "version": self.config.agent_version,
                        "status": "active",
                        "uptime": self._calculate_uptime()
                    },
                    "reasoning_metrics": {
                        "total_chains": len(self.agent_sdk.reasoning_chains),
                        "active_chains": sum(1 for c in self.agent_sdk.reasoning_chains.values() if c.get("active", True)),
                        "completed_chains": sum(1 for c in self.agent_sdk.reasoning_chains.values() if not c.get("active", True)),
                        "average_confidence": self._calculate_average_confidence(),
                        "reasoning_types_distribution": self._get_reasoning_type_distribution()
                    },
                    "knowledge_base_stats": {
                        "total_domains": len(self.agent_sdk.knowledge_graph),
                        "total_items": sum(len(items) for items in self.agent_sdk.knowledge_graph.values()),
                        "items_by_type": self._get_knowledge_type_distribution()
                    },
                    "performance_metrics": {
                        "average_response_time": self.agent_sdk.performance_metrics.get("avg_response_time", 0),
                        "success_rate": self.agent_sdk.performance_metrics.get("success_rate", 1.0),
                        "cache_hit_rate": self.agent_sdk.performance_metrics.get("cache_hit_rate", 0)
                    },
                    "recent_activity": self._get_recent_activity(time_range)
                }
                
                # Filter metrics if specific ones requested
                if "all" not in metrics:
                    filtered_data = {"agent_info": dashboard_data["agent_info"]}
                    for metric in metrics:
                        if metric in dashboard_data:
                            filtered_data[metric] = dashboard_data[metric]
                    result = filtered_data
                else:
                    result = dashboard_data

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_dashboard_data",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_dashboard_data: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("get_reasoning_options")
        async def handle_get_reasoning_options(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle get_reasoning_options operation"""
            try:
                # Get available reasoning options and capabilities
                result = {
                    "reasoning_types": [
                        {
                            "type": "deductive",
                            "description": "Logical deduction from premises to conclusion",
                            "confidence": "high",
                            "use_cases": ["Mathematical proofs", "Rule-based inference"]
                        },
                        {
                            "type": "inductive",
                            "description": "Pattern recognition and generalization",
                            "confidence": "medium",
                            "use_cases": ["Trend analysis", "Hypothesis generation"]
                        },
                        {
                            "type": "abductive",
                            "description": "Best explanation reasoning",
                            "confidence": "medium",
                            "use_cases": ["Diagnostics", "Root cause analysis"]
                        },
                        {
                            "type": "causal",
                            "description": "Cause-effect relationship analysis",
                            "confidence": "medium-high",
                            "use_cases": ["Impact analysis", "Prediction"]
                        },
                        {
                            "type": "analogical",
                            "description": "Reasoning by comparison and similarity",
                            "confidence": "medium",
                            "use_cases": ["Problem solving", "Knowledge transfer"]
                        }
                    ],
                    "supported_domains": list(self.agent_sdk.domain_expertise.keys()),
                    "collaboration_strategies": [
                        "consensus", "weighted_voting", "expert_selection", "hierarchical"
                    ],
                    "confidence_thresholds": {
                        "high": 0.8,
                        "medium": 0.6,
                        "low": 0.4
                    },
                    "max_reasoning_depth": 10,
                    "parallel_processing": True,
                    "explanation_formats": ["natural_language", "structured", "visual"]
                }

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="get_reasoning_options",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to get_reasoning_options: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("health_check")
        async def handle_health_check(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Health check for agent"""
            try:
                health_status = {
                    "status": "healthy",
                    "agent": self.config.agent_name,
                    "version": self.config.agent_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "a2a_compliant": True,
                    "blockchain_connected": await self._check_blockchain_connection()
                }
                result = health_status

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="health_check",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to health_check: {e}")
                return self.create_secure_response(str(e), status="error")

        # Registry capability handlers
        @self.secure_handler("logical_reasoning")
        async def handle_logical_reasoning(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle logical reasoning operations"""
            try:
                result = await self.agent_sdk.perform_logical_reasoning(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="logical_reasoning",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to logical_reasoning: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("inference_generation")
        async def handle_inference_generation(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle inference generation"""
            try:
                result = await self.agent_sdk.generate_inferences_enhanced(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="inference_generation",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to inference_generation: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("decision_making")
        async def handle_decision_making(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle decision making operations"""
            try:
                result = await self.agent_sdk.make_decisions_enhanced(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="decision_making",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to decision_making: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("knowledge_synthesis")
        async def handle_knowledge_synthesis(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle knowledge synthesis operations"""
            try:
                result = await self.agent_sdk.synthesize_knowledge_enhanced(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="knowledge_synthesis",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to knowledge_synthesis: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("problem_solving")
        async def handle_problem_solving(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle problem solving operations"""
            try:
                result = await self.agent_sdk.solve_problems_enhanced(data)

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="problem_solving",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to problem_solving: {e}")
                return self.create_secure_response(str(e), status="error")

    async def process_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Main entry point for A2A messages
        Routes messages to appropriate handlers based on operation
        """
        try:
            # Extract operation from message
            operation = None
            data = {}

            if message.parts and len(message.parts) > 0:
                part = message.parts[0]
                if part.data:
                    operation = part.data.get("operation")
                    data = part.data.get("data", {})

            if not operation:
                return self.create_secure_response(
                    "No operation specified in message",
                    status="error"
                )

            # Get handler for operation
            handler = self.handlers.get(operation)
            if not handler:
                return self.create_secure_response(
                    f"Unknown operation: {operation}",
                    status="error"
                )

            # Create context ID
            context_id = f"{message.sender_id}:{operation}:{datetime.utcnow().timestamp()}"

            # Process through handler
            return await handler(message, context_id, data)

        except Exception as e:
            logger.error(f"Failed to process A2A message: {e}")
            return self.create_secure_response(str(e), status="error")

    async def _log_blockchain_transaction(self, operation: str, data_hash: str, result_hash: str, context_id: str):
        """Log transaction to blockchain for audit trail"""
        try:
            transaction_data = {
                "agent_id": self.config.agent_id,
                "operation": operation,
                "data_hash": data_hash,
                "result_hash": result_hash,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Send to blockchain through A2A client
            await self.a2a_client.log_transaction(transaction_data)

        except Exception as e:
            logger.error(f"Failed to log blockchain transaction: {e}")

    def _hash_data(self, data: Any) -> str:
        """Create hash of data for blockchain logging"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _check_blockchain_connection(self) -> bool:
        """Check if blockchain connection is active"""
        try:
            return await self.a2a_client.is_connected()
        except Exception:
            return False

    async def start(self):
        """Start the A2A handler"""
        logger.info(f"Starting A2A handler for {self.config.agent_name}")

        # Connect to blockchain
        await self.a2a_client.connect()

        # Register agent on blockchain
        await self.a2a_client.register_agent({
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "capabilities": list(self.config.allowed_operations),
            "version": self.config.agent_version
        })

        logger.info(f"A2A handler started and registered on blockchain")

    async def stop(self):
        """Stop the A2A handler"""
        logger.info(f"Stopping A2A handler for {self.config.agent_name}")

        # Unregister from blockchain
        await self.a2a_client.unregister_agent(self.config.agent_id)

        # Disconnect
        await self.a2a_client.disconnect()

        # Parent cleanup
        await self.shutdown()

        logger.info(f"A2A handler stopped")
    
    # Helper methods for TODO implementations
    def _check_contradiction(self, item1: Dict, item2: Dict) -> bool:
        """Check if two knowledge items contradict each other"""
        # Simple contradiction detection - can be enhanced with NLP
        content1 = item1.get("content", "").lower()
        content2 = item2.get("content", "").lower()
        
        # Check for negation patterns
        negation_words = ["not", "never", "no", "false", "incorrect"]
        for neg in negation_words:
            if (neg in content1 and neg not in content2) or (neg in content2 and neg not in content1):
                # Check if they refer to the same subject
                words1 = set(content1.split())
                words2 = set(content2.split())
                common_words = words1.intersection(words2)
                if len(common_words) > 2:  # Likely about the same topic
                    return True
        return False
    
    def _validate_rule_completeness(self, rule: Dict) -> bool:
        """Validate if a rule has all required components"""
        content = rule.get("content", "")
        # A complete rule should have condition and consequence
        return "if" in content.lower() and ("then" in content.lower() or "->" in content)
    
    def _calculate_uptime(self) -> str:
        """Calculate agent uptime"""
        # This would typically track actual start time
        return "Active"
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all reasoning chains"""
        confidences = []
        for chain in self.agent_sdk.reasoning_chains.values():
            if "confidence_scores" in chain and "overall" in chain["confidence_scores"]:
                confidences.append(chain["confidence_scores"]["overall"])
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _get_reasoning_type_distribution(self) -> Dict[str, int]:
        """Get distribution of reasoning types used"""
        distribution = {}
        for chain in self.agent_sdk.reasoning_chains.values():
            r_type = chain.get("reasoning_type", "unknown")
            distribution[r_type] = distribution.get(r_type, 0) + 1
        return distribution
    
    def _get_knowledge_type_distribution(self) -> Dict[str, int]:
        """Get distribution of knowledge item types"""
        distribution = {}
        for items in self.agent_sdk.knowledge_graph.values():
            for item in items:
                k_type = item.get("type", "unknown")
                distribution[k_type] = distribution.get(k_type, 0) + 1
        return distribution
    
    def _get_recent_activity(self, time_range: str) -> List[Dict]:
        """Get recent reasoning activity"""
        # Parse time range
        hours = 24
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
        elif time_range.endswith('d'):
            hours = int(time_range[:-1]) * 24
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_activity = []
        
        # Get recent chains
        for chain_id, chain in self.agent_sdk.reasoning_chains.items():
            created_at_str = chain.get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    if created_at > cutoff_time:
                        recent_activity.append({
                            "type": "reasoning_chain",
                            "id": chain_id,
                            "timestamp": created_at_str,
                            "details": {
                                "reasoning_type": chain.get("reasoning_type"),
                                "status": "active" if chain.get("active", True) else "completed"
                            }
                        })
                except:
                    pass
        
        # Sort by timestamp
        recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent_activity[:10]  # Return last 10 activities


# Factory function to create A2A handler
def create_agent9Router_a2a_handler(agent_sdk: ComprehensiveReasoningAgentSdk) -> Agent9RouterA2AHandler:
    """Create A2A-compliant handler for Agent 9 - Reasoning"""
    return Agent9RouterA2AHandler(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_agent9Router_a2a_handler(agent9Router_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""
