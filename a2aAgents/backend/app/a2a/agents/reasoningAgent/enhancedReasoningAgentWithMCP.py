import os
"""
Enhanced Reasoning Agent with MCP Tool Integration
Example of how to update agent SDKs to use the new MCP tools
"""

import logging
from typing import Dict, List, Any, Optional
from ...sdk import A2AAgentBase, a2a_handler, a2a_skill
from ...sdk.mcpSkillCoordination import MCPSkillCoordinator
from .mcpReasoningConfidenceCalculator import mcp_confidence_calculator
from .mcpSemanticSimilarityCalculator import mcp_similarity_calculator
from app.a2a.core.security_base import SecureA2AAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
logger = logging.getLogger(__name__)


class EnhancedReasoningAgentWithMCP(SecureA2AAgent):
    """
    Enhanced reasoning agent that uses MCP tools for calculations
    Demonstrates integration of MCP tools into existing agent architecture
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="enhanced_reasoning_agent",
            name="Enhanced Reasoning Agent with MCP",
            description="Reasoning agent using MCP tools for cross-agent calculations",
            version="2.0.0",
            base_url=base_url
        )
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        
        # Initialize MCP skill coordinator
        self.mcp_coordinator = MCPSkillCoordinator(self.agent_id)
        
        # Register MCP tool providers
        self.mcp_confidence_calculator = mcp_confidence_calculator
        self.mcp_similarity_calculator = mcp_similarity_calculator
        
        logger.info(f"Initialized {self.name} with MCP tool integration")
    
    async def initialize(self) -> None:
        """Initialize agent with MCP tools"""
        await super().initialize()
        
        # Register MCP tools with coordinator
        await self.mcp_coordinator.register_skill_provider(
            "confidence_calculation",
            self.mcp_confidence_calculator
        )
        
        await self.mcp_coordinator.register_skill_provider(
            "text_similarity",
            self.mcp_similarity_calculator
        )
        
        logger.info("MCP tools registered and ready")
    
    @a2a_handler("analyze_reasoning")
    async def handle_analyze_reasoning(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler that uses MCP confidence calculation tool
        """
        try:
            reasoning_context = message.get("reasoning_context", {})
            
            # Use MCP tool for confidence calculation
            confidence_result = await self.mcp_confidence_calculator.calculate_reasoning_confidence_mcp(
                reasoning_context=reasoning_context,
                include_explanation=True
            )
            
            # Extract key metrics
            confidence_score = confidence_result["confidence"]
            factor_breakdown = confidence_result["factor_breakdown"]
            recommendations = confidence_result["recommendations"]
            
            # Determine reasoning quality
            quality = self._determine_reasoning_quality(confidence_score)
            
            return {
                "status": "success",
                "confidence_score": confidence_score,
                "quality_assessment": quality,
                "factor_analysis": factor_breakdown,
                "recommendations": recommendations,
                "mcp_tool_used": "calculate_reasoning_confidence"
            }
            
        except Exception as e:
            logger.error(f"Reasoning analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @a2a_handler("compare_responses")
    async def handle_compare_responses(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handler that uses MCP text similarity tool
        """
        try:
            responses = message.get("responses", [])
            
            if len(responses) < 2:
                return {
                    "status": "error",
                    "error": "Need at least 2 responses to compare"
                }
            
            # Use MCP tool for group similarity
            group_result = await self.mcp_similarity_calculator.calculate_group_similarity_mcp(
                texts=responses,
                return_matrix=True
            )
            
            # Find most similar pairs
            similarity_pairs = []
            matrix = group_result.get("similarity_matrix", [])
            
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity_pairs.append({
                        "response1_idx": i,
                        "response2_idx": j,
                        "similarity": matrix[i][j]
                    })
            
            # Sort by similarity
            similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "status": "success",
                "average_similarity": group_result["average_similarity"],
                "consensus_level": self._determine_consensus_level(group_result["average_similarity"]),
                "most_similar_pair": similarity_pairs[0] if similarity_pairs else None,
                "similarity_statistics": {
                    "min": group_result["min_similarity"],
                    "max": group_result["max_similarity"],
                    "std_dev": group_result["std_deviation"]
                },
                "mcp_tool_used": "calculate_group_similarity"
            }
            
        except Exception as e:
            logger.error(f"Response comparison failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @a2a_skill("enhanced_reasoning_validation")
    async def enhanced_reasoning_validation_skill(self, 
                                            reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Skill that combines multiple MCP tools for comprehensive validation
        """
        validation_results = {}
        
        # 1. Calculate confidence using MCP tool
        confidence_result = await self.mcp_confidence_calculator.calculate_reasoning_confidence_mcp(
            reasoning_data,
            include_explanation=True
        )
        
        validation_results["confidence_analysis"] = {
            "score": confidence_result["confidence"],
            "breakdown": confidence_result["factor_breakdown"],
            "recommendations": confidence_result["recommendations"]
        }
        
        # 2. Check semantic alignment
        question = reasoning_data.get("question", "")
        answer = reasoning_data.get("answer", "")
        
        if question and answer:
            alignment_result = await self.mcp_confidence_calculator.calculate_semantic_alignment_mcp(
                question,
                answer,
                analyze_keywords=True
            )
            
            validation_results["semantic_alignment"] = {
                "score": alignment_result["alignment_score"],
                "question_type": alignment_result["question_type"],
                "keyword_overlap": alignment_result["keyword_overlap"]
            }
        
        # 3. Evidence quality assessment
        evidence = reasoning_data.get("evidence", [])
        if evidence:
            evidence_result = await self.mcp_confidence_calculator.calculate_evidence_quality_mcp(
                evidence,
                return_details=True
            )
            
            validation_results["evidence_quality"] = {
                "score": evidence_result["quality_score"],
                "composition": evidence_result["evidence_composition"],
                "has_academic": evidence_result["has_academic_sources"]
            }
        
        # Overall validation score
        scores = [
            validation_results["confidence_analysis"]["score"],
            validation_results.get("semantic_alignment", {}).get("score", 0.5),
            validation_results.get("evidence_quality", {}).get("score", 0.5)
        ]
        
        validation_results["overall_score"] = sum(scores) / len(scores)
        validation_results["is_valid"] = validation_results["overall_score"] > 0.6
        
        return validation_results
    
    @a2a_skill("cross_agent_similarity_search")
    async def cross_agent_similarity_search_skill(self,
                                            query: str,
                                            candidate_texts: List[str],
                                            top_k: int = 5) -> Dict[str, Any]:
        """
        Skill that uses MCP similarity tool for cross-agent search
        """
        # Prepare candidates with IDs
        candidates = [
            {"id": f"text_{i}", "text": text}
            for i, text in enumerate(candidate_texts)
        ]
        
        # Use MCP tool to find similar texts
        search_result = await self.mcp_similarity_calculator.find_similar_texts_mcp(
            query=query,
            candidates=candidate_texts,
            top_k=top_k,
            method="hybrid"
        )
        
        # Enhance results with additional analysis
        enhanced_results = []
        for result in search_result["results"]:
            # Extract semantic features for each result
            features = await self.mcp_similarity_calculator.extract_semantic_features_mcp(
                result["text"],
                include_categories=True
            )
            
            enhanced_results.append({
                "text": result["text"],
                "similarity": result["similarity"],
                "semantic_categories": list(features.get("semantic_categories", {}).keys()),
                "unique_words": len(features.get("unique_words", []))
            })
        
        return {
            "query": query,
            "results": enhanced_results,
            "total_candidates": len(candidate_texts),
            "mcp_tools_used": ["find_similar_texts", "extract_semantic_features"]
        }
    
    def _determine_reasoning_quality(self, confidence_score: float) -> str:
        """Determine reasoning quality based on confidence score"""
        if confidence_score >= 0.8:
            return "excellent"
        elif confidence_score >= 0.6:
            return "good"
        elif confidence_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _determine_consensus_level(self, average_similarity: float) -> str:
        """Determine consensus level based on similarity"""
        if average_similarity >= 0.8:
            return "strong_consensus"
        elif average_similarity >= 0.6:
            return "moderate_consensus"
        elif average_similarity >= 0.4:
            return "weak_consensus"
        else:
            return "no_consensus"
    
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        # Unregister MCP tools
        await self.mcp_coordinator.cleanup()
        await super().cleanup()


# Example usage patterns for other agents to follow:

async def example_usage():
    """
    Example of how to use MCP tools in agent code
    """
    # Initialize agent
    agent = EnhancedReasoningAgentWithMCP("os.getenv("A2A_BASE_URL")")
    await agent.initialize()
    
    # Example 1: Direct MCP tool usage
    confidence_result = await agent.mcp_confidence_calculator.calculate_reasoning_confidence_mcp(
        reasoning_context={
            "evidence": [{"source_type": "academic"}],
            "question": "What is AI?",
            "answer": "AI is artificial intelligence"
        }
    )
    
    # Example 2: Using MCP tools in skills
    validation_result = await agent.enhanced_reasoning_validation_skill({
        "question": "How does machine learning work?",
        "answer": "Machine learning uses algorithms to learn from data",
        "evidence": [
            {"source_type": "academic", "content": "Research paper"},
            {"source_type": "empirical", "content": "Experiment results"}
        ]
    })
    
    # Example 3: Cross-agent similarity search
    search_result = await agent.cross_agent_similarity_search_skill(
        query="natural language processing",
        candidate_texts=[
            "NLP is a branch of AI focused on language",
            "Computer vision processes images",
            "Text analysis and language understanding",
            "Database management systems"
        ],
        top_k=2
    )
    
    await agent.cleanup()


# Integration guide for existing agents:

"""
INTEGRATION GUIDE FOR EXISTING AGENTS

1. Import MCP calculators:
   from .mcpReasoningConfidenceCalculator import mcp_confidence_calculator
   from .mcpSemanticSimilarityCalculator import mcp_similarity_calculator
   from ..agent3VectorProcessing.active.mcpHybridRankingSkills import mcp_hybrid_ranking
   from ..agent3VectorProcessing.active.mcpVectorSimilarityCalculator import mcp_vector_similarity

2. Initialize in __init__:
   self.mcp_confidence_calculator = mcp_confidence_calculator
   self.mcp_similarity_calculator = mcp_similarity_calculator

3. Replace hardcoded calculations:
   # Old way:
   confidence = self._calculate_confidence_hardcoded(data)
   
   # New way:
   result = await self.mcp_confidence_calculator.calculate_reasoning_confidence_mcp(data)
   confidence = result["confidence"]

4. Use in handlers and skills:
   @a2a_handler("my_handler")
   async def my_handler(self, message):
        # Security validation
        if not self.validate_input(request_data)[0]:
            return create_error_response("Invalid input data")
        
        # Rate limiting check
        client_id = request_data.get('client_id', 'unknown')
        if not self.check_rate_limit(client_id):
            return create_error_response("Rate limit exceeded")
        
       # Use MCP tools for calculations
       similarity = await self.mcp_similarity_calculator.calculate_text_similarity_mcp(
           text1, text2, method="hybrid"
       )

5. Leverage MCP resources:
   # Get available metrics
   metrics = await self.mcp_vector_similarity.get_available_metrics()
   
   # Get calculator configuration
   config = await self.mcp_similarity_calculator.get_calculator_config()

6. Use MCP prompts for analysis:
   # Get analysis prompt
   analysis = await self.mcp_confidence_calculator.confidence_analysis_prompt(
       context, focus_area="evidence"
   )
"""