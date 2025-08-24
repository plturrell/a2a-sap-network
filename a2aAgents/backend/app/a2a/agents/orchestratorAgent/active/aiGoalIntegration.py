"""
AI Goal Integration Module
Integrates AI optimization with the existing orchestrator agent
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from .aiOptimizedGoalManagement import get_ai_optimizer, initialize_ai_optimization

logger = logging.getLogger(__name__)

class AIGoalIntegrationMixin:
    """Mixin to add AI capabilities to orchestrator agent"""
    
    def __init__(self):
        self.ai_optimizer = None
        self.ai_enabled = True
        
    async def initialize_ai_integration(self):
        """Initialize AI integration"""
        try:
            self.ai_optimizer = get_ai_optimizer(self)
            await initialize_ai_optimization(self)
            logger.info("AI goal optimization integrated successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI integration: {e}")
            self.ai_enabled = False
    
    async def _register_ai_handlers(self):
        """Register AI-enhanced message handlers"""
        
        @self.secure_handler("predict_goal_completion")
        async def handle_predict_goal_completion(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """AI-powered goal completion prediction"""
            try:
                agent_id = data.get("agent_id")
                if not agent_id:
                    return self.create_secure_response("agent_id is required", status="error")
                
                if not self.ai_enabled or not self.ai_optimizer:
                    return self.create_secure_response("AI optimization not available", status="error")
                
                prediction = await self.ai_optimizer.predict_goal_completion(agent_id)
                
                if prediction:
                    result = {
                        "agent_id": agent_id,
                        "prediction": {
                            "completion_date": prediction.predicted_completion_date.isoformat(),
                            "confidence_score": prediction.confidence_score,
                            "risk_factors": prediction.risk_factors,
                            "recommendations": prediction.recommended_actions,
                            "progress_curve": [
                                {"date": date.isoformat(), "progress": progress}
                                for date, progress in prediction.expected_progress_curve
                            ]
                        }
                    }
                    
                    # Log blockchain transaction
                    await self._log_blockchain_transaction(
                        operation="predict_goal_completion",
                        data_hash=self._hash_data(data),
                        result_hash=self._hash_data(result),
                        context_id=context_id
                    )
                    
                    return self.create_secure_response(result)
                else:
                    return self.create_secure_response("Unable to generate prediction", status="error")
                    
            except Exception as e:
                logger.error(f"Failed to predict goal completion: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("detect_intelligent_milestones")
        async def handle_detect_intelligent_milestones(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """AI-powered intelligent milestone detection"""
            try:
                agent_id = data.get("agent_id")
                if not agent_id:
                    return self.create_secure_response("agent_id is required", status="error")
                
                if not self.ai_enabled or not self.ai_optimizer:
                    return self.create_secure_response("AI optimization not available", status="error")
                
                milestones = await self.ai_optimizer.detect_intelligent_milestones(agent_id)
                
                result = {
                    "agent_id": agent_id,
                    "intelligent_milestones": [
                        {
                            "name": m.milestone_name,
                            "achievement_probability": m.achievement_probability,
                            "estimated_date": m.estimated_date.isoformat(),
                            "dependencies": m.dependencies,
                            "impact_score": m.impact_score
                        }
                        for m in milestones
                    ],
                    "total_milestones": len(milestones)
                }
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="detect_intelligent_milestones",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to detect intelligent milestones: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("optimize_goal_strategy")
        async def handle_optimize_goal_strategy(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """AI-powered goal strategy optimization"""
            try:
                agent_id = data.get("agent_id")
                if not agent_id:
                    return self.create_secure_response("agent_id is required", status="error")
                
                if not self.ai_enabled or not self.ai_optimizer:
                    return self.create_secure_response("AI optimization not available", status="error")
                
                optimization_result = await self.ai_optimizer.optimize_goal_strategy(agent_id)
                
                if optimization_result["status"] == "success":
                    # Log blockchain transaction
                    await self._log_blockchain_transaction(
                        operation="optimize_goal_strategy",
                        data_hash=self._hash_data(data),
                        result_hash=self._hash_data(optimization_result["strategy"]),
                        context_id=context_id
                    )
                    
                    return self.create_secure_response(optimization_result["strategy"])
                else:
                    return self.create_secure_response(optimization_result["message"], status="error")
                
            except Exception as e:
                logger.error(f"Failed to optimize goal strategy: {e}")
                return self.create_secure_response(str(e), status="error")

        @self.secure_handler("ai_goal_insights")
        async def handle_ai_goal_insights(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Get comprehensive AI insights for goal management"""
            try:
                agent_id = data.get("agent_id")
                
                if not self.ai_enabled or not self.ai_optimizer:
                    return self.create_secure_response("AI optimization not available", status="error")
                
                if agent_id:
                    # Agent-specific insights
                    prediction = await self.ai_optimizer.predict_goal_completion(agent_id)
                    milestones = await self.ai_optimizer.detect_intelligent_milestones(agent_id)
                    optimization = await self.ai_optimizer.optimize_goal_strategy(agent_id)
                    
                    result = {
                        "agent_id": agent_id,
                        "ai_insights": {
                            "completion_prediction": {
                                "available": prediction is not None,
                                "confidence": prediction.confidence_score if prediction else 0.0,
                                "estimated_completion": prediction.predicted_completion_date.isoformat() if prediction else None
                            },
                            "intelligent_milestones": {
                                "count": len(milestones),
                                "high_probability_count": len([m for m in milestones if m.achievement_probability > 0.7])
                            },
                            "optimization_score": optimization.get("strategy", {}).get("optimization_score", 0.0) if optimization.get("status") == "success" else 0.0,
                            "ai_recommendations": len(prediction.recommended_actions) if prediction else 0
                        }
                    }
                else:
                    # System-wide AI insights
                    total_agents = len(self.agent_goals)
                    ai_predictions = 0
                    avg_confidence = 0.0
                    
                    for aid in self.agent_goals.keys():
                        pred = await self.ai_optimizer.predict_goal_completion(aid)
                        if pred:
                            ai_predictions += 1
                            avg_confidence += pred.confidence_score
                    
                    if ai_predictions > 0:
                        avg_confidence /= ai_predictions
                    
                    result = {
                        "system_ai_insights": {
                            "total_agents_with_goals": total_agents,
                            "agents_with_ai_predictions": ai_predictions,
                            "average_prediction_confidence": avg_confidence,
                            "ai_optimization_enabled": self.ai_enabled,
                            "ml_models_trained": self.ai_optimizer.progress_predictor is not None
                        }
                    }
                
                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="ai_goal_insights",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )
                
                return self.create_secure_response(result)
                
            except Exception as e:
                logger.error(f"Failed to get AI goal insights: {e}")
                return self.create_secure_response(str(e), status="error")

    async def _enhance_progress_tracking_with_ai(self, agent_id: str, progress_data: Dict[str, Any]):
        """Enhance progress tracking with AI insights"""
        try:
            if not self.ai_enabled or not self.ai_optimizer:
                return
            
            # Get AI prediction for this agent
            prediction = await self.ai_optimizer.predict_goal_completion(agent_id)
            
            if prediction:
                # Add AI insights to progress data
                progress_data["ai_insights"] = {
                    "predicted_completion": prediction.predicted_completion_date.isoformat(),
                    "confidence_score": prediction.confidence_score,
                    "risk_assessment": {
                        "risk_level": "high" if len(prediction.risk_factors) > 2 else "medium" if prediction.risk_factors else "low",
                        "risk_factors": prediction.risk_factors
                    },
                    "ai_recommendations": prediction.recommended_actions[:3]  # Top 3 recommendations
                }
                
                # Detect and add intelligent milestones
                milestones = await self.ai_optimizer.detect_intelligent_milestones(agent_id)
                if milestones:
                    progress_data["ai_insights"]["next_milestones"] = [
                        {
                            "name": m.milestone_name,
                            "probability": m.achievement_probability,
                            "estimated_date": m.estimated_date.isoformat()
                        }
                        for m in milestones[:2]  # Next 2 milestones
                    ]
                
                logger.debug(f"Enhanced progress tracking with AI insights for {agent_id}")
                
        except Exception as e:
            logger.error(f"Failed to enhance progress tracking with AI for {agent_id}: {e}")

    async def _ai_enhanced_goal_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI-enhanced analytics"""
        try:
            if not self.ai_enabled or not self.ai_optimizer:
                return {}
            
            ai_analytics = {}
            
            if agent_id:
                # Agent-specific AI analytics
                prediction = await self.ai_optimizer.predict_goal_completion(agent_id)
                milestones = await self.ai_optimizer.detect_intelligent_milestones(agent_id)
                
                if prediction:
                    days_to_completion = (prediction.predicted_completion_date - datetime.utcnow()).days
                    
                    ai_analytics = {
                        "ai_prediction": {
                            "completion_in_days": days_to_completion,
                            "confidence_level": "high" if prediction.confidence_score > 0.8 else "medium" if prediction.confidence_score > 0.6 else "low",
                            "risk_assessment": len(prediction.risk_factors),
                            "optimization_potential": len(prediction.recommended_actions)
                        },
                        "intelligent_milestones": {
                            "upcoming_count": len([m for m in milestones if m.achievement_probability > 0.5]),
                            "high_impact_count": len([m for m in milestones if m.impact_score > 0.7])
                        }
                    }
            else:
                # System-wide AI analytics
                total_predictions = 0
                total_confidence = 0.0
                high_risk_agents = 0
                
                for aid in self.agent_goals.keys():
                    pred = await self.ai_optimizer.predict_goal_completion(aid)
                    if pred:
                        total_predictions += 1
                        total_confidence += pred.confidence_score
                        if len(pred.risk_factors) > 2:
                            high_risk_agents += 1
                
                ai_analytics = {
                    "system_ai_metrics": {
                        "prediction_coverage": f"{total_predictions}/{len(self.agent_goals)}",
                        "average_confidence": total_confidence / max(total_predictions, 1),
                        "high_risk_agents": high_risk_agents,
                        "ai_optimization_active": self.ai_enabled
                    }
                }
            
            return ai_analytics
            
        except Exception as e:
            logger.error(f"Failed to generate AI-enhanced analytics: {e}")
            return {}

# Integration function to add AI capabilities to existing orchestrator
def integrate_ai_capabilities(orchestrator_handler):
    """Integrate AI capabilities into existing orchestrator handler"""
    
    # Add AI mixin methods to orchestrator
    ai_mixin = AIGoalIntegrationMixin()
    
    # Copy AI methods to orchestrator
    orchestrator_handler.initialize_ai_integration = ai_mixin.initialize_ai_integration.__get__(orchestrator_handler)
    orchestrator_handler._register_ai_handlers = ai_mixin._register_ai_handlers.__get__(orchestrator_handler)
    orchestrator_handler._enhance_progress_tracking_with_ai = ai_mixin._enhance_progress_tracking_with_ai.__get__(orchestrator_handler)
    orchestrator_handler._ai_enhanced_goal_analytics = ai_mixin._ai_enhanced_goal_analytics.__get__(orchestrator_handler)
    
    # Initialize AI properties
    orchestrator_handler.ai_optimizer = None
    orchestrator_handler.ai_enabled = True
    
    logger.info("AI capabilities integrated into orchestrator handler")
