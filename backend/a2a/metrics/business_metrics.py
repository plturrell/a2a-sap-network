"""
A2A Business Metrics Collection and Analysis
Custom metrics for A2A-specific business logic and KPIs
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from opentelemetry import metrics
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, Info, Enum as PromEnum
from prometheus_client import start_http_server, generate_latest

logger = logging.getLogger(__name__)


class A2AMetricType(str, Enum):
    """Types of A2A business metrics"""
    AGENT_INTERACTION = "agent_interaction"
    MESSAGE_FLOW = "message_flow" 
    WORKFLOW_EXECUTION = "workflow_execution"
    TRUST_SCORE = "trust_score"
    BLOCKCHAIN_ACTIVITY = "blockchain_activity"
    DATA_PIPELINE = "data_pipeline"
    SERVICE_UTILIZATION = "service_utilization"
    USER_ENGAGEMENT = "user_engagement"
    BUSINESS_VALUE = "business_value"


@dataclass
class A2AMetric:
    """A2A business metric data structure"""
    name: str
    metric_type: A2AMetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class A2ABusinessMetrics:
    """
    A2A Business Metrics Collector
    Tracks business-specific KPIs and operational metrics
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.meter = metrics.get_meter(__name__)
        
        # Initialize all business metrics
        self._init_agent_metrics()
        self._init_message_metrics()
        self._init_workflow_metrics()
        self._init_trust_metrics()
        self._init_blockchain_metrics()
        self._init_data_pipeline_metrics()
        self._init_service_metrics()
        self._init_user_engagement_metrics()
        self._init_business_value_metrics()
        
        # Metric collection state
        self.collected_metrics: List[A2AMetric] = []
        self.collection_interval = 30  # seconds
        self._collection_task = None
        
        logger.info("A2A Business Metrics initialized")

    def _init_agent_metrics(self):
        """Initialize agent-specific business metrics"""
        # Agent interaction metrics
        self.agent_interactions_total = Counter(
            'a2a_agent_interactions_total',
            'Total number of agent-to-agent interactions',
            ['source_agent', 'target_agent', 'interaction_type', 'protocol_version'],
            registry=self.registry
        )
        
        self.agent_collaboration_success_rate = Gauge(
            'a2a_agent_collaboration_success_rate',
            'Success rate of agent collaborations',
            ['agent_pair', 'workflow_type'],
            registry=self.registry
        )
        
        self.agent_discovery_time = Histogram(
            'a2a_agent_discovery_duration_seconds',
            'Time taken to discover and connect to agents',
            ['discovery_method', 'agent_type'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.agent_specialization_score = Gauge(
            'a2a_agent_specialization_score',
            'Agent specialization effectiveness score (0-100)',
            ['agent_id', 'specialty'],
            registry=self.registry
        )
        
        # Agent network topology
        self.agent_network_density = Gauge(
            'a2a_agent_network_density',
            'Density of agent interaction network (0-1)',
            registry=self.registry
        )
        
        self.agent_centrality_score = Gauge(
            'a2a_agent_centrality_score',
            'Agent centrality in the network',
            ['agent_id', 'centrality_type'],
            registry=self.registry
        )

    def _init_message_metrics(self):
        """Initialize message flow business metrics"""
        # Message complexity and intelligence
        self.message_complexity_score = Histogram(
            'a2a_message_complexity_score',
            'Complexity score of A2A messages (0-100)',
            ['message_type', 'agent_type'],
            registry=self.registry,
            buckets=[10, 25, 50, 75, 90, 95, 99, 100]
        )
        
        self.message_intelligence_level = Gauge(
            'a2a_message_intelligence_level',
            'Intelligence level of message content',
            ['content_type', 'processing_stage'],
            registry=self.registry
        )
        
        # Semantic understanding
        self.semantic_understanding_accuracy = Gauge(
            'a2a_semantic_understanding_accuracy',
            'Accuracy of semantic message understanding',
            ['language_model', 'context_type'],
            registry=self.registry
        )
        
        self.intent_classification_confidence = Histogram(
            'a2a_intent_classification_confidence',
            'Confidence score for message intent classification',
            ['intent_category'],
            registry=self.registry,
            buckets=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        # Cross-protocol adaptation
        self.protocol_adaptation_success = Counter(
            'a2a_protocol_adaptation_total',
            'Protocol adaptation attempts and successes',
            ['source_protocol', 'target_protocol', 'status'],
            registry=self.registry
        )
        
        self.message_transformation_efficiency = Histogram(
            'a2a_message_transformation_efficiency',
            'Efficiency of message format transformations',
            ['transformation_type'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

    def _init_workflow_metrics(self):
        """Initialize workflow execution business metrics"""
        # Workflow orchestration
        self.workflow_orchestration_complexity = Gauge(
            'a2a_workflow_orchestration_complexity',
            'Complexity of workflow orchestration',
            ['workflow_id', 'orchestration_pattern'],
            registry=self.registry
        )
        
        self.workflow_adaptation_score = Gauge(
            'a2a_workflow_adaptation_score',
            'How well workflow adapts to changing conditions',
            ['workflow_type', 'adaptation_trigger'],
            registry=self.registry
        )
        
        # Dynamic workflow optimization
        self.workflow_optimization_impact = Gauge(
            'a2a_workflow_optimization_impact',
            'Impact of dynamic workflow optimizations',
            ['optimization_type', 'metric_improved'],
            registry=self.registry
        )
        
        self.workflow_resource_utilization = Gauge(
            'a2a_workflow_resource_utilization',
            'Resource utilization efficiency in workflows',
            ['workflow_id', 'resource_type'],
            registry=self.registry
        )
        
        # Workflow intelligence
        self.workflow_decision_accuracy = Gauge(
            'a2a_workflow_decision_accuracy',
            'Accuracy of automated workflow decisions',
            ['decision_type', 'context'],
            registry=self.registry
        )

    def _init_trust_metrics(self):
        """Initialize trust and reputation metrics"""
        # Trust network dynamics
        self.trust_network_stability = Gauge(
            'a2a_trust_network_stability',
            'Stability of the trust network over time',
            registry=self.registry
        )
        
        self.trust_propagation_speed = Histogram(
            'a2a_trust_propagation_speed_seconds',
            'Speed of trust score propagation through network',
            registry=self.registry,
            buckets=[1, 5, 15, 30, 60, 300, 900]
        )
        
        self.reputation_volatility = Gauge(
            'a2a_reputation_volatility',
            'Volatility of agent reputation scores',
            ['agent_id', 'time_window'],
            registry=self.registry
        )
        
        # Trust-based decision making
        self.trust_based_routing_efficiency = Gauge(
            'a2a_trust_based_routing_efficiency',
            'Efficiency of trust-based message routing',
            ['routing_algorithm'],
            registry=self.registry
        )
        
        self.trust_verification_accuracy = Gauge(
            'a2a_trust_verification_accuracy',
            'Accuracy of trust verification mechanisms',
            ['verification_method'],
            registry=self.registry
        )

    def _init_blockchain_metrics(self):
        """Initialize blockchain business metrics"""
        # Blockchain network health
        self.blockchain_network_participation = Gauge(
            'a2a_blockchain_network_participation',
            'Participation rate in blockchain network',
            ['network_type', 'participant_role'],
            registry=self.registry
        )
        
        self.smart_contract_execution_complexity = Histogram(
            'a2a_smart_contract_execution_complexity',
            'Complexity of smart contract executions',
            ['contract_type', 'function_name'],
            registry=self.registry,
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000]
        )
        
        # Cross-chain interoperability
        self.cross_chain_bridge_efficiency = Gauge(
            'a2a_cross_chain_bridge_efficiency',
            'Efficiency of cross-chain bridges',
            ['source_chain', 'target_chain', 'bridge_protocol'],
            registry=self.registry
        )
        
        self.blockchain_consensus_participation = Gauge(
            'a2a_blockchain_consensus_participation',
            'Participation in blockchain consensus',
            ['consensus_type', 'role'],
            registry=self.registry
        )
        
        # Decentralized governance
        self.governance_proposal_engagement = Gauge(
            'a2a_governance_proposal_engagement',
            'Engagement in governance proposals',
            ['proposal_type', 'voting_power_tier'],
            registry=self.registry
        )

    def _init_data_pipeline_metrics(self):
        """Initialize data pipeline business metrics"""
        # Data quality and lineage
        self.data_quality_score = Gauge(
            'a2a_data_quality_score',
            'Data quality score across pipelines',
            ['pipeline_id', 'quality_dimension'],
            registry=self.registry
        )
        
        self.data_lineage_completeness = Gauge(
            'a2a_data_lineage_completeness',
            'Completeness of data lineage tracking',
            ['data_source', 'lineage_depth'],
            registry=self.registry
        )
        
        # Real-time processing
        self.stream_processing_throughput = Gauge(
            'a2a_stream_processing_throughput_events_per_second',
            'Throughput of real-time stream processing',
            ['stream_type', 'processing_stage'],
            registry=self.registry
        )
        
        self.data_freshness_score = Gauge(
            'a2a_data_freshness_score',
            'Freshness score of processed data',
            ['data_type', 'consumer_agent'],
            registry=self.registry
        )
        
        # AI/ML pipeline metrics
        self.model_prediction_confidence = Histogram(
            'a2a_model_prediction_confidence',
            'Confidence scores of ML model predictions',
            ['model_id', 'prediction_type'],
            registry=self.registry,
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        self.feature_importance_stability = Gauge(
            'a2a_feature_importance_stability',
            'Stability of feature importance over time',
            ['model_id', 'feature_category'],
            registry=self.registry
        )

    def _init_service_metrics(self):
        """Initialize service utilization metrics"""
        # Service mesh intelligence
        self.service_mesh_optimization_score = Gauge(
            'a2a_service_mesh_optimization_score',
            'Optimization score of service mesh routing',
            ['optimization_algorithm'],
            registry=self.registry
        )
        
        self.microservice_cohesion_score = Gauge(
            'a2a_microservice_cohesion_score',
            'Cohesion score of microservice boundaries',
            ['service_cluster'],
            registry=self.registry
        )
        
        # Adaptive scaling
        self.adaptive_scaling_accuracy = Gauge(
            'a2a_adaptive_scaling_accuracy',
            'Accuracy of adaptive scaling decisions',
            ['service_type', 'scaling_trigger'],
            registry=self.registry
        )
        
        self.resource_prediction_accuracy = Gauge(
            'a2a_resource_prediction_accuracy',
            'Accuracy of resource demand predictions',
            ['resource_type', 'prediction_horizon'],
            registry=self.registry
        )
        
        # Service discovery intelligence
        self.service_discovery_intelligence_score = Gauge(
            'a2a_service_discovery_intelligence_score',
            'Intelligence of service discovery mechanisms',
            ['discovery_protocol', 'network_topology'],
            registry=self.registry
        )

    def _init_user_engagement_metrics(self):
        """Initialize user engagement metrics"""
        # User interaction patterns
        self.user_workflow_complexity_preference = Histogram(
            'a2a_user_workflow_complexity_preference',
            'User preference for workflow complexity',
            ['user_type', 'experience_level'],
            registry=self.registry,
            buckets=[1, 3, 5, 7, 9, 12, 15, 20]
        )
        
        self.user_agent_interaction_satisfaction = Gauge(
            'a2a_user_agent_interaction_satisfaction',
            'User satisfaction with agent interactions',
            ['interaction_type', 'user_segment'],
            registry=self.registry
        )
        
        # Personalization effectiveness
        self.personalization_effectiveness_score = Gauge(
            'a2a_personalization_effectiveness_score',
            'Effectiveness of AI-driven personalization',
            ['personalization_type', 'user_context'],
            registry=self.registry
        )
        
        # Learning and adaptation
        self.user_behavior_prediction_accuracy = Gauge(
            'a2a_user_behavior_prediction_accuracy',
            'Accuracy of user behavior predictions',
            ['behavior_type', 'prediction_model'],
            registry=self.registry
        )

    def _init_business_value_metrics(self):
        """Initialize business value metrics"""
        # ROI and efficiency
        self.automation_roi_score = Gauge(
            'a2a_automation_roi_score',
            'ROI score of A2A automation',
            ['automation_type', 'business_process'],
            registry=self.registry
        )
        
        self.process_optimization_impact = Gauge(
            'a2a_process_optimization_impact',
            'Impact of A2A process optimization',
            ['process_type', 'optimization_dimension'],
            registry=self.registry
        )
        
        # Innovation metrics
        self.innovation_velocity = Gauge(
            'a2a_innovation_velocity',
            'Velocity of innovation enabled by A2A platform',
            ['innovation_category'],
            registry=self.registry
        )
        
        self.collaboration_effectiveness = Gauge(
            'a2a_collaboration_effectiveness',
            'Effectiveness of inter-agent collaboration',
            ['collaboration_pattern', 'outcome_type'],
            registry=self.registry
        )
        
        # Knowledge network value
        self.knowledge_network_density = Gauge(
            'a2a_knowledge_network_density',
            'Density of knowledge sharing network',
            ['knowledge_domain'],
            registry=self.registry
        )
        
        self.collective_intelligence_score = Gauge(
            'a2a_collective_intelligence_score',
            'Collective intelligence score of agent network',
            ['intelligence_dimension'],
            registry=self.registry
        )

    # Metric collection methods
    
    def record_agent_interaction(self, source_agent: str, target_agent: str, 
                               interaction_type: str, protocol_version: str = "0.2.9"):
        """Record an agent interaction"""
        self.agent_interactions_total.labels(
            source_agent=source_agent,
            target_agent=target_agent,
            interaction_type=interaction_type,
            protocol_version=protocol_version
        ).inc()
        
        # Record the metric for analysis
        metric = A2AMetric(
            name="agent_interaction",
            metric_type=A2AMetricType.AGENT_INTERACTION,
            value=1,
            labels={
                "source_agent": source_agent,
                "target_agent": target_agent,
                "interaction_type": interaction_type,
                "protocol_version": protocol_version
            }
        )
        self.collected_metrics.append(metric)

    def record_message_complexity(self, message_type: str, agent_type: str, complexity_score: float):
        """Record message complexity score"""
        self.message_complexity_score.labels(
            message_type=message_type,
            agent_type=agent_type
        ).observe(complexity_score)
        
        metric = A2AMetric(
            name="message_complexity",
            metric_type=A2AMetricType.MESSAGE_FLOW,
            value=complexity_score,
            labels={
                "message_type": message_type,
                "agent_type": agent_type
            }
        )
        self.collected_metrics.append(metric)

    def record_workflow_execution(self, workflow_id: str, workflow_type: str, 
                                 execution_time: float, success: bool):
        """Record workflow execution metrics"""
        # Update workflow adaptation score based on success
        adaptation_score = 100 if success else max(0, 100 - (execution_time * 10))
        self.workflow_adaptation_score.labels(
            workflow_type=workflow_type,
            adaptation_trigger="execution_result"
        ).set(adaptation_score)
        
        metric = A2AMetric(
            name="workflow_execution",
            metric_type=A2AMetricType.WORKFLOW_EXECUTION,
            value=execution_time,
            labels={
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "success": str(success)
            },
            metadata={
                "adaptation_score": adaptation_score
            }
        )
        self.collected_metrics.append(metric)

    def record_trust_update(self, agent_id: str, old_score: float, new_score: float):
        """Record trust score update"""
        volatility = abs(new_score - old_score)
        self.reputation_volatility.labels(
            agent_id=agent_id,
            time_window="1h"
        ).set(volatility)
        
        metric = A2AMetric(
            name="trust_update",
            metric_type=A2AMetricType.TRUST_SCORE,
            value=new_score,
            labels={
                "agent_id": agent_id
            },
            metadata={
                "old_score": old_score,
                "volatility": volatility
            }
        )
        self.collected_metrics.append(metric)

    def record_blockchain_transaction(self, contract_type: str, function_name: str, 
                                    gas_used: int, success: bool):
        """Record blockchain transaction metrics"""
        self.smart_contract_execution_complexity.labels(
            contract_type=contract_type,
            function_name=function_name
        ).observe(gas_used)
        
        metric = A2AMetric(
            name="blockchain_transaction",
            metric_type=A2AMetricType.BLOCKCHAIN_ACTIVITY,
            value=gas_used,
            labels={
                "contract_type": contract_type,
                "function_name": function_name,
                "success": str(success)
            }
        )
        self.collected_metrics.append(metric)

    def record_data_quality(self, pipeline_id: str, quality_dimension: str, score: float):
        """Record data quality metrics"""
        self.data_quality_score.labels(
            pipeline_id=pipeline_id,
            quality_dimension=quality_dimension
        ).set(score)
        
        metric = A2AMetric(
            name="data_quality",
            metric_type=A2AMetricType.DATA_PIPELINE,
            value=score,
            labels={
                "pipeline_id": pipeline_id,
                "quality_dimension": quality_dimension
            }
        )
        self.collected_metrics.append(metric)

    def record_user_satisfaction(self, interaction_type: str, user_segment: str, 
                               satisfaction_score: float):
        """Record user satisfaction metrics"""
        self.user_agent_interaction_satisfaction.labels(
            interaction_type=interaction_type,
            user_segment=user_segment
        ).set(satisfaction_score)
        
        metric = A2AMetric(
            name="user_satisfaction",
            metric_type=A2AMetricType.USER_ENGAGEMENT,
            value=satisfaction_score,
            labels={
                "interaction_type": interaction_type,
                "user_segment": user_segment
            }
        )
        self.collected_metrics.append(metric)

    def record_business_value(self, automation_type: str, business_process: str, roi_score: float):
        """Record business value metrics"""
        self.automation_roi_score.labels(
            automation_type=automation_type,
            business_process=business_process
        ).set(roi_score)
        
        metric = A2AMetric(
            name="business_value",
            metric_type=A2AMetricType.BUSINESS_VALUE,
            value=roi_score,
            labels={
                "automation_type": automation_type,
                "business_process": business_process
            }
        )
        self.collected_metrics.append(metric)

    # Advanced analytics methods
    
    def calculate_network_intelligence_score(self) -> float:
        """Calculate overall network intelligence score"""
        # This is a composite metric based on various intelligence indicators
        
        scores = []
        
        # Agent specialization effectiveness
        recent_specialization = [
            m for m in self.collected_metrics 
            if m.name == "agent_interaction" and 
            m.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if recent_specialization:
            specialization_score = len(set(
                (m.labels.get("source_agent"), m.labels.get("interaction_type"))
                for m in recent_specialization
            )) / len(recent_specialization) * 100
            scores.append(specialization_score)
        
        # Message complexity handling
        recent_complexity = [
            m for m in self.collected_metrics
            if m.name == "message_complexity" and
            m.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if recent_complexity:
            avg_complexity = sum(m.value for m in recent_complexity) / len(recent_complexity)
            scores.append(avg_complexity)
        
        # Workflow adaptation success
        recent_workflows = [
            m for m in self.collected_metrics
            if m.name == "workflow_execution" and
            m.timestamp > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if recent_workflows:
            success_rate = sum(
                1 for m in recent_workflows 
                if m.labels.get("success") == "True"
            ) / len(recent_workflows) * 100
            scores.append(success_rate)
        
        # Calculate weighted average
        if scores:
            network_intelligence = sum(scores) / len(scores)
        else:
            network_intelligence = 50  # Default neutral score
        
        # Record the composite metric
        self.collective_intelligence_score.labels(
            intelligence_dimension="network_overall"
        ).set(network_intelligence)
        
        return network_intelligence

    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze agent interaction patterns"""
        interactions = [
            m for m in self.collected_metrics
            if m.name == "agent_interaction" and
            m.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        if not interactions:
            return {"pattern": "insufficient_data"}
        
        # Calculate interaction frequency by agent pair
        pair_counts = {}
        for interaction in interactions:
            source = interaction.labels.get("source_agent")
            target = interaction.labels.get("target_agent")
            pair = f"{source}->{target}"
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # Find most active pairs
        most_active_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate network density
        unique_agents = set()
        for interaction in interactions:
            unique_agents.add(interaction.labels.get("source_agent"))
            unique_agents.add(interaction.labels.get("target_agent"))
        
        max_connections = len(unique_agents) * (len(unique_agents) - 1)
        actual_connections = len(pair_counts)
        density = actual_connections / max_connections if max_connections > 0 else 0
        
        self.agent_network_density.set(density)
        
        return {
            "pattern": "network_analysis",
            "total_interactions": len(interactions),
            "unique_agents": len(unique_agents),
            "network_density": density,
            "most_active_pairs": most_active_pairs,
            "interaction_types": list(set(
                m.labels.get("interaction_type") for m in interactions
            ))
        }

    def generate_business_insights(self) -> List[Dict[str, Any]]:
        """Generate business insights from collected metrics"""
        insights = []
        
        # Network intelligence analysis
        intelligence_score = self.calculate_network_intelligence_score()
        if intelligence_score > 80:
            insights.append({
                "type": "positive",
                "metric": "network_intelligence",
                "message": f"High network intelligence score: {intelligence_score:.1f}%",
                "recommendation": "Continue current optimization strategies"
            })
        elif intelligence_score < 60:
            insights.append({
                "type": "warning",
                "metric": "network_intelligence", 
                "message": f"Low network intelligence score: {intelligence_score:.1f}%",
                "recommendation": "Review agent specialization and workflow complexity"
            })
        
        # Trust network stability
        trust_updates = [
            m for m in self.collected_metrics
            if m.name == "trust_update" and
            m.timestamp > datetime.utcnow() - timedelta(hours=6)
        ]
        
        if trust_updates:
            avg_volatility = sum(
                m.metadata.get("volatility", 0) for m in trust_updates
            ) / len(trust_updates)
            
            if avg_volatility > 20:
                insights.append({
                    "type": "warning",
                    "metric": "trust_volatility",
                    "message": f"High trust volatility detected: {avg_volatility:.1f}",
                    "recommendation": "Investigate trust score fluctuations"
                })
        
        # Workflow efficiency trends
        workflow_executions = [
            m for m in self.collected_metrics
            if m.name == "workflow_execution" and
            m.timestamp > datetime.utcnow() - timedelta(hours=12)
        ]
        
        if workflow_executions:
            success_rate = sum(
                1 for m in workflow_executions 
                if m.labels.get("success") == "True"
            ) / len(workflow_executions)
            
            if success_rate < 0.85:
                insights.append({
                    "type": "alert",
                    "metric": "workflow_success_rate",
                    "message": f"Workflow success rate below threshold: {success_rate:.1%}",
                    "recommendation": "Review workflow configurations and agent capabilities"
                })
        
        return insights

    # Utility methods
    
    def start_collection(self):
        """Start automated metric collection"""
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Started automatic metric collection")

    def stop_collection(self):
        """Stop automated metric collection"""
        if self._collection_task:
            self._collection_task.cancel()
            self._collection_task = None
            logger.info("Stopped automatic metric collection")

    async def _collection_loop(self):
        """Main metric collection loop"""
        while True:
            try:
                # Collect derived metrics
                self.calculate_network_intelligence_score()
                self.analyze_interaction_patterns()
                
                # Clean up old metrics (keep 7 days)
                cutoff = datetime.utcnow() - timedelta(days=7)
                self.collected_metrics = [
                    m for m in self.collected_metrics
                    if m.timestamp > cutoff
                ]
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_prometheus_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        summary = {
            "total_metrics": len(self.collected_metrics),
            "metrics_by_type": {},
            "recent_activity": {},
            "collection_stats": {
                "collection_interval": self.collection_interval,
                "running": self._collection_task is not None,
                "oldest_metric": None,
                "newest_metric": None
            }
        }
        
        # Group by metric type
        for metric in self.collected_metrics:
            metric_type = metric.metric_type.value
            if metric_type not in summary["metrics_by_type"]:
                summary["metrics_by_type"][metric_type] = 0
            summary["metrics_by_type"][metric_type] += 1
        
        # Recent activity (last hour)
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in self.collected_metrics if m.timestamp > recent_cutoff]
        
        for metric in recent_metrics:
            name = metric.name
            if name not in summary["recent_activity"]:
                summary["recent_activity"][name] = 0
            summary["recent_activity"][name] += 1
        
        # Time bounds
        if self.collected_metrics:
            timestamps = [m.timestamp for m in self.collected_metrics]
            summary["collection_stats"]["oldest_metric"] = min(timestamps).isoformat()
            summary["collection_stats"]["newest_metric"] = max(timestamps).isoformat()
        
        return summary


# Global instance
_business_metrics = None

def get_business_metrics() -> A2ABusinessMetrics:
    """Get global business metrics instance"""
    global _business_metrics
    if _business_metrics is None:
        _business_metrics = A2ABusinessMetrics()
    return _business_metrics


# Convenience functions for common operations
def record_agent_interaction(source: str, target: str, interaction_type: str):
    """Record agent interaction"""
    metrics = get_business_metrics()
    metrics.record_agent_interaction(source, target, interaction_type)


def record_workflow_success(workflow_id: str, workflow_type: str, execution_time: float):
    """Record successful workflow execution"""
    metrics = get_business_metrics()
    metrics.record_workflow_execution(workflow_id, workflow_type, execution_time, True)


def record_trust_change(agent_id: str, old_score: float, new_score: float):
    """Record trust score change"""
    metrics = get_business_metrics()
    metrics.record_trust_update(agent_id, old_score, new_score)


def get_network_intelligence() -> float:
    """Get current network intelligence score"""
    metrics = get_business_metrics()
    return metrics.calculate_network_intelligence_score()


def start_metrics_collection():
    """Start automatic metric collection"""
    metrics = get_business_metrics()
    metrics.start_collection()


# Export main classes and functions
__all__ = [
    'A2ABusinessMetrics',
    'A2AMetric',
    'A2AMetricType',
    'get_business_metrics',
    'record_agent_interaction',
    'record_workflow_success',
    'record_trust_change',
    'get_network_intelligence',
    'start_metrics_collection'
]