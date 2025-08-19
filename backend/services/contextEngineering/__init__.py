"""
Context Engineering Agent Package
Production-ready context engineering system for A2A Network
"""

from .contextEngineeringAgentProduction import (
    ContextEngineeringAgent,
    create_production_agent,
    EnhancedContextStructure,
    ContextOptimizationResult,
    ContextQualityLevel
)

from .contextCoordination import (
    DistributedContextManager,
    ContextVersionControl,
    ContextConflictResolver,
    ContextPropagationManager,
    ConflictType,
    PropagationStrategy,
    ContextVersion,
    ContextConflict,
    SynchronizationState
)

from .orchestration import (
    ProcessOrchestrator,
    ProcessState,
    GatewayType,
    ProcessEvent,
    ProcessTask,
    ProcessGateway,
    TimerEventHandler
)

from .qualityFeedback import (
    QualityMonitor,
    FeedbackLoop,
    ABTestManager,
    CostOptimizer,
    ErrorAnalyzer,
    AnalyticsDashboard,
    QualityMetric,
    FeedbackItem,
    ABTestConfig,
    FeedbackType,
    ImprovementStrategy
)

__version__ = "2.0.0"

__all__ = [
    # Main agent
    "ContextEngineeringAgent",
    "create_production_agent",
    
    # Data structures
    "EnhancedContextStructure",
    "ContextOptimizationResult",
    "ContextQualityLevel",
    
    # Coordination
    "DistributedContextManager",
    "ContextVersionControl",
    "ContextConflictResolver",
    "ContextPropagationManager",
    "ConflictType",
    "PropagationStrategy",
    "ContextVersion",
    "ContextConflict",
    "SynchronizationState",
    
    # Orchestration
    "ProcessOrchestrator",
    "ProcessState",
    "GatewayType",
    "ProcessEvent",
    "ProcessTask",
    "ProcessGateway",
    "TimerEventHandler",
    
    # Quality & Feedback
    "QualityMonitor",
    "FeedbackLoop",
    "ABTestManager",
    "CostOptimizer",
    "ErrorAnalyzer",
    "AnalyticsDashboard",
    "QualityMetric",
    "FeedbackItem",
    "ABTestConfig",
    "FeedbackType",
    "ImprovementStrategy"
]

# Module metadata
MODULE_INFO = {
    "name": "Context Engineering Agent",
    "description": "Advanced context engineering for multi-agent reasoning systems",
    "version": __version__,
    "capabilities": [
        "context_parsing",
        "relevance_assessment",
        "context_optimization",
        "quality_assessment",
        "multi_agent_coordination",
        "semantic_memory",
        "workflow_integration"
    ],
    "requirements": [
        "spacy>=3.0",
        "sentence-transformers>=2.0",
        "scikit-learn>=1.0",
        "numpy>=1.20",
        "networkx>=2.5",
        "aioredis>=2.0",
        "prometheus-client>=0.12"
    ]
}