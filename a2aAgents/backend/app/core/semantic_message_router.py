"""
Semantic Message Routing System for A2A Platform
AI-powered intelligent message routing based on semantic understanding and context
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import re

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.getLogger(__name__).warning("ML libraries not available for semantic routing")

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """AI-classified message types"""
    TASK_REQUEST = "task_request"
    QUERY = "query"
    NOTIFICATION = "notification"
    SYSTEM_COMMAND = "system_command"
    DATA_TRANSFER = "data_transfer"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    WORKFLOW = "workflow"


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "direct"  # Route directly to specific agent
    BROADCAST = "broadcast"  # Send to all relevant agents
    LOAD_BALANCED = "load_balanced"  # Route based on load balancing
    CAPABILITY_BASED = "capability_based"  # Route based on agent capabilities
    SEMANTIC_MATCH = "semantic_match"  # Route based on semantic similarity
    PRIORITY_QUEUE = "priority_queue"  # Route through priority queues
    WORKFLOW_CHAIN = "workflow_chain"  # Route through workflow sequence


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


@dataclass
class SemanticMessage:
    """Enhanced message with semantic analysis"""
    message_id: str
    sender_id: str
    content: str
    message_type: MessageType
    priority: MessagePriority
    timestamp: float

    # Semantic analysis results
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = None
    keywords: List[str] = None
    semantic_vector: Optional[List[float]] = None
    context_vector: Optional[List[float]] = None

    # Routing metadata
    target_agents: List[str] = None
    routing_strategy: Optional[RoutingStrategy] = None
    routing_confidence: float = 0.0
    routing_reasons: List[str] = None

    # Processing metadata
    processing_started: Optional[float] = None
    processing_completed: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.keywords is None:
            self.keywords = []
        if self.target_agents is None:
            self.target_agents = []
        if self.routing_reasons is None:
            self.routing_reasons = []


@dataclass
class AgentCapabilityProfile:
    """Agent capability profile for semantic routing"""
    agent_id: str
    capabilities: Set[str]
    specializations: Dict[str, float]  # capability -> expertise score (0-1)
    message_history: deque
    current_load: int = 0
    max_capacity: int = 10
    response_time_avg: float = 1.0
    success_rate: float = 1.0
    preferred_message_types: Set[MessageType] = None
    semantic_profile: Optional[List[float]] = None  # Learned semantic preferences
    last_activity: float = 0.0
    availability_score: float = 1.0

    def __post_init__(self):
        if self.preferred_message_types is None:
            self.preferred_message_types = set()
        if not hasattr(self, 'message_history') or self.message_history is None:
            self.message_history = deque(maxlen=100)


class SemanticAnalyzer:
    """AI-powered semantic message analysis"""

    def __init__(self):
        self.tfidf_vectorizer = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.context_analyzer = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None

        # Knowledge base for intent classification
        self.intent_patterns = {
            'task_request': [
                r'\b(execute|run|perform|process|handle|do)\b',
                r'\b(please|can you|could you)\b.*\b(task|job|work)\b',
                r'\b(need|want|require)\b.*\b(done|completed|finished)\b'
            ],
            'query': [
                r'\b(what|how|when|where|why|which)\b',
                r'\b(status|information|data|details)\b.*\?',
                r'\b(get|retrieve|fetch|find)\b.*\b(info|data)\b'
            ],
            'system_command': [
                r'\b(start|stop|restart|shutdown|initialize)\b',
                r'\b(config|configure|setup|deploy)\b',
                r'\b(update|upgrade|install|uninstall)\b'
            ],
            'notification': [
                r'\b(alert|notify|inform|announce)\b',
                r'\b(completed|finished|failed|error)\b',
                r'\b(update|status|progress)\b.*\b(report)\b'
            ]
        }

        # Entity extraction patterns
        self.entity_patterns = {
            'agent_id': r'\bagent[_-]?(\w+)\b',
            'task_id': r'\btask[_-]?(\w+)\b',
            'file_name': r'\b[\w-]+\.(txt|json|xml|csv|log)\b',
            'url': r'https?://[\w\.-]+\.\w+[\w\._~:/?#[\]@!\$&\'()*+,;=-]*',
            'timestamp': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            'number': r'\b\d+\.?\d*\b',
            'email': r'\b[\w\.-]+@[\w\.-]+\.\w+\b'
        }

        # Initialize ML components
        if ML_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.intent_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        self.trained = False
        self.message_corpus = []

    def analyze_message(self, message: SemanticMessage) -> SemanticMessage:
        """Perform comprehensive semantic analysis of message"""

        # Intent classification
        message.intent = self._classify_intent(message.content)

        # Entity extraction
        message.entities = self._extract_entities(message.content)

        # Keyword extraction
        message.keywords = self._extract_keywords(message.content)

        # Semantic vectorization
        if ML_AVAILABLE and self.trained:
            message.semantic_vector = self._vectorize_content(message.content)
            message.context_vector = self._extract_context_vector(message)

        return message

    def _classify_intent(self, content: str) -> str:
        """Classify message intent using patterns and ML"""

        content_lower = content.lower()

        # Pattern-based classification
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        # ML-based classification if available and trained
        if ML_AVAILABLE and self.trained and self.intent_classifier:
            try:
                features = self.tfidf_vectorizer.transform([content])
                predicted_intent = self.intent_classifier.predict(features)[0]
                confidence = max(self.intent_classifier.predict_proba(features)[0])

                if confidence > 0.7:
                    return predicted_intent
            except Exception as e:
                logger.warning(f"ML intent classification failed: {e}")

        return "unknown"

    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities from message content"""

        entities = []

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity = {
                    'type': entity_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8  # Pattern-based confidence
                }
                entities.append(entity)

        return entities

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from message content"""

        # Simple keyword extraction based on word frequency and importance
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}

        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = defaultdict(int)

        for word in words:
            if word not in stop_words:
                word_freq[word] += 1

        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

    def _vectorize_content(self, content: str) -> Optional[List[float]]:
        """Convert content to semantic vector"""

        if not ML_AVAILABLE or not self.trained:
            return None

        try:
            tfidf_vector = self.tfidf_vectorizer.transform([content])
            return tfidf_vector.toarray()[0].tolist()
        except Exception as e:
            logger.error(f"Content vectorization failed: {e}")
            return None

    def _extract_context_vector(self, message: SemanticMessage) -> Optional[List[float]]:
        """Extract context vector considering message metadata"""

        if not ML_AVAILABLE:
            return None

        try:
            # Create context features
            context_features = [
                message.priority.value,
                len(message.content),
                len(message.keywords),
                len(message.entities),
                hash(message.sender_id) % 1000 / 1000.0,  # Normalized sender hash
                time.time() % 86400 / 86400.0,  # Time of day normalized
                1.0 if message.intent != "unknown" else 0.0
            ]

            return context_features
        except Exception as e:
            logger.error(f"Context vector extraction failed: {e}")
            return None

    async def train_models(self, training_messages: List[SemanticMessage]):
        """Train ML models with message data"""

        if not ML_AVAILABLE or len(training_messages) < 100:
            return

        try:
            # Prepare training data
            contents = [msg.content for msg in training_messages]
            intents = [msg.intent for msg in training_messages if msg.intent != "unknown"]

            if len(contents) < 50 or len(intents) < 50:
                logger.info("Insufficient training data for semantic models")
                return

            # Train TF-IDF vectorizer
            self.tfidf_vectorizer.fit(contents)

            # Train intent classifier
            if len(set(intents)) >= 2:  # Need at least 2 classes
                X = self.tfidf_vectorizer.transform([msg.content for msg in training_messages if msg.intent != "unknown"])
                y = intents

                self.intent_classifier.fit(X, y)

            self.trained = True
            self.message_corpus.extend(contents)

            logger.info(f"Semantic models trained with {len(training_messages)} messages")

        except Exception as e:
            logger.error(f"Model training error: {e}")


class IntelligentRouter:
    """AI-powered intelligent message routing engine"""

    def __init__(self):
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.routing_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(lambda: {'success': 0, 'total': 0})

        # Routing algorithms
        self.routing_algorithms = {
            RoutingStrategy.DIRECT: self._route_direct,
            RoutingStrategy.BROADCAST: self._route_broadcast,
            RoutingStrategy.LOAD_BALANCED: self._route_load_balanced,
            RoutingStrategy.CAPABILITY_BASED: self._route_capability_based,
            RoutingStrategy.SEMANTIC_MATCH: self._route_semantic_match,
            RoutingStrategy.PRIORITY_QUEUE: self._route_priority_queue,
            RoutingStrategy.WORKFLOW_CHAIN: self._route_workflow_chain
        }

        # Learning components
        self.routing_optimizer = None
        if ML_AVAILABLE:
            self.routing_optimizer = RandomForestClassifier(n_estimators=50, random_state=42)

    def register_agent(self, agent_profile: AgentCapabilityProfile):
        """Register an agent with the routing system"""
        self.agent_profiles[agent_profile.agent_id] = agent_profile
        logger.info(f"Registered agent {agent_profile.agent_id} with {len(agent_profile.capabilities)} capabilities")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the routing system"""
        if agent_id in self.agent_profiles:
            del self.agent_profiles[agent_id]
            logger.info(f"Unregistered agent {agent_id}")

    async def route_message(self, message: SemanticMessage) -> Tuple[List[str], RoutingStrategy, float]:
        """Intelligently route message to optimal agents"""

        # Determine optimal routing strategy
        strategy, confidence = self._select_routing_strategy(message)
        message.routing_strategy = strategy
        message.routing_confidence = confidence

        # Execute routing algorithm
        routing_func = self.routing_algorithms.get(strategy, self._route_capability_based)
        target_agents, routing_reasons = await routing_func(message)

        message.target_agents = target_agents
        message.routing_reasons = routing_reasons

        # Record routing decision for learning
        self._record_routing_decision(message, strategy, target_agents, confidence)

        return target_agents, strategy, confidence

    def _select_routing_strategy(self, message: SemanticMessage) -> Tuple[RoutingStrategy, float]:
        """Select optimal routing strategy based on message analysis"""

        strategy_scores = {}

        # Message type based strategy selection
        if message.message_type == MessageType.SYSTEM_COMMAND:
            strategy_scores[RoutingStrategy.BROADCAST] = 0.8
        elif message.message_type == MessageType.TASK_REQUEST:
            strategy_scores[RoutingStrategy.CAPABILITY_BASED] = 0.9
            strategy_scores[RoutingStrategy.SEMANTIC_MATCH] = 0.7
        elif message.message_type == MessageType.QUERY:
            strategy_scores[RoutingStrategy.SEMANTIC_MATCH] = 0.8
            strategy_scores[RoutingStrategy.CAPABILITY_BASED] = 0.6
        elif message.message_type == MessageType.NOTIFICATION:
            strategy_scores[RoutingStrategy.BROADCAST] = 0.7
        elif message.message_type == MessageType.WORKFLOW:
            strategy_scores[RoutingStrategy.WORKFLOW_CHAIN] = 0.9

        # Priority based adjustments
        if message.priority == MessagePriority.CRITICAL:
            strategy_scores[RoutingStrategy.PRIORITY_QUEUE] = strategy_scores.get(RoutingStrategy.PRIORITY_QUEUE, 0) + 0.3

        # Load balancing considerations
        if self._is_system_under_load():
            strategy_scores[RoutingStrategy.LOAD_BALANCED] = strategy_scores.get(RoutingStrategy.LOAD_BALANCED, 0) + 0.2

        # Intent based adjustments
        if message.intent and "specific" in message.intent:
            strategy_scores[RoutingStrategy.DIRECT] = strategy_scores.get(RoutingStrategy.DIRECT, 0) + 0.4

        if not strategy_scores:
            return RoutingStrategy.CAPABILITY_BASED, 0.5

        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]

    async def _route_direct(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route directly to a specific agent"""

        # Look for explicit agent mentions in entities
        target_agent = None
        reasons = ["Direct routing requested"]

        for entity in message.entities:
            if entity['type'] == 'agent_id':
                agent_id = entity['value'].replace('agent_', '').replace('agent-', '')
                if agent_id in self.agent_profiles:
                    target_agent = agent_id
                    reasons.append(f"Agent {agent_id} explicitly mentioned")
                    break

        if not target_agent:
            # Fallback to capability-based routing
            return await self._route_capability_based(message)

        return [target_agent], reasons

    async def _route_broadcast(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Broadcast message to all relevant agents"""

        # Filter agents based on message type preferences
        relevant_agents = []
        reasons = ["Broadcast routing selected"]

        for agent_id, profile in self.agent_profiles.items():
            if (not profile.preferred_message_types or
                message.message_type in profile.preferred_message_types):
                relevant_agents.append(agent_id)
                reasons.append(f"Agent {agent_id} accepts {message.message_type.value}")

        return relevant_agents, reasons

    async def _route_load_balanced(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route to agent with lowest current load"""

        if not self.agent_profiles:
            return [], ["No agents available"]

        # Calculate load scores for all agents
        agent_loads = []
        for agent_id, profile in self.agent_profiles.items():
            load_ratio = profile.current_load / max(1, profile.max_capacity)
            availability_factor = profile.availability_score
            response_factor = 1.0 / max(0.1, profile.response_time_avg)
            success_factor = profile.success_rate

            # Combined score (lower is better for load, higher is better for performance)
            score = load_ratio - (availability_factor + response_factor + success_factor) / 3
            agent_loads.append((agent_id, score))

        # Sort by score (best first)
        agent_loads.sort(key=lambda x: x[1])
        best_agent = agent_loads[0][0]

        reasons = [
            "Load-balanced routing selected",
            f"Agent {best_agent} has optimal load/performance ratio"
        ]

        return [best_agent], reasons

    async def _route_capability_based(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route based on agent capabilities and specializations"""

        if not self.agent_profiles:
            return [], ["No agents available"]

        # Score agents based on capability match
        agent_scores = []
        reasons = ["Capability-based routing selected"]

        for agent_id, profile in self.agent_profiles.items():
            score = 0
            matched_capabilities = []

            # Check keyword matches with capabilities
            for keyword in message.keywords:
                for capability in profile.capabilities:
                    if keyword.lower() in capability.lower():
                        expertise = profile.specializations.get(capability, 0.5)
                        score += expertise
                        matched_capabilities.append(capability)

            # Check intent matching
            if message.intent and message.intent != "unknown":
                for capability in profile.capabilities:
                    if message.intent.lower() in capability.lower():
                        score += profile.specializations.get(capability, 0.5) * 1.5
                        matched_capabilities.append(capability)

            # Apply performance factors
            score *= profile.success_rate * profile.availability_score

            if score > 0:
                agent_scores.append((agent_id, score, matched_capabilities))

        if not agent_scores:
            # Fallback to any available agent
            available_agents = [aid for aid, profile in self.agent_profiles.items()
                             if profile.current_load < profile.max_capacity]
            if available_agents:
                return [available_agents[0]], ["Fallback to available agent"]
            return [], ["No suitable agents found"]

        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_score, capabilities = agent_scores[0]

        reasons.append(f"Agent {best_agent} matched capabilities: {', '.join(capabilities)}")
        reasons.append(f"Capability match score: {best_score:.2f}")

        return [best_agent], reasons

    async def _route_semantic_match(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route based on semantic similarity"""

        if not ML_AVAILABLE or not message.semantic_vector:
            # Fallback to capability-based routing
            return await self._route_capability_based(message)

        agent_similarities = []
        reasons = ["Semantic matching routing selected"]

        for agent_id, profile in self.agent_profiles.items():
            if profile.semantic_profile:
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [message.semantic_vector],
                    [profile.semantic_profile]
                )[0][0]

                # Apply performance weighting
                weighted_similarity = similarity * profile.success_rate * profile.availability_score
                agent_similarities.append((agent_id, weighted_similarity))

        if not agent_similarities:
            return await self._route_capability_based(message)

        # Sort by similarity (highest first)
        agent_similarities.sort(key=lambda x: x[1], reverse=True)
        best_agent, similarity_score = agent_similarities[0]

        reasons.append(f"Agent {best_agent} has highest semantic similarity: {similarity_score:.3f}")

        return [best_agent], reasons

    async def _route_priority_queue(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route through priority-aware queuing"""

        # Find agents with lowest priority queue backlog
        if not self.agent_profiles:
            return [], ["No agents available"]

        priority_scores = []
        for agent_id, profile in self.agent_profiles.items():
            # Estimate queue backlog based on current load and priority
            queue_factor = 1.0 / max(0.1, profile.current_load + 1)
            priority_factor = message.priority.value / 4.0  # Normalize to 0-1
            response_factor = 1.0 / max(0.1, profile.response_time_avg)

            combined_score = queue_factor * priority_factor * response_factor * profile.availability_score
            priority_scores.append((agent_id, combined_score))

        priority_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent = priority_scores[0][0]

        reasons = [
            "Priority queue routing selected",
            f"Agent {best_agent} has optimal priority handling capacity"
        ]

        return [best_agent], reasons

    async def _route_workflow_chain(self, message: SemanticMessage) -> Tuple[List[str], List[str]]:
        """Route through workflow chain sequence"""

        # This is a simplified implementation
        # In practice, would maintain workflow state and chain definitions

        workflow_agents = []
        reasons = ["Workflow chain routing selected"]

        # Look for workflow-related keywords
        workflow_keywords = ['workflow', 'chain', 'sequence', 'pipeline']
        has_workflow = any(kw in message.content.lower() for kw in workflow_keywords)

        if has_workflow:
            # Find agents capable of workflow coordination
            for agent_id, profile in self.agent_profiles.items():
                if any('workflow' in cap.lower() or 'coordination' in cap.lower()
                      for cap in profile.capabilities):
                    workflow_agents.append(agent_id)
                    reasons.append(f"Agent {agent_id} has workflow capabilities")

        if not workflow_agents:
            return await self._route_capability_based(message)

        return workflow_agents, reasons

    def _is_system_under_load(self) -> bool:
        """Check if system is under high load"""
        if not self.agent_profiles:
            return False

        total_load = sum(profile.current_load for profile in self.agent_profiles.values())
        total_capacity = sum(profile.max_capacity for profile in self.agent_profiles.values())

        if total_capacity == 0:
            return False

        load_ratio = total_load / total_capacity
        return load_ratio > 0.8  # 80% load threshold

    def _record_routing_decision(self,
                               message: SemanticMessage,
                               strategy: RoutingStrategy,
                               target_agents: List[str],
                               confidence: float):
        """Record routing decision for learning and analysis"""

        routing_record = {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'message_type': message.message_type.value,
            'intent': message.intent,
            'keywords': message.keywords,
            'strategy': strategy.value,
            'target_agents': target_agents,
            'confidence': confidence,
            'timestamp': time.time()
        }

        self.routing_history.append(routing_record)

    async def update_routing_outcome(self,
                                   message_id: str,
                                   target_agent: str,
                                   success: bool,
                                   response_time: float):
        """Update routing outcome for learning"""

        # Update agent profile
        if target_agent in self.agent_profiles:
            profile = self.agent_profiles[target_agent]

            # Update response time (exponential moving average)
            alpha = 0.1
            profile.response_time_avg = (alpha * response_time +
                                       (1 - alpha) * profile.response_time_avg)

            # Update success rate
            profile.success_rate = (profile.success_rate * 0.95 +
                                  (1.0 if success else 0.0) * 0.05)

            # Update last activity
            profile.last_activity = time.time()

        # Update performance metrics
        self.performance_metrics[target_agent]['total'] += 1
        if success:
            self.performance_metrics[target_agent]['success'] += 1

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""

        if not self.routing_history:
            return {"error": "No routing data available"}

        # Strategy distribution
        strategy_counts = defaultdict(int)
        for record in self.routing_history:
            strategy_counts[record['strategy']] += 1

        # Performance metrics
        total_routes = len(self.routing_history)
        avg_confidence = np.mean([r['confidence'] for r in self.routing_history])

        # Agent utilization
        agent_utilization = {}
        for agent_id, profile in self.agent_profiles.items():
            utilization = profile.current_load / max(1, profile.max_capacity)
            agent_utilization[agent_id] = {
                'utilization': utilization,
                'success_rate': profile.success_rate,
                'avg_response_time': profile.response_time_avg,
                'capabilities_count': len(profile.capabilities)
            }

        return {
            'total_messages_routed': total_routes,
            'strategy_distribution': dict(strategy_counts),
            'average_confidence': avg_confidence,
            'registered_agents': len(self.agent_profiles),
            'agent_utilization': agent_utilization,
            'ml_available': ML_AVAILABLE,
            'system_load': self._is_system_under_load()
        }


class SemanticMessageRouter:
    """Main semantic message routing system"""

    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.intelligent_router = IntelligentRouter()
        self.message_queue = asyncio.Queue()
        self.processing_tasks = []
        self.running = False

        # Performance tracking
        self.total_messages = 0
        self.processing_times = deque(maxlen=1000)

    async def initialize(self):
        """Initialize the semantic message router"""
        logger.info("Semantic message router initialized")

    async def start(self, num_workers: int = 3):
        """Start message processing workers"""
        self.running = True

        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._message_worker(f"worker-{i}"))
            self.processing_tasks.append(task)

        logger.info(f"Started {num_workers} message processing workers")

    async def stop(self):
        """Stop the message router"""
        self.running = False

        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        logger.info("Semantic message router stopped")

    async def route_message(self,
                          sender_id: str,
                          content: str,
                          message_type: MessageType = MessageType.QUERY,
                          priority: MessagePriority = MessagePriority.NORMAL) -> Dict[str, Any]:
        """Route a message through the semantic routing system"""

        start_time = time.time()

        # Create semantic message
        message = SemanticMessage(
            message_id=hashlib.md5(f"{sender_id}{content}{time.time()}".encode()).hexdigest(),
            sender_id=sender_id,
            content=content,
            message_type=message_type,
            priority=priority,
            timestamp=time.time()
        )

        # Semantic analysis
        message = self.semantic_analyzer.analyze_message(message)

        # Intelligent routing
        target_agents, strategy, confidence = await self.intelligent_router.route_message(message)

        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.total_messages += 1

        return {
            'message_id': message.message_id,
            'target_agents': target_agents,
            'routing_strategy': strategy.value,
            'confidence': confidence,
            'intent': message.intent,
            'keywords': message.keywords,
            'entities': message.entities,
            'processing_time': processing_time,
            'routing_reasons': message.routing_reasons
        }

    async def _message_worker(self, worker_id: str):
        """Message processing worker"""

        while self.running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Process message
                await self._process_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, message: SemanticMessage):
        """Process a single message"""

        message.processing_started = time.time()

        try:
            # Perform semantic analysis
            message = self.semantic_analyzer.analyze_message(message)

            # Route message
            target_agents, strategy, confidence = await self.intelligent_router.route_message(message)

            # Update processing completion
            message.processing_completed = time.time()

            logger.info(f"Routed message {message.message_id} to {len(target_agents)} agents using {strategy.value}")

        except Exception as e:
            logger.error(f"Message processing error: {e}")

    def register_agent(self,
                      agent_id: str,
                      capabilities: List[str],
                      specializations: Optional[Dict[str, float]] = None):
        """Register an agent with the routing system"""

        profile = AgentCapabilityProfile(
            agent_id=agent_id,
            capabilities=set(capabilities),
            specializations=specializations or {},
            message_history=deque(maxlen=100)
        )

        self.intelligent_router.register_agent(profile)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        self.intelligent_router.unregister_agent(agent_id)

    async def update_agent_status(self,
                                agent_id: str,
                                current_load: int,
                                availability_score: float = 1.0):
        """Update agent status information"""

        if agent_id in self.intelligent_router.agent_profiles:
            profile = self.intelligent_router.agent_profiles[agent_id]
            profile.current_load = current_load
            profile.availability_score = availability_score
            profile.last_activity = time.time()

    async def report_message_outcome(self,
                                   message_id: str,
                                   target_agent: str,
                                   success: bool,
                                   response_time: float):
        """Report the outcome of message processing"""

        await self.intelligent_router.update_routing_outcome(
            message_id, target_agent, success, response_time
        )

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""

        routing_analytics = self.intelligent_router.get_routing_analytics()

        avg_processing_time = (np.mean(self.processing_times)
                             if self.processing_times else 0.0)

        return {
            'total_messages_processed': self.total_messages,
            'average_processing_time': avg_processing_time,
            'queue_size': self.message_queue.qsize(),
            'active_workers': len(self.processing_tasks),
            'semantic_analyzer_trained': self.semantic_analyzer.trained,
            **routing_analytics
        }


# Global instance
semantic_router = None

async def initialize_semantic_router():
    """Initialize the global semantic message router"""
    global semantic_router
    semantic_router = SemanticMessageRouter()
    await semantic_router.initialize()
    await semantic_router.start()
    logger.info("Global semantic message router initialized")

async def shutdown_semantic_router():
    """Shutdown the global semantic message router"""
    global semantic_router
    if semantic_router:
        await semantic_router.stop()
        logger.info("Semantic message router shutdown complete")
