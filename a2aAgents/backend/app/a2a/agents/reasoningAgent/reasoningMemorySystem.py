"""
Reasoning Memory and Learning System
Implements persistent memory, learning from experience, and adaptive reasoning
"""

import asyncio
import logging
import json
import pickle
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ReasoningExperience:
    """Individual reasoning experience for learning"""
    experience_id: str
    question: str
    decomposition_strategy: str
    patterns_found: Dict[str, Any]
    reasoning_chain: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    timestamp: datetime
    success_metrics: Dict[str, float] = field(default_factory=dict)
    user_feedback: Optional[str] = None
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class MemoryPattern:
    """Learned pattern from past experiences"""
    pattern_id: str
    pattern_type: str
    trigger_conditions: Dict[str, Any]
    successful_strategies: List[str]
    success_rate: float
    usage_count: int
    last_used: datetime
    confidence_decay: float = 0.95

@dataclass
class ReasoningTemplate:
    """Template for common reasoning patterns"""
    template_id: str
    name: str
    description: str
    question_patterns: List[str]
    decomposition_steps: List[Dict[str, Any]]
    success_rate: float
    usage_count: int
    adaptations: List[Dict[str, Any]] = field(default_factory=list)

class ReasoningMemoryStore:
    """Persistent storage for reasoning experiences and patterns"""
    
    def __init__(self, db_path: str = "reasoning_memory.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for memory storage"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Experiences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    question TEXT,
                    decomposition_strategy TEXT,
                    patterns_found TEXT,
                    reasoning_chain TEXT,
                    final_answer TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    success_metrics TEXT,
                    user_feedback TEXT,
                    improvement_suggestions TEXT
                )
            """)
            
            # Patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    trigger_conditions TEXT,
                    successful_strategies TEXT,
                    success_rate REAL,
                    usage_count INTEGER,
                    last_used TEXT,
                    confidence_decay REAL
                )
            """)
            
            # Templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    question_patterns TEXT,
                    decomposition_steps TEXT,
                    success_rate REAL,
                    usage_count INTEGER,
                    adaptations TEXT
                )
            """)
            
            self.connection.commit()
            logger.info("Reasoning memory database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
    
    async def store_experience(self, experience: ReasoningExperience) -> bool:
        """Store reasoning experience"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO experiences 
                (experience_id, question, decomposition_strategy, patterns_found, 
                 reasoning_chain, final_answer, confidence, timestamp, 
                 success_metrics, user_feedback, improvement_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.experience_id,
                experience.question,
                experience.decomposition_strategy,
                json.dumps(experience.patterns_found),
                json.dumps(experience.reasoning_chain),
                experience.final_answer,
                experience.confidence,
                experience.timestamp.isoformat(),
                json.dumps(experience.success_metrics),
                experience.user_feedback,
                json.dumps(experience.improvement_suggestions)
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return False
    
    async def retrieve_similar_experiences(self, question: str, limit: int = 10) -> List[ReasoningExperience]:
        """Retrieve similar past experiences"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM experiences 
                ORDER BY confidence DESC, timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            experiences = []
            
            for row in rows:
                experience = ReasoningExperience(
                    experience_id=row[0],
                    question=row[1],
                    decomposition_strategy=row[2],
                    patterns_found=json.loads(row[3]) if row[3] else {},
                    reasoning_chain=json.loads(row[4]) if row[4] else [],
                    final_answer=row[5],
                    confidence=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    success_metrics=json.loads(row[8]) if row[8] else {},
                    user_feedback=row[9],
                    improvement_suggestions=json.loads(row[10]) if row[10] else []
                )
                experiences.append(experience)
            
            return experiences
            
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    async def store_pattern(self, pattern: MemoryPattern) -> bool:
        """Store learned pattern"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO patterns 
                (pattern_id, pattern_type, trigger_conditions, successful_strategies,
                 success_rate, usage_count, last_used, confidence_decay)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                json.dumps(pattern.trigger_conditions),
                json.dumps(pattern.successful_strategies),
                pattern.success_rate,
                pattern.usage_count,
                pattern.last_used.isoformat(),
                pattern.confidence_decay
            ))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False
    
    async def retrieve_matching_patterns(self, conditions: Dict[str, Any]) -> List[MemoryPattern]:
        """Retrieve patterns matching given conditions"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM patterns 
                ORDER BY success_rate DESC, usage_count DESC
            """)
            
            rows = cursor.fetchall()
            matching_patterns = []
            
            for row in rows:
                pattern = MemoryPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    trigger_conditions=json.loads(row[2]) if row[2] else {},
                    successful_strategies=json.loads(row[3]) if row[3] else [],
                    success_rate=row[4],
                    usage_count=row[5],
                    last_used=datetime.fromisoformat(row[6]),
                    confidence_decay=row[7]
                )
                
                # Simple matching logic (can be enhanced)
                if self._pattern_matches_conditions(pattern, conditions):
                    matching_patterns.append(pattern)
            
            return matching_patterns[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []
    
    def _pattern_matches_conditions(self, pattern: MemoryPattern, conditions: Dict[str, Any]) -> bool:
        """Check if pattern matches given conditions"""
        trigger_conditions = pattern.trigger_conditions
        
        # Check for overlapping keys
        common_keys = set(trigger_conditions.keys()) & set(conditions.keys())
        if not common_keys:
            return False
        
        # Check value similarity
        matches = 0
        for key in common_keys:
            if str(trigger_conditions[key]).lower() in str(conditions[key]).lower():
                matches += 1
        
        return matches >= len(common_keys) * 0.5  # At least 50% match


class ReasoningLearningEngine:
    """Engine for learning from reasoning experiences"""
    
    def __init__(self, memory_store: ReasoningMemoryStore):
        self.memory_store = memory_store
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.adaptation_history = deque(maxlen=1000)
        
        # Learning metrics
        self.performance_tracker = defaultdict(list)
        self.strategy_effectiveness = defaultdict(float)
        self.pattern_usage = defaultdict(int)
    
    async def learn_from_experience(self, experience: ReasoningExperience) -> Dict[str, Any]:
        """Learn patterns and improvements from a reasoning experience"""
        
        # Store the experience
        await self.memory_store.store_experience(experience)
        
        # Extract learning insights
        insights = {
            "new_patterns": [],
            "strategy_updates": {},
            "template_adaptations": [],
            "performance_insights": {}
        }
        
        # 1. Pattern extraction
        new_patterns = await self._extract_patterns_from_experience(experience)
        insights["new_patterns"] = new_patterns
        
        # 2. Strategy effectiveness analysis
        strategy_update = await self._analyze_strategy_effectiveness(experience)
        insights["strategy_updates"] = strategy_update
        
        # 3. Template adaptation
        adaptations = await self._adapt_templates(experience)
        insights["template_adaptations"] = adaptations
        
        # 4. Performance tracking
        performance = await self._track_performance_metrics(experience)
        insights["performance_insights"] = performance
        
        return insights
    
    async def _extract_patterns_from_experience(self, experience: ReasoningExperience) -> List[MemoryPattern]:
        """Extract reusable patterns from successful experiences"""
        patterns = []
        
        if experience.confidence > self.pattern_threshold:
            # Extract question type pattern
            question_pattern = self._classify_question_type(experience.question)
            
            # Extract successful decomposition pattern
            if experience.reasoning_chain:
                pattern_id = hashlib.md5(
                    f"{question_pattern}_{experience.decomposition_strategy}".encode()
                ).hexdigest()[:8]
                
                pattern = MemoryPattern(
                    pattern_id=pattern_id,
                    pattern_type="decomposition_strategy",
                    trigger_conditions={
                        "question_type": question_pattern,
                        "semantic_type": experience.patterns_found.get("semantic_type", "unknown"),
                        "entity_count": len(experience.patterns_found.get("entities", []))
                    },
                    successful_strategies=[experience.decomposition_strategy],
                    success_rate=experience.confidence,
                    usage_count=1,
                    last_used=experience.timestamp
                )
                
                # Store the pattern
                await self.memory_store.store_pattern(pattern)
                patterns.append(pattern)
        
        return patterns
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question into general types for pattern matching"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["how", "what is the process", "steps"]):
            return "process_inquiry"
        elif any(word in question_lower for word in ["why", "cause", "reason"]):
            return "causal_inquiry"
        elif any(word in question_lower for word in ["what", "define", "explain"]):
            return "definitional_inquiry"
        elif any(word in question_lower for word in ["compare", "difference", "versus"]):
            return "comparative_inquiry"
        elif any(word in question_lower for word in ["when", "timeline", "sequence"]):
            return "temporal_inquiry"
        else:
            return "general_inquiry"
    
    async def _analyze_strategy_effectiveness(self, experience: ReasoningExperience) -> Dict[str, float]:
        """Analyze effectiveness of reasoning strategies"""
        strategy = experience.decomposition_strategy
        confidence = experience.confidence
        
        # Update strategy effectiveness
        current_effectiveness = self.strategy_effectiveness.get(strategy, 0.5)
        new_effectiveness = (current_effectiveness * 0.9) + (confidence * 0.1)
        self.strategy_effectiveness[strategy] = new_effectiveness
        
        # Track performance history
        self.performance_tracker[strategy].append({
            "confidence": confidence,
            "timestamp": experience.timestamp,
            "success_metrics": experience.success_metrics
        })
        
        return {
            "strategy": strategy,
            "updated_effectiveness": new_effectiveness,
            "performance_trend": self._calculate_trend(self.performance_tracker[strategy][-10:])
        }
    
    def _calculate_trend(self, recent_performances: List[Dict[str, Any]]) -> str:
        """Calculate performance trend"""
        if len(recent_performances) < 3:
            return "insufficient_data"
        
        confidences = [p["confidence"] for p in recent_performances]
        
        # Simple trend calculation
        early_avg = sum(confidences[:len(confidences)//2]) / (len(confidences)//2)
        late_avg = sum(confidences[len(confidences)//2:]) / (len(confidences) - len(confidences)//2)
        
        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def _adapt_templates(self, experience: ReasoningExperience) -> List[Dict[str, Any]]:
        """Adapt reasoning templates based on successful experiences"""
        adaptations = []
        
        if experience.confidence > 0.8 and experience.reasoning_chain:
            # Create or update template
            question_type = self._classify_question_type(experience.question)
            
            # Check if we have an existing template for this question type
            similar_experiences = await self.memory_store.retrieve_similar_experiences(
                experience.question, limit=20
            )
            
            same_type_experiences = [
                exp for exp in similar_experiences 
                if self._classify_question_type(exp.question) == question_type
            ]
            
            if len(same_type_experiences) >= 3:
                # Enough data to create/update template
                template_adaptation = {
                    "question_type": question_type,
                    "successful_strategies": list(set(exp.decomposition_strategy for exp in same_type_experiences)),
                    "common_patterns": self._extract_common_patterns(same_type_experiences),
                    "average_confidence": sum(exp.confidence for exp in same_type_experiences) / len(same_type_experiences)
                }
                adaptations.append(template_adaptation)
        
        return adaptations
    
    def _extract_common_patterns(self, experiences: List[ReasoningExperience]) -> Dict[str, Any]:
        """Extract common patterns from multiple experiences"""
        pattern_counts = defaultdict(int)
        
        for exp in experiences:
            # Count semantic types
            semantic_type = exp.patterns_found.get("semantic_type", "unknown")
            pattern_counts[f"semantic_type_{semantic_type}"] += 1
            
            # Count reasoning chain lengths
            chain_length = len(exp.reasoning_chain)
            if chain_length <= 2:
                pattern_counts["short_chain"] += 1
            elif chain_length <= 4:
                pattern_counts["medium_chain"] += 1
            else:
                pattern_counts["long_chain"] += 1
        
        # Return most common patterns
        total_experiences = len(experiences)
        common_patterns = {
            pattern: count/total_experiences 
            for pattern, count in pattern_counts.items() 
            if count/total_experiences > 0.5
        }
        
        return common_patterns
    
    async def _track_performance_metrics(self, experience: ReasoningExperience) -> Dict[str, Any]:
        """Track various performance metrics"""
        
        # Calculate metrics
        metrics = {
            "confidence_score": experience.confidence,
            "reasoning_chain_length": len(experience.reasoning_chain),
            "patterns_found_count": sum(len(patterns) for patterns in experience.patterns_found.values() if isinstance(patterns, list)),
            "processing_complexity": self._calculate_complexity(experience),
            "user_satisfaction": self._infer_user_satisfaction(experience)
        }
        
        # Store in performance tracker
        strategy = experience.decomposition_strategy
        self.performance_tracker[f"{strategy}_metrics"].append(metrics)
        
        return metrics
    
    def _calculate_complexity(self, experience: ReasoningExperience) -> float:
        """Calculate processing complexity score"""
        complexity = 0.0
        
        # Question complexity
        question_words = len(experience.question.split())
        complexity += min(question_words / 20.0, 1.0) * 0.3
        
        # Reasoning chain complexity
        chain_length = len(experience.reasoning_chain)
        complexity += min(chain_length / 5.0, 1.0) * 0.4
        
        # Pattern complexity
        pattern_count = sum(len(patterns) for patterns in experience.patterns_found.values() if isinstance(patterns, list))
        complexity += min(pattern_count / 10.0, 1.0) * 0.3
        
        return complexity
    
    def _infer_user_satisfaction(self, experience: ReasoningExperience) -> float:
        """Infer user satisfaction from available signals"""
        satisfaction = experience.confidence  # Base satisfaction on confidence
        
        # Adjust based on feedback
        if experience.user_feedback:
            if any(word in experience.user_feedback.lower() for word in ["good", "helpful", "correct"]):
                satisfaction = min(satisfaction + 0.2, 1.0)
            elif any(word in experience.user_feedback.lower() for word in ["wrong", "bad", "incorrect"]):
                satisfaction = max(satisfaction - 0.3, 0.0)
        
        # Adjust based on improvement suggestions
        if experience.improvement_suggestions:
            suggestion_penalty = len(experience.improvement_suggestions) * 0.1
            satisfaction = max(satisfaction - suggestion_penalty, 0.0)
        
        return satisfaction


class AdaptiveReasoningSystem:
    """System that adapts reasoning strategies based on learned patterns"""
    
    def __init__(self, memory_store: ReasoningMemoryStore, learning_engine: ReasoningLearningEngine):
        self.memory_store = memory_store
        self.learning_engine = learning_engine
        self.adaptation_cache = {}
        
        # Strategy selection weights
        self.strategy_weights = {
            "causal_decomposition": 1.0,
            "temporal_decomposition": 1.0,
            "structural_decomposition": 1.0,
            "comparative_decomposition": 1.0,
            "hypothetical_decomposition": 1.0,
            "definitional_decomposition": 1.0
        }
    
    async def get_adaptive_strategy(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive reasoning strategy based on learned patterns"""
        
        # Classify question
        question_type = self.learning_engine._classify_question_type(question)
        
        # Retrieve matching patterns
        patterns = await self.memory_store.retrieve_matching_patterns({
            "question_type": question_type,
            "context": context
        })
        
        # Retrieve similar experiences
        similar_experiences = await self.memory_store.retrieve_similar_experiences(question, limit=5)
        
        # Select optimal strategy
        strategy_recommendation = await self._select_optimal_strategy(
            question_type, patterns, similar_experiences, context
        )
        
        # Generate adaptive parameters
        adaptive_params = await self._generate_adaptive_parameters(
            strategy_recommendation, patterns, context
        )
        
        return {
            "recommended_strategy": strategy_recommendation,
            "adaptive_parameters": adaptive_params,
            "confidence": self._calculate_strategy_confidence(patterns, similar_experiences),
            "learning_insights": self._extract_learning_insights(patterns, similar_experiences)
        }
    
    async def _select_optimal_strategy(self, question_type: str, patterns: List[MemoryPattern],
                                     experiences: List[ReasoningExperience], context: Dict[str, Any]) -> str:
        """Select optimal reasoning strategy"""
        
        strategy_scores = {}
        
        # Score based on learned patterns
        for pattern in patterns:
            for strategy in pattern.successful_strategies:
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = 0.0
                
                # Weight by success rate and recency
                age_factor = self._calculate_age_factor(pattern.last_used)
                strategy_scores[strategy] += pattern.success_rate * age_factor * 0.4
        
        # Score based on similar experiences
        for experience in experiences:
            strategy = experience.decomposition_strategy
            if strategy not in strategy_scores:
                strategy_scores[strategy] = 0.0
            
            # Weight by confidence and recency
            age_factor = self._calculate_age_factor(experience.timestamp)
            strategy_scores[strategy] += experience.confidence * age_factor * 0.6
        
        # Apply global strategy weights
        for strategy in strategy_scores:
            if strategy in self.strategy_weights:
                strategy_scores[strategy] *= self.strategy_weights[strategy]
        
        # Select highest scoring strategy
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            return best_strategy
        else:
            # Fallback to question type mapping
            strategy_mapping = {
                "process_inquiry": "structural_decomposition",
                "causal_inquiry": "causal_decomposition",
                "definitional_inquiry": "definitional_decomposition",
                "comparative_inquiry": "comparative_decomposition",
                "temporal_inquiry": "temporal_decomposition",
                "general_inquiry": "structural_decomposition"
            }
            return strategy_mapping.get(question_type, "structural_decomposition")
    
    def _calculate_age_factor(self, timestamp: datetime) -> float:
        """Calculate age factor for weighting recent experiences more heavily"""
        age = datetime.now() - timestamp
        age_days = age.total_seconds() / (24 * 3600)
        
        # Exponential decay with half-life of 30 days
        return np.exp(-age_days / 30.0)
    
    async def _generate_adaptive_parameters(self, strategy: str, patterns: List[MemoryPattern],
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive parameters for the selected strategy"""
        
        params = {
            "max_depth": 3,
            "confidence_threshold": 0.5,
            "pattern_weight": 0.7,
            "validation_enabled": True
        }
        
        # Adapt based on learned patterns
        if patterns:
            avg_success_rate = sum(p.success_rate for p in patterns) / len(patterns)
            
            # Adjust depth based on success rate
            if avg_success_rate > 0.8:
                params["max_depth"] = 4  # More depth for high-success patterns
            elif avg_success_rate < 0.6:
                params["max_depth"] = 2  # Less depth for low-success patterns
            
            # Adjust thresholds
            params["confidence_threshold"] = max(0.3, avg_success_rate - 0.2)
            params["pattern_weight"] = min(0.9, avg_success_rate + 0.1)
        
        # Strategy-specific adaptations
        if strategy == "causal_decomposition":
            params["causal_depth"] = 3
            params["mechanism_focus"] = True
        elif strategy == "temporal_decomposition":
            params["sequence_extraction"] = True
            params["temporal_ordering"] = True
        elif strategy == "comparative_decomposition":
            params["similarity_threshold"] = 0.7
            params["contrast_emphasis"] = True
        
        return params
    
    def _calculate_strategy_confidence(self, patterns: List[MemoryPattern],
                                     experiences: List[ReasoningExperience]) -> float:
        """Calculate confidence in strategy recommendation"""
        
        if not patterns and not experiences:
            from .reasoningConfidenceCalculator import calculate_fallback_confidence
            return calculate_fallback_confidence("no_historical_data")
        
        # Pattern-based confidence
        pattern_confidence = 0.0
        if patterns:
            pattern_confidence = sum(p.success_rate for p in patterns) / len(patterns)
        
        # Experience-based confidence
        experience_confidence = 0.0
        if experiences:
            experience_confidence = sum(e.confidence for e in experiences) / len(experiences)
        
        # Combine confidences
        if patterns and experiences:
            return (pattern_confidence * 0.6) + (experience_confidence * 0.4)
        elif patterns:
            return pattern_confidence * 0.8
        elif experiences:
            return experience_confidence * 0.8
        else:
            from .reasoningConfidenceCalculator import calculate_fallback_confidence
            return calculate_fallback_confidence("no_historical_data")
    
    def _extract_learning_insights(self, patterns: List[MemoryPattern],
                                 experiences: List[ReasoningExperience]) -> List[str]:
        """Extract insights from learning data"""
        insights = []
        
        if patterns:
            most_successful_pattern = max(patterns, key=lambda p: p.success_rate)
            insights.append(f"Most successful pattern: {most_successful_pattern.pattern_type} (success rate: {most_successful_pattern.success_rate:.2f})")
        
        if experiences:
            avg_confidence = sum(e.confidence for e in experiences) / len(experiences)
            insights.append(f"Average historical confidence: {avg_confidence:.2f}")
            
            strategies_used = set(e.decomposition_strategy for e in experiences)
            insights.append(f"Strategies with proven success: {', '.join(strategies_used)}")
        
        if len(experiences) >= 3:
            recent_trend = self.learning_engine._calculate_trend(
                [{"confidence": e.confidence} for e in experiences[-3:]]
            )
            insights.append(f"Recent performance trend: {recent_trend}")
        
        return insights
    
    async def update_strategy_weights(self, feedback: Dict[str, Any]):
        """Update strategy weights based on feedback"""
        
        strategy = feedback.get("strategy")
        success = feedback.get("success", False)
        confidence = feedback.get("confidence", 0.5)
        
        if strategy in self.strategy_weights:
            if success and confidence > 0.7:
                # Boost successful strategies
                self.strategy_weights[strategy] = min(self.strategy_weights[strategy] * 1.1, 2.0)
            elif not success or confidence < 0.4:
                # Reduce weight for unsuccessful strategies
                self.strategy_weights[strategy] = max(self.strategy_weights[strategy] * 0.9, 0.5)
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")


# Factory function for creating the complete memory system
def create_reasoning_memory_system(db_path: str = None) -> Tuple[ReasoningMemoryStore, ReasoningLearningEngine, AdaptiveReasoningSystem]:
    """Create a complete reasoning memory and learning system"""
    
    if db_path is None:
        db_path = Path(__file__).parent / "reasoning_memory.db"
    
    memory_store = ReasoningMemoryStore(str(db_path))
    learning_engine = ReasoningLearningEngine(memory_store)
    adaptive_system = AdaptiveReasoningSystem(memory_store, learning_engine)
    
    return memory_store, learning_engine, adaptive_system