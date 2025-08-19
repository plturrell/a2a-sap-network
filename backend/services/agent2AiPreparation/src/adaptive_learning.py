"""
Adaptive Learning System for Financial Domain Preprocessing
Implements continuous learning and vocabulary expansion capabilities
"""

import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import logging
import sqlite3
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """A pattern learned from user feedback or data analysis"""
    pattern_type: str  # 'synonym', 'risk_indicator', 'regulatory_mapping', 'hierarchy'
    source_term: str
    target_term: str
    confidence_score: float
    usage_count: int
    first_seen: datetime
    last_updated: datetime
    context: str  # business context where pattern applies


@dataclass
class UsageStats:
    """Statistics about term usage and effectiveness"""
    term: str
    search_count: int
    selection_count: int  # How often users select results with this term
    effectiveness_score: float  # selection_count / search_count
    contexts: List[str]  # Business contexts where term appears
    last_used: datetime


@dataclass
class FeedbackEvent:
    """User feedback event for learning"""
    event_id: str
    timestamp: datetime
    query_terms: List[str]
    returned_entities: List[str]
    selected_entities: List[str]  # What user actually selected
    entity_types: List[str]
    business_context: str


class AdaptiveLearningStorage:
    """Persistent storage for learned patterns and statistics"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "adaptive_learning.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for learning storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    source_term TEXT NOT NULL,
                    target_term TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    first_seen TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    context TEXT,
                    UNIQUE(pattern_type, source_term, target_term, context)
                );
                
                CREATE TABLE IF NOT EXISTS usage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    search_count INTEGER DEFAULT 0,
                    selection_count INTEGER DEFAULT 0,
                    effectiveness_score REAL DEFAULT 0.0,
                    contexts TEXT,  -- JSON array
                    last_used TIMESTAMP NOT NULL,
                    UNIQUE(term)
                );
                
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    query_terms TEXT,  -- JSON array
                    returned_entities TEXT,  -- JSON array
                    selected_entities TEXT,  -- JSON array
                    entity_types TEXT,  -- JSON array
                    business_context TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns(pattern_type);
                CREATE INDEX IF NOT EXISTS idx_patterns_source ON learned_patterns(source_term);
                CREATE INDEX IF NOT EXISTS idx_stats_term ON usage_stats(term);
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_events(timestamp);
            """)
    
    def save_learned_pattern(self, pattern: LearnedPattern) -> bool:
        """Save or update a learned pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learned_patterns 
                    (pattern_type, source_term, target_term, confidence_score, 
                     usage_count, first_seen, last_updated, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_type, pattern.source_term, pattern.target_term,
                    pattern.confidence_score, pattern.usage_count,
                    pattern.first_seen.isoformat(), pattern.last_updated.isoformat(),
                    pattern.context
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to save learned pattern: {e}")
            return False
    
    def get_learned_patterns(self, pattern_type: Optional[str] = None, 
                           context: Optional[str] = None) -> List[LearnedPattern]:
        """Retrieve learned patterns with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM learned_patterns WHERE 1=1"
                params = []
                
                if pattern_type:
                    query += " AND pattern_type = ?"
                    params.append(pattern_type)
                
                if context:
                    query += " AND context = ?"
                    params.append(context)
                
                query += " ORDER BY confidence_score DESC, usage_count DESC"
                
                cursor = conn.execute(query, params)
                patterns = []
                
                for row in cursor.fetchall():
                    patterns.append(LearnedPattern(
                        pattern_type=row[1],
                        source_term=row[2],
                        target_term=row[3],
                        confidence_score=row[4],
                        usage_count=row[5],
                        first_seen=datetime.fromisoformat(row[6]),
                        last_updated=datetime.fromisoformat(row[7]),
                        context=row[8] or ""
                    ))
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to retrieve learned patterns: {e}")
            return []
    
    def update_usage_stats(self, term: str, searched: bool = False, 
                          selected: bool = False, context: str = "") -> bool:
        """Update usage statistics for a term"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get existing stats
                cursor = conn.execute("SELECT * FROM usage_stats WHERE term = ?", (term,))
                row = cursor.fetchone()
                
                if row:
                    # Update existing
                    search_count = row[2] + (1 if searched else 0)
                    selection_count = row[3] + (1 if selected else 0)
                    effectiveness = selection_count / max(search_count, 1)
                    contexts = json.loads(row[5] or "[]")
                    if context and context not in contexts:
                        contexts.append(context)
                    
                    conn.execute("""
                        UPDATE usage_stats 
                        SET search_count = ?, selection_count = ?, 
                            effectiveness_score = ?, contexts = ?, last_used = ?
                        WHERE term = ?
                    """, (search_count, selection_count, effectiveness, 
                          json.dumps(contexts), datetime.now().isoformat(), term))
                else:
                    # Create new
                    search_count = 1 if searched else 0
                    selection_count = 1 if selected else 0
                    effectiveness = selection_count / max(search_count, 1)
                    contexts = [context] if context else []
                    
                    conn.execute("""
                        INSERT INTO usage_stats 
                        (term, search_count, selection_count, effectiveness_score, 
                         contexts, last_used)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (term, search_count, selection_count, effectiveness,
                          json.dumps(contexts), datetime.now().isoformat()))
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to update usage stats: {e}")
            return False
    
    def save_feedback_event(self, feedback: FeedbackEvent) -> bool:
        """Save user feedback event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO feedback_events 
                    (event_id, timestamp, query_terms, returned_entities, 
                     selected_entities, entity_types, business_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.event_id,
                    feedback.timestamp.isoformat(),
                    json.dumps(feedback.query_terms),
                    json.dumps(feedback.returned_entities),
                    json.dumps(feedback.selected_entities),
                    json.dumps(feedback.entity_types),
                    feedback.business_context
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback event: {e}")
            return False


class PatternDiscoveryEngine:
    """Discovers new patterns from data and usage"""
    
    def __init__(self, storage: AdaptiveLearningStorage):
        self.storage = storage
        self.min_confidence_threshold = 0.6
        self.min_usage_threshold = 3
    
    async def discover_synonym_patterns(self, entity_corpus: List[Dict[str, Any]]) -> List[LearnedPattern]:
        """Discover new synonym patterns from entity corpus"""
        discovered_patterns = []
        
        # Analyze co-occurrence of terms
        term_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for entity in entity_corpus:
            entity_text = json.dumps(entity, default=str).lower()
            words = set(entity_text.split())
            
            # Find words that appear together frequently
            for word1 in words:
                for word2 in words:
                    if word1 != word2 and len(word1) > 3 and len(word2) > 3:
                        term_cooccurrence[word1][word2] += 1
        
        # Identify potential synonyms based on co-occurrence
        for term1, cooccurrences in term_cooccurrence.items():
            for term2, count in cooccurrences.items():
                if count >= self.min_usage_threshold:
                    # Calculate confidence based on co-occurrence frequency
                    confidence = min(count / 10.0, 0.95)  # Max 0.95 confidence
                    
                    if confidence >= self.min_confidence_threshold:
                        pattern = LearnedPattern(
                            pattern_type="synonym",
                            source_term=term1,
                            target_term=term2,
                            confidence_score=confidence,
                            usage_count=count,
                            first_seen=datetime.now(),
                            last_updated=datetime.now(),
                            context="discovered_from_corpus"
                        )
                        discovered_patterns.append(pattern)
        
        return discovered_patterns
    
    async def analyze_search_patterns(self) -> List[LearnedPattern]:
        """Analyze search patterns to discover new terminology relationships"""
        patterns = []
        
        # Get recent feedback events
        cutoff_date = datetime.now() - timedelta(days=30)
        
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                cursor = conn.execute("""
                    SELECT query_terms, selected_entities, business_context 
                    FROM feedback_events 
                    WHERE timestamp > ?
                """, (cutoff_date.isoformat(),))
                
                feedback_events = cursor.fetchall()
                
            # Analyze patterns in successful searches
            successful_term_pairs = defaultdict(int)
            
            for query_terms_json, selected_entities_json, context in feedback_events:
                query_terms = json.loads(query_terms_json or "[]")
                selected_entities = json.loads(selected_entities_json or "[]")
                
                if selected_entities:  # Successful search
                    for term in query_terms:
                        for entity_id in selected_entities:
                            # This indicates the term was useful for finding this entity
                            successful_term_pairs[f"{term}|{context}"] += 1
            
            # Convert successful patterns to learned patterns
            for term_context, usage_count in successful_term_pairs.items():
                if usage_count >= self.min_usage_threshold:
                    term, context = term_context.split("|", 1)
                    confidence = min(usage_count / 20.0, 0.9)
                    
                    if confidence >= self.min_confidence_threshold:
                        pattern = LearnedPattern(
                            pattern_type="effective_search_term",
                            source_term=term,
                            target_term=context,
                            confidence_score=confidence,
                            usage_count=usage_count,
                            first_seen=datetime.now(),
                            last_updated=datetime.now(),
                            context=context
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error analyzing search patterns: {e}")
        
        return patterns
    
    async def discover_regulatory_patterns(self, entity_corpus: List[Dict[str, Any]]) -> List[LearnedPattern]:
        """Discover new regulatory classification patterns"""
        patterns = []
        
        # Known regulatory indicators
        regulatory_indicators = {
            'sox': ['internal_control', 'audit', 'certification', 'disclosure'],
            'mifid': ['investor_protection', 'best_execution', 'transaction_reporting'],
            'basel': ['capital_adequacy', 'risk_weighted', 'tier_1', 'leverage_ratio'],
            'ifrs': ['fair_value', 'impairment', 'expected_credit_loss', 'hedge_accounting']
        }
        
        # Analyze entities for new regulatory patterns
        for entity in entity_corpus:
            entity_text = json.dumps(entity, default=str).lower()
            
            # Check for regulatory framework indicators
            for framework, indicators in regulatory_indicators.items():
                indicator_matches = sum(1 for indicator in indicators if indicator in entity_text)
                
                if indicator_matches >= 2:  # Strong regulatory signal
                    # Extract potential new terms associated with this framework
                    words = entity_text.split()
                    for word in words:
                        if (len(word) > 4 and 
                            word not in indicators and 
                            any(ind in entity_text for ind in indicators)):
                            
                            pattern = LearnedPattern(
                                pattern_type="regulatory_mapping",
                                source_term=word,
                                target_term=framework,
                                confidence_score=0.7,
                                usage_count=1,
                                first_seen=datetime.now(),
                                last_updated=datetime.now(),
                                context=f"regulatory_{framework}"
                            )
                            patterns.append(pattern)
        
        return patterns


class AdaptiveVocabularyExpander:
    """Expands vocabulary based on learned patterns"""
    
    def __init__(self, storage: AdaptiveLearningStorage):
        self.storage = storage
    
    def get_expanded_synonyms(self, term: str, context: str = "") -> List[str]:
        """Get expanded synonyms including learned ones"""
        base_synonyms = []
        
        # Get learned synonym patterns
        learned_patterns = self.storage.get_learned_patterns("synonym", context)
        
        for pattern in learned_patterns:
            if (pattern.source_term.lower() == term.lower() or 
                pattern.target_term.lower() == term.lower()):
                if pattern.confidence_score >= 0.6:
                    if pattern.source_term.lower() != term.lower():
                        base_synonyms.append(pattern.source_term)
                    if pattern.target_term.lower() != term.lower():
                        base_synonyms.append(pattern.target_term)
        
        return list(set(base_synonyms))
    
    def get_contextual_terms(self, entity_type: str, business_context: str) -> List[str]:
        """Get terms that work well in specific context"""
        effective_terms = []
        
        # Get effective search terms for this context
        patterns = self.storage.get_learned_patterns("effective_search_term", business_context)
        
        for pattern in patterns:
            if pattern.confidence_score >= 0.7:
                effective_terms.append(pattern.source_term)
        
        return effective_terms
    
    def update_terminology_mappings(self, base_mappings: Dict[str, List[str]], 
                                   context: str = "") -> Dict[str, List[str]]:
        """Update terminology mappings with learned patterns"""
        enhanced_mappings = base_mappings.copy()
        
        # Add learned synonyms
        learned_synonyms = self.storage.get_learned_patterns("synonym", context)
        
        for pattern in learned_synonyms:
            if pattern.confidence_score >= 0.6:
                # Find the base mapping that contains either term
                for standard_term, synonyms in enhanced_mappings.items():
                    if (pattern.source_term in synonyms or 
                        pattern.target_term in synonyms or
                        pattern.source_term == standard_term):
                        
                        # Add the new synonym
                        if pattern.source_term not in synonyms and pattern.source_term != standard_term:
                            synonyms.append(pattern.source_term)
                        if pattern.target_term not in synonyms and pattern.target_term != standard_term:
                            synonyms.append(pattern.target_term)
        
        return enhanced_mappings


class ContinuousLearner:
    """Orchestrates continuous learning across the system"""
    
    def __init__(self, storage: AdaptiveLearningStorage):
        self.storage = storage
        self.discovery_engine = PatternDiscoveryEngine(storage)
        self.vocabulary_expander = AdaptiveVocabularyExpander(storage)
        self.learning_enabled = True
        self.last_learning_cycle = datetime.now()
    
    async def process_feedback(self, feedback: FeedbackEvent) -> bool:
        """Process user feedback to update learning"""
        if not self.learning_enabled:
            return False
        
        # Save feedback event
        success = self.storage.save_feedback_event(feedback)
        if not success:
            return False
        
        # Update usage statistics
        for term in feedback.query_terms:
            self.storage.update_usage_stats(
                term, 
                searched=True, 
                selected=len(feedback.selected_entities) > 0,
                context=feedback.business_context
            )
        
        # If user selected specific entities, learn from that
        if feedback.selected_entities and feedback.returned_entities:
            selection_ratio = len(feedback.selected_entities) / len(feedback.returned_entities)
            
            # If selection ratio is high, the terms were effective
            if selection_ratio >= 0.5:
                for term in feedback.query_terms:
                    # Create or update effective term pattern
                    pattern = LearnedPattern(
                        pattern_type="effective_search_term",
                        source_term=term,
                        target_term=feedback.business_context,
                        confidence_score=selection_ratio,
                        usage_count=1,
                        first_seen=datetime.now(),
                        last_updated=datetime.now(),
                        context=feedback.business_context
                    )
                    self.storage.save_learned_pattern(pattern)
        
        return True
    
    async def run_discovery_cycle(self, entity_corpus: List[Dict[str, Any]]) -> Dict[str, int]:
        """Run a complete discovery cycle"""
        if not self.learning_enabled:
            return {"status": "learning_disabled"}
        
        results = {
            "synonyms_discovered": 0,
            "search_patterns_discovered": 0,
            "regulatory_patterns_discovered": 0
        }
        
        try:
            # Discover new synonym patterns
            synonym_patterns = await self.discovery_engine.discover_synonym_patterns(entity_corpus)
            for pattern in synonym_patterns:
                self.storage.save_learned_pattern(pattern)
            results["synonyms_discovered"] = len(synonym_patterns)
            
            # Analyze search patterns
            search_patterns = await self.discovery_engine.analyze_search_patterns()
            for pattern in search_patterns:
                self.storage.save_learned_pattern(pattern)
            results["search_patterns_discovered"] = len(search_patterns)
            
            # Discover regulatory patterns
            regulatory_patterns = await self.discovery_engine.discover_regulatory_patterns(entity_corpus)
            for pattern in regulatory_patterns:
                self.storage.save_learned_pattern(pattern)
            results["regulatory_patterns_discovered"] = len(regulatory_patterns)
            
            self.last_learning_cycle = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in discovery cycle: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        try:
            with sqlite3.connect(self.storage.db_path) as conn:
                # Count patterns by type
                cursor = conn.execute("""
                    SELECT pattern_type, COUNT(*), AVG(confidence_score) 
                    FROM learned_patterns 
                    GROUP BY pattern_type
                """)
                pattern_stats = {}
                for row in cursor.fetchall():
                    pattern_stats[row[0]] = {
                        "count": row[1],
                        "avg_confidence": round(row[2], 3)
                    }
                
                # Get usage stats
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(effectiveness_score) 
                    FROM usage_stats
                """)
                usage_row = cursor.fetchone()
                
                # Get feedback events count
                cursor = conn.execute("SELECT COUNT(*) FROM feedback_events")
                feedback_count = cursor.fetchone()[0]
                
                return {
                    "patterns_by_type": pattern_stats,
                    "total_terms_tracked": usage_row[0] if usage_row else 0,
                    "avg_term_effectiveness": round(usage_row[1] or 0, 3),
                    "feedback_events_count": feedback_count,
                    "last_learning_cycle": self.last_learning_cycle.isoformat(),
                    "learning_enabled": self.learning_enabled
                }
                
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {"error": str(e)}