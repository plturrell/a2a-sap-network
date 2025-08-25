"""
Async Reasoning Memory System
Replaces SQLite blocking operations with async storage for performance improvements
"""

import asyncio
import aiosqlite
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ReasoningExperience:
    """Represents a reasoning experience for learning"""
    question: str
    answer: str
    reasoning_chain: List[Dict[str, Any]]
    confidence: float
    context: Dict[str, Any]
    timestamp: datetime
    architecture_used: str
    performance_metrics: Dict[str, Any]
    experience_id: Optional[str] = None

    def __post_init__(self):
        if not self.experience_id:
            # Generate unique ID based on question and timestamp
            content = f"{self.question}_{self.timestamp.isoformat()}"
            self.experience_id = hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class MemoryPattern:
    """Represents a learned reasoning pattern"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    success_rate: float
    usage_count: int
    last_used: datetime
    contexts: List[str]
    pattern_id: Optional[str] = None

    def __post_init__(self):
        if not self.pattern_id:
            content = f"{self.pattern_type}_{self.pattern_data}"
            self.pattern_id = hashlib.md5(str(content).encode()).hexdigest()[:16]


class AsyncReasoningMemoryStore:
    """Async persistent storage for reasoning experiences and patterns"""

    def __init__(self, db_path: str = "reasoning_memory_async.db"):
        self.db_path = db_path
        self.connection_pool_size = 5
        self.connection_timeout = 30.0
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the async database with proper schema"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                async with aiosqlite.connect(self.db_path) as db:
                    # Enable WAL mode for better concurrency
                    await db.execute("PRAGMA journal_mode = WAL")
                    await db.execute("PRAGMA synchronous = NORMAL")
                    await db.execute("PRAGMA cache_size = 10000")
                    await db.execute("PRAGMA temp_store = MEMORY")

                    # Create experiences table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS reasoning_experiences (
                            experience_id TEXT PRIMARY KEY,
                            question TEXT NOT NULL,
                            answer TEXT NOT NULL,
                            reasoning_chain TEXT NOT NULL,
                            confidence REAL NOT NULL,
                            context TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            architecture_used TEXT NOT NULL,
                            performance_metrics TEXT NOT NULL,
                            question_hash TEXT NOT NULL,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create patterns table
                    await db.execute("""
                        CREATE TABLE IF NOT EXISTS reasoning_patterns (
                            pattern_id TEXT PRIMARY KEY,
                            pattern_type TEXT NOT NULL,
                            pattern_data TEXT NOT NULL,
                            success_rate REAL NOT NULL,
                            usage_count INTEGER NOT NULL,
                            last_used TEXT NOT NULL,
                            contexts TEXT NOT NULL,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create indexes for better performance
                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_experiences_question_hash
                        ON reasoning_experiences(question_hash)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_experiences_timestamp
                        ON reasoning_experiences(timestamp)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_experiences_architecture
                        ON reasoning_experiences(architecture_used)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_patterns_type
                        ON reasoning_patterns(pattern_type)
                    """)

                    await db.execute("""
                        CREATE INDEX IF NOT EXISTS idx_patterns_success_rate
                        ON reasoning_patterns(success_rate)
                    """)

                    await db.commit()

                self._initialized = True
                logger.info("Async reasoning memory store initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize async memory store: {e}")
                raise

    async def store_experience(self, experience: ReasoningExperience) -> bool:
        """Store a reasoning experience asynchronously"""
        try:
            await self.initialize()

            # Create question hash for efficient lookups
            question_hash = hashlib.md5(experience.question.encode()).hexdigest()

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO reasoning_experiences (
                        experience_id, question, answer, reasoning_chain, confidence,
                        context, timestamp, architecture_used, performance_metrics,
                        question_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience.experience_id,
                    experience.question,
                    experience.answer,
                    json.dumps(experience.reasoning_chain),
                    experience.confidence,
                    json.dumps(experience.context),
                    experience.timestamp.isoformat(),
                    experience.architecture_used,
                    json.dumps(experience.performance_metrics),
                    question_hash
                ))

                await db.commit()

            logger.debug(f"Stored reasoning experience: {experience.experience_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return False

    async def retrieve_similar_experiences(self,
                                         question: str,
                                         limit: int = 10,
                                         architecture_filter: Optional[str] = None,
                                         min_confidence: float = 0.0) -> List[ReasoningExperience]:
        """Retrieve similar reasoning experiences asynchronously"""
        try:
            await self.initialize()

            question_hash = hashlib.md5(question.encode()).hexdigest()

            # Build query with optional filters
            query = """
                SELECT experience_id, question, answer, reasoning_chain, confidence,
                       context, timestamp, architecture_used, performance_metrics
                FROM reasoning_experiences
                WHERE confidence >= ?
            """
            params = [min_confidence]

            if architecture_filter:
                query += " AND architecture_used = ?"
                params.append(architecture_filter)

            # Order by similarity (exact match first, then by confidence and recency)
            query += """
                ORDER BY
                    CASE WHEN question_hash = ? THEN 0 ELSE 1 END,
                    confidence DESC,
                    timestamp DESC
                LIMIT ?
            """
            params.extend([question_hash, limit])

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

            experiences = []
            for row in rows:
                experience = ReasoningExperience(
                    experience_id=row[0],
                    question=row[1],
                    answer=row[2],
                    reasoning_chain=json.loads(row[3]),
                    confidence=row[4],
                    context=json.loads(row[5]),
                    timestamp=datetime.fromisoformat(row[6]),
                    architecture_used=row[7],
                    performance_metrics=json.loads(row[8])
                )
                experiences.append(experience)

            logger.debug(f"Retrieved {len(experiences)} similar experiences")
            return experiences

        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []

    async def store_pattern(self, pattern: MemoryPattern) -> bool:
        """Store a reasoning pattern asynchronously"""
        try:
            await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                # Check if pattern exists and update, otherwise insert
                async with db.execute(
                    "SELECT usage_count FROM reasoning_patterns WHERE pattern_id = ?",
                    (pattern.pattern_id,)
                ) as cursor:
                    existing = await cursor.fetchone()

                if existing:
                    # Update existing pattern
                    new_usage_count = existing[0] + pattern.usage_count
                    await db.execute("""
                        UPDATE reasoning_patterns
                        SET pattern_data = ?, success_rate = ?, usage_count = ?,
                            last_used = ?, contexts = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE pattern_id = ?
                    """, (
                        json.dumps(pattern.pattern_data),
                        pattern.success_rate,
                        new_usage_count,
                        pattern.last_used.isoformat(),
                        json.dumps(pattern.contexts),
                        pattern.pattern_id
                    ))
                else:
                    # Insert new pattern
                    await db.execute("""
                        INSERT INTO reasoning_patterns (
                            pattern_id, pattern_type, pattern_data, success_rate,
                            usage_count, last_used, contexts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        json.dumps(pattern.pattern_data),
                        pattern.success_rate,
                        pattern.usage_count,
                        pattern.last_used.isoformat(),
                        json.dumps(pattern.contexts)
                    ))

                await db.commit()

            logger.debug(f"Stored reasoning pattern: {pattern.pattern_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False

    async def get_successful_patterns(self,
                                    pattern_type: Optional[str] = None,
                                    min_success_rate: float = 0.7,
                                    limit: int = 20) -> List[MemoryPattern]:
        """Get successful reasoning patterns asynchronously"""
        try:
            await self.initialize()

            query = """
                SELECT pattern_id, pattern_type, pattern_data, success_rate,
                       usage_count, last_used, contexts
                FROM reasoning_patterns
                WHERE success_rate >= ?
            """
            params = [min_success_rate]

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            query += " ORDER BY success_rate DESC, usage_count DESC LIMIT ?"
            params.append(limit)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

            patterns = []
            for row in rows:
                pattern = MemoryPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    pattern_data=json.loads(row[2]),
                    success_rate=row[3],
                    usage_count=row[4],
                    last_used=datetime.fromisoformat(row[5]),
                    contexts=json.loads(row[6])
                )
                patterns.append(pattern)

            logger.debug(f"Retrieved {len(patterns)} successful patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to get patterns: {e}")
            return []

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics asynchronously"""
        try:
            await self.initialize()

            async with aiosqlite.connect(self.db_path) as db:
                # Get experience statistics
                async with db.execute("""
                    SELECT
                        COUNT(*) as total_experiences,
                        AVG(confidence) as avg_confidence,
                        MAX(confidence) as max_confidence,
                        COUNT(DISTINCT architecture_used) as architectures_used
                    FROM reasoning_experiences
                """) as cursor:
                    exp_stats = await cursor.fetchone()

                # Get pattern statistics
                async with db.execute("""
                    SELECT
                        COUNT(*) as total_patterns,
                        AVG(success_rate) as avg_success_rate,
                        SUM(usage_count) as total_usage
                    FROM reasoning_patterns
                """) as cursor:
                    pattern_stats = await cursor.fetchone()

                # Get architecture performance
                async with db.execute("""
                    SELECT
                        architecture_used,
                        COUNT(*) as usage_count,
                        AVG(confidence) as avg_confidence
                    FROM reasoning_experiences
                    GROUP BY architecture_used
                    ORDER BY avg_confidence DESC
                """) as cursor:
                    arch_stats = await cursor.fetchall()

            return {
                "experiences": {
                    "total": exp_stats[0] if exp_stats else 0,
                    "avg_confidence": exp_stats[1] if exp_stats else 0.0,
                    "max_confidence": exp_stats[2] if exp_stats else 0.0,
                    "architectures_used": exp_stats[3] if exp_stats else 0
                },
                "patterns": {
                    "total": pattern_stats[0] if pattern_stats else 0,
                    "avg_success_rate": pattern_stats[1] if pattern_stats else 0.0,
                    "total_usage": pattern_stats[2] if pattern_stats else 0
                },
                "architecture_performance": [
                    {
                        "architecture": row[0],
                        "usage_count": row[1],
                        "avg_confidence": row[2]
                    }
                    for row in arch_stats
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}

    async def cleanup_old_experiences(self, days_to_keep: int = 30) -> int:
        """Clean up old experiences to manage storage asynchronously"""
        try:
            await self.initialize()

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    DELETE FROM reasoning_experiences
                    WHERE timestamp < ? AND confidence < 0.7
                """, (cutoff_date.isoformat(),)) as cursor:
                    deleted_count = cursor.rowcount

                await db.commit()

            logger.info(f"Cleaned up {deleted_count} old reasoning experiences")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup experiences: {e}")
            return 0

    async def close(self):
        """Close the async memory store gracefully"""
        # aiosqlite connections are automatically closed when context exits
        # No persistent connections to close
        logger.info("Async reasoning memory store closed")


class AsyncAdaptiveReasoningSystem:
    """Async adaptive reasoning system with learning capabilities"""

    def __init__(self, memory_store: AsyncReasoningMemoryStore):
        self.memory_store = memory_store
        self.learning_rate = 0.1
        self.pattern_threshold = 0.8
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}

    async def learn_from_experience(self, experience: ReasoningExperience) -> List[MemoryPattern]:
        """Learn patterns from reasoning experience asynchronously"""
        try:
            # Store the experience
            await self.memory_store.store_experience(experience)

            # Extract patterns from the experience
            patterns = await self._extract_patterns_async(experience)

            # Store successful patterns
            stored_patterns = []
            for pattern in patterns:
                if pattern.success_rate >= self.pattern_threshold:
                    success = await self.memory_store.store_pattern(pattern)
                    if success:
                        stored_patterns.append(pattern)

            logger.debug(f"Learned {len(stored_patterns)} patterns from experience")
            return stored_patterns

        except Exception as e:
            logger.error(f"Failed to learn from experience: {e}")
            return []

    async def get_reasoning_suggestions(self, question: str, architecture: str) -> Dict[str, Any]:
        """Get reasoning suggestions based on past experiences asynchronously"""
        try:
            # Check cache first
            cache_key = f"{question}_{architecture}"
            if self._is_cached(cache_key):
                return self.cache[cache_key]

            # Get similar experiences
            experiences = await self.memory_store.retrieve_similar_experiences(
                question, limit=5, architecture_filter=architecture
            )

            # Get relevant patterns
            patterns = await self.memory_store.get_successful_patterns(
                pattern_type=f"{architecture}_reasoning"
            )

            suggestions = {
                "similar_cases": len(experiences),
                "confidence_range": {
                    "min": min((e.confidence for e in experiences), default=0.0),
                    "max": max((e.confidence for e in experiences), default=0.0),
                    "avg": sum(e.confidence for e in experiences) / len(experiences) if experiences else 0.0
                },
                "recommended_patterns": [
                    {
                        "type": p.pattern_type,
                        "success_rate": p.success_rate,
                        "usage_count": p.usage_count
                    }
                    for p in patterns[:3]
                ],
                "architecture_performance": await self._get_architecture_performance_async(architecture)
            }

            # Cache the result
            self._cache_result(cache_key, suggestions)

            return suggestions

        except Exception as e:
            logger.error(f"Failed to get reasoning suggestions: {e}")
            return {}

    async def _extract_patterns_async(self, experience: ReasoningExperience) -> List[MemoryPattern]:
        """Extract patterns from experience asynchronously"""
        patterns = []

        try:
            # Question type pattern
            question_words = experience.question.lower().split()
            if any(word in question_words for word in ["what", "how", "why", "when", "where"]):
                question_type = next((word for word in ["what", "how", "why", "when", "where"]
                                    if word in question_words), "unknown")

                patterns.append(MemoryPattern(
                    pattern_type=f"{experience.architecture_used}_question_type",
                    pattern_data={
                        "question_type": question_type,
                        "avg_confidence": experience.confidence,
                        "reasoning_steps": len(experience.reasoning_chain)
                    },
                    success_rate=experience.confidence,
                    usage_count=1,
                    last_used=experience.timestamp,
                    contexts=[experience.context.get("domain", "general")]
                ))

            # Performance pattern
            if experience.performance_metrics:
                patterns.append(MemoryPattern(
                    pattern_type=f"{experience.architecture_used}_performance",
                    pattern_data=experience.performance_metrics,
                    success_rate=experience.confidence,
                    usage_count=1,
                    last_used=experience.timestamp,
                    contexts=[experience.context.get("domain", "general")]
                ))

            return patterns

        except Exception as e:
            logger.error(f"Failed to extract patterns: {e}")
            return []

    async def _get_architecture_performance_async(self, architecture: str) -> Dict[str, Any]:
        """Get architecture performance metrics asynchronously"""
        try:
            experiences = await self.memory_store.retrieve_similar_experiences(
                "", limit=100, architecture_filter=architecture
            )

            if not experiences:
                return {"usage_count": 0, "avg_confidence": 0.0, "success_rate": 0.0}

            return {
                "usage_count": len(experiences),
                "avg_confidence": sum(e.confidence for e in experiences) / len(experiences),
                "success_rate": sum(1 for e in experiences if e.confidence > 0.7) / len(experiences)
            }

        except Exception as e:
            logger.error(f"Failed to get architecture performance: {e}")
            return {}

    def _is_cached(self, cache_key: str) -> bool:
        """Check if result is cached and still valid"""
        if cache_key not in self.cache:
            return False

        timestamp = self.cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.cache_ttl

    def _cache_result(self, cache_key: str, result: Any):
        """Cache a result with timestamp"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()

        # Simple cache cleanup - remove oldest if too many
        if len(self.cache) > 100:
            oldest_key = min(self.cache_timestamps.keys(),
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics asynchronously"""
        stats = await self.memory_store.get_performance_stats()
        stats["cache_size"] = len(self.cache)
        stats["cache_hit_rate"] = getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        return stats

    async def cleanup(self):
        """Cleanup the adaptive reasoning system"""
        self.cache.clear()
        self.cache_timestamps.clear()
        await self.memory_store.close()


# Example usage and testing
async def test_async_memory_system():
    """Test the async memory system"""
    try:
        # Initialize async memory store
        memory_store = AsyncReasoningMemoryStore("test_async_memory.db")

        # Create test experience
        experience = ReasoningExperience(
            question="What causes global warming?",
            answer="Greenhouse gas emissions from human activities",
            reasoning_chain=[
                {"step": 1, "type": "decomposition", "content": "Identify greenhouse gases"},
                {"step": 2, "type": "analysis", "content": "Analyze emission sources"},
                {"step": 3, "type": "synthesis", "content": "Connect to warming effect"}
            ],
            confidence=0.85,
            context={"domain": "climate_science"},
            timestamp=datetime.utcnow(),
            architecture_used="blackboard",
            performance_metrics={"duration": 2.5, "api_calls": 3}
        )

        # Test storing experience
        success = await memory_store.store_experience(experience)
        print(f"✅ Store experience: {success}")

        # Test retrieving experiences
        similar = await memory_store.retrieve_similar_experiences("climate change")
        print(f"✅ Retrieved {len(similar)} similar experiences")

        # Test adaptive system
        adaptive_system = AsyncAdaptiveReasoningSystem(memory_store)
        patterns = await adaptive_system.learn_from_experience(experience)
        print(f"✅ Learned {len(patterns)} patterns")

        # Test suggestions
        suggestions = await adaptive_system.get_reasoning_suggestions(
            "How does climate change work?", "blackboard"
        )
        print(f"✅ Got reasoning suggestions: {bool(suggestions)}")

        # Test performance stats
        stats = await memory_store.get_performance_stats()
        print(f"✅ Performance stats: {stats}")

        # Cleanup
        await adaptive_system.cleanup()
        print("✅ Async memory system test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_async_memory_system())