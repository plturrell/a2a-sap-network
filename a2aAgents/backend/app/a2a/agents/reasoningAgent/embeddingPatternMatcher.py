"""
Embedding-Based Pattern Matcher
Uses embeddings for semantic similarity instead of keyword matching
"""
import random

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedPattern:
    """Pattern with embedding representation"""
    pattern_id: str
    pattern_text: str
    pattern_type: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class EmbeddingPatternMatcher:
    """
    Advanced pattern matching using embeddings for semantic similarity
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.pattern_library: Dict[str, EmbeddedPattern] = {}
        self.pattern_cache: Dict[str, np.ndarray] = {}

        # Initialize pattern types with embeddings
        self._initialize_semantic_patterns()

    def _initialize_semantic_patterns(self):
        """Initialize semantic patterns with pre-computed embeddings"""
        # These would normally come from a real embedding model
        # For now, using synthetic embeddings that capture semantic relationships

        semantic_patterns = {
            # Question patterns
            "causal_question": {
                "examples": ["why does", "what causes", "how does X lead to Y"],
                "embedding": self._generate_synthetic_embedding("causal_reasoning")
            },
            "definitional_question": {
                "examples": ["what is", "define", "meaning of"],
                "embedding": self._generate_synthetic_embedding("definition_seeking")
            },
            "procedural_question": {
                "examples": ["how to", "steps to", "process of"],
                "embedding": self._generate_synthetic_embedding("procedure_explanation")
            },
            "comparative_question": {
                "examples": ["difference between", "compare", "versus"],
                "embedding": self._generate_synthetic_embedding("comparison_analysis")
            },

            # Domain patterns
            "technical_domain": {
                "examples": ["algorithm", "system", "architecture", "implementation"],
                "embedding": self._generate_synthetic_embedding("technical_content")
            },
            "scientific_domain": {
                "examples": ["hypothesis", "experiment", "theory", "evidence"],
                "embedding": self._generate_synthetic_embedding("scientific_method")
            },
            "business_domain": {
                "examples": ["strategy", "market", "revenue", "customer"],
                "embedding": self._generate_synthetic_embedding("business_analysis")
            }
        }

        # Store patterns
        for pattern_type, pattern_data in semantic_patterns.items():
            pattern = EmbeddedPattern(
                pattern_id=hashlib.md5(pattern_type.encode()).hexdigest()[:8],
                pattern_text=" ".join(pattern_data["examples"]),
                pattern_type=pattern_type,
                embedding=pattern_data["embedding"],
                metadata={"examples": pattern_data["examples"]}
            )
            self.pattern_library[pattern_type] = pattern

    def _generate_synthetic_embedding(self, concept: str) -> np.ndarray:
        """Generate synthetic embedding for testing (replace with real model)"""
        # Create deterministic embedding based on concept
        np.random.seed(hash(concept) % 2**32)
        base_embedding = np.random.randn(self.embedding_dim)

        # Add concept-specific signal
        if "causal" in concept:
            base_embedding[:100] += 0.5
        elif "definition" in concept:
            base_embedding[100:200] += 0.5
        elif "procedure" in concept:
            base_embedding[200:300] += 0.5
        elif "comparison" in concept:
            base_embedding[300:400] += 0.5
        elif "technical" in concept:
            base_embedding[400:500] += 0.5
        elif "scientific" in concept:
            base_embedding[500:600] += 0.5
        elif "business" in concept:
            base_embedding[600:700] += 0.5

        # Normalize
        return base_embedding / np.linalg.norm(base_embedding)

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text (would use real model in production)"""
        # Check cache
        if text in self.pattern_cache:
            return self.pattern_cache[text]

        # In production, this would call a real embedding model
        # For now, create synthetic embedding based on text features
        embedding = self._create_text_embedding(text)

        # Cache result
        self.pattern_cache[text] = embedding

        return embedding

    def _create_text_embedding(self, text: str) -> np.ndarray:
        """Create synthetic embedding based on text features"""
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)

        # Add signals based on keywords (simplified)
        keyword_signals = {
            "why": (0, 100),
            "what": (100, 200),
            "how": (200, 300),
            "when": (300, 400),
            "where": (400, 500),
            "algorithm": (500, 600),
            "system": (550, 650),
            "business": (600, 700),
            "scientific": (650, 750)
        }

        for word in words:
            for keyword, (start, end) in keyword_signals.items():
                if keyword in word:
                    embedding[start:end] += 0.3

        # Add some noise for realism
        embedding += np.random.randn(self.embedding_dim) * 0.1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def find_similar_patterns(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[str, float, EmbeddedPattern]]:
        """Find patterns similar to input text using embeddings"""
        # Get text embedding
        text_embedding = await self.get_text_embedding(text)

        # Calculate similarities
        similarities = []
        for pattern_type, pattern in self.pattern_library.items():
            similarity = self.cosine_similarity(text_embedding, pattern.embedding)
            if similarity >= threshold:
                similarities.append((pattern_type, similarity, pattern))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def analyze_semantic_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for semantic patterns using embeddings"""
        # Get text embedding
        text_embedding = await self.get_text_embedding(text)

        # Find similar patterns
        similar_patterns = await self.find_similar_patterns(text)

        # Analyze domain
        domain_scores = {}
        for pattern_type, similarity, pattern in similar_patterns:
            if "domain" in pattern_type:
                domain = pattern_type.replace("_domain", "")
                domain_scores[domain] = similarity

        # Determine primary domain
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "general"

        # Analyze question type
        question_types = {}
        for pattern_type, similarity, pattern in similar_patterns:
            if "question" in pattern_type:
                q_type = pattern_type.replace("_question", "")
                question_types[q_type] = similarity

        # Extract semantic features
        features = {
            "has_causal_reasoning": any("causal" in p[0] for p in similar_patterns),
            "has_comparison": any("compar" in p[0] for p in similar_patterns),
            "has_procedural": any("procedur" in p[0] for p in similar_patterns),
            "is_definitional": any("definition" in p[0] for p in similar_patterns)
        }

        return {
            "text": text,
            "embedding_norm": float(np.linalg.norm(text_embedding)),
            "primary_domain": primary_domain,
            "domain_scores": domain_scores,
            "question_types": question_types,
            "semantic_features": features,
            "similar_patterns": [
                {
                    "type": p[0],
                    "similarity": float(p[1]),
                    "examples": p[2].metadata.get("examples", [])
                }
                for p in similar_patterns
            ],
            "confidence": float(similar_patterns[0][1]) if similar_patterns else 0.0
        }

    async def compute_pattern_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        embedding1 = await self.get_text_embedding(text1)
        embedding2 = await self.get_text_embedding(text2)

        return self.cosine_similarity(embedding1, embedding2)

    def cluster_patterns(self, texts: List[str], num_clusters: int = 5) -> Dict[int, List[str]]:
        """Cluster texts by semantic similarity (simplified k-means)"""
        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            # Synchronous version for simplicity
            embedding = self._create_text_embedding(text)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Simple k-means clustering
        clusters = defaultdict(list)

        # Initialize cluster centers randomly
        indices = np.secrets.choice(len(texts), min(num_clusters, len(texts)), replace=False)
        centers = embeddings[indices]

        # Assign texts to nearest cluster
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            similarities = [self.cosine_similarity(embedding, center) for center in centers]
            cluster_id = np.argmax(similarities)
            clusters[cluster_id].append(text)

        return dict(clusters)


# Integration with existing NLP pattern matcher
class EnhancedNLPPatternMatcher:
    """Enhanced pattern matcher that combines keyword and embedding approaches"""

    def __init__(self):
        self.embedding_matcher = EmbeddingPatternMatcher()
        # Can still use keyword patterns as fallback
        self.keyword_patterns = {
            "causal": ["why", "because", "cause", "effect", "lead to", "result in"],
            "comparative": ["compare", "versus", "different", "similar", "contrast"],
            "procedural": ["how to", "steps", "process", "method", "approach"]
        }

    async def analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze patterns using both embeddings and keywords"""
        # Get embedding-based analysis
        embedding_analysis = await self.embedding_matcher.analyze_semantic_patterns(text)

        # Fallback keyword analysis
        keyword_matches = []
        for pattern_type, keywords in self.keyword_patterns.items():
            if any(keyword in text.lower() for keyword in keywords):
                keyword_matches.append(pattern_type)

        # Combine results
        return {
            "semantic_analysis": embedding_analysis,
            "keyword_matches": keyword_matches,
            "combined_confidence": max(
                embedding_analysis["confidence"],
                0.6 if keyword_matches else 0.0
            ),
            "recommended_approach": self._recommend_approach(embedding_analysis, keyword_matches)
        }

    def _recommend_approach(self, semantic: Dict[str, Any], keywords: List[str]) -> str:
        """Recommend reasoning approach based on analysis"""
        # Check semantic features
        features = semantic.get("semantic_features", {})

        if features.get("has_causal_reasoning") or "causal" in keywords:
            return "causal_chain_reasoning"
        elif features.get("has_comparison") or "comparative" in keywords:
            return "comparative_analysis"
        elif features.get("has_procedural") or "procedural" in keywords:
            return "step_by_step_reasoning"
        elif features.get("is_definitional"):
            return "definitional_analysis"
        else:
            return "general_reasoning"


# Factory function
def create_embedding_pattern_matcher() -> EmbeddingPatternMatcher:
    """Create an embedding-based pattern matcher"""
    return EmbeddingPatternMatcher()