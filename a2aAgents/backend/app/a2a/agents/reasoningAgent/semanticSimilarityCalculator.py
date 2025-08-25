"""
Semantic Similarity Calculator
Enhanced text similarity using multiple methods
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
from collections import Counter
import math

class SemanticSimilarityCalculator:
    """Calculate semantic similarity between texts using multiple methods"""

    def __init__(self):
        # Semantic word categories for enhanced matching
        self.semantic_categories = {
            "causation": ["cause", "effect", "result", "lead", "produce", "trigger", "consequence"],
            "process": ["step", "stage", "phase", "process", "procedure", "method", "approach"],
            "comparison": ["similar", "different", "like", "unlike", "same", "opposite", "compare"],
            "temporal": ["before", "after", "during", "while", "then", "now", "later", "earlier"],
            "structure": ["part", "component", "element", "system", "whole", "structure", "framework"]
        }

        # Synonyms for common words
        self.synonym_groups = [
            {"create", "make", "produce", "generate", "build", "construct"},
            {"use", "utilize", "employ", "apply", "implement"},
            {"show", "demonstrate", "display", "present", "exhibit"},
            {"understand", "comprehend", "grasp", "realize", "recognize"},
            {"improve", "enhance", "optimize", "refine", "upgrade"}
        ]

    def calculate_similarity(self, text1: str, text2: str,
                           method: str = "hybrid") -> float:
        """Calculate similarity between two texts"""

        if method == "jaccard":
            return self._jaccard_similarity(text1, text2)
        elif method == "cosine":
            return self._cosine_similarity(text1, text2)
        elif method == "semantic":
            return self._semantic_similarity(text1, text2)
        elif method == "hybrid":
            return self._hybrid_similarity(text1, text2)
        else:
            return self._jaccard_similarity(text1, text2)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for similarity calculation"""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation but keep word boundaries
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split into words
        words = text.split()

        # Remove stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "might", "must", "shall", "can", "may", "of", "to", "in",
            "for", "on", "with", "at", "by", "from", "about", "as", "and", "or",
            "but", "if", "then", "else", "when", "where", "why", "how", "all",
            "each", "every", "some", "any", "few", "more", "most", "other"
        }

        words = [w for w in words if w not in stop_words and len(w) > 2]

        return words

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity"""

        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF"""

        words1 = self._preprocess_text(text1)
        words2 = self._preprocess_text(text2)

        # Create vocabulary
        vocabulary = list(set(words1 + words2))

        if not vocabulary:
            return 0.0

        # Calculate term frequencies
        tf1 = Counter(words1)
        tf2 = Counter(words2)

        # Create vectors
        vector1 = [tf1.get(word, 0) for word in vocabulary]
        vector2 = [tf2.get(word, 0) for word in vocabulary]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using word categories and synonyms"""

        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))

        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0

        # Direct matches
        direct_matches = len(words1.intersection(words2))

        # Semantic category matches
        category_matches = 0
        for category, category_words in self.semantic_categories.items():
            words1_in_category = words1.intersection(set(category_words))
            words2_in_category = words2.intersection(set(category_words))

            if words1_in_category and words2_in_category:
                category_matches += 1

        # Synonym matches
        synonym_matches = 0
        for word1 in words1:
            for word2 in words2:
                if word1 != word2 and self._are_synonyms(word1, word2):
                    synonym_matches += 1

        # Calculate weighted similarity
        total_possible = max(len(words1), len(words2))

        similarity = (
            (direct_matches * 1.0) +
            (category_matches * 0.3) +
            (synonym_matches * 0.5)
        ) / total_possible

        return min(similarity, 1.0)

    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms"""

        for synonym_group in self.synonym_groups:
            if word1 in synonym_group and word2 in synonym_group:
                return True

        return False

    def _hybrid_similarity(self, text1: str, text2: str) -> float:
        """Calculate hybrid similarity combining multiple methods"""

        # Calculate individual similarities
        jaccard = self._jaccard_similarity(text1, text2)
        cosine = self._cosine_similarity(text1, text2)
        semantic = self._semantic_similarity(text1, text2)

        # Weighted combination
        weights = {
            "jaccard": 0.3,
            "cosine": 0.3,
            "semantic": 0.4
        }

        hybrid_score = (
            jaccard * weights["jaccard"] +
            cosine * weights["cosine"] +
            semantic * weights["semantic"]
        )

        return hybrid_score

    def calculate_group_similarity(self, texts: List[str]) -> float:
        """Calculate average pairwise similarity for a group of texts"""

        if len(texts) < 2:
            return 1.0

        similarities = []

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = self.calculate_similarity(texts[i], texts[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def find_most_similar(self, query: str, candidates: List[str],
                         top_k: int = 1) -> List[Tuple[str, float]]:
        """Find most similar texts from candidates"""

        similarities = []

        for candidate in candidates:
            sim = self.calculate_similarity(query, candidate)
            similarities.append((candidate, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def calculate_semantic_distance(self, text1: str, text2: str) -> float:
        """Calculate semantic distance (inverse of similarity)"""

        similarity = self.calculate_similarity(text1, text2)
        return 1.0 - similarity

    def extract_common_concepts(self, text1: str, text2: str) -> Dict[str, List[str]]:
        """Extract common concepts between texts"""

        words1 = set(self._preprocess_text(text1))
        words2 = set(self._preprocess_text(text2))

        common_concepts = {
            "direct_matches": list(words1.intersection(words2)),
            "category_matches": [],
            "semantic_relations": []
        }

        # Find category matches
        for category, category_words in self.semantic_categories.items():
            words1_in_category = words1.intersection(set(category_words))
            words2_in_category = words2.intersection(set(category_words))

            if words1_in_category and words2_in_category:
                common_concepts["category_matches"].append({
                    "category": category,
                    "words_text1": list(words1_in_category),
                    "words_text2": list(words2_in_category)
                })

        # Find semantic relations
        for word1 in words1:
            for word2 in words2:
                if word1 != word2 and self._are_synonyms(word1, word2):
                    common_concepts["semantic_relations"].append({
                        "word1": word1,
                        "word2": word2,
                        "relation": "synonym"
                    })

        return common_concepts


# Singleton instance
similarity_calculator = SemanticSimilarityCalculator()

# Helper functions
def calculate_text_similarity(text1: str, text2: str, method: str = "hybrid") -> float:
    """Calculate similarity between two texts"""
    return similarity_calculator.calculate_similarity(text1, text2, method)

def calculate_group_consensus(texts: List[str]) -> float:
    """Calculate consensus score for a group of texts"""
    return similarity_calculator.calculate_group_similarity(texts)

def find_similar_texts(query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Find most similar texts from candidates"""
    return similarity_calculator.find_most_similar(query, candidates, top_k)
