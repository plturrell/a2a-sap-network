"""
MCP-enabled Semantic Similarity Calculator
Exposes text similarity calculations as MCP tools for cross-agent usage
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
from collections import Counter
import math
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...sdk.mcpSkillCoordination import skill_provides, skill_depends_on

class MCPSemanticSimilarityCalculator:
    """MCP-enabled semantic similarity calculator with exposed tools"""
    
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
        
        # Stop words for preprocessing
        self.stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "might", "must", "shall", "can", "may", "of", "to", "in",
            "for", "on", "with", "at", "by", "from", "about", "as", "and", "or",
            "but", "if", "then", "else", "when", "where", "why", "how", "all",
            "each", "every", "some", "any", "few", "more", "most", "other"
        }
    
    @mcp_tool(
        name="calculate_text_similarity",
        description="Calculate semantic similarity between two texts using multiple methods",
        input_schema={
            "type": "object",
            "properties": {
                "text1": {"type": "string", "description": "First text to compare"},
                "text2": {"type": "string", "description": "Second text to compare"},
                "method": {
                    "type": "string",
                    "enum": ["jaccard", "cosine", "semantic", "hybrid"],
                    "default": "hybrid",
                    "description": "Similarity calculation method"
                },
                "return_components": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return breakdown of similarity components"
                }
            },
            "required": ["text1", "text2"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "similarity": {"type": "number"},
                "method": {"type": "string"},
                "components": {"type": "object"},
                "common_concepts": {"type": "object"}
            }
        }
    )
    @skill_provides("text_similarity", "semantic_analysis")
    async def calculate_text_similarity_mcp(self,
                                      text1: str,
                                      text2: str,
                                      method: str = "hybrid",
                                      return_components: bool = False) -> Dict[str, Any]:
        """MCP tool for calculating text similarity"""
        
        similarity = self.calculate_similarity(text1, text2, method)
        
        result = {
            "similarity": similarity,
            "method": method
        }
        
        if return_components and method == "hybrid":
            # Calculate individual components
            result["components"] = {
                "jaccard": self._jaccard_similarity(text1, text2),
                "cosine": self._cosine_similarity(text1, text2),
                "semantic": self._semantic_similarity(text1, text2)
            }
            
            # Extract common concepts
            result["common_concepts"] = self.extract_common_concepts(text1, text2)
        
        return result
    
    @mcp_tool(
        name="find_similar_texts",
        description="Find most similar texts from a list of candidates",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query text to match"},
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate texts"
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top results to return"
                },
                "threshold": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Minimum similarity threshold"
                },
                "method": {
                    "type": "string",
                    "enum": ["jaccard", "cosine", "semantic", "hybrid"],
                    "default": "hybrid"
                }
            },
            "required": ["query", "candidates"]
        }
    )
    @skill_provides("text_search", "similarity_ranking")
    async def find_similar_texts_mcp(self,
                               query: str,
                               candidates: List[str],
                               top_k: int = 5,
                               threshold: float = 0.0,
                               method: str = "hybrid") -> Dict[str, Any]:
        """MCP tool for finding similar texts"""
        
        similarities = []
        
        for idx, candidate in enumerate(candidates):
            sim = self.calculate_similarity(query, candidate, method)
            if sim >= threshold:
                similarities.append({
                    "text": candidate,
                    "index": idx,
                    "similarity": sim
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top k
        top_results = similarities[:top_k]
        
        return {
            "query": query,
            "method": method,
            "results": top_results,
            "total_candidates": len(candidates),
            "above_threshold": len(similarities)
        }
    
    @mcp_tool(
        name="calculate_group_similarity",
        description="Calculate consensus/similarity among a group of texts",
        input_schema={
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of texts to analyze"
                },
                "return_matrix": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return full similarity matrix"
                }
            },
            "required": ["texts"]
        }
    )
    @skill_provides("group_analysis", "consensus_calculation")
    async def calculate_group_similarity_mcp(self,
                                       texts: List[str],
                                       return_matrix: bool = False) -> Dict[str, Any]:
        """MCP tool for group similarity analysis"""
        
        if len(texts) < 2:
            return {
                "average_similarity": 1.0,
                "text_count": len(texts),
                "message": "Need at least 2 texts for comparison"
            }
        
        similarities = []
        similarity_matrix = []
        
        for i in range(len(texts)):
            row = []
            for j in range(len(texts)):
                if i == j:
                    sim = 1.0
                elif j < i:
                    # Use previously calculated value (symmetric matrix)
                    sim = similarity_matrix[j][i]
                else:
                    sim = self.calculate_similarity(texts[i], texts[j])
                    similarities.append(sim)
                row.append(sim)
            similarity_matrix.append(row)
        
        average_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        result = {
            "average_similarity": average_similarity,
            "text_count": len(texts),
            "comparison_count": len(similarities),
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
            "std_deviation": np.std(similarities) if similarities else 0.0
        }
        
        if return_matrix:
            result["similarity_matrix"] = similarity_matrix
        
        return result
    
    @mcp_tool(
        name="preprocess_text",
        description="Preprocess text for similarity calculations",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "remove_stop_words": {"type": "boolean", "default": True},
                "min_word_length": {"type": "integer", "default": 2}
            },
            "required": ["text"]
        }
    )
    async def preprocess_text_mcp(self,
                            text: str,
                            remove_stop_words: bool = True,
                            min_word_length: int = 2) -> Dict[str, Any]:
        """MCP tool for text preprocessing"""
        
        # Original text stats
        original_words = text.split()
        
        # Preprocess
        words = self._preprocess_text_custom(text, remove_stop_words, min_word_length)
        
        return {
            "original_text": text,
            "processed_words": words,
            "original_word_count": len(original_words),
            "processed_word_count": len(words),
            "removed_count": len(original_words) - len(words),
            "unique_words": list(set(words))
        }
    
    @mcp_tool(
        name="extract_semantic_features",
        description="Extract semantic features and concepts from text",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "include_categories": {"type": "boolean", "default": True},
                "include_synonyms": {"type": "boolean", "default": True}
            },
            "required": ["text"]
        }
    )
    @skill_provides("feature_extraction", "semantic_analysis")
    async def extract_semantic_features_mcp(self,
                                      text: str,
                                      include_categories: bool = True,
                                      include_synonyms: bool = True) -> Dict[str, Any]:
        """MCP tool for semantic feature extraction"""
        
        words = set(self._preprocess_text(text))
        
        features = {
            "word_count": len(words),
            "unique_words": list(words)
        }
        
        if include_categories:
            category_features = {}
            for category, category_words in self.semantic_categories.items():
                matching_words = words.intersection(set(category_words))
                if matching_words:
                    category_features[category] = list(matching_words)
            features["semantic_categories"] = category_features
        
        if include_synonyms:
            synonym_features = []
            for word in words:
                for group_idx, synonym_group in enumerate(self.synonym_groups):
                    if word in synonym_group:
                        synonym_features.append({
                            "word": word,
                            "synonym_group": group_idx,
                            "synonyms": list(synonym_group - {word})
                        })
            features["synonym_matches"] = synonym_features
        
        return features
    
    @mcp_resource(
        uri="similarity://calculator/config",
        name="Similarity Calculator Configuration",
        description="Current configuration and available methods",
        mime_type="application/json"
    )
    async def get_calculator_config(self) -> Dict[str, Any]:
        """Get calculator configuration"""
        return {
            "available_methods": ["jaccard", "cosine", "semantic", "hybrid"],
            "semantic_categories": list(self.semantic_categories.keys()),
            "synonym_group_count": len(self.synonym_groups),
            "stop_word_count": len(self.stop_words),
            "hybrid_weights": {
                "jaccard": 0.3,
                "cosine": 0.3,
                "semantic": 0.4
            }
        }
    
    @mcp_prompt(
        name="similarity_analysis",
        description="Interactive similarity analysis assistant",
        arguments=[
            {
                "name": "texts",
                "description": "List of texts to analyze",
                "required": True
            },
            {
                "name": "analysis_type",
                "description": "Type of analysis (pairwise, group, semantic)",
                "required": False
            }
        ]
    )
    async def similarity_analysis_prompt(self, 
                                   texts: List[str],
                                   analysis_type: str = "comprehensive") -> str:
        """Interactive prompt for similarity analysis"""
        
        prompt = f"Similarity Analysis for {len(texts)} texts:\n\n"
        
        if analysis_type in ["pairwise", "comprehensive"]:
            prompt += "Pairwise Similarities:\n"
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    sim = self.calculate_similarity(texts[i], texts[j])
                    prompt += f"Text {i+1} vs Text {j+1}: {sim:.2%}\n"
        
        if analysis_type in ["group", "comprehensive"]:
            group_sim = await self.calculate_group_similarity_mcp(texts)
            prompt += f"\nGroup Consensus: {group_sim['average_similarity']:.2%}\n"
            prompt += f"Min: {group_sim['min_similarity']:.2%}, Max: {group_sim['max_similarity']:.2%}\n"
        
        if analysis_type in ["semantic", "comprehensive"]:
            prompt += "\nSemantic Features:\n"
            for idx, text in enumerate(texts[:3]):  # Limit to first 3 for brevity
                features = await self.extract_semantic_features_mcp(text)
                prompt += f"Text {idx+1}: {features['word_count']} unique words\n"
                if features.get('semantic_categories'):
                    prompt += f"  Categories: {', '.join(features['semantic_categories'].keys())}\n"
        
        return prompt
    
    # Internal calculation methods (keep existing implementations)
    def calculate_similarity(self, text1: str, text2: str, method: str = "hybrid") -> float:
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
        return self._preprocess_text_custom(text, True, 2)
    
    def _preprocess_text_custom(self, text: str, remove_stop_words: bool, min_length: int) -> List[str]:
        """Custom preprocessing with parameters"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep word boundaries
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter based on parameters
        if remove_stop_words:
            words = [w for w in words if w not in self.stop_words and len(w) > min_length]
        else:
            words = [w for w in words if len(w) > min_length]
        
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
mcp_similarity_calculator = MCPSemanticSimilarityCalculator()