import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
import math
from collections import defaultdict
import logging
from ....sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ....sdk.mcpSkillCoordination import skill_provides, skill_depends_on

from app.a2a.core.security_base import SecureA2AAgent
"""
MCP-enabled Vector Similarity Calculator
Exposes vector similarity operations as MCP tools for cross-agent usage
"""

logger = logging.getLogger(__name__)


class MCPVectorSimilarityCalculator(SecureA2AAgent):
    """MCP-enabled vector similarity calculator with multiple distance metrics"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self):
        super().__init__()
        self.supported_metrics = [
            "cosine",
            "euclidean",
            "manhattan",
            "dot_product",
            "jaccard",
            "hamming"
        ]
        
        # Cache for normalized vectors
        self.normalization_cache = {}
        self.cache_size_limit = 1000
    
    @mcp_tool(
        name="calculate_vector_similarity",
        description="Calculate similarity between vectors using various metrics",
        input_schema={
            "type": "object",
            "properties": {
                "vector1": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First vector"
                },
                "vector2": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second vector"
                },
                "metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot_product", "jaccard", "hamming"],
                    "default": "cosine",
                    "description": "Similarity/distance metric to use"
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Normalize vectors before calculation (for cosine)"
                }
            },
            "required": ["vector1", "vector2"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "similarity": {"type": "number"},
                "distance": {"type": "number"},
                "metric": {"type": "string"},
                "normalized": {"type": "boolean"},
                "vector_dimensions": {"type": "integer"}
            }
        }
    )
    @skill_provides("vector_similarity", "distance_calculation")
    async def calculate_vector_similarity_mcp(self,
                                        vector1: List[float],
                                        vector2: List[float],
                                        metric: str = "cosine",
                                        normalize: bool = True) -> Dict[str, Any]:
        """MCP tool for calculating vector similarity/distance"""
        
        if len(vector1) != len(vector2):
            raise ValueError(f"Vector dimensions must match: {len(vector1)} != {len(vector2)}")
        
        if not vector1 or not vector2:
            raise ValueError("Vectors cannot be empty")
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(vector1)
        vec2 = np.array(vector2)
        
        result = {
            "metric": metric,
            "normalized": False,
            "vector_dimensions": len(vector1)
        }
        
        if metric == "cosine":
            similarity = self._cosine_similarity(vec1, vec2, normalize)
            result["similarity"] = float(similarity)
            result["distance"] = float(1.0 - similarity)
            result["normalized"] = normalize
            
        elif metric == "euclidean":
            distance = self._euclidean_distance(vec1, vec2)
            result["distance"] = float(distance)
            result["similarity"] = float(1.0 / (1.0 + distance))
            
        elif metric == "manhattan":
            distance = self._manhattan_distance(vec1, vec2)
            result["distance"] = float(distance)
            result["similarity"] = float(1.0 / (1.0 + distance))
            
        elif metric == "dot_product":
            dot_prod = self._dot_product(vec1, vec2)
            result["similarity"] = float(dot_prod)
            result["distance"] = float(-dot_prod)  # Negative for distance interpretation
            
        elif metric == "jaccard":
            similarity = self._jaccard_similarity(vec1, vec2)
            result["similarity"] = float(similarity)
            result["distance"] = float(1.0 - similarity)
            
        elif metric == "hamming":
            distance = self._hamming_distance(vec1, vec2)
            result["distance"] = float(distance)
            result["similarity"] = float(1.0 - (distance / len(vec1)))
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return result
    
    @mcp_tool(
        name="batch_vector_similarity",
        description="Calculate similarity between a query vector and multiple candidate vectors",
        input_schema={
            "type": "object",
            "properties": {
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query vector"
                },
                "candidate_vectors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "vector": {"type": "array", "items": {"type": "number"}}
                        }
                    },
                    "description": "List of candidate vectors with IDs"
                },
                "metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "manhattan", "dot_product"],
                    "default": "cosine"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Return top K most similar vectors"
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity threshold"
                }
            },
            "required": ["query_vector", "candidate_vectors"]
        }
    )
    @skill_provides("batch_similarity", "vector_search")
    async def batch_vector_similarity_mcp(self,
                                    query_vector: List[float],
                                    candidate_vectors: List[Dict[str, Any]],
                                    metric: str = "cosine",
                                    top_k: int = 10,
                                    threshold: Optional[float] = None) -> Dict[str, Any]:
        """MCP tool for batch vector similarity calculation"""
        
        if not candidate_vectors:
            return {
                "results": [],
                "query_vector_dim": len(query_vector),
                "candidates_processed": 0
            }
        
        # Calculate similarities for all candidates
        similarities = []
        
        for candidate in candidate_vectors:
            candidate_id = candidate.get("id", "unknown")
            candidate_vector = candidate.get("vector", [])
            
            if len(candidate_vector) != len(query_vector):
                logger.warning(f"Skipping candidate {candidate_id}: dimension mismatch")
                continue
            
            try:
                sim_result = await self.calculate_vector_similarity_mcp(
                    query_vector,
                    candidate_vector,
                    metric=metric
                )
                
                similarity = sim_result.get("similarity", sim_result.get("distance", 0))
                
                # Apply threshold if specified
                if threshold is None or similarity >= threshold:
                    similarities.append({
                        "id": candidate_id,
                        "similarity": similarity,
                        "distance": sim_result.get("distance"),
                        "metric": metric
                    })
                    
            except Exception as e:
                logger.error(f"Error calculating similarity for {candidate_id}: {e}")
                continue
        
        # Sort by similarity (descending) or distance (ascending)
        if metric in ["euclidean", "manhattan", "hamming"]:
            similarities.sort(key=lambda x: x.get("distance", float("inf")))
        else:
            similarities.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Take top K
        top_results = similarities[:top_k]
        
        return {
            "results": top_results,
            "query_vector_dim": len(query_vector),
            "candidates_processed": len(candidate_vectors),
            "results_returned": len(top_results),
            "metric_used": metric,
            "threshold_applied": threshold
        }
    
    @mcp_tool(
        name="vector_statistics",
        description="Calculate statistics for a vector or set of vectors",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "One or more vectors to analyze"
                },
                "calculate_pairwise": {
                    "type": "boolean",
                    "default": False,
                    "description": "Calculate pairwise similarities"
                }
            },
            "required": ["vectors"]
        }
    )
    @skill_provides("vector_analysis", "statistics")
    async def vector_statistics_mcp(self,
                              vectors: List[List[float]],
                              calculate_pairwise: bool = False) -> Dict[str, Any]:
        """MCP tool for vector statistics calculation"""
        
        if not vectors:
            return {"error": "No vectors provided"}
        
        # Convert to numpy for efficient computation
        vec_array = np.array(vectors)
        
        stats = {
            "count": len(vectors),
            "dimensions": vec_array.shape[1] if vec_array.ndim > 1 else len(vectors[0]),
            "mean_vector": vec_array.mean(axis=0).tolist(),
            "std_vector": vec_array.std(axis=0).tolist(),
            "min_values": vec_array.min(axis=0).tolist(),
            "max_values": vec_array.max(axis=0).tolist()
        }
        
        # Calculate norms
        norms = [np.linalg.norm(vec) for vec in vec_array]
        stats["norm_statistics"] = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
            "min_norm": float(np.min(norms)),
            "max_norm": float(np.max(norms))
        }
        
        # Calculate pairwise similarities if requested
        if calculate_pairwise and len(vectors) > 1:
            pairwise_sims = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = self._cosine_similarity(vec_array[i], vec_array[j])
                    pairwise_sims.append(float(sim))
            
            stats["pairwise_similarities"] = {
                "count": len(pairwise_sims),
                "mean": float(np.mean(pairwise_sims)),
                "std": float(np.std(pairwise_sims)),
                "min": float(np.min(pairwise_sims)),
                "max": float(np.max(pairwise_sims))
            }
        
        return stats
    
    @mcp_tool(
        name="normalize_vectors",
        description="Normalize vectors to unit length",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Vectors to normalize"
                },
                "norm_type": {
                    "type": "string",
                    "enum": ["l2", "l1", "max"],
                    "default": "l2",
                    "description": "Type of normalization"
                }
            },
            "required": ["vectors"]
        }
    )
    @skill_provides("vector_normalization", "preprocessing")
    async def normalize_vectors_mcp(self,
                              vectors: List[List[float]],
                              norm_type: str = "l2") -> Dict[str, Any]:
        """MCP tool for vector normalization"""
        
        normalized_vectors = []
        normalization_factors = []
        
        for vector in vectors:
            vec = np.array(vector)
            
            if norm_type == "l2":
                norm = np.linalg.norm(vec)
            elif norm_type == "l1":
                norm = np.sum(np.abs(vec))
            elif norm_type == "max":
                norm = np.max(np.abs(vec))
            else:
                raise ValueError(f"Unsupported norm type: {norm_type}")
            
            if norm > 0:
                normalized_vec = (vec / norm).tolist()
                normalized_vectors.append(normalized_vec)
                normalization_factors.append(float(norm))
            else:
                # Handle zero vector
                normalized_vectors.append(vector)
                normalization_factors.append(0.0)
        
        return {
            "normalized_vectors": normalized_vectors,
            "normalization_factors": normalization_factors,
            "norm_type": norm_type,
            "count": len(normalized_vectors)
        }
    
    @mcp_resource(
        uri="vectorsim://calculator/metrics",
        name="Available Vector Metrics",
        description="List of supported vector similarity/distance metrics",
        mime_type="application/json"
    )
    async def get_available_metrics(self) -> Dict[str, Any]:
        """Get available metrics and their properties"""
        return {
            "metrics": {
                "cosine": {
                    "type": "similarity",
                    "range": [0, 1],
                    "description": "Cosine similarity between vectors",
                    "best_for": "Text embeddings, direction-based similarity"
                },
                "euclidean": {
                    "type": "distance",
                    "range": [0, "infinity"],
                    "description": "Euclidean (L2) distance",
                    "best_for": "Spatial data, continuous features"
                },
                "manhattan": {
                    "type": "distance",
                    "range": [0, "infinity"],
                    "description": "Manhattan (L1) distance",
                    "best_for": "Grid-based data, categorical features"
                },
                "dot_product": {
                    "type": "similarity",
                    "range": ["-infinity", "infinity"],
                    "description": "Dot product between vectors",
                    "best_for": "When vector magnitudes matter"
                },
                "jaccard": {
                    "type": "similarity",
                    "range": [0, 1],
                    "description": "Jaccard similarity for binary/sparse vectors",
                    "best_for": "Binary features, set similarity"
                },
                "hamming": {
                    "type": "distance",
                    "range": [0, "vector_length"],
                    "description": "Hamming distance for binary vectors",
                    "best_for": "Binary strings, categorical data"
                }
            }
        }
    
    @mcp_prompt(
        name="vector_similarity_analysis",
        description="Analyze vector similarity results",
        arguments=[
            {
                "name": "similarity_results",
                "description": "Results from similarity calculations",
                "required": True
            },
            {
                "name": "analysis_focus",
                "description": "Focus area (distribution, outliers, clusters)",
                "required": False
            }
        ]
    )
    async def vector_similarity_analysis_prompt(self,
                                          similarity_results: Dict[str, Any],
                                          analysis_focus: str = "comprehensive") -> str:
        """Analyze vector similarity results"""
        
        prompt = "Vector Similarity Analysis:\n\n"
        
        # Extract results
        results = similarity_results.get("results", [])
        if not results:
            return prompt + "No results to analyze."
        
        # Basic statistics
        similarities = [r.get("similarity", r.get("distance", 0)) for r in results]
        prompt += f"Results Summary:\n"
        prompt += f"- Total results: {len(results)}\n"
        prompt += f"- Metric used: {similarity_results.get('metric_used', 'unknown')}\n"
        prompt += f"- Score range: [{min(similarities):.4f}, {max(similarities):.4f}]\n"
        prompt += f"- Mean score: {np.mean(similarities):.4f}\n"
        prompt += f"- Std deviation: {np.std(similarities):.4f}\n\n"
        
        if analysis_focus in ["distribution", "comprehensive"]:
            prompt += "Score Distribution:\n"
            # Create simple histogram
            hist, bins = np.histogram(similarities, bins=5)
            for i, count in enumerate(hist):
                prompt += f"  [{bins[i]:.3f}-{bins[i+1]:.3f}]: {'*' * count} ({count})\n"
        
        if analysis_focus in ["outliers", "comprehensive"]:
            prompt += "\nPotential Outliers:\n"
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            outliers = [r for r, s in zip(results, similarities) 
                       if abs(s - mean_sim) > 2 * std_sim]
            for outlier in outliers[:5]:
                prompt += f"  - ID: {outlier['id']}, Score: {outlier.get('similarity', outlier.get('distance')):.4f}\n"
        
        return prompt
    
    # Internal calculation methods
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray, normalize: bool = True) -> float:
        """Calculate cosine similarity"""
        if normalize:
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            vec1 = vec1 / norm1
            vec2 = vec2 / norm2
        
        return float(np.dot(vec1, vec2))
    
    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return float(np.linalg.norm(vec1 - vec2))
    
    def _manhattan_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan distance"""
        return float(np.sum(np.abs(vec1 - vec2)))
    
    def _dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate dot product"""
        return float(np.dot(vec1, vec2))
    
    def _jaccard_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Jaccard similarity for binary/sparse vectors"""
        # Convert to binary
        bin1 = vec1 > 0
        bin2 = vec2 > 0
        
        intersection = np.sum(bin1 & bin2)
        union = np.sum(bin1 | bin2)
        
        if union == 0:
            return 1.0
        
        return float(intersection / union)
    
    def _hamming_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Hamming distance"""
        # For continuous vectors, discretize first
        if not np.array_equal(vec1, vec1.astype(bool).astype(float)):
            # Not binary, so discretize at 0.5
            vec1 = vec1 > 0.5
            vec2 = vec2 > 0.5
        
        return float(np.sum(vec1 != vec2))


# Singleton instance
mcp_vector_similarity = MCPVectorSimilarityCalculator()