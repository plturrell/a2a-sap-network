"""
Advanced MCP Vector Processing Agent (Agent 3)
Enhanced vector processing and embeddings with comprehensive MCP tool integration
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import faiss

from ...sdk.agentBase import A2AAgentBase
from ...sdk.decorators import a2a_handler, a2a_skill, a2a_task
from ...sdk.types import A2AMessage, MessageRole, TaskStatus, AgentCard
from ...sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ...common.mcpPerformanceTools import MCPPerformanceTools
from ...common.mcpValidationTools import MCPValidationTools
from ...common.mcpQualityAssessmentTools import MCPQualityAssessmentTools
from app.a2a.core.security_base import SecureA2AAgent

logger = logging.getLogger(__name__)


class AdvancedMCPVectorProcessingAgent(SecureA2AAgent):
    """
    Advanced Vector Processing Agent with comprehensive MCP tool integration
    Handles vector operations, embeddings, similarity search, and cross-agent coordination
    """
    
    def __init__(self, base_url: str):
        super().__init__(
            agent_id="advanced_mcp_vector_processing_agent",
            name="Advanced MCP Vector Processing Agent",
            description="Enhanced vector processing with comprehensive MCP tool integration",
            version="2.0.0",
            base_url=base_url
        )
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        
        # Initialize MCP tool providers
        self.performance_tools = MCPPerformanceTools()
        self.validation_tools = MCPValidationTools()
        self.quality_tools = MCPQualityAssessmentTools()
        
        
        # Vector processing state
        self.vector_stores = {}
        self.embedding_models = {}
        self.similarity_indices = {}
        self.clustering_models = {}
        self.processing_pipelines = {}
        
        # Initialize FAISS indices
        self.faiss_indices = {}
        
        logger.info(f"Initialized {self.name} with comprehensive MCP tool integration")
    
    @mcp_tool(
        name="intelligent_vector_processing",
        description="Process vectors with intelligent optimization and quality assessment",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "description": "Input vectors for processing"
                },
                "processing_config": {
                    "type": "object",
                    "description": "Processing configuration and parameters"
                },
                "operations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of operations to perform"
                },
                "quality_requirements": {
                    "type": "object",
                    "description": "Quality requirements for processing"
                },
                "optimization_level": {
                    "type": "string",
                    "enum": ["basic", "standard", "advanced"],
                    "default": "standard"
                },
                "cross_validation": {"type": "boolean", "default": True},
                "performance_monitoring": {"type": "boolean", "default": True}
            },
            "required": ["vectors", "operations"]
        }
    )
    async def intelligent_vector_processing(
        self,
        vectors: List[List[float]],
        operations: List[str],
        processing_config: Optional[Dict[str, Any]] = None,
        quality_requirements: Optional[Dict[str, Any]] = None,
        optimization_level: str = "standard",
        cross_validation: bool = True,
        performance_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Process vectors with intelligent optimization and quality assessment
        """
        processing_id = f"vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Validate input vectors using MCP tools
            vector_validation = await self.validation_tools.validate_vector_data(
                vectors=vectors,
                expected_dimensions=processing_config.get("expected_dimensions") if processing_config else None,
                validation_level="comprehensive"
            )
            
            if not vector_validation["is_valid"]:
                return {
                    "status": "error",
                    "error": "Vector validation failed",
                    "validation_details": vector_validation,
                    "processing_id": processing_id
                }
            
            # Step 2: Analyze vector characteristics
            vector_analysis = await self._analyze_vector_characteristics_mcp(vectors, processing_config)
            
            # Step 3: Optimize processing strategy based on analysis
            optimization_strategy = await self._optimize_processing_strategy_mcp(
                vectors, operations, vector_analysis, optimization_level
            )
            
            # Step 4: Execute vector operations
            operation_results = await self._execute_vector_operations_mcp(
                vectors, operations, optimization_strategy, processing_config
            )
            
            # Step 5: Cross-validation with other agents if enabled
            cross_validation_results = {}
            if cross_validation:
                cross_validation_results = await self._perform_cross_agent_validation_mcp(
                    operation_results, vector_analysis
                )
            
            # Step 6: Quality assessment using MCP tools
            quality_assessment = await self.quality_tools.assess_vector_processing_quality(
                original_vectors=vectors,
                processed_results=operation_results,
                operations_performed=operations,
                quality_requirements=quality_requirements or {},
                assessment_criteria=["accuracy", "efficiency", "consistency", "completeness"]
            )
            
            # Step 7: Performance measurement
            end_time = datetime.now().timestamp()
            if performance_monitoring:
                performance_metrics = await self.performance_tools.measure_performance_metrics(
                    operation_id=processing_id,
                    start_time=start_time,
                    end_time=end_time,
                    operation_count=len(operations),
                    custom_metrics={
                        "vectors_processed": len(vectors),
                        "vector_dimensions": len(vectors[0]) if vectors else 0,
                        "operations_count": len(operations),
                        "optimization_level": optimization_level,
                        "quality_score": quality_assessment.get("overall_score", 0)
                    }
                )
            else:
                performance_metrics = {}
            
            return {
                "status": "success",
                "processing_id": processing_id,
                "vector_validation": vector_validation,
                "vector_analysis": vector_analysis,
                "optimization_strategy": optimization_strategy,
                "operation_results": operation_results,
                "cross_validation": cross_validation_results,
                "quality_assessment": quality_assessment,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_vector_data",
                    "analyze_vector_characteristics",
                    "optimize_processing_strategy",
                    "execute_vector_operations",
                    "cross_agent_validation",
                    "assess_vector_processing_quality"
                ]
            }
            
        except Exception as e:
            logger.error(f"Intelligent vector processing failed: {e}")
            return {
                "status": "error",
                "processing_id": processing_id,
                "error": str(e)
            }
    
    @mcp_tool(
        name="advanced_similarity_search",
        description="Perform advanced similarity search with multiple algorithms and cross-validation",
        input_schema={
            "type": "object",
            "properties": {
                "query_vector": {
                    "type": "array",
                    "description": "Query vector for similarity search"
                },
                "search_space": {
                    "type": "array",
                    "description": "Vector space to search in"
                },
                "similarity_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Similarity metrics to use"
                },
                "search_parameters": {
                    "type": "object",
                    "description": "Search parameters and configurations"
                },
                "result_count": {"type": "integer", "default": 10},
                "quality_filtering": {"type": "boolean", "default": True},
                "cross_metric_validation": {"type": "boolean", "default": True}
            },
            "required": ["query_vector", "search_space"]
        }
    )
    async def advanced_similarity_search(
        self,
        query_vector: List[float],
        search_space: List[List[float]],
        similarity_metrics: Optional[List[str]] = None,
        search_parameters: Optional[Dict[str, Any]] = None,
        result_count: int = 10,
        quality_filtering: bool = True,
        cross_metric_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Perform advanced similarity search with multiple algorithms and cross-validation
        """
        search_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Validate query vector and search space
            validation_result = await self._validate_similarity_search_inputs_mcp(
                query_vector, search_space, similarity_metrics
            )
            
            if not validation_result["is_valid"]:
                return {
                    "status": "error",
                    "error": "Similarity search validation failed",
                    "validation_details": validation_result,
                    "search_id": search_id
                }
            
            # Step 2: Prepare search indices and optimize for performance
            index_preparation = await self._prepare_similarity_indices_mcp(
                search_space, similarity_metrics or ["cosine"], search_parameters
            )
            
            # Step 3: Execute similarity search with multiple metrics
            search_results = await self._execute_multi_metric_similarity_search_mcp(
                query_vector, search_space, similarity_metrics or ["cosine"], 
                index_preparation, result_count
            )
            
            # Step 4: Cross-metric validation and consensus
            consensus_results = {}
            if cross_metric_validation and len(similarity_metrics or []) > 1:
                consensus_results = await self._perform_cross_metric_validation_mcp(
                    search_results, query_vector, search_space
                )
            
            # Step 5: Quality filtering and ranking refinement
            filtered_results = search_results
            if quality_filtering:
                filtered_results = await self._apply_quality_filtering_mcp(
                    search_results, query_vector, search_space, search_parameters
                )
            
            # Step 6: Cross-agent validation for specialized domains
            domain_validation = await self._perform_domain_specific_validation_mcp(
                query_vector, filtered_results, search_parameters
            )
            
            # Step 7: Performance and quality assessment
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=search_id,
                start_time=start_time,
                end_time=end_time,
                custom_metrics={
                    "search_space_size": len(search_space),
                    "query_dimensions": len(query_vector),
                    "metrics_used": len(similarity_metrics or ["cosine"]),
                    "results_returned": len(filtered_results.get("results", [])),
                    "quality_filtering_applied": quality_filtering
                }
            )
            
            return {
                "status": "success",
                "search_id": search_id,
                "validation_result": validation_result,
                "index_preparation": index_preparation,
                "search_results": search_results,
                "consensus_results": consensus_results,
                "filtered_results": filtered_results,
                "domain_validation": domain_validation,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_similarity_search_inputs",
                    "prepare_similarity_indices",
                    "execute_multi_metric_search",
                    "cross_metric_validation",
                    "apply_quality_filtering",
                    "domain_specific_validation"
                ]
            }
            
        except Exception as e:
            logger.error(f"Advanced similarity search failed: {e}")
            return {
                "status": "error",
                "search_id": search_id,
                "error": str(e)
            }
    
    @mcp_tool(
        name="intelligent_vector_clustering",
        description="Perform intelligent vector clustering with adaptive algorithm selection",
        input_schema={
            "type": "object",
            "properties": {
                "vectors": {
                    "type": "array",
                    "description": "Vectors to cluster"
                },
                "clustering_config": {
                    "type": "object",
                    "description": "Clustering configuration parameters"
                },
                "adaptive_selection": {"type": "boolean", "default": True},
                "quality_optimization": {"type": "boolean", "default": True},
                "cross_validation": {"type": "boolean", "default": True},
                "performance_tuning": {"type": "boolean", "default": True}
            },
            "required": ["vectors"]
        }
    )
    async def intelligent_vector_clustering(
        self,
        vectors: List[List[float]],
        clustering_config: Optional[Dict[str, Any]] = None,
        adaptive_selection: bool = True,
        quality_optimization: bool = True,
        cross_validation: bool = True,
        performance_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Perform intelligent vector clustering with adaptive algorithm selection
        """
        clustering_id = f"clust_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now().timestamp()
        
        try:
            # Step 1: Validate input vectors and analyze characteristics
            vector_validation = await self._validate_clustering_inputs_mcp(vectors, clustering_config)
            
            if not vector_validation["is_valid"]:
                return {
                    "status": "error",
                    "error": "Clustering input validation failed",
                    "validation_details": vector_validation,
                    "clustering_id": clustering_id
                }
            
            # Step 2: Analyze vector distribution and characteristics
            distribution_analysis = await self._analyze_vector_distribution_mcp(vectors)
            
            # Step 3: Adaptive algorithm selection based on data characteristics
            algorithm_selection = {}
            if adaptive_selection:
                algorithm_selection = await self._select_clustering_algorithm_mcp(
                    vectors, distribution_analysis, clustering_config
                )
            else:
                algorithm_selection = {
                    "selected_algorithm": clustering_config.get("algorithm", "kmeans"),
                    "parameters": clustering_config or {}
                }
            
            # Step 4: Execute clustering with selected algorithm
            clustering_results = await self._execute_clustering_mcp(
                vectors, algorithm_selection, performance_tuning
            )
            
            # Step 5: Quality optimization and refinement
            optimized_results = clustering_results
            if quality_optimization:
                optimized_results = await self._optimize_clustering_quality_mcp(
                    vectors, clustering_results, algorithm_selection
                )
            
            # Step 6: Cross-validation with statistical measures
            validation_metrics = {}
            if cross_validation:
                validation_metrics = await self._validate_clustering_quality_mcp(
                    vectors, optimized_results, algorithm_selection
                )
            
            # Step 7: Cross-agent validation for domain-specific insights
            domain_insights = await self._get_domain_clustering_insights_mcp(
                vectors, optimized_results, clustering_config
            )
            
            # Step 8: Performance measurement and quality assessment
            end_time = datetime.now().timestamp()
            performance_metrics = await self.performance_tools.measure_performance_metrics(
                operation_id=clustering_id,
                start_time=start_time,
                end_time=end_time,
                custom_metrics={
                    "vectors_clustered": len(vectors),
                    "clusters_generated": len(optimized_results.get("cluster_centers", [])),
                    "algorithm_used": algorithm_selection.get("selected_algorithm"),
                    "silhouette_score": validation_metrics.get("silhouette_score", 0),
                    "adaptive_selection_used": adaptive_selection
                }
            )
            
            return {
                "status": "success",
                "clustering_id": clustering_id,
                "vector_validation": vector_validation,
                "distribution_analysis": distribution_analysis,
                "algorithm_selection": algorithm_selection,
                "clustering_results": clustering_results,
                "optimized_results": optimized_results,
                "validation_metrics": validation_metrics,
                "domain_insights": domain_insights,
                "performance_metrics": performance_metrics,
                "total_duration": end_time - start_time,
                "mcp_tools_used": [
                    "validate_clustering_inputs",
                    "analyze_vector_distribution",
                    "select_clustering_algorithm",
                    "execute_clustering",
                    "optimize_clustering_quality",
                    "validate_clustering_quality",
                    "get_domain_insights"
                ]
            }
            
        except Exception as e:
            logger.error(f"Intelligent vector clustering failed: {e}")
            return {
                "status": "error",
                "clustering_id": clustering_id,
                "error": str(e)
            }
    
    @mcp_resource(
        uri="vector-processing://vector-stores",
        name="Vector Stores Registry",
        description="Registry of all vector stores and their metadata"
    )
    async def get_vector_stores(self) -> Dict[str, Any]:
        """Provide access to vector stores registry as MCP resource"""
        return {
            "vector_stores": {
                store_id: {
                    "store_id": store_id,
                    "name": store.get("name", "Unknown"),
                    "dimensions": store.get("dimensions", 0),
                    "vector_count": store.get("vector_count", 0),
                    "index_type": store.get("index_type", "Unknown"),
                    "created": store.get("created_time"),
                    "last_accessed": store.get("last_accessed")
                }
                for store_id, store in self.vector_stores.items()
            },
            "total_stores": len(self.vector_stores),
            "total_vectors": sum(s.get("vector_count", 0) for s in self.vector_stores.values()),
            "index_types": self._get_index_type_summary(),
            "last_updated": datetime.now().isoformat()
        }
    
    @mcp_resource(
        uri="vector-processing://clustering-models",
        name="Clustering Models Registry", 
        description="Registry of trained clustering models and their performance"
    )
    async def get_clustering_models(self) -> Dict[str, Any]:
        """Provide access to clustering models registry as MCP resource"""
        return {
            "clustering_models": {
                model_id: {
                    "model_id": model_id,
                    "algorithm": model.get("algorithm", "Unknown"),
                    "parameters": model.get("parameters", {}),
                    "performance_score": model.get("performance_score", 0),
                    "vectors_trained_on": model.get("training_vector_count", 0),
                    "clusters_generated": model.get("cluster_count", 0),
                    "created": model.get("created_time"),
                    "usage_count": model.get("usage_count", 0)
                }
                for model_id, model in self.clustering_models.items()
            },
            "total_models": len(self.clustering_models),
            "algorithm_distribution": self._get_algorithm_distribution(),
            "performance_summary": self._get_performance_summary(),
            "last_updated": datetime.now().isoformat()
        }
    
    @mcp_prompt(
        name="vector_processing_advisor",
        description="Provide intelligent advice on vector processing strategies and optimization",
        arguments=[
            {"name": "task_type", "type": "string", "description": "Type of vector processing task"},
            {"name": "data_characteristics", "type": "object", "description": "Characteristics of the vector data"},
            {"name": "performance_requirements", "type": "object", "description": "Performance and quality requirements"}
        ]
    )
    async def vector_processing_advisor_prompt(
        self,
        task_type: str = "general",
        data_characteristics: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Provide intelligent advice on vector processing strategies and optimization
        """
        try:
            # Analyze current vector processing state
            current_state = await self._analyze_vector_processing_state_mcp()
            
            # Generate task-specific advice
            advice = await self._generate_vector_processing_advice_mcp(
                task_type, data_characteristics or {}, performance_requirements or {}, current_state
            )
            
            return advice
            
        except Exception as e:
            logger.error(f"Vector processing advisor failed: {e}")
            return f"I'm having trouble analyzing your vector processing needs. Error: {str(e)}"
    
    # Private helper methods for MCP operations
    
    async def _analyze_vector_characteristics_mcp(
        self, 
        vectors: List[List[float]], 
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze vector characteristics using MCP tools"""
        
        vectors_array = np.array(vectors)
        
        analysis = {
            "vector_count": len(vectors),
            "dimensions": len(vectors[0]) if vectors else 0,
            "statistics": {
                "mean": np.mean(vectors_array, axis=0).tolist() if len(vectors) > 0 else [],
                "std": np.std(vectors_array, axis=0).tolist() if len(vectors) > 0 else [],
                "min": np.min(vectors_array, axis=0).tolist() if len(vectors) > 0 else [],
                "max": np.max(vectors_array, axis=0).tolist() if len(vectors) > 0 else []
            },
            "sparsity": self._calculate_sparsity(vectors_array),
            "distribution_type": self._analyze_distribution_type(vectors_array),
            "recommended_algorithms": []
        }
        
        # Add algorithm recommendations based on characteristics
        if analysis["sparsity"] > 0.8:
            analysis["recommended_algorithms"].append("sparse_clustering")
        if analysis["dimensions"] > 1000:
            analysis["recommended_algorithms"].append("dimensionality_reduction")
        
        return analysis
    
    async def _optimize_processing_strategy_mcp(
        self,
        vectors: List[List[float]],
        operations: List[str],
        analysis: Dict[str, Any],
        optimization_level: str
    ) -> Dict[str, Any]:
        """Optimize processing strategy based on vector analysis"""
        
        strategy = {
            "optimization_level": optimization_level,
            "recommended_operations": [],
            "performance_optimizations": [],
            "memory_management": {}
        }
        
        # Optimize based on vector characteristics
        if analysis["dimensions"] > 500:
            strategy["performance_optimizations"].append("use_batch_processing")
            strategy["memory_management"]["batch_size"] = min(1000, len(vectors) // 4)
        
        if "similarity_search" in operations and len(vectors) > 10000:
            strategy["recommended_operations"].append("build_faiss_index")
        
        if "clustering" in operations:
            if analysis["vector_count"] > 50000:
                strategy["recommended_operations"].append("use_mini_batch_kmeans")
            else:
                strategy["recommended_operations"].append("use_standard_kmeans")
        
        return strategy
    
    async def _execute_vector_operations_mcp(
        self,
        vectors: List[List[float]],
        operations: List[str],
        strategy: Dict[str, Any],
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute vector operations with optimization strategy"""
        
        results = {
            "operations_performed": [],
            "results": {},
            "performance_stats": {}
        }
        
        vectors_array = np.array(vectors)
        
        for operation in operations:
            operation_start = datetime.now().timestamp()
            
            try:
                if operation == "normalize":
                    # L2 normalization
                    norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
                    norms[norms == 0] = 1  # Avoid division by zero
                    normalized = vectors_array / norms
                    results["results"][operation] = normalized.tolist()
                
                elif operation == "dimensionality_reduction":
                    # PCA for dimensionality reduction
                    target_dims = config.get("target_dimensions", min(50, vectors_array.shape[1] // 2)) if config else 50
                    pca = PCA(n_components=target_dims)
                    reduced = pca.fit_transform(vectors_array)
                    results["results"][operation] = {
                        "reduced_vectors": reduced.tolist(),
                        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist()
                    }
                
                elif operation == "similarity_matrix":
                    # Compute similarity matrix
                    similarity_matrix = cosine_similarity(vectors_array)
                    results["results"][operation] = similarity_matrix.tolist()
                
                elif operation == "clustering":
                    # Basic K-means clustering
                    n_clusters = config.get("n_clusters", min(8, len(vectors) // 10)) if config else 5
                    if len(vectors) >= n_clusters:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = kmeans.fit_predict(vectors_array)
                        results["results"][operation] = {
                            "cluster_labels": cluster_labels.tolist(),
                            "cluster_centers": kmeans.cluster_centers_.tolist(),
                            "inertia": float(kmeans.inertia_)
                        }
                    else:
                        results["results"][operation] = {"error": "Not enough vectors for clustering"}
                
                operation_end = datetime.now().timestamp()
                results["performance_stats"][operation] = {
                    "duration": operation_end - operation_start,
                    "success": True
                }
                results["operations_performed"].append(operation)
                
            except Exception as e:
                operation_end = datetime.now().timestamp()
                results["performance_stats"][operation] = {
                    "duration": operation_end - operation_start,
                    "success": False,
                    "error": str(e)
                }
                logger.warning(f"Operation {operation} failed: {e}")
        
        return results
    
    async def _perform_cross_agent_validation_mcp(
        self, 
        operation_results: Dict[str, Any], 
        vector_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cross-agent validation using MCP"""
        validation_results = {}
        
        try:
            # Request validation from Data Product Agent for data quality
            if "clustering" in operation_results.get("results", {}):
                data_quality_validation = await self.mcp_client.call_skill_tool(
                    "agent_0_data_product",
                    "validate_clustering_results",
                    {
                        "clustering_results": operation_results["results"]["clustering"],
                        "vector_analysis": vector_analysis,
                        "validation_level": "standard"
                    }
                )
                validation_results["data_quality"] = data_quality_validation.get("result", {})
            
            # Request validation from Calculation Agent for numerical accuracy
            if "similarity_matrix" in operation_results.get("results", {}):
                numerical_validation = await self.mcp_client.call_skill_tool(
                    "calculation_agent",
                    "validate_matrix_calculations",
                    {
                        "matrix_data": operation_results["results"]["similarity_matrix"],
                        "expected_properties": ["symmetric", "positive_semidefinite"],
                        "tolerance": 1e-6
                    }
                )
                validation_results["numerical_accuracy"] = numerical_validation.get("result", {})
            
        except Exception as e:
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _calculate_sparsity(self, vectors_array: np.ndarray) -> float:
        """Calculate sparsity of vector array"""
        if vectors_array.size == 0:
            return 0.0
        zero_elements = np.count_nonzero(vectors_array == 0)
        total_elements = vectors_array.size
        return zero_elements / total_elements
    
    def _analyze_distribution_type(self, vectors_array: np.ndarray) -> str:
        """Analyze the distribution type of vectors"""
        if vectors_array.size == 0:
            return "empty"
        
        # Simple heuristic based on statistics
        means = np.mean(vectors_array, axis=0)
        stds = np.std(vectors_array, axis=0)
        
        # Check if approximately normal (mean near 0, std near 1)
        if np.allclose(means, 0, atol=0.5) and np.allclose(stds, 1, atol=0.5):
            return "approximately_normal"
        elif np.all(vectors_array >= 0):
            return "positive_definite"
        else:
            return "mixed_distribution"
    
    def _get_index_type_summary(self) -> Dict[str, int]:
        """Get summary of index types in vector stores"""
        type_counts = {}
        for store in self.vector_stores.values():
            index_type = store.get("index_type", "unknown")
            type_counts[index_type] = type_counts.get(index_type, 0) + 1
        return type_counts
    
    def _get_algorithm_distribution(self) -> Dict[str, int]:
        """Get distribution of clustering algorithms"""
        algo_counts = {}
        for model in self.clustering_models.values():
            algorithm = model.get("algorithm", "unknown")
            algo_counts[algorithm] = algo_counts.get(algorithm, 0) + 1
        return algo_counts
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary of clustering models"""
        if not self.clustering_models:
            return {"average_score": 0, "best_score": 0, "worst_score": 0}
        
        scores = [m.get("performance_score", 0) for m in self.clustering_models.values()]
        return {
            "average_score": np.mean(scores),
            "best_score": np.max(scores),
            "worst_score": np.min(scores),
            "total_models": len(scores)
        }


# Factory function for creating advanced MCP vector processing agent
def create_advanced_mcp_vector_processing_agent(base_url: str) -> AdvancedMCPVectorProcessingAgent:
    """Create and configure advanced MCP vector processing agent"""
    return AdvancedMCPVectorProcessingAgent(base_url)