"""
Vector Quantization Skills for Agent 3 - Phase 2 Advanced Features
Implements Product Quantization (PQ) and compression for efficient vector storage
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import faiss
import json
import os
from pathlib import Path

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task
from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
try:
    from app.a2a.core.trustIdentity import TrustIdentity
except ImportError:
    class TrustIdentity:
        def __init__(self, **kwargs): pass
        def validate(self, *args): return True

try:
    from app.a2a.core.dataValidation import DataValidator
except ImportError:
    class DataValidator:
        def __init__(self, **kwargs): pass
        def validate(self, *args): return {"valid": True}


@dataclass
class QuantizationConfig:
    """Configuration for vector quantization parameters"""
    num_subquantizers: int = 8  # Number of subquantizers (m)
    subquantizer_bits: int = 8  # Bits per subquantizer (nbits)
    training_vectors_count: int = 10000  # Vectors for training
    compression_ratio: float = 32.0  # Target compression ratio
    enable_pca: bool = True  # Enable PCA preprocessing
    pca_dimensions: int = 256  # PCA reduced dimensions


class VectorQuantizationSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Real A2A agent skills for advanced vector quantization and compression
    Implements Product Quantization (PQ) with FAISS integration
    """

    def __init__(self, trust_identity: TrustIdentity):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataValidator()
        
        # Initialize quantization models
        self.pq_models: Dict[str, faiss.ProductQuantizer] = {}
        self.pca_models: Dict[str, PCA] = {}
        self.quantization_configs: Dict[str, QuantizationConfig] = {}
        self.index_metadata: Dict[str, Dict] = {}
        
        # Performance tracking
        self.compression_metrics = {
            'total_vectors_quantized': 0,
            'storage_saved_bytes': 0,
            'search_time_improvement': 0.0,
            'quantization_errors': []
        }

    @a2a_skill(
        name="createProductQuantizationIndex",
        description="Create Product Quantization index for vector compression",
        input_schema={
            "type": "object",
            "properties": {
                "index_name": {"type": "string"},
                "vector_dimension": {"type": "integer"},
                "training_vectors": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "config": {
                    "type": "object",
                    "properties": {
                        "num_subquantizers": {"type": "integer", "default": 8},
                        "subquantizer_bits": {"type": "integer", "default": 8},
                        "enable_pca": {"type": "boolean", "default": True},
                        "pca_dimensions": {"type": "integer", "default": 256}
                    }
                }
            },
            "required": ["index_name", "vector_dimension", "training_vectors"]
        }
    )
    def create_product_quantization_index(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Product Quantization index for efficient vector compression"""
        try:
            index_name = request_data["index_name"]
            vector_dimension = request_data["vector_dimension"]
            training_vectors = np.array(request_data["training_vectors"], dtype=np.float32)
            
            # Create quantization config
            config_data = request_data.get("config", {})
            config = QuantizationConfig(
                num_subquantizers=config_data.get("num_subquantizers", 8),
                subquantizer_bits=config_data.get("subquantizer_bits", 8),
                enable_pca=config_data.get("enable_pca", True),
                pca_dimensions=config_data.get("pca_dimensions", 256)
            )
            
            self.quantization_configs[index_name] = config
            
            # Validate inputs
            if training_vectors.shape[0] < config.num_subquantizers:
                raise ValueError("Need more training vectors than subquantizers")
            
            # Apply PCA if enabled
            processed_vectors = training_vectors
            if config.enable_pca and vector_dimension > config.pca_dimensions:
                self.logger.info(f"Applying PCA reduction: {vector_dimension} -> {config.pca_dimensions}")
                pca = PCA(n_components=config.pca_dimensions)
                processed_vectors = pca.fit_transform(training_vectors)
                self.pca_models[index_name] = pca
                vector_dimension = config.pca_dimensions
            
            # Create Product Quantizer
            pq = faiss.ProductQuantizer(
                vector_dimension,
                config.num_subquantizers,
                config.subquantizer_bits
            )
            
            # Train the quantizer
            self.logger.info(f"Training PQ with {len(processed_vectors)} vectors")
            pq.train(processed_vectors)
            
            # Store the trained model
            self.pq_models[index_name] = pq
            
            # Calculate compression ratio
            original_size = vector_dimension * 4  # float32 = 4 bytes
            compressed_size = config.num_subquantizers * (config.subquantizer_bits / 8)
            actual_compression_ratio = original_size / compressed_size
            
            # Store metadata
            self.index_metadata[index_name] = {
                'vector_dimension': vector_dimension,
                'original_dimension': request_data["vector_dimension"],
                'compression_ratio': actual_compression_ratio,
                'training_vectors_count': len(training_vectors),
                'pca_enabled': config.enable_pca,
                'created_timestamp': np.datetime64('now').astype(str)
            }
            
            self.logger.info(f"PQ index '{index_name}' created with {actual_compression_ratio:.1f}x compression")
            
            return {
                'success': True,
                'index_name': index_name,
                'compression_ratio': actual_compression_ratio,
                'vector_dimension': vector_dimension,
                'subquantizers': config.num_subquantizers,
                'bits_per_subquantizer': config.subquantizer_bits,
                'pca_applied': config.enable_pca,
                'metadata': self.index_metadata[index_name]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create PQ index: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'quantization_creation_error'
            }

    @a2a_skill(
        name="quantizeVectors",
        description="Quantize vectors using trained Product Quantization model",
        input_schema={
            "type": "object",
            "properties": {
                "index_name": {"type": "string"},
                "vectors": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "vector_ids": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["index_name", "vectors"]
        }
    )
    def quantize_vectors(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize vectors using the trained PQ model"""
        try:
            index_name = request_data["index_name"]
            vectors = np.array(request_data["vectors"], dtype=np.float32)
            vector_ids = request_data.get("vector_ids", [f"vec_{i}" for i in range(len(vectors))])
            
            if index_name not in self.pq_models:
                raise ValueError(f"PQ model '{index_name}' not found. Train model first.")
            
            pq = self.pq_models[index_name]
            
            # Apply PCA if model uses it
            processed_vectors = vectors
            if index_name in self.pca_models:
                processed_vectors = self.pca_models[index_name].transform(vectors)
            
            # Quantize vectors
            quantized_codes = pq.compute_codes(processed_vectors)
            
            # Calculate storage savings
            original_size = vectors.nbytes
            compressed_size = quantized_codes.nbytes
            storage_saved = original_size - compressed_size
            
            # Update metrics
            self.compression_metrics['total_vectors_quantized'] += len(vectors)
            self.compression_metrics['storage_saved_bytes'] += storage_saved
            
            # Store quantized vectors with metadata
            quantized_data = []
            for i, (vector_id, codes) in enumerate(zip(vector_ids, quantized_codes)):
                quantized_data.append({
                    'vector_id': vector_id,
                    'quantized_codes': codes.tolist(),
                    'original_norm': float(np.linalg.norm(vectors[i])),
                    'compression_error': self._calculate_reconstruction_error(
                        vectors[i], pq.decode(codes.reshape(1, -1))[0]
                    )
                })
            
            self.logger.info(f"Quantized {len(vectors)} vectors, saved {storage_saved} bytes")
            
            return {
                'success': True,
                'quantized_vectors': quantized_data,
                'compression_stats': {
                    'original_size_bytes': int(original_size),
                    'compressed_size_bytes': int(compressed_size),
                    'storage_saved_bytes': int(storage_saved),
                    'compression_ratio': float(original_size / compressed_size)
                },
                'index_metadata': self.index_metadata[index_name]
            }
            
        except Exception as e:
            self.logger.error(f"Vector quantization failed: {str(e)}")
            self.compression_metrics['quantization_errors'].append(str(e))
            return {
                'success': False,
                'error': str(e),
                'error_type': 'vector_quantization_error'
            }

    @a2a_skill(
        name="searchQuantizedVectors",
        description="Perform similarity search on quantized vectors",
        input_schema={
            "type": "object",
            "properties": {
                "index_name": {"type": "string"},
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "k": {"type": "integer", "default": 10},
                "quantized_database": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "vector_id": {"type": "string"},
                            "quantized_codes": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            "required": ["index_name", "query_vector", "quantized_database"]
        }
    )
    def search_quantized_vectors(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar vectors in quantized space"""
        try:
            index_name = request_data["index_name"]
            query_vector = np.array(request_data["query_vector"], dtype=np.float32)
            k = request_data.get("k", 10)
            quantized_database = request_data["quantized_database"]
            
            if index_name not in self.pq_models:
                raise ValueError(f"PQ model '{index_name}' not found")
            
            pq = self.pq_models[index_name]
            
            # Apply PCA to query if model uses it
            processed_query = query_vector
            if index_name in self.pca_models:
                processed_query = self.pca_models[index_name].transform(query_vector.reshape(1, -1))[0]
            
            # Extract quantized codes from database
            db_codes = []
            db_ids = []
            for item in quantized_database:
                db_codes.append(item["quantized_codes"])
                db_ids.append(item["vector_id"])
            
            db_codes = np.array(db_codes, dtype=np.uint8)
            
            # Compute asymmetric distances (query vs quantized database)
            distances = pq.compute_distance_table(processed_query.reshape(1, -1))
            
            # Calculate distances to all database vectors
            vector_distances = []
            for i, codes in enumerate(db_codes):
                distance = np.sum(distances[0, np.arange(len(codes)), codes])
                vector_distances.append((distance, db_ids[i], i))
            
            # Sort by distance and return top k
            vector_distances.sort(key=lambda x: x[0])
            results = vector_distances[:k]
            
            search_results = []
            for distance, vector_id, idx in results:
                search_results.append({
                    'vector_id': vector_id,
                    'distance': float(distance),
                    'similarity_score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                    'rank': len(search_results) + 1
                })
            
            return {
                'success': True,
                'search_results': search_results,
                'query_stats': {
                    'total_database_size': len(quantized_database),
                    'results_returned': len(search_results),
                    'search_method': 'asymmetric_distance_computation'
                },
                'index_name': index_name
            }
            
        except Exception as e:
            self.logger.error(f"Quantized vector search failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'quantized_search_error'
            }

    @a2a_skill(
        name="getCompressionMetrics",
        description="Get detailed compression and performance metrics",
        input_schema={
            "type": "object",
            "properties": {
                "index_names": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    )
    def get_compression_metrics(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve comprehensive compression metrics and statistics"""
        try:
            index_names = request_data.get("index_names", list(self.index_metadata.keys()))
            
            metrics_report = {
                'global_metrics': self.compression_metrics.copy(),
                'index_specific_metrics': {},
                'performance_summary': {
                    'total_indexes': len(self.pq_models),
                    'active_indexes': len([name for name in index_names if name in self.pq_models]),
                    'total_storage_saved_mb': self.compression_metrics['storage_saved_bytes'] / (1024 * 1024)
                }
            }
            
            for index_name in index_names:
                if index_name in self.index_metadata:
                    metadata = self.index_metadata[index_name]
                    config = self.quantization_configs.get(index_name)
                    
                    metrics_report['index_specific_metrics'][index_name] = {
                        'metadata': metadata,
                        'configuration': {
                            'num_subquantizers': config.num_subquantizers if config else None,
                            'subquantizer_bits': config.subquantizer_bits if config else None,
                            'pca_enabled': config.enable_pca if config else None,
                            'pca_dimensions': config.pca_dimensions if config else None
                        },
                        'model_status': {
                            'pq_model_trained': index_name in self.pq_models,
                            'pca_model_available': index_name in self.pca_models,
                            'ready_for_quantization': index_name in self.pq_models
                        }
                    }
            
            return {
                'success': True,
                'compression_metrics': metrics_report,
                'timestamp': np.datetime64('now').astype(str)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get compression metrics: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'metrics_retrieval_error'
            }

    def _calculate_reconstruction_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate L2 reconstruction error between original and reconstructed vector"""
        if original.shape != reconstructed.shape:
            return float('inf')
        
        error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
        return float(error)

    @a2a_task(
        task_type="optimizeQuantizationParameters",
        description="Automatically optimize quantization parameters for best compression/accuracy trade-off"
    )
    def optimize_quantization_parameters(self, training_data: np.ndarray, target_compression: float = 16.0) -> QuantizationConfig:
        """Optimize PQ parameters based on training data characteristics"""
        try:
            vector_dim = training_data.shape[1]
            num_vectors = training_data.shape[0]
            
            # Analyze vector characteristics
            variance_per_dim = np.var(training_data, axis=0)
            mean_variance = np.mean(variance_per_dim)
            
            # Optimize number of subquantizers
            optimal_m = min(vector_dim // 4, max(4, int(np.log2(target_compression))))
            
            # Optimize bits per subquantizer
            optimal_nbits = max(4, min(8, int(np.log2(num_vectors / optimal_m))))
            
            # Decide on PCA based on dimensionality and variance distribution
            enable_pca = vector_dim > 128 and np.std(variance_per_dim) / mean_variance > 0.5
            pca_dim = min(vector_dim // 2, 256) if enable_pca else vector_dim
            
            optimized_config = QuantizationConfig(
                num_subquantizers=optimal_m,
                subquantizer_bits=optimal_nbits,
                enable_pca=enable_pca,
                pca_dimensions=pca_dim,
                training_vectors_count=min(num_vectors, 50000)
            )
            
            self.logger.info(f"Optimized PQ config: m={optimal_m}, nbits={optimal_nbits}, PCA={enable_pca}")
            
            return optimized_config
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {str(e)}")
            return QuantizationConfig()  # Return default config