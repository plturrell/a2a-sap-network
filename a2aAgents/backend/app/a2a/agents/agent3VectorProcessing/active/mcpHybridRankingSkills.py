from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import logging
import json
import math
from collections import Counter, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ....sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ....sdk.mcpSkillCoordination import skill_provides, skill_depends_on

from app.a2a.core.security_base import SecureA2AAgent
"""
MCP-enabled Hybrid Ranking Skills
Exposes BM25, PageRank, and hybrid ranking as MCP tools for cross-agent usage
"""

logger = logging.getLogger(__name__)


class MCPHybridRankingSkills(SecureA2AAgent):
    """MCP-enabled hybrid ranking with exposed tools for cross-agent usage"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, hanaConnection=None):
        super().__init__()
        self.hanaConnection = hanaConnection
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.defaultWeights = {
            'vectorSimilarity': 0.4,
            'bm25Score': 0.3,
            'pageRankScore': 0.2,
            'contextualRelevance': 0.1
        }
        self.documentFrequencyCache = {}
        self.pageRankCache = {}
        
        # BM25 default parameters
        self.bm25_k1 = 1.2  # Term frequency saturation
        self.bm25_b = 0.75  # Length normalization
    
    @mcp_tool(
        name="hybrid_ranking_search",
        description="Perform hybrid ranking combining vector similarity, BM25, PageRank, and contextual relevance",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query embedding vector"
                },
                "candidate_documents": {
                    "type": "array",
                    "description": "List of candidate documents to rank"
                },
                "ranking_weights": {
                    "type": "object",
                    "description": "Custom weights for ranking components",
                    "properties": {
                        "vectorSimilarity": {"type": "number"},
                        "bm25Score": {"type": "number"},
                        "pageRankScore": {"type": "number"},
                        "contextualRelevance": {"type": "number"}
                    }
                },
                "search_context": {
                    "type": "object",
                    "description": "Additional search context"
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of top results to return"
                },
                "apply_diversity": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply result diversification"
                }
            },
            "required": ["query", "candidate_documents"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "ranked_results": {"type": "array"},
                "ranking_metadata": {"type": "object"},
                "performance_metrics": {"type": "object"}
            }
        }
    )
    @skill_provides("document_ranking", "hybrid_search")
    async def hybrid_ranking_search_mcp(self,
                                  query: str,
                                  candidate_documents: List[Dict[str, Any]],
                                  query_vector: Optional[List[float]] = None,
                                  ranking_weights: Optional[Dict[str, float]] = None,
                                  search_context: Optional[Dict[str, Any]] = None,
                                  top_k: int = 10,
                                  apply_diversity: bool = True) -> Dict[str, Any]:
        """MCP-exposed hybrid ranking search"""
        
        start_time = datetime.now()
        
        if not candidate_documents:
            return {
                "ranked_results": [],
                "ranking_metadata": {"message": "No documents to rank"},
                "performance_metrics": {"total_time_ms": 0}
            }
        
        # Use provided weights or defaults
        weights = ranking_weights or self.defaultWeights
        context = search_context or {}
        
        logger.info(f"Hybrid ranking {len(candidate_documents)} documents")
        
        # Compute individual ranking components
        ranking_components = await self._compute_all_ranking_components(
            query, query_vector, candidate_documents, context
        )
        
        # Combine rankings
        final_scores = self._combine_rankings_weighted(
            candidate_documents,
            ranking_components,
            weights
        )
        
        # Sort and prepare results
        ranked_results = self._prepare_ranked_results(
            candidate_documents,
            final_scores,
            ranking_components
        )
        
        # Apply diversification if requested
        if apply_diversity:
            ranked_results = await self._apply_diversification(ranked_results, context)
        
        # Take top k results
        final_results = ranked_results[:top_k]
        
        # Calculate performance metrics
        end_time = datetime.now()
        performance_metrics = {
            "total_time_ms": (end_time - start_time).total_seconds() * 1000,
            "documents_processed": len(candidate_documents),
            "results_returned": len(final_results)
        }
        
        return {
            "ranked_results": final_results,
            "ranking_metadata": {
                "weights_used": weights,
                "diversity_applied": apply_diversity,
                "total_candidates": len(candidate_documents)
            },
            "performance_metrics": performance_metrics
        }
    
    @mcp_tool(
        name="calculate_bm25_scores",
        description="Calculate BM25 relevance scores for documents",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "documents": {
                    "type": "array",
                    "description": "Documents to score"
                },
                "k1": {
                    "type": "number",
                    "default": 1.2,
                    "description": "BM25 k1 parameter"
                },
                "b": {
                    "type": "number",
                    "default": 0.75,
                    "description": "BM25 b parameter"
                },
                "return_details": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return detailed scoring breakdown"
                }
            },
            "required": ["query", "documents"]
        }
    )
    @skill_provides("text_relevance", "bm25_scoring")
    async def calculate_bm25_scores_mcp(self,
                                  query: str,
                                  documents: List[Dict[str, Any]],
                                  k1: float = 1.2,
                                  b: float = 0.75,
                                  return_details: bool = False) -> Dict[str, Any]:
        """MCP tool for BM25 scoring"""
        
        # Tokenize query
        query_terms = self._tokenize_text(query)
        
        if not query_terms:
            return {
                "scores": {doc.get('docId', idx): 0.0 for idx, doc in enumerate(documents)},
                "query_terms": [],
                "message": "Empty query after tokenization"
            }
        
        # Calculate corpus statistics
        corpus_stats = self._calculate_corpus_statistics(documents)
        total_docs = corpus_stats['total_documents']
        avg_doc_length = corpus_stats['avg_doc_length']
        
        scores = {}
        score_details = {}
        
        for idx, doc in enumerate(documents):
            doc_id = doc.get('docId', str(idx))
            content = doc.get('content', '') or doc.get('text', '')
            
            if not content:
                scores[doc_id] = 0.0
                continue
            
            # Tokenize document
            doc_terms = self._tokenize_text(content)
            doc_length = len(doc_terms)
            term_freqs = Counter(doc_terms)
            
            score = 0.0
            term_scores = {}
            
            for term in query_terms:
                if term in term_freqs:
                    tf = term_freqs[term]
                    df = self._get_document_frequency(term, documents)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    term_score = idf * (numerator / denominator)
                    
                    score += term_score
                    term_scores[term] = {
                        "tf": tf,
                        "df": df,
                        "idf": idf,
                        "score": term_score
                    }
            
            scores[doc_id] = max(0.0, score)
            
            if return_details:
                score_details[doc_id] = {
                    "total_score": scores[doc_id],
                    "doc_length": doc_length,
                    "term_scores": term_scores
                }
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {doc_id: score/max_score for doc_id, score in scores.items()}
        
        result = {
            "scores": scores,
            "query_terms": query_terms,
            "corpus_stats": corpus_stats,
            "parameters": {"k1": k1, "b": b}
        }
        
        if return_details:
            result["score_details"] = score_details
        
        return result
    
    @mcp_tool(
        name="calculate_pagerank_scores",
        description="Calculate PageRank-style authority scores for documents",
        input_schema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "Documents with link/reference information"
                },
                "damping_factor": {
                    "type": "number",
                    "default": 0.85,
                    "description": "PageRank damping factor"
                },
                "iterations": {
                    "type": "integer",
                    "default": 30,
                    "description": "Number of iterations"
                },
                "relationship_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of relationships to consider"
                }
            },
            "required": ["documents"]
        }
    )
    @skill_provides("authority_scoring", "pagerank")
    async def calculate_pagerank_scores_mcp(self,
                                      documents: List[Dict[str, Any]],
                                      damping_factor: float = 0.85,
                                      iterations: int = 30,
                                      relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """MCP tool for PageRank scoring"""
        
        if not documents:
            return {
                "scores": {},
                "metadata": {"message": "No documents provided"}
            }
        
        # Build link graph
        link_graph = self._build_link_graph(documents, relationship_types)
        
        # Initialize PageRank scores
        num_docs = len(documents)
        doc_ids = [doc.get('docId', str(idx)) for idx, doc in enumerate(documents)]
        
        # Initial uniform distribution
        pagerank_scores = {doc_id: 1.0 / num_docs for doc_id in doc_ids}
        
        # Power iteration
        for iteration in range(iterations):
            new_scores = {}
            
            for doc_id in doc_ids:
                # Base score (random surfer)
                score = (1 - damping_factor) / num_docs
                
                # Add contributions from incoming links
                for other_id, links in link_graph.items():
                    if doc_id in links:
                        # Contribution from other_id to doc_id
                        num_outlinks = len(links)
                        if num_outlinks > 0:
                            score += damping_factor * pagerank_scores[other_id] / num_outlinks
                
                new_scores[doc_id] = score
            
            # Check convergence
            converged = all(
                abs(new_scores[doc_id] - pagerank_scores[doc_id]) < 1e-6
                for doc_id in doc_ids
            )
            
            pagerank_scores = new_scores
            
            if converged:
                break
        
        # Normalize scores
        total_score = sum(pagerank_scores.values())
        if total_score > 0:
            pagerank_scores = {
                doc_id: score / total_score 
                for doc_id, score in pagerank_scores.items()
            }
        
        return {
            "scores": pagerank_scores,
            "metadata": {
                "iterations_run": iteration + 1,
                "converged": converged,
                "damping_factor": damping_factor,
                "num_documents": num_docs,
                "link_density": sum(len(links) for links in link_graph.values()) / (num_docs * num_docs)
            }
        }
    
    @mcp_tool(
        name="calculate_vector_similarity",
        description="Calculate cosine similarity between vectors",
        input_schema={
            "type": "object",
            "properties": {
                "query_vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query vector"
                },
                "document_vectors": {
                    "type": "array",
                    "description": "List of document vectors"
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Normalize vectors before comparison"
                }
            },
            "required": ["query_vector", "document_vectors"]
        }
    )
    @skill_provides("vector_similarity", "cosine_similarity")
    async def calculate_vector_similarity_mcp(self,
                                        query_vector: List[float],
                                        document_vectors: List[Dict[str, Any]],
                                        normalize: bool = True) -> Dict[str, Any]:
        """MCP tool for vector similarity calculation"""
        
        if not query_vector or not document_vectors:
            return {
                "similarities": {},
                "metadata": {"message": "Missing vectors"}
            }
        
        # Convert query vector to numpy array
        q_vec = np.array(query_vector)
        
        if normalize:
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
        
        similarities = {}
        
        for doc in document_vectors:
            doc_id = doc.get('docId', doc.get('id'))
            doc_vector = doc.get('vector', doc.get('embedding'))
            
            if doc_vector and doc_id:
                d_vec = np.array(doc_vector)
                
                if normalize:
                    d_norm = np.linalg.norm(d_vec)
                    if d_norm > 0:
                        d_vec = d_vec / d_norm
                
                # Cosine similarity
                similarity = np.dot(q_vec, d_vec)
                similarities[doc_id] = float(similarity)
        
        return {
            "similarities": similarities,
            "metadata": {
                "query_vector_dim": len(query_vector),
                "documents_processed": len(similarities),
                "normalized": normalize
            }
        }
    
    @mcp_resource(
        uri="ranking://config/default",
        name="Default Ranking Configuration",
        description="Get default ranking weights and parameters",
        mime_type="application/json"
    )
    async def get_ranking_config(self) -> Dict[str, Any]:
        """Get current ranking configuration"""
        return {
            "default_weights": self.defaultWeights,
            "bm25_parameters": {
                "k1": self.bm25_k1,
                "b": self.bm25_b
            },
            "pagerank_parameters": {
                "damping_factor": 0.85,
                "default_iterations": 30
            },
            "available_methods": [
                "hybrid_ranking",
                "bm25_only",
                "vector_only",
                "pagerank_only"
            ]
        }
    
    @mcp_prompt(
        name="ranking_analysis",
        description="Analyze ranking results and provide insights",
        arguments=[
            {
                "name": "ranked_results",
                "description": "Results from ranking operation",
                "required": True
            },
            {
                "name": "analysis_type",
                "description": "Type of analysis (distribution, diversity, quality)",
                "required": False
            }
        ]
    )
    async def ranking_analysis_prompt(self,
                                ranked_results: List[Dict[str, Any]],
                                analysis_type: str = "comprehensive") -> str:
        """Analyze ranking results"""
        
        prompt = f"Ranking Analysis for {len(ranked_results)} results:\n\n"
        
        if not ranked_results:
            return prompt + "No results to analyze."
        
        # Score distribution
        scores = [r.get('hybridScore', r.get('score', 0)) for r in ranked_results]
        prompt += f"Score Distribution:\n"
        prompt += f"- Highest: {max(scores):.3f}\n"
        prompt += f"- Lowest: {min(scores):.3f}\n"
        prompt += f"- Average: {np.mean(scores):.3f}\n"
        prompt += f"- Std Dev: {np.std(scores):.3f}\n\n"
        
        # Component analysis
        if ranked_results[0].get('rankingComponents'):
            prompt += "Component Contributions (top 5):\n"
            for idx, result in enumerate(ranked_results[:5]):
                components = result.get('rankingComponents', {})
                prompt += f"{idx+1}. Document {result.get('docId', idx)}:\n"
                for comp, score in components.items():
                    prompt += f"   - {comp}: {score:.3f}\n"
        
        return prompt
    
    # Internal helper methods
    async def _compute_all_ranking_components(self,
                                            query: str,
                                            query_vector: Optional[List[float]],
                                            documents: List[Dict[str, Any]],
                                            context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute all ranking components in parallel"""
        
        tasks = []
        
        # Vector similarity
        if query_vector:
            tasks.append(self._compute_vector_scores_batch(query_vector, documents))
        else:
            tasks.append(self._create_zero_scores(documents))
        
        # BM25 scores
        tasks.append(self.calculate_bm25_scores_mcp(query, documents))
        
        # PageRank scores
        tasks.append(self.calculate_pagerank_scores_mcp(documents))
        
        # Contextual relevance
        tasks.append(self._compute_contextual_scores(documents, context))
        
        results = await asyncio.gather(*tasks)
        
        return {
            'vectorSimilarity': results[0] if isinstance(results[0], dict) else results[0].get('similarities', {}),
            'bm25Score': results[1].get('scores', {}),
            'pageRankScore': results[2].get('scores', {}),
            'contextualRelevance': results[3]
        }
    
    async def _compute_vector_scores_batch(self,
                                         query_vector: List[float],
                                         documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute vector similarities for documents"""
        result = await self.calculate_vector_similarity_mcp(
            query_vector,
            documents,
            normalize=True
        )
        return result.get('similarities', {})
    
    async def _create_zero_scores(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Create zero scores for documents when no vector is available"""
        return {doc.get('docId', str(idx)): 0.0 for idx, doc in enumerate(documents)}
    
    async def _compute_contextual_scores(self,
                                       documents: List[Dict[str, Any]],
                                       context: Dict[str, Any]) -> Dict[str, float]:
        """Compute contextual relevance scores"""
        scores = {}
        
        # Example contextual factors
        user_preferences = context.get('user_preferences', {})
        recency_weight = context.get('recency_weight', 0.0)
        
        for idx, doc in enumerate(documents):
            doc_id = doc.get('docId', str(idx))
            score = 0.0
            
            # Recency scoring
            if recency_weight > 0 and 'timestamp' in doc:
                doc_age_days = (datetime.now() - doc['timestamp']).days
                recency_score = max(0, 1 - (doc_age_days / 365))  # Decay over a year
                score += recency_weight * recency_score
            
            # User preference matching
            if user_preferences:
                doc_categories = set(doc.get('categories', []))
                pref_categories = set(user_preferences.get('preferred_categories', []))
                if doc_categories and pref_categories:
                    overlap = len(doc_categories.intersection(pref_categories))
                    score += overlap / len(pref_categories)
            
            scores[doc_id] = min(1.0, score)
        
        return scores
    
    def _combine_rankings_weighted(self,
                                 documents: List[Dict[str, Any]],
                                 components: Dict[str, Dict[str, float]],
                                 weights: Dict[str, float]) -> Dict[str, float]:
        """Combine ranking components with weights"""
        final_scores = {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            norm_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            norm_weights = weights
        
        for idx, doc in enumerate(documents):
            doc_id = doc.get('docId', str(idx))
            score = 0.0
            
            for component, weight in norm_weights.items():
                component_scores = components.get(component, {})
                component_score = component_scores.get(doc_id, 0.0)
                score += weight * component_score
            
            final_scores[doc_id] = score
        
        return final_scores
    
    def _prepare_ranked_results(self,
                              documents: List[Dict[str, Any]],
                              final_scores: Dict[str, float],
                              components: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Prepare final ranked results with metadata"""
        ranked_results = []
        
        for doc_id, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
            # Find original document
            doc = next((d for d in documents if d.get('docId', str(documents.index(d))) == doc_id), None)
            if doc:
                enhanced_doc = doc.copy()
                enhanced_doc.update({
                    'hybridScore': score,
                    'rankingComponents': {
                        comp: comp_scores.get(doc_id, 0.0)
                        for comp, comp_scores in components.items()
                    },
                    'rank': len(ranked_results) + 1
                })
                ranked_results.append(enhanced_doc)
        
        return ranked_results
    
    async def _apply_diversification(self,
                                   ranked_results: List[Dict[str, Any]],
                                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply result diversification to reduce redundancy"""
        
        if len(ranked_results) <= 1:
            return ranked_results
        
        diversity_threshold = context.get('diversity_threshold', 0.7)
        diversified = [ranked_results[0]]  # Keep top result
        
        for candidate in ranked_results[1:]:
            # Check similarity to already selected documents
            is_diverse = True
            
            for selected in diversified:
                # Simple content-based diversity check
                if 'content' in candidate and 'content' in selected:
                    # You could use the text similarity MCP tool here
                    similarity = self._simple_text_similarity(
                        candidate['content'],
                        selected['content']
                    )
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diversified.append(candidate)
        
        return diversified
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation and filter
        cleaned_tokens = []
        for token in tokens:
            # Remove common punctuation
            cleaned = token.strip('.,!?;:"')
            if len(cleaned) > 2:  # Skip very short tokens
                cleaned_tokens.append(cleaned)
        
        return cleaned_tokens
    
    def _calculate_corpus_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate corpus-level statistics for BM25"""
        total_length = 0
        doc_count = 0
        
        for doc in documents:
            content = doc.get('content', '') or doc.get('text', '')
            if content:
                tokens = self._tokenize_text(content)
                total_length += len(tokens)
                doc_count += 1
        
        avg_length = total_length / doc_count if doc_count > 0 else 0
        
        return {
            'total_documents': len(documents),
            'avg_doc_length': avg_length,
            'total_tokens': total_length
        }
    
    def _get_document_frequency(self, term: str, documents: List[Dict[str, Any]]) -> int:
        """Get document frequency for a term"""
        if term in self.documentFrequencyCache:
            return self.documentFrequencyCache[term]
        
        df = 0
        for doc in documents:
            content = doc.get('content', '') or doc.get('text', '')
            if content:
                tokens = self._tokenize_text(content)
                if term in tokens:
                    df += 1
        
        self.documentFrequencyCache[term] = df
        return df
    
    def _build_link_graph(self, 
                         documents: List[Dict[str, Any]], 
                         relationship_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Build link graph for PageRank"""
        link_graph = defaultdict(list)
        
        for idx, doc in enumerate(documents):
            doc_id = doc.get('docId', str(idx))
            
            # Extract links/references
            links = doc.get('links', [])
            references = doc.get('references', [])
            related = doc.get('related_documents', [])
            
            # Combine all relationships
            all_links = set()
            all_links.update(links)
            all_links.update(references)
            all_links.update(related)
            
            # Filter by relationship types if specified
            if relationship_types:
                # Filter links based on relationship types
                filtered_links = []
                for link in all_links:
                    link_type = link.get('relationship_type', 'unknown')
                    if link_type in relationship_types:
                        filtered_links.append(link)
                all_links = filtered_links
            
            link_graph[doc_id] = list(all_links)
        
        return dict(link_graph)
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for diversification"""
        tokens1 = set(self._tokenize_text(text1))
        tokens2 = set(self._tokenize_text(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0


# Singleton instance
mcp_hybrid_ranking = MCPHybridRankingSkills()