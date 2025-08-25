import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import json
import hashlib
from pathlib import Path

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task
from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
from app.a2a.core.trustIdentity import TrustIdentity
from app.a2a.core.dataValidation import DataValidator


from app.a2a.core.security_base import SecureA2AAgent
"""
Semantic Chunking and Hierarchical Embeddings Skills for Agent 2 - Phase 2 Advanced Features
Implements intelligent document chunking with semantic understanding and multi-level embeddings
"""

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TOPIC_MODELING = "topic_modeling"
    HIERARCHICAL_STRUCTURE = "hierarchical_structure"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE_BOUNDARY = "adaptive_boundary"


@dataclass
class ChunkMetadata:
    """Metadata for each semantic chunk"""
    chunk_id: str
    start_position: int
    end_position: int
    level: int  # Hierarchical level (0=document, 1=section, 2=paragraph, etc.)
    parent_chunk_id: Optional[str]
    child_chunk_ids: List[str] = field(default_factory=list)
    semantic_coherence_score: float = 0.0
    topic_keywords: List[str] = field(default_factory=list)
    entity_mentions: List[Dict[str, Any]] = field(default_factory=list)
    linguistic_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalEmbedding:
    """Multi-level embedding representation"""
    chunk_id: str
    embeddings: Dict[int, np.ndarray]  # level -> embedding
    aggregation_method: str
    confidence_scores: Dict[int, float]
    semantic_fingerprint: str


class SemanticChunkingSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Real A2A agent skills for semantic chunking and hierarchical embeddings
    Provides intelligent document segmentation and multi-level semantic representation
    """

    def __init__(self, trust_identity: TrustIdentity):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataValidator()
        
        # Initialize NLP models
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("spaCy model not found. Installing...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Chunking parameters
        self.chunking_configs = {
            'min_chunk_size': 50,
            'max_chunk_size': 1000,
            'semantic_threshold': 0.75,
            'overlap_ratio': 0.1,
            'hierarchy_levels': 4
        }
        
        # Performance tracking
        self.chunking_metrics = {
            'documents_processed': 0,
            'chunks_created': 0,
            'average_chunk_size': 0,
            'semantic_coherence_scores': [],
            'processing_times': []
        }

    @a2a_skill(
        name="performSemanticChunking",
        description="Chunk document using semantic understanding and coherence",
        input_schema={
            "type": "object",
            "properties": {
                "document_text": {"type": "string"},
                "document_id": {"type": "string"},
                "chunking_strategy": {
                    "type": "string",
                    "enum": ["semantic_similarity", "topic_modeling", "hierarchical_structure", "sliding_window", "adaptive_boundary"],
                    "default": "semantic_similarity"
                },
                "config": {
                    "type": "object",
                    "properties": {
                        "min_chunk_size": {"type": "integer", "default": 50},
                        "max_chunk_size": {"type": "integer", "default": 1000},
                        "semantic_threshold": {"type": "number", "default": 0.75},
                        "overlap_ratio": {"type": "number", "default": 0.1}
                    }
                }
            },
            "required": ["document_text", "document_id"]
        }
    )
    def perform_semantic_chunking(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic chunking on document text"""
        try:
            document_text = request_data["document_text"]
            document_id = request_data["document_id"]
            strategy = ChunkingStrategy(request_data.get("chunking_strategy", "semantic_similarity"))
            config = request_data.get("config", {})
            
            # Update configuration
            current_config = self.chunking_configs.copy()
            current_config.update(config)
            
            # Preprocess document
            preprocessed_text = self._preprocess_document(document_text)
            
            # Extract sentences for analysis
            sentences = self._extract_sentences(preprocessed_text)
            if len(sentences) < 2:
                return {
                    'success': False,
                    'error': 'Document too short for semantic chunking',
                    'error_type': 'insufficient_content'
                }
            
            # Perform chunking based on strategy
            chunks_data = []
            if strategy == ChunkingStrategy.SEMANTIC_SIMILARITY:
                chunks_data = self._chunk_by_semantic_similarity(sentences, current_config)
            elif strategy == ChunkingStrategy.TOPIC_MODELING:
                chunks_data = self._chunk_by_topic_modeling(sentences, current_config)
            elif strategy == ChunkingStrategy.HIERARCHICAL_STRUCTURE:
                chunks_data = self._chunk_by_hierarchical_structure(document_text, current_config)
            elif strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunks_data = self._chunk_by_sliding_window(sentences, current_config)
            elif strategy == ChunkingStrategy.ADAPTIVE_BOUNDARY:
                chunks_data = self._chunk_by_adaptive_boundary(sentences, current_config)
            
            # Create chunk metadata
            semantic_chunks = []
            for i, chunk_data in enumerate(chunks_data):
                chunk_id = f"{document_id}_chunk_{i:04d}"
                
                # Calculate semantic coherence
                coherence_score = self._calculate_semantic_coherence(chunk_data['sentences'])
                
                # Extract entities and keywords
                entities = self._extract_entities(chunk_data['text'])
                keywords = self._extract_keywords(chunk_data['text'])
                
                # Create metadata
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    start_position=chunk_data['start_pos'],
                    end_position=chunk_data['end_pos'],
                    level=chunk_data.get('level', 1),
                    parent_chunk_id=chunk_data.get('parent_id'),
                    semantic_coherence_score=coherence_score,
                    topic_keywords=keywords,
                    entity_mentions=entities,
                    linguistic_features={
                        'sentence_count': len(chunk_data['sentences']),
                        'word_count': len(chunk_data['text'].split()),
                        'avg_sentence_length': np.mean([len(s.split()) for s in chunk_data['sentences']]),
                        'lexical_diversity': len(set(chunk_data['text'].lower().split())) / len(chunk_data['text'].split())
                    }
                )
                
                semantic_chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_data['text'],
                    'metadata': metadata.__dict__,
                    'sentences': chunk_data['sentences']
                })
            
            # Update metrics
            self.chunking_metrics['documents_processed'] += 1
            self.chunking_metrics['chunks_created'] += len(semantic_chunks)
            self.chunking_metrics['average_chunk_size'] = np.mean([len(chunk['text']) for chunk in semantic_chunks])
            self.chunking_metrics['semantic_coherence_scores'].extend([chunk['metadata']['semantic_coherence_score'] for chunk in semantic_chunks])
            
            self.logger.info(f"Created {len(semantic_chunks)} semantic chunks for document {document_id}")
            
            return {
                'success': True,
                'document_id': document_id,
                'chunking_strategy': strategy.value,
                'semantic_chunks': semantic_chunks,
                'chunk_statistics': {
                    'total_chunks': len(semantic_chunks),
                    'average_chunk_size': int(self.chunking_metrics['average_chunk_size']),
                    'average_coherence_score': float(np.mean([chunk['metadata']['semantic_coherence_score'] for chunk in semantic_chunks])),
                    'total_entities_found': sum(len(chunk['metadata']['entity_mentions']) for chunk in semantic_chunks),
                    'unique_keywords': len(set([kw for chunk in semantic_chunks for kw in chunk['metadata']['topic_keywords']]))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Semantic chunking failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'semantic_chunking_error'
            }

    @a2a_skill(
        name="generateHierarchicalEmbeddings",
        description="Generate multi-level hierarchical embeddings for document chunks",
        input_schema={
            "type": "object",
            "properties": {
                "semantic_chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "text": {"type": "string"},
                            "metadata": {"type": "object"},
                            "sentences": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "hierarchy_levels": {"type": "integer", "default": 3},
                "aggregation_methods": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["mean", "weighted_mean", "attention", "max_pooling"]
                    },
                    "default": ["mean", "weighted_mean", "attention"]
                }
            },
            "required": ["semantic_chunks"]
        }
    )
    def generate_hierarchical_embeddings(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hierarchical embeddings at multiple levels of granularity"""
        try:
            semantic_chunks = request_data["semantic_chunks"]
            hierarchy_levels = request_data.get("hierarchy_levels", 3)
            aggregation_methods = request_data.get("aggregation_methods", ["mean", "weighted_mean", "attention"])
            
            hierarchical_embeddings = []
            
            for chunk in semantic_chunks:
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]
                sentences = chunk["sentences"]
                metadata = chunk["metadata"]
                
                # Generate embeddings at different levels
                embeddings_by_level = {}
                confidence_scores = {}
                
                # Level 0: Full chunk embedding (document-level)
                chunk_embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                embeddings_by_level[0] = chunk_embedding
                confidence_scores[0] = float(metadata.get('semantic_coherence_score', 0.5))
                
                # Level 1: Sentence-level embeddings aggregated
                if sentences and len(sentences) > 1:
                    sentence_embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
                    
                    for method in aggregation_methods:
                        level_key = f"1_{method}"
                        if method == "mean":
                            aggregated = np.mean(sentence_embeddings, axis=0)
                        elif method == "weighted_mean":
                            # Weight by sentence length and position
                            weights = np.array([len(s.split()) / sum(len(sent.split()) for sent in sentences) for s in sentences])
                            weights = weights * np.linspace(1.0, 0.8, len(sentences))  # Slight position weighting
                            aggregated = np.average(sentence_embeddings, axis=0, weights=weights)
                        elif method == "attention":
                            # Simple attention mechanism
                            similarities = cosine_similarity(sentence_embeddings)
                            attention_weights = np.mean(similarities, axis=1)
                            attention_weights = attention_weights / np.sum(attention_weights)
                            aggregated = np.average(sentence_embeddings, axis=0, weights=attention_weights)
                        elif method == "max_pooling":
                            aggregated = np.max(sentence_embeddings, axis=0)
                        
                        embeddings_by_level[level_key] = aggregated
                        confidence_scores[level_key] = float(np.mean(cosine_similarity([aggregated], sentence_embeddings)[0]))
                
                # Level 2: Sub-sentence level (if chunk is large enough)
                if len(text) > 200:
                    # Split into sub-sentences or phrases
                    phrases = self._extract_phrases(text)
                    if len(phrases) > 2:
                        phrase_embeddings = self.sentence_model.encode(phrases, normalize_embeddings=True)
                        phrase_aggregated = np.mean(phrase_embeddings, axis=0)
                        embeddings_by_level[2] = phrase_aggregated
                        confidence_scores[2] = float(np.mean(cosine_similarity([phrase_aggregated], phrase_embeddings)[0]))
                
                # Generate semantic fingerprint
                fingerprint_text = f"{chunk_id}_{text[:100]}_{'_'.join(metadata.get('topic_keywords', []))}"
                semantic_fingerprint = hashlib.md5(fingerprint_text.encode()).hexdigest()[:16]
                
                # Create hierarchical embedding object
                hierarchical_embedding = HierarchicalEmbedding(
                    chunk_id=chunk_id,
                    embeddings=embeddings_by_level,
                    aggregation_method='multi_level',
                    confidence_scores=confidence_scores,
                    semantic_fingerprint=semantic_fingerprint
                )
                
                # Convert numpy arrays to lists for JSON serialization
                serializable_embeddings = {}
                for level, embedding in embeddings_by_level.items():
                    serializable_embeddings[str(level)] = embedding.tolist()
                
                hierarchical_embeddings.append({
                    'chunk_id': chunk_id,
                    'embeddings': serializable_embeddings,
                    'aggregation_method': 'multi_level',
                    'confidence_scores': {str(k): v for k, v in confidence_scores.items()},
                    'semantic_fingerprint': semantic_fingerprint,
                    'embedding_dimensions': {str(k): len(v) for k, v in embeddings_by_level.items()},
                    'hierarchy_info': {
                        'max_level': max([int(str(k).split('_')[0]) for k in embeddings_by_level.keys()]),
                        'aggregation_methods_used': aggregation_methods,
                        'sentence_count': len(sentences),
                        'phrase_count': len(phrases) if len(text) > 200 else 0
                    }
                })
            
            # Calculate overall statistics
            total_embeddings = sum(len(he['embeddings']) for he in hierarchical_embeddings)
            avg_confidence = np.mean([np.mean(list(he['confidence_scores'].values())) for he in hierarchical_embeddings])
            
            self.logger.info(f"Generated hierarchical embeddings for {len(semantic_chunks)} chunks with {total_embeddings} total embeddings")
            
            return {
                'success': True,
                'hierarchical_embeddings': hierarchical_embeddings,
                'embedding_statistics': {
                    'total_chunks_processed': len(semantic_chunks),
                    'total_embeddings_generated': total_embeddings,
                    'average_confidence_score': float(avg_confidence),
                    'hierarchy_levels_used': hierarchy_levels,
                    'aggregation_methods': aggregation_methods,
                    'embedding_models_used': ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Hierarchical embedding generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'hierarchical_embedding_error'
            }

    def _preprocess_document(self, text: str) -> str:
        """Preprocess document text for chunking"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.!?])', r'\1', text)
        return text.strip()

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences using spaCy"""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        return sentences

    def _chunk_by_semantic_similarity(self, sentences: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk sentences based on semantic similarity"""
        if len(sentences) <= 2:
            return [{
                'text': ' '.join(sentences),
                'sentences': sentences,
                'start_pos': 0,
                'end_pos': len(' '.join(sentences)),
                'level': 1
            }]
        
        # Generate sentence embeddings
        embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find natural breakpoints where similarity drops
        similarity_scores = []
        for i in range(len(sentences) - 1):
            similarity_scores.append(similarity_matrix[i][i + 1])
        
        # Find breakpoints
        threshold = config['semantic_threshold']
        breakpoints = [0]
        
        for i, score in enumerate(similarity_scores):
            if score < threshold:
                breakpoints.append(i + 1)
        
        breakpoints.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append({
                'text': chunk_text,
                'sentences': chunk_sentences,
                'start_pos': sum(len(s) + 1 for s in sentences[:start_idx]),
                'end_pos': sum(len(s) + 1 for s in sentences[:end_idx]),
                'level': 1
            })
        
        return chunks

    def _chunk_by_topic_modeling(self, sentences: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk based on topic coherence using clustering"""
        if len(sentences) <= 3:
            return self._chunk_by_semantic_similarity(sentences, config)
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        
        # Perform hierarchical clustering
        n_clusters = min(max(2, len(sentences) // 3), 8)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((i, sentences[i]))
        
        # Create chunks from clusters, maintaining order
        chunks = []
        for cluster_id in sorted(clusters.keys()):
            cluster_sentences = [sent for _, sent in sorted(clusters[cluster_id], key=lambda x: x[0])]
            chunk_text = ' '.join(cluster_sentences)
            
            start_idx = min(idx for idx, _ in clusters[cluster_id])
            end_idx = max(idx for idx, _ in clusters[cluster_id]) + 1
            
            chunks.append({
                'text': chunk_text,
                'sentences': cluster_sentences,
                'start_pos': sum(len(s) + 1 for s in sentences[:start_idx]),
                'end_pos': sum(len(s) + 1 for s in sentences[:end_idx]),
                'level': 1,
                'topic_cluster': int(cluster_id)
            })
        
        return chunks

    def _chunk_by_hierarchical_structure(self, text: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk based on document structure (headers, paragraphs, etc.)"""
        # Detect structural elements
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_pos = 0
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < config['min_chunk_size']:
                continue
            
            sentences = self._extract_sentences(paragraph)
            if not sentences:
                continue
            
            # Determine level based on content patterns
            level = 1
            if re.match(r'^#{1,6}\s', paragraph) or paragraph.isupper():
                level = 0  # Header
            elif len(paragraph) > config['max_chunk_size']:
                level = 2  # Large paragraph - needs sub-chunking
            
            chunks.append({
                'text': paragraph.strip(),
                'sentences': sentences,
                'start_pos': current_pos,
                'end_pos': current_pos + len(paragraph),
                'level': level,
                'paragraph_index': i
            })
            
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return chunks

    def _chunk_by_sliding_window(self, sentences: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk using sliding window with semantic-aware overlap"""
        chunks = []
        window_size = max(3, config['max_chunk_size'] // 100)  # Approximate sentences per chunk
        overlap_size = max(1, int(window_size * config['overlap_ratio']))
        
        i = 0
        while i < len(sentences):
            end_idx = min(i + window_size, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append({
                'text': chunk_text,
                'sentences': chunk_sentences,
                'start_pos': sum(len(s) + 1 for s in sentences[:i]),
                'end_pos': sum(len(s) + 1 for s in sentences[:end_idx]),
                'level': 1,
                'window_start': i,
                'window_end': end_idx
            })
            
            i += window_size - overlap_size
        
        return chunks

    def _chunk_by_adaptive_boundary(self, sentences: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adaptive chunking that adjusts boundaries based on content complexity"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_complexity = self._calculate_sentence_complexity(sentence)
            adjusted_size = len(sentence) * (1 + sentence_complexity)  # Weight by complexity
            
            if (current_size + adjusted_size > config['max_chunk_size'] and 
                current_size >= config['min_chunk_size'] and 
                len(current_chunk) > 0):
                
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'sentences': current_chunk.copy(),
                    'start_pos': sum(len(s) + 1 for s in sentences[:i - len(current_chunk)]),
                    'end_pos': sum(len(s) + 1 for s in sentences[:i]),
                    'level': 1,
                    'adaptive_complexity': float(np.mean([self._calculate_sentence_complexity(s) for s in current_chunk]))
                })
                
                current_chunk = [sentence]
                current_size = adjusted_size
            else:
                current_chunk.append(sentence)
                current_size += adjusted_size
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'sentences': current_chunk,
                'start_pos': sum(len(s) + 1 for s in sentences[:len(sentences) - len(current_chunk)]),
                'end_pos': sum(len(s) + 1 for s in sentences),
                'level': 1,
                'adaptive_complexity': float(np.mean([self._calculate_sentence_complexity(s) for s in current_chunk]))
            })
        
        return chunks

    def _calculate_sentence_complexity(self, sentence: str) -> float:
        """Calculate sentence complexity based on linguistic features"""
        doc = self.nlp(sentence)
        
        # Feature calculations
        word_count = len([token for token in doc if not token.is_punct])
        avg_word_length = np.mean([len(token.text) for token in doc if not token.is_punct]) if word_count > 0 else 0
        named_entities = len(doc.ents)
        dependency_depth = max([token.depth for token in doc]) if doc else 0
        
        # Normalize and combine features
        complexity = (
            min(word_count / 20.0, 1.0) * 0.3 +  # Word count factor
            min(avg_word_length / 10.0, 1.0) * 0.2 +  # Word length factor
            min(named_entities / 5.0, 1.0) * 0.3 +  # Entity density factor
            min(dependency_depth / 10.0, 1.0) * 0.2  # Syntactic complexity
        )
        
        return complexity

    def _calculate_semantic_coherence(self, sentences: List[str]) -> float:
        """Calculate semantic coherence score for a chunk"""
        if len(sentences) <= 1:
            return 1.0
        
        embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        similarities = cosine_similarity(embeddings)
        
        # Calculate average pairwise similarity
        coherence_scores = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                coherence_scores.append(similarities[i][j])
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms and phrases from text"""
        doc = self.nlp(text)
        
        # Extract key terms based on POS tags and frequency
        keywords = []
        word_freq = {}
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                len(token.text) > 2):
                
                lemma = token.lemma_.lower()
                word_freq[lemma] = word_freq.get(lemma, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun phrases
        for np in doc.noun_chunks:
            if len(np.text.strip()) > 3:
                phrases.append(np.text.strip())
        
        # Extract clauses by splitting on conjunctions and punctuation
        sentences = [sent.text.strip() for sent in doc.sents]
        for sentence in sentences:
            # Split on common conjunctions
            sub_phrases = re.split(r'\b(?:and|but|or|because|while|although|if|when|where|which|that)\b', sentence)
            for phrase in sub_phrases:
                phrase = phrase.strip()
                if len(phrase) > 10 and len(phrase) < 100:
                    phrases.append(phrase)
        
        return list(set(phrases))  # Remove duplicates