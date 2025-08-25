import asyncio
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from dataclasses import dataclass
from enum import Enum
import re
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from app.a2a.core.security_base import SecureA2AAgent
"""
Enhanced Catalog Management Skills for Production-Ready ORD Operations
Implements advanced metadata enhancement, semantic search, and caching strategies
"""

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK data download failed: {e}")
    NLTK_AVAILABLE = False


class MetadataQuality(Enum):
    """Metadata quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[timedelta] = None
    priority: int = 0
    size_bytes: int = 0


@dataclass
class SemanticSearchResult:
    """Enhanced search result with relevance scoring"""
    document_id: str
    score: float
    highlights: List[str]
    metadata_matches: Dict[str, float]
    semantic_similarity: float
    ranking_factors: Dict[str, float]


class EnhancedORDMetadataProcessor(SecureA2AAgent):
    """Advanced ORD metadata enhancement processor"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self):
        
        super().__init__()
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.metadata_patterns = self._load_metadata_patterns()
        self.enhancement_rules = self._load_enhancement_rules()
        self.quality_thresholds = {
            MetadataQuality.EXCELLENT: 0.9,
            MetadataQuality.GOOD: 0.75,
            MetadataQuality.FAIR: 0.6,
            MetadataQuality.POOR: 0.4,
            MetadataQuality.CRITICAL: 0.0
        }
    
    def _load_metadata_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load regex patterns for metadata extraction"""
        return {
            "version": [
                re.compile(r'v?(\d+)\.(\d+)\.(\d+)(?:-(\w+))?'),
                re.compile(r'version\s*[:=]\s*(["\']?)(\d+\.\d+(?:\.\d+)?)\1', re.IGNORECASE)
            ],
            "api_type": [
                re.compile(r'(REST|SOAP|GraphQL|gRPC|WebSocket)', re.IGNORECASE),
                re.compile(r'api[_\s]?type[:\s]+(\w+)', re.IGNORECASE)
            ],
            "authentication": [
                re.compile(r'(OAuth2?|SAML|JWT|Basic|Bearer|API[_\s]?Key)', re.IGNORECASE),
                re.compile(r'auth(?:entication)?[:\s]+(\w+)', re.IGNORECASE)
            ],
            "protocol": [
                re.compile(r'(HTTP/\d\.\d|HTTPS|WSS?|AMQP|MQTT)', re.IGNORECASE),
                re.compile(r'protocol[:\s]+(\w+)', re.IGNORECASE)
            ],
            "data_format": [
                re.compile(r'(JSON|XML|CSV|Protobuf|MessagePack|Avro)', re.IGNORECASE),
                re.compile(r'(?:data[_\s]?)?format[:\s]+(\w+)', re.IGNORECASE)
            ]
        }
    
    def _load_enhancement_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load metadata enhancement rules"""
        return {
            "title": [
                {"pattern": r'^[a-z]', "action": "capitalize", "priority": 1},
                {"pattern": r'api$', "action": "append", "value": " Service", "priority": 2},
                {"pattern": r'^\w{1,3}$', "action": "expand_acronym", "priority": 3}
            ],
            "description": [
                {"min_length": 50, "action": "enrich_description", "priority": 1},
                {"pattern": r'^\s*$', "action": "generate_description", "priority": 2}
            ],
            "tags": [
                {"action": "extract_keywords", "min_tags": 5, "priority": 1},
                {"action": "add_category_tags", "priority": 2},
                {"action": "add_technology_tags", "priority": 3}
            ]
        }
    
    async def enhance_metadata(self, ord_document: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhance ORD document metadata with advanced algorithms
        Returns: (enhanced_document, enhancement_report)
        """
        enhanced = ord_document.copy()
        report = {
            "original_quality": self._assess_metadata_quality(ord_document),
            "enhancements_applied": [],
            "fields_added": [],
            "fields_enhanced": [],
            "quality_improvement": 0.0
        }
        
        # Extract metadata from content
        extracted_metadata = await self._extract_metadata_from_content(enhanced)
        for field, value in extracted_metadata.items():
            if field not in enhanced or not enhanced[field]:
                enhanced[field] = value
                report["fields_added"].append(field)
        
        # Apply enhancement rules
        for field, rules in self.enhancement_rules.items():
            if field in enhanced:
                original_value = enhanced[field]
                enhanced_value = await self._apply_enhancement_rules(field, original_value, rules)
                if enhanced_value != original_value:
                    enhanced[field] = enhanced_value
                    report["fields_enhanced"].append(field)
                    report["enhancements_applied"].append(f"Enhanced {field}")
        
        # Generate missing metadata
        missing_fields = self._identify_missing_metadata(enhanced)
        for field in missing_fields:
            generated_value = await self._generate_metadata_field(field, enhanced)
            if generated_value:
                enhanced[field] = generated_value
                report["fields_added"].append(field)
        
        # Add semantic tags
        semantic_tags = await self._generate_semantic_tags(enhanced)
        if "tags" in enhanced:
            enhanced["tags"] = list(set(enhanced["tags"] + semantic_tags))
        else:
            enhanced["tags"] = semantic_tags
        
        # Add quality metadata
        enhanced["_metadata_quality"] = {
            "score": self._assess_metadata_quality(enhanced),
            "enhanced_at": datetime.utcnow().isoformat(),
            "enhancement_version": "2.0"
        }
        
        # Calculate quality improvement
        final_quality = self._assess_metadata_quality(enhanced)
        report["final_quality"] = final_quality
        report["quality_improvement"] = final_quality - report["original_quality"]
        
        return enhanced, report
    
    async def _extract_metadata_from_content(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document content using patterns"""
        extracted = {}
        
        # Combine all text fields for analysis
        text_content = " ".join([
            str(v) for v in document.values() 
            if isinstance(v, str) and len(str(v)) > 10
        ])
        
        # Apply extraction patterns
        for metadata_type, patterns in self.metadata_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_content)
                if match:
                    extracted[metadata_type] = match.group(1) if match.groups() else match.group(0)
                    break
        
        # Extract relationships
        relationships = await self._extract_relationships(text_content)
        if relationships:
            extracted["relationships"] = relationships
        
        # Extract capabilities
        capabilities = await self._extract_capabilities(text_content)
        if capabilities:
            extracted["capabilities"] = capabilities
        
        return extracted
    
    async def _generate_semantic_tags(self, document: Dict[str, Any]) -> List[str]:
        """Generate semantic tags using NLP"""
        tags = set()
        
        # Extract from title and description
        text = f"{document.get('title', '')} {document.get('description', '')}"
        
        if NLTK_AVAILABLE and text:
            # Tokenize and POS tag
            tokens = word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract nouns and important terms
            for word, pos in pos_tags:
                if pos in ['NN', 'NNS', 'NNP', 'NNPS'] and len(word) > 3:
                    lemma = self.lemmatizer.lemmatize(word)
                    tags.add(lemma)
            
            # Add technology-specific tags
            tech_keywords = {
                'api', 'service', 'endpoint', 'integration', 'data',
                'cloud', 'microservice', 'rest', 'soap', 'graphql',
                'authentication', 'authorization', 'security'
            }
            
            for keyword in tech_keywords:
                if keyword in text.lower():
                    tags.add(keyword)
        
        # Add category tags based on content
        if 'api' in text.lower():
            tags.add('api-service')
        if 'data' in text.lower():
            tags.add('data-service')
        if 'auth' in text.lower():
            tags.add('security-service')
        
        return list(tags)
    
    def _assess_metadata_quality(self, document: Dict[str, Any]) -> float:
        """Assess the quality of metadata (0.0 to 1.0)"""
        score = 0.0
        weights = {
            "title": 0.15,
            "description": 0.20,
            "version": 0.10,
            "tags": 0.15,
            "api_type": 0.10,
            "authentication": 0.10,
            "documentation_url": 0.10,
            "capabilities": 0.10
        }
        
        # Check required fields
        for field, weight in weights.items():
            if field in document and document[field]:
                field_score = self._assess_field_quality(field, document[field])
                score += field_score * weight
        
        # Bonus for additional metadata
        bonus_fields = ["relationships", "dependencies", "examples", "schemas"]
        bonus_score = sum(0.025 for field in bonus_fields if field in document)
        score += min(bonus_score, 0.1)  # Cap bonus at 0.1
        
        return min(score, 1.0)
    
    def _assess_field_quality(self, field: str, value: Any) -> float:
        """Assess quality of individual field"""
        if not value:
            return 0.0
        
        if field == "title":
            # Good title: 5-50 chars, capitalized, descriptive
            if isinstance(value, str):
                score = 0.5
                if 5 <= len(value) <= 50:
                    score += 0.25
                if value[0].isupper():
                    score += 0.25
                return score
        
        elif field == "description":
            # Good description: 50+ chars, complete sentences
            if isinstance(value, str):
                length_score = min(len(value) / 200, 1.0)  # Full score at 200+ chars
                return length_score
        
        elif field == "tags":
            # Good tags: 5+ relevant tags
            if isinstance(value, list):
                return min(len(value) / 10, 1.0)  # Full score at 10+ tags
        
        elif field == "version":
            # Valid version format
            if isinstance(value, str) and re.match(r'\d+\.\d+', value):
                return 1.0
        
        # Default: field exists = 0.8
        return 0.8


class AdvancedSemanticSearchEngine(SecureA2AAgent):
    """Production-ready semantic search with advanced ranking"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, embedding_model):
        
        super().__init__()
        self.embedding_model = embedding_model
        self.index_cache = {}
        self.query_cache = {}
        self.ranking_weights = {
            "semantic_similarity": 0.4,
            "keyword_match": 0.2,
            "metadata_match": 0.2,
            "freshness": 0.1,
            "popularity": 0.1
        }
    
    async def search(
        self, 
        query: str, 
        documents: Dict[str, Dict[str, Any]], 
        embeddings: Dict[str, np.ndarray],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[SemanticSearchResult]:
        """
        Advanced semantic search with multi-factor ranking
        """
        # Check query cache
        cache_key = hashlib.md5(f"{query}{filters}".encode()).hexdigest()
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if (datetime.utcnow() - cached_result["timestamp"]).seconds < 300:  # 5 min cache
                return cached_result["results"]
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        
        # Calculate similarities
        results = []
        for doc_id, document in documents.items():
            if doc_id not in embeddings:
                continue
            
            # Apply filters
            if filters and not self._apply_filters(document, filters):
                continue
            
            # Calculate semantic similarity
            semantic_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                embeddings[doc_id].reshape(1, -1)
            )[0][0]
            
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_match(query, document)
            
            # Calculate metadata match score
            metadata_score = self._calculate_metadata_match(query, document)
            
            # Calculate freshness score
            freshness_score = self._calculate_freshness_score(document)
            
            # Calculate popularity score (based on access patterns)
            popularity_score = self._calculate_popularity_score(doc_id)
            
            # Combine scores
            ranking_factors = {
                "semantic_similarity": semantic_sim,
                "keyword_match": keyword_score,
                "metadata_match": metadata_score,
                "freshness": freshness_score,
                "popularity": popularity_score
            }
            
            final_score = sum(
                score * self.ranking_weights[factor]
                for factor, score in ranking_factors.items()
            )
            
            # Extract highlights
            highlights = self._extract_highlights(query, document)
            
            results.append(SemanticSearchResult(
                document_id=doc_id,
                score=final_score,
                highlights=highlights,
                metadata_matches={},
                semantic_similarity=semantic_sim,
                ranking_factors=ranking_factors
            ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        top_results = results[:top_k]
        
        # Cache results
        self.query_cache[cache_key] = {
            "results": top_results,
            "timestamp": datetime.utcnow()
        }
        
        return top_results
    
    def _calculate_keyword_match(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate keyword matching score"""
        query_tokens = set(query.lower().split())
        doc_text = f"{document.get('title', '')} {document.get('description', '')}"
        doc_tokens = set(doc_text.lower().split())
        
        if not query_tokens:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_tokens.intersection(doc_tokens)
        union = query_tokens.union(doc_tokens)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_metadata_match(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate metadata field matching score"""
        score = 0.0
        query_lower = query.lower()
        
        # Check specific metadata fields
        metadata_fields = ["tags", "categories", "api_type", "version"]
        for field in metadata_fields:
            if field in document:
                field_value = str(document[field]).lower()
                if query_lower in field_value or field_value in query_lower:
                    score += 0.25
        
        return min(score, 1.0)
    
    def _calculate_freshness_score(self, document: Dict[str, Any]) -> float:
        """Calculate document freshness score"""
        if "updated_at" in document:
            try:
                updated = datetime.fromisoformat(document["updated_at"].replace('Z', '+00:00'))
                age_days = (datetime.utcnow() - updated.replace(tzinfo=None)).days
                # Exponential decay: full score for < 7 days, 0.5 at 30 days
                return np.exp(-age_days / 30)
            except:
                pass
        return 0.5  # Default middle score
    
    def _calculate_popularity_score(self, doc_id: str) -> float:
        """Calculate popularity based on access patterns"""
        # This would typically use real access logs
        # For now, return a default score
        return 0.5
    
    def _extract_highlights(self, query: str, document: Dict[str, Any], max_highlights: int = 3) -> List[str]:
        """Extract relevant text highlights"""
        highlights = []
        query_tokens = query.lower().split()
        
        # Search in description
        description = document.get("description", "")
        if description:
            sentences = description.split('.')
            for sentence in sentences[:max_highlights]:
                if any(token in sentence.lower() for token in query_tokens):
                    highlights.append(sentence.strip())
        
        return highlights


class IntelligentCacheManager(SecureA2AAgent):
    """Advanced caching system with multiple strategies"""
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, max_size_mb: int = 100, default_strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        
        super().__init__()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.strategy = default_strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns = defaultdict(list)  # Track access times
        self.ttl_default = timedelta(hours=1)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with strategy-based management"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if entry.ttl and (datetime.utcnow() - entry.created_at) > entry.ttl:
            await self.invalidate(key)
            return None
        
        # Update access metadata
        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1
        self.access_patterns[key].append(datetime.utcnow())
        
        # Adaptive strategy adjustments
        if self.strategy == CacheStrategy.ADAPTIVE:
            await self._adjust_entry_priority(key, entry)
        
        return entry.value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[timedelta] = None,
        priority: int = 0
    ) -> bool:
        """Set value in cache with intelligent eviction"""
        # Calculate size
        size_bytes = self._estimate_size(value)
        
        # Check if we need to evict
        if self.current_size_bytes + size_bytes > self.max_size_bytes:
            await self._evict_entries(size_bytes)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            ttl=ttl or self.ttl_default,
            priority=priority,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        self.current_size_bytes += size_bytes
        
        return True
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry"""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
            del self.access_patterns[key]
            return True
        return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        import re
        regex = re.compile(pattern)
        keys_to_remove = [k for k in self.cache.keys() if regex.match(k)]
        
        for key in keys_to_remove:
            await self.invalidate(key)
        
        return len(keys_to_remove)
    
    async def _evict_entries(self, required_space: int):
        """Evict entries based on configured strategy"""
        if self.strategy == CacheStrategy.LRU:
            await self._evict_lru(required_space)
        elif self.strategy == CacheStrategy.LFU:
            await self._evict_lfu(required_space)
        elif self.strategy == CacheStrategy.TTL:
            await self._evict_expired()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            await self._evict_adaptive(required_space)
    
    async def _evict_lru(self, required_space: int):
        """Evict least recently used entries"""
        # Sort by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            freed_space += entry.size_bytes
            await self.invalidate(key)
    
    async def _evict_adaptive(self, required_space: int):
        """Adaptive eviction based on access patterns and value"""
        # Calculate eviction scores
        eviction_scores = {}
        
        for key, entry in self.cache.items():
            # Factor 1: Recency (time since last access)
            recency_score = (datetime.utcnow() - entry.last_accessed).total_seconds()
            
            # Factor 2: Frequency (access rate)
            access_rate = entry.access_count / max(
                (datetime.utcnow() - entry.created_at).total_seconds() / 3600, 1
            )
            frequency_score = 1 / (1 + access_rate)
            
            # Factor 3: Size efficiency (value per byte)
            size_score = entry.size_bytes / self.max_size_bytes
            
            # Factor 4: Priority
            priority_score = 1 / (1 + entry.priority)
            
            # Combined score (lower is better for eviction)
            eviction_scores[key] = (
                recency_score * 0.3 +
                frequency_score * 0.3 +
                size_score * 0.2 +
                priority_score * 0.2
            )
        
        # Sort by eviction score
        sorted_keys = sorted(eviction_scores.keys(), key=lambda k: eviction_scores[k], reverse=True)
        
        freed_space = 0
        for key in sorted_keys:
            if freed_space >= required_space:
                break
            entry = self.cache[key]
            freed_space += entry.size_bytes
            await self.invalidate(key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (list, dict)):
            return len(json.dumps(obj).encode('utf-8'))
        elif isinstance(obj, bytes):
            return len(obj)
        else:
            # Rough estimate for other types
            return 1024
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if entry.ttl and (datetime.utcnow() - entry.created_at) > entry.ttl
        )
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "hit_rate": self._calculate_hit_rate(),
            "strategy": self.strategy.value,
            "avg_entry_size_kb": (self.current_size_bytes / max(total_entries, 1)) / 1024
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from recent accesses"""
        # This would typically track hits vs misses
        # For now, estimate based on access patterns
        if not self.access_patterns:
            return 0.0
        
        recent_window = datetime.utcnow() - timedelta(minutes=5)
        recent_accesses = sum(
            len([t for t in times if t > recent_window])
            for times in self.access_patterns.values()
        )
        
        return min(recent_accesses / (len(self.cache) * 2), 1.0)  # Rough estimate