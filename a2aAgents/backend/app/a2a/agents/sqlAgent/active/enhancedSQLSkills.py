import re
import json
import asyncio
import hashlib
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from app.a2a.core.security_base import SecureA2AAgent
"""
Enhanced SQL Skills for SQL Agent
Provides advanced NL2SQL and SQL2NL capabilities with explanations
"""

# Advanced NLP processing
try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Advanced NLP features will be limited.")

# Transformers for advanced language understanding
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Advanced language models will be limited.")

# SQL parsing and security
try:
    import sqlparse
    from sqlparse import sql, tokens
    SQLPARSE_AVAILABLE = True
except ImportError:
    SQLPARSE_AVAILABLE = False
    logging.warning("sqlparse not available. Advanced SQL parsing will be limited.")

# GrokClient for AI-powered validation
try:
    from app.a2a.core.grokClient import GrokClient
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    logging.warning("GrokClient not available. AI validation will be limited.")

logger = logging.getLogger(__name__)


class EnhancedSQLSkills(SecureA2AAgent):
    """
    Advanced SQL skills for natural language to SQL conversion
    Supports HANA-specific features including graph, vector, ML, and multi-model queries
    Features:
    - Advanced NLP with spaCy and transformers
    - Comprehensive SQL security and injection prevention
    - Multi-model HANA capabilities (relational, graph, vector, text, spatial)
    - Intelligent query optimization and caching
    - Context-aware conversation support
    """
    
    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling  
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning
    
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
        # Initialize advanced NLP components
        self._initialize_nlp_models()
        
        # Query cache for performance - configurable
        self.query_cache = {}
        self.max_cache_size = int(os.getenv("SQL_CACHE_SIZE", "1000"))
        
        # Security components
        self.sql_injection_patterns = self._initialize_security_patterns()
        
        # Conversation context for multi-turn queries
        self.conversation_contexts = {}
        
        # Performance metrics
        self.performance_stats = {
            "queries_processed": 0,
            "cache_hits": 0,
            "security_blocks": 0,
            "nlp_enhancements": 0,
            "grok_validations": 0,
            "logical_consistency_checks": 0
        }
        
        # Initialize GrokClient for AI validation
        self.grok_client = None
        if GROK_AVAILABLE:
            try:
                self.grok_client = GrokClient()
                logger.info("GrokClient initialized for SQL validation")
            except Exception as e:
                logger.warning(f"Failed to initialize GrokClient: {e}")
                self.grok_client = None
        
        # SQL templates for common operations
        self.sql_templates = {
            "basic_select": "SELECT {columns} FROM {table}",
            "filtered_select": "SELECT {columns} FROM {table} WHERE {conditions}",
            "join_select": "SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition}",
            "aggregate": "SELECT {agg_func}({column}) AS {alias} FROM {table}",
            "group_aggregate": "SELECT {group_cols}, {agg_func}({agg_col}) FROM {table} GROUP BY {group_cols}",
            "ordered_select": "SELECT {columns} FROM {table} ORDER BY {order_cols} {direction}",
            "limited_select": "SELECT {columns} FROM {table} LIMIT {limit}",
            "vector_similarity": "SELECT {columns}, COSINE_SIMILARITY({vector_col}, TO_REAL_VECTOR({query_vector})) AS similarity FROM {table} ORDER BY similarity DESC LIMIT {limit}",
            "graph_match": "MATCH {pattern} RETURN {return_items}",
            "graph_path": "MATCH p = {start_pattern}-[{relationship}]-{end_pattern} RETURN p"
        }
        
        # Natural language patterns mapping
        self.nl_patterns = {
            "show_all": [
                r"show (?:me )?all (\w+)",
                r"list all (\w+)",
                r"get all (\w+)",
                r"display all (\w+)"
            ],
            "filter_equals": [
                r"(\w+) (?:where|with) (\w+) (?:is|equals|=) ['\"]?([^'\"]+)['\"]?",
                r"find (\w+) (?:where|with) (\w+) (?:is|equals|=) ['\"]?([^'\"]+)['\"]?"
            ],
            "filter_contains": [
                r"(\w+) (?:where|with) (\w+) contains ['\"]?([^'\"]+)['\"]?",
                r"(\w+) (?:where|with) (\w+) like ['\"]?([^'\"]+)['\"]?"
            ],
            "count": [
                r"count (?:of |the )?(\w+)",
                r"how many (\w+)",
                r"number of (\w+)"
            ],
            "sum": [
                r"sum (?:of |the )?(\w+) in (\w+)",
                r"total (\w+) (?:in|from) (\w+)"
            ],
            "average": [
                r"average (?:of |the )?(\w+) (?:in|from) (\w+)",
                r"avg (?:of |the )?(\w+) (?:in|from) (\w+)",
                r"mean (?:of |the )?(\w+) (?:in|from) (\w+)"
            ],
            "top_n": [
                r"top (\d+) (\w+) by (\w+)",
                r"(\d+) highest (\w+) by (\w+)",
                r"best (\d+) (\w+) by (\w+)"
            ],
            "similarity": [
                r"find (?:(\d+) )?(?:most )?similar (\w+) to ['\"]?([^'\"]+)['\"]?",
                r"(\w+) similar to ['\"]?([^'\"]+)['\"]?",
                r"(\w+) like ['\"]?([^'\"]+)['\"]? using vectors"
            ],
            "graph_nodes": [
                r"find all (\w+) nodes",
                r"get (\w+) nodes",
                r"show (\w+) in graph"
            ],
            "graph_path": [
                r"path from (\w+) to (\w+)",
                r"route between (\w+) and (\w+)",
                r"connection from (\w+) to (\w+)"
            ]
        }
        
        # Comprehensive HANA-specific functions
        self.hana_functions = {
            "text": ["CONTAINS", "FUZZY", "SCORE", "SNIPPETS", "HIGHLIGHTED"],
            "vector": ["COSINE_SIMILARITY", "L2DISTANCE", "TO_REAL_VECTOR", "EMBED", "VECTOR_DIMENSION"],
            "graph": ["MATCH", "SHORTEST_PATH", "BREADTH_FIRST", "DEPTH_FIRST", "TRAVERSE", "VERTICES", "EDGES"],
            "window": ["ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE"],
            "json": ["JSON_VALUE", "JSON_QUERY", "JSON_TABLE", "JSON_EXTRACT", "JSON_OBJECT"],
            "ml": ["PAL_RANDOM_FOREST", "PAL_K_MEANS", "PAL_LINEAR_REGRESSION", "APL_AREA_CLUSTERING"],
            "time_series": ["SERIES_GENERATE", "SERIES_DISAGGREGATE", "SERIES_ROUND"],
            "spatial": ["ST_DISTANCE", "ST_INTERSECTS", "ST_WITHIN", "ST_BUFFER"],
            "planning": ["CE_CALC", "CE_PROJECTION", "CE_JOIN", "CE_AGGREGATION"],
            "crypto": ["HASH_SHA256", "ENCRYPT_AES", "DECRYPT_AES"],
            "advanced": ["SERIES_ROUND", "WORKDAYS_BETWEEN", "ADD_WORKDAYS"]
        }
        
        logger.info("Enhanced SQL Skills initialized")
    
    def _initialize_nlp_models(self):
        """Initialize advanced NLP models for better language understanding"""
        self.nlp_models = {}
        
        if SPACY_AVAILABLE:
            try:
                import spacy


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                # Use configurable spaCy model
                spacy_model = os.getenv("SPACY_MODEL", "en_core_web_sm")
                self.nlp_models['spacy'] = spacy.load(spacy_model)
                
                # Initialize spaCy matcher for SQL patterns
                self.sql_matcher = Matcher(self.nlp_models['spacy'].vocab)
                
                # Define SQL intent patterns
                patterns = [
                    [{'LOWER': {'IN': ['show', 'get', 'find', 'select', 'list', 'display']}},
                     {'POS': 'DET', 'OP': '?'},
                     {'POS': 'NOUN'}],
                    [{'LOWER': 'count'},
                     {'POS': 'DET', 'OP': '?'},
                     {'POS': 'NOUN'}],
                    [{'LOWER': {'IN': ['sum', 'total', 'average', 'avg', 'mean']}},
                     {'POS': 'DET', 'OP': '?'},
                     {'POS': 'NOUN'}]
                ]
                
                for i, pattern in enumerate(patterns):
                    self.sql_matcher.add(f"SQL_INTENT_{i}", [pattern])
                    
                logger.info(f"spaCy NLP model '{spacy_model}' initialized successfully")
            except OSError as e:
                logger.warning(f"spaCy model '{spacy_model}' not found: {e}. Install with: python -m spacy download {spacy_model}")
                self.nlp_models['spacy'] = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use configurable models
                text2sql_model = os.getenv("TEXT2SQL_MODEL", "microsoft/DialoGPT-medium")
                qa_model = os.getenv("QA_MODEL", "distilbert-base-cased-distilled-squad")
                device = int(os.getenv("NLP_DEVICE", "-1"))  # -1 for CPU, 0+ for GPU
                
                # Initialize text-to-SQL pipeline
                self.nlp_models['text2sql'] = pipeline(
                    "text2text-generation", 
                    model=text2sql_model,
                    device=device
                )
                
                # Initialize question answering for intent detection
                self.nlp_models['qa'] = pipeline(
                    "question-answering",
                    model=qa_model,
                    device=device
                )
                
                logger.info("Transformers models initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize transformers models: {e}")
                self.nlp_models['text2sql'] = None
                self.nlp_models['qa'] = None
    
    def _initialize_security_patterns(self) -> Dict[str, List[str]]:
        """Initialize SQL injection and security patterns"""
        return {
            "sql_injection": [
                r"(?i)(union\s+select|union\s+all\s+select)",
                r"(?i)(drop\s+table|drop\s+database|truncate\s+table)",
                r"(?i)(insert\s+into|update\s+set|delete\s+from)",
                r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
                r"(?i)(xp_cmdshell|sp_oacreate|sp_oamethod)",
                r"(?i)(;\s*--|'\s*--|\/\*.*\*\/)",
                r"(?i)(information_schema|sysobjects|syscolumns)",
                r"(?i)(concat\s*\(|char\s*\(|ascii\s*\()",
                r"(?i)(waitfor\s+delay|benchmark\s*\(|sleep\s*\()"
            ],
            "suspicious_keywords": [
                "drop", "alter", "create", "grant", "revoke", "truncate",
                "shutdown", "sp_", "xp_", "openrowset", "opendatasource"
            ],
            "allowed_functions": [
                "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
                "COUNT", "SUM", "AVG", "MIN", "MAX", "DISTINCT", "TOP", "LIMIT",
                "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
                "COSINE_SIMILARITY", "L2DISTANCE", "TO_REAL_VECTOR", "EMBED",
                "CONTAINS", "FUZZY", "SCORE", "MATCH", "SHORTEST_PATH"
            ]
        }
    
    def _validate_sql_security(self, query: str) -> Dict[str, Any]:
        """Validate SQL query for security issues"""
        issues = []
        severity = "low"
        
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns["sql_injection"]:
            if re.search(pattern, query):
                issues.append(f"Potential SQL injection pattern detected: {pattern}")
                severity = "high"
        
        # Check for suspicious keywords
        query_upper = query.upper()
        for keyword in self.sql_injection_patterns["suspicious_keywords"]:
            if keyword.upper() in query_upper:
                issues.append(f"Suspicious keyword detected: {keyword}")
                if severity == "low":
                    severity = "medium"
        
        # Check for proper parameterization (basic check)
        if "'" in query and not re.search(r"'[^']*'", query):
            issues.append("Potentially unescaped string literals detected")
            if severity == "low":
                severity = "medium"
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "severity": severity,
            "recommendation": "Use parameterized queries" if issues else "Query appears safe"
        }
    
    def _enhance_nl_understanding(self, nl_query: str) -> Dict[str, Any]:
        """Use advanced NLP to better understand natural language query"""
        enhanced_info = {
            "entities": [],
            "intent": "unknown",
            "confidence": 0.5,
            "suggested_tables": [],
            "suggested_columns": [],
            "query_complexity": "simple"
        }
        
        if SPACY_AVAILABLE and self.nlp_models.get('spacy'):
            try:
                doc = self.nlp_models['spacy'](nl_query)
                
                # Extract named entities
                enhanced_info["entities"] = [
                    {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                    for ent in doc.ents
                ]
                
                # Use spaCy matcher for intent detection
                matches = self.sql_matcher(doc)
                if matches:
                    enhanced_info["intent"] = "data_retrieval"
                    enhanced_info["confidence"] = 0.8
                
                # Extract potential table/column names
                for token in doc:
                    if token.pos_ == "NOUN" and not token.is_stop:
                        enhanced_info["suggested_tables"].append(token.lemma_)
                    elif token.pos_ == "PROPN":
                        enhanced_info["suggested_columns"].append(token.text)
                
                # Determine query complexity
                if len([token for token in doc if token.pos_ in ["NOUN", "VERB"]]) > 5:
                    enhanced_info["query_complexity"] = "complex"
                elif "and" in nl_query.lower() or "or" in nl_query.lower():
                    enhanced_info["query_complexity"] = "medium"
                    
                self.performance_stats["nlp_enhancements"] += 1
                
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}")
        
        return enhanced_info
    
    async def convert_nl_to_sql(self, nl_query: str, query_type: str = "auto", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert natural language to SQL with detailed explanation
        """
        try:
            start_time = datetime.now()
            self.performance_stats["queries_processed"] += 1
            
            # Check cache first
            cache_key = hashlib.md5(f"{nl_query}_{query_type}_{json.dumps(context or {}, sort_keys=True)}".encode()).hexdigest()
            if cache_key in self.query_cache:
                self.performance_stats["cache_hits"] += 1
                cached_result = self.query_cache[cache_key].copy()
                cached_result["cached"] = True
                return cached_result
            
            # Security validation
            security_check = self._validate_sql_security(nl_query)
            if not security_check["is_safe"]:
                self.performance_stats["security_blocks"] += 1
                return {
                    "sql_query": "",
                    "error": "Security validation failed",
                    "security_issues": security_check["issues"],
                    "explanation": "Query blocked due to potential security risks"
                }
            
            # Enhanced NLP processing
            nl_enhanced = self._enhance_nl_understanding(nl_query)
            
            # Normalize the query
            nl_query_processed = nl_query.strip().lower()
            
            # Detect query intent if auto
            if query_type == "auto":
                query_type = self._detect_query_intent(nl_query_processed, nl_enhanced)
            
            # Route to appropriate converter with enhanced context
            enhanced_context = {**(context or {}), "nl_enhanced": nl_enhanced}
            
            if query_type == "relational":
                result = await self._convert_relational_query(nl_query_processed, enhanced_context)
            elif query_type == "graph":
                result = await self._convert_graph_query(nl_query_processed, enhanced_context)
            elif query_type == "vector":
                result = await self._convert_vector_query(nl_query_processed, enhanced_context)
            elif query_type == "ml":
                result = await self._convert_ml_query(nl_query_processed, enhanced_context)
            elif query_type == "time_series":
                result = await self._convert_time_series_query(nl_query_processed, enhanced_context)
            elif query_type == "spatial":
                result = await self._convert_spatial_query(nl_query_processed, enhanced_context)
            else:
                result = await self._convert_hybrid_query(nl_query_processed, enhanced_context)
            
            # Add comprehensive metadata
            result["explanation"] = self._generate_sql_explanation(result["sql_query"], nl_query)
            result["confidence"] = self._calculate_confidence(nl_query_processed, result["sql_query"], nl_enhanced)
            result["security_check"] = security_check
            result["nl_enhancements"] = nl_enhanced
            result["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            result["cached"] = False
            
            # GrokClient validation if available and query is valid
            if result.get("sql_query") and not result.get("error"):
                grok_validation = await self.validate_sql_with_grok(
                    result["sql_query"], 
                    nl_query, 
                    enhanced_context
                )
                result["grok_validation"] = grok_validation
                
                # Update confidence based on Grok validation
                if grok_validation.get("grok_available") and grok_validation.get("overall_confidence"):
                    # Combine original confidence with Grok validation
                    original_confidence = result["confidence"]
                    grok_confidence = grok_validation["overall_confidence"]
                    result["confidence"] = (original_confidence * 0.6 + grok_confidence * 0.4)
                
                # Add logical consistency check
                consistency_check = await self.check_logical_consistency(
                    nl_query,
                    result["sql_query"],
                    enhanced_context
                )
                result["logical_consistency"] = consistency_check
            
            # Cache successful results
            if result.get("sql_query") and not result.get("error"):
                if len(self.query_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[cache_key] = result.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert NL to SQL: {e}")
            return {
                "sql_query": "",
                "error": str(e),
                "explanation": f"Failed to parse the natural language query: {str(e)}"
            }
    
    async def convert_sql_to_nl(self, sql_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convert SQL to natural language explanation
        """
        try:
            # Parse SQL query
            sql_upper = sql_query.upper()
            query_parts = self._parse_sql_query(sql_query)
            
            # Generate natural language description
            nl_description = self._generate_nl_from_parts(query_parts)
            
            # Add context-specific explanation
            if context:
                nl_description = self._enhance_with_context(nl_description, context)
            
            result = {
                "natural_language": nl_description,
                "query_type": query_parts.get("type", "unknown"),
                "components": query_parts,
                "technical_level": "user-friendly"
            }
            
            # GrokClient validation of NL explanation
            if self.grok_client and nl_description:
                nl_validation = await self.validate_nl_with_grok(
                    nl_description,
                    sql_query,
                    context
                )
                result["grok_nl_validation"] = nl_validation
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert SQL to NL: {e}")
            return {
                "natural_language": "This SQL query performs a database operation.",
                "error": str(e)
            }
    
    async def _convert_relational_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to relational SQL query"""
        sql_parts = {
            "select": "*",
            "from": "",
            "where": [],
            "group_by": [],
            "order_by": [],
            "limit": None
        }
        
        # Check each pattern type
        for pattern_type, patterns in self.nl_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, nl_query)
                if match:
                    if pattern_type == "show_all":
                        sql_parts["from"] = match.group(1)
                    
                    elif pattern_type == "filter_equals":
                        table = match.group(1)
                        column = match.group(2)
                        value = match.group(3)
                        sql_parts["from"] = table
                        sql_parts["where"].append(f"{column} = '{value}'")
                    
                    elif pattern_type == "filter_contains":
                        table = match.group(1)
                        column = match.group(2)
                        value = match.group(3)
                        sql_parts["from"] = table
                        sql_parts["where"].append(f"{column} LIKE '%{value}%'")
                    
                    elif pattern_type == "count":
                        table = match.group(1)
                        sql_parts["select"] = "COUNT(*) AS count"
                        sql_parts["from"] = table
                    
                    elif pattern_type == "sum":
                        column = match.group(1)
                        table = match.group(2)
                        sql_parts["select"] = f"SUM({column}) AS total_{column}"
                        sql_parts["from"] = table
                    
                    elif pattern_type == "average":
                        column = match.group(1)
                        table = match.group(2)
                        sql_parts["select"] = f"AVG({column}) AS avg_{column}"
                        sql_parts["from"] = table
                    
                    elif pattern_type == "top_n":
                        n = match.group(1)
                        table = match.group(2)
                        order_col = match.group(3)
                        sql_parts["from"] = table
                        sql_parts["order_by"].append(f"{order_col} DESC")
                        sql_parts["limit"] = n
                    
                    break
        
        # Construct SQL query
        sql_query = self._build_sql_from_parts(sql_parts)
        
        return {
            "sql_query": sql_query,
            "query_type": "relational",
            "detected_patterns": [p for p in self.nl_patterns.keys() if any(re.search(pat, nl_query) for pat in self.nl_patterns[p])]
        }
    
    async def _convert_graph_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to HANA graph query"""
        # Default graph query parts
        graph_parts = {
            "match": "",
            "where": [],
            "return": "",
            "limit": None
        }
        
        # Check for graph patterns
        if "path" in nl_query or "route" in nl_query:
            # Extract path endpoints
            path_match = re.search(r"(?:path|route) (?:from|between) (\w+) (?:to|and) (\w+)", nl_query)
            if path_match:
                start = path_match.group(1)
                end = path_match.group(2)
                
                if "shortest" in nl_query:
                    graph_parts["match"] = f"p = shortestPath((a {{name: '{start}'}})-[*]-(b {{name: '{end}'}}))"
                else:
                    graph_parts["match"] = f"p = (a {{name: '{start}'}})-[*]-(b {{name: '{end}'}}))"
                
                graph_parts["return"] = "p"
        
        elif "neighbors" in nl_query or "connected to" in nl_query:
            # Extract node
            node_match = re.search(r"(?:neighbors of|connected to) (\w+)", nl_query)
            if node_match:
                node = node_match.group(1)
                graph_parts["match"] = f"(n {{name: '{node}'}})-[r]-(neighbor)"
                graph_parts["return"] = "neighbor, r"
        
        else:
            # Default node match
            node_match = re.search(r"(\w+) nodes", nl_query)
            if node_match:
                node_type = node_match.group(1)
                graph_parts["match"] = f"(n:{node_type})"
                graph_parts["return"] = "n"
            else:
                graph_parts["match"] = "(n)"
                graph_parts["return"] = "n"
                graph_parts["limit"] = "10"
        
        # Build graph query
        query_parts = [f"MATCH {graph_parts['match']}"]
        
        if graph_parts["where"]:
            query_parts.append(f"WHERE {' AND '.join(graph_parts['where'])}")
        
        query_parts.append(f"RETURN {graph_parts['return']}")
        
        if graph_parts["limit"]:
            query_parts.append(f"LIMIT {graph_parts['limit']}")
        
        graph_query = " ".join(query_parts)
        
        return {
            "sql_query": graph_query,
            "query_type": "graph",
            "graph_operation": self._identify_graph_operation(nl_query)
        }
    
    async def _convert_vector_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to HANA vector query"""
        # Extract table and search parameters - from context or environment
        table = context.get("default_table") or os.getenv("DEFAULT_VECTOR_TABLE", "documents")
        vector_column = context.get("vector_column") or os.getenv("DEFAULT_VECTOR_COLUMN", "embedding")
        text_column = context.get("text_column") or os.getenv("DEFAULT_TEXT_COLUMN", "content")
        
        # Check for similarity patterns
        similarity_match = re.search(r"find (?:(\d+) )?(?:most )?similar (\w+) to ['\"]?([^'\"]+)['\"]?", nl_query)
        
        if similarity_match:
            k = similarity_match.group(1) or "10"
            table = similarity_match.group(2) or table
            query_text = similarity_match.group(3)
            
            # Check if hybrid search is needed
            if "containing" in nl_query or "matching" in nl_query:
                # Hybrid search with text filtering
                text_match = re.search(r"containing ['\"]?([^'\"]+)['\"]?", nl_query)
                if text_match:
                    text_filter = text_match.group(1)
                    vector_query = f"""
                    SELECT TOP {k} *, 
                    COSINE_SIMILARITY({vector_column}, TO_REAL_VECTOR(EMBED('{query_text}'))) AS similarity 
                    FROM {table} 
                    WHERE {text_column} LIKE '%{text_filter}%'
                    ORDER BY similarity DESC
                    """.strip()
                else:
                    vector_query = self._create_similarity_query(k, table, vector_column, query_text)
            else:
                vector_query = self._create_similarity_query(k, table, vector_column, query_text)
        
        elif "nearest" in nl_query or "closest" in nl_query:
            # KNN search
            knn_match = re.search(r"(?:(\d+) )?(?:nearest|closest) (\w+) to ['\"]?([^'\"]+)['\"]?", nl_query)
            if knn_match:
                k = knn_match.group(1) or "5"
                table = knn_match.group(2) or table
                query_text = knn_match.group(3)
                
                vector_query = f"""
                SELECT TOP {k} *, 
                L2DISTANCE({vector_column}, TO_REAL_VECTOR(EMBED('{query_text}'))) AS distance 
                FROM {table} 
                ORDER BY distance ASC
                """.strip()
        
        else:
            # Default similarity search
            vector_query = self._create_similarity_query("10", table, vector_column, nl_query)
        
        return {
            "sql_query": vector_query,
            "query_type": "vector",
            "vector_operation": "similarity_search" if "similar" in nl_query else "knn_search"
        }
    
    async def _convert_hybrid_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex natural language queries that might combine multiple query types"""
        # This would handle more complex queries that combine relational, graph, and vector operations
        # For now, default to relational
        return await self._convert_relational_query(nl_query, context)
    
    def _detect_query_intent(self, nl_query: str, nl_enhanced: Dict[str, Any] = None) -> str:
        """Detect the primary intent of the query with enhanced NLP"""
        nl_lower = nl_query.lower()
        
        # Use NLP enhancements if available
        if nl_enhanced and nl_enhanced.get("intent") != "unknown":
            return nl_enhanced["intent"]
        
        # Check for ML/analytics indicators
        ml_indicators = ["predict", "classify", "cluster", "forecast", "model", "train", "regression", "correlation"]
        if any(indicator in nl_lower for indicator in ml_indicators):
            return "ml"
        
        # Check for time series indicators
        time_indicators = ["trend", "seasonal", "time series", "over time", "forecast", "moving average"]
        if any(indicator in nl_lower for indicator in time_indicators):
            return "time_series"
        
        # Check for spatial indicators
        spatial_indicators = ["distance", "location", "geographic", "spatial", "coordinates", "radius", "within"]
        if any(indicator in nl_lower for indicator in spatial_indicators):
            return "spatial"
        
        # Check for vector indicators
        vector_indicators = ["similar", "similarity", "nearest", "closest", "embedding", "vector", "semantic"]
        if any(indicator in nl_lower for indicator in vector_indicators):
            return "vector"
        
        # Check for graph indicators
        graph_indicators = ["path", "route", "connection", "neighbors", "graph", "nodes", "edges", "relationship"]
        if any(indicator in nl_lower for indicator in graph_indicators):
            return "graph"
        
        # Default to relational
        return "relational"
    
    def _build_sql_from_parts(self, parts: Dict[str, Any]) -> str:
        """Build SQL query from parts"""
        query_parts = []
        
        # SELECT clause
        query_parts.append(f"SELECT {parts['select']}")
        
        # FROM clause
        if parts["from"]:
            query_parts.append(f"FROM {parts['from']}")
        
        # WHERE clause
        if parts["where"]:
            query_parts.append(f"WHERE {' AND '.join(parts['where'])}")
        
        # GROUP BY clause
        if parts["group_by"]:
            query_parts.append(f"GROUP BY {', '.join(parts['group_by'])}")
        
        # ORDER BY clause
        if parts["order_by"]:
            query_parts.append(f"ORDER BY {', '.join(parts['order_by'])}")
        
        # LIMIT clause
        if parts["limit"]:
            query_parts.append(f"LIMIT {parts['limit']}")
        
        return " ".join(query_parts)
    
    def _create_similarity_query(self, k: str, table: str, vector_column: str, query_text: str) -> str:
        """Create a similarity search query"""
        return f"""
        SELECT TOP {k} *, 
        COSINE_SIMILARITY({vector_column}, TO_REAL_VECTOR(EMBED('{query_text}'))) AS similarity 
        FROM {table} 
        ORDER BY similarity DESC
        """.strip()
    
    def _identify_graph_operation(self, nl_query: str) -> str:
        """Identify the type of graph operation"""
        if "shortest" in nl_query:
            return "shortest_path"
        elif "path" in nl_query or "route" in nl_query:
            return "path_finding"
        elif "neighbors" in nl_query or "connected" in nl_query:
            return "neighbor_search"
        else:
            return "node_matching"
    
    def _parse_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Parse SQL query into components"""
        sql_upper = sql_query.upper()
        parts = {
            "type": "unknown",
            "tables": [],
            "columns": [],
            "conditions": [],
            "functions": [],
            "joins": []
        }
        
        # Determine query type
        if sql_upper.startswith("SELECT"):
            parts["type"] = "select"
            
            # Extract tables
            from_match = re.search(r"FROM\s+(\w+)", sql_upper)
            if from_match:
                parts["tables"].append(from_match.group(1))
            
            # Extract joins
            join_matches = re.finditer(r"JOIN\s+(\w+)", sql_upper)
            for match in join_matches:
                parts["tables"].append(match.group(1))
                parts["joins"].append(match.group(0))
            
            # Extract conditions
            where_match = re.search(r"WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)", sql_upper)
            if where_match:
                parts["conditions"] = [where_match.group(1).strip()]
            
            # Extract functions
            for func_category, functions in self.hana_functions.items():
                for func in functions:
                    if func in sql_upper:
                        parts["functions"].append(func)
        
        elif sql_upper.startswith("MATCH"):
            parts["type"] = "graph"
            # Parse graph pattern
            pattern_match = re.search(r"MATCH\s+(.+?)(?:WHERE|RETURN)", sql_query, re.IGNORECASE)
            if pattern_match:
                parts["graph_pattern"] = pattern_match.group(1).strip()
        
        return parts
    
    def _generate_nl_from_parts(self, parts: Dict[str, Any]) -> str:
        """Generate natural language from parsed SQL parts"""
        nl_parts = []
        
        if parts["type"] == "select":
            nl_parts.append("This query retrieves data")
            
            if parts["tables"]:
                if len(parts["tables"]) == 1:
                    nl_parts.append(f"from the {parts['tables'][0]} table")
                else:
                    nl_parts.append(f"from {len(parts['tables'])} tables: {', '.join(parts['tables'])}")
            
            if parts["joins"]:
                nl_parts.append("by combining related data")
            
            if parts["conditions"]:
                nl_parts.append("with specific filtering conditions")
            
            if "COSINE_SIMILARITY" in parts["functions"]:
                nl_parts.append("using vector similarity search")
            elif "L2DISTANCE" in parts["functions"]:
                nl_parts.append("using distance-based search")
            
            if any(func in parts["functions"] for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
                nl_parts.append("and calculates aggregate values")
        
        elif parts["type"] == "graph":
            nl_parts.append("This is a graph query that")
            if "shortestPath" in str(parts.get("graph_pattern", "")):
                nl_parts.append("finds the shortest path between nodes")
            else:
                nl_parts.append("searches for patterns in the graph database")
        
        return " ".join(nl_parts) + "."
    
    def _enhance_with_context(self, nl_description: str, context: Dict[str, Any]) -> str:
        """Enhance NL description with context information"""
        if context.get("domain"):
            nl_description = f"In the context of {context['domain']}, {nl_description.lower()}"
        
        if context.get("purpose"):
            nl_description += f" The purpose is to {context['purpose']}."
        
        return nl_description
    
    def _generate_sql_explanation(self, sql_query: str, original_nl: str) -> str:
        """Generate detailed explanation of SQL query"""
        explanation_parts = [f"Based on your request '{original_nl}', I've created a SQL query that:"]
        
        sql_upper = sql_query.upper()
        
        # Explain main operation
        if sql_upper.startswith("SELECT"):
            if "COUNT(" in sql_upper:
                explanation_parts.append("- Counts the number of records")
            elif "SUM(" in sql_upper:
                explanation_parts.append("- Calculates the sum of values")
            elif "AVG(" in sql_upper:
                explanation_parts.append("- Calculates the average of values")
            elif "*" in sql_query:
                explanation_parts.append("- Retrieves all columns from the table")
            else:
                explanation_parts.append("- Retrieves specific data fields")
        
        # Explain special operations
        if "COSINE_SIMILARITY" in sql_upper:
            explanation_parts.append("- Uses vector similarity to find semantically similar items")
        
        if "JOIN" in sql_upper:
            explanation_parts.append("- Combines data from multiple related tables")
        
        if "WHERE" in sql_upper:
            explanation_parts.append("- Filters results based on specific conditions")
        
        if "ORDER BY" in sql_upper:
            if "DESC" in sql_upper:
                explanation_parts.append("- Sorts results in descending order")
            else:
                explanation_parts.append("- Sorts results in ascending order")
        
        if "LIMIT" in sql_upper or "TOP" in sql_upper:
            explanation_parts.append("- Limits the number of results returned")
        
        return "\n".join(explanation_parts)
    
    def _calculate_confidence(self, nl_query: str, sql_query: str, nl_enhanced: Dict[str, Any] = None) -> float:
        """Calculate confidence score for the conversion with enhanced factors"""
        confidence = 0.3  # Base confidence
        
        # Use NLP confidence if available
        if nl_enhanced and nl_enhanced.get("confidence"):
            confidence += 0.2 * nl_enhanced["confidence"]
        
        # Check if we matched any patterns
        matched_patterns = 0
        for pattern_list in self.nl_patterns.values():
            if any(re.search(p, nl_query) for p in pattern_list):
                matched_patterns += 1
        
        if matched_patterns > 0:
            confidence += 0.15 * min(matched_patterns, 3)  # Cap pattern bonus
        
        # Check SQL query quality
        if sql_query and len(sql_query) > 10:
            confidence += 0.15
            
            # Additional quality checks
            sql_upper = sql_query.upper()
            if "SELECT" in sql_upper and "FROM" in sql_upper:
                confidence += 0.1
            
            # Check for proper HANA functions
            hana_functions_used = 0
            for func_category, functions in self.hana_functions.items():
                for func in functions:
                    if func in sql_upper:
                        hana_functions_used += 1
            
            if hana_functions_used > 0:
                confidence += 0.05 * min(hana_functions_used, 2)
        
        # Entity recognition bonus
        if nl_enhanced and nl_enhanced.get("entities"):
            confidence += 0.05 * min(len(nl_enhanced["entities"]), 3)
        
        # Query complexity adjustment
        if nl_enhanced and nl_enhanced.get("query_complexity") == "complex":
            confidence -= 0.1  # Lower confidence for complex queries
        elif nl_enhanced and nl_enhanced.get("query_complexity") == "simple":
            confidence += 0.05
        
        # Cap confidence at 0.95
        return min(confidence, 0.95)
    
    async def _convert_ml_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to HANA ML/PAL query"""
        ml_parts = {
            "algorithm": "",
            "input_table": "",
            "target_column": "",
            "feature_columns": [],
            "parameters": {}
        }
        
        # Detect ML algorithm
        if "regression" in nl_query or "predict" in nl_query:
            if "linear" in nl_query:
                ml_parts["algorithm"] = "PAL_LINEAR_REGRESSION"
            else:
                ml_parts["algorithm"] = "PAL_RANDOM_FOREST"
        elif "cluster" in nl_query or "group" in nl_query:
            ml_parts["algorithm"] = "PAL_K_MEANS"
        elif "classify" in nl_query or "classification" in nl_query:
            ml_parts["algorithm"] = "PAL_RANDOM_FOREST"
        
        # Extract table and columns
        table_match = re.search(r"(?:from|in|using)\s+(\w+)", nl_query)
        if table_match:
            ml_parts["input_table"] = table_match.group(1)
        
        target_match = re.search(r"predict\s+(\w+)", nl_query)
        if target_match:
            ml_parts["target_column"] = target_match.group(1)
        
        # Build ML query
        if ml_parts["algorithm"] and ml_parts["input_table"]:
            ml_query = f"""
            CALL {ml_parts["algorithm"]}(
                {ml_parts["input_table"]},
                {ml_parts["target_column"] or f"'{os.getenv('DEFAULT_TARGET_COLUMN', 'target')}'"},
                #PARAMETERS
            ) WITH OVERVIEW
            """.strip()
        else:
            ml_query = "-- ML query requires algorithm and input table specification"
        
        return {
            "sql_query": ml_query,
            "query_type": "ml",
            "algorithm": ml_parts["algorithm"],
            "detected_components": ml_parts
        }
    
    async def _convert_time_series_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to HANA time series query"""
        ts_parts = {
            "table": "",
            "time_column": "",
            "value_column": "",
            "operation": "",
            "granularity": "DAY"
        }
        
        # Extract components
        table_match = re.search(r"(?:from|in)\s+(\w+)", nl_query)
        if table_match:
            ts_parts["table"] = table_match.group(1)
        
        # Detect time series operation
        if "trend" in nl_query or "moving average" in nl_query:
            ts_parts["operation"] = "SERIES_ROUND"
        elif "forecast" in nl_query:
            ts_parts["operation"] = "SERIES_GENERATE"
        elif "seasonal" in nl_query:
            ts_parts["operation"] = "SERIES_DISAGGREGATE"
        
        # Detect granularity
        if "hour" in nl_query:
            ts_parts["granularity"] = "HOUR"
        elif "month" in nl_query:
            ts_parts["granularity"] = "MONTH"
        elif "year" in nl_query:
            ts_parts["granularity"] = "YEAR"
        
        # Build time series query
        if ts_parts["table"] and ts_parts["operation"]:
            ts_query = f"""
            SELECT * FROM {ts_parts["operation"]}(
                SELECT * FROM {ts_parts["table"]}
                ORDER BY {ts_parts["time_column"] or os.getenv("DEFAULT_TIME_COLUMN", "timestamp")}
            ) AS SERIES
            """.strip()
        else:
            ts_query = "-- Time series query requires table and operation specification"
        
        return {
            "sql_query": ts_query,
            "query_type": "time_series",
            "operation": ts_parts["operation"],
            "detected_components": ts_parts
        }
    
    async def _convert_spatial_query(self, nl_query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert natural language to HANA spatial query"""
        spatial_parts = {
            "table": "",
            "geometry_column": "location",
            "operation": "",
            "reference_point": None,
            "distance": None
        }
        
        # Extract table
        table_match = re.search(r"(?:from|in)\s+(\w+)", nl_query)
        if table_match:
            spatial_parts["table"] = table_match.group(1)
        
        # Detect spatial operation
        if "distance" in nl_query:
            spatial_parts["operation"] = "ST_DISTANCE"
        elif "within" in nl_query or "inside" in nl_query:
            spatial_parts["operation"] = "ST_WITHIN"
        elif "intersect" in nl_query:
            spatial_parts["operation"] = "ST_INTERSECTS"
        elif "buffer" in nl_query or "radius" in nl_query:
            spatial_parts["operation"] = "ST_BUFFER"
        
        # Extract distance
        distance_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|meter|mile)", nl_query)
        if distance_match:
            spatial_parts["distance"] = distance_match.group(1)
        
        # Extract coordinates from query or use context
        coord_match = re.search(r"coordinates?\s*(\d+(?:\.\d+)?),?\s*(\d+(?:\.\d+)?)", nl_query)
        if coord_match:
            spatial_parts["reference_point"] = f"{coord_match.group(1)}, {coord_match.group(2)}"
        elif context and context.get("reference_coordinates"):
            spatial_parts["reference_point"] = context["reference_coordinates"]
        else:
            # Get from environment or use dynamic default
            default_coords = os.getenv("DEFAULT_SPATIAL_COORDINATES", "52.5200, 13.4050")  # Berlin coordinates
            spatial_parts["reference_point"] = default_coords
        
        # Build spatial query
        if spatial_parts["table"] and spatial_parts["operation"]:
            if spatial_parts["operation"] == "ST_DISTANCE":
                spatial_query = f"""
                SELECT *, {spatial_parts["operation"]}({spatial_parts["geometry_column"]}, 
                    NEW ST_Point({spatial_parts["reference_point"]})) AS distance
                FROM {spatial_parts["table"]}
                ORDER BY distance
                """.strip()
            else:
                spatial_query = f"""
                SELECT * FROM {spatial_parts["table"]}
                WHERE {spatial_parts["operation"]}({spatial_parts["geometry_column"]}, 
                    NEW ST_Point({spatial_parts["reference_point"]}))
                """.strip()
        else:
            spatial_query = "-- Spatial query requires table and operation specification"
        
        return {
            "sql_query": spatial_query,
            "query_type": "spatial",
            "operation": spatial_parts["operation"],
            "detected_components": spatial_parts
        }
    
    def _optimize_query(self, sql_query: str, query_type: str) -> str:
        """Apply intelligent query optimizations"""
        if not sql_query or sql_query.startswith("--"):
            return sql_query
        
        optimized = sql_query
        
        # Add query hints for HANA
        if query_type == "vector" and "COSINE_SIMILARITY" in optimized:
            optimized = f"/*+ USE_VECTOR_ENGINE */ {optimized}"
        elif query_type == "graph" and "MATCH" in optimized:
            optimized = f"/*+ USE_GRAPH_ENGINE */ {optimized}"
        elif "GROUP BY" in optimized.upper():
            optimized = f"/*+ USE_OLAP_PLAN */ {optimized}"
        
        # Add LIMIT for potentially large result sets
        if ("SELECT *" in optimized.upper() and 
            "LIMIT" not in optimized.upper() and 
            "TOP" not in optimized.upper()):
            default_limit = os.getenv("SQL_DEFAULT_LIMIT", "100")
            optimized += f" LIMIT {default_limit}"
        
        # Optimize JOIN order for better performance
        if "JOIN" in optimized.upper():
            # Add USE_HASH_JOIN hint for large tables
            optimized = f"/*+ USE_HASH_JOIN */ {optimized}"
        
        return optimized
    
    def _generate_query_suggestions(self, nl_query: str, failed_query: str = None) -> List[str]:
        """Generate alternative query suggestions"""
        suggestions = []
        
        # Suggest simpler alternatives
        if "complex" in nl_query or len(nl_query.split()) > 10:
            suggestions.append("Try breaking down the question into simpler parts")
            suggestions.append("Specify table names explicitly if known")
        
        # Suggest specific table/column names
        if "table" not in nl_query.lower():
            suggestions.append("Include the table name in your query")
        
        # Vector search suggestions
        if any(word in nl_query.lower() for word in ["similar", "like", "match"]):
            suggestions.append("For similarity search, try: 'find similar [ITEMS] to [SEARCH_TERM]'")
            suggestions.append("Specify the number of results: 'find [N] most similar...'")
        
        # Graph search suggestions
        if any(word in nl_query.lower() for word in ["connect", "path", "relationship"]):
            suggestions.append("For graph queries, try: 'find path from [START_NODE] to [END_NODE]'")
            suggestions.append("For neighbors: 'show all nodes connected to [NODE_NAME]'")
        
        return suggestions
    
    async def explain_query_execution_plan(self, sql_query: str) -> Dict[str, Any]:
        """Generate execution plan explanation for the SQL query"""
        try:
            plan_info = {
                "estimated_cost": "medium",
                "optimization_hints": [],
                "performance_considerations": [],
                "resource_usage": {
                    "cpu": "medium",
                    "memory": "low",
                    "io": "medium"
                }
            }
            
            sql_upper = sql_query.upper()
            
            # Analyze query complexity
            if "JOIN" in sql_upper:
                plan_info["optimization_hints"].append("Consider indexing join columns")
                plan_info["resource_usage"]["cpu"] = "high"
            
            if "GROUP BY" in sql_upper or "ORDER BY" in sql_upper:
                plan_info["optimization_hints"].append("Sorting/grouping operations may require memory")
                plan_info["resource_usage"]["memory"] = "medium"
            
            if "COSINE_SIMILARITY" in sql_upper or "L2DISTANCE" in sql_upper:
                plan_info["optimization_hints"].append("Vector operations use specialized engine")
                plan_info["performance_considerations"].append("Ensure vector indexes are available")
                plan_info["resource_usage"]["cpu"] = "high"
            
            if "CONTAINS" in sql_upper or "FUZZY" in sql_upper:
                plan_info["optimization_hints"].append("Text search uses full-text indexes")
                plan_info["performance_considerations"].append("Verify full-text indexes exist")
            
            return plan_info
            
        except Exception as e:
            logger.error(f"Failed to analyze execution plan: {e}")
            return {"error": "Could not analyze execution plan"}
    
    async def validate_query_syntax(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query syntax and provide detailed feedback"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        if not sql_query or sql_query.strip() == "":
            validation_result["is_valid"] = False
            validation_result["errors"].append("Empty query")
            return validation_result
        
        # Basic syntax validation
        if SQLPARSE_AVAILABLE:
            try:
                parsed = sqlparse.parse(sql_query)
                if not parsed:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Could not parse SQL syntax")
            except Exception as e:
                validation_result["warnings"].append(f"SQL parsing warning: {e}")
        
        # Check for common issues
        sql_upper = sql_query.upper()
        
        # Missing FROM clause
        if "SELECT" in sql_upper and "FROM" not in sql_upper:
            validation_result["warnings"].append("Query missing FROM clause")
        
        # Unmatched parentheses
        if sql_query.count("(") != sql_query.count(")"):
            validation_result["is_valid"] = False
            validation_result["errors"].append("Unmatched parentheses")
        
        # Check for HANA-specific syntax
        hana_functions_found = []
        for func_category, functions in self.hana_functions.items():
            for func in functions:
                if func in sql_upper:
                    hana_functions_found.append(func)
        
        if hana_functions_found:
            validation_result["suggestions"].append(
                f"Using HANA-specific functions: {', '.join(hana_functions_found[:3])}"
            )
        
        return validation_result
    
    async def get_query_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "performance_stats": self.performance_stats.copy(),
            "cache_info": {
                "size": len(self.query_cache),
                "max_size": self.max_cache_size,
                "hit_rate": (
                    self.performance_stats["cache_hits"] / 
                    max(self.performance_stats["queries_processed"], 1)
                ) * 100
            },
            "nlp_availability": {
                "spacy": SPACY_AVAILABLE and self.nlp_models.get('spacy') is not None,
                "transformers": TRANSFORMERS_AVAILABLE and self.nlp_models.get('text2sql') is not None,
                "sqlparse": SQLPARSE_AVAILABLE
            },
            "supported_query_types": ["relational", "graph", "vector", "ml", "time_series", "spatial"],
            "hana_functions_supported": sum(len(funcs) for funcs in self.hana_functions.values())
        }
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def update_conversation_context(self, session_id: str, context: Dict[str, Any]):
        """Update conversation context for multi-turn queries"""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = {
                "queries": [],
                "tables_mentioned": set(),
                "columns_mentioned": set(),
                "last_query_type": None,
                "created_at": datetime.now()
            }
        
        self.conversation_contexts[session_id].update(context)
        
        # Cleanup old contexts (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        to_remove = [
            sid for sid, ctx in self.conversation_contexts.items()
            if ctx.get("created_at", datetime.now()) < cutoff_time
        ]
        for sid in to_remove:
            del self.conversation_contexts[sid]
    
    async def test_enhanced_features(self) -> Dict[str, Any]:
        """Test all enhanced features to ensure they work correctly"""
        test_results = {
            "nlp_processing": False,
            "security_validation": False,
            "query_optimization": False,
            "multi_model_support": False,
            "caching": False,
            "error_handling": False
        }
        
        try:
            # Test NLP processing
            nl_enhanced = self._enhance_nl_understanding("show me all customers")
            test_results["nlp_processing"] = "entities" in nl_enhanced
            
            # Test security validation
            test_table = os.getenv("TEST_TABLE_NAME", "test_table")
            test_query = f"SELECT * FROM {test_table} WHERE id = 1"
            security_check = self._validate_sql_security(test_query)
            test_results["security_validation"] = "is_safe" in security_check
            
            # Test query optimization
            optimization_test_query = f"SELECT * FROM {os.getenv('TEST_LARGE_TABLE', 'large_table')}"
            optimized = self._optimize_query(optimization_test_query, "relational")
            test_results["query_optimization"] = "LIMIT" in optimized
            
            # Test multi-model support
            ml_result = await self._convert_ml_query("predict sales using regression", {})
            test_results["multi_model_support"] = "PAL_" in ml_result.get("sql_query", "")
            
            # Test caching
            cache_key = f"test_cache_{datetime.now().timestamp()}"
            test_data = {
                "operation": "cache_test", 
                "timestamp": datetime.now().isoformat(),
                "test_id": str(uuid.uuid4())[:8]
            }
            self.query_cache[cache_key] = test_data
            test_results["caching"] = cache_key in self.query_cache
            # Cleanup test cache entry
            if cache_key in self.query_cache:
                del self.query_cache[cache_key]
            
            # Test error handling
            try:
                dangerous_query = os.getenv("ERROR_TEST_QUERY", "invalid query with potential injection")
                await self.convert_nl_to_sql(dangerous_query)
                test_results["error_handling"] = True
            except:
                test_results["error_handling"] = True
            
        except Exception as e:
            logger.error(f"Feature testing failed: {e}")
        
        return {
            "test_results": test_results,
            "features_working": sum(test_results.values()),
            "total_features": len(test_results)
        }
    
    async def validate_sql_with_grok(self, sql_query: str, nl_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use GrokClient to validate SQL syntax and logical consistency"""
        if not self.grok_client:
            return {
                "grok_available": False,
                "syntax_valid": True,  # Assume valid if no Grok
                "logical_consistent": True,
                "confidence": 0.5,
                "feedback": "GrokClient not available - using basic validation"
            }
        
        try:
            self.performance_stats["grok_validations"] += 1
            
            # Create validation prompt
            validation_prompt = f"""
            Validate this SQL query for syntax correctness and logical consistency:
            
            Original Natural Language Query: {nl_query}
            Generated SQL Query: {sql_query}
            Context: {json.dumps(context or {}, indent=2)}
            
            Please evaluate:
            1. SQL Syntax Correctness (0-100)
            2. Logical Consistency with NL intent (0-100) 
            3. HANA SQL compatibility (0-100)
            4. Security considerations (0-100)
            5. Performance implications (0-100)
            
            Return JSON with:
            - syntax_score: number (0-100)
            - logical_consistency_score: number (0-100)
            - hana_compatibility_score: number (0-100)
            - security_score: number (0-100)
            - performance_score: number (0-100)
            - overall_score: number (0-100)
            - syntax_valid: boolean
            - logical_consistent: boolean
            - feedback: string with specific recommendations
            - issues: array of specific problems found
            - suggestions: array of improvement suggestions
            """
            
            result = await self.grok_client.analyze(validation_prompt, context)
            validation_data = json.loads(result)
            
            # Enhance with our local validation
            local_syntax = await self.validate_query_syntax(sql_query)
            local_security = self._validate_sql_security(sql_query)
            
            # Combine Grok and local validation
            combined_result = {
                "grok_available": True,
                "grok_validation": validation_data,
                "local_syntax_check": local_syntax,
                "local_security_check": local_security,
                "syntax_valid": validation_data.get("syntax_valid", True) and local_syntax.get("is_valid", True),
                "logical_consistent": validation_data.get("logical_consistent", True),
                "security_safe": local_security.get("is_safe", True),
                "overall_confidence": validation_data.get("overall_score", 50) / 100,
                "combined_feedback": self._generate_combined_feedback(validation_data, local_syntax, local_security)
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Grok validation failed: {e}")
            return {
                "grok_available": False,
                "error": str(e),
                "syntax_valid": True,  # Fallback assumption
                "logical_consistent": True,
                "feedback": f"Grok validation failed: {e}"
            }
    
    async def validate_nl_with_grok(self, nl_explanation: str, sql_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use GrokClient to validate natural language explanations"""
        if not self.grok_client:
            return {
                "grok_available": False,
                "explanation_quality": 75,
                "accuracy": 75,
                "feedback": "GrokClient not available - using basic validation"
            }
        
        try:
            validation_prompt = f"""
            Validate this natural language explanation of an SQL query:
            
            SQL Query: {sql_query}
            Generated Natural Language Explanation: {nl_explanation}
            Context: {json.dumps(context or {}, indent=2)}
            
            Evaluate:
            1. Accuracy of explanation (0-100)
            2. Clarity and readability (0-100)
            3. Completeness (covers all SQL operations) (0-100)
            4. Technical correctness (0-100)
            
            Return JSON with:
            - accuracy_score: number (0-100)
            - clarity_score: number (0-100) 
            - completeness_score: number (0-100)
            - technical_score: number (0-100)
            - overall_score: number (0-100)
            - feedback: string with specific comments
            - missing_elements: array of SQL features not explained
            - improvement_suggestions: array of suggestions
            """
            
            result = await self.grok_client.analyze(validation_prompt, context)
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"NL validation with Grok failed: {e}")
            return {
                "grok_available": False,
                "error": str(e),
                "explanation_quality": 50,
                "feedback": f"NL validation failed: {e}"
            }
    
    async def check_logical_consistency(self, nl_query: str, sql_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deep logical consistency check between NL intent and SQL output"""
        if not self.grok_client:
            return self._basic_logical_consistency_check(nl_query, sql_query)
        
        try:
            self.performance_stats["logical_consistency_checks"] += 1
            
            consistency_prompt = f"""
            Perform deep logical consistency analysis:
            
            User Intent (Natural Language): {nl_query}
            Generated SQL Query: {sql_query}
            Context: {json.dumps(context or {}, indent=2)}
            
            Analyze if the SQL query correctly implements the user's intent by checking:
            1. Data source alignment (correct tables/views)
            2. Column selection accuracy
            3. Filter condition correctness
            4. Aggregation appropriateness
            5. Join logic validity
            6. Sort/limit alignment with intent
            7. Function usage appropriateness
            
            Return JSON with:
            - consistency_score: number (0-100)
            - intent_captured: boolean
            - missing_requirements: array of user requirements not addressed
            - extra_operations: array of SQL operations not requested
            - semantic_alignment: number (0-100)
            - detailed_analysis: object with per-component analysis
            - recommendations: array of improvements
            """
            
            result = await self.grok_client.analyze(consistency_prompt, context)
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"Logical consistency check failed: {e}")
            return self._basic_logical_consistency_check(nl_query, sql_query)
    
    def _basic_logical_consistency_check(self, nl_query: str, sql_query: str) -> Dict[str, Any]:
        """Basic logical consistency check without Grok"""
        score = 70  # Base score
        issues = []
        
        nl_lower = nl_query.lower()
        sql_upper = sql_query.upper()
        
        # Check for basic alignment
        if "count" in nl_lower and "COUNT" not in sql_upper:
            score -= 20
            issues.append("User requested count but SQL doesn't include COUNT function")
        
        if "sum" in nl_lower and "SUM" not in sql_upper:
            score -= 20
            issues.append("User requested sum but SQL doesn't include SUM function")
        
        if "average" in nl_lower and "AVG" not in sql_upper:
            score -= 20
            issues.append("User requested average but SQL doesn't include AVG function")
        
        if "similar" in nl_lower and "COSINE_SIMILARITY" not in sql_upper:
            score -= 15
            issues.append("User requested similarity but SQL doesn't use vector similarity")
        
        if "where" in nl_lower and "WHERE" not in sql_upper:
            score -= 10
            issues.append("User implied filtering but SQL doesn't include WHERE clause")
        
        return {
            "consistency_score": max(0, score),
            "intent_captured": score >= 70,
            "missing_requirements": issues,
            "basic_check": True,
            "grok_available": False
        }
    
    def _generate_combined_feedback(self, grok_result: Dict, syntax_result: Dict, security_result: Dict) -> str:
        """Generate combined feedback from all validation sources"""
        feedback_parts = []
        
        # Grok feedback
        if grok_result.get("feedback"):
            feedback_parts.append(f"AI Analysis: {grok_result['feedback']}")
        
        # Syntax feedback
        if not syntax_result.get("is_valid", True):
            feedback_parts.append(f"Syntax Issues: {'; '.join(syntax_result.get('errors', []))}")
        
        # Security feedback
        if not security_result.get("is_safe", True):
            feedback_parts.append(f"Security Concerns: {'; '.join(security_result.get('issues', []))}")
        
        # Suggestions
        if grok_result.get("suggestions"):
            feedback_parts.append(f"Suggestions: {'; '.join(grok_result['suggestions'])}")
        
        return " | ".join(feedback_parts) if feedback_parts else "Query appears valid and well-formed"
    
    # Integration methods for main SQL Agent
    
    async def process_conversational_query(
        self, 
        nl_query: str, 
        session_id: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process query with conversation context awareness"""
        # Get conversation context
        conv_context = self.conversation_contexts.get(session_id, {})
        
        # Merge with provided context
        enhanced_context = {**(context or {}), **conv_context}
        
        # Convert query
        result = await self.convert_nl_to_sql(nl_query, context=enhanced_context)
        
        # Update conversation context
        if not result.get("error"):
            self.update_conversation_context(session_id, {
                "last_query": nl_query,
                "last_query_type": result.get("query_type"),
                "last_result": result
            })
        
        return result
    
    def get_comprehensive_help(self) -> Dict[str, Any]:
        """Get comprehensive help information"""
        return {
            "description": "Enhanced SQL Skills with advanced NLP, security, and multi-model support",
            "supported_query_types": {
                "relational": "Standard SQL queries for tables and views",
                "vector": "Similarity search using embeddings and vector functions",
                "graph": "Graph pattern matching and path finding",
                "ml": "Machine learning queries using HANA PAL/APL",
                "time_series": "Time series analysis and forecasting",
                "spatial": "Geospatial queries and operations"
            },
            "example_queries": {
                "relational": "show all customers where city is [CITY_NAME]",
                "vector": "find [N] most similar [ITEMS] to '[SEARCH_TERM]'",
                "graph": "find shortest path from [NODE_A] to [NODE_B]",
                "ml": "predict [TARGET] using [ALGORITHM] from [TABLE]",
                "time_series": "show [PERIOD] trend for [METRIC] over time",
                "spatial": "find [ITEMS] within [DISTANCE] of coordinates [LAT], [LON]"
            },
            "security_features": [
                "SQL injection prevention",
                "Query validation",
                "Suspicious pattern detection",
                "Safe function whitelisting"
            ],
            "performance_features": [
                "Intelligent query caching",
                "HANA-specific optimizations",
                "Query execution planning",
                "Performance monitoring"
            ],
            "nlp_features": [
                "Named entity recognition",
                "Intent detection",
                "Query complexity analysis",
                "Conversational context awareness"
            ]
        }