"""
Advanced SQL query optimization and caching system for A2A SQL Agent.
Provides intelligent query optimization, result caching, and performance monitoring.
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import OrderedDict
from sqlparse import parse, format as sql_format
from sqlparse.sql import Statement, IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML

from config.agentConfig import config
from common.errorHandling import with_circuit_breaker, with_retry
from monitoring.prometheusConfig import create_agent_metrics

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """SQL query types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    UNKNOWN = "unknown"


class OptimizationLevel(Enum):
    """Query optimization levels."""
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class QueryAnalysis:
    """Query analysis result."""
    query_hash: str
    query_type: QueryType
    tables: List[str]
    columns: List[str]
    joins: List[str]
    where_conditions: List[str]
    complexity_score: float
    estimated_cost: float
    optimization_suggestions: List[str] = field(default_factory=list)
    index_recommendations: List[str] = field(default_factory=list)


@dataclass
class CachedResult:
    """Cached query result."""
    query_hash: str
    result: Any
    timestamp: datetime
    execution_time: float
    hit_count: int = 0
    size_bytes: int = 0


@dataclass
class QueryPlan:
    """Query execution plan."""
    original_query: str
    optimized_query: str
    execution_plan: Dict[str, Any]
    estimated_rows: int
    estimated_cost: float
    optimization_applied: List[str]


class LRUCache:
    """
    Least Recently Used cache implementation for query results.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of cached items
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
    
    def get(self, key: str) -> Optional[CachedResult]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                cached_item = self.cache.pop(key)
                self.cache[key] = cached_item
                cached_item.hit_count += 1
                self.stats['hits'] += 1
                return cached_item
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, result: CachedResult):
        """Put item in cache."""
        with self.lock:
            # Calculate result size
            result_size = self._calculate_size(result.result)
            result.size_bytes = result_size
            
            # Remove existing item if present
            if key in self.cache:
                old_item = self.cache.pop(key)
                self.current_memory -= old_item.size_bytes
            
            # Check memory limits
            while (self.current_memory + result_size > self.max_memory_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Check size limits
            while len(self.cache) >= self.max_size and len(self.cache) > 0:
                self._evict_lru()
            
            # Add new item
            self.cache[key] = result
            self.current_memory += result_size
            self.stats['memory_usage'] = self.current_memory
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            _, evicted_item = self.cache.popitem(last=False)
            self.current_memory -= evicted_item.size_bytes
            self.stats['evictions'] += 1
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object."""
        try:
            return len(json.dumps(obj, default=str).encode('utf-8'))
        except:
            # Fallback estimation
            return len(str(obj).encode('utf-8'))
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
            self.stats['memory_usage'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'memory_usage_mb': self.current_memory / (1024 * 1024)
            }


class SQLQueryOptimizer:
    """
    Advanced SQL query optimizer with caching and performance monitoring.
    """
    
    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_memory_mb: int = 512,
        enable_metrics: bool = True
    ):
        """
        Initialize SQL query optimizer.
        
        Args:
            optimization_level: Level of optimization to apply
            enable_caching: Enable result caching
            cache_size: Maximum number of cached queries
            cache_memory_mb: Maximum cache memory in MB
            enable_metrics: Enable performance metrics
        """
        self.optimization_level = optimization_level
        self.enable_caching = enable_caching
        self.enable_metrics = enable_metrics
        
        # Initialize cache
        if enable_caching:
            self.result_cache = LRUCache(cache_size, cache_memory_mb)
        else:
            self.result_cache = None
        
        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'cached_queries': 0,
            'optimized_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Optimization rules
        self.optimization_rules = self._load_optimization_rules()
        
        # Performance metrics
        if enable_metrics:
            self.metrics = create_agent_metrics("sql_agent", "query_optimizer")
        else:
            self.metrics = None
        
        # Index recommendations cache
        self.index_recommendations = {}
        
        logger.info(f"SQL Query Optimizer initialized with {optimization_level.value} optimization")
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load SQL optimization rules."""
        return {
            'select_star_warning': {
                'pattern': r'SELECT\s+\*\s+FROM',
                'suggestion': 'Specify explicit column names instead of SELECT *',
                'severity': 'medium'
            },
            'missing_where_clause': {
                'pattern': r'SELECT\s+.*\s+FROM\s+\w+(?!\s+WHERE)',
                'suggestion': 'Consider adding WHERE clause to limit results',
                'severity': 'low'
            },
            'inefficient_like': {
                'pattern': r'LIKE\s+[\'"]%.*%[\'"]',
                'suggestion': 'Leading wildcards in LIKE prevent index usage',
                'severity': 'high'
            },
            'subquery_optimization': {
                'pattern': r'IN\s*\(\s*SELECT',
                'suggestion': 'Consider using EXISTS instead of IN with subquery',
                'severity': 'medium'
            },
            'function_in_where': {
                'pattern': r'WHERE\s+\w+\s*\(',
                'suggestion': 'Functions in WHERE clause prevent index usage',
                'severity': 'high'
            }
        }
    
    async def optimize_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, QueryAnalysis, List[str]]:
        """
        Optimize SQL query and provide analysis.
        
        Args:
            query: SQL query to optimize
            parameters: Query parameters
            
        Returns:
            Tuple of (optimized_query, analysis, warnings)
        """
        start_time = time.time()
        
        try:
            # Parse and analyze query
            analysis = await self._analyze_query(query)
            
            # Apply optimizations
            optimized_query = await self._apply_optimizations(query, analysis)
            
            # Generate warnings and suggestions
            warnings = self._generate_warnings(query, analysis)
            
            # Update statistics
            self.query_stats['total_queries'] += 1
            if optimized_query != query:
                self.query_stats['optimized_queries'] += 1
            
            optimization_time = time.time() - start_time
            
            if self.metrics:
                self.metrics.record_task("query_optimization", "success", optimization_time)
            
            logger.debug(f"Query optimization completed in {optimization_time:.3f}s")
            
            return optimized_query, analysis, warnings
            
        except Exception as e:
            self.query_stats['failed_queries'] += 1
            
            if self.metrics:
                self.metrics.record_task("query_optimization", "failure")
            
            logger.error(f"Query optimization failed: {e}")
            raise
    
    async def execute_with_cache(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        executor_func: callable = None,
        cache_ttl: int = 3600
    ) -> Tuple[Any, bool]:
        """
        Execute query with caching support.
        
        Args:
            query: SQL query
            parameters: Query parameters
            executor_func: Function to execute the query
            cache_ttl: Cache TTL in seconds
            
        Returns:
            Tuple of (result, from_cache)
        """
        if not self.enable_caching or not executor_func:
            # Execute without caching
            result = await executor_func(query, parameters)
            return result, False
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, parameters)
        
        # Check cache first
        cached_result = self.result_cache.get(cache_key)
        if cached_result and self._is_cache_valid(cached_result, cache_ttl):
            self.query_stats['cached_queries'] += 1
            
            if self.metrics:
                self.metrics.record_validation("cache_lookup", "hit")
            
            return cached_result.result, True
        
        # Execute query
        start_time = time.time()
        result = await executor_func(query, parameters)
        execution_time = time.time() - start_time
        
        # Cache result
        if self.result_cache:
            cached_result = CachedResult(
                query_hash=cache_key,
                result=result,
                timestamp=datetime.now(),
                execution_time=execution_time
            )
            self.result_cache.put(cache_key, cached_result)
        
        # Update statistics
        self.query_stats['total_execution_time'] += execution_time
        
        if self.metrics:
            self.metrics.record_validation("cache_lookup", "miss")
            self.metrics.record_task("query_execution", "success", execution_time)
        
        return result, False
    
    async def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze SQL query structure and complexity."""
        # Parse SQL
        parsed = parse(query)[0]
        
        # Extract query type
        query_type = self._extract_query_type(parsed)
        
        # Extract tables
        tables = self._extract_tables(parsed)
        
        # Extract columns
        columns = self._extract_columns(parsed)
        
        # Extract joins
        joins = self._extract_joins(parsed)
        
        # Extract WHERE conditions
        where_conditions = self._extract_where_conditions(parsed)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            tables, columns, joins, where_conditions
        )
        
        # Estimate cost
        estimated_cost = self._estimate_query_cost(
            query_type, tables, joins, where_conditions
        )
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(query, parsed)
        
        # Generate index recommendations
        index_recommendations = self._generate_index_recommendations(
            tables, columns, where_conditions, joins
        )
        
        # Generate query hash
        query_hash = self._generate_cache_key(query)
        
        return QueryAnalysis(
            query_hash=query_hash,
            query_type=query_type,
            tables=tables,
            columns=columns,
            joins=joins,
            where_conditions=where_conditions,
            complexity_score=complexity_score,
            estimated_cost=estimated_cost,
            optimization_suggestions=optimization_suggestions,
            index_recommendations=index_recommendations
        )
    
    def _extract_query_type(self, parsed: Statement) -> QueryType:
        """Extract query type from parsed SQL."""
        for token in parsed.tokens:
            if token.ttype is DML:
                query_type_str = token.value.upper()
                if query_type_str == 'SELECT':
                    return QueryType.SELECT
                elif query_type_str == 'INSERT':
                    return QueryType.INSERT
                elif query_type_str == 'UPDATE':
                    return QueryType.UPDATE
                elif query_type_str == 'DELETE':
                    return QueryType.DELETE
            elif token.ttype is Keyword:
                keyword = token.value.upper()
                if keyword == 'CREATE':
                    return QueryType.CREATE
                elif keyword == 'DROP':
                    return QueryType.DROP
                elif keyword == 'ALTER':
                    return QueryType.ALTER
        
        return QueryType.UNKNOWN
    
    def _extract_tables(self, parsed: Statement) -> List[str]:
        """Extract table names from parsed SQL."""
        tables = []
        from_seen = False
        
        for token in parsed.flatten():
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif from_seen and token.ttype is None:
                # This is likely a table name
                table_name = token.value.strip()
                if table_name and not table_name.upper() in ['WHERE', 'GROUP', 'ORDER', 'HAVING']:
                    tables.append(table_name)
                    from_seen = False
        
        return tables
    
    def _extract_columns(self, parsed: Statement) -> List[str]:
        """Extract column names from parsed SQL."""
        columns = []
        select_seen = False
        
        for token in parsed.flatten():
            if token.ttype is DML and token.value.upper() == 'SELECT':
                select_seen = True
            elif select_seen and token.ttype is Keyword and token.value.upper() == 'FROM':
                break
            elif select_seen and token.ttype is None:
                column_name = token.value.strip().rstrip(',')
                if column_name and column_name != '*':
                    columns.append(column_name)
        
        return columns
    
    def _extract_joins(self, parsed: Statement) -> List[str]:
        """Extract JOIN clauses from parsed SQL."""
        joins = []
        query_str = str(parsed)
        
        # Find JOIN patterns
        join_patterns = [
            r'INNER\s+JOIN\s+(\w+)',
            r'LEFT\s+JOIN\s+(\w+)',
            r'RIGHT\s+JOIN\s+(\w+)',
            r'FULL\s+JOIN\s+(\w+)',
            r'JOIN\s+(\w+)'
        ]
        
        for pattern in join_patterns:
            matches = re.finditer(pattern, query_str, re.IGNORECASE)
            for match in matches:
                joins.append(match.group())
        
        return joins
    
    def _extract_where_conditions(self, parsed: Statement) -> List[str]:
        """Extract WHERE conditions from parsed SQL."""
        conditions = []
        query_str = str(parsed)
        
        # Simple WHERE extraction
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', 
                               query_str, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR but this is simplified
            conditions = [cond.strip() for cond in re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE)]
        
        return conditions
    
    def _calculate_complexity_score(
        self,
        tables: List[str],
        columns: List[str],
        joins: List[str],
        where_conditions: List[str]
    ) -> float:
        """Calculate query complexity score."""
        score = 0.0
        
        # Base score for number of tables
        score += len(tables) * 1.0
        
        # Add score for joins
        score += len(joins) * 2.0
        
        # Add score for WHERE conditions
        score += len(where_conditions) * 1.5
        
        # Add score for SELECT *
        if '*' in columns:
            score += 1.0
        else:
            score += len(columns) * 0.1
        
        return score
    
    def _estimate_query_cost(
        self,
        query_type: QueryType,
        tables: List[str],
        joins: List[str],
        where_conditions: List[str]
    ) -> float:
        """Estimate query execution cost."""
        base_cost = 1.0
        
        if query_type == QueryType.SELECT:
            # Cost increases with joins
            cost = base_cost + (len(joins) * 10.0)
            
            # Cost increases with tables
            cost += len(tables) * 5.0
            
            # Cost decreases with WHERE conditions (more selective)
            cost = cost / max(len(where_conditions), 1)
            
        elif query_type in [QueryType.INSERT, QueryType.UPDATE, QueryType.DELETE]:
            cost = base_cost + (len(tables) * 3.0)
            
        else:
            cost = base_cost
        
        return cost
    
    async def _apply_optimizations(self, query: str, analysis: QueryAnalysis) -> str:
        """Apply optimization rules to query."""
        optimized_query = query
        
        if self.optimization_level == OptimizationLevel.BASIC:
            # Basic optimizations only
            optimized_query = self._apply_basic_optimizations(optimized_query)
            
        elif self.optimization_level == OptimizationLevel.MODERATE:
            # Basic + moderate optimizations
            optimized_query = self._apply_basic_optimizations(optimized_query)
            optimized_query = self._apply_moderate_optimizations(optimized_query, analysis)
            
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # All optimizations
            optimized_query = self._apply_basic_optimizations(optimized_query)
            optimized_query = self._apply_moderate_optimizations(optimized_query, analysis)
            optimized_query = self._apply_aggressive_optimizations(optimized_query, analysis)
        
        return optimized_query
    
    def _apply_basic_optimizations(self, query: str) -> str:
        """Apply basic SQL optimizations."""
        # Remove unnecessary whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Format SQL for readability
        try:
            query = sql_format(query, reindent=True, keyword_case='upper')
        except:
            pass  # If formatting fails, use original
        
        return query
    
    def _apply_moderate_optimizations(self, query: str, analysis: QueryAnalysis) -> str:
        """Apply moderate SQL optimizations."""
        # Convert IN subqueries to EXISTS when beneficial
        if analysis.query_type == QueryType.SELECT:
            query = re.sub(
                r'(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)(\s+WHERE\s+[^)]+)?\)',
                r'EXISTS (SELECT 1 FROM \3 WHERE \3.\2 = \1\4)',
                query,
                flags=re.IGNORECASE
            )
        
        return query
    
    def _apply_aggressive_optimizations(self, query: str, analysis: QueryAnalysis) -> str:
        """Apply aggressive SQL optimizations."""
        # Replace SELECT * with specific columns if we know the schema
        # This would require schema knowledge, so skip for now
        
        # Add LIMIT if none exists for potentially large result sets
        if (analysis.query_type == QueryType.SELECT and 
            'LIMIT' not in query.upper() and 
            len(analysis.where_conditions) == 0):
            
            # Only add LIMIT if there are no WHERE conditions (could be dangerous)
            # query += " LIMIT 1000"  # Commented out for safety
            pass
        
        return query
    
    def _generate_optimization_suggestions(self, query: str, parsed: Statement) -> List[str]:
        """Generate optimization suggestions for query."""
        suggestions = []
        
        for rule_name, rule in self.optimization_rules.items():
            if re.search(rule['pattern'], query, re.IGNORECASE):
                suggestions.append(f"{rule['suggestion']} (Severity: {rule['severity']})")
        
        return suggestions
    
    def _generate_index_recommendations(
        self,
        tables: List[str],
        columns: List[str],
        where_conditions: List[str],
        joins: List[str]
    ) -> List[str]:
        """Generate index recommendations."""
        recommendations = []
        
        # Recommend indexes for WHERE clause columns
        for condition in where_conditions:
            # Extract column names from conditions (simplified)
            column_match = re.match(r'(\w+)\s*[=<>!]', condition)
            if column_match:
                column = column_match.group(1)
                recommendations.append(f"Consider index on column: {column}")
        
        # Recommend indexes for JOIN columns
        for join in joins:
            # This is simplified - would need better JOIN analysis
            recommendations.append(f"Consider indexes for JOIN: {join}")
        
        return recommendations
    
    def _generate_warnings(self, query: str, analysis: QueryAnalysis) -> List[str]:
        """Generate warnings for potentially problematic queries."""
        warnings = []
        
        # Warn about SELECT *
        if '*' in analysis.columns:
            warnings.append("Using SELECT * may impact performance")
        
        # Warn about missing WHERE clause
        if analysis.query_type == QueryType.SELECT and not analysis.where_conditions:
            warnings.append("Query has no WHERE clause - may return large result set")
        
        # Warn about complex queries
        if analysis.complexity_score > 10:
            warnings.append(f"High complexity query (score: {analysis.complexity_score:.1f})")
        
        # Warn about estimated high cost
        if analysis.estimated_cost > 50:
            warnings.append(f"High estimated cost query (cost: {analysis.estimated_cost:.1f})")
        
        return warnings
    
    def _generate_cache_key(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query and parameters."""
        # Normalize query (remove extra whitespace, convert to lowercase)
        normalized_query = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Include parameters in hash
        cache_data = {
            'query': normalized_query,
            'parameters': parameters or {}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: CachedResult, ttl: int) -> bool:
        """Check if cached result is still valid."""
        age_seconds = (datetime.now() - cached_result.timestamp).total_seconds()
        return age_seconds <= ttl
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.result_cache:
            cache_stats = self.result_cache.get_stats()
        else:
            cache_stats = {'cache_disabled': True}
        
        return {
            'query_stats': self.query_stats,
            'cache_stats': cache_stats,
            'optimization_level': self.optimization_level.value
        }
    
    def clear_cache(self):
        """Clear query result cache."""
        if self.result_cache:
            self.result_cache.clear()
            logger.info("Query cache cleared")
    
    def get_index_recommendations_summary(self) -> Dict[str, List[str]]:
        """Get summary of index recommendations."""
        return dict(self.index_recommendations)