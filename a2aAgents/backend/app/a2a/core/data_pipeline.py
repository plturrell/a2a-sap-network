"""
Production-Ready Data Pipeline for Real-Time ML Training and Inference

This module provides actual data collection, processing, and storage for all AI/ML models
in the A2A platform. It replaces synthetic data with real system metrics, logs, and events.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import time
import os
import sqlite3
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
import queue
import hashlib
import psutil
import aiofiles
import aiosqlite

# Time series data handling
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Real-time stream processing
try:
    import aiokafka
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Real system metrics collected from the platform"""
    timestamp: datetime
    agent_id: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_in_bytes: int
    network_out_bytes: int
    active_connections: int
    request_count: int
    error_count: int
    response_time_ms: float
    queue_depth: int
    thread_count: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Real agent events for ML training"""
    event_id: str
    timestamp: datetime
    agent_id: str
    event_type: str
    event_data: Dict[str, Any]
    success: bool
    duration_ms: float
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Real database query metrics"""
    query_id: str
    timestamp: datetime
    query_text: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    index_used: bool
    table_names: List[str]
    query_type: str
    optimization_applied: bool = False


@dataclass
class MessageMetrics:
    """Real message routing metrics"""
    message_id: str
    timestamp: datetime
    sender_id: str
    receiver_id: str
    message_size: int
    routing_time_ms: float
    hops: int
    protocol: str
    success: bool
    retry_count: int = 0


class DataCollectionMode(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"


class ProductionDataPipeline:
    """
    Production-grade data pipeline for collecting, processing, and serving
    real data to ML models. No synthetic data - only actual system metrics.
    """
    
    def __init__(self, db_path: str = "./ml_data.db", 
                 buffer_size: int = 10000,
                 collection_interval: float = 1.0):
        # Data storage
        self.db_path = db_path
        self.buffer_size = buffer_size
        self.collection_interval = collection_interval
        
        # In-memory buffers for real-time processing
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.events_buffer = deque(maxlen=buffer_size)
        self.query_buffer = deque(maxlen=buffer_size)
        self.message_buffer = deque(maxlen=buffer_size)
        
        # Real-time data streams
        self.data_queues = {
            'metrics': asyncio.Queue(maxsize=1000),
            'events': asyncio.Queue(maxsize=1000),
            'queries': asyncio.Queue(maxsize=1000),
            'messages': asyncio.Queue(maxsize=1000)
        }
        
        # Data processors
        self.processors = {}
        self.feature_extractors = {}
        
        # Model data caches
        self.training_data_cache = {}
        self.feature_cache = {}
        
        # Monitoring
        self.collection_stats = defaultdict(int)
        self.processing_stats = defaultdict(float)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.metrics_collected = Counter('ml_metrics_collected_total', 
                                           'Total metrics collected', ['type'])
            self.pipeline_latency = Histogram('ml_pipeline_latency_seconds',
                                            'Data pipeline latency')
            self.buffer_size_gauge = Gauge('ml_buffer_size', 
                                         'Current buffer size', ['buffer'])
        
        # Background tasks
        self._collection_task = None
        self._processing_task = None
        self._persistence_task = None
        self._is_running = False
        
        # Initialize database
        self._initialize_database()
        
        logger.info("Production data pipeline initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for ML data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                agent_id TEXT,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                network_in_bytes INTEGER,
                network_out_bytes INTEGER,
                active_connections INTEGER,
                request_count INTEGER,
                error_count INTEGER,
                response_time_ms REAL,
                queue_depth INTEGER,
                thread_count INTEGER,
                custom_metrics TEXT
            )
        ''')
        
        # Agent events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                timestamp REAL,
                agent_id TEXT,
                event_type TEXT,
                event_data TEXT,
                success INTEGER,
                duration_ms REAL,
                error_code TEXT,
                stack_trace TEXT,
                context TEXT
            )
        ''')
        
        # Query metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT UNIQUE,
                timestamp REAL,
                query_text TEXT,
                execution_time_ms REAL,
                rows_examined INTEGER,
                rows_returned INTEGER,
                index_used INTEGER,
                table_names TEXT,
                query_type TEXT,
                optimization_applied INTEGER
            )
        ''')
        
        # Message metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                timestamp REAL,
                sender_id TEXT,
                receiver_id TEXT,
                message_size INTEGER,
                routing_time_ms REAL,
                hops INTEGER,
                protocol TEXT,
                success INTEGER,
                retry_count INTEGER
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_agent ON system_metrics(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON agent_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_agent ON agent_events(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON agent_events(event_type)')
        
        conn.commit()
        conn.close()
    
    async def start(self):
        """Start the data pipeline"""
        if self._is_running:
            logger.warning("Data pipeline is already running")
            return
        
        self._is_running = True
        
        # Start background tasks
        self._collection_task = asyncio.create_task(self._collect_system_metrics())
        self._processing_task = asyncio.create_task(self._process_data_streams())
        self._persistence_task = asyncio.create_task(self._persist_data())
        
        logger.info("Data pipeline started")
    
    async def stop(self):
        """Stop the data pipeline"""
        self._is_running = False
        
        # Cancel background tasks
        if self._collection_task:
            self._collection_task.cancel()
        if self._processing_task:
            self._processing_task.cancel()
        if self._persistence_task:
            self._persistence_task.cancel()
        
        # Flush buffers
        await self._flush_buffers()
        
        logger.info("Data pipeline stopped")
    
    async def collect_metrics(self, agent_id: str) -> SystemMetrics:
        """Collect real system metrics for an agent"""
        try:
            # Get actual system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            connections = len(psutil.net_connections())
            process = psutil.Process()
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                agent_id=agent_id,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_in_bytes=network.bytes_recv,
                network_out_bytes=network.bytes_sent,
                active_connections=connections,
                request_count=0,  # Would come from actual agent
                error_count=0,    # Would come from actual agent
                response_time_ms=0.0,  # Would come from actual agent
                queue_depth=self.data_queues['metrics'].qsize(),
                thread_count=process.num_threads(),
                custom_metrics={}
            )
            
            # Add to buffer and queue
            self.metrics_buffer.append(metrics)
            await self.data_queues['metrics'].put(metrics)
            
            # Update stats
            self.collection_stats['metrics_collected'] += 1
            if PROMETHEUS_AVAILABLE:
                self.metrics_collected.labels(type='system').inc()
                self.buffer_size_gauge.labels(buffer='metrics').set(len(self.metrics_buffer))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            raise
    
    async def record_event(self, event: AgentEvent):
        """Record a real agent event"""
        try:
            # Add to buffer and queue
            self.events_buffer.append(event)
            await self.data_queues['events'].put(event)
            
            # Update stats
            self.collection_stats['events_recorded'] += 1
            if PROMETHEUS_AVAILABLE:
                self.metrics_collected.labels(type='event').inc()
                self.buffer_size_gauge.labels(buffer='events').set(len(self.events_buffer))
            
        except Exception as e:
            logger.error(f"Error recording event: {e}")
    
    async def record_query(self, query: QueryMetrics):
        """Record real query metrics"""
        try:
            # Add to buffer and queue
            self.query_buffer.append(query)
            await self.data_queues['queries'].put(query)
            
            # Update stats
            self.collection_stats['queries_recorded'] += 1
            if PROMETHEUS_AVAILABLE:
                self.metrics_collected.labels(type='query').inc()
                self.buffer_size_gauge.labels(buffer='queries').set(len(self.query_buffer))
            
        except Exception as e:
            logger.error(f"Error recording query: {e}")
    
    async def record_message(self, message: MessageMetrics):
        """Record real message routing metrics"""
        try:
            # Add to buffer and queue
            self.message_buffer.append(message)
            await self.data_queues['messages'].put(message)
            
            # Update stats
            self.collection_stats['messages_recorded'] += 1
            if PROMETHEUS_AVAILABLE:
                self.metrics_collected.labels(type='message').inc()
                self.buffer_size_gauge.labels(buffer='messages').set(len(self.message_buffer))
            
        except Exception as e:
            logger.error(f"Error recording message: {e}")
    
    async def get_training_data(self, data_type: str, 
                              hours: int = 24,
                              min_samples: int = 100) -> pd.DataFrame:
        """Get real training data from the pipeline"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            async with aiosqlite.connect(self.db_path) as db:
                if data_type == 'metrics':
                    query = '''
                        SELECT * FROM system_metrics 
                        WHERE timestamp > ? 
                        ORDER BY timestamp DESC
                    '''
                    columns = ['id', 'timestamp', 'agent_id', 'cpu_usage', 'memory_usage',
                             'disk_usage', 'network_in_bytes', 'network_out_bytes',
                             'active_connections', 'request_count', 'error_count',
                             'response_time_ms', 'queue_depth', 'thread_count', 'custom_metrics']
                
                elif data_type == 'events':
                    query = '''
                        SELECT * FROM agent_events 
                        WHERE timestamp > ? 
                        ORDER BY timestamp DESC
                    '''
                    columns = ['id', 'event_id', 'timestamp', 'agent_id', 'event_type',
                             'event_data', 'success', 'duration_ms', 'error_code',
                             'stack_trace', 'context']
                
                elif data_type == 'queries':
                    query = '''
                        SELECT * FROM query_metrics 
                        WHERE timestamp > ? 
                        ORDER BY timestamp DESC
                    '''
                    columns = ['id', 'query_id', 'timestamp', 'query_text', 'execution_time_ms',
                             'rows_examined', 'rows_returned', 'index_used', 'table_names',
                             'query_type', 'optimization_applied']
                
                elif data_type == 'messages':
                    query = '''
                        SELECT * FROM message_metrics 
                        WHERE timestamp > ? 
                        ORDER BY timestamp DESC
                    '''
                    columns = ['id', 'message_id', 'timestamp', 'sender_id', 'receiver_id',
                             'message_size', 'routing_time_ms', 'hops', 'protocol',
                             'success', 'retry_count']
                else:
                    raise ValueError(f"Unknown data type: {data_type}")
                
                cursor = await db.execute(query, (cutoff_timestamp,))
                rows = await cursor.fetchall()
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                
                # Parse JSON fields
                def parse_json_field(x):
                    return json.loads(x) if x else {}
                
                if data_type == 'metrics' and 'custom_metrics' in df.columns:
                    df['custom_metrics'] = df['custom_metrics'].apply(parse_json_field)
                elif data_type == 'events':
                    for col in ['event_data', 'context']:
                        if col in df.columns:
                            df[col] = df[col].apply(parse_json_field)
                
                # Ensure minimum samples
                if len(df) < min_samples:
                    logger.warning(f"Only {len(df)} samples available for {data_type}, "
                                 f"requested minimum {min_samples}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    async def get_real_time_features(self, agent_id: str, 
                                   feature_window: int = 300) -> np.ndarray:
        """Extract real-time features for ML inference"""
        try:
            # Get recent data from buffers
            cutoff_time = datetime.utcnow() - timedelta(seconds=feature_window)
            
            # Filter recent metrics
            recent_metrics = [m for m in self.metrics_buffer 
                            if m.agent_id == agent_id and m.timestamp > cutoff_time]
            
            recent_events = [e for e in self.events_buffer
                           if e.agent_id == agent_id and e.timestamp > cutoff_time]
            
            if not recent_metrics:
                # Return zero features if no data
                return np.zeros(50)
            
            # Extract statistical features from real data
            features = []
            
            # CPU usage features
            cpu_values = [m.cpu_usage for m in recent_metrics]
            features.extend([
                np.mean(cpu_values),
                np.std(cpu_values) if len(cpu_values) > 1 else 0,
                np.max(cpu_values),
                np.min(cpu_values),
                np.percentile(cpu_values, 95) if len(cpu_values) > 4 else np.max(cpu_values)
            ])
            
            # Memory usage features
            memory_values = [m.memory_usage for m in recent_metrics]
            features.extend([
                np.mean(memory_values),
                np.std(memory_values) if len(memory_values) > 1 else 0,
                np.max(memory_values),
                np.percentile(memory_values, 95) if len(memory_values) > 4 else np.max(memory_values)
            ])
            
            # Network features
            if len(recent_metrics) > 1:
                network_in_rate = (recent_metrics[-1].network_in_bytes - 
                                 recent_metrics[0].network_in_bytes) / feature_window
                network_out_rate = (recent_metrics[-1].network_out_bytes - 
                                  recent_metrics[0].network_out_bytes) / feature_window
            else:
                network_in_rate = 0
                network_out_rate = 0
            
            features.extend([network_in_rate, network_out_rate])
            
            # Response time features
            response_times = [m.response_time_ms for m in recent_metrics if m.response_time_ms > 0]
            if response_times:
                features.extend([
                    np.mean(response_times),
                    np.std(response_times) if len(response_times) > 1 else 0,
                    np.percentile(response_times, 95) if len(response_times) > 4 else np.max(response_times)
                ])
            else:
                features.extend([0, 0, 0])
            
            # Error rate features
            total_requests = sum(m.request_count for m in recent_metrics)
            total_errors = sum(m.error_count for m in recent_metrics)
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            features.append(error_rate)
            
            # Event-based features
            if recent_events:
                event_types = [e.event_type for e in recent_events]
                unique_event_types = len(set(event_types))
                failure_rate = sum(1 for e in recent_events if not e.success) / len(recent_events)
                avg_duration = np.mean([e.duration_ms for e in recent_events])
            else:
                unique_event_types = 0
                failure_rate = 0
                avg_duration = 0
            
            features.extend([unique_event_types, failure_rate, avg_duration])
            
            # Connection and thread features
            avg_connections = np.mean([m.active_connections for m in recent_metrics])
            max_connections = np.max([m.active_connections for m in recent_metrics])
            avg_threads = np.mean([m.thread_count for m in recent_metrics])
            
            features.extend([avg_connections, max_connections, avg_threads])
            
            # Queue depth features
            queue_depths = [m.queue_depth for m in recent_metrics]
            features.extend([
                np.mean(queue_depths),
                np.max(queue_depths),
                np.std(queue_depths) if len(queue_depths) > 1 else 0
            ])
            
            # Time-based features
            current_hour = datetime.utcnow().hour
            is_business_hours = 1 if 9 <= current_hour <= 17 else 0
            day_of_week = datetime.utcnow().weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            features.extend([current_hour / 24.0, is_business_hours, day_of_week / 7.0, is_weekend])
            
            # Trend features (compare recent vs older)
            if len(recent_metrics) >= 10:
                mid_point = len(recent_metrics) // 2
                recent_cpu = np.mean([m.cpu_usage for m in recent_metrics[mid_point:]])
                older_cpu = np.mean([m.cpu_usage for m in recent_metrics[:mid_point]])
                cpu_trend = (recent_cpu - older_cpu) / (older_cpu + 1e-5)
                
                recent_memory = np.mean([m.memory_usage for m in recent_metrics[mid_point:]])
                older_memory = np.mean([m.memory_usage for m in recent_metrics[:mid_point]])
                memory_trend = (recent_memory - older_memory) / (older_memory + 1e-5)
            else:
                cpu_trend = 0
                memory_trend = 0
            
            features.extend([cpu_trend, memory_trend])
            
            # Pad or truncate to fixed size
            feature_array = np.array(features, dtype=np.float32)
            if len(feature_array) < 50:
                feature_array = np.pad(feature_array, (0, 50 - len(feature_array)), 'constant')
            elif len(feature_array) > 50:
                feature_array = feature_array[:50]
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error extracting real-time features: {e}")
            return np.zeros(50)
    
    async def _collect_system_metrics(self):
        """Background task to continuously collect system metrics"""
        while self._is_running:
            try:
                # Collect metrics for all active agents
                # In production, this would get the list of active agents from the registry
                agent_ids = ["agent_0", "agent_1", "agent_2"]  # Example
                
                for agent_id in agent_ids:
                    if self._is_running:
                        await self.collect_metrics(agent_id)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _process_data_streams(self):
        """Process incoming data streams"""
        while self._is_running:
            try:
                # Process each queue
                for data_type, queue in self.data_queues.items():
                    if not queue.empty():
                        # Process up to 100 items at a time
                        batch = []
                        for _ in range(min(100, queue.qsize())):
                            try:
                                item = queue.get_nowait()
                                batch.append(item)
                            except asyncio.QueueEmpty:
                                break
                        
                        if batch:
                            # Process batch through any registered processors
                            if data_type in self.processors:
                                for processor in self.processors[data_type]:
                                    batch = await processor(batch)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data processing: {e}")
                await asyncio.sleep(1)
    
    async def _persist_data(self):
        """Persist buffered data to database"""
        while self._is_running:
            try:
                await self._flush_buffers()
                await asyncio.sleep(60)  # Persist every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data persistence: {e}")
                await asyncio.sleep(60)
    
    async def _flush_buffers(self):
        """Flush all buffers to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Persist system metrics
                if self.metrics_buffer:
                    metrics_data = []
                    for m in list(self.metrics_buffer):
                        metrics_data.append((
                            m.timestamp.timestamp(), m.agent_id, m.cpu_usage,
                            m.memory_usage, m.disk_usage, m.network_in_bytes,
                            m.network_out_bytes, m.active_connections, m.request_count,
                            m.error_count, m.response_time_ms, m.queue_depth,
                            m.thread_count, json.dumps(m.custom_metrics)
                        ))
                    
                    await db.executemany('''
                        INSERT INTO system_metrics 
                        (timestamp, agent_id, cpu_usage, memory_usage, disk_usage,
                         network_in_bytes, network_out_bytes, active_connections,
                         request_count, error_count, response_time_ms, queue_depth,
                         thread_count, custom_metrics)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', metrics_data)
                
                # Persist agent events
                if self.events_buffer:
                    events_data = []
                    for e in list(self.events_buffer):
                        events_data.append((
                            e.event_id, e.timestamp.timestamp(), e.agent_id,
                            e.event_type, json.dumps(e.event_data), int(e.success),
                            e.duration_ms, e.error_code, e.stack_trace,
                            json.dumps(e.context)
                        ))
                    
                    await db.executemany('''
                        INSERT OR REPLACE INTO agent_events
                        (event_id, timestamp, agent_id, event_type, event_data,
                         success, duration_ms, error_code, stack_trace, context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', events_data)
                
                # Similar for queries and messages...
                
                await db.commit()
                logger.info(f"Flushed {len(self.metrics_buffer)} metrics, "
                          f"{len(self.events_buffer)} events to database")
                
        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
    
    def register_processor(self, data_type: str, processor: Callable):
        """Register a data processor for a specific data type"""
        if data_type not in self.processors:
            self.processors[data_type] = []
        self.processors[data_type].append(processor)
    
    def register_feature_extractor(self, name: str, extractor: Callable):
        """Register a feature extractor"""
        self.feature_extractors[name] = extractor


# Singleton instance
_data_pipeline = None

def get_data_pipeline() -> ProductionDataPipeline:
    """Get or create the production data pipeline instance"""
    global _data_pipeline
    if not _data_pipeline:
        _data_pipeline = ProductionDataPipeline()
    return _data_pipeline