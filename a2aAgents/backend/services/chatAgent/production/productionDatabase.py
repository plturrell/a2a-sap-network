"""
Production Database Layer for A2A Chat Agent
Supports SQLite for development and SAP HANA for production
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
import aiosqlite
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, MetaData, Table, Column, String, Integer, DateTime, Boolean, Text, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConversationModel(Base):
    """SQLAlchemy model for conversations"""
    __tablename__ = 'conversations'
    
    conversation_id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(String(500))
    type = Column(String(50), default='direct')
    participants = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)
    status = Column(String(50), default='active')
    settings = Column(JSON, default=dict)
    meta_data = Column(JSON, default=dict)
    message_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_user_status', 'user_id', 'status'),
        Index('idx_updated_at', 'updated_at'),
    )


class MessageModel(Base):
    """SQLAlchemy model for messages"""
    __tablename__ = 'messages'
    
    message_id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey('conversations.conversation_id'), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    parts = Column(JSON, default=list)
    task_id = Column(String(36))
    parent_message_id = Column(String(36), ForeignKey('messages.message_id'))
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime)
    status = Column(String(50), default='sent')
    meta_data = Column(JSON, default=dict)
    embedding = Column(JSON)  # For semantic search
    
    __table_args__ = (
        Index('idx_conv_created', 'conversation_id', 'created_at'),
        Index('idx_task_id', 'task_id'),
    )


class AgentResponseModel(Base):
    """SQLAlchemy model for agent responses"""
    __tablename__ = 'agent_responses'
    
    response_id = Column(String(36), primary_key=True)
    message_id = Column(String(36), ForeignKey('messages.message_id'), nullable=False, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    response_content = Column(Text, nullable=False)
    response_type = Column(String(50), default='text')
    processing_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime, nullable=False)
    meta_data = Column(JSON, default=dict)
    confidence_score = Column(Float)
    
    __table_args__ = (
        Index('idx_message_agent', 'message_id', 'agent_id'),
    )


class UserModel(Base):
    """SQLAlchemy model for users"""
    __tablename__ = 'users'
    
    user_id = Column(String(255), primary_key=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), index=True)
    created_at = Column(DateTime, nullable=False)
    last_active = Column(DateTime)
    settings = Column(JSON, default=dict)
    preferences = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    api_key_hash = Column(String(255))  # For API authentication
    rate_limit_tier = Column(String(50), default='standard')
    total_messages = Column(Integer, default=0)
    total_conversations = Column(Integer, default=0)


class MetricsModel(Base):
    """SQLAlchemy model for metrics and analytics"""
    __tablename__ = 'metrics'
    
    metric_id = Column(String(36), primary_key=True)
    metric_type = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Float, nullable=False)
    dimensions = Column(JSON, default=dict)
    timestamp = Column(DateTime, nullable=False, index=True)
    user_id = Column(String(255), index=True)
    conversation_id = Column(String(36), index=True)
    
    __table_args__ = (
        Index('idx_metric_type_time', 'metric_type', 'timestamp'),
        Index('idx_user_metrics', 'user_id', 'metric_type', 'timestamp'),
    )


class ProductionDatabase:
    """
    Production-grade database layer with support for multiple backends
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_type = config.get('type', 'sqlite')  # sqlite, postgresql, hana
        self.connection_string = config.get('connection_string')
        self.pool_size = config.get('pool_size', 20)
        self.max_overflow = config.get('max_overflow', 10)
        self.echo_sql = config.get('echo_sql', False)
        
        self.engine: Optional[AsyncEngine] = None
        self.async_session: Optional[sessionmaker] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize database connection and create tables"""
        if self._initialized:
            return
            
        try:
            # Build connection string based on type
            if self.db_type == 'sqlite':
                if not self.connection_string:
                    self.connection_string = 'sqlite+aiosqlite:///production_chat.db'
            elif self.db_type == 'postgresql':
                if not self.connection_string:
                    # Use environment variables for database configuration
                    db_host = os.getenv('POSTGRES_HOST', 'localhost')
                    db_port = os.getenv('POSTGRES_PORT', '5432')
                    db_user = os.getenv('POSTGRES_USER', 'user')
                    db_pass = os.getenv('POSTGRES_PASSWORD', 'pass')
                    db_name = os.getenv('POSTGRES_DB', 'a2a_chat')
                    self.connection_string = f'postgresql+asyncpg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}'
            elif self.db_type == 'hana':
                # SAP HANA connection
                if not self.connection_string:
                    raise ValueError("HANA connection string required")
            
            # Create async engine with appropriate settings for each database type
            engine_kwargs = {
                'echo': self.echo_sql,
            }
            
            if self.db_type == 'sqlite':
                # SQLite doesn't support pool settings
                engine_kwargs.update({
                    'pool_pre_ping': True,
                })
            else:
                # PostgreSQL and other databases support connection pooling
                engine_kwargs.update({
                    'pool_size': self.pool_size,
                    'max_overflow': self.max_overflow,
                    'pool_pre_ping': True,  # Verify connections
                    'pool_recycle': 3600,   # Recycle connections after 1 hour
                })
            
            self.engine = create_async_engine(self.connection_string, **engine_kwargs)
            
            # Create session factory
            self.async_session = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # Create additional indexes for performance
            await self._create_custom_indexes()
            
            self._initialized = True
            logger.info(f"Production database initialized: {self.db_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _create_custom_indexes(self):
        """Create custom indexes for performance optimization"""
        try:
            async with self.engine.begin() as conn:
                # Full-text search index for messages (PostgreSQL)
                if self.db_type == 'postgresql':
                    await conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_messages_content_fts 
                        ON messages USING gin(to_tsvector('english', content))
                    """))
                
                # Composite indexes for common queries
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_conv_user_updated 
                    ON conversations(user_id, updated_at DESC)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_composite 
                    ON metrics(user_id, metric_type, timestamp DESC)
                """))
                
        except Exception as e:
            logger.warning(f"Some custom indexes may not have been created: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup"""
        if not self.async_session:
            raise RuntimeError("Database not initialized")
            
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    # User Management
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user"""
        async with self.get_session() as session:
            user = UserModel(
                user_id=user_data['user_id'],
                username=user_data['username'],
                email=user_data.get('email'),
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                settings=user_data.get('settings', {}),
                preferences=user_data.get('preferences', {}),
                api_key_hash=user_data.get('api_key_hash'),
                rate_limit_tier=user_data.get('rate_limit_tier', 'standard')
            )
            session.add(user)
            await session.flush()
            return user.user_id
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE user_id = :user_id"),
                {"user_id": user_id}
            )
            row = result.first()
            return dict(row) if row else None
    
    async def get_user_by_api_key(self, api_key_hash: str) -> Optional[Dict[str, Any]]:
        """Get user by API key hash"""
        async with self.get_session() as session:
            result = await session.execute(
                text("SELECT * FROM users WHERE api_key_hash = :api_key_hash"),
                {"api_key_hash": api_key_hash}
            )
            row = result.first()
            return dict(row) if row else None
    
    async def update_user_activity(self, user_id: str):
        """Update user's last active timestamp"""
        async with self.get_session() as session:
            await session.execute(
                text("UPDATE users SET last_active = :now WHERE user_id = :user_id"),
                {"now": datetime.utcnow(), "user_id": user_id}
            )
    
    # Conversation Management
    async def create_conversation(self, conv_data: Dict[str, Any]) -> str:
        """Create a new conversation"""
        async with self.get_session() as session:
            conv = ConversationModel(
                conversation_id=conv_data['conversation_id'],
                user_id=conv_data['user_id'],
                title=conv_data.get('title'),
                type=conv_data.get('type', 'direct'),
                participants=conv_data.get('participants', []),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                settings=conv_data.get('settings', {}),
                meta_data=conv_data.get('metadata', {})
            )
            session.add(conv)
            
            # Update user stats
            await session.execute(
                text("UPDATE users SET total_conversations = total_conversations + 1 WHERE user_id = :user_id"),
                {"user_id": conv_data['user_id']}
            )
            
            await session.flush()
            return conv.conversation_id
    
    async def get_conversations(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user's conversations with pagination"""
        async with self.get_session() as session:
            query = """
                SELECT c.*, 
                       COUNT(DISTINCT m.message_id) as message_count,
                       MAX(m.created_at) as last_message_at
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.user_id = :user_id
            """
            params = {"user_id": user_id}
            
            if status:
                query += " AND c.status = :status"
                params["status"] = status
            
            query += """
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT :limit OFFSET :offset
            """
            params.update({"limit": limit, "offset": offset})
            
            result = await session.execute(text(query), params)
            return [dict(row) for row in result]
    
    # Message Management
    async def save_message(self, message_data: Dict[str, Any]) -> str:
        """Save a message to the database"""
        async with self.get_session() as session:
            msg = MessageModel(
                message_id=message_data['message_id'],
                conversation_id=message_data['conversation_id'],
                role=message_data['role'],
                content=message_data['content'],
                parts=message_data.get('parts', []),
                task_id=message_data.get('task_id'),
                parent_message_id=message_data.get('parent_message_id'),
                created_at=datetime.utcnow(),
                status=message_data.get('status', 'sent'),
                meta_data=message_data.get('metadata', {}),
                embedding=message_data.get('embedding')
            )
            session.add(msg)
            
            # Update conversation
            await session.execute(
                text("""
                    UPDATE conversations 
                    SET updated_at = :now, 
                        message_count = message_count + 1 
                    WHERE conversation_id = :conv_id
                """),
                {"now": datetime.utcnow(), "conv_id": message_data['conversation_id']}
            )
            
            # Update user stats
            await session.execute(
                text("""
                    UPDATE users 
                    SET total_messages = total_messages + 1,
                        last_active = :now
                    WHERE user_id = (
                        SELECT user_id FROM conversations 
                        WHERE conversation_id = :conv_id
                    )
                """),
                {"now": datetime.utcnow(), "conv_id": message_data['conversation_id']}
            )
            
            await session.flush()
            return msg.message_id
    
    async def get_messages(
        self, 
        conversation_id: str, 
        limit: int = 100, 
        offset: int = 0,
        include_responses: bool = True
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        async with self.get_session() as session:
            # Get messages
            msg_result = await session.execute(
                text("""
                    SELECT * FROM messages 
                    WHERE conversation_id = :conv_id 
                    ORDER BY created_at ASC 
                    LIMIT :limit OFFSET :offset
                """),
                {"conv_id": conversation_id, "limit": limit, "offset": offset}
            )
            
            messages = []
            for row in msg_result:
                message = dict(row)
                
                if include_responses:
                    # Get agent responses
                    resp_result = await session.execute(
                        text("""
                            SELECT * FROM agent_responses 
                            WHERE message_id = :msg_id 
                            ORDER BY created_at ASC
                        """),
                        {"msg_id": message['message_id']}
                    )
                    message['agent_responses'] = [dict(r) for r in resp_result]
                
                messages.append(message)
            
            return messages
    
    # Agent Response Management
    async def save_agent_response(self, response_data: Dict[str, Any]) -> str:
        """Save an agent response"""
        async with self.get_session() as session:
            resp = AgentResponseModel(
                response_id=response_data['response_id'],
                message_id=response_data['message_id'],
                agent_id=response_data['agent_id'],
                response_content=response_data['response_content'],
                response_type=response_data.get('response_type', 'text'),
                processing_time_ms=response_data.get('processing_time_ms'),
                success=response_data.get('success', True),
                error_message=response_data.get('error_message'),
                created_at=datetime.utcnow(),
                meta_data=response_data.get('metadata', {}),
                confidence_score=response_data.get('confidence_score')
            )
            session.add(resp)
            await session.flush()
            return resp.response_id
    
    # Search and Analytics
    async def search_conversations(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        async with self.get_session() as session:
            if self.db_type == 'postgresql':
                # Use PostgreSQL full-text search
                search_query = """
                    SELECT DISTINCT c.*, 
                           ts_rank(to_tsvector('english', m.content), 
                                  plainto_tsquery('english', :query)) as rank
                    FROM conversations c
                    JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE c.user_id = :user_id
                      AND to_tsvector('english', m.content) @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC, c.updated_at DESC
                    LIMIT :limit
                """
            else:
                # Basic LIKE search for SQLite
                search_query = """
                    SELECT DISTINCT c.*
                    FROM conversations c
                    JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE c.user_id = :user_id
                      AND (c.title LIKE :query_pattern OR m.content LIKE :query_pattern)
                    ORDER BY c.updated_at DESC
                    LIMIT :limit
                """
                
            params = {
                "user_id": user_id,
                "query": query,
                "query_pattern": f"%{query}%",
                "limit": limit
            }
            
            result = await session.execute(text(search_query), params)
            return [dict(row) for row in result]
    
    # Metrics and Analytics
    async def record_metric(self, metric_data: Dict[str, Any]):
        """Record a metric for analytics"""
        async with self.get_session() as session:
            from uuid import uuid4
            metric = MetricsModel(
                metric_id=metric_data.get('metric_id', str(uuid4())),
                metric_type=metric_data['metric_type'],
                metric_name=metric_data['metric_name'],
                metric_value=metric_data['metric_value'],
                dimensions=metric_data.get('dimensions', {}),
                timestamp=datetime.utcnow(),
                user_id=metric_data.get('user_id'),
                conversation_id=metric_data.get('conversation_id')
            )
            session.add(metric)
    
    async def get_metrics(
        self,
        metric_type: str,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        aggregation: str = 'sum'
    ) -> List[Dict[str, Any]]:
        """Get aggregated metrics"""
        async with self.get_session() as session:
            query = f"""
                SELECT 
                    metric_name,
                    {aggregation}(metric_value) as value,
                    COUNT(*) as count,
                    DATE_TRUNC('hour', timestamp) as hour
                FROM metrics
                WHERE metric_type = :metric_type
                  AND timestamp BETWEEN :start_time AND :end_time
            """
            params = {
                "metric_type": metric_type,
                "start_time": start_time,
                "end_time": end_time
            }
            
            if user_id:
                query += " AND user_id = :user_id"
                params["user_id"] = user_id
            
            query += " GROUP BY metric_name, hour ORDER BY hour DESC"
            
            result = await session.execute(text(query), params)
            return [dict(row) for row in result]
    
    # Cleanup and Maintenance
    async def cleanup_old_data(self, days_old: int = 90):
        """Clean up old archived data"""
        async with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Delete old archived conversations and their messages
            await session.execute(
                text("""
                    DELETE FROM agent_responses 
                    WHERE message_id IN (
                        SELECT m.message_id FROM messages m
                        JOIN conversations c ON m.conversation_id = c.conversation_id
                        WHERE c.status = 'archived' AND c.updated_at < :cutoff
                    )
                """),
                {"cutoff": cutoff_date}
            )
            
            await session.execute(
                text("""
                    DELETE FROM messages 
                    WHERE conversation_id IN (
                        SELECT conversation_id FROM conversations
                        WHERE status = 'archived' AND updated_at < :cutoff
                    )
                """),
                {"cutoff": cutoff_date}
            )
            
            result = await session.execute(
                text("""
                    DELETE FROM conversations
                    WHERE status = 'archived' AND updated_at < :cutoff
                    RETURNING conversation_id
                """),
                {"cutoff": cutoff_date}
            )
            
            deleted_count = result.rowcount
            logger.info(f"Cleaned up {deleted_count} old conversations")
            
            # Clean up old metrics
            await session.execute(
                text("DELETE FROM metrics WHERE timestamp < :cutoff"),
                {"cutoff": cutoff_date}
            )
            
            return deleted_count
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.get_session() as session:
            stats = {}
            
            # Table sizes
            for table in ['users', 'conversations', 'messages', 'agent_responses', 'metrics']:
                result = await session.execute(
                    text(f"SELECT COUNT(*) as count FROM {table}")
                )
                stats[f"{table}_count"] = result.scalar()
            
            # Active users (last 24h)
            result = await session.execute(
                text("""
                    SELECT COUNT(DISTINCT user_id) as active_users 
                    FROM users 
                    WHERE last_active > :cutoff
                """),
                {"cutoff": datetime.utcnow() - timedelta(days=1)}
            )
            stats['active_users_24h'] = result.scalar()
            
            # Message volume (last 24h)
            result = await session.execute(
                text("""
                    SELECT COUNT(*) as message_count 
                    FROM messages 
                    WHERE created_at > :cutoff
                """),
                {"cutoff": datetime.utcnow() - timedelta(days=1)}
            )
            stats['messages_24h'] = result.scalar()
            
            return stats
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Factory function
def create_production_database(config: Dict[str, Any]) -> ProductionDatabase:
    """Create production database instance based on configuration"""
    return ProductionDatabase(config)