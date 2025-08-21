"""
Production-grade persistent storage for A2A Chat Agent
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager

from ..sdk.types import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class ChatStorage:
    """
    Production-grade persistent storage for chat conversations
    Supports SQLite with async operations, connection pooling, and data integrity
    """
    
    def __init__(
        self,
        database_path: str = "a2a_chat.db",
        max_connections: int = 10,
        enable_encryption: bool = False,
        encryption_key: Optional[str] = None
    ):
        self.database_path = database_path
        self.max_connections = max_connections
        self.enable_encryption = enable_encryption
        self.encryption_key = encryption_key
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Initialize database and create tables"""
        if self._initialized:
            return
            
        # Create database schema
        async with aiosqlite.connect(self.database_path) as db:
            await self._create_tables(db)
            await db.commit()
        
        # Initialize connection pool
        await self._init_connection_pool()
        self._initialized = True
        logger.info(f"ChatStorage initialized with database: {self.database_path}")
    
    async def _create_tables(self, db: aiosqlite.Connection):
        """Create database tables for chat storage"""
        
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT,
                settings TEXT DEFAULT '{}',
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Conversations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                type TEXT DEFAULT 'direct',
                participants TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                settings TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                parts TEXT DEFAULT '[]',
                task_id TEXT,
                parent_message_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                status TEXT DEFAULT 'sent',
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
                FOREIGN KEY (parent_message_id) REFERENCES messages (message_id)
            )
        """)
        
        # Agent responses table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agent_responses (
                response_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                response_content TEXT NOT NULL,
                response_type TEXT DEFAULT 'text',
                processing_time_ms INTEGER,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (message_id) REFERENCES messages (message_id)
            )
        """)
        
        # Chat sessions table (for grouping related conversations)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_name TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                conversation_count INTEGER DEFAULT 0,
                settings TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        """)
        
        # Create indexes for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_responses_message_id ON agent_responses(message_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id)")
    
    async def _init_connection_pool(self):
        """Initialize connection pool for better performance"""
        async with self._pool_lock:
            for _ in range(self.max_connections):
                conn = await aiosqlite.connect(self.database_path)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                await conn.execute("PRAGMA cache_size=10000")
                self._connection_pool.append(conn)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        async with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
            else:
                conn = await aiosqlite.connect(self.database_path)
        
        try:
            yield conn
        finally:
            async with self._pool_lock:
                if len(self._connection_pool) < self.max_connections:
                    self._connection_pool.append(conn)
                else:
                    await conn.close()
    
    async def create_user(
        self,
        user_id: str,
        username: str,
        email: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new user"""
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO users (user_id, username, email, created_at, last_active, settings)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    username,
                    email,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                    json.dumps(settings or {})
                ))
                await db.commit()
                logger.info(f"Created user: {username} ({user_id})")
                return True
        except Exception as e:
            logger.error(f"Error creating user {user_id}: {e}")
            return False
    
    async def create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        conversation_type: str = "direct",
        participants: Optional[List[str]] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation"""
        conv_id = conversation_id or str(uuid4())
        now = datetime.utcnow().isoformat()
        
        async with self.get_connection() as db:
            await db.execute("""
                INSERT INTO conversations (
                    conversation_id, user_id, title, type, participants,
                    created_at, updated_at, settings
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conv_id,
                user_id,
                title or f"Conversation {conv_id[:8]}",
                conversation_type,
                json.dumps(participants or []),
                now,
                now,
                json.dumps(settings or {})
            ))
            await db.commit()
        
        logger.info(f"Created conversation {conv_id} for user {user_id}")
        return conv_id
    
    async def save_message(
        self,
        conversation_id: str,
        message: A2AMessage,
        status: str = "sent",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save a message to the conversation"""
        try:
            # Extract text content from message parts
            content = ""
            for part in message.parts:
                if part.kind == "text" and part.text:
                    content = part.text
                    break
            
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO messages (
                        message_id, conversation_id, role, content, parts,
                        task_id, created_at, status, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.messageId,
                    conversation_id,
                    message.role.value,
                    content,
                    json.dumps([part.dict() for part in message.parts]),
                    message.taskId,
                    message.timestamp,
                    status,
                    json.dumps(metadata or {})
                ))
                
                # Update conversation timestamp
                await db.execute("""
                    UPDATE conversations 
                    SET updated_at = ? 
                    WHERE conversation_id = ?
                """, (datetime.utcnow().isoformat(), conversation_id))
                
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving message {message.messageId}: {e}")
            return False
    
    async def save_agent_response(
        self,
        message_id: str,
        agent_id: str,
        response_content: str,
        response_type: str = "text",
        processing_time_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save an agent response"""
        response_id = str(uuid4())
        
        try:
            async with self.get_connection() as db:
                await db.execute("""
                    INSERT INTO agent_responses (
                        response_id, message_id, agent_id, response_content,
                        response_type, processing_time_ms, success, error_message,
                        created_at, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    response_id,
                    message_id,
                    agent_id,
                    response_content,
                    response_type,
                    processing_time_ms,
                    success,
                    error_message,
                    datetime.utcnow().isoformat(),
                    json.dumps(metadata or {})
                ))
                await db.commit()
                return response_id
        except Exception as e:
            logger.error(f"Error saving agent response: {e}")
            return ""
    
    async def get_conversations(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversations for a user"""
        try:
            async with self.get_connection() as db:
                query = """
                    SELECT conversation_id, title, type, participants, created_at,
                           updated_at, status, settings, metadata,
                           (SELECT COUNT(*) FROM messages WHERE conversation_id = c.conversation_id) as message_count
                    FROM conversations c
                    WHERE user_id = ?
                """
                params = [user_id]
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                conversations = []
                for row in rows:
                    conversations.append({
                        "conversation_id": row[0],
                        "title": row[1],
                        "type": row[2],
                        "participants": json.loads(row[3] or "[]"),
                        "created_at": row[4],
                        "updated_at": row[5],
                        "status": row[6],
                        "settings": json.loads(row[7] or "{}"),
                        "metadata": json.loads(row[8] or "{}"),
                        "message_count": row[9]
                    })
                
                return conversations
        except Exception as e:
            logger.error(f"Error getting conversations for user {user_id}: {e}")
            return []
    
    async def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        include_responses: bool = True
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation"""
        try:
            async with self.get_connection() as db:
                # Get messages
                async with db.execute("""
                    SELECT message_id, role, content, parts, task_id, parent_message_id,
                           created_at, status, metadata
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                    LIMIT ? OFFSET ?
                """, (conversation_id, limit, offset)) as cursor:
                    message_rows = await cursor.fetchall()
                
                messages = []
                for row in message_rows:
                    message = {
                        "message_id": row[0],
                        "role": row[1],
                        "content": row[2],
                        "parts": json.loads(row[3] or "[]"),
                        "task_id": row[4],
                        "parent_message_id": row[5],
                        "created_at": row[6],
                        "status": row[7],
                        "metadata": json.loads(row[8] or "{}"),
                        "agent_responses": []
                    }
                    
                    # Get agent responses if requested
                    if include_responses:
                        async with db.execute("""
                            SELECT response_id, agent_id, response_content, response_type,
                                   processing_time_ms, success, error_message, created_at, metadata
                            FROM agent_responses
                            WHERE message_id = ?
                            ORDER BY created_at ASC
                        """, (row[0],)) as resp_cursor:
                            response_rows = await resp_cursor.fetchall()
                            
                            for resp_row in response_rows:
                                message["agent_responses"].append({
                                    "response_id": resp_row[0],
                                    "agent_id": resp_row[1],
                                    "response_content": resp_row[2],
                                    "response_type": resp_row[3],
                                    "processing_time_ms": resp_row[4],
                                    "success": bool(resp_row[5]),
                                    "error_message": resp_row[6],
                                    "created_at": resp_row[7],
                                    "metadata": json.loads(resp_row[8] or "{}")
                                })
                    
                    messages.append(message)
                
                return messages
        except Exception as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {e}")
            return []
    
    async def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        try:
            async with self.get_connection() as db:
                # Search in conversation titles and message content
                async with db.execute("""
                    SELECT DISTINCT c.conversation_id, c.title, c.type, c.created_at, c.updated_at,
                           COUNT(m.message_id) as message_count
                    FROM conversations c
                    LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                    WHERE c.user_id = ? AND (
                        c.title LIKE ? OR 
                        m.content LIKE ?
                    )
                    GROUP BY c.conversation_id
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                """, (user_id, f"%{query}%", f"%{query}%", limit)) as cursor:
                    rows = await cursor.fetchall()
                    
                results = []
                for row in rows:
                    results.append({
                        "conversation_id": row[0],
                        "title": row[1],
                        "type": row[2],
                        "created_at": row[3],
                        "updated_at": row[4],
                        "message_count": row[5]
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
        hard_delete: bool = False
    ) -> bool:
        """Delete or archive a conversation"""
        try:
            async with self.get_connection() as db:
                if hard_delete:
                    # Delete all related data
                    await db.execute("DELETE FROM agent_responses WHERE message_id IN (SELECT message_id FROM messages WHERE conversation_id = ?)", (conversation_id,))
                    await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                    await db.execute("DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id))
                else:
                    # Soft delete - mark as archived
                    await db.execute("UPDATE conversations SET status = 'archived' WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id))
                
                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            async with self.get_connection() as db:
                # Get conversation count
                async with db.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)) as cursor:
                    conv_count = (await cursor.fetchone())[0]
                
                # Get message count
                async with db.execute("""
                    SELECT COUNT(*) FROM messages m
                    JOIN conversations c ON m.conversation_id = c.conversation_id
                    WHERE c.user_id = ?
                """, (user_id,)) as cursor:
                    msg_count = (await cursor.fetchone())[0]
                
                # Get active conversations
                async with db.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ? AND status = 'active'", (user_id,)) as cursor:
                    active_conv = (await cursor.fetchone())[0]
                
                return {
                    "total_conversations": conv_count,
                    "total_messages": msg_count,
                    "active_conversations": active_conv,
                    "archived_conversations": conv_count - active_conv
                }
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            return {}
    
    async def cleanup_old_data(self, days_old: int = 90):
        """Clean up old archived conversations"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
            
            async with self.get_connection() as db:
                # Get old conversation IDs
                async with db.execute("""
                    SELECT conversation_id FROM conversations 
                    WHERE status = 'archived' AND updated_at < ?
                """, (cutoff_date,)) as cursor:
                    old_conversations = [row[0] for row in await cursor.fetchall()]
                
                if old_conversations:
                    # Delete related data
                    for conv_id in old_conversations:
                        await db.execute("DELETE FROM agent_responses WHERE message_id IN (SELECT message_id FROM messages WHERE conversation_id = ?)", (conv_id,))
                        await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
                        await db.execute("DELETE FROM conversations WHERE conversation_id = ?", (conv_id,))
                    
                    await db.commit()
                    logger.info(f"Cleaned up {len(old_conversations)} old conversations")
                    
                return len(old_conversations)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    async def close(self):
        """Close all connections and cleanup"""
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()
        
        logger.info("ChatStorage closed")