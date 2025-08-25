#!/usr/bin/env python3
"""
Chat Persistence Layer - Production-ready database storage for chat history
Supports multiple database backends and efficient querying
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiosqlite
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"

@dataclass
class ChatMessage:
    """Chat message data model"""
    message_id: str
    conversation_id: str
    sender: str
    recipient: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class ChatConversation:
    """Chat conversation model"""
    conversation_id: str
    notification_id: Optional[str]
    user_id: str
    agent_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    summary: Optional[str] = None

class ChatPersistenceLayer:
    """Unified persistence layer for chat history with multiple backend support"""

    def __init__(self, db_type: DatabaseType, connection_string: str):
        self.db_type = db_type
        self.connection_string = connection_string
        self.connection = None
        self._initialized = False

        # Connection pools
        self._pg_pool = None
        self._mongo_client = None

        logger.info(f"Initializing chat persistence with {db_type.value}")

    async def initialize(self):
        """Initialize database connection and schema"""
        if self._initialized:
            return

        try:
            if self.db_type == DatabaseType.SQLITE:
                await self._init_sqlite()
            elif self.db_type == DatabaseType.POSTGRESQL:
                await self._init_postgresql()
            elif self.db_type == DatabaseType.MONGODB:
                await self._init_mongodb()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")

            self._initialized = True
            logger.info("Chat persistence layer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize persistence layer: {e}")
            raise

    async def _init_sqlite(self):
        """Initialize SQLite database"""
        self.connection = await aiosqlite.connect(self.connection_string)

        # Create tables
        await self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                notification_id TEXT,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                metadata TEXT,
                summary TEXT,
                FOREIGN KEY (notification_id) REFERENCES notifications(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                metadata TEXT,
                embedding BLOB,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_status
                ON conversations(status);
        """)

        await self.connection.commit()

    async def _init_postgresql(self):
        """Initialize PostgreSQL database with connection pool"""
        self._pg_pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

        # Create schema
        async with self._pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id UUID PRIMARY KEY,
                    notification_id UUID,
                    user_id VARCHAR(255) NOT NULL,
                    agent_id VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id UUID PRIMARY KEY,
                    conversation_id UUID NOT NULL REFERENCES conversations(conversation_id),
                    sender VARCHAR(255) NOT NULL,
                    recipient VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metadata JSONB,
                    embedding vector(1536)  -- For OpenAI embeddings
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                    ON messages(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                    ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);

                -- Full text search
                CREATE INDEX IF NOT EXISTS idx_messages_fts
                    ON messages USING gin(to_tsvector('english', message));
            """)

    async def _init_mongodb(self):
        """Initialize MongoDB with proper indexes"""
        self._mongo_client = AsyncIOMotorClient(self.connection_string)
        self.db = self._mongo_client.a2a_chat

        # Create indexes
        await self.db.conversations.create_index([
            ("conversation_id", 1)
        ], unique=True)

        await self.db.conversations.create_index([
            ("user_id", 1),
            ("updated_at", -1)
        ])

        await self.db.messages.create_index([
            ("conversation_id", 1),
            ("timestamp", -1)
        ])

        await self.db.messages.create_index([
            ("message", "text")
        ])

    async def save_conversation(self, conversation: ChatConversation) -> bool:
        """Save or update a conversation"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                return await self._save_conversation_sqlite(conversation)
            elif self.db_type == DatabaseType.POSTGRESQL:
                return await self._save_conversation_postgresql(conversation)
            elif self.db_type == DatabaseType.MONGODB:
                return await self._save_conversation_mongodb(conversation)

        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def _save_conversation_sqlite(self, conversation: ChatConversation) -> bool:
        """Save conversation to SQLite"""
        await self.connection.execute("""
            INSERT OR REPLACE INTO conversations
            (conversation_id, notification_id, user_id, agent_id, status,
             created_at, updated_at, metadata, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation.conversation_id,
            conversation.notification_id,
            conversation.user_id,
            conversation.agent_id,
            conversation.status,
            conversation.created_at.isoformat(),
            conversation.updated_at.isoformat(),
            json.dumps(conversation.metadata),
            conversation.summary
        ))

        await self.connection.commit()
        return True

    async def _save_conversation_postgresql(self, conversation: ChatConversation) -> bool:
        """Save conversation to PostgreSQL"""
        async with self._pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations
                (conversation_id, notification_id, user_id, agent_id, status,
                 created_at, updated_at, metadata, summary)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata,
                    summary = EXCLUDED.summary
            """,
                conversation.conversation_id,
                conversation.notification_id,
                conversation.user_id,
                conversation.agent_id,
                conversation.status,
                conversation.created_at,
                conversation.updated_at,
                json.dumps(conversation.metadata),
                conversation.summary
            )
        return True

    async def _save_conversation_mongodb(self, conversation: ChatConversation) -> bool:
        """Save conversation to MongoDB"""
        doc = asdict(conversation)
        doc['_id'] = conversation.conversation_id

        await self.db.conversations.replace_one(
            {'_id': doc['_id']},
            doc,
            upsert=True
        )
        return True

    async def save_message(self, message: ChatMessage) -> bool:
        """Save a chat message"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                return await self._save_message_sqlite(message)
            elif self.db_type == DatabaseType.POSTGRESQL:
                return await self._save_message_postgresql(message)
            elif self.db_type == DatabaseType.MONGODB:
                return await self._save_message_mongodb(message)

        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    async def _save_message_sqlite(self, message: ChatMessage) -> bool:
        """Save message to SQLite"""
        embedding_bytes = None
        if message.embedding:
            embedding_bytes = json.dumps(message.embedding).encode()

        await self.connection.execute("""
            INSERT INTO messages
            (message_id, conversation_id, sender, recipient, message,
             timestamp, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.message_id,
            message.conversation_id,
            message.sender,
            message.recipient,
            message.message,
            message.timestamp.isoformat(),
            json.dumps(message.metadata),
            embedding_bytes
        ))

        # Update conversation timestamp
        await self.connection.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ?
        """, (message.timestamp.isoformat(), message.conversation_id))

        await self.connection.commit()
        return True

    async def _save_message_postgresql(self, message: ChatMessage) -> bool:
        """Save message to PostgreSQL"""
        async with self._pg_pool.acquire() as conn:
            # Use transaction for atomicity
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO messages
                    (message_id, conversation_id, sender, recipient, message,
                     timestamp, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    message.message_id,
                    message.conversation_id,
                    message.sender,
                    message.recipient,
                    message.message,
                    message.timestamp,
                    json.dumps(message.metadata),
                    message.embedding
                )

                # Update conversation
                await conn.execute("""
                    UPDATE conversations
                    SET updated_at = $1
                    WHERE conversation_id = $2
                """, message.timestamp, message.conversation_id)

        return True

    async def _save_message_mongodb(self, message: ChatMessage) -> bool:
        """Save message to MongoDB"""
        doc = asdict(message)
        doc['_id'] = message.message_id

        # Insert message
        await self.db.messages.insert_one(doc)

        # Update conversation
        await self.db.conversations.update_one(
            {'_id': message.conversation_id},
            {'$set': {'updated_at': message.timestamp}}
        )

        return True

    async def get_conversation(self, conversation_id: str) -> Optional[ChatConversation]:
        """Retrieve a conversation by ID"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                return await self._get_conversation_sqlite(conversation_id)
            elif self.db_type == DatabaseType.POSTGRESQL:
                return await self._get_conversation_postgresql(conversation_id)
            elif self.db_type == DatabaseType.MONGODB:
                return await self._get_conversation_mongodb(conversation_id)

        except Exception as e:
            logger.error(f"Error retrieving conversation: {e}")
            return None

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatMessage]:
        """Retrieve messages for a conversation"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                return await self._get_messages_sqlite(conversation_id, limit, offset)
            elif self.db_type == DatabaseType.POSTGRESQL:
                return await self._get_messages_postgresql(conversation_id, limit, offset)
            elif self.db_type == DatabaseType.MONGODB:
                return await self._get_messages_mongodb(conversation_id, limit, offset)

        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return []

    async def search_messages(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Tuple[ChatMessage, float]]:
        """Search messages with relevance scoring"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                return await self._search_messages_sqlite(query, user_id, limit)
            elif self.db_type == DatabaseType.POSTGRESQL:
                return await self._search_messages_postgresql(query, user_id, limit)
            elif self.db_type == DatabaseType.MONGODB:
                return await self._search_messages_mongodb(query, user_id, limit)

        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []

    async def get_user_conversations(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[ChatConversation]:
        """Get conversations for a user"""
        try:
            if self.db_type == DatabaseType.SQLITE:
                query = """
                    SELECT * FROM conversations
                    WHERE user_id = ?
                """
                params = [user_id]

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor = await self.connection.execute(query, params)
                rows = await cursor.fetchall()

                return [self._row_to_conversation(row) for row in rows]

        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []

    async def archive_old_conversations(self, days: int = 30) -> int:
        """Archive conversations older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        try:
            if self.db_type == DatabaseType.SQLITE:
                cursor = await self.connection.execute("""
                    UPDATE conversations
                    SET status = 'archived'
                    WHERE updated_at < ? AND status != 'archived'
                """, (cutoff_date.isoformat(),))

                await self.connection.commit()
                return cursor.rowcount

        except Exception as e:
            logger.error(f"Error archiving conversations: {e}")
            return 0

    def _row_to_conversation(self, row) -> ChatConversation:
        """Convert database row to ChatConversation object"""
        return ChatConversation(
            conversation_id=row[0],
            notification_id=row[1],
            user_id=row[2],
            agent_id=row[3],
            status=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            metadata=json.loads(row[7]) if row[7] else {},
            summary=row[8]
        )

    def _row_to_message(self, row) -> ChatMessage:
        """Convert database row to ChatMessage object"""
        embedding = None
        if row[7]:
            if isinstance(row[7], bytes):
                embedding = json.loads(row[7].decode())
            else:
                embedding = row[7]

        return ChatMessage(
            message_id=row[0],
            conversation_id=row[1],
            sender=row[2],
            recipient=row[3],
            message=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
            embedding=embedding
        )

    async def close(self):
        """Close database connections"""
        if self.db_type == DatabaseType.SQLITE and self.connection:
            await self.connection.close()
        elif self.db_type == DatabaseType.POSTGRESQL and self._pg_pool:
            await self._pg_pool.close()
        elif self.db_type == DatabaseType.MONGODB and self._mongo_client:
            self._mongo_client.close()


# Factory function for creating persistence layer
def create_chat_persistence(
    db_type: str = "sqlite",
    connection_string: Optional[str] = None
) -> ChatPersistenceLayer:
    """Create chat persistence layer with specified database"""

    if not connection_string:
        # Default connection strings
        defaults = {
            "sqlite": "chat_history.db",
            "postgresql": "postgresql://user:pass@localhost/a2a_chat",
            "mongodb": "mongodb://localhost:27017/a2a_chat"
        }
        connection_string = defaults.get(db_type, "chat_history.db")

    return ChatPersistenceLayer(
        DatabaseType(db_type),
        connection_string
    )
