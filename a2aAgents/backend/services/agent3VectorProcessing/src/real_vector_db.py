"""
Real Vector Database Implementation for A2A
Uses PostgreSQL with pgvector extension for production-grade vector storage
"""

import os
import asyncio
import asyncpg
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RealVectorDB:
    """Production-ready vector database using PostgreSQL with pgvector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = None
        self.dimension = 1536  # Default embedding dimension (OpenAI ada-002)
        
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Try PostgreSQL with pgvector first
            if os.getenv("POSTGRES_HOST"):
                await self._init_postgres()
            else:
                # Fallback to SQLite with vector extension
                await self._init_sqlite_vector()
                
        except Exception as e:
            logger.error(f"Failed to initialize real vector DB: {e}")
            raise
            
    async def _init_postgres(self):
        """Initialize PostgreSQL with pgvector"""
        try:
            dsn = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'a2a_vectors')}"
            
            self.pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
            
            # Create pgvector extension and tables
            async with self.pool.acquire() as conn:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
                
                # Create vectors table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS vectors (
                        entity_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        embedding vector($1),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                ''', self.dimension)
                
                # Create graph table for relationships
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS graph_edges (
                        id SERIAL PRIMARY KEY,
                        source_id TEXT NOT NULL,
                        target_id TEXT NOT NULL,
                        edge_type TEXT NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(source_id, target_id, edge_type)
                    )
                ''')
                
                # Create indexes
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_vectors_type ON vectors(entity_type)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target_id)')
                
            logger.info("PostgreSQL vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")
            raise
            
    async def _init_sqlite_vector(self):
        """Initialize SQLite with vector support using numpy"""
        import aiosqlite
        import pickle
        
        db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'vectors.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db = await aiosqlite.connect(db_path)
        
        # Create tables
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                embedding BLOB,  -- Store serialized numpy array
                metadata TEXT,   -- JSON string
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id, edge_type)
            )
        ''')
        
        await self.db.execute('CREATE INDEX IF NOT EXISTS idx_vectors_type ON vectors(entity_type)')
        await self.db.execute('CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source_id)')
        await self.db.execute('CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target_id)')
        
        await self.db.commit()
        
        logger.info("SQLite vector database initialized successfully")
        self.db_type = 'sqlite'
        
    async def store_vector(self, entity_id: str, vector: List[float], metadata: Dict[str, Any]):
        """Store a vector with metadata"""
        try:
            vector_array = np.array(vector, dtype=np.float32)
            
            if hasattr(self, 'pool') and self.pool:
                # PostgreSQL
                async with self.pool.acquire() as conn:
                    await conn.execute('''
                        INSERT INTO vectors (entity_id, entity_type, embedding, metadata)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (entity_id) 
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    ''', entity_id, metadata.get('type', 'unknown'), vector_array.tolist(), json.dumps(metadata))
            else:
                # SQLite
                import pickle
                vector_blob = pickle.dumps(vector_array)
                metadata_json = json.dumps(metadata)
                
                await self.db.execute('''
                    INSERT OR REPLACE INTO vectors (entity_id, entity_type, embedding, metadata, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (entity_id, metadata.get('type', 'unknown'), vector_blob, metadata_json))
                
                await self.db.commit()
                
            logger.debug(f"Stored vector for entity {entity_id}")
            
        except Exception as e:
            logger.error(f"Failed to store vector: {e}")
            raise
            
    async def search_similar(self, query_vector: List[float], top_k: int = 10, 
                           entity_type: Optional[str] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors using cosine similarity"""
        try:
            query_array = np.array(query_vector, dtype=np.float32)
            
            if hasattr(self, 'pool') and self.pool:
                # PostgreSQL with pgvector
                async with self.pool.acquire() as conn:
                    if entity_type:
                        results = await conn.fetch('''
                            SELECT entity_id, 
                                   1 - (embedding <=> $1::vector) as similarity,
                                   metadata
                            FROM vectors
                            WHERE entity_type = $2
                            ORDER BY embedding <=> $1::vector
                            LIMIT $3
                        ''', query_array.tolist(), entity_type, top_k)
                    else:
                        results = await conn.fetch('''
                            SELECT entity_id,
                                   1 - (embedding <=> $1::vector) as similarity,
                                   metadata
                            FROM vectors
                            ORDER BY embedding <=> $1::vector
                            LIMIT $2
                        ''', query_array.tolist(), top_k)
                        
                    return [(r['entity_id'], r['similarity'], json.loads(r['metadata'])) for r in results]
                    
            else:
                # SQLite with numpy cosine similarity
                import pickle
                
                if entity_type:
                    cursor = await self.db.execute(
                        'SELECT entity_id, embedding, metadata FROM vectors WHERE entity_type = ?',
                        (entity_type,)
                    )
                else:
                    cursor = await self.db.execute('SELECT entity_id, embedding, metadata FROM vectors')
                    
                similarities = []
                async for row in cursor:
                    stored_vector = pickle.loads(row[1])
                    # Cosine similarity
                    similarity = np.dot(query_array, stored_vector) / (np.linalg.norm(query_array) * np.linalg.norm(stored_vector))
                    similarities.append((row[0], float(similarity), json.loads(row[2])))
                    
                # Sort by similarity and return top_k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
                
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise
            
    async def add_graph_edge(self, source_id: str, target_id: str, edge_type: str, 
                           weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Add an edge to the graph"""
        try:
            metadata_json = json.dumps(metadata or {})
            
            if hasattr(self, 'pool') and self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute('''
                        INSERT INTO graph_edges (source_id, target_id, edge_type, weight, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (source_id, target_id, edge_type)
                        DO UPDATE SET weight = EXCLUDED.weight, metadata = EXCLUDED.metadata
                    ''', source_id, target_id, edge_type, weight, metadata_json)
            else:
                await self.db.execute('''
                    INSERT OR REPLACE INTO graph_edges (source_id, target_id, edge_type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (source_id, target_id, edge_type, weight, metadata_json))
                await self.db.commit()
                
        except Exception as e:
            logger.error(f"Failed to add graph edge: {e}")
            raise
            
    async def get_graph_neighbors(self, entity_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get graph neighbors for an entity"""
        try:
            if hasattr(self, 'pool') and self.pool:
                async with self.pool.acquire() as conn:
                    if edge_type:
                        results = await conn.fetch('''
                            SELECT target_id, edge_type, weight, metadata
                            FROM graph_edges
                            WHERE source_id = $1 AND edge_type = $2
                        ''', entity_id, edge_type)
                    else:
                        results = await conn.fetch('''
                            SELECT target_id, edge_type, weight, metadata
                            FROM graph_edges
                            WHERE source_id = $1
                        ''', entity_id)
                        
                    return [
                        {
                            'target_id': r['target_id'],
                            'edge_type': r['edge_type'],
                            'weight': r['weight'],
                            'metadata': json.loads(r['metadata']) if r['metadata'] else {}
                        }
                        for r in results
                    ]
            else:
                if edge_type:
                    cursor = await self.db.execute(
                        'SELECT target_id, edge_type, weight, metadata FROM graph_edges WHERE source_id = ? AND edge_type = ?',
                        (entity_id, edge_type)
                    )
                else:
                    cursor = await self.db.execute(
                        'SELECT target_id, edge_type, weight, metadata FROM graph_edges WHERE source_id = ?',
                        (entity_id,)
                    )
                    
                results = []
                async for row in cursor:
                    results.append({
                        'target_id': row[0],
                        'edge_type': row[1],
                        'weight': row[2],
                        'metadata': json.loads(row[3]) if row[3] else {}
                    })
                    
                return results
                
        except Exception as e:
            logger.error(f"Failed to get graph neighbors: {e}")
            raise
            
    async def close(self):
        """Close database connections"""
        if hasattr(self, 'pool') and self.pool:
            await self.pool.close()
        elif hasattr(self, 'db'):
            await self.db.close()