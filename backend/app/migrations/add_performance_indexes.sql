-- Performance optimization indexes for A2A database

-- Agent data indexes
CREATE INDEX IF NOT EXISTS idx_agent_data_agent_id_type ON agent_data(agent_id, data_type);
CREATE INDEX IF NOT EXISTS idx_agent_data_created_at ON agent_data(created_at DESC);

-- Agent interactions indexes
CREATE INDEX IF NOT EXISTS idx_agent_interactions_agent_id ON agent_interactions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_timestamp ON agent_interactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_type ON agent_interactions(interaction_type);

-- Financial data indexes
CREATE INDEX IF NOT EXISTS idx_financial_data_source_type ON financial_data(data_source, data_type);
CREATE INDEX IF NOT EXISTS idx_financial_data_validation ON financial_data(validation_status);
CREATE INDEX IF NOT EXISTS idx_financial_data_created ON financial_data(created_at DESC);

-- Message persistence indexes
CREATE INDEX IF NOT EXISTS idx_messages_agent_id_status ON messages(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_context_id ON messages(context_id);

-- Task persistence indexes
CREATE INDEX IF NOT EXISTS idx_persisted_tasks_agent_status ON persisted_tasks(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_persisted_tasks_created ON persisted_tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_persisted_tasks_status ON persisted_tasks(status);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_agent_recent_activity ON agent_interactions(agent_id, timestamp DESC)
WHERE success = 1;

-- Vector search optimization (if using vector embeddings)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (vector vector_l2_ops)
WITH (lists = 100);

-- Full text search indexes
CREATE INDEX IF NOT EXISTS idx_financial_data_fts ON financial_data USING GIN(to_tsvector('english', record_data::text));

-- Analyze tables to update statistics
ANALYZE agent_data;
ANALYZE agent_interactions;
ANALYZE financial_data;
ANALYZE messages;
ANALYZE persisted_tasks;