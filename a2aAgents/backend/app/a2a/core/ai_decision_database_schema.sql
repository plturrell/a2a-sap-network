-- AI Decision Logger Database Schema
-- Optimized for Data Manager Agent integration with HANA and SQLite support

-- Main decisions table
CREATE TABLE ai_decisions (
    decision_id VARCHAR(36) PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    decision_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question TEXT,
    context JSON,
    ai_response JSON,
    confidence_score DECIMAL(3,2),
    response_time DECIMAL(8,3),
    metadata JSON,
    
    -- Indexes for performance
    INDEX idx_agent_type (agent_id, decision_type),
    INDEX idx_timestamp (timestamp),
    INDEX idx_confidence (confidence_score),
    INDEX idx_agent_timestamp (agent_id, timestamp)
);

-- Decision outcomes table
CREATE TABLE ai_decision_outcomes (
    decision_id VARCHAR(36) PRIMARY KEY,
    outcome_status VARCHAR(20) NOT NULL,
    outcome_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success_metrics JSON,
    failure_reason TEXT,
    side_effects JSON,
    feedback TEXT,
    actual_duration DECIMAL(8,3),
    
    -- Foreign key relationship
    FOREIGN KEY (decision_id) REFERENCES ai_decisions(decision_id) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_outcome_status (outcome_status),
    INDEX idx_outcome_timestamp (outcome_timestamp)
);

-- Learned patterns table
CREATE TABLE ai_learned_patterns (
    pattern_id VARCHAR(36) PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    description TEXT,
    confidence DECIMAL(3,2),
    evidence_count INT,
    success_rate DECIMAL(3,2),
    applicable_contexts JSON,
    recommendations JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_agent_pattern (agent_id, pattern_type),
    INDEX idx_confidence_pattern (confidence),
    INDEX idx_success_rate (success_rate),
    INDEX idx_evidence_count (evidence_count)
);

-- Global cross-agent analytics view
CREATE VIEW ai_global_analytics AS
SELECT 
    agent_id,
    decision_type,
    COUNT(*) as total_decisions,
    COUNT(o.decision_id) as decisions_with_outcomes,
    SUM(CASE WHEN o.outcome_status = 'success' THEN 1 ELSE 0 END) as successful_outcomes,
    AVG(confidence_score) as avg_confidence,
    AVG(response_time) as avg_response_time,
    DATE(d.timestamp) as decision_date
FROM ai_decisions d
LEFT JOIN ai_decision_outcomes o ON d.decision_id = o.decision_id
GROUP BY agent_id, decision_type, DATE(d.timestamp);

-- Pattern effectiveness view
CREATE VIEW ai_pattern_effectiveness AS
SELECT 
    p.pattern_id,
    p.agent_id,
    p.pattern_type,
    p.confidence,
    p.success_rate,
    p.evidence_count,
    COUNT(d.decision_id) as recent_usage,
    AVG(CASE WHEN o.outcome_status = 'success' THEN 1.0 ELSE 0.0 END) as actual_success_rate
FROM ai_learned_patterns p
LEFT JOIN ai_decisions d ON d.agent_id = p.agent_id 
    AND d.decision_type = p.pattern_type 
    AND d.timestamp > (CURRENT_TIMESTAMP - INTERVAL '30' DAY)
LEFT JOIN ai_decision_outcomes o ON d.decision_id = o.decision_id
GROUP BY p.pattern_id, p.agent_id, p.pattern_type, p.confidence, p.success_rate, p.evidence_count;

-- Decision performance metrics view
CREATE VIEW ai_decision_performance AS
SELECT 
    d.agent_id,
    d.decision_type,
    d.confidence_score,
    o.outcome_status,
    d.response_time,
    d.timestamp,
    CASE 
        WHEN o.outcome_status = 'success' THEN 1.0
        WHEN o.outcome_status = 'partial_success' THEN 0.5
        ELSE 0.0
    END as success_score,
    -- Extract common context features for pattern matching
    JSON_EXTRACT(d.context, '$.complexity') as complexity,
    JSON_EXTRACT(d.context, '$.domain') as domain,
    JSON_EXTRACT(d.context, '$.urgency') as urgency
FROM ai_decisions d
LEFT JOIN ai_decision_outcomes o ON d.decision_id = o.decision_id;

-- Indexes on views for better performance
CREATE INDEX idx_global_analytics_agent_date ON ai_global_analytics(agent_id, decision_date);
CREATE INDEX idx_pattern_effectiveness_agent ON ai_pattern_effectiveness(agent_id, pattern_type);
CREATE INDEX idx_decision_performance_agent_type ON ai_decision_performance(agent_id, decision_type);

-- Table for tracking schema version and migrations
CREATE TABLE ai_decision_schema_version (
    version VARCHAR(10) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO ai_decision_schema_version (version, description) 
VALUES ('1.0.0', 'Initial AI Decision Logger database schema');