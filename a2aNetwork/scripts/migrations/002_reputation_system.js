/**
 * Migration 002: Reputation System
 * Adds comprehensive reputation tracking and scoring system
 */

async function up(db) {
    console.log('🏆 Adding reputation system...');

    // Reputation Scores table
    await db.run(`
        CREATE TABLE IF NOT EXISTS ReputationScores (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36) NOT NULL,
            overall_score DECIMAL(5,2) DEFAULT 100.00,
            reliability_score DECIMAL(5,2) DEFAULT 100.00,
            performance_score DECIMAL(5,2) DEFAULT 100.00,
            security_score DECIMAL(5,2) DEFAULT 100.00,
            total_interactions INTEGER DEFAULT 0,
            successful_interactions INTEGER DEFAULT 0,
            failed_interactions INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Reputation Events table
    await db.run(`
        CREATE TABLE IF NOT EXISTS ReputationEvents (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36) NOT NULL,
            event_type NVARCHAR(50) NOT NULL,
            impact_score DECIMAL(5,2),
            description NVARCHAR(1000),
            metadata NCLOB,
            reported_by NVARCHAR(36),
            verified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID),
            FOREIGN KEY (reported_by) REFERENCES Agents(ID)
        )
    `);

    // Trust Relationships table
    await db.run(`
        CREATE TABLE IF NOT EXISTS TrustRelationships (
            ID NVARCHAR(36) PRIMARY KEY,
            truster_agent_ID NVARCHAR(36) NOT NULL,
            trustee_agent_ID NVARCHAR(36) NOT NULL,
            trust_level DECIMAL(5,2) DEFAULT 50.00,
            relationship_type NVARCHAR(50) DEFAULT 'peer',
            established_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_interaction TIMESTAMP,
            interaction_count INTEGER DEFAULT 0,
            FOREIGN KEY (truster_agent_ID) REFERENCES Agents(ID),
            FOREIGN KEY (trustee_agent_ID) REFERENCES Agents(ID),
            UNIQUE(truster_agent_ID, trustee_agent_ID)
        )
    `);

    // Reputation History table for tracking changes
    await db.run(`
        CREATE TABLE IF NOT EXISTS ReputationHistory (
            ID NVARCHAR(36) PRIMARY KEY,
            agent_ID NVARCHAR(36) NOT NULL,
            score_type NVARCHAR(50) NOT NULL,
            old_score DECIMAL(5,2),
            new_score DECIMAL(5,2),
            change_reason NVARCHAR(500),
            changed_by NVARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Reputation Feedback table
    await db.run(`
        CREATE TABLE IF NOT EXISTS ReputationFeedback (
            ID NVARCHAR(36) PRIMARY KEY,
            from_agent_ID NVARCHAR(36) NOT NULL,
            to_agent_ID NVARCHAR(36) NOT NULL,
            interaction_ID NVARCHAR(36),
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            feedback_text NVARCHAR(2000),
            feedback_type NVARCHAR(50),
            is_anonymous BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_agent_ID) REFERENCES Agents(ID),
            FOREIGN KEY (to_agent_ID) REFERENCES Agents(ID)
        )
    `);

    // Create reputation-related indexes
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_scores_agent_id ON ReputationScores(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_scores_overall ON ReputationScores(overall_score)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_events_agent_id ON ReputationEvents(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_events_type ON ReputationEvents(event_type)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_trust_relationships_truster ON TrustRelationships(truster_agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_trust_relationships_trustee ON TrustRelationships(trustee_agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_history_agent_id ON ReputationHistory(agent_ID)');
    await db.run('CREATE INDEX IF NOT EXISTS idx_reputation_feedback_to_agent ON ReputationFeedback(to_agent_ID)');

    // Add reputation score column to Agents table if not exists
    try {
        await db.run('ALTER TABLE Agents ADD COLUMN reputation_score DECIMAL(5,2) DEFAULT 100.00');
        console.log('✅ Added reputation_score column to Agents table');
    } catch (error) {
        console.log('ℹ️  reputation_score column already exists in Agents table');
    }

    console.log('✅ Reputation system added successfully');
}

async function down(db) {
    console.log('🔄 Rolling back reputation system...');

    const tables = [
        'ReputationFeedback',
        'ReputationHistory',
        'TrustRelationships',
        'ReputationEvents',
        'ReputationScores'
    ];

    for (const table of tables) {
        await db.run(`DROP TABLE IF EXISTS ${table}`);
    }

    // Remove reputation_score column from Agents table (if supported)
    try {
        await db.run('ALTER TABLE Agents DROP COLUMN reputation_score');
        console.log('✅ Removed reputation_score column from Agents table');
    } catch (error) {
        console.log('⚠️  Could not remove reputation_score column (may not be supported)');
    }

    console.log('✅ Reputation system rollback completed');
}

module.exports = { up, down };