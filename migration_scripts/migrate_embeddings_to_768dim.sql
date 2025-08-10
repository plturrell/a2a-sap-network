-- HANA Database Migration Script: Upgrade Embeddings from 384 to 768 Dimensions
-- Version: 1.0
-- Purpose: Migrate existing vector storage to support all-mpnet-base-v2 model
-- Date: 2025-01-10

-- =============================================================================
-- PRE-MIGRATION VALIDATION AND BACKUP
-- =============================================================================

-- 1. Validate current schema
DO BEGIN
    DECLARE table_exists INTEGER;
    DECLARE current_dimension INTEGER;
    
    -- Check if A2A_VECTORS table exists
    SELECT COUNT(*) INTO table_exists 
    FROM SYS.TABLES 
    WHERE SCHEMA_NAME = CURRENT_SCHEMA 
    AND TABLE_NAME = 'A2A_VECTORS';
    
    IF table_exists = 0 THEN
        SIGNAL SQL_ERROR_CODE 10001 
        SET MESSAGE_TEXT = 'A2A_VECTORS table does not exist. Migration cannot proceed.';
    END IF;
    
    -- Check current vector dimension
    SELECT VECTOR_TYPE_DIMENSION INTO current_dimension
    FROM SYS.TABLE_COLUMNS 
    WHERE SCHEMA_NAME = CURRENT_SCHEMA 
    AND TABLE_NAME = 'A2A_VECTORS' 
    AND COLUMN_NAME = 'VECTOR_EMBEDDING';
    
    IF current_dimension != 384 THEN
        SIGNAL SQL_ERROR_CODE 10002 
        SET MESSAGE_TEXT = 'Current vector dimension is not 384. Migration may not be applicable.';
    END IF;
    
    -- Log validation success
    INSERT INTO A2A_MIGRATION_LOG VALUES (
        'VALIDATION', 
        CURRENT_TIMESTAMP, 
        'Pre-migration validation passed. Current dimension: ' || current_dimension
    );
END;

-- 2. Create migration log table
CREATE COLUMN TABLE IF NOT EXISTS A2A_MIGRATION_LOG (
    OPERATION NVARCHAR(50),
    TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    MESSAGE NVARCHAR(500)
);

-- 3. Create backup of existing data
CREATE COLUMN TABLE A2A_VECTORS_384_BACKUP AS (
    SELECT 
        DOC_ID,
        CONTENT,
        METADATA,
        VECTOR_EMBEDDING AS VECTOR_EMBEDDING_384,
        ENTITY_TYPE,
        SOURCE_AGENT,
        CREATED_AT,
        CURRENT_TIMESTAMP AS BACKUP_TIMESTAMP
    FROM A2A_VECTORS
) WITH DATA;

-- Log backup creation
INSERT INTO A2A_MIGRATION_LOG VALUES (
    'BACKUP', 
    CURRENT_TIMESTAMP, 
    'Created backup table A2A_VECTORS_384_BACKUP with ' || 
    (SELECT COUNT(*) FROM A2A_VECTORS_384_BACKUP) || ' records'
);

-- =============================================================================
-- SCHEMA MIGRATION
-- =============================================================================

-- 4. Create new table with 768-dimensional vectors
CREATE COLUMN TABLE A2A_VECTORS_768_NEW (
    DOC_ID NVARCHAR(255) PRIMARY KEY,
    CONTENT NCLOB,
    METADATA NCLOB,
    VECTOR_EMBEDDING REAL_VECTOR(768),
    ENTITY_TYPE NVARCHAR(100),
    SOURCE_AGENT NVARCHAR(100),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    MIGRATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    MIGRATION_VERSION NVARCHAR(20) DEFAULT '768-mpnet-v1.0',
    -- Migration tracking columns
    ORIGINAL_DIMENSION INTEGER DEFAULT 384,
    RE_EMBEDDED BOOLEAN DEFAULT FALSE,
    EMBEDDING_MODEL NVARCHAR(50) DEFAULT 'all-mpnet-base-v2'
);

-- Create indexes for performance
CREATE INDEX IDX_A2A_VECTORS_768_ENTITY_TYPE ON A2A_VECTORS_768_NEW (ENTITY_TYPE);
CREATE INDEX IDX_A2A_VECTORS_768_SOURCE_AGENT ON A2A_VECTORS_768_NEW (SOURCE_AGENT);
CREATE INDEX IDX_A2A_VECTORS_768_MIGRATED ON A2A_VECTORS_768_NEW (RE_EMBEDDED);

-- 5. Update knowledge graph tables for 768-dimensional references
-- Update A2A_GRAPH_NODES if it exists
DO BEGIN
    DECLARE node_table_exists INTEGER;
    
    SELECT COUNT(*) INTO node_table_exists 
    FROM SYS.TABLES 
    WHERE SCHEMA_NAME = CURRENT_SCHEMA 
    AND TABLE_NAME = 'A2A_GRAPH_NODES';
    
    IF node_table_exists > 0 THEN
        -- Add migration tracking columns to graph nodes
        ALTER TABLE A2A_GRAPH_NODES 
        ADD (
            VECTOR_DIMENSION INTEGER DEFAULT 384,
            EMBEDDING_MODEL NVARCHAR(50) DEFAULT 'all-MiniLM-L6-v2',
            MIGRATED_TO_768 BOOLEAN DEFAULT FALSE
        );
        
        INSERT INTO A2A_MIGRATION_LOG VALUES (
            'SCHEMA_UPDATE', 
            CURRENT_TIMESTAMP, 
            'Updated A2A_GRAPH_NODES table with migration tracking columns'
        );
    END IF;
END;

-- =============================================================================
-- DATA MIGRATION PROCEDURES
-- =============================================================================

-- 6. Create procedure for zero-vector migration (immediate compatibility)
CREATE OR REPLACE PROCEDURE MIGRATE_WITH_ZERO_VECTORS()
AS BEGIN
    DECLARE record_count INTEGER;
    
    -- Insert existing records with zero-padded vectors for immediate compatibility
    INSERT INTO A2A_VECTORS_768_NEW (
        DOC_ID,
        CONTENT,
        METADATA,
        VECTOR_EMBEDDING,
        ENTITY_TYPE,
        SOURCE_AGENT,
        CREATED_AT,
        RE_EMBEDDED,
        ORIGINAL_DIMENSION
    )
    SELECT 
        DOC_ID,
        CONTENT,
        METADATA,
        -- Create 768-dim vector by zero-padding existing 384-dim vector
        ARRAY_CONCAT(
            TO_REAL_VECTOR(VECTOR_EMBEDDING), 
            ARRAY_AGG(CAST(0.0 AS REAL))[1:384]  -- Add 384 zeros
        ) AS VECTOR_EMBEDDING_768,
        ENTITY_TYPE,
        SOURCE_AGENT,
        CREATED_AT,
        FALSE,  -- Mark as not re-embedded
        384     -- Original dimension
    FROM A2A_VECTORS;
    
    GET DIAGNOSTICS record_count = ROW_COUNT;
    
    INSERT INTO A2A_MIGRATION_LOG VALUES (
        'ZERO_VECTOR_MIGRATION', 
        CURRENT_TIMESTAMP, 
        'Migrated ' || record_count || ' records with zero-padded vectors'
    );
END;

-- 7. Create procedure for re-embedding (requires application-level processing)
CREATE OR REPLACE PROCEDURE UPDATE_VECTOR_FOR_RE_EMBEDDING(
    IN p_doc_id NVARCHAR(255),
    IN p_new_vector REAL_VECTOR(768)
)
AS BEGIN
    UPDATE A2A_VECTORS_768_NEW 
    SET 
        VECTOR_EMBEDDING = p_new_vector,
        RE_EMBEDDED = TRUE,
        MIGRATED_AT = CURRENT_TIMESTAMP,
        EMBEDDING_MODEL = 'all-mpnet-base-v2'
    WHERE DOC_ID = p_doc_id;
    
    IF SQL%ROWCOUNT = 0 THEN
        SIGNAL SQL_ERROR_CODE 10003 
        SET MESSAGE_TEXT = 'Document ID not found for re-embedding: ' || p_doc_id;
    END IF;
END;

-- =============================================================================
-- EXECUTION AND CUTOVER
-- =============================================================================

-- 8. Execute zero-vector migration
CALL MIGRATE_WITH_ZERO_VECTORS();

-- 9. Create view for backward compatibility during transition
CREATE OR REPLACE VIEW A2A_VECTORS_COMPATIBILITY AS 
SELECT 
    DOC_ID,
    CONTENT,
    METADATA,
    -- Truncate 768-dim vector to 384-dim for legacy compatibility
    SUBARRAY(VECTOR_EMBEDDING, 1, 384) AS VECTOR_EMBEDDING_384,
    VECTOR_EMBEDDING AS VECTOR_EMBEDDING_768,
    ENTITY_TYPE,
    SOURCE_AGENT,
    CREATED_AT,
    RE_EMBEDDED,
    EMBEDDING_MODEL
FROM A2A_VECTORS_768_NEW;

-- 10. Rename tables for cutover (EXECUTE MANUALLY AFTER VALIDATION)
-- This section should be executed manually after validation
/*
-- Validation steps before cutover:
-- 1. Verify record counts match
-- 2. Test application connectivity with new schema
-- 3. Validate that Agent 2 and Agent 3 are configured for 768-dim

-- CUTOVER COMMANDS (EXECUTE MANUALLY):
RENAME TABLE A2A_VECTORS TO A2A_VECTORS_384_OLD;
RENAME TABLE A2A_VECTORS_768_NEW TO A2A_VECTORS;

-- Update any existing views or stored procedures
CREATE OR REPLACE VIEW A2A_CURRENT_VECTORS AS 
SELECT * FROM A2A_VECTORS;

INSERT INTO A2A_MIGRATION_LOG VALUES (
    'CUTOVER_COMPLETE', 
    CURRENT_TIMESTAMP, 
    'Successfully completed migration to 768-dimensional vectors'
);
*/

-- =============================================================================
-- POST-MIGRATION VALIDATION
-- =============================================================================

-- 11. Create validation procedure
CREATE OR REPLACE PROCEDURE VALIDATE_MIGRATION()
AS BEGIN
    DECLARE original_count INTEGER;
    DECLARE migrated_count INTEGER;
    DECLARE zero_embedded_count INTEGER;
    DECLARE re_embedded_count INTEGER;
    
    -- Count original records
    SELECT COUNT(*) INTO original_count FROM A2A_VECTORS;
    
    -- Count migrated records
    SELECT COUNT(*) INTO migrated_count FROM A2A_VECTORS_768_NEW;
    
    -- Count records by embedding status
    SELECT COUNT(*) INTO zero_embedded_count 
    FROM A2A_VECTORS_768_NEW WHERE RE_EMBEDDED = FALSE;
    
    SELECT COUNT(*) INTO re_embedded_count 
    FROM A2A_VECTORS_768_NEW WHERE RE_EMBEDDED = TRUE;
    
    -- Validation checks
    IF original_count != migrated_count THEN
        SIGNAL SQL_ERROR_CODE 10004 
        SET MESSAGE_TEXT = 'Record count mismatch: Original=' || original_count || ', Migrated=' || migrated_count;
    END IF;
    
    -- Log validation results
    INSERT INTO A2A_MIGRATION_LOG VALUES (
        'VALIDATION', 
        CURRENT_TIMESTAMP, 
        'Migration validation passed. Records: ' || migrated_count || 
        ', Zero-embedded: ' || zero_embedded_count || 
        ', Re-embedded: ' || re_embedded_count
    );
END;

-- Execute validation
CALL VALIDATE_MIGRATION();

-- =============================================================================
-- MONITORING AND MAINTENANCE VIEWS
-- =============================================================================

-- 12. Create monitoring views
CREATE OR REPLACE VIEW A2A_MIGRATION_STATUS AS
SELECT 
    COUNT(*) AS total_records,
    COUNT(CASE WHEN RE_EMBEDDED = TRUE THEN 1 END) AS re_embedded_count,
    COUNT(CASE WHEN RE_EMBEDDED = FALSE THEN 1 END) AS zero_padded_count,
    ROUND(
        COUNT(CASE WHEN RE_EMBEDDED = TRUE THEN 1 END) * 100.0 / COUNT(*), 2
    ) AS re_embedding_percentage,
    MIN(MIGRATED_AT) AS migration_start,
    MAX(MIGRATED_AT) AS last_update
FROM A2A_VECTORS_768_NEW;

-- 13. Re-embedding progress by entity type
CREATE OR REPLACE VIEW A2A_RE_EMBEDDING_PROGRESS AS
SELECT 
    ENTITY_TYPE,
    COUNT(*) AS total_entities,
    COUNT(CASE WHEN RE_EMBEDDED = TRUE THEN 1 END) AS re_embedded,
    ROUND(
        COUNT(CASE WHEN RE_EMBEDDED = TRUE THEN 1 END) * 100.0 / COUNT(*), 2
    ) AS completion_percentage
FROM A2A_VECTORS_768_NEW
GROUP BY ENTITY_TYPE
ORDER BY completion_percentage DESC;

-- =============================================================================
-- CLEANUP PROCEDURES (USE AFTER FULL VALIDATION)
-- =============================================================================

-- 14. Create cleanup procedure (execute after successful migration and validation)
CREATE OR REPLACE PROCEDURE CLEANUP_MIGRATION_ARTIFACTS()
AS BEGIN
    DECLARE backup_count INTEGER;
    
    -- Verify backup table exists
    SELECT COUNT(*) INTO backup_count FROM A2A_VECTORS_384_BACKUP;
    
    INSERT INTO A2A_MIGRATION_LOG VALUES (
        'CLEANUP_INFO', 
        CURRENT_TIMESTAMP, 
        'Backup table contains ' || backup_count || ' records. Ready for cleanup after validation period.'
    );
    
    -- NOTE: Uncomment these lines only after extended validation period
    -- DROP TABLE A2A_VECTORS_384_BACKUP;
    -- DROP TABLE A2A_VECTORS_384_OLD;  -- Only after cutover
    -- DROP VIEW A2A_VECTORS_COMPATIBILITY;
END;

-- =============================================================================
-- MIGRATION SUMMARY
-- =============================================================================

-- Display migration summary
SELECT 'MIGRATION SUMMARY' AS status;
SELECT * FROM A2A_MIGRATION_LOG ORDER BY TIMESTAMP DESC;
SELECT * FROM A2A_MIGRATION_STATUS;
SELECT * FROM A2A_RE_EMBEDDING_PROGRESS;

-- Final success message
INSERT INTO A2A_MIGRATION_LOG VALUES (
    'MIGRATION_COMPLETE', 
    CURRENT_TIMESTAMP, 
    'Migration to 768-dimensional vectors completed successfully. ' ||
    'Schema ready for all-mpnet-base-v2 model. ' ||
    'Re-embedding can proceed incrementally via Agent 2.'
);

/*
MIGRATION EXECUTION CHECKLIST:
☐ 1. Review and understand all migration steps
☐ 2. Ensure Agent 2 and Agent 3 are updated to use all-mpnet-base-v2
☐ 3. Stop applications accessing A2A_VECTORS during migration
☐ 4. Execute this migration script
☐ 5. Validate record counts and data integrity  
☐ 6. Update application configurations for 768-dimensional vectors
☐ 7. Test applications with new schema
☐ 8. Execute manual cutover commands (step 10)
☐ 9. Begin re-embedding process through Agent 2
☐ 10. Monitor re-embedding progress via monitoring views
☐ 11. Clean up migration artifacts after validation period

ROLLBACK PLAN:
If issues occur, restore from A2A_VECTORS_384_BACKUP:
- RENAME TABLE A2A_VECTORS TO A2A_VECTORS_768_FAILED;
- RENAME TABLE A2A_VECTORS_384_BACKUP TO A2A_VECTORS;
- Reconfigure agents back to all-MiniLM-L6-v2 model
*/