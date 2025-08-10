const cds = require('@sap/cds');
const { createClient } = require('@supabase/supabase-js');

/**
 * Database Service - HANA Primary, Supabase Backup
 * Implements dual database architecture as used in the Python backend
 */
class DatabaseService extends cds.Service {
    async init() {
        // Initialize HANA connection (primary)
        try {
            this.hana = await cds.connect.to('db');
            this.log.info('Connected to HANA (primary database)');
        } catch (error) {
            this.log.error('HANA connection failed:', error);
            this.hanaAvailable = false;
        }
        
        // Initialize Supabase connection (backup)
        try {
            this.supabase = createClient(
                process.env.SUPABASE_URL,
                process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_KEY
            );
            
            // Test Supabase connection
            const { data, error } = await this.supabase.from('agents').select('count').limit(1);
            if (!error) {
                this.log.info('Connected to Supabase (backup database)');
                this.supabaseAvailable = true;
            }
        } catch (error) {
            this.log.warn('Supabase connection failed:', error);
            this.supabaseAvailable = false;
        }
        
        // Register handlers for database operations
        this.on('query', this._handleQuery);
        this.on('insert', this._handleInsert);
        this.on('update', this._handleUpdate);
        this.on('delete', this._handleDelete);
        
        await super.init();
    }
    
    async _handleQuery(req) {
        const { entity, query } = req.data;
        
        try {
            // Try HANA first (primary)
            if (this.hana) {
                const result = await this.hana.run(query);
                this.log.debug(`Query executed on HANA for entity: ${entity}`);
                return result;
            }
        } catch (error) {
            this.log.warn(`HANA query failed for ${entity}:`, error);
            // Fall back to Supabase
        }
        
        // Fallback to Supabase
        if (this.supabaseAvailable) {
            try {
                const result = await this._executeSupabaseQuery(entity, query);
                this.log.info(`Query executed on Supabase (fallback) for entity: ${entity}`);
                return result;
            } catch (error) {
                this.log.error(`Supabase query failed for ${entity}:`, error);
                throw error;
            }
        }
        
        throw new Error('No database connection available');
    }
    
    async _handleInsert(req) {
        const { entity, data } = req.data;
        
        // Try to insert to HANA first
        let hanaSuccess = false;
        if (this.hana) {
            try {
                await this.hana.run(INSERT.into(entity).entries(data));
                hanaSuccess = true;
                this.log.debug(`Insert to HANA successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`HANA insert failed for ${entity}:`, error);
            }
        }
        
        // Also insert to Supabase for redundancy
        if (this.supabaseAvailable) {
            try {
                const tableName = this._getSupabaseTableName(entity);
                await this.supabase.from(tableName).insert(data);
                this.log.debug(`Insert to Supabase successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`Supabase insert failed for ${entity}:`, error);
            }
        }
        
        if (!hanaSuccess && !this.supabaseAvailable) {
            throw new Error('Insert failed on all databases');
        }
        
        return { success: true, primary: hanaSuccess, backup: this.supabaseAvailable };
    }
    
    async _handleUpdate(req) {
        const { entity, data, where } = req.data;
        
        // Try to update HANA first
        let hanaSuccess = false;
        if (this.hana) {
            try {
                await this.hana.run(UPDATE(entity).set(data).where(where));
                hanaSuccess = true;
                this.log.debug(`Update to HANA successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`HANA update failed for ${entity}:`, error);
            }
        }
        
        // Also update Supabase
        if (this.supabaseAvailable) {
            try {
                const tableName = this._getSupabaseTableName(entity);
                await this.supabase.from(tableName).update(data).match(where);
                this.log.debug(`Update to Supabase successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`Supabase update failed for ${entity}:`, error);
            }
        }
        
        if (!hanaSuccess && !this.supabaseAvailable) {
            throw new Error('Update failed on all databases');
        }
        
        return { success: true, primary: hanaSuccess, backup: this.supabaseAvailable };
    }
    
    async _executeSupabaseQuery(entity, query) {
        const tableName = this._getSupabaseTableName(entity);
        
        // Parse CDS query and convert to Supabase query
        // This is a simplified implementation - in production would need full query parser
        if (query.SELECT) {
            let supabaseQuery = this.supabase.from(tableName).select('*');
            
            if (query.SELECT.where) {
                // Apply where conditions
                for (const [field, value] of Object.entries(query.SELECT.where)) {
                    supabaseQuery = supabaseQuery.eq(field, value);
                }
            }
            
            if (query.SELECT.orderBy) {
                const { column, order } = query.SELECT.orderBy;
                supabaseQuery = supabaseQuery.order(column, { ascending: order === 'asc' });
            }
            
            if (query.SELECT.limit) {
                supabaseQuery = supabaseQuery.limit(query.SELECT.limit);
            }
            
            const { data, error } = await supabaseQuery;
            if (error) throw error;
            return data;
        }
        
        throw new Error('Unsupported query type for Supabase fallback');
    }
    
    _getSupabaseTableName(entity) {
        // Convert CAP entity names to Supabase table names
        const tableMap = {
            'a2a.network.Agents': 'agents',
            'a2a.network.Services': 'services',
            'a2a.network.Capabilities': 'capabilities',
            'a2a.network.AgentCapabilities': 'agent_capabilities',
            'a2a.network.Messages': 'messages',
            'a2a.network.Workflows': 'workflows',
            'a2a.network.WorkflowExecutions': 'workflow_executions',
            'a2a.network.NetworkStats': 'network_stats'
        };
        
        return tableMap[entity] || entity.toLowerCase().replace(/\./g, '_');
    }
    
    // Health check for both databases
    async getHealthStatus() {
        const status = {
            hana: { available: false, responseTime: null },
            supabase: { available: false, responseTime: null },
            primary: 'hana',
            backup: 'supabase'
        };
        
        // Check HANA
        if (this.hana) {
            try {
                const start = Date.now();
                await this.hana.run('SELECT 1 FROM DUMMY');
                status.hana.available = true;
                status.hana.responseTime = Date.now() - start;
            } catch (error) {
                status.hana.error = error.message;
            }
        }
        
        // Check Supabase
        if (this.supabaseAvailable) {
            try {
                const start = Date.now();
                await this.supabase.from('agents').select('count').limit(1);
                status.supabase.available = true;
                status.supabase.responseTime = Date.now() - start;
            } catch (error) {
                status.supabase.error = error.message;
            }
        }
        
        return status;
    }
}

module.exports = DatabaseService;