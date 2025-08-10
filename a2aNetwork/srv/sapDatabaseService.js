/**
 * @fileoverview SAP Database Service - Dual architecture with HANA primary and SQLite backup
 * @description Implements database abstraction layer supporting both SAP HANA Cloud 
 * and SQLite databases for high availability and disaster recovery scenarios.
 * @module sapDatabaseService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.db
 */

const cds = require('@sap/cds');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

/**
 * Database Service - HANA Primary, SQLite Backup
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
        
        // Initialize SQLite connection (backup)
        try {
            const dbPath = process.env.SQLITE_DB_PATH || path.join(__dirname, '../data/a2a_fallback.db');
            const dbDir = path.dirname(dbPath);
            
            // Ensure directory exists
            if (!fs.existsSync(dbDir)) {
                fs.mkdirSync(dbDir, { recursive: true });
            }
            
            this.sqlite = new sqlite3.Database(dbPath, (err) => {
                if (err) {
                    this.log.error('SQLite connection error:', err);
                    this.sqliteAvailable = false;
                } else {
                    this.log.info('Connected to SQLite (backup database)');
                    this.sqliteAvailable = true;
                    this._initSQLiteTables();
                }
            });
        } catch (error) {
            this.log.warn('SQLite connection failed:', error);
            this.sqliteAvailable = false;
        }
        
        // Register handlers for database operations
        this.on('query', this._handleQuery);
        this.on('insert', this._handleInsert);
        this.on('update', this._handleUpdate);
        this.on('delete', this._handleDelete);
        
        await super.init();
    }
    
    _initSQLiteTables() {
        // Initialize SQLite tables
        const tables = [
            `CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )`,
            `CREATE TABLE IF NOT EXISTS services (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                endpoint TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )`,
            `CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT
            )`,
            `CREATE TABLE IF NOT EXISTS agent_capabilities (
                agent_id TEXT,
                capability_id TEXT,
                PRIMARY KEY (agent_id, capability_id),
                FOREIGN KEY (agent_id) REFERENCES agents(id),
                FOREIGN KEY (capability_id) REFERENCES capabilities(id)
            )`,
            `CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                from_agent TEXT,
                to_agent TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )`,
            `CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                definition TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )`,
            `CREATE TABLE IF NOT EXISTS workflow_executions (
                id TEXT PRIMARY KEY,
                workflow_id TEXT,
                status TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )`,
            `CREATE TABLE IF NOT EXISTS network_stats (
                id TEXT PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )`
        ];
        
        tables.forEach(sql => {
            this.sqlite.run(sql, (err) => {
                if (err) {
                    this.log.error('Error creating SQLite table:', err);
                }
            });
        });
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
            // Fall back to SQLite
        }
        
        // Fallback to SQLite
        if (this.sqliteAvailable) {
            try {
                const result = await this._executeSQLiteQuery(entity, query);
                this.log.info(`Query executed on SQLite (fallback) for entity: ${entity}`);
                return result;
            } catch (error) {
                this.log.error(`SQLite query failed for ${entity}:`, error);
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
        
        // Also insert to SQLite for redundancy
        if (this.sqliteAvailable) {
            try {
                const tableName = this._getSQLiteTableName(entity);
                await this._insertSQLite(tableName, data);
                this.log.debug(`Insert to SQLite successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`SQLite insert failed for ${entity}:`, error);
            }
        }
        
        if (!hanaSuccess && !this.sqliteAvailable) {
            throw new Error('Insert failed on all databases');
        }
        
        return { success: true, primary: hanaSuccess, backup: this.sqliteAvailable };
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
        
        // Also update SQLite
        if (this.sqliteAvailable) {
            try {
                const tableName = this._getSQLiteTableName(entity);
                await this._updateSQLite(tableName, data, where);
                this.log.debug(`Update to SQLite successful for entity: ${entity}`);
            } catch (error) {
                this.log.warn(`SQLite update failed for ${entity}:`, error);
            }
        }
        
        if (!hanaSuccess && !this.sqliteAvailable) {
            throw new Error('Update failed on all databases');
        }
        
        return { success: true, primary: hanaSuccess, backup: this.sqliteAvailable };
    }
    
    async _executeSQLiteQuery(entity, query) {
        const tableName = this._getSQLiteTableName(entity);
        
        // Parse CDS query and convert to SQLite query
        // This is a simplified implementation - in production would need full query parser
        if (query.SELECT) {
            return new Promise((resolve, reject) => {
                let sql = `SELECT * FROM ${tableName}`;
                const params = [];
                
                if (query.SELECT.where) {
                    const whereConditions = [];
                    for (const [field, value] of Object.entries(query.SELECT.where)) {
                        whereConditions.push(`${field} = ?`);
                        params.push(value);
                    }
                    sql += ` WHERE ${whereConditions.join(' AND ')}`;
                }
                
                if (query.SELECT.orderBy) {
                    const { column, order } = query.SELECT.orderBy;
                    sql += ` ORDER BY ${column} ${order === 'asc' ? 'ASC' : 'DESC'}`;
                }
                
                if (query.SELECT.limit) {
                    sql += ` LIMIT ${query.SELECT.limit}`;
                }
                
                this.sqlite.all(sql, params, (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                });
            });
        }
        
        throw new Error('Unsupported query type for SQLite fallback');
    }
    
    async _insertSQLite(tableName, data) {
        return new Promise((resolve, reject) => {
            const entries = Array.isArray(data) ? data : [data];
            
            entries.forEach(entry => {
                const fields = Object.keys(entry);
                const placeholders = fields.map(() => '?').join(',');
                const values = fields.map(f => entry[f]);
                
                const sql = `INSERT INTO ${tableName} (${fields.join(',')}) VALUES (${placeholders})`;
                
                this.sqlite.run(sql, values, function(err) {
                    if (err) reject(err);
                    else resolve({ id: this.lastID });
                });
            });
        });
    }
    
    async _updateSQLite(tableName, data, where) {
        return new Promise((resolve, reject) => {
            const setClause = Object.keys(data).map(field => `${field} = ?`).join(', ');
            const whereClause = Object.keys(where).map(field => `${field} = ?`).join(' AND ');
            const values = [...Object.values(data), ...Object.values(where)];
            
            const sql = `UPDATE ${tableName} SET ${setClause} WHERE ${whereClause}`;
            
            this.sqlite.run(sql, values, function(err) {
                if (err) reject(err);
                else resolve({ changes: this.changes });
            });
        });
    }
    
    _getSQLiteTableName(entity) {
        // Convert CAP entity names to SQLite table names
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
            sqlite: { available: false, responseTime: null },
            primary: 'hana',
            backup: 'sqlite'
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
        
        // Check SQLite
        if (this.sqliteAvailable) {
            try {
                const start = Date.now();
                await new Promise((resolve, reject) => {
                    this.sqlite.get('SELECT 1', (err) => {
                        if (err) reject(err);
                        else resolve();
                    });
                });
                status.sqlite.available = true;
                status.sqlite.responseTime = Date.now() - start;
            } catch (error) {
                status.sqlite.error = error.message;
            }
        }
        
        return status;
    }
}

module.exports = DatabaseService;