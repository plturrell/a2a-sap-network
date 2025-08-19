/**
 * @fileoverview SAP Database Service - Enterprise Connection Pool Management
 * @description Implements advanced database connection pooling with SAP HANA Cloud optimization,
 * circuit breaker pattern, connection monitoring, and automatic failover capabilities.
 * @module sapDatabaseService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.db
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');
const EventEmitter = require('events');

/**
 * Advanced Connection Pool for SAP HANA Cloud
 */
class ConnectionPool extends EventEmitter {
    constructor(options = {}) {
        super();
        this.options = {
            minConnections: options.minConnections || 5,
            maxConnections: options.maxConnections || 50,
            acquireTimeout: options.acquireTimeout || 30000,
            idleTimeout: options.idleTimeout || 300000,
            checkInterval: options.checkInterval || 30000,
            maxRetries: options.maxRetries || 3,
            retryDelay: options.retryDelay || 1000,
            connectionTimeout: options.connectionTimeout || 10000,
            healthCheckInterval: options.healthCheckInterval || 60000,
            ...options
        };
        
        this.connections = new Set();
        this.available = [];
        this.pending = [];
        this.metrics = {
            totalConnections: 0,
            activeConnections: 0,
            waitingRequests: 0,
            successfulAcquisitions: 0,
            failedAcquisitions: 0,
            timeouts: 0,
            errors: 0,
            averageWaitTime: 0,
            averageConnectionTime: 0,
            peakConnections: 0,
            poolUtilization: 0
        };
        
        this.circuitBreaker = {
            state: 'CLOSED', // CLOSED, OPEN, HALF_OPEN
            failures: 0,
            lastFailureTime: null,
            threshold: options.circuitBreakerThreshold || 5,
            timeout: options.circuitBreakerTimeout || 60000,
            resetTimeout: options.circuitBreakerResetTimeout || 30000
        };
        
        this.log = cds.log('connection-pool');
        this.isInitialized = false;
        this.isDestroyed = false;
        
        // Start monitoring intervals
        this._startMonitoring();
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            // Create minimum connections
            for (let i = 0; i < this.options.minConnections; i++) {
                await this._createConnection();
            }
            
            this.isInitialized = true;
            this.log.info(`Connection pool initialized with ${this.options.minConnections} connections`);
            this.emit('initialized');
        } catch (error) {
            this.log.error('Failed to initialize connection pool:', error);
            throw error;
        }
    }
    
    async acquire() {
        if (this.isDestroyed) {
            throw new Error('Connection pool has been destroyed');
        }
        
        if (this.circuitBreaker.state === 'OPEN') {
            if (Date.now() - this.circuitBreaker.lastFailureTime < this.circuitBreaker.timeout) {
                throw new Error('Circuit breaker is OPEN - database connections blocked');
            } else {
                this.circuitBreaker.state = 'HALF_OPEN';
                this.log.info('Circuit breaker moved to HALF_OPEN state');
            }
        }
        
        const startTime = Date.now();
        this.metrics.waitingRequests++;
        
        try {
            const connection = await this._acquireConnection();
            const waitTime = Date.now() - startTime;
            
            this.metrics.successfulAcquisitions++;
            this.metrics.waitingRequests--;
            this.metrics.activeConnections++;
            this._updateAverageWaitTime(waitTime);
            this._updatePoolUtilization();
            
            // Reset circuit breaker on success
            if (this.circuitBreaker.state === 'HALF_OPEN') {
                this.circuitBreaker.state = 'CLOSED';
                this.circuitBreaker.failures = 0;
                this.log.info('Circuit breaker reset to CLOSED state');
            }
            
            return connection;
        } catch (error) {
            this.metrics.failedAcquisitions++;
            this.metrics.waitingRequests--;
            this._handleCircuitBreakerFailure();
            
            this.log.error('Failed to acquire connection:', error);
            throw error;
        }
    }
    
    release(connection) {
        if (!this.connections.has(connection)) {
            this.log.warn('Attempting to release unknown connection');
            return;
        }
        
        this.metrics.activeConnections--;
        this._updatePoolUtilization();
        
        // Check if connection is still healthy
        if (this._isConnectionHealthy(connection)) {
            connection.lastUsed = Date.now();
            this.available.push(connection);
            
            // Process pending requests
            if (this.pending.length > 0) {
                const { resolve } = this.pending.shift();
                resolve(this._borrowConnection());
            }
        } else {
            this._removeConnection(connection);
            // Replace with new connection if below minimum
            if (this.connections.size < this.options.minConnections) {
                this._createConnection().catch(error => {
                    this.log.error('Failed to create replacement connection:', error);
                });
            }
        }
        
        this.emit('released', { connection, poolSize: this.connections.size });
    }
    
    async _acquireConnection() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.metrics.timeouts++;
                reject(new Error('Connection acquisition timeout'));
            }, this.options.acquireTimeout);
            
            const attemptAcquisition = async () => {
                try {
                    // Check available connections
                    if (this.available.length > 0) {
                        clearTimeout(timeout);
                        resolve(this._borrowConnection());
                        return;
                    }
                    
                    // Create new connection if under limit
                    if (this.connections.size < this.options.maxConnections) {
                        const connection = await this._createConnection();
                        clearTimeout(timeout);
                        resolve(connection);
                        return;
                    }
                    
                    // Add to pending queue
                    this.pending.push({ resolve: (conn) => {
                        clearTimeout(timeout);
                        resolve(conn);
                    }, reject, timeout });
                    
                } catch (error) {
                    clearTimeout(timeout);
                    reject(error);
                }
            };
            
            attemptAcquisition();
        });
    }
    
    async _createConnection() {
        const startTime = Date.now();
        
        try {
            const connection = await cds.connect.to('db');
            const connectionTime = Date.now() - startTime;
            
            // Enhance connection with metadata
            connection.id = `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            connection.createdAt = new Date();
            connection.lastUsed = Date.now();
            connection.queryCount = 0;
            connection.errorCount = 0;
            connection.isHealthy = true;
            
            this.connections.add(connection);
            this.available.push(connection);
            this.metrics.totalConnections++;
            this.metrics.peakConnections = Math.max(this.metrics.peakConnections, this.connections.size);
            this._updateAverageConnectionTime(connectionTime);
            
            this.log.debug(`Created new connection ${connection.id} in ${connectionTime}ms`);
            this.emit('created', { connection, poolSize: this.connections.size });
            
            return connection;
        } catch (error) {
            this.metrics.errors++;
            this.log.error('Failed to create database connection:', error);
            throw error;
        }
    }
    
    _borrowConnection() {
        const connection = this.available.shift();
        if (connection) {
            connection.lastUsed = Date.now();
            return connection;
        }
        throw new Error('No available connections');
    }
    
    _removeConnection(connection) {
        this.connections.delete(connection);
        const index = this.available.indexOf(connection);
        if (index > -1) {
            this.available.splice(index, 1);
        }
        
        // Close connection gracefully
        try {
            if (connection.disconnect) {
                connection.disconnect();
            }
        } catch (error) {
            this.log.warn('Error closing connection:', error);
        }
        
        this.emit('removed', { connection, poolSize: this.connections.size });
    }
    
    _isConnectionHealthy(connection) {
        const now = Date.now();
        const isIdle = (now - connection.lastUsed) > this.options.idleTimeout;
        const hasErrors = connection.errorCount > 5;
        
        return connection.isHealthy && !isIdle && !hasErrors;
    }
    
    _startMonitoring() {
        // Health check interval
        this.healthCheckInterval = setInterval(() => {
            this._performHealthCheck();
        }, this.options.healthCheckInterval);
        
        // Cleanup idle connections
        this.cleanupInterval = setInterval(() => {
            this._cleanupIdleConnections();
        }, this.options.checkInterval);
        
        // Metrics reporting
        this.metricsInterval = setInterval(() => {
            this._reportMetrics();
        }, 60000); // Every minute
    }
    
    async _performHealthCheck() {
        const healthyConnections = [];
        
        for (const connection of this.available) {
            try {
                await connection.run('SELECT 1');
                connection.isHealthy = true;
                healthyConnections.push(connection);
            } catch (error) {
                connection.isHealthy = false;
                connection.errorCount++;
                this.log.warn(`Connection ${connection.id} failed health check:`, error);
                this._removeConnection(connection);
            }
        }
        
        this.available = healthyConnections;
        
        // Ensure minimum connections
        const neededConnections = this.options.minConnections - this.connections.size;
        for (let i = 0; i < neededConnections; i++) {
            try {
                await this._createConnection();
            } catch (error) {
                this.log.error('Failed to create connection during health check:', error);
                break;
            }
        }
    }
    
    _cleanupIdleConnections() {
        const now = Date.now();
        const toRemove = [];
        
        for (const connection of this.available) {
            if ((now - connection.lastUsed) > this.options.idleTimeout && 
                this.connections.size > this.options.minConnections) {
                toRemove.push(connection);
            }
        }
        
        toRemove.forEach(connection => this._removeConnection(connection));
        
        if (toRemove.length > 0) {
            this.log.debug(`Removed ${toRemove.length} idle connections`);
        }
    }
    
    _handleCircuitBreakerFailure() {
        this.circuitBreaker.failures++;
        this.circuitBreaker.lastFailureTime = Date.now();
        
        if (this.circuitBreaker.failures >= this.circuitBreaker.threshold) {
            this.circuitBreaker.state = 'OPEN';
            this.log.warn(`Circuit breaker opened after ${this.circuitBreaker.failures} failures`);
            this.emit('circuitBreakerOpened');
            
            // Schedule reset attempt
            setTimeout(() => {
                if (this.circuitBreaker.state === 'OPEN') {
                    this.circuitBreaker.state = 'HALF_OPEN';
                    this.log.info('Circuit breaker moved to HALF_OPEN for testing');
                }
            }, this.circuitBreaker.resetTimeout);
        }
    }
    
    _updateAverageWaitTime(waitTime) {
        const alpha = 0.1;
        this.metrics.averageWaitTime = this.metrics.averageWaitTime === 0 
            ? waitTime 
            : (alpha * waitTime) + ((1 - alpha) * this.metrics.averageWaitTime);
    }
    
    _updateAverageConnectionTime(connectionTime) {
        const alpha = 0.1;
        this.metrics.averageConnectionTime = this.metrics.averageConnectionTime === 0 
            ? connectionTime 
            : (alpha * connectionTime) + ((1 - alpha) * this.metrics.averageConnectionTime);
    }
    
    _updatePoolUtilization() {
        this.metrics.poolUtilization = (this.metrics.activeConnections / this.options.maxConnections) * 100;
    }
    
    _reportMetrics() {
        this.log.info('Connection Pool Metrics:', {
            totalConnections: this.connections.size,
            availableConnections: this.available.length,
            activeConnections: this.metrics.activeConnections,
            waitingRequests: this.metrics.waitingRequests,
            poolUtilization: this.metrics.poolUtilization.toFixed(2) + '%',
            averageWaitTime: this.metrics.averageWaitTime.toFixed(2) + 'ms',
            averageConnectionTime: this.metrics.averageConnectionTime.toFixed(2) + 'ms',
            circuitBreakerState: this.circuitBreaker.state,
            successRate: this.metrics.totalConnections > 0 
                ? ((this.metrics.successfulAcquisitions / (this.metrics.successfulAcquisitions + this.metrics.failedAcquisitions)) * 100).toFixed(2) + '%'
                : '0%'
        });
        
        this.emit('metrics', this.metrics);
    }
    
    getMetrics() {
        return {
            ...this.metrics,
            poolSize: this.connections.size,
            availableConnections: this.available.length,
            circuitBreakerState: this.circuitBreaker.state,
            isInitialized: this.isInitialized
        };
    }
    
    async destroy() {
        this.isDestroyed = true;
        
        // Clear intervals
        clearInterval(this.healthCheckInterval);
        clearInterval(this.cleanupInterval);
        clearInterval(this.metricsInterval);
        
        // Close all connections
        const closePromises = Array.from(this.connections).map(async (connection) => {
            try {
                if (connection.disconnect) {
                    await connection.disconnect();
                }
            } catch (error) {
                this.log.warn('Error closing connection during destroy:', error);
            }
        });
        
        await Promise.allSettled(closePromises);
        
        this.connections.clear();
        this.available = [];
        this.pending = [];
        
        this.log.info('Connection pool destroyed');
        this.emit('destroyed');
    }
}

/**
 * Database Service with Advanced Connection Pool Management
 * Implements enterprise-grade connection pooling with monitoring and failover
 */
class DatabaseService extends cds.Service {
    async init() {
        // Initialize advanced connection pool
        this.connectionPool = new ConnectionPool({
            minConnections: parseInt(process.env.DB_MIN_CONNECTIONS) || 5,
            maxConnections: parseInt(process.env.DB_MAX_CONNECTIONS) || 50,
            acquireTimeout: parseInt(process.env.DB_ACQUIRE_TIMEOUT) || 30000,
            idleTimeout: parseInt(process.env.DB_IDLE_TIMEOUT) || 300000,
            circuitBreakerThreshold: parseInt(process.env.DB_CIRCUIT_BREAKER_THRESHOLD) || 5,
            healthCheckInterval: parseInt(process.env.DB_HEALTH_CHECK_INTERVAL) || 60000
        });
        
        try {
            await this.connectionPool.initialize();
            this.log.info('Advanced connection pool initialized successfully');
            
            // Set up connection pool monitoring
            this.connectionPool.on('circuitBreakerOpened', () => {
                this.log.error('Database connection circuit breaker opened - connections blocked');
                this.emit('database.circuitBreakerOpened');
            });
            
            this.connectionPool.on('metrics', (metrics) => {
                this.emit('database.metrics', metrics);
            });
            
        } catch (error) {
            this.log.error('Failed to initialize connection pool:', error);
            this.hanaAvailable = false;
        }
        
        // SQLite backup database disabled in production - use HANA only
        this.sqliteAvailable = false;
        this.log.info('SQLite backup database disabled - using HANA primary database only');
        
        // Register handlers for database operations
        this.on('query', this._handleQuery);
        this.on('insert', this._handleInsert);
        this.on('update', this._handleUpdate);
        this.on('delete', this._handleDelete);
        
        // Start connection pool monitoring
        this._startPoolMonitoring();
        
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
        const startTime = Date.now();
        let connection = null;
        
        try {
            // Acquire connection from pool
            connection = await this.connectionPool.acquire();
            connection.queryCount++;
            
            const result = await connection.run(query);
            const duration = Date.now() - startTime;
            
            this.log.debug(`Query executed for entity ${entity} in ${duration}ms`);
            this._recordQueryMetrics('query', entity, duration, true);
            
            return result;
            
        } catch (error) {
            const duration = Date.now() - startTime;
            this.log.warn(`Query failed for ${entity} after ${duration}ms:`, error);
            this._recordQueryMetrics('query', entity, duration, false);
            
            if (connection) {
                connection.errorCount++;
            }
            
            throw new Error(`Database query failed: ${error.message}`);
        } finally {
            if (connection) {
                this.connectionPool.release(connection);
            }
        }
    }
    
    async _handleInsert(req) {
        const { entity, data } = req.data;
        const startTime = Date.now();
        let connection = null;
        
        try {
            // Acquire connection from pool
            connection = await this.connectionPool.acquire();
            connection.queryCount++;
            
            const result = await connection.run(INSERT.into(entity).entries(data));
            const duration = Date.now() - startTime;
            
            this.log.debug(`Insert to entity ${entity} successful in ${duration}ms`);
            this._recordQueryMetrics('insert', entity, duration, true);
            
            return { success: true, result };
            
        } catch (error) {
            const duration = Date.now() - startTime;
            this.log.warn(`Insert to ${entity} failed after ${duration}ms:`, error);
            this._recordQueryMetrics('insert', entity, duration, false);
            
            if (connection) {
                connection.errorCount++;
            }
            
            throw new Error(`Database insert failed: ${error.message}`);
        } finally {
            if (connection) {
                this.connectionPool.release(connection);
            }
        }
    }
    
    async _handleUpdate(req) {
        const { entity, data, where } = req.data;
        const startTime = Date.now();
        let connection = null;
        
        try {
            // Acquire connection from pool
            connection = await this.connectionPool.acquire();
            connection.queryCount++;
            
            const result = await connection.run(UPDATE(entity).set(data).where(where));
            const duration = Date.now() - startTime;
            
            this.log.debug(`Update to entity ${entity} successful in ${duration}ms`);
            this._recordQueryMetrics('update', entity, duration, true);
            
            return { success: true, result };
            
        } catch (error) {
            const duration = Date.now() - startTime;
            this.log.warn(`Update to ${entity} failed after ${duration}ms:`, error);
            this._recordQueryMetrics('update', entity, duration, false);
            
            if (connection) {
                connection.errorCount++;
            }
            
            throw new Error(`Database update failed: ${error.message}`);
        } finally {
            if (connection) {
                this.connectionPool.release(connection);
            }
        }
    }
    
    async _handleDelete(req) {
        const { entity, where } = req.data;
        const startTime = Date.now();
        let connection = null;
        
        try {
            // Acquire connection from pool
            connection = await this.connectionPool.acquire();
            connection.queryCount++;
            
            const result = await connection.run(DELETE.from(entity).where(where));
            const duration = Date.now() - startTime;
            
            this.log.debug(`Delete from entity ${entity} successful in ${duration}ms`);
            this._recordQueryMetrics('delete', entity, duration, true);
            
            return { success: true, result };
            
        } catch (error) {
            const duration = Date.now() - startTime;
            this.log.warn(`Delete from ${entity} failed after ${duration}ms:`, error);
            this._recordQueryMetrics('delete', entity, duration, false);
            
            if (connection) {
                connection.errorCount++;
            }
            
            throw new Error(`Database delete failed: ${error.message}`);
        } finally {
            if (connection) {
                this.connectionPool.release(connection);
            }
        }
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
    
    /**
     * Start connection pool monitoring and metrics collection
     */
    _startPoolMonitoring() {
        this.poolMetrics = {
            queryMetrics: new Map(),
            operationCounts: {
                query: 0,
                insert: 0,
                update: 0,
                delete: 0
            },
            errorCounts: {
                query: 0,
                insert: 0,
                update: 0,
                delete: 0
            },
            avgResponseTimes: {
                query: 0,
                insert: 0,
                update: 0,
                delete: 0
            }
        };
        
        // Monitor connection pool metrics
        this.connectionPool.on('metrics', (metrics) => {
            this.log.debug('Connection pool metrics updated:', metrics);
        });
        
        // Report database metrics every 5 minutes
        setInterval(() => {
            this._reportDatabaseMetrics();
        }, 300000);
    }
    
    /**
     * Record query performance metrics
     */
    _recordQueryMetrics(operation, entity, duration, success) {
        const key = `${operation}.${entity}`;
        
        if (!this.poolMetrics.queryMetrics.has(key)) {
            this.poolMetrics.queryMetrics.set(key, {
                count: 0,
                errors: 0,
                totalDuration: 0,
                avgDuration: 0,
                minDuration: Infinity,
                maxDuration: 0
            });
        }
        
        const metrics = this.poolMetrics.queryMetrics.get(key);
        metrics.count++;
        
        if (success) {
            metrics.totalDuration += duration;
            metrics.avgDuration = metrics.totalDuration / metrics.count;
            metrics.minDuration = Math.min(metrics.minDuration, duration);
            metrics.maxDuration = Math.max(metrics.maxDuration, duration);
            
            this.poolMetrics.operationCounts[operation]++;
            
            // Update global average response time
            const alpha = 0.1;
            this.poolMetrics.avgResponseTimes[operation] = 
                this.poolMetrics.avgResponseTimes[operation] === 0 
                    ? duration 
                    : (alpha * duration) + ((1 - alpha) * this.poolMetrics.avgResponseTimes[operation]);
        } else {
            metrics.errors++;
            this.poolMetrics.errorCounts[operation]++;
        }
    }
    
    /**
     * Report comprehensive database metrics
     */
    _reportDatabaseMetrics() {
        const poolMetrics = this.connectionPool.getMetrics();
        const dbMetrics = {
            connectionPool: poolMetrics,
            operations: this.poolMetrics.operationCounts,
            errors: this.poolMetrics.errorCounts,
            avgResponseTimes: this.poolMetrics.avgResponseTimes,
            topQueries: this._getTopQueries(),
            timestamp: new Date().toISOString()
        };
        
        this.log.info('Database Performance Metrics:', dbMetrics);
        this.emit('database.performanceMetrics', dbMetrics);
    }
    
    /**
     * Get top performing/problematic queries
     */
    _getTopQueries() {
        const queries = Array.from(this.poolMetrics.queryMetrics.entries())
            .map(([key, metrics]) => ({
                query: key,
                count: metrics.count,
                avgDuration: parseFloat(metrics.avgDuration.toFixed(2)),
                maxDuration: metrics.maxDuration,
                errorRate: metrics.errors > 0 ? (metrics.errors / metrics.count * 100).toFixed(2) : 0
            }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);
            
        return queries;
    }
    
    /**
     * Execute query with automatic retry and connection management
     */
    async executeQuery(query, retries = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= retries; attempt++) {
            let connection = null;
            try {
                connection = await this.connectionPool.acquire();
                const result = await connection.run(query);
                return result;
                
            } catch (error) {
                lastError = error;
                this.log.warn(`Query attempt ${attempt} failed:`, error);
                
                if (connection) {
                    connection.errorCount++;
                }
                
                // Wait before retry (exponential backoff)
                if (attempt < retries) {
                    await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
                }
            } finally {
                if (connection) {
                    this.connectionPool.release(connection);
                }
            }
        }
        
        throw new Error(`Query failed after ${retries} attempts: ${lastError.message}`);
    }
    
    /**
     * Execute transaction with connection pool management
     */
    async executeTransaction(operations) {
        let connection = null;
        
        try {
            connection = await this.connectionPool.acquire();
            
            // Begin transaction
            await connection.run('BEGIN TRANSACTION');
            
            const results = [];
            for (const operation of operations) {
                const result = await connection.run(operation);
                results.push(result);
            }
            
            // Commit transaction
            await connection.run('COMMIT');
            
            this.log.debug(`Transaction with ${operations.length} operations completed successfully`);
            return results;
            
        } catch (error) {
            // Rollback on error
            if (connection) {
                try {
                    await connection.run('ROLLBACK');
                } catch (rollbackError) {
                    this.log.error('Failed to rollback transaction:', rollbackError);
                }
                connection.errorCount++;
            }
            
            this.log.error('Transaction failed:', error);
            throw new Error(`Transaction failed: ${error.message}`);
        } finally {
            if (connection) {
                this.connectionPool.release(connection);
            }
        }
    }
    
    /**
     * Enhanced health check with connection pool status
     */
    async getHealthStatus() {
        const poolMetrics = this.connectionPool.getMetrics();
        
        const status = {
            connectionPool: {
                available: this.connectionPool.isInitialized,
                totalConnections: poolMetrics.poolSize,
                activeConnections: poolMetrics.activeConnections,
                availableConnections: poolMetrics.availableConnections,
                utilization: poolMetrics.poolUtilization.toFixed(2) + '%',
                circuitBreakerState: poolMetrics.circuitBreakerState,
                averageWaitTime: poolMetrics.averageWaitTime.toFixed(2) + 'ms',
                successRate: poolMetrics.successfulAcquisitions > 0 
                    ? ((poolMetrics.successfulAcquisitions / (poolMetrics.successfulAcquisitions + poolMetrics.failedAcquisitions)) * 100).toFixed(2) + '%'
                    : '100%'
            },
            database: {
                primary: 'hana',
                responseTime: null,
                available: false
            },
            performance: {
                avgQueryTime: this.poolMetrics.avgResponseTimes.query.toFixed(2) + 'ms',
                avgInsertTime: this.poolMetrics.avgResponseTimes.insert.toFixed(2) + 'ms',
                avgUpdateTime: this.poolMetrics.avgResponseTimes.update.toFixed(2) + 'ms',
                avgDeleteTime: this.poolMetrics.avgResponseTimes.delete.toFixed(2) + 'ms',
                totalOperations: Object.values(this.poolMetrics.operationCounts).reduce((a, b) => a + b, 0),
                totalErrors: Object.values(this.poolMetrics.errorCounts).reduce((a, b) => a + b, 0)
            }
        };
        
        // Test database connectivity
        try {
            const start = Date.now();
            await this.executeQuery('SELECT 1');
            status.database.available = true;
            status.database.responseTime = Date.now() - start;
        } catch (error) {
            status.database.error = error.message;
        }
        
        return status;
    }
    
    /**
     * Get detailed connection pool statistics
     */
    getConnectionPoolStats() {
        return {
            pool: this.connectionPool.getMetrics(),
            queries: this._getTopQueries(),
            operations: this.poolMetrics.operationCounts,
            errors: this.poolMetrics.errorCounts,
            avgResponseTimes: this.poolMetrics.avgResponseTimes
        };
    }
    
    /**
     * Gracefully shutdown the database service
     */
    async shutdown() {
        this.log.info('Shutting down database service...');
        
        try {
            await this.connectionPool.destroy();
            this.log.info('Database service shutdown completed');
        } catch (error) {
            this.log.error('Error during database service shutdown:', error);
            throw error;
        }
    }
}

module.exports = DatabaseService;