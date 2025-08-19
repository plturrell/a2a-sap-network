/**
 * WebSocket Connection Monitor
 * Tracks real WebSocket connections and updates metrics in the database
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const cds = require('@sap/cds');

class WebSocketMonitor {
    constructor() {
        this.connections = new Map();
        this.dbPath = path.join(__dirname, '../a2aNetwork.db');
        this.db = null;
        this.initDatabase();
    }

    initDatabase() {
        this.log = cds.log('websocket-monitor');
        this.db = new sqlite3.Database(this.dbPath, (err) => {
            if (err) {
                this.log.error('Failed to connect to database:', err);
            } else {
                this.log.info('Connected to database');
            }
        });
    }

    // Track a new WebSocket connection
    addConnection(socketId, clientInfo = {}) {
        this.connections.set(socketId, {
            id: socketId,
            connectedAt: new Date(),
            clientInfo: clientInfo,
            messageCount: 0,
            lastActivity: new Date()
        });
        
        this.updateDatabaseMetrics();
        // console.log(`🔌 WebSocket connected: ${socketId} (Total: ${this.connections.size})`);
    }

    // Remove a disconnected WebSocket
    removeConnection(socketId) {
        if (this.connections.has(socketId)) {
            this.connections.delete(socketId);
            this.updateDatabaseMetrics();
            // console.log(`🔌 WebSocket disconnected: ${socketId} (Total: ${this.connections.size})`);
        }
    }

    // Update activity for a connection
    updateActivity(socketId) {
        const connection = this.connections.get(socketId);
        if (connection) {
            connection.lastActivity = new Date();
            connection.messageCount++;
        }
    }

    // Get current connection count
    getConnectionCount() {
        return this.connections.size;
    }

    // Get active connections (activity within last 5 minutes)
    getActiveConnections() {
        const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
        let activeCount = 0;
        
        for (const [, connection] of this.connections) {
            if (connection.lastActivity > fiveMinutesAgo) {
                activeCount++;
            }
        }
        
        return activeCount;
    }

    // Update WebSocket metrics in the database
    updateDatabaseMetrics() {
        const connectionCount = this.connections.size;
        const activeCount = this.getActiveConnections();
        
        if (!this.db) return;

        // Update the WebSocket connections metric
        const updateSQL = `
            UPDATE a2a_network_NetworkHealthMetrics 
            SET metricValue = ?, 
                lastCheckTime = ?,
                modifiedAt = ?,
                status = CASE 
                    WHEN ? >= thresholdCritical THEN 'critical'
                    WHEN ? >= thresholdWarning THEN 'warning'
                    ELSE 'healthy'
                END
            WHERE metricName = 'WebSocket Connections'
        `;

        const now = new Date().toISOString();
        
        this.db.run(updateSQL, [
            connectionCount,
            now,
            now,
            connectionCount,
            connectionCount
        ], (err) => {
            if (err) {
                this.log.error('Failed to update WebSocket metrics:', err);
            }
        });

        // Also update a separate active connections metric if it exists
        const updateActiveSQL = `
            INSERT OR REPLACE INTO a2a_network_NetworkHealthMetrics (
                ID,
                metricName,
                metricValue,
                unit,
                status,
                thresholdWarning,
                thresholdCritical,
                lastCheckTime,
                createdAt,
                modifiedAt
            ) VALUES (
                (SELECT ID FROM a2a_network_NetworkHealthMetrics WHERE metricName = 'Active WebSocket Connections'),
                'Active WebSocket Connections',
                ?,
                'connections',
                'healthy',
                800,
                950,
                ?,
                COALESCE((SELECT createdAt FROM a2a_network_NetworkHealthMetrics WHERE metricName = 'Active WebSocket Connections'), ?),
                ?
            )
        `;

        this.db.run(updateActiveSQL, [activeCount, now, now, now], (err) => {
            if (err && !err.message.includes('NOT NULL constraint failed')) {
                this.log.error('Failed to update active WebSocket metrics:', err);
            }
        });
    }

    // Get connection statistics
    getStatistics() {
        const stats = {
            totalConnections: this.connections.size,
            activeConnections: this.getActiveConnections(),
            connectionDetails: []
        };

        // Add connection details
        for (const [id, connection] of this.connections) {
            stats.connectionDetails.push({
                id: id,
                connectedAt: connection.connectedAt,
                messageCount: connection.messageCount,
                lastActivity: connection.lastActivity,
                duration: Date.now() - connection.connectedAt.getTime()
            });
        }

        return stats;
    }

    // Clean up stale connections (no activity for 10 minutes)
    cleanupStaleConnections() {
        const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000);
        const staleConnections = [];

        for (const [id, connection] of this.connections) {
            if (connection.lastActivity < tenMinutesAgo) {
                staleConnections.push(id);
            }
        }

        staleConnections.forEach(id => {
            // console.log(`🧹 Cleaning up stale WebSocket connection: ${id}`);
            this.removeConnection(id);
        });

        return staleConnections.length;
    }

    // Close database connection
    close() {
        if (this.db) {
            this.db.close((err) => {
                if (err) {
                    this.log.error('Error closing WebSocket monitor database:', err);
                } else {
                    // console.log('✅ WebSocket monitor database connection closed');
                }
            });
        }
    }
}

// Export singleton instance
module.exports = new WebSocketMonitor();