#!/usr/bin/env node

/**
 * Database Health Check for A2A Network
 * Monitors database integrity, performance, and consistency
 */

const cds = require('@sap/cds');

class DatabaseHealthChecker {
    constructor() {
        this.db = null;
        this.healthStatus = {
            overall: 'healthy',
            checks: [],
            timestamp: new Date().toISOString(),
            performance: {}
        };
    }

    async initialize() {
        try {
            const env = process.env.NODE_ENV || 'development';

            if (env === 'production') {
                this.db = await cds.connect.to('db');
                log.debug('✅ Connected to SAP HANA Cloud');
            } else {
                this.db = await cds.connect.to('db', {
                    kind: 'sqlite',
                    credentials: { url: './data/a2a-network.db' }
                });
                log.debug('✅ Connected to SQLite database');
            }
        } catch (error) {
            console.error('❌ Database connection failed:', error);
            this.addHealthCheck('connection', 'critical', false, error.message);
            throw error;
        }
    }

    addHealthCheck(component, severity, passed, details = null) {
        this.healthStatus.checks.push({
            component,
            severity,
            passed,
            details,
            timestamp: new Date().toISOString()
        });

        if (!passed && (severity === 'critical' || severity === 'high')) {
            this.healthStatus.overall = 'unhealthy';
        } else if (!passed && this.healthStatus.overall === 'healthy') {
            this.healthStatus.overall = 'warning';
        }
    }

    async checkTableIntegrity() {
        log.debug('🔍 Checking table integrity...');

        const requiredTables = [
            'Agents', 'Services', 'Capabilities', 'Messages', 'Workflows',
            'PerformanceMetrics', 'ReputationScores', 'NetworkStatistics'
        ];

        for (const table of requiredTables) {
            try {
                const startTime = Date.now();
                const result = await this.db.run(`SELECT COUNT(*) as count FROM ${table}`);
                const queryTime = Date.now() - startTime;

                this.addHealthCheck(
                    `table_${table.toLowerCase()}`,
                    'critical',
                    true,
                    `Table exists with ${result[0]?.count || 0} records (${queryTime}ms)`
                );

                this.healthStatus.performance[`${table}_query_time`] = queryTime;
            } catch (error) {
                this.addHealthCheck(
                    `table_${table.toLowerCase()}`,
                    'critical',
                    false,
                    `Table missing or inaccessible: ${error.message}`
                );
            }
        }
    }

    async checkDataConsistency() {
        log.debug('🔗 Checking data consistency...');

        try {
            // Check for agents without reputation scores
            const agentsWithoutReputation = await this.db.run(`
                SELECT COUNT(*) as count
                FROM Agents a
                LEFT JOIN ReputationScores r ON a.ID = r.agent_ID
                WHERE r.agent_ID IS NULL
            `);

            if (agentsWithoutReputation[0]?.count > 0) {
                this.addHealthCheck(
                    'agents_reputation_consistency',
                    'medium',
                    false,
                    `${agentsWithoutReputation[0].count} agents missing reputation scores`
                );
            } else {
                this.addHealthCheck(
                    'agents_reputation_consistency',
                    'medium',
                    true,
                    'All agents have reputation scores'
                );
            }

            // Check for services without valid agents
            const orphanedServices = await this.db.run(`
                SELECT COUNT(*) as count
                FROM Services s
                LEFT JOIN Agents a ON s.agent_ID = a.ID
                WHERE a.ID IS NULL
            `);

            if (orphanedServices[0]?.count > 0) {
                this.addHealthCheck(
                    'services_agent_consistency',
                    'high',
                    false,
                    `${orphanedServices[0].count} services reference non-existent agents`
                );
            } else {
                this.addHealthCheck(
                    'services_agent_consistency',
                    'high',
                    true,
                    'All services reference valid agents'
                );
            }

            // Check for capabilities without valid agents
            const orphanedCapabilities = await this.db.run(`
                SELECT COUNT(*) as count
                FROM Capabilities c
                LEFT JOIN Agents a ON c.agent_ID = a.ID
                WHERE a.ID IS NULL
            `);

            if (orphanedCapabilities[0]?.count > 0) {
                this.addHealthCheck(
                    'capabilities_agent_consistency',
                    'medium',
                    false,
                    `${orphanedCapabilities[0].count} capabilities reference non-existent agents`
                );
            } else {
                this.addHealthCheck(
                    'capabilities_agent_consistency',
                    'medium',
                    true,
                    'All capabilities reference valid agents'
                );
            }

        } catch (error) {
            this.addHealthCheck(
                'data_consistency',
                'high',
                false,
                `Consistency check failed: ${error.message}`
            );
        }
    }

    async checkPerformanceMetrics() {
        log.debug('⚡ Checking performance metrics...');

        try {
            // Test query performance on large tables
            const performanceTests = [
                {
                    name: 'agent_lookup',
                    query: 'SELECT * FROM Agents WHERE status = ? LIMIT 10',
                    params: ['active']
                },
                {
                    name: 'message_history',
                    query: 'SELECT * FROM Messages ORDER BY created_at DESC LIMIT 100',
                    params: []
                },
                {
                    name: 'reputation_ranking',
                    query: 'SELECT * FROM ReputationScores ORDER BY overall_score DESC LIMIT 20',
                    params: []
                }
            ];

            for (const test of performanceTests) {
                const startTime = Date.now();
                await this.db.run(test.query, test.params);
                const queryTime = Date.now() - startTime;

                this.healthStatus.performance[test.name] = queryTime;

                if (queryTime > 1000) {
                    this.addHealthCheck(
                        `performance_${test.name}`,
                        'medium',
                        false,
                        `Query took ${queryTime}ms (> 1000ms threshold)`
                    );
                } else {
                    this.addHealthCheck(
                        `performance_${test.name}`,
                        'low',
                        true,
                        `Query completed in ${queryTime}ms`
                    );
                }
            }

        } catch (error) {
            this.addHealthCheck(
                'performance_metrics',
                'medium',
                false,
                `Performance check failed: ${error.message}`
            );
        }
    }

    async checkIndexEffectiveness() {
        log.debug('📈 Checking index effectiveness...');

        // This is more relevant for production databases with EXPLAIN PLAN
        // For now, we'll do basic checks
        try {
            const indexChecks = [
                {
                    name: 'agents_status_index',
                    query: 'SELECT COUNT(*) FROM Agents WHERE status = ?',
                    params: ['active']
                },
                {
                    name: 'messages_timestamp_index',
                    query: 'SELECT COUNT(*) FROM Messages WHERE created_at > ?',
                    params: [new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()]
                }
            ];

            for (const check of indexChecks) {
                const startTime = Date.now();
                await this.db.run(check.query, check.params);
                const queryTime = Date.now() - startTime;

                this.addHealthCheck(
                    check.name,
                    'low',
                    queryTime < 500,
                    `Index query took ${queryTime}ms`
                );
            }

        } catch (error) {
            this.addHealthCheck(
                'index_effectiveness',
                'low',
                false,
                `Index check failed: ${error.message}`
            );
        }
    }

    async checkDiskSpace() {
        log.debug('💾 Checking disk space...');

        try {
            const fs = require('fs');
            const path = require('path');

            // Check available space (for SQLite)
            if (process.env.NODE_ENV !== 'production') {
                const dbPath = './data/a2a-network.db';
                if (fs.existsSync(dbPath)) {
                    const stats = fs.statSync(dbPath);
                    const sizeInMB = (stats.size / (1024 * 1024)).toFixed(2);

                    this.addHealthCheck(
                        'disk_usage',
                        'low',
                        true,
                        `Database file size: ${sizeInMB} MB`
                    );

                    this.healthStatus.performance.db_size_mb = parseFloat(sizeInMB);
                }
            }

        } catch (error) {
            this.addHealthCheck(
                'disk_space',
                'low',
                false,
                `Disk space check failed: ${error.message}`
            );
        }
    }

    async runAllChecks() {
        log.info('🏥 Starting database health check...');

        try {
            await this.checkTableIntegrity();
            await this.checkDataConsistency();
            await this.checkPerformanceMetrics();
            await this.checkIndexEffectiveness();
            await this.checkDiskSpace();

            log.debug('\n📊 Health Check Summary:');
            log.debug(`Overall Status: ${this.healthStatus.overall.toUpperCase()}`);

            const criticalFailures = this.healthStatus.checks.filter(c => c.severity === 'critical' && !c.passed);
            const highFailures = this.healthStatus.checks.filter(c => c.severity === 'high' && !c.passed);
            const mediumFailures = this.healthStatus.checks.filter(c => c.severity === 'medium' && !c.passed);

            if (criticalFailures.length > 0) {
                log.error(`❌ Critical Issues: ${criticalFailures.length}`);
                criticalFailures.forEach(check => {
                    log.debug(`   - ${check.component}: ${check.details}`);
                });
            }

            if (highFailures.length > 0) {
                log.debug(`⚠️  High Priority Issues: ${highFailures.length}`);
                highFailures.forEach(check => {
                    log.debug(`   - ${check.component}: ${check.details}`);
                });
            }

            if (mediumFailures.length > 0) {
                log.debug(`🔶 Medium Priority Issues: ${mediumFailures.length}`);
            }

            const passedChecks = this.healthStatus.checks.filter(c => c.passed).length;
            const totalChecks = this.healthStatus.checks.length;
            log.debug(`✅ Passed: ${passedChecks}/${totalChecks} checks`);

            // Performance summary
            log.debug('\n⚡ Performance Metrics:');
            Object.entries(this.healthStatus.performance).forEach(([metric, value]) => {
                log.debug(`   ${metric}: ${value}${metric.includes('time') ? 'ms' : metric.includes('size') ? 'MB' : ''}`);
            });

        } catch (error) {
            console.error('❌ Health check failed:', error);
            this.healthStatus.overall = 'unhealthy';
        }

        return this.healthStatus;
    }

    async disconnect() {
        if (this.db) {
            await this.db.disconnect();
            log.debug('🔌 Database disconnected');
        }
    }
}

// CLI Interface
async function main() {
    const healthChecker = new DatabaseHealthChecker();

    try {
        await healthChecker.initialize();
        const healthStatus = await healthChecker.runAllChecks();

        // Exit with appropriate code
        if (healthStatus.overall === 'unhealthy') {
            process.exit(1);
        } else if (healthStatus.overall === 'warning') {
            process.exit(2);
        } else {
            process.exit(0);
        }

    } catch (error) {
        console.error('💥 Health check failed:', error);
        process.exit(1);
    } finally {
        await healthChecker.disconnect();
    }
}

if (require.main === module) {
    main();
}

module.exports = DatabaseHealthChecker;