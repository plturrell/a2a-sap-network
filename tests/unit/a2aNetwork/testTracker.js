/**
 * SQLite Test Execution Tracker
 * Tracks test executions, results, and performance metrics
 */

const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

class TestTracker {
    constructor(dbPath = null) {
        this.dbPath = dbPath || path.join(__dirname, '../data/test_tracking.db');
        this.db = null;
        this.currentRun = null;
    }

    /**
     * Initialize database and create tables
     */
    async initialize() {
        // Ensure data directory exists
        const dataDir = path.dirname(this.dbPath);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }

        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database(this.dbPath, (err) => {
                if (err) {
                    reject(err);
                    return;
                }
                console.log('📊 Test tracking database connected');
                this.createTables().then(resolve).catch(reject);
            });
        });
    }

    /**
     * Create database tables
     */
    async createTables() {
        const tables = {
            // Test runs table
            test_runs: `
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    environment TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    status TEXT NOT NULL DEFAULT 'running',
                    total_suites INTEGER DEFAULT 0,
                    passed_suites INTEGER DEFAULT 0,
                    failed_suites INTEGER DEFAULT 0,
                    total_tests INTEGER DEFAULT 0,
                    passed_tests INTEGER DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    coverage_percentage REAL DEFAULT 0,
                    node_version TEXT,
                    git_commit TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            `,

            // Test suites table
            test_suites: `
                CREATE TABLE IF NOT EXISTS test_suites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    suite_type TEXT NOT NULL,
                    description TEXT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    status TEXT NOT NULL DEFAULT 'running',
                    total_tests INTEGER DEFAULT 0,
                    passed_tests INTEGER DEFAULT 0,
                    failed_tests INTEGER DEFAULT 0,
                    duration_ms INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
                )
            `,

            // Individual tests table
            test_cases: `
                CREATE TABLE IF NOT EXISTS test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    test_id TEXT NOT NULL,
                    component TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    status TEXT NOT NULL DEFAULT 'running',
                    duration_ms INTEGER DEFAULT 0,
                    error_message TEXT,
                    stack_trace TEXT,
                    assertions_total INTEGER DEFAULT 0,
                    assertions_passed INTEGER DEFAULT 0,
                    performance_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
                )
            `,

            // Test metrics table
            test_metrics: `
                CREATE TABLE IF NOT EXISTS test_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    component TEXT,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
                )
            `,

            // Test environments table
            test_environments: `
                CREATE TABLE IF NOT EXISTS test_environments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    environment_name TEXT NOT NULL,
                    node_version TEXT,
                    os_platform TEXT,
                    os_version TEXT,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    disk_space_mb REAL,
                    configuration TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
                )
            `,

            // Test coverage table
            test_coverage: `
                CREATE TABLE IF NOT EXISTS test_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    total_lines INTEGER NOT NULL,
                    covered_lines INTEGER NOT NULL,
                    total_functions INTEGER NOT NULL,
                    covered_functions INTEGER NOT NULL,
                    total_branches INTEGER NOT NULL,
                    covered_branches INTEGER NOT NULL,
                    coverage_percentage REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES test_runs (run_id)
                )
            `
        };

        const promises = Object.entries(tables).map(([tableName, sql]) => {
            return new Promise((resolve, reject) => {
                this.db.run(sql, (err) => {
                    if (err) {
                        console.error(`Error creating table ${tableName}:`, err);
                        reject(err);
                    } else {
                        console.log(`✅ Table ${tableName} ready`);
                        resolve();
                    }
                });
            });
        });

        await Promise.all(promises);
        console.log('📊 All test tracking tables created');
    }

    /**
     * Start new test run
     */
    async startTestRun(options = {}) {
        const runId = options.runId || `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const environment = options.environment || 'test';
        
        this.currentRun = {
            runId,
            environment,
            startTime: new Date(),
            status: 'running',
            suites: [],
            totalTests: 0,
            passedTests: 0,
            failedTests: 0
        };

        return new Promise((resolve, reject) => {
            const sql = `
                INSERT INTO test_runs (
                    run_id, environment, start_time, status, node_version, git_commit
                ) VALUES (?, ?, ?, ?, ?, ?)
            `;

            this.db.run(sql, [
                runId,
                environment,
                this.currentRun.startTime.toISOString(),
                'running',
                process.version,
                options.gitCommit || null
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`🚀 Test run started: ${runId}`);
                    resolve(runId);
                }
            });
        });
    }

    /**
     * Start test suite
     */
    async startTestSuite(suiteName, suiteType = 'integration', description = '') {
        if (!this.currentRun) {
            throw new Error('No active test run. Call startTestRun() first.');
        }

        const suite = {
            name: suiteName,
            type: suiteType,
            description,
            startTime: new Date(),
            status: 'running',
            tests: []
        };

        this.currentRun.suites.push(suite);

        return new Promise((resolve, reject) => {
            const sql = `
                INSERT INTO test_suites (
                    run_id, suite_name, suite_type, description, start_time, status
                ) VALUES (?, ?, ?, ?, ?, ?)
            `;

            this.db.run(sql, [
                this.currentRun.runId,
                suiteName,
                suiteType,
                description,
                suite.startTime.toISOString(),
                'running'
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`📋 Test suite started: ${suiteName}`);
                    resolve(suite);
                }
            });
        });
    }

    /**
     * Record test case result
     */
    async recordTestCase(testData) {
        if (!this.currentRun) {
            throw new Error('No active test run');
        }

        const {
            suiteName,
            testName,
            testId,
            component,
            status,
            duration,
            error = null,
            stackTrace = null,
            assertions = { total: 0, passed: 0 },
            performanceData = null
        } = testData;

        return new Promise((resolve, reject) => {
            const sql = `
                INSERT INTO test_cases (
                    run_id, suite_name, test_name, test_id, component,
                    start_time, end_time, status, duration_ms,
                    error_message, stack_trace, assertions_total, assertions_passed,
                    performance_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `;

            const now = new Date();
            const startTime = new Date(now.getTime() - duration);

            this.db.run(sql, [
                this.currentRun.runId,
                suiteName,
                testName,
                testId,
                component,
                startTime.toISOString(),
                now.toISOString(),
                status,
                duration,
                error,
                stackTrace,
                assertions.total,
                assertions.passed,
                JSON.stringify(performanceData)
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    if (status === 'passed') {
                        this.currentRun.passedTests++;
                    } else {
                        this.currentRun.failedTests++;
                    }
                    this.currentRun.totalTests++;
                    resolve();
                }
            }.bind(this));
        });
    }

    /**
     * Record performance metric
     */
    async recordMetric(metricName, value, unit = null, component = null) {
        if (!this.currentRun) {
            throw new Error('No active test run');
        }

        return new Promise((resolve, reject) => {
            const sql = `
                INSERT INTO test_metrics (
                    run_id, metric_name, metric_value, metric_unit, component, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            `;

            this.db.run(sql, [
                this.currentRun.runId,
                metricName,
                value,
                unit,
                component,
                new Date().toISOString()
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }

    /**
     * End test suite
     */
    async endTestSuite(suiteName, status, error = null) {
        if (!this.currentRun) {
            throw new Error('No active test run');
        }

        return new Promise((resolve, reject) => {
            const sql = `
                UPDATE test_suites 
                SET end_time = ?, status = ?, error_message = ?,
                    total_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ? AND suite_name = ?),
                    passed_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ? AND suite_name = ? AND status = 'passed'),
                    failed_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ? AND suite_name = ? AND status = 'failed'),
                    duration_ms = (strftime('%s', ?) - strftime('%s', start_time)) * 1000
                WHERE run_id = ? AND suite_name = ?
            `;

            const endTime = new Date().toISOString();

            this.db.run(sql, [
                endTime, status, error,
                this.currentRun.runId, suiteName,
                this.currentRun.runId, suiteName,
                this.currentRun.runId, suiteName,
                endTime,
                this.currentRun.runId, suiteName
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`✅ Test suite ended: ${suiteName} (${status})`);
                    resolve();
                }
            });
        });
    }

    /**
     * End test run
     */
    async endTestRun(status = 'completed', coverage = null) {
        if (!this.currentRun) {
            throw new Error('No active test run');
        }

        return new Promise((resolve, reject) => {
            const sql = `
                UPDATE test_runs 
                SET end_time = ?, status = ?, 
                    total_suites = (SELECT COUNT(*) FROM test_suites WHERE run_id = ?),
                    passed_suites = (SELECT COUNT(*) FROM test_suites WHERE run_id = ? AND status = 'passed'),
                    failed_suites = (SELECT COUNT(*) FROM test_suites WHERE run_id = ? AND status = 'failed'),
                    total_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ?),
                    passed_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ? AND status = 'passed'),
                    failed_tests = (SELECT COUNT(*) FROM test_cases WHERE run_id = ? AND status = 'failed'),
                    duration_ms = (strftime('%s', ?) - strftime('%s', start_time)) * 1000,
                    coverage_percentage = ?
                WHERE run_id = ?
            `;

            const endTime = new Date().toISOString();

            this.db.run(sql, [
                endTime, status,
                this.currentRun.runId,
                this.currentRun.runId,
                this.currentRun.runId,
                this.currentRun.runId,
                this.currentRun.runId,
                this.currentRun.runId,
                endTime,
                coverage,
                this.currentRun.runId
            ], function(err) {
                if (err) {
                    reject(err);
                } else {
                    console.log(`🏁 Test run completed: ${this.currentRun.runId} (${status})`);
                    this.currentRun = null;
                    resolve();
                }
            }.bind(this));
        });
    }

    /**
     * Get test run summary
     */
    async getTestRunSummary(runId = null) {
        const targetRunId = runId || (this.currentRun && this.currentRun.runId);
        if (!targetRunId) {
            throw new Error('No run ID specified and no active run');
        }

        return new Promise((resolve, reject) => {
            const sql = `
                SELECT * FROM test_runs WHERE run_id = ?
            `;

            this.db.get(sql, [targetRunId], (err, row) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(row);
                }
            });
        });
    }

    /**
     * Get recent test runs
     */
    async getRecentTestRuns(limit = 10) {
        return new Promise((resolve, reject) => {
            const sql = `
                SELECT * FROM test_runs 
                ORDER BY start_time DESC 
                LIMIT ?
            `;

            this.db.all(sql, [limit], (err, rows) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(rows);
                }
            });
        });
    }

    /**
     * Get test statistics
     */
    async getTestStatistics(days = 7) {
        return new Promise((resolve, reject) => {
            const sql = `
                SELECT 
                    COUNT(*) as total_runs,
                    AVG(duration_ms) as avg_duration,
                    AVG(coverage_percentage) as avg_coverage,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                    SUM(total_tests) as total_tests_executed,
                    SUM(passed_tests) as total_tests_passed
                FROM test_runs 
                WHERE start_time >= datetime('now', '-${days} days')
            `;

            this.db.get(sql, (err, row) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(row);
                }
            });
        });
    }

    /**
     * Close database connection
     */
    async close() {
        if (this.db) {
            return new Promise((resolve) => {
                this.db.close((err) => {
                    if (err) {
                        console.error('Error closing database:', err);
                    } else {
                        console.log('📊 Test tracking database closed');
                    }
                    resolve();
                });
            });
        }
    }
}

module.exports = TestTracker;