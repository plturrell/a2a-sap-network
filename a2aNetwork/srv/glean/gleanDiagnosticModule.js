/**
 * @fileoverview Glean Integration Module for A2A Diagnostic Tool
 * @module gleanDiagnosticModule
 * @since 1.0.0
 * 
 * Extends the existing diagnostic tool with Glean code intelligence capabilities
 */

const cds = require('@sap/cds');
const fetch = require('node-fetch');

class GleanDiagnosticModule {
    constructor(diagnosticTool) {
        this.diagnosticTool = diagnosticTool;
        this.gleanService = null;
        this.gleanUrl = process.env.GLEAN_URL || 'http://localhost:8080';
        this.isAvailable = false;
        this.log = cds.log('glean-diagnostic');
    }

    async initialize() {
        try {
            // Check if Glean service is available
            const health = await blockchainClient.sendMessage(`${this.gleanUrl}/health`);
            this.isAvailable = health.ok;
            
            if (this.isAvailable) {
                this.log.info('Glean service is available for enhanced diagnostics');
                // Connect to Glean service via CDS
                this.gleanService = await cds.connect.to('GleanService');
            } else {
                this.log.warn('Glean service not available, code analysis features disabled');
            }
        } catch (error) {
            this.log.error('Failed to initialize Glean module:', error);
            this.isAvailable = false;
        }
    }

    async runCodeDiagnostics() {
        if (!this.isAvailable) {
            return {
                status: 'SKIPPED',
                reason: 'Glean service not available'
            };
        }

        const results = {
            timestamp: new Date().toISOString(),
            codeHealth: {},
            dependencies: {},
            security: {},
            performance: {},
            recommendations: []
        };

        try {
            // Run all code diagnostics in parallel
            const [health, deps, security, perf] = await Promise.all([
                this.analyzeCodeHealth(),
                this.analyzeDependencies(),
                this.analyzeSecurityIssues(),
                this.analyzePerformance()
            ]);

            results.codeHealth = health;
            results.dependencies = deps;
            results.security = security;
            results.performance = perf;
            results.recommendations = this.generateRecommendations(results);

            // Add to main diagnostic results
            this.diagnosticTool.results.codeAnalysis = results;
            
            return results;
        } catch (error) {
            this.log.error('Code diagnostics failed:', error);
            return {
                status: 'FAILED',
                error: error.message
            };
        }
    }

    async analyzeCodeHealth() {
        const paths = [
            './srv',
            './app/a2aFiori',
            './contracts',
            './pythonSdk'
        ];

        const health = await this.gleanService.send({
            event: 'getCodeHealth',
            data: { paths }
        });

        // Calculate health score
        const score = this.calculateHealthScore(health);
        
        return {
            score,
            metrics: health,
            status: score > 80 ? 'HEALTHY' : score > 60 ? 'NEEDS_ATTENTION' : 'UNHEALTHY',
            topIssues: this.extractTopIssues(health)
        };
    }

    calculateHealthScore(health) {
        let score = 100;
        
        // Deduct points for various issues
        if (health.averageComplexity > 10) score -= 10;
        if (health.averageComplexity > 20) score -= 10;
        
        // Deduct for code smells
        score -= Math.min(health.codeSmells.length * 2, 20);
        
        // Deduct for technical debt
        score -= Math.min(health.technicalDebt.length * 3, 30);
        
        // Bonus for good documentation
        if (health.documentationCoverage > 80) score += 5;
        
        return Math.max(0, Math.min(100, score));
    }

    extractTopIssues(health) {
        const issues = [];
        
        // Add high complexity files
        health.codeSmells
            .filter(smell => smell.issue === 'High complexity')
            .slice(0, 5)
            .forEach(smell => {
                issues.push({
                    type: 'complexity',
                    severity: 'medium',
                    file: smell.file,
                    message: `Complexity: ${smell.complexity} (threshold: 10)`
                });
            });
        
        // Add documentation issues
        health.technicalDebt
            .filter(debt => debt.issue === 'Low documentation coverage')
            .slice(0, 5)
            .forEach(debt => {
                issues.push({
                    type: 'documentation',
                    severity: 'low',
                    file: debt.file,
                    message: `Documentation coverage: ${debt.coverage}%`
                });
            });
        
        return issues;
    }

    async analyzeDependencies() {
        const criticalPaths = [
            './srv/server.js',
            './srv/sapA2aService.js',
            './srv/sapBlockchainService.js',
            './app/a2aFiori/webapp/Component.js'
        ];

        const dependencies = {};
        
        for (const path of criticalPaths) {
            try {
                const deps = await this.gleanService.send({
                    event: 'findDependencies',
                    data: { sourcePath: path, depth: 3 }
                });
                
                dependencies[path] = {
                    direct: deps.dependencies.filter(d => d.depth === 1).length,
                    transitive: deps.dependencies.length,
                    graph: deps.graph
                };
            } catch (error) {
                dependencies[path] = {
                    error: error.message
                };
            }
        }

        // Check for circular dependencies
        const circular = await this.detectCircularDependencies(dependencies);
        
        return {
            components: dependencies,
            circularDependencies: circular,
            totalDependencies: Object.values(dependencies).reduce((sum, d) => sum + (d.transitive || 0), 0)
        };
    }

    async detectCircularDependencies(dependencies) {
        const circular = [];
        
        for (const [path, data] of Object.entries(dependencies)) {
            if (data.graph && data.graph.edges) {
                const cycles = this.findCycles(data.graph);
                cycles.forEach(cycle => {
                    circular.push({
                        component: path,
                        cycle: cycle.join(' -> ')
                    });
                });
            }
        }
        
        return circular;
    }

    findCycles(graph) {
        const cycles = [];
        const visited = new Set();
        const recursionStack = new Set();
        
        const dfs = (node, path = []) => {
            if (recursionStack.has(node)) {
                const cycleStart = path.indexOf(node);
                if (cycleStart !== -1) {
                    cycles.push(path.slice(cycleStart));
                }
                return;
            }
            
            if (visited.has(node)) return;
            
            visited.add(node);
            recursionStack.add(node);
            path.push(node);
            
            const edges = graph.edges.filter(e => e.from === node);
            for (const edge of edges) {
                dfs(edge.to, [...path]);
            }
            
            recursionStack.delete(node);
        };
        
        for (const node of graph.nodes) {
            if (!visited.has(node)) {
                dfs(node);
            }
        }
        
        return cycles;
    }

    async analyzeSecurityIssues() {
        const securityScan = await this.gleanService.send({
            event: 'detectSecurityIssues',
            data: {
                paths: ['./srv', './app', './contracts'],
                severity: 'all'
            }
        });

        // Categorize by severity
        const categorized = {
            critical: [],
            high: [],
            medium: [],
            low: []
        };

        securityScan.issues.forEach(issue => {
            categorized[issue.severity].push(issue);
        });

        // Check for specific A2A security concerns
        const a2aSpecificIssues = await this.checkA2ASecurityPatterns();
        
        return {
            summary: securityScan.bySeverity,
            critical: categorized.critical,
            high: categorized.high,
            medium: categorized.medium,
            low: categorized.low,
            a2aSpecific: a2aSpecificIssues,
            totalIssues: securityScan.totalIssues
        };
    }

    async checkA2ASecurityPatterns() {
        const issues = [];
        
        // Check blockchain key management
        const keyManagement = await this.gleanService.send({
            event: 'queryCode',
            data: {
                query: 'privateKey|private_key|secretKey|secret_key',
                language: 'javascript',
                limit: 10
            }
        });
        
        if (keyManagement.results && keyManagement.results.length > 0) {
            issues.push({
                type: 'key_exposure',
                severity: 'critical',
                message: 'Potential private key exposure detected',
                locations: keyManagement.results
            });
        }
        
        // Check agent authentication
        const authPatterns = await this.gleanService.send({
            event: 'queryCode',
            data: {
                query: 'verifyAgent|authenticateAgent|agentAuth',
                language: 'javascript',
                limit: 10
            }
        });
        
        if (!authPatterns.results || authPatterns.results.length === 0) {
            issues.push({
                type: 'missing_auth',
                severity: 'high',
                message: 'Agent authentication mechanisms may be missing'
            });
        }
        
        return issues;
    }

    async analyzePerformance() {
        const perfAnalysis = await this.gleanService.send({
            event: 'analyzePerformance',
            data: {
                paths: ['./srv', './app']
            }
        });

        // Analyze database query patterns
        const dbPatterns = await this.analyzeDatabasePatterns();
        
        return {
            slowQueries: perfAnalysis.slowQueries,
            memoryLeaks: perfAnalysis.memoryLeaks,
            inefficientLoops: perfAnalysis.inefficientLoops,
            blockingOperations: perfAnalysis.blockingOperations,
            databasePatterns: dbPatterns,
            recommendations: this.generatePerformanceRecommendations(perfAnalysis)
        };
    }

    async analyzeDatabasePatterns() {
        // Look for N+1 query patterns
        const n1Patterns = await this.gleanService.send({
            event: 'queryCode',
            data: {
                query: 'forEach.*await.*SELECT|map.*await.*SELECT',
                language: 'javascript',
                limit: 20
            }
        });

        // Look for missing indexes
        const queryPatterns = await this.gleanService.send({
            event: 'queryCode',
            data: {
                query: 'SELECT.*WHERE|cds.run.*SELECT',
                language: 'javascript',
                limit: 50
            }
        });

        return {
            potentialN1Queries: n1Patterns.results ? n1Patterns.results.length : 0,
            totalQueries: queryPatterns.results ? queryPatterns.results.length : 0,
            recommendations: []
        };
    }

    generatePerformanceRecommendations(analysis) {
        const recommendations = [];
        
        if (analysis.slowQueries.length > 0) {
            recommendations.push({
                type: 'database',
                priority: 'high',
                message: 'Optimize slow database queries using batch operations or joins'
            });
        }
        
        if (analysis.memoryLeaks.length > 0) {
            recommendations.push({
                type: 'memory',
                priority: 'medium',
                message: 'Add cleanup handlers for timers and event listeners'
            });
        }
        
        if (analysis.blockingOperations.length > 0) {
            recommendations.push({
                type: 'async',
                priority: 'high',
                message: 'Convert blocking operations to async/await patterns'
            });
        }
        
        return recommendations;
    }

    generateRecommendations(results) {
        const recommendations = [];
        
        // Code health recommendations
        if (results.codeHealth.score < 70) {
            recommendations.push({
                category: 'code_quality',
                priority: 'high',
                action: 'Refactor high-complexity functions',
                impact: 'Improve maintainability and reduce bugs'
            });
        }
        
        // Security recommendations
        if (results.security.critical.length > 0) {
            recommendations.push({
                category: 'security',
                priority: 'critical',
                action: 'Address critical security vulnerabilities immediately',
                impact: 'Prevent potential security breaches'
            });
        }
        
        // Dependency recommendations
        if (results.dependencies.circularDependencies.length > 0) {
            recommendations.push({
                category: 'architecture',
                priority: 'medium',
                action: 'Resolve circular dependencies',
                impact: 'Improve build times and code organization'
            });
        }
        
        // Performance recommendations
        if (results.performance.slowQueries.length > 5) {
            recommendations.push({
                category: 'performance',
                priority: 'high',
                action: 'Optimize database queries',
                impact: 'Reduce response times and server load'
            });
        }
        
        return recommendations.sort((a, b) => {
            const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
            return priorityOrder[a.priority] - priorityOrder[b.priority];
        });
    }

    async generateDetailedReport() {
        const report = {
            executionTime: new Date().toISOString(),
            gleanVersion: await this.getGleanVersion(),
            indexStatus: await this.getIndexStatus(),
            analysis: this.diagnosticTool.results.codeAnalysis,
            actionItems: this.generateActionItems()
        };

        return report;
    }

    async getGleanVersion() {
        try {
            const response = await blockchainClient.sendMessage(`${this.gleanUrl}/api/v1/version`);
            if (response.ok) {
                const data = await response.json();
                return data.version;
            }
        } catch (error) {
            this.log.error('Failed to get Glean version:', error);
        }
        return 'unknown';
    }

    async getIndexStatus() {
        try {
            const response = await blockchainClient.sendMessage(`${this.gleanUrl}/api/v1/index/status`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            this.log.error('Failed to get index status:', error);
        }
        return { status: 'unknown' };
    }

    generateActionItems() {
        const actionItems = [];
        const analysis = this.diagnosticTool.results.codeAnalysis;
        
        if (!analysis) return actionItems;
        
        // Critical security issues
        if (analysis.security && analysis.security.critical.length > 0) {
            analysis.security.critical.forEach(issue => {
                actionItems.push({
                    priority: 'P0',
                    category: 'Security',
                    task: `Fix ${issue.type} in ${issue.file}`,
                    assignee: 'Security Team',
                    deadline: 'Immediate'
                });
            });
        }
        
        // High complexity code
        if (analysis.codeHealth && analysis.codeHealth.topIssues) {
            analysis.codeHealth.topIssues
                .filter(issue => issue.type === 'complexity')
                .slice(0, 3)
                .forEach(issue => {
                    actionItems.push({
                        priority: 'P2',
                        category: 'Code Quality',
                        task: `Refactor complex code in ${issue.file}`,
                        assignee: 'Development Team',
                        deadline: 'Next Sprint'
                    });
                });
        }
        
        return actionItems;
    }
}

module.exports = GleanDiagnosticModule;