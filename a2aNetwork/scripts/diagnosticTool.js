#!/usr/bin/env node

/**
 * A2A Network - Enhanced CLI Diagnostic Tool with Glean Integration
 * Production-ready diagnostic tool with code intelligence capabilities
 */

const A2ADiagnosticTool = require('./diagnosticTool');
const GleanDiagnosticModule = require('../srv/glean/gleanDiagnosticModule');
const cds = require('@sap/cds');

class EnhancedA2ADiagnosticTool extends A2ADiagnosticTool {
    constructor() {
        super();
        this.gleanModule = new GleanDiagnosticModule(this);
        this.results.codeAnalysis = null;
        this.results.enhancedDiagnostics = {
            timestamp: new Date().toISOString(),
            gleanEnabled: false,
            codeIntelligence: {}
        };
    }

    async run() {
        this.log('🚀 Starting A2A Network Enhanced Diagnostic Tool with Glean', 'info');
        this.log('='.repeat(60), 'info');

        try {
            // Initialize Glean module
            await this.initializeGlean();

            // Run base diagnostics
            await this.runTest('Server Connectivity', () => this.testServerConnectivity());
            await this.runTest('API Endpoints', () => this.testAPIEndpoints());
            await this.runTest('File System Check', () => this.checkFileSystem());
            await this.runTest('Server Logs Capture', () => this.captureServerLogs());
            await this.runTest('SAP CAP Server Diagnostics', () => this.diagnoseSAPCAPServer());
            await this.runTest('Diagnostic Page Content', () => this.testDiagnosticPage());

            // Run enhanced diagnostics if Glean is available
            if (this.results.enhancedDiagnostics.gleanEnabled) {
                await this.runEnhancedDiagnostics();
            }

            // Generate comprehensive report
            this.generateEnhancedSummary();

        } catch (error) {
            this.log(`Critical error during diagnostics: ${error.message}`, 'error');
        }

        // Generate reports
        await this.generateReports();

        return this.results;
    }

    async initializeGlean() {
        try {
            await this.gleanModule.initialize();
            this.results.enhancedDiagnostics.gleanEnabled = this.gleanModule.isAvailable;

            if (this.gleanModule.isAvailable) {
                this.log('✅ Glean code intelligence service is available', 'success');
            } else {
                this.log('⚠️  Glean service not available - running basic diagnostics only', 'warning');
            }
        } catch (error) {
            this.log(`Failed to initialize Glean: ${error.message}`, 'error');
            this.results.enhancedDiagnostics.gleanEnabled = false;
        }
    }

    async runEnhancedDiagnostics() {
        this.log('🔍 Running enhanced code intelligence diagnostics...', 'info');

        try {
            // Run code analysis
            await this.runTest('Code Health Analysis', async () => {
                const codeAnalysis = await this.gleanModule.runCodeDiagnostics();
                this.results.enhancedDiagnostics.codeIntelligence = codeAnalysis;

                // Log summary
                if (codeAnalysis.codeHealth) {
                    this.log(`Code Health Score: ${codeAnalysis.codeHealth.score}/100`,
                             codeAnalysis.codeHealth.score > 80 ? 'success' : 'warning');
                }

                if (codeAnalysis.security) {
                    this.log(`Security Issues: ${codeAnalysis.security.totalIssues} found`,
                             codeAnalysis.security.totalIssues === 0 ? 'success' : 'warning');
                }

                return codeAnalysis;
            });

            // Analyze specific components
            await this.runTest('Critical Component Analysis', async () => {
                const criticalComponents = [
                    '/Users/apple/projects/a2a/a2aNetwork/srv/server.js',
                    '/Users/apple/projects/a2a/a2aNetwork/srv/sapBlockchainService.js',
                    '/Users/apple/projects/a2a/a2aNetwork/contracts/AgentServiceMarketplace.sol'
                ];

                const componentAnalysis = {};

                for (const component of criticalComponents) {
                    try {
                        const analysis = await this.gleanModule.gleanService.send({
                            event: 'analyzeComponents',
                            data: { componentPath: component }
                        });
                        componentAnalysis[component] = analysis;

                        this.log(`${component}: Complexity ${analysis.complexity.cyclomaticComplexity}`,
                                 analysis.complexity.cyclomaticComplexity < 20 ? 'success' : 'warning');
                    } catch (error) {
                        componentAnalysis[component] = { error: error.message };
                    }
                }

                this.results.enhancedDiagnostics.componentAnalysis = componentAnalysis;
                return componentAnalysis;
            });

            // Check for A2A specific patterns
            await this.runTest('A2A Architecture Validation', async () => {
                const validation = await this.validateA2AArchitecture();
                this.results.enhancedDiagnostics.architectureValidation = validation;
                return validation;
            });

        } catch (error) {
            this.log(`Enhanced diagnostics failed: ${error.message}`, 'error');
            this.results.enhancedDiagnostics.error = error.message;
        }
    }

    async validateA2AArchitecture() {
        const validation = {
            sapIntegration: {},
            blockchainIntegration: {},
            agentCommunication: {},
            securityPatterns: {}
        };

        // Validate SAP integration patterns
        validation.sapIntegration = await this.validateSAPPatterns();

        // Validate blockchain integration
        validation.blockchainIntegration = await this.validateBlockchainPatterns();

        // Validate agent communication
        validation.agentCommunication = await this.validateAgentPatterns();

        // Validate security implementation
        validation.securityPatterns = await this.validateSecurityImplementation();

        return validation;
    }

    async validateSAPPatterns() {
        const patterns = {
            cdsServices: false,
            xsuaaAuth: false,
            sapLogging: false,
            enterpriseMessaging: false
        };

        try {
            // Check for CDS service definitions
            const cdsQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'class.*extends.*cds\\.Service',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.cdsServices = cdsQuery.results && cdsQuery.results.length > 0;

            // Check for XSUAA authentication
            const authQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'xssec|xsuaa|@sap/xssec',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.xsuaaAuth = authQuery.results && authQuery.results.length > 0;

            // Check for SAP logging
            const loggingQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: '@sap/logging|cds\\.log',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.sapLogging = loggingQuery.results && loggingQuery.results.length > 0;

        } catch (error) {
            this.log(`SAP pattern validation error: ${error.message}`, 'error');
        }

        return patterns;
    }

    async validateBlockchainPatterns() {
        const patterns = {
            web3Integration: false,
            smartContracts: false,
            eventListeners: false,
            transactionHandling: false
        };

        try {
            // Check Web3 integration
            const web3Query = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'Web3|ethers|web3',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.web3Integration = web3Query.results && web3Query.results.length > 0;

            // Check smart contract interaction
            const contractQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'contract\\.methods|contract\\.call|sendTransaction',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.smartContracts = contractQuery.results && contractQuery.results.length > 0;

        } catch (error) {
            this.log(`Blockchain pattern validation error: ${error.message}`, 'error');
        }

        return patterns;
    }

    async validateAgentPatterns() {
        const patterns = {
            agentRegistration: false,
            messagePassing: false,
            reputationSystem: false,
            serviceDiscovery: false
        };

        try {
            // Check agent registration
            const regQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'registerAgent|agentRegistry',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.agentRegistration = regQuery.results && regQuery.results.length > 0;

            // Check message passing
            const msgQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'sendMessage|receiveMessage|messageQueue',
                    language: 'javascript',
                    limit: 10
                }
            });
            patterns.messagePassing = msgQuery.results && msgQuery.results.length > 0;

        } catch (error) {
            this.log(`Agent pattern validation error: ${error.message}`, 'error');
        }

        return patterns;
    }

    async validateSecurityImplementation() {
        const security = {
            authentication: false,
            authorization: false,
            encryption: false,
            inputValidation: false,
            auditLogging: false
        };

        try {
            // Check authentication
            const authQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'authenticate|verifyToken|passport',
                    language: 'javascript',
                    limit: 10
                }
            });
            security.authentication = authQuery.results && authQuery.results.length > 0;

            // Check input validation
            const validationQuery = await this.gleanModule.gleanService.send({
                event: 'queryCode',
                data: {
                    query: 'joi\\.validate|express-validator|sanitize',
                    language: 'javascript',
                    limit: 10
                }
            });
            security.inputValidation = validationQuery.results && validationQuery.results.length > 0;

        } catch (error) {
            this.log(`Security validation error: ${error.message}`, 'error');
        }

        return security;
    }

    generateEnhancedSummary() {
        // Call parent summary generation
        super.generateSummary();

        // Add enhanced diagnostics summary
        if (this.results.enhancedDiagnostics.gleanEnabled && this.results.enhancedDiagnostics.codeIntelligence) {
            const codeAnalysis = this.results.enhancedDiagnostics.codeIntelligence;

            this.results.summary.codeHealthScore = codeAnalysis.codeHealth?.score || 0;
            this.results.summary.securityIssues = codeAnalysis.security?.totalIssues || 0;
            this.results.summary.criticalSecurityIssues = codeAnalysis.security?.critical?.length || 0;
            this.results.summary.recommendations = codeAnalysis.recommendations || [];
        }

        // Update overall status based on enhanced findings
        if (this.results.summary.criticalSecurityIssues > 0) {
            this.results.summary.overallStatus = 'CRITICAL_ISSUES';
        }
    }

    async generateReports() {
        const timestamp = Date.now();

        // Save main diagnostic results
        const mainReportFile = `/Users/apple/projects/a2a/enhanced-diagnostic-results-${timestamp}.json`;
        await require('fs').promises.writeFile(
            mainReportFile,
            JSON.stringify(this.results, null, 2)
        );
        this.log(`📄 Enhanced diagnostic results saved to: ${mainReportFile}`, 'info');

        // Generate executive summary if Glean was used
        if (this.results.enhancedDiagnostics.gleanEnabled) {
            const summaryReport = await this.generateExecutiveSummary();
            const summaryFile = `/Users/apple/projects/a2a/diagnostic-summary-${timestamp}.md`;
            await require('fs').promises.writeFile(summaryFile, summaryReport);
            this.log(`📊 Executive summary saved to: ${summaryFile}`, 'info');
        }

        // Generate action items report
        if (this.results.summary.recommendations && this.results.summary.recommendations.length > 0) {
            const actionReport = this.generateActionItemsReport();
            const actionFile = `/Users/apple/projects/a2a/diagnostic-actions-${timestamp}.md`;
            await require('fs').promises.writeFile(actionFile, actionReport);
            this.log(`📋 Action items saved to: ${actionFile}`, 'info');
        }
    }

    async generateExecutiveSummary() {
        const report = await this.gleanModule.generateDetailedReport();

        return `# A2A Network Diagnostic Executive Summary

Generated: ${new Date().toISOString()}

## Overall Health Status: ${this.results.summary.overallStatus}

### Key Metrics
- **Code Health Score**: ${this.results.summary.codeHealthScore}/100
- **Test Success Rate**: ${this.results.summary.successRate}%
- **Total Errors**: ${this.results.summary.totalErrors}
- **Security Issues**: ${this.results.summary.securityIssues} (${this.results.summary.criticalSecurityIssues} critical)

### System Status
- **Server Connectivity**: ${this.results.networkTests['SAP CAP Server']?.status || 'UNKNOWN'}
- **API Health**: ${this.results.summary.passedTests}/${this.results.summary.totalTests} endpoints responding
- **Glean Service**: ${this.results.enhancedDiagnostics.gleanEnabled ? 'ACTIVE' : 'INACTIVE'}

### Code Analysis Findings
${this.formatCodeAnalysisFindings()}

### Architecture Validation
${this.formatArchitectureValidation()}

### Top Recommendations
${this.formatTopRecommendations()}

### Action Items
${report.actionItems ? report.actionItems.map(item =>
    `- **[${item.priority}]** ${item.task} (${item.category}) - ${item.deadline}`
).join('\n') : 'No critical action items'}

### Next Steps
1. Address critical security vulnerabilities immediately
2. Review and refactor high-complexity components
3. Update missing documentation
4. Schedule regular diagnostic runs

---
*Report generated by A2A Enhanced Diagnostic Tool with Glean Integration*
`;
    }

    formatCodeAnalysisFindings() {
        const analysis = this.results.enhancedDiagnostics.codeIntelligence;
        if (!analysis || !analysis.codeHealth) return 'No code analysis data available';

        const findings = [];

        if (analysis.codeHealth.topIssues) {
            findings.push('#### Top Code Issues');
            analysis.codeHealth.topIssues.slice(0, 5).forEach(issue => {
                findings.push(`- **${issue.type}**: ${issue.file} - ${issue.message}`);
            });
        }

        if (analysis.dependencies && analysis.dependencies.circularDependencies.length > 0) {
            findings.push('\n#### Circular Dependencies Detected');
            analysis.dependencies.circularDependencies.forEach(dep => {
                findings.push(`- ${dep.component}: ${dep.cycle}`);
            });
        }

        return findings.join('\n') || 'No significant code issues found';
    }

    formatArchitectureValidation() {
        const validation = this.results.enhancedDiagnostics.architectureValidation;
        if (!validation) return 'Architecture validation not performed';

        const results = [];

        results.push('| Component | Status |');
        results.push('|-----------|--------|');

        // SAP Integration
        results.push(`| SAP CDS Services | ${validation.sapIntegration?.cdsServices ? '✅' : '❌'} |`);
        results.push(`| XSUAA Authentication | ${validation.sapIntegration?.xsuaaAuth ? '✅' : '❌'} |`);
        results.push(`| SAP Logging | ${validation.sapIntegration?.sapLogging ? '✅' : '❌'} |`);

        // Blockchain
        results.push(`| Web3 Integration | ${validation.blockchainIntegration?.web3Integration ? '✅' : '❌'} |`);
        results.push(`| Smart Contracts | ${validation.blockchainIntegration?.smartContracts ? '✅' : '❌'} |`);

        // Security
        results.push(`| Authentication | ${validation.securityPatterns?.authentication ? '✅' : '❌'} |`);
        results.push(`| Input Validation | ${validation.securityPatterns?.inputValidation ? '✅' : '❌'} |`);

        return results.join('\n');
    }

    formatTopRecommendations() {
        const recommendations = this.results.summary.recommendations || [];
        if (recommendations.length === 0) return 'No specific recommendations';

        return recommendations.slice(0, 5).map(rec =>
            `- **${rec.priority.toUpperCase()}**: ${rec.action} (${rec.category}) - ${rec.impact}`
        ).join('\n');
    }

    generateActionItemsReport() {
        const items = this.gleanModule.generateActionItems();

        return `# A2A Network - Diagnostic Action Items

Generated: ${new Date().toISOString()}

## Priority Breakdown

### P0 - Critical (Immediate Action Required)
${items.filter(i => i.priority === 'P0').map(i =>
    `- [ ] ${i.task}\n  - Category: ${i.category}\n  - Assignee: ${i.assignee}\n  - Deadline: ${i.deadline}`
).join('\n\n') || 'No critical items'}

### P1 - High Priority (This Week)
${items.filter(i => i.priority === 'P1').map(i =>
    `- [ ] ${i.task}\n  - Category: ${i.category}\n  - Assignee: ${i.assignee}\n  - Deadline: ${i.deadline}`
).join('\n\n') || 'No high priority items'}

### P2 - Medium Priority (Next Sprint)
${items.filter(i => i.priority === 'P2').map(i =>
    `- [ ] ${i.task}\n  - Category: ${i.category}\n  - Assignee: ${i.assignee}\n  - Deadline: ${i.deadline}`
).join('\n\n') || 'No medium priority items'}

## Summary
- Total Action Items: ${items.length}
- Critical Items: ${items.filter(i => i.priority === 'P0').length}
- Estimated Effort: ${this.estimateEffort(items)}

---
*Track progress in your project management system*
`;
    }

    estimateEffort(items) {
        const effortMap = {
            'P0': 4,  // hours
            'P1': 8,  // hours
            'P2': 16  // hours
        };

        const totalHours = items.reduce((sum, item) =>
            sum + (effortMap[item.priority] || 8), 0
        );

        return `${totalHours} hours (${Math.ceil(totalHours / 8)} days)`;
    }
}

// Run the enhanced diagnostic tool if called directly
if (require.main === module) {
    const tool = new EnhancedA2ADiagnosticTool();
    tool.run().then((results) => {
        process.exit(results.summary.overallStatus === 'HEALTHY' ? 0 : 1);
    }).catch((error) => {
        console.error('❌ Enhanced diagnostic tool failed:', error.message);
        process.exit(1);
    });
}

module.exports = EnhancedA2ADiagnosticTool;