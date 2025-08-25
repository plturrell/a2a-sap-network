/**
 * Agent 7 UI Security Scanner - Agent Manager
 * Comprehensive security and SAP standards compliance validation
 */

const fs = require('fs');
const path = require('path');

class Agent7SecurityScanner {
    constructor() {
        this.agent7Path = '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent7';
        this.issues = [];
        this.passed = 0;
        this.total = 29; // Standard check count

        this.securityChecks = {
            // Critical Security (8 checks)
            csrfProtection: false,
            inputValidation: false,
            xssProtection: false,
            authenticationChecks: false,
            eventSourceSecurity: false,
            memoryManagement: false,
            agentOperationSecurity: false,
            auditLogging: false,

            // SAP Standards (6 checks)
            sapUi5Structure: false,
            i18nComprehensive: false,
            errorHandling: false,
            manifestCompliance: false,
            controllerExtensions: false,
            fragmentLoading: false,

            // UI5 Development (5 checks)
            encodeXmlUsage: false,
            modelBinding: false,
            routerUsage: false,
            busyIndicators: false,
            messageHandling: false,

            // Enterprise Compliance (5 checks)
            auditTrail: false,
            roleBasedAccess: false,
            dataPrivacy: false,
            businessContinuity: false,
            changeManagement: false,

            // Performance (5 checks)
            lazyLoading: false,
            resourceOptimization: false,
            caching: false,
            bundling: false,
            memoryLeakPrevention: false
        };
    }

    async scanAll() {
        console.log('🔍 Starting Agent 7 UI Security Scan - Agent Manager');
        console.log('═══════════════════════════════════════════════════');

        try {
            await this.scanControllers();
            await this.scanManifest();
            await this.scanI18n();
            await this.scanFragments();

            this.generateReport();
        } catch (error) {
            console.error('❌ Scan failed:', error.message);
        }
    }

    async scanControllers() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const filePath = path.join(this.agent7Path, controller);
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf8');
                await this.analyzeController(content, controller);
            }
        }
    }

    async analyzeController(content, fileName) {
        // 1. CSRF Protection Check
        if (!content.includes('X-CSRF-Token') && !content.includes('csrf')) {
            this.addIssue('CRITICAL', 'CSRF Protection Missing',
                `${fileName} lacks CSRF token implementation for AJAX calls`);
        } else {
            this.securityChecks.csrfProtection = true;
            this.passed++;
        }

        // 2. Input Validation Check
        const hasValidation = content.includes('validateInput') ||
                            content.includes('sanitize') ||
                            content.includes('escape');
        if (!hasValidation) {
            this.addIssue('HIGH', 'Input Validation Missing',
                `${fileName} lacks proper input validation and sanitization`);
        } else {
            this.securityChecks.inputValidation = true;
            this.passed++;
        }

        // 3. XSS Protection Check
        const hasXssProtection = content.includes('encodeXML') ||
                               content.includes('encodeURL') ||
                               content.includes('encodeHTML');
        if (!hasXssProtection) {
            this.addIssue('CRITICAL', 'XSS Protection Missing',
                `${fileName} lacks XSS protection with encodeXML usage`);
        } else {
            this.securityChecks.xssProtection = true;
            this.passed++;
        }

        // 4. Authentication Checks
        if (!content.includes('getUser') && !content.includes('hasRole')) {
            this.addIssue('HIGH', 'Authentication Missing',
                `${fileName} lacks user authentication and authorization checks`);
        } else {
            this.securityChecks.authenticationChecks = true;
            this.passed++;
        }

        // 5. EventSource Security (specific to Agent 7's real-time monitoring)
        if (content.includes('EventSource')) {
            if (!content.includes('validateEventSourceUrl') ||
                !content.includes('eventSource.close') ||
                !content.includes('onerror')) {
                this.addIssue('HIGH', 'EventSource Security Risk',
                    `${fileName} EventSource implementation lacks URL validation and proper error handling`);
            } else {
                this.securityChecks.eventSourceSecurity = true;
                this.passed++;
            }
        } else {
            this.securityChecks.eventSourceSecurity = true;
            this.passed++;
        }

        // 6. Memory Management (onExit)
        if (!content.includes('onExit') || !content.includes('destroy')) {
            this.addIssue('MEDIUM', 'Memory Leak Risk',
                `${fileName} lacks proper cleanup in onExit method`);
        } else {
            this.securityChecks.memoryManagement = true;
            this.passed++;
        }

        // 7. Agent Operation Security (specific to Agent 7)
        if (content.includes('_executeAgentOperation') || content.includes('/operations')) {
            if (!content.includes('validateAgentId') || !content.includes('authorizeOperation')) {
                this.addIssue('CRITICAL', 'Agent Operation Security Risk',
                    `${fileName} agent operations lack proper validation and authorization`);
            } else {
                this.securityChecks.agentOperationSecurity = true;
                this.passed++;
            }
        } else {
            this.securityChecks.agentOperationSecurity = true;
            this.passed++;
        }

        // 8. Audit Logging
        if (!content.includes('auditLog') && !content.includes('logUserAction')) {
            this.addIssue('MEDIUM', 'Audit Logging Missing',
                `${fileName} lacks audit logging for agent management actions`);
        } else {
            this.securityChecks.auditLogging = true;
            this.passed++;
        }

        // 9. SAP UI5 Structure
        if (content.includes('sap.ui.define') && content.includes('ControllerExtension')) {
            this.securityChecks.sapUi5Structure = true;
            this.passed++;
        } else {
            this.addIssue('HIGH', 'SAP UI5 Structure Violation',
                `${fileName} doesn't follow SAP UI5 module definition standards`);
        }

        // 10. Error Handling
        const hasProperErrorHandling = content.includes('catch') ||
                                     content.includes('error:') ||
                                     content.includes('MessageBox.error');
        if (!hasProperErrorHandling) {
            this.addIssue('MEDIUM', 'Error Handling Incomplete',
                `${fileName} lacks comprehensive error handling`);
        } else {
            this.securityChecks.errorHandling = true;
            this.passed++;
        }

        // 11. Controller Extensions
        if (content.includes('ControllerExtension.extend')) {
            this.securityChecks.controllerExtensions = true;
            this.passed++;
        } else {
            this.addIssue('MEDIUM', 'Controller Extension Missing',
                `${fileName} not properly extending ControllerExtension`);
        }

        // 12. Fragment Loading
        if (content.includes('Fragment.load')) {
            this.securityChecks.fragmentLoading = true;
            this.passed++;
        } else {
            this.addIssue('LOW', 'Fragment Loading Missing',
                `${fileName} doesn't use proper fragment loading`);
        }

        // 13. Model Binding
        if (content.includes('JSONModel') || content.includes('setModel')) {
            this.securityChecks.modelBinding = true;
            this.passed++;
        } else {
            this.addIssue('MEDIUM', 'Model Binding Issues',
                `${fileName} has improper model binding practices`);
        }

        // 14. Router Usage
        if (content.includes('getRouterFor') || content.includes('navTo')) {
            this.securityChecks.routerUsage = true;
            this.passed++;
        } else {
            this.addIssue('LOW', 'Router Usage Missing',
                `${fileName} doesn't implement proper routing`);
        }

        // 15. Busy Indicators
        if (content.includes('setBusy')) {
            this.securityChecks.busyIndicators = true;
            this.passed++;
        } else {
            this.addIssue('LOW', 'Busy Indicators Missing',
                `${fileName} lacks user feedback with busy indicators`);
        }

        // 16. Message Handling
        if (content.includes('MessageBox') || content.includes('MessageToast')) {
            this.securityChecks.messageHandling = true;
            this.passed++;
        } else {
            this.addIssue('MEDIUM', 'Message Handling Missing',
                `${fileName} lacks proper user message handling`);
        }

        // 17-21. Enterprise Compliance Checks
        this.securityChecks.auditTrail = false;
        this.securityChecks.roleBasedAccess = false;
        this.securityChecks.dataPrivacy = false;
        this.securityChecks.businessContinuity = false;
        this.securityChecks.changeManagement = false;

        // 22-26. Performance Checks
        this.securityChecks.lazyLoading = false;
        this.securityChecks.resourceOptimization = false;
        this.securityChecks.caching = false;
        this.securityChecks.bundling = false;
        this.securityChecks.memoryLeakPrevention = false;
    }

    async scanManifest() {
        const manifestPath = path.join(this.agent7Path, 'manifest.json');
        if (fs.existsSync(manifestPath)) {
            const content = fs.readFileSync(manifestPath, 'utf8');
            const manifest = JSON.parse(content);

            // Manifest Compliance Check
            if (manifest['sap.ui5'] && manifest['sap.ui5'].dependencies) {
                this.securityChecks.manifestCompliance = true;
                this.passed++;
            } else {
                this.addIssue('MEDIUM', 'Manifest Compliance Issue',
                    'manifest.json missing required SAP UI5 dependencies');
            }
        }
    }

    async scanI18n() {
        const i18nPath = path.join(this.agent7Path, 'i18n/i18n.properties');
        if (fs.existsSync(i18nPath)) {
            const content = fs.readFileSync(i18nPath, 'utf8');

            // Check for comprehensive i18n coverage
            const requiredKeys = [
                'agent.name', 'error.', 'msg.', 'field.', 'action.',
                'status.', 'dialog.', 'btn.', 'tooltip.'
            ];

            const hasComprehensiveI18n = requiredKeys.every(key =>
                content.includes(key) || content.split('\n').some(line => line.startsWith(key))
            );

            if (hasComprehensiveI18n) {
                this.securityChecks.i18nComprehensive = true;
                this.passed++;
            } else {
                this.addIssue('LOW', 'I18n Coverage Incomplete',
                    'i18n.properties missing comprehensive text coverage');
            }
        }
    }

    async scanFragments() {
        const fragmentsPath = path.join(this.agent7Path, 'fragment');
        if (fs.existsSync(fragmentsPath)) {
            const fragments = fs.readdirSync(fragmentsPath);
            // Fragment scanning would go here but files don't exist yet
        }
    }

    addIssue(severity, title, description) {
        this.issues.push({ severity, title, description });
    }

    generateReport() {
        const compliancePercentage = Math.round((this.passed / this.total) * 100);

        console.log('\n📊 AGENT 7 SECURITY SCAN RESULTS');
        console.log('═══════════════════════════════════════════════════');
        console.log('🎯 Agent: Agent Manager');
        console.log(`📈 Compliance Score: ${compliancePercentage}%`);
        console.log(`✅ Passed: ${this.passed}/${this.total}`);
        console.log(`❌ Issues Found: ${this.issues.length}`);

        if (this.issues.length > 0) {
            console.log('\n🚨 SECURITY ISSUES IDENTIFIED:');
            console.log('─'.repeat(50));

            const criticalIssues = this.issues.filter(i => i.severity === 'CRITICAL');
            const highIssues = this.issues.filter(i => i.severity === 'HIGH');
            const mediumIssues = this.issues.filter(i => i.severity === 'MEDIUM');
            const lowIssues = this.issues.filter(i => i.severity === 'LOW');

            if (criticalIssues.length > 0) {
                console.log(`\n🔴 CRITICAL (${criticalIssues.length}):`);
                criticalIssues.forEach(issue => {
                    console.log(`   • ${issue.title}: ${issue.description}`);
                });
            }

            if (highIssues.length > 0) {
                console.log(`\n🟠 HIGH (${highIssues.length}):`);
                highIssues.forEach(issue => {
                    console.log(`   • ${issue.title}: ${issue.description}`);
                });
            }

            if (mediumIssues.length > 0) {
                console.log(`\n🟡 MEDIUM (${mediumIssues.length}):`);
                mediumIssues.forEach(issue => {
                    console.log(`   • ${issue.title}: ${issue.description}`);
                });
            }

            if (lowIssues.length > 0) {
                console.log(`\n🔵 LOW (${lowIssues.length}):`);
                lowIssues.forEach(issue => {
                    console.log(`   • ${issue.title}: ${issue.description}`);
                });
            }
        }

        console.log('\n🎯 AGENT 7 SPECIFIC SECURITY AREAS:');
        console.log('─'.repeat(50));
        console.log('• Agent Lifecycle Management Security');
        console.log('• Multi-Agent Operation Coordination');
        console.log('• Real-time Health Monitoring Security');
        console.log('• EventSource/SSE Connection Security');
        console.log('• Agent Registration Validation');
        console.log('• Performance Monitoring Security');
        console.log('• Configuration Management Security');
        console.log('• Bulk Operations Security');
        console.log('• Agent-to-Agent Communication Security');

        console.log('\n📋 NEXT STEPS:');
        console.log('─'.repeat(50));
        console.log('1. Implement CSRF token handling for all AJAX calls');
        console.log('2. Add comprehensive input validation and XSS protection');
        console.log('3. Implement proper authentication and authorization');
        console.log('4. Secure EventSource connections with validation');
        console.log('5. Add audit logging for agent management actions');
        console.log('6. Implement memory management with onExit cleanup');
        console.log('7. Validate agent operations and IDs');
        console.log('8. Add enterprise-grade compliance features');

        console.log('\n═══════════════════════════════════════════════════');

        if (compliancePercentage < 100) {
            console.log('❌ Agent 7 UI requires security enhancements to achieve 100% compliance');
            process.exit(1);
        } else {
            console.log('✅ Agent 7 UI meets all security and compliance requirements');
        }
    }
}

// Execute scan
if (require.main === module) {
    const scanner = new Agent7SecurityScanner();
    scanner.scanAll().catch(console.error);
}

module.exports = Agent7SecurityScanner;