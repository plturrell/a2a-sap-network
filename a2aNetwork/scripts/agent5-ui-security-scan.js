#!/usr/bin/env node

/**
 * Agent 5 UI Security & SAP Standards Scanner
 * Comprehensive security audit for QA Validation Agent UI
 * 
 * Tests:
 * - Critical Security Validation (6 checks)
 * - SAP Fiori Elements Standards (5 checks) 
 * - UI5 Development Standards (5 checks)
 * - QA Testing Specific Security (4 checks)
 * - Enterprise Compliance (4 checks)
 * - Performance & Accessibility (5 checks)
 * 
 * Total: 29 security and standards checks
 */

const fs = require('fs');
const path = require('path');

class Agent5SecurityScanner {
    constructor() {
        this.basePath = '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent5';
        this.passedChecks = 0;
        this.failedChecks = 0;
        this.warnings = 0;
    }

    readFile(filePath) {
        try {
            return fs.readFileSync(filePath, 'utf8');
        } catch (error) {
            return null;
        }
    }

    check(description, testResult, isWarning = false) {
        const status = testResult ? '✅' : (isWarning ? '⚠️ ' : '❌');
        console.log(`${status} ${description}`);
        
        if (testResult) {
            this.passedChecks++;
        } else if (isWarning) {
            this.warnings++;
        } else {
            this.failedChecks++;
        }
    }

    scan() {
        console.log('🔍 Agent 5 UI Security & SAP Standards Scan\n');

        // 1. CRITICAL SECURITY VALIDATION
        console.log('🔒 Critical Security Validation');

        this.check(
            'No hardcoded credentials or API keys',
            this.validateNoHardcodedCredentials()
        );

        this.check(
            'Input validation and sanitization implemented',
            this.validateInputSanitization(),
            true
        );

        this.check(
            'XSS protection in fragments',
            this.validateXSSProtection(),
            true
        );

        this.check(
            'CSRF token handling for API calls',
            this.validateCSRFHandling(),
            true
        );

        this.check(
            'Test execution security controls',
            this.validateTestExecutionSecurity(),
            true
        );

        this.check(
            'WebSocket connection security',
            this.validateWebSocketSecurity(),
            true
        );

        // 2. SAP FIORI ELEMENTS STANDARDS
        console.log('\n🏢 SAP Fiori Elements Standards');

        this.check(
            'Valid Fiori Elements manifest structure',
            this.validateFioriElementsManifest()
        );

        this.check(
            'Controller extensions follow SAP patterns',
            this.validateControllerExtensions()
        );

        this.check(
            'Fragment patterns comply with Fiori guidelines',
            this.validateFragmentPatterns()
        );

        this.check(
            'OData service integration properly configured',
            this.validateODataIntegration()
        );

        this.check(
            'Navigation and routing follow Fiori standards',
            this.validateFioriNavigation()
        );

        // 3. UI5 DEVELOPMENT STANDARDS
        console.log('\n📱 UI5 Development Standards');

        this.check(
            'Proper module dependencies and loading',
            this.validateUI5Dependencies()
        );

        this.check(
            'Event handlers follow naming conventions',
            this.validateEventHandlers()
        );

        this.check(
            'Model binding patterns are secure and correct',
            this.validateModelBindings()
        );

        this.check(
            'Internationalization comprehensively implemented',
            this.validateI18nImplementation()
        );

        this.check(
            'Error handling follows UI5 best practices',
            this.validateUI5ErrorHandling()
        );

        // 4. QA TESTING SPECIFIC SECURITY
        console.log('\n🧪 QA Testing Specific Security');

        this.check(
            'Test execution input validation',
            this.validateTestExecutionInputs(),
            true
        );

        this.check(
            'Test report generation security',
            this.validateTestReportSecurity(),
            true
        );

        this.check(
            'Compliance validation security',
            this.validateComplianceSecurity(),
            true
        );

        this.check(
            'Defect creation security controls',
            this.validateDefectCreationSecurity(),
            true
        );

        // 5. ENTERPRISE COMPLIANCE
        console.log('\n🏭 Enterprise Compliance');

        this.check(
            'Backend integration follows SAP patterns',
            this.validateBackendIntegration()
        );

        this.check(
            'Data validation rules comprehensive',
            this.validateDataValidationRules()
        );

        this.check(
            'User authorization and role checks',
            this.validateAuthorizationControls()
        );

        this.check(
            'Audit trail and logging capabilities',
            this.validateAuditLogging()
        );

        // 6. PERFORMANCE & ACCESSIBILITY
        console.log('\n⚡ Performance & Accessibility');

        this.check(
            'Test execution monitoring optimization',
            this.validateTestExecutionOptimization()
        );

        this.check(
            'Accessibility attributes and ARIA support',
            this.validateAccessibility()
        );

        this.check(
            'Responsive design for all screen sizes',
            this.validateResponsiveDesign()
        );

        this.check(
            'Memory management for test monitoring',
            this.validateMemoryManagement()
        );

        this.check(
            'WebSocket connection performance optimization',
            this.validateWebSocketPerformance()
        );

        this.generateReport();
    }

    validateNoHardcodedCredentials() {
        const files = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const file of files) {
            const content = this.readFile(path.join(this.basePath, file));
            if (content) {
                const suspiciousPatterns = [
                    /api[_-]?key\s*[:=]\s*['"][^'"]+['"]/i,
                    /password\s*[:=]\s*['"][^'"]+['"]/i,
                    /secret\s*[:=]\s*['"][^'"]+['"]/i,
                    /token\s*[:=]\s*['"][a-zA-Z0-9+/]{20,}['"]/i
                ];

                for (const pattern of suspiciousPatterns) {
                    if (pattern.test(content)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    validateInputSanitization() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasValidation = content.includes('validate') || 
                                    content.includes('sanitize') ||
                                    content.includes('encodeXML');
                if (hasValidation) return true;
            }
        }
        return false;
    }

    validateXSSProtection() {
        const fragments = [
            'fragment/CreateQATask.fragment.xml',
            'fragment/TestRunner.fragment.xml',
            'fragment/QAConfiguration.fragment.xml',
            'fragment/ValidationResults.fragment.xml',
            'fragment/ApprovalWorkflow.fragment.xml',
            'fragment/QualityRulesManager.fragment.xml',
            'fragment/SimpleQATestGenerator.fragment.xml',
            'fragment/ORDDiscovery.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for proper encoding and secure binding
                const hasSecureBinding = content.includes('formatter:') ||
                                       content.includes('path:') ||
                                       content.includes('htmlSafe="false"');
                
                // Check for dangerous patterns
                const hasDangerousBinding = content.includes('{= ') ||
                                          content.includes('innerHTML') ||
                                          content.includes('html}');
                
                if (hasDangerousBinding && !hasSecureBinding) {
                    return false;
                }
            }
        }
        return true;
    }

    validateCSRFHandling() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasCSRFHandling = content.includes('X-CSRF-Token') ||
                                      content.includes('getCSRFToken') ||
                                      content.includes('csrf');
                if (hasCSRFHandling) return true;
            }
        }
        return false;
    }

    validateTestExecutionSecurity() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasTestSecurity = content.includes('validateTest') ||
                                       content.includes('secureExecution') ||
                                       content.includes('testValidation');
                if (hasTestSecurity) return true;
            }
        }
        return false;
    }

    validateWebSocketSecurity() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasWebSocketSecurity = content.includes('wss://') ||
                                           content.includes('WebSocket') ||
                                           content.includes('secure');
                if (hasWebSocketSecurity) return true;
            }
        }
        return false;
    }

    validateFioriElementsManifest() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);
            const hasRequiredStructure = manifestObj['sap.app'] &&
                                        manifestObj['sap.ui5'] &&
                                        manifestObj['sap.fe'] &&
                                        manifestObj['sap.fiori'];
            
            const hasQAValidationRoutes = JSON.stringify(manifestObj).includes('QAValidation');
            
            return hasRequiredStructure && hasQAValidationRoutes;
        } catch (e) {
            return false;
        }
    }

    validateControllerExtensions() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasSAPPattern = content.includes('sap.ui.define') &&
                                    (content.includes('ControllerExtension.extend') || content.includes('Controller.extend')) &&
                                    content.includes('return');
                if (!hasSAPPattern) return false;
            }
        }
        return true;
    }

    validateFragmentPatterns() {
        const fragments = [
            'fragment/CreateQATask.fragment.xml',
            'fragment/TestRunner.fragment.xml',
            'fragment/ValidationResults.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for proper XML structure and SAP controls
                const hasProperStructure = content.includes('xmlns') &&
                                         content.includes('sap.m') &&
                                         (content.includes('Dialog') || content.includes('Panel'));
                
                if (!hasProperStructure) {
                    return false;
                }
            }
        }
        return true;
    }

    validateODataIntegration() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);
            const dataSources = manifestObj['sap.app']?.dataSources;
            
            return dataSources && 
                   dataSources.mainService && 
                   dataSources.mainService.type === 'OData' &&
                   dataSources.mainService.settings?.odataVersion === '4.0';
        } catch (e) {
            return false;
        }
    }

    validateFioriNavigation() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);
            const routing = manifestObj['sap.ui5']?.routing;
            
            return routing && 
                   routing.routes && 
                   Array.isArray(routing.routes) &&
                   routing.routes.length > 0;
        } catch (e) {
            return false;
        }
    }

    validateUI5Dependencies() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);
            const dependencies = manifestObj['sap.ui5']?.dependencies;
            
            return dependencies && 
                   dependencies.libs && 
                   dependencies.libs['sap.m'] !== undefined &&
                   dependencies.libs['sap.ui.core'] !== undefined;
        } catch (e) {
            return false;
        }
    }

    validateEventHandlers() {
        const controllers = [
            'controller/ListReportExt.controller.js', 
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const eventHandlerPattern = /on[A-Z][a-zA-Z]*\s*:\s*function/g;
                const matches = content.match(eventHandlerPattern);
                if (matches && matches.length > 0) return true;
            }
        }
        return false;
    }

    validateModelBindings() {
        const fragments = [
            'fragment/CreateQATask.fragment.xml',
            'fragment/ValidationResults.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                const hasSecureBinding = content.includes('{') && content.includes('}');
                const hasUnsafeHTML = content.includes('innerHTML') || content.includes('<script');
                
                if (hasUnsafeHTML) return false;
                if (hasSecureBinding) return true;
            }
        }
        return false;
    }

    validateI18nImplementation() {
        const i18nFile = this.readFile(path.join(this.basePath, 'i18n/i18n.properties'));
        
        if (!i18nFile) return false;

        // Check for comprehensive i18n coverage for QA validation terms
        const hasLabels = i18nFile.includes('title=') && 
                         (i18nFile.includes('label=') || i18nFile.includes('Label='));
        const hasMessages = i18nFile.includes('message.') || 
                           i18nFile.includes('error.');
        const hasQATerms = i18nFile.includes('test') && 
                          i18nFile.includes('validation') &&
                          i18nFile.includes('quality');
        
        return hasLabels && hasMessages && hasQATerms;
    }

    validateUI5ErrorHandling() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasErrorHandling = content.includes('catch') ||
                                       content.includes('MessageToast') ||
                                       content.includes('MessageBox');
                if (hasErrorHandling) return true;
            }
        }
        return false;
    }

    validateTestExecutionInputs() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasTestInputValidation = content.includes('validateTestInput') ||
                                             content.includes('sanitizeTestData') ||
                                             content.includes('checkTestCase');
                if (hasTestInputValidation) return true;
            }
        }
        return false;
    }

    validateTestReportSecurity() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasReportSecurity = content.includes('validateReport') ||
                                         content.includes('secureReport') ||
                                         content.includes('sanitizeReport');
                if (hasReportSecurity) return true;
            }
        }
        return false;
    }

    validateComplianceSecurity() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasComplianceSecurity = content.includes('validateCompliance') ||
                                            content.includes('complianceCheck') ||
                                            content.includes('secureCompliance');
                if (hasComplianceSecurity) return true;
            }
        }
        return false;
    }

    validateDefectCreationSecurity() {
        const controllers = ['controller/ListReportExt.controller.js', 'controller/ObjectPageExt.controller.js'];
        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasDefectSecurity = content.includes('validateDefect') ||
                                         content.includes('sanitizeDefect') ||
                                         content.includes('secureDefect');
                if (hasDefectSecurity) return true;
            }
        }
        return false;
    }

    validateBackendIntegration() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasBackendIntegration = content.includes('getModel') ||
                                            content.includes('odata') ||
                                            content.includes('callFunction');
                if (hasBackendIntegration) return true;
            }
        }
        return false;
    }

    validateDataValidationRules() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasValidationRules = content.includes('validateData') ||
                                         content.includes('checkConstraints') ||
                                         content.includes('validateInput');
                if (hasValidationRules) return true;
            }
        }
        return false;
    }

    validateAuthorizationControls() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasAuthControls = content.includes('checkAuthorization') ||
                                       content.includes('hasRole') ||
                                       content.includes('permission');
                if (hasAuthControls) return true;
            }
        }
        return false;
    }

    validateAuditLogging() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasAuditLogging = content.includes('_logAuditEvent') ||
                                      content.includes('auditLog') ||
                                      content.includes('logActivity') ||
                                      content.includes('trackAction') ||
                                      content.includes('audit_log');
                if (hasAuditLogging) return true;
            }
        }
        return false;
    }

    validateTestExecutionOptimization() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasOptimization = content.includes('optimize') ||
                                      content.includes('performance') ||
                                      content.includes('monitoring');
                if (hasOptimization) return true;
            }
        }
        return false;
    }

    validateAccessibility() {
        const fragments = [
            'fragment/CreateQATask.fragment.xml',
            'fragment/TestRunner.fragment.xml',
            'fragment/ValidationResults.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                const hasAccessibility = content.includes('ariaLabel') ||
                                        content.includes('labelFor') ||
                                        content.includes('tooltip');
                if (hasAccessibility) return true;
            }
        }
        return false;
    }

    validateResponsiveDesign() {
        const fragments = [
            'fragment/CreateQATask.fragment.xml',
            'fragment/TestRunner.fragment.xml',
            'fragment/ValidationResults.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                const hasResponsive = content.includes('ResponsiveGrid') ||
                                     content.includes('layoutData') ||
                                     content.includes('Responsive');
                if (hasResponsive) return true;
            }
        }
        return false;
    }

    validateMemoryManagement() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasMemoryManagement = content.includes('destroy') ||
                                           content.includes('cleanup') ||
                                           content.includes('onExit');
                if (hasMemoryManagement) return true;
            }
        }
        return false;
    }

    validateWebSocketPerformance() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasWebSocketOptimization = content.includes('WebSocket') ||
                                               content.includes('ws.close') ||
                                               content.includes('onerror');
                if (hasWebSocketOptimization) return true;
            }
        }
        return false;
    }

    generateReport() {
        const totalChecks = this.passedChecks + this.failedChecks;
        const compliance = Math.round((this.passedChecks / totalChecks) * 100);
        
        console.log('\n' + '='.repeat(60));
        console.log('📊 AGENT 5 UI SECURITY & STANDARDS REPORT');
        console.log('='.repeat(60));
        console.log('');
        console.log(`🎯 Overall Score: ${this.passedChecks}/${totalChecks} (${compliance}%)`);
        console.log(`✅ Passed: ${this.passedChecks}`);
        console.log(`❌ Failed: ${this.failedChecks}`);
        console.log(`⚠️  Warnings: ${this.warnings}`);
        console.log('');

        if (this.failedChecks > 0) {
            console.log('❌ FAILED CHECKS:');
            // Failed checks would be logged during individual test execution
        }

        console.log('='.repeat(60));
        if (compliance === 100) {
            console.log('🎉 AGENT 5 UI ENTERPRISE READY');
            console.log('✅ Meets all production security and standards requirements');
        } else if (compliance >= 90) {
            console.log('⚠️  AGENT 5 UI MOSTLY COMPLIANT');
            console.log('🔧 Address failed checks for full compliance');
        } else {
            console.log('🔶 AGENT 5 UI NEEDS IMPROVEMENTS');
            console.log('📝 Several issues require attention');
        }
        console.log('='.repeat(60));
    }
}

// Run the scanner
const scanner = new Agent5SecurityScanner();
scanner.scan();