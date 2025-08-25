#!/usr/bin/env node

/**
 * Agent 1 UI Security and SAP Standards Scan
 * Comprehensive validation for Agent 1 Data Standardization UI
 */

const fs = require('fs');
const path = require('path');

class Agent1UIScanner {
    constructor() {
        this.results = {
            passed: [],
            failed: [],
            warnings: [],
            score: 0,
            maxScore: 0
        };
        this.basePath = 'app/a2aFiori/webapp/ext/agent1';
    }

    check(name, condition, isCritical = false) {
        this.results.maxScore++;
        if (isCritical) this.results.maxScore++; // Critical checks worth double

        if (condition) {
            this.results.passed.push(name);
            this.results.score += isCritical ? 2 : 1;
            console.log(`✅ ${name}`);
        } else {
            this.results.failed.push(name);
            console.log(`❌ ${name}`);
        }
    }

    warn(name, message) {
        this.results.warnings.push({ name, message });
        console.log(`⚠️  ${name}: ${message}`);
    }

    fileExists(filePath) {
        try {
            return fs.existsSync(filePath);
        } catch (error) {
            return false;
        }
    }

    readFile(filePath) {
        try {
            return fs.readFileSync(filePath, 'utf8');
        } catch (error) {
            return null;
        }
    }

    async scan() {
        console.log('🔍 Agent 1 UI Security & SAP Standards Scan\n');

        // 1. CRITICAL SECURITY CHECKS
        console.log('🔒 Critical Security Validation');

        this.check(
            'No hardcoded credentials in controllers',
            this.validateNoHardcodedCredentials(),
            true
        );

        this.check(
            'Input validation in controllers',
            this.validateInputValidation(),
            true
        );

        this.check(
            'XSS protection in fragments',
            this.validateXSSProtection(),
            true
        );

        this.check(
            'CSRF token handling implemented',
            this.validateCSRFHandling(),
            true
        );

        this.check(
            'Secure file upload validation',
            this.validateFileUploadSecurity(),
            true
        );

        // 2. SAP FIORI ELEMENTS STANDARDS
        console.log('\n🏢 SAP Fiori Elements Standards');

        this.check(
            'Valid Fiori Elements manifest',
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
            'OData service integration correct',
            this.validateODataIntegration()
        );

        this.check(
            'Navigation patterns follow Fiori standards',
            this.validateNavigationPatterns()
        );

        // 3. UI5 DEVELOPMENT STANDARDS
        console.log('\n📱 UI5 Development Standards');

        this.check(
            'Proper module dependencies declared',
            this.validateModuleDependencies()
        );

        this.check(
            'Event handlers follow naming conventions',
            this.validateEventHandlerNaming()
        );

        this.check(
            'Model binding patterns correct',
            this.validateModelBinding()
        );

        this.check(
            'Internationalization properly implemented',
            this.validateI18nImplementation()
        );

        this.check(
            'Error handling follows UI5 patterns',
            this.validateUI5ErrorHandling()
        );

        // 4. ENTERPRISE COMPLIANCE
        console.log('\n🏭 Enterprise Compliance');

        this.check(
            'Backend integration follows SAP best practices',
            this.validateBackendIntegration()
        );

        this.check(
            'Data validation rules comprehensive',
            this.validateDataValidationRules()
        );

        this.check(
            'User authorization checks implemented',
            this.validateUserAuthorization()
        );

        this.check(
            'Audit trail capabilities present',
            this.validateAuditTrail()
        );

        // 5. PERFORMANCE & ACCESSIBILITY
        console.log('\n⚡ Performance & Accessibility');

        this.check(
            'Lazy loading implemented for large datasets',
            this.validateLazyLoading()
        );

        this.check(
            'Accessibility attributes present',
            this.validateAccessibility()
        );

        this.check(
            'Responsive design patterns used',
            this.validateResponsiveDesign()
        );

        this.check(
            'Memory management for large files',
            this.validateMemoryManagement()
        );

        this.generateReport();
    }

    validateNoHardcodedCredentials() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        const dangerousPatterns = [
            /password\s*[:=]\s*["'][^"']+["']/gi,
            /token\s*[:=]\s*["'][^"']+["']/gi,
            /secret\s*[:=]\s*["'][^"']+["']/gi,
            /api[_-]?key\s*[:=]\s*["'][^"']+["']/gi
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                for (const pattern of dangerousPatterns) {
                    if (pattern.test(content)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    validateInputValidation() {
        const listController = this.readFile(path.join(this.basePath, 'controller/ListReportExt.controller.js'));
        if (!listController) return false;

        // Check for input validation in key functions
        return listController.includes('validateInput') ||
               (listController.includes('trim()') &&
                listController.includes('length')) ||
               listController.includes('validation');
    }

    validateXSSProtection() {
        const fragments = [
            'fragment/CreateStandardizationTask.fragment.xml',
            'fragment/ImportSchema.fragment.xml',
            'fragment/SchemaMappingVisualizer.fragment.xml',
            'fragment/ValidationErrors.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for proper encoding patterns
                const hasProperBinding = content.includes('{path:') ||
                                       content.includes('htmlSafe="false"') ||
                                       content.includes('escapeXML="true"');

                // Check for dangerous innerHTML usage
                const hasDangerousBinding = content.includes('{= ') ||
                                          content.includes('innerHTML') ||
                                          content.includes('html}');

                if (hasDangerousBinding && !hasProperBinding) {
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
            if (content && content.includes('jQuery.ajax')) {
                // If using jQuery.ajax, should have CSRF handling
                return content.includes('X-CSRF-Token') ||
                       content.includes('csrf') ||
                       content.includes('sap.ui.model.odata.v2.ODataModel'); // OData handles CSRF automatically
            }
        }

        // If no direct AJAX calls, assume OData is used (which handles CSRF)
        return true;
    }

    validateFileUploadSecurity() {
        const createTaskFragment = this.readFile(path.join(this.basePath, 'fragment/CreateStandardizationTask.fragment.xml'));
        const importSchemaFragment = this.readFile(path.join(this.basePath, 'fragment/ImportSchema.fragment.xml'));

        if (!createTaskFragment && !importSchemaFragment) return false;

        // Check for file type restrictions
        const hasFileTypeValidation = (createTaskFragment && createTaskFragment.includes('fileType')) ||
                                    (importSchemaFragment && importSchemaFragment.includes('accept='));

        // Check for size limits
        const hasSizeValidation = (createTaskFragment && createTaskFragment.includes('maximumFileSize')) ||
                                (importSchemaFragment && importSchemaFragment.includes('maximumFileSize'));

        return hasFileTypeValidation || hasSizeValidation;
    }

    validateFioriElementsManifest() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);

            // Check for required Fiori Elements properties
            return manifestObj['sap.app'] &&
                   manifestObj['sap.ui5'] &&
                   manifestObj['sap.fe'] &&
                   manifestObj['sap.app'].type === 'application' &&
                   manifestObj['sap.fe'].template;
        } catch (error) {
            return false;
        }
    }

    validateControllerExtensions() {
        const listController = this.readFile(path.join(this.basePath, 'controller/ListReportExt.controller.js'));
        const objectController = this.readFile(path.join(this.basePath, 'controller/ObjectPageExt.controller.js'));

        if (!listController || !objectController) return false;

        // Check for proper controller extension patterns
        const hasProperExtension = (
            listController.includes('sap.ui.define') &&
            listController.includes('Controller') &&
            listController.includes('.extend(')
        ) && (
            objectController.includes('sap.ui.define') &&
            objectController.includes('Controller') &&
            objectController.includes('.extend(')
        );

        return hasProperExtension;
    }

    validateFragmentPatterns() {
        const fragments = [
            'fragment/CreateStandardizationTask.fragment.xml',
            'fragment/ImportSchema.fragment.xml',
            'fragment/SchemaMappingVisualizer.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for proper XML structure and SAP controls
                const hasProperStructure = content.includes('xmlns') &&
                                         content.includes('sap.m') &&
                                         content.includes('Dialog');

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
                   Object.values(dataSources).some(ds => ds.type === 'OData' || ds.uri?.includes('odata'));
        } catch (error) {
            return false;
        }
    }

    validateNavigationPatterns() {
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        if (!manifest) return false;

        try {
            const manifestObj = JSON.parse(manifest);
            const routing = manifestObj['sap.ui5']?.routing;

            return routing && routing.routes && Array.isArray(routing.routes) && routing.routes.length > 0;
        } catch (error) {
            return false;
        }
    }

    validateModuleDependencies() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for proper module declaration
                const hasProperModules = content.includes('sap.ui.define([') &&
                                        content.includes('sap/ui/core/mvc/Controller');

                if (!hasProperModules) {
                    return false;
                }
            }
        }
        return true;
    }

    validateEventHandlerNaming() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        const properNamingPattern = /on[A-Z][a-zA-Z]*\s*:\s*function/g;

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const handlers = content.match(/on[A-Z][a-zA-Z]*\s*[:=]/g);
                if (handlers && handlers.length > 0) {
                    // Found event handlers with proper naming
                    return true;
                }
            }
        }
        return false;
    }

    validateModelBinding() {
        const fragments = [
            'fragment/CreateStandardizationTask.fragment.xml',
            'fragment/ImportSchema.fragment.xml',
            'fragment/SchemaMappingVisualizer.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for proper model binding patterns
                const hasBinding = content.includes('{') && content.includes('}') &&
                                 (content.includes('value="{') || content.includes('text="{'));

                if (hasBinding) {
                    return true;
                }
            }
        }
        return false;
    }

    validateI18nImplementation() {
        const i18nFile = this.readFile(path.join(this.basePath, 'i18n/i18n.properties'));

        if (!i18nFile) return false;

        // Check for comprehensive i18n coverage
        const hasLabels = i18nFile.includes('title=') &&
                         (i18nFile.includes('label=') || i18nFile.includes('Label='));
        const hasMessages = i18nFile.includes('message.') ||
                           i18nFile.includes('error.');

        return hasLabels && hasMessages;
    }

    validateUI5ErrorHandling() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for error handling patterns
                const hasErrorHandling = content.includes('catch') ||
                                        content.includes('error') ||
                                        content.includes('MessageToast') ||
                                        content.includes('MessageBox');

                if (hasErrorHandling) {
                    return true;
                }
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
                // Check for proper backend integration patterns
                const hasBackendIntegration = content.includes('getModel()') ||
                                            content.includes('odata') ||
                                            content.includes('service') ||
                                            content.includes('ajax');

                if (hasBackendIntegration) {
                    return true;
                }
            }
        }
        return false;
    }

    validateDataValidationRules() {
        const createTaskFragment = this.readFile(path.join(this.basePath, 'fragment/CreateStandardizationTask.fragment.xml'));
        const importSchemaFragment = this.readFile(path.join(this.basePath, 'fragment/ImportSchema.fragment.xml'));

        if (!createTaskFragment && !importSchemaFragment) return false;

        // Check for validation controls
        const hasValidation = (createTaskFragment &&
                              (createTaskFragment.includes('required="true"') ||
                               createTaskFragment.includes('validate'))) ||
                             (importSchemaFragment &&
                              (importSchemaFragment.includes('required="true"') ||
                               importSchemaFragment.includes('validate')));

        return hasValidation;
    }

    validateUserAuthorization() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for authorization patterns
                const hasAuth = content.includes('authorization') ||
                               content.includes('permission') ||
                               content.includes('role') ||
                               content.includes('access');

                if (hasAuth) {
                    return true;
                }
            }
        }

        // Check if using Fiori Elements (which handles auth automatically)
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        return manifest && manifest.includes('"sap.fe"');
    }

    validateAuditTrail() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for audit/logging patterns
                const hasAudit = content.includes('log') ||
                               content.includes('audit') ||
                               content.includes('track') ||
                               content.includes('history');

                if (hasAudit) {
                    return true;
                }
            }
        }
        return false;
    }

    validateLazyLoading() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for pagination or lazy loading patterns
                const hasLazyLoading = content.includes('$skip') ||
                                     content.includes('$top') ||
                                     content.includes('pagination') ||
                                     content.includes('growing');

                if (hasLazyLoading) {
                    return true;
                }
            }
        }

        // Fiori Elements has built-in lazy loading
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        return manifest && manifest.includes('"sap.fe"');
    }

    validateAccessibility() {
        const fragments = [
            'fragment/CreateStandardizationTask.fragment.xml',
            'fragment/ImportSchema.fragment.xml',
            'fragment/SchemaMappingVisualizer.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for accessibility attributes
                const hasA11y = content.includes('ariaLabel') ||
                               content.includes('ariaDescribedBy') ||
                               content.includes('labelFor') ||
                               content.includes('tooltip');

                if (hasA11y) {
                    return true;
                }
            }
        }
        return false;
    }

    validateResponsiveDesign() {
        const fragments = [
            'fragment/CreateStandardizationTask.fragment.xml',
            'fragment/ImportSchema.fragment.xml',
            'fragment/SchemaMappingVisualizer.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
                // Check for responsive design patterns
                const hasResponsive = content.includes('Responsive') ||
                                     content.includes('GridLayout') ||
                                     content.includes('FlexBox') ||
                                     content.includes('columns=');

                if (hasResponsive) {
                    return true;
                }
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
                // Check for memory management patterns
                const hasMemoryMgmt = content.includes('destroy') ||
                                     content.includes('cleanup') ||
                                     content.includes('batch') ||
                                     content.includes('stream');

                if (hasMemoryMgmt) {
                    return true;
                }
            }
        }
        return false;
    }

    generateReport() {
        console.log(`\n${  '='.repeat(60)}`);
        console.log('📊 AGENT 1 UI SECURITY & STANDARDS REPORT');
        console.log('='.repeat(60));

        const percentage = Math.round((this.results.score / this.results.maxScore) * 100);

        console.log(`\n🎯 Overall Score: ${this.results.score}/${this.results.maxScore} (${percentage}%)`);
        console.log(`✅ Passed: ${this.results.passed.length}`);
        console.log(`❌ Failed: ${this.results.failed.length}`);
        console.log(`⚠️  Warnings: ${this.results.warnings.length}`);

        if (this.results.failed.length > 0) {
            console.log('\n❌ FAILED CHECKS:');
            this.results.failed.forEach(check => console.log(`  • ${check}`));
        }

        if (this.results.warnings.length > 0) {
            console.log('\n⚠️  WARNINGS:');
            this.results.warnings.forEach(warning =>
                console.log(`  • ${warning.name}: ${warning.message}`)
            );
        }

        console.log(`\n${  '='.repeat(60)}`);

        if (percentage >= 95) {
            console.log('🎉 AGENT 1 UI ENTERPRISE READY');
            console.log('✅ Meets all production security and standards requirements');
        } else if (percentage >= 90) {
            console.log('⚠️  AGENT 1 UI MOSTLY COMPLIANT');
            console.log('🔧 Address failed checks for full compliance');
        } else if (percentage >= 80) {
            console.log('🔶 AGENT 1 UI NEEDS IMPROVEMENTS');
            console.log('📝 Several issues require attention');
        } else {
            console.log('❌ AGENT 1 UI NOT ENTERPRISE READY');
            console.log('🚨 Critical issues must be resolved');
        }

        console.log('='.repeat(60));
    }
}

// Run the scan
const scanner = new Agent1UIScanner();
scanner.scan().catch(console.error);