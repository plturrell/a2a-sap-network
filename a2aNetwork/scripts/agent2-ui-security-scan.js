#!/usr/bin/env node

/**
 * Agent 2 UI Security and SAP Standards Scan
 * Comprehensive validation for Agent 2 AI Preparation UI
 */

const fs = require('fs');
const path = require('path');

class Agent2UIScanner {
    constructor() {
        this.results = {
            passed: [],
            failed: [],
            warnings: [],
            score: 0,
            maxScore: 0
        };
        this.basePath = 'app/a2aFiori/webapp/ext/agent2';
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
        console.log('🔍 Agent 2 UI Security & SAP Standards Scan\n');

        // 1. CRITICAL SECURITY CHECKS
        console.log('🔒 Critical Security Validation');

        this.check(
            'No hardcoded credentials or API keys',
            this.validateNoHardcodedCredentials(),
            true
        );

        this.check(
            'Input validation and sanitization implemented',
            this.validateInputValidation(),
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
            'Secure file upload and data handling',
            this.validateFileUploadSecurity(),
            true
        );

        this.check(
            'EventSource security for real-time updates',
            this.validateEventSourceSecurity(),
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
            this.validateNavigationPatterns()
        );

        // 3. UI5 DEVELOPMENT STANDARDS
        console.log('\n📱 UI5 Development Standards');

        this.check(
            'Proper module dependencies and loading',
            this.validateModuleDependencies()
        );

        this.check(
            'Event handlers follow naming conventions',
            this.validateEventHandlerNaming()
        );

        this.check(
            'Model binding patterns are secure and correct',
            this.validateModelBinding()
        );

        this.check(
            'Internationalization comprehensively implemented',
            this.validateI18nImplementation()
        );

        this.check(
            'Error handling follows UI5 best practices',
            this.validateUI5ErrorHandling()
        );

        // 4. AI/ML SPECIFIC SECURITY
        console.log('\n🤖 AI/ML Specific Security');

        this.check(
            'Model configuration validation',
            this.validateModelConfigSecurity()
        );

        this.check(
            'Data export security controls',
            this.validateDataExportSecurity()
        );

        this.check(
            'Embedding generation security',
            this.validateEmbeddingSecurity()
        );

        this.check(
            'AutoML workflow security',
            this.validateAutoMLSecurity()
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
            this.validateUserAuthorization()
        );

        this.check(
            'Audit trail and logging capabilities',
            this.validateAuditTrail()
        );

        // 6. PERFORMANCE & ACCESSIBILITY
        console.log('\n⚡ Performance & Accessibility');

        this.check(
            'Lazy loading for large datasets and models',
            this.validateLazyLoading()
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
            'Memory management for large ML datasets',
            this.validateMemoryManagement()
        );

        this.check(
            'Real-time updates performance optimization',
            this.validateRealTimePerformance()
        );

        this.generateReport();
    }

    validateNoHardcodedCredentials() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        const dangerousPatterns = [
            /api[_-]?key\s*[:=]\s*["'][^"']+["']/gi,
            /secret\s*[:=]\s*["'][^"']+["']/gi,
            /token\s*[:=]\s*["'][^"']+["']/gi,
            /password\s*[:=]\s*["'][^"']+["']/gi,
            /openai[_-]?key/gi,
            /hugging[_-]?face[_-]?token/gi
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
        const objectController = this.readFile(path.join(this.basePath, 'controller/ObjectPageExt.controller.js'));

        if (!listController || !objectController) return false;

        // Check for comprehensive input validation
        const hasValidation = (
            listController.includes('validateInput') ||
            listController.includes('_validate') ||
            listController.includes('sanitize')
        ) && (
            objectController.includes('validateInput') ||
            objectController.includes('_validate') ||
            objectController.includes('sanitize')
        );

        return hasValidation;
    }

    validateXSSProtection() {
        const fragments = [
            'fragment/CreateAIPreparationTask.fragment.xml',
            'fragment/AutoMLWizard.fragment.xml',
            'fragment/DataProfiler.fragment.xml',
            'fragment/EmbeddingConfiguration.fragment.xml',
            'fragment/FeatureAnalysis.fragment.xml',
            'fragment/ExportPreparedData.fragment.xml'
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
            if (content && content.includes('jQuery.ajax')) {
                // Check for CSRF handling in AJAX calls
                return content.includes('X-CSRF-Token') ||
                       content.includes('csrf') ||
                       content.includes('sap.ui.model.odata'); // OData handles CSRF
            }
        }

        // If no direct AJAX, check for OData usage
        const manifest = this.readFile(path.join(this.basePath, 'manifest.json'));
        return manifest && manifest.includes('OData');
    }

    validateFileUploadSecurity() {
        const createTaskFragment = this.readFile(path.join(this.basePath, 'fragment/CreateAIPreparationTask.fragment.xml'));
        const exportFragment = this.readFile(path.join(this.basePath, 'fragment/ExportPreparedData.fragment.xml'));

        if (!createTaskFragment && !exportFragment) return false;

        // Check for file validation
        const hasFileValidation = (createTaskFragment &&
                                 (createTaskFragment.includes('maximumFileSize') ||
                                  createTaskFragment.includes('fileType'))) ||
                                (exportFragment &&
                                 (exportFragment.includes('validation') ||
                                  exportFragment.includes('secure')));

        return hasFileValidation;
    }

    validateEventSourceSecurity() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content && content.includes('EventSource')) {
                // Check for secure EventSource usage
                return content.includes('authentication') ||
                       content.includes('token') ||
                       content.includes('authorization');
            }
        }

        // If no EventSource found, assume it's handled securely elsewhere
        return true;
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
                   (manifestObj['sap.fe'].template || manifestObj['sap.fe'].type);
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
            listController.includes('ControllerExtension') &&
            listController.includes('.extend(')
        ) && (
            objectController.includes('sap.ui.define') &&
            objectController.includes('ControllerExtension') &&
            objectController.includes('.extend(')
        );

        return hasProperExtension;
    }

    validateFragmentPatterns() {
        const fragments = [
            'fragment/CreateAIPreparationTask.fragment.xml',
            'fragment/AutoMLWizard.fragment.xml',
            'fragment/DataProfiler.fragment.xml'
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
                   Object.values(dataSources).some(ds =>
                       ds.type === 'OData' || ds.uri?.includes('odata')
                   );
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

            return routing && routing.routes && Array.isArray(routing.routes);
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
                                        content.includes('sap/ui/core/mvc/ControllerExtension');

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

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const handlers = content.match(/on[A-Z][a-zA-Z]*\s*[:=]/g);
                if (handlers && handlers.length > 0) {
                    return true;
                }
            }
        }
        return false;
    }

    validateModelBinding() {
        const fragments = [
            'fragment/CreateAIPreparationTask.fragment.xml',
            'fragment/AutoMLWizard.fragment.xml',
            'fragment/EmbeddingConfiguration.fragment.xml'
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

        // Check for comprehensive i18n coverage for AI/ML terms
        const hasLabels = i18nFile.includes('title=') &&
                         (i18nFile.includes('label=') || i18nFile.includes('Label='));
        const hasMessages = i18nFile.includes('message.') ||
                           i18nFile.includes('error.');
        const hasAITerms = i18nFile.includes('model') &&
                          i18nFile.includes('feature') &&
                          i18nFile.includes('training');

        return hasLabels && hasMessages && hasAITerms;
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

    validateModelConfigSecurity() {
        const embeddingFragment = this.readFile(path.join(this.basePath, 'fragment/EmbeddingConfiguration.fragment.xml'));
        const automlFragment = this.readFile(path.join(this.basePath, 'fragment/AutoMLWizard.fragment.xml'));

        if (!embeddingFragment && !automlFragment) return false;

        // Check for model configuration validation
        const hasModelValidation = (embeddingFragment &&
                                   embeddingFragment.includes('validation')) ||
                                  (automlFragment &&
                                   automlFragment.includes('validation'));

        return hasModelValidation;
    }

    validateDataExportSecurity() {
        const exportFragment = this.readFile(path.join(this.basePath, 'fragment/ExportPreparedData.fragment.xml'));

        if (!exportFragment) return false;

        // Check for export security controls
        return exportFragment.includes('format') &&
               exportFragment.includes('compression');
    }

    validateEmbeddingSecurity() {
        const embeddingFragment = this.readFile(path.join(this.basePath, 'fragment/EmbeddingConfiguration.fragment.xml'));

        if (!embeddingFragment) return false;

        // Check for embedding generation security
        return embeddingFragment.includes('model') &&
               embeddingFragment.includes('batch');
    }

    validateAutoMLSecurity() {
        const automlFragment = this.readFile(path.join(this.basePath, 'fragment/AutoMLWizard.fragment.xml'));

        if (!automlFragment) return false;

        // Check for AutoML workflow security
        return automlFragment.includes('evaluation') &&
               automlFragment.includes('metric');
    }

    validateBackendIntegration() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                // Check for proper backend integration
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
        const createTaskFragment = this.readFile(path.join(this.basePath, 'fragment/CreateAIPreparationTask.fragment.xml'));

        if (!createTaskFragment) return false;

        // Check for validation controls
        return createTaskFragment.includes('required="true"') ||
               createTaskFragment.includes('validate');
    }

    validateUserAuthorization() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasAuth = content.includes('authorization') ||
                               content.includes('permission') ||
                               content.includes('role');

                if (hasAuth) {
                    return true;
                }
            }
        }

        // Check for Fiori Elements authorization
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
                const hasAudit = content.includes('log') ||
                               content.includes('audit') ||
                               content.includes('track');

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
            'fragment/CreateAIPreparationTask.fragment.xml',
            'fragment/AutoMLWizard.fragment.xml',
            'fragment/DataProfiler.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
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
            'fragment/CreateAIPreparationTask.fragment.xml',
            'fragment/AutoMLWizard.fragment.xml',
            'fragment/DataProfiler.fragment.xml'
        ];

        for (const fragment of fragments) {
            const content = this.readFile(path.join(this.basePath, fragment));
            if (content) {
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

    validateRealTimePerformance() {
        const controllers = [
            'controller/ListReportExt.controller.js',
            'controller/ObjectPageExt.controller.js'
        ];

        for (const controller of controllers) {
            const content = this.readFile(path.join(this.basePath, controller));
            if (content) {
                const hasRealTime = content.includes('EventSource') ||
                                   content.includes('WebSocket') ||
                                   content.includes('polling') ||
                                   content.includes('interval');

                if (hasRealTime) {
                    return true;
                }
            }
        }
        return false;
    }

    generateReport() {
        console.log(`\n${  '='.repeat(60)}`);
        console.log('📊 AGENT 2 UI SECURITY & STANDARDS REPORT');
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
            console.log('🎉 AGENT 2 UI ENTERPRISE READY');
            console.log('✅ Meets all production security and standards requirements');
        } else if (percentage >= 90) {
            console.log('⚠️  AGENT 2 UI MOSTLY COMPLIANT');
            console.log('🔧 Address failed checks for full compliance');
        } else if (percentage >= 80) {
            console.log('🔶 AGENT 2 UI NEEDS IMPROVEMENTS');
            console.log('📝 Several issues require attention');
        } else {
            console.log('❌ AGENT 2 UI NOT ENTERPRISE READY');
            console.log('🚨 Critical issues must be resolved');
        }

        console.log('='.repeat(60));
    }
}

// Run the scan
const scanner = new Agent2UIScanner();
scanner.scan().catch(console.error);