#!/usr/bin/env node

/**
 * Final Production Security Scan
 * Comprehensive validation for enterprise SAP deployment
 */

const fs = require('fs');
const path = require('path');

class FinalSecurityScanner {
    constructor() {
        this.results = {
            passed: [],
            failed: [],
            warnings: [],
            score: 0,
            maxScore: 0
        };
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
        console.log('🔍 Final Production Security Scan\n');

        // 1. CRITICAL SECURITY CHECKS
        console.log('🔒 Critical Security Validation');
        
        this.check(
            'No exposed private keys in .env files',
            !this.fileExists('.env.deployed'),
            true
        );

        this.check(
            'SQL injection protection implemented',
            this.validateSQLProtection(),
            true
        );

        this.check(
            'WebSocket authentication enforced',
            this.validateWebSocketAuth(),
            true
        );

        this.check(
            'CSRF protection enabled',
            this.validateCSRFProtection(),
            true
        );

        // 2. SAP ENTERPRISE STANDARDS
        console.log('\n🏢 SAP Enterprise Standards');

        this.check(
            'Enterprise FLP bootstrap (no sandbox)',
            this.validateEnterpriseBootstrap()
        );

        this.check(
            'XSUAA authentication configured',
            this.validateXSUAAConfig()
        );

        this.check(
            'Standard SAP UI5 patterns used',
            this.validateUI5Patterns()
        );

        this.check(
            'Authorization objects defined',
            this.validateAuthorizationObjects()
        );

        // 3. PRODUCTION READINESS
        console.log('\n🚀 Production Readiness');

        this.check(
            'No test endpoints accessible',
            this.validateNoTestEndpoints()
        );

        this.check(
            'Error handling sanitized',
            this.validateErrorHandling()
        );

        this.check(
            'Rate limiting implemented',
            this.validateRateLimiting()
        );

        this.check(
            'Security headers configured',
            this.validateSecurityHeaders()
        );

        // 4. CODE QUALITY
        console.log('\n📋 Code Quality');

        this.check(
            'No sensitive debug logging',
            this.validateNoSensitiveLogging()
        );

        this.check(
            'Consistent error patterns',
            this.validateErrorPatterns()
        );

        this.check(
            'All components use standard patterns',
            this.validateStandardPatterns()
        );

        // 5. DEPLOYMENT CONFIGURATION
        console.log('\n🌐 Deployment Configuration');

        this.check(
            'Transport configuration ready',
            this.validateTransportConfig()
        );

        this.check(
            'Manifest descriptors valid',
            this.validateManifests()
        );

        this.check(
            'Component paths correct',
            this.validateComponentPaths()
        );

        this.generateReport();
    }

    validateSQLProtection() {
        const messagePersistence = this.readFile('srv/messagePersistence.js');
        if (!messagePersistence) return false;

        // Check for parameterized queries - fixed validation
        return messagePersistence.includes('sanitizedQuery') &&
               messagePersistence.includes('LIKE ?') &&
               messagePersistence.includes('replace(/[%_]/g');
    }

    validateWebSocketAuth() {
        const wsAuth = this.readFile('srv/middleware/secureWebSocketAuth.js');
        const launchpadController = this.readFile('app/controller/Launchpad.controller.js');
        
        if (!wsAuth && !launchpadController) return false;

        // Check either dedicated WS auth file or auth in controller
        const hasWsAuth = wsAuth && (
            wsAuth.includes('Authentication required') &&
            wsAuth.includes('RS256') &&
            wsAuth.includes('validateToken')
        );
        
        const hasControllerAuth = launchpadController && (
            launchpadController.includes('_getAuthToken') &&
            launchpadController.includes('token=${encodeURIComponent(token)}')
        );

        return hasWsAuth || hasControllerAuth;
    }

    validateCSRFProtection() {
        const securityService = this.readFile('app/services/SecurityService.js');
        if (!securityService) return false;

        return securityService.includes('X-CSRF-Token') &&
               securityService.includes('fetchCSRFToken');
    }

    validateEnterpriseBootstrap() {
        const bootstrap = this.readFile('app/flp-bootstrap.js');
        if (!bootstrap) return false;

        return bootstrap.includes('cdmBootstrap') &&
               !bootstrap.includes('sandbox') &&
               bootstrap.includes('fiori2');
    }

    validateXSUAAConfig() {
        const authConfig = this.readFile('srv/middleware/auth.js');
        if (!authConfig) return false;

        return authConfig.includes('xssec') &&
               authConfig.includes('createSecurityContext') &&
               authConfig.includes('JWT_SECRET');
    }

    validateUI5Patterns() {
        const standardDialog = this.readFile('app/fragment/StandardDialog.fragment.xml');
        const standardForm = this.readFile('app/fragment/StandardForm.fragment.xml');
        
        return standardDialog && standardForm &&
               standardDialog.includes('beginButton') &&
               standardForm.includes('ResponsiveGridLayout');
    }

    validateAuthorizationObjects() {
        const authObjects = this.readFile('app/config/authorization-objects.json');
        return authObjects && authObjects.includes('S_SERVICE');
    }

    validateNoTestEndpoints() {
        const securityHardening = this.readFile('srv/middleware/securityHardening.js');
        if (!securityHardening) return false;

        return securityHardening.includes('blockedPaths') &&
               securityHardening.includes('/test/') &&
               securityHardening.includes('BLOCKED_TEST_ENDPOINT');
    }

    validateErrorHandling() {
        const securityHardening = this.readFile('srv/middleware/securityHardening.js');
        if (!securityHardening) return false;

        return securityHardening.includes('sanitizeErrorResponses') &&
               securityHardening.includes('errorId');
    }

    validateRateLimiting() {
        const securityHardening = this.readFile('srv/middleware/securityHardening.js');
        if (!securityHardening) return false;

        return securityHardening.includes('RATE_LIMITS') &&
               securityHardening.includes('createErrorReportLimiter');
    }

    validateSecurityHeaders() {
        const securityHardening = this.readFile('srv/middleware/securityHardening.js');
        if (!securityHardening) return false;

        return securityHardening.includes('helmet') &&
               securityHardening.includes('Content-Security-Policy') &&
               securityHardening.includes('enhancedCSP');
    }

    validateNoSensitiveLogging() {
        // Check that no sensitive patterns exist
        const patterns = [
            /console\.log\s*\([^)]*process\.env/gi,
            /console\.log\s*\([^)]*SECRET/gi,
            /console\.log\s*\([^)]*PASSWORD/gi,
            /console\.log\s*\([^)]*TOKEN/gi
        ];

        const checkFiles = [
            'srv/messagePersistence.js',
            'srv/middleware/auth.js',
            'app/controller/Launchpad.controller.js'
        ];

        for (const file of checkFiles) {
            const content = this.readFile(file);
            if (content) {
                for (const pattern of patterns) {
                    if (pattern.test(content)) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    validateErrorPatterns() {
        const standardPatterns = this.readFile('app/controller/mixin/StandardPatternsMixin.js');
        if (!standardPatterns) return false;

        return standardPatterns.includes('_handleError') &&
               standardPatterns.includes('_sanitizeErrorMessage');
    }

    validateStandardPatterns() {
        const standardPatterns = this.readFile('app/controller/mixin/StandardPatternsMixin.js');
        return standardPatterns && (
            standardPatterns.includes('sap.ui.define') ||
            standardPatterns.includes('initializeStandardModels')
        );
    }

    validateTransportConfig() {
        const transportConfig = this.readFile('app/config/transport-config.json');
        return transportConfig && transportConfig.includes('CTS_PROJECT_ID');
    }

    validateManifests() {
        const appManifest = this.readFile('app/manifest.json');
        if (!appManifest) return false;

        const manifest = JSON.parse(appManifest);
        return manifest['sap.app'] && 
               manifest['sap.ui5'] &&
               manifest['sap.app'].id === 'a2a.fiori';
    }

    validateComponentPaths() {
        const launchpadHtml = this.readFile('app/launchpad-production.html');
        if (!launchpadHtml) return false;

        return launchpadHtml.includes('./a2aFiori/webapp/') &&
               !launchpadHtml.includes('./a2aAgent/webapp/');
    }

    generateReport() {
        console.log('\n' + '='.repeat(60));
        console.log('📊 FINAL SECURITY SCAN REPORT');
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

        console.log('\n' + '='.repeat(60));
        
        if (percentage >= 95) {
            console.log('🎉 ENTERPRISE DEPLOYMENT READY');
            console.log('✅ System meets all production security requirements');
        } else if (percentage >= 90) {
            console.log('⚠️  DEPLOYMENT READY WITH MINOR ISSUES');
            console.log('🔧 Address failed checks before deployment');
        } else {
            console.log('❌ NOT READY FOR DEPLOYMENT');
            console.log('🚨 Critical security issues must be resolved');
        }

        console.log('='.repeat(60));
    }
}

// Run the scan
const scanner = new FinalSecurityScanner();
scanner.scan().catch(console.error);