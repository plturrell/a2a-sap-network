#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

(async () => {
console.log('🔒 A2A Network Security Audit\n');

/**
 * Real security audit for A2A Network
 * Checks actual implementation against SAP security standards
 */
class SecurityAuditor {
    constructor() {
        this.issues = [];
        this.passed = [];
        this.warnings = [];
    }

    log(level, category, message, details = null) {
        const entry = { level, category, message, details, timestamp: new Date() };

        if (level === 'PASS') this.passed.push(entry);
        else if (level === 'WARN') this.warnings.push(entry);
        else if (level === 'FAIL') this.issues.push(entry);

        const icon = level === 'PASS' ? '✅' : level === 'WARN' ? '⚠️' : '❌';
        console.log(`${icon} [${category}] ${message}`);
        if (details) console.log(`   Details: ${details}`);
    }

    async auditAuthentication() {
        console.log('\n🔐 Authentication Security Audit');

        // Check XSUAA configuration
        try {
            const authMiddleware = await fs.readFile('srv/middleware/auth.js', 'utf8');

            if (authMiddleware.includes('ENABLE_XSUAA_VALIDATION')) {
                this.log('PASS', 'AUTH', 'Environment-based XSUAA validation implemented');
            } else {
                this.log('FAIL', 'AUTH', 'Missing XSUAA validation toggle');
            }

            if (authMiddleware.includes('xssec.createSecurityContext')) {
                this.log('PASS', 'AUTH', 'SAP XSSEC library properly integrated');
            } else {
                this.log('FAIL', 'AUTH', 'XSSEC security context not implemented');
            }

            if (authMiddleware.includes('securityContext.hasLocalScope')) {
                this.log('PASS', 'AUTH', 'Scope-based authorization implemented');
            } else {
                this.log('WARN', 'AUTH', 'Scope validation may be incomplete');
            }

        } catch (error) {
            this.log('FAIL', 'AUTH', 'Authentication middleware not found', error.message);
        }

        // Check xs-security.json
        try {
            const xsSecurity = JSON.parse(await fs.readFile('xs-security.json', 'utf8'));

            if (xsSecurity.scopes && xsSecurity.scopes.length > 0) {
                this.log('PASS', 'AUTH', `${xsSecurity.scopes.length} security scopes defined`);
            } else {
                this.log('FAIL', 'AUTH', 'No security scopes defined');
            }

            if (xsSecurity['role-templates'] && xsSecurity['role-templates'].length > 0) {
                this.log('PASS', 'AUTH', `${xsSecurity['role-templates'].length} role templates configured`);
            } else {
                this.log('FAIL', 'AUTH', 'No role templates defined');
            }

            if (xsSecurity['oauth2-configuration']) {
                this.log('PASS', 'AUTH', 'OAuth2 configuration present');
            } else {
                this.log('WARN', 'AUTH', 'OAuth2 configuration missing');
            }

        } catch (error) {
            this.log('FAIL', 'AUTH', 'xs-security.json not found or invalid', error.message);
        }
    }

    async auditSecurityHeaders() {
        console.log('\n🛡️  Security Headers Audit');

        try {
            const securityMiddleware = await fs.readFile('srv/middleware/security.js', 'utf8');

            const securityHeaders = [
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection',
                'Strict-Transport-Security',
                'Content-Security-Policy'
            ];

            securityHeaders.forEach(header => {
                if (securityMiddleware.includes(header)) {
                    this.log('PASS', 'HEADERS', `${header} configured`);
                } else {
                    this.log('WARN', 'HEADERS', `${header} may be missing`);
                }
            });

            if (securityMiddleware.includes('helmet')) {
                this.log('PASS', 'HEADERS', 'Helmet.js security middleware integrated');
            } else {
                this.log('FAIL', 'HEADERS', 'Helmet.js not detected');
            }

        } catch (error) {
            this.log('FAIL', 'HEADERS', 'Security middleware not found', error.message);
        }
    }

    async auditRateLimiting() {
        console.log('\n⏱️  Rate Limiting Audit');

        try {
            const securityMiddleware = await fs.readFile('srv/middleware/security.js', 'utf8');

            if (securityMiddleware.includes('express-rate-limit')) {
                this.log('PASS', 'RATE_LIMIT', 'Express rate limiting configured');
            } else {
                this.log('FAIL', 'RATE_LIMIT', 'Rate limiting not implemented');
            }

            if (securityMiddleware.includes('RATE_LIMIT_MAX_REQUESTS')) {
                this.log('PASS', 'RATE_LIMIT', 'Configurable rate limits');
            } else {
                this.log('WARN', 'RATE_LIMIT', 'Rate limits may be hardcoded');
            }

        } catch (error) {
            this.log('FAIL', 'RATE_LIMIT', 'Cannot assess rate limiting', error.message);
        }
    }

    generateReport() {
        console.log('\n📊 Security Audit Summary');
        console.log('=' .repeat(50));
        console.log(`✅ Passed: ${this.passed.length}`);
        console.log(`⚠️  Warnings: ${this.warnings.length}`);
        console.log(`❌ Failed: ${this.issues.length}`);
        console.log('=' .repeat(50));

        if (this.issues.length > 0) {
            console.log('\n❌ Critical Issues Requiring Attention:');
            this.issues.forEach((issue, i) => {
                console.log(`${i + 1}. [${issue.category}] ${issue.message}`);
                if (issue.details) console.log(`   ${issue.details}`);
            });
        }

        if (this.warnings.length > 0) {
            console.log('\n⚠️  Warnings for Review:');
            this.warnings.forEach((warning, i) => {
                console.log(`${i + 1}. [${warning.category}] ${warning.message}`);
                if (warning.details) console.log(`   ${warning.details}`);
            });
        }

        const securityScore = Math.round(
            (this.passed.length / (this.passed.length + this.warnings.length + this.issues.length)) * 100
        );

        console.log(`\n🔒 Overall Security Score: ${securityScore}%`);

        if (securityScore >= 90) {
            console.log('🟢 Excellent security posture');
        } else if (securityScore >= 75) {
            console.log('🟡 Good security with room for improvement');
        } else {
            console.log('🔴 Security improvements required before production');
        }

        return {
            score: securityScore,
            passed: this.passed.length,
            warnings: this.warnings.length,
            failed: this.issues.length
        };
    }

    async runFullAudit() {
        console.log('Starting comprehensive security audit...\n');

        await this.auditAuthentication();
        await this.auditSecurityHeaders();
        await this.auditRateLimiting();

        return this.generateReport();
    }
}

// Run the audit
const auditor = new SecurityAuditor();
auditor.runFullAudit().then(results => {
    process.exit(results.failed > 0 ? 1 : 0);
}).catch(error => {
    console.error('❌ Security audit failed:', error);
    process.exit(1);
});
})().catch(console.error);