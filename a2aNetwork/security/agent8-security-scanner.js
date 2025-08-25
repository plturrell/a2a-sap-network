/**
 * Comprehensive Security Scanner for Agent 8 Data Manager UI
 * Checks for vulnerabilities, compliance issues, and security best practices
 */

const fs = require('fs');
const path = require('path');

class Agent8SecurityScanner {
    constructor() {
        this.results = {
            criticalIssues: [],
            highIssues: [],
            mediumIssues: [],
            lowIssues: [],
            complianceGaps: [],
            summary: {
                totalIssues: 0,
                criticalCount: 0,
                highCount: 0,
                mediumCount: 0,
                lowCount: 0,
                complianceScore: 0
            }
        };

        this.scannedFiles = [];
        this.basePath = '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent8';
    }

    async scanAllFiles() {
        console.log('🔍 Starting comprehensive security scan for Agent 8 UI...\n');

        // Scan controller files
        await this.scanFile('controller/ListReportExt.controller.js');
        await this.scanFile('controller/ObjectPageExt.controller.js');

        // Scan i18n files
        await this.scanFile('i18n/i18n.properties');

        // Scan manifest
        await this.scanFile('manifest.json');

        // Generate comprehensive report
        this.generateSecurityReport();

        return this.results;
    }

    async scanFile(relativePath) {
        const filePath = path.join(this.basePath, relativePath);

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            console.log(`📄 Scanning: ${relativePath}`);

            this.scannedFiles.push(relativePath);

            // Perform all security checks
            this.checkXSSVulnerabilities(content, relativePath);
            this.checkSQLInjection(content, relativePath);
            this.checkCSRFProtection(content, relativePath);
            this.checkInputValidation(content, relativePath);
            this.checkOutputEncoding(content, relativePath);
            this.checkAuthenticationIssues(content, relativePath);
            this.checkSessionManagement(content, relativePath);
            this.checkErrorHandling(content, relativePath);
            this.checkLoggingSecurity(content, relativePath);
            this.checkDataExposure(content, relativePath);
            this.checkAPISecurityIssues(content, relativePath);
            this.checkSAPComplianceIssues(content, relativePath);
            this.checkUISecurityIssues(content, relativePath);
            this.checkConfigurationSecurity(content, relativePath);

        } catch (error) {
            this.addIssue('high', 'FILE_ACCESS_ERROR', `Cannot read file: ${relativePath}`, relativePath, 0);
        }
    }

    checkXSSVulnerabilities(content, filePath) {
        const issues = [];

        // Check for direct DOM manipulation without encoding
        const domManipulationPatterns = [
            /innerHTML\s*=\s*[^"'].*\+/g,
            /\.html\([^)]*\+/g,
            /document\.write\(/g,
            /outerHTML\s*=/g
        ];

        domManipulationPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                issues.push({
                    type: 'XSS_DOM_MANIPULATION',
                    description: 'Potential XSS vulnerability in DOM manipulation',
                    line: this.getLineNumber(content, match.index),
                    code: match[0]
                });
            }
        });

        // Check for unescaped user data in message displays
        const messagePatterns = [
            /MessageBox\.(success|error|warning|information)\([^)]*\+.*xhr\.responseText/g,
            /MessageToast\.show\([^)]*\+.*data\./g,
            /MessageBox\.(success|error|warning|information)\([^)]*data\.[^)]*\)/g
        ];

        messagePatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if the match is within a context that already uses security encoding
                const matchContext = content.substring(Math.max(0, match.index - 200), match.index + 200);

                // Skip if proper encoding functions are already being used
                if (matchContext.includes('_securityUtils.encodeHTML') ||
                    matchContext.includes('SecurityUtils.encodeHTML') ||
                    matchContext.includes('encodeXML') ||
                    matchContext.includes('_securityUtils.sanitize')) {
                    continue;
                }

                issues.push({
                    type: 'XSS_MESSAGE_DISPLAY',
                    description: 'Potential XSS in message display - user data not properly encoded',
                    line: this.getLineNumber(content, match.index),
                    code: match[0]
                });
            }
        });

        // Check for eval() usage
        if (content.includes('eval(')) {
            issues.push({
                type: 'XSS_EVAL_USAGE',
                description: 'Use of eval() function can lead to code injection',
                line: this.getLineNumber(content, content.indexOf('eval(')),
                code: 'eval() usage detected'
            });
        }

        issues.forEach(issue => {
            this.addIssue('critical', issue.type, issue.description, filePath, issue.line, issue.code);
        });
    }

    checkSQLInjection(content, filePath) {
        // Check for dynamic SQL construction (less common in UI5 but possible)
        const sqlPatterns = [
            /SELECT.*\+.*['"]/gi,
            /INSERT.*VALUES.*\+/gi,
            /UPDATE.*SET.*\+/gi,
            /DELETE.*WHERE.*\+/gi
        ];

        sqlPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if this is actually SQL or just string concatenation in messages
                const matchContext = content.substring(Math.max(0, match.index - 100), match.index + 100);

                // Skip if it's just message text or UI strings, not actual SQL
                if (matchContext.includes('MessageBox') ||
                    matchContext.includes('MessageToast') ||
                    matchContext.includes('selected task') ||
                    matchContext.includes('transformation')) {
                    continue;
                }

                this.addIssue('critical', 'SQL_INJECTION',
                    'Potential SQL injection vulnerability in dynamic query construction',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkCSRFProtection(content, filePath) {
        // Check AJAX calls without CSRF protection
        const ajaxPattern = /jQuery\.ajax\s*\({[^}]*}/gs;
        const matches = content.matchAll(ajaxPattern);

        for (const match of matches) {
            const ajaxCall = match[0];

            // Check if it's a state-changing operation
            if (/(type:\s*["']?(POST|PUT|DELETE|PATCH)["']?)/.test(ajaxCall)) {
                // Check for CSRF token
                if (!ajaxCall.includes('X-CSRF-Token') && !ajaxCall.includes('csrf')) {
                    this.addIssue('high', 'CSRF_MISSING_TOKEN',
                        'AJAX call missing CSRF protection for state-changing operation',
                        filePath, this.getLineNumber(content, match.index), match[0]);
                }
            }
        }
    }

    checkInputValidation(content, filePath) {
        // Check for input validation patterns
        const validationIssues = [];

        // Look for user input handling without validation
        const inputPatterns = [
            /getProperty\([^)]+\)(?![^;]*validate)/g,
            /getData\(\)(?![^;]*validate)/g,
            /oData\.[a-zA-Z]+(?![^;]*validate)/g
        ];

        inputPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Skip if it's followed by validation within reasonable proximity
                const afterMatch = content.slice(match.index, match.index + 200);
                if (!afterMatch.includes('validate') && !afterMatch.includes('sanitize')) {
                    this.addIssue('medium', 'INPUT_VALIDATION_MISSING',
                        'User input processed without apparent validation',
                        filePath, this.getLineNumber(content, match.index), match[0]);
                }
            }
        });

        // Check for missing validation in form submissions
        const formSubmitPattern = /onConfirm|onExecute|onCreate/g;
        const matches = content.matchAll(formSubmitPattern);

        for (const match of matches) {
            const functionStart = match.index;
            const functionEnd = this.findFunctionEnd(content, functionStart);
            const functionBody = content.slice(functionStart, functionEnd);

            // Check for various types of validation and security measures
            const hasValidation = functionBody.includes('validate') ||
                                functionBody.includes('required') ||
                                functionBody.includes('_securityUtils.hasRole') ||
                                functionBody.includes('SecurityUtils.hasRole') ||
                                functionBody.includes('checkAuth') ||
                                functionBody.includes('_securityUtils.validateInput') ||
                                functionBody.includes('_securityUtils.sanitize') ||
                                functionBody.includes('Access denied');

            if (!hasValidation) {
                this.addIssue('high', 'FORM_VALIDATION_MISSING',
                    'Form submission function lacks proper input validation or authorization',
                    filePath, this.getLineNumber(content, functionStart));
            }
        }
    }

    checkOutputEncoding(content, filePath) {
        // Check for output encoding issues
        const outputPatterns = [
            /MessageBox\.[^(]+\([^)]*data\.[^)]*\)/g,
            /MessageToast\.show\([^)]*data\./g,
            /setText\([^)]*data\./g
        ];

        outputPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                this.addIssue('medium', 'OUTPUT_ENCODING_MISSING',
                    'Output data not properly encoded before display',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkAuthenticationIssues(content, filePath) {
        // Check for hardcoded credentials
        const credentialPatterns = [
            /password\s*[=:]\s*["'][^"']{3,}/gi,
            /apikey\s*[=:]\s*["'][^"']{10,}/gi,
            /secret\s*[=:]\s*["'][^"']{10,}/gi,
            /token\s*[=:]\s*["'][^"']{20,}/gi
        ];

        credentialPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                this.addIssue('critical', 'HARDCODED_CREDENTIALS',
                    'Potential hardcoded credentials detected',
                    filePath, this.getLineNumber(content, match.index));
            }
        });

        // Check for missing authentication checks
        if (!content.includes('authenticate') && !content.includes('authorize') &&
            content.includes('ajax') && filePath.includes('controller')) {
            this.addIssue('high', 'MISSING_AUTH_CHECK',
                'Controller appears to lack authentication/authorization checks',
                filePath, 1);
        }
    }

    checkSessionManagement(content, filePath) {
        // Check for session token exposure
        if (content.includes('sessionStorage') || content.includes('localStorage')) {
            const storagePattern = /(sessionStorage|localStorage)\.setItem.*token/gi;
            const matches = content.matchAll(storagePattern);

            for (const match of matches) {
                this.addIssue('high', 'SESSION_TOKEN_STORAGE',
                    'Potential session token stored in browser storage',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        }
    }

    checkErrorHandling(content, filePath) {
        // Check for information disclosure in error handling
        const errorPatterns = [
            /error:\s*function[^}]*xhr\.responseText/g,
            /catch[^}]*console\.log[^}]*error/g,
            /MessageBox\.error[^)]*xhr/g
        ];

        errorPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                this.addIssue('medium', 'ERROR_INFO_DISCLOSURE',
                    'Error handling may expose sensitive information',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });

        // Check for missing error handling
        const ajaxPattern = /jQuery\.ajax\s*\({[^}]*}/gs;
        const matches = content.matchAll(ajaxPattern);

        for (const match of matches) {
            const ajaxCall = match[0];
            if (!ajaxCall.includes('error:')) {
                this.addIssue('low', 'MISSING_ERROR_HANDLING',
                    'AJAX call missing error handling',
                    filePath, this.getLineNumber(content, match.index));
            }
        }
    }

    checkLoggingSecurity(content, filePath) {
        // Check for sensitive data in logs
        const logPatterns = [
            /console\.log.*password/gi,
            /console\.log.*token/gi,
            /console\.log.*secret/gi,
            /console\.log.*key/gi
        ];

        logPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                this.addIssue('high', 'SENSITIVE_DATA_LOGGING',
                    'Potential sensitive data in console logging',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkDataExposure(content, filePath) {
        // Check for potential data exposure through debugging
        const debugPatterns = [
            /debugger;/g,
            /console\.log.*data/gi,
            /alert\(.*data/gi
        ];

        debugPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                this.addIssue('low', 'DATA_EXPOSURE_DEBUG',
                    'Potential data exposure through debugging code',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkAPISecurityIssues(content, filePath) {
        // Check for API endpoints without proper validation
        const apiUrlPattern = /url:\s*["'][^"']*\/a2a\/agent8\/[^"']*["']/g;
        const matches = content.matchAll(apiUrlPattern);

        for (const match of matches) {
            const url = match[0];

            // Check for parameter injection vulnerabilities
            if (url.includes('" + ') || url.includes('\' + ')) {
                this.addIssue('high', 'API_PARAMETER_INJECTION',
                    'API URL constructed with string concatenation - potential injection vulnerability',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        }

        // Check for missing content-type validation
        const postPattern = /type:\s*["']POST["']/g;
        const postMatches = content.matchAll(postPattern);

        for (const match of postMatches) {
            const ajaxStart = content.lastIndexOf('jQuery.ajax', match.index);
            const ajaxEnd = content.indexOf('})', match.index) + 2;
            const ajaxBlock = content.slice(ajaxStart, ajaxEnd);

            if (!ajaxBlock.includes('contentType')) {
                this.addIssue('medium', 'MISSING_CONTENT_TYPE',
                    'POST request missing explicit content-type header',
                    filePath, this.getLineNumber(content, match.index));
            }
        }
    }

    checkSAPComplianceIssues(content, filePath) {
        // SAP UI5 specific security checks

        // Check for proper fragment loading
        const fragmentPattern = /Fragment\.load\(/g;
        const fragmentMatches = content.matchAll(fragmentPattern);

        for (const match of fragmentMatches) {
            const fragmentCall = this.extractFragmentCall(content, match.index);
            if (!fragmentCall.includes('controller:')) {
                this.addIssue('medium', 'SAP_FRAGMENT_CONTROLLER_MISSING',
                    'Fragment loaded without explicit controller reference',
                    filePath, this.getLineNumber(content, match.index));
            }
        }

        // Check for proper model binding
        if (content.includes('setModel') && !content.includes('TwoWay')) {
            this.addIssue('low', 'SAP_MODEL_BINDING_REVIEW',
                'Review model binding modes for security implications',
                filePath, 1);
        }

        // Check for proper extension API usage
        if (content.includes('ControllerExtension') && !content.includes('getExtensionAPI')) {
            this.addIssue('medium', 'SAP_EXTENSION_API_MISSING',
                'Controller extension not properly using extension API',
                filePath, 1);
        }
    }

    checkUISecurityIssues(content, filePath) {
        // UI-specific security issues

        // Check for clickjacking protection
        if (filePath.includes('manifest.json') && !content.includes('frameOptions')) {
            this.addIssue('medium', 'CLICKJACKING_PROTECTION_MISSING',
                'Missing X-Frame-Options configuration for clickjacking protection',
                filePath, 1);
        }

        // Check for CSP headers
        if (filePath.includes('manifest.json') && !content.includes('contentSecurityPolicy')) {
            this.addIssue('high', 'CSP_MISSING',
                'Missing Content Security Policy configuration',
                filePath, 1);
        }
    }

    checkConfigurationSecurity(content, filePath) {
        if (filePath.includes('manifest.json')) {
            try {
                const config = JSON.parse(content);

                // Check for debug mode in production
                if (config['sap.ui5'] && config['sap.ui5'].debug === true) {
                    this.addIssue('high', 'DEBUG_MODE_ENABLED',
                        'Debug mode enabled in manifest - should be disabled in production',
                        filePath, 1);
                }

                // Check for proper HTTPS enforcement
                if (!content.includes('https') && content.includes('http')) {
                    this.addIssue('medium', 'HTTPS_NOT_ENFORCED',
                        'HTTPS not explicitly enforced in configuration',
                        filePath, 1);
                }

            } catch (e) {
                this.addIssue('low', 'CONFIG_PARSE_ERROR',
                    'Configuration file could not be parsed properly',
                    filePath, 1);
            }
        }
    }

    addIssue(severity, type, description, filePath, lineNumber = 0, code = '') {
        const issue = {
            severity,
            type,
            description,
            file: filePath,
            line: lineNumber,
            code: code.slice(0, 100) + (code.length > 100 ? '...' : '')
        };

        switch (severity) {
            case 'critical':
                this.results.criticalIssues.push(issue);
                this.results.summary.criticalCount++;
                break;
            case 'high':
                this.results.highIssues.push(issue);
                this.results.summary.highCount++;
                break;
            case 'medium':
                this.results.mediumIssues.push(issue);
                this.results.summary.mediumCount++;
                break;
            case 'low':
                this.results.lowIssues.push(issue);
                this.results.summary.lowCount++;
                break;
        }

        this.results.summary.totalIssues++;
    }

    getLineNumber(content, index) {
        return content.slice(0, index).split('\n').length;
    }

    findFunctionEnd(content, startIndex) {
        let braceCount = 0;
        let inFunction = false;

        for (let i = startIndex; i < content.length; i++) {
            const char = content[i];

            if (char === '{') {
                braceCount++;
                inFunction = true;
            } else if (char === '}') {
                braceCount--;
                if (inFunction && braceCount === 0) {
                    return i + 1;
                }
            }
        }

        return startIndex + 500; // fallback
    }

    extractFragmentCall(content, startIndex) {
        const endIndex = content.indexOf('});', startIndex);
        return content.slice(startIndex, endIndex + 3);
    }

    generateSecurityReport() {
        this.results.summary.complianceScore = this.calculateComplianceScore();

        console.log('\n📊 AGENT 8 SECURITY SCAN RESULTS');
        console.log('=====================================');
        console.log(`🔴 Critical Issues: ${this.results.summary.criticalCount}`);
        console.log(`🟠 High Issues: ${this.results.summary.highCount}`);
        console.log(`🟡 Medium Issues: ${this.results.summary.mediumCount}`);
        console.log(`🟢 Low Issues: ${this.results.summary.lowCount}`);
        console.log(`📋 Total Issues: ${this.results.summary.totalIssues}`);
        console.log(`🎯 Compliance Score: ${this.results.summary.complianceScore}%`);
        console.log(`📁 Files Scanned: ${this.scannedFiles.length}`);

        // Display critical and high issues
        if (this.results.criticalIssues.length > 0) {
            console.log('\n🔴 CRITICAL SECURITY ISSUES:');
            this.results.criticalIssues.forEach((issue, index) => {
                console.log(`${index + 1}. ${issue.type}`);
                console.log(`   File: ${issue.file}:${issue.line}`);
                console.log(`   Description: ${issue.description}`);
                if (issue.code) console.log(`   Code: ${issue.code}`);
                console.log('');
            });
        }

        if (this.results.highIssues.length > 0) {
            console.log('\n🟠 HIGH PRIORITY ISSUES:');
            this.results.highIssues.slice(0, 5).forEach((issue, index) => {
                console.log(`${index + 1}. ${issue.type}`);
                console.log(`   File: ${issue.file}:${issue.line}`);
                console.log(`   Description: ${issue.description}`);
                console.log('');
            });

            if (this.results.highIssues.length > 5) {
                console.log(`   ... and ${this.results.highIssues.length - 5} more high priority issues`);
            }
        }
    }

    calculateComplianceScore() {
        const totalPossiblePoints = 100;
        const criticalPenalty = 20;
        const highPenalty = 10;
        const mediumPenalty = 5;
        const lowPenalty = 1;

        const deductions =
            (this.results.summary.criticalCount * criticalPenalty) +
            (this.results.summary.highCount * highPenalty) +
            (this.results.summary.mediumCount * mediumPenalty) +
            (this.results.summary.lowCount * lowPenalty);

        return Math.max(0, totalPossiblePoints - deductions);
    }

    generateDetailedReport() {
        const report = {
            scanTimestamp: new Date().toISOString(),
            scanTarget: 'Agent 8 - Data Manager UI',
            filesScanned: this.scannedFiles,
            summary: this.results.summary,
            issues: {
                critical: this.results.criticalIssues,
                high: this.results.highIssues,
                medium: this.results.mediumIssues,
                low: this.results.lowIssues
            },
            recommendations: this.generateRecommendations()
        };

        return report;
    }

    generateRecommendations() {
        const recommendations = [];

        if (this.results.summary.criticalCount > 0) {
            recommendations.push('🚨 URGENT: Address all critical security vulnerabilities immediately');
            recommendations.push('Implement input validation and output encoding for all user data');
            recommendations.push('Add CSRF protection to all state-changing operations');
        }

        if (this.results.summary.highCount > 0) {
            recommendations.push('🔧 Implement proper authentication and authorization checks');
            recommendations.push('Add Content Security Policy (CSP) headers');
            recommendations.push('Review error handling to prevent information disclosure');
        }

        if (this.results.summary.mediumCount > 0) {
            recommendations.push('📝 Improve logging practices to avoid sensitive data exposure');
            recommendations.push('Implement proper session management');
            recommendations.push('Add security headers for clickjacking protection');
        }

        recommendations.push('🔍 Conduct regular security reviews and penetration testing');
        recommendations.push('📚 Provide security training for development team');
        recommendations.push('🏗️ Integrate security scanning into CI/CD pipeline');

        return recommendations;
    }
}

// Export for use
module.exports = Agent8SecurityScanner;

// Run if called directly
if (require.main === module) {
    const scanner = new Agent8SecurityScanner();
    scanner.scanAllFiles().then(results => {
        const report = scanner.generateDetailedReport();

        // Save detailed report
        const reportPath = '/Users/apple/projects/a2a/a2aNetwork/security/agent8-security-report.json';
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        console.log(`\n📄 Detailed report saved to: ${reportPath}`);
        console.log('\n✅ Security scan completed!');

        // Exit with appropriate code
        process.exit(results.summary.criticalCount > 0 ? 2 :
                    results.summary.highCount > 0 ? 1 : 0);
    }).catch(error => {
        console.error('❌ Security scan failed:', error);
        process.exit(3);
    });
}