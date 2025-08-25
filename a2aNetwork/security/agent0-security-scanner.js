#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 0 (Data Product Agent) Security Scanner
 * Specialized scanner for data product management and Dublin Core metadata vulnerabilities
 */
class Agent0SecurityScanner {
    constructor() {
        this.vulnerabilities = [];
        this.severityLevels = {
            CRITICAL: 'CRITICAL',
            HIGH: 'HIGH',
            MEDIUM: 'MEDIUM',
            LOW: 'LOW',
            WARNING: 'WARNING'
        };
        this.scanStartTime = Date.now();
        this.filesScanned = 0;

        // Data Product-specific vulnerability patterns
        this.dataProductPatterns = {
            // Metadata injection vulnerabilities
            METADATA_INJECTION: {
                patterns: [
                    /eval\s*\(/gi,
                    /new\s+Function\s*\(/gi,
                    /dublinCore\s*=.*eval/gi,
                    /metadata\s*=.*Function/gi,
                    /schema\s*=.*\$\{/gi,
                    /\.setMetadata\s*\([^)]*\+/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'DATA_METADATA_INJECTION',
                message: 'Potential metadata injection vulnerability',
                impact: 'Could allow malicious metadata injection and data manipulation'
            },

            // Schema manipulation vulnerabilities
            SCHEMA_MANIPULATION: {
                patterns: [
                    /schema\s*=.*eval/gi,
                    /\.validateSchema\s*\([^)]*Function/gi,
                    /schemaDefinition\s*=.*\+/gi,
                    /\.updateSchema\s*\([^)]*\$\{/gi,
                    /dataType\s*=.*user/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'DATA_SCHEMA_MANIPULATION',
                message: 'Potential schema manipulation vulnerability',
                impact: 'Could allow unauthorized schema modifications and data corruption'
            },

            // Data lineage tampering
            LINEAGE_TAMPERING: {
                patterns: [
                    /lineage\s*=.*eval/gi,
                    /\.addLineage\s*\([^)]*Function/gi,
                    /lineageData\s*=.*\+/gi,
                    /\.traceData\s*\([^)]*\$\{/gi,
                    /source\s*=.*user/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_LINEAGE_TAMPERING',
                message: 'Data lineage tampering vulnerability',
                impact: 'Could allow falsification of data lineage and provenance tracking'
            },

            // Quality metrics manipulation
            QUALITY_MANIPULATION: {
                patterns: [
                    /qualityScore\s*=.*eval/gi,
                    /\.assessQuality\s*\([^)]*Function/gi,
                    /qualityMetrics\s*=.*\+/gi,
                    /\.updateQuality\s*\([^)]*\$\{/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_QUALITY_MANIPULATION',
                message: 'Quality metrics manipulation vulnerability',
                impact: 'Could allow falsification of data quality assessments'
            },

            // Data product publication vulnerabilities
            PUBLICATION_BYPASS: {
                patterns: [
                    /publish\s*=\s*true/gi,
                    /\.skipValidation\s*\(/gi,
                    /approved\s*=\s*true/gi,
                    /\.overrideApproval\s*\(/gi,
                    /\.forcePublish\s*\(/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_PUBLICATION_BYPASS',
                message: 'Data product publication bypass',
                impact: 'Could allow bypassing of data product approval workflows'
            },

            // Metadata export vulnerabilities
            EXPORT_EXPOSURE: {
                patterns: [
                    /\.exportAll\s*\(/gi,
                    /includePrivate\s*=\s*true/gi,
                    /\.exportSensitive\s*\(/gi,
                    /filterSensitive\s*=\s*false/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'DATA_EXPORT_EXPOSURE',
                message: 'Sensitive data export exposure',
                impact: 'Could allow export of sensitive metadata or private data products'
            }
        };
    }

    scanFile(filePath) {
        console.log(`🔎 Scanning: ${filePath}`);
        this.filesScanned++;

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const lines = content.split('\n');

            // Check for general OWASP vulnerabilities
            this.checkOWASPVulnerabilities(content, filePath, lines);

            // Check for data product-specific vulnerabilities
            this.checkDataProductVulnerabilities(content, filePath, lines);

            // Check for SAP Fiori specific issues
            this.checkSAPFioriCompliance(content, filePath, lines);

        } catch (error) {
            console.error(`❌ Error scanning ${filePath}: ${error.message}`);
        }
    }

    checkOWASPVulnerabilities(content, filePath, lines) {
        // XSS vulnerabilities
        const xssPatterns = [
            { pattern: /innerHTML\s*=\s*[^'"`]+/gi, type: 'XSS', message: 'Potential XSS via innerHTML' },
            { pattern: /document\.write\s*\(/gi, type: 'XSS', message: 'Potential XSS via document.write' },
            { pattern: /\.html\s*\([^)]*\$\{/gi, type: 'XSS', message: 'Potential XSS via jQuery html()' },
            { pattern: /dangerouslySetInnerHTML/gi, type: 'XSS', message: 'Potential XSS via React dangerouslySetInnerHTML' }
        ];

        xssPatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                this.addVulnerability({
                    type: type,
                    severity: this.severityLevels.HIGH,
                    file: filePath,
                    line: lineNumber,
                    code: lines[lineNumber - 1]?.trim() || '',
                    message: message,
                    impact: 'Could allow execution of malicious scripts',
                    fix: 'Use proper output encoding and sanitization'
                });
            }
        });

        // CSRF vulnerabilities
        const csrfPatterns = [
            /\$\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["']/gi,
            /\.post\s*\(/gi,
            /\.put\s*\(/gi,
            /\.delete\s*\(/gi,
            /fetch\s*\([^,]+,\s*\{[^}]*method\s*:\s*["'](POST|PUT|DELETE)["']/gi
        ];

        csrfPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if CSRF token is present nearby or using SecurityUtils
                const surroundingCode = content.substring(Math.max(0, match.index - 200), match.index + 200);
                if (!surroundingCode.includes('X-CSRF-Token') &&
                    !surroundingCode.includes('csrf') &&
                    !surroundingCode.includes('SecurityUtils.secureCallFunction') &&
                    !surroundingCode.includes('refreshSecurityToken')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection',
                        impact: 'Could allow unauthorized state changes',
                        fix: 'Add CSRF token to all state-changing requests'
                    });
                }
            }
        });

        // Insecure connections
        const insecurePatterns = [
            { pattern: /http:\/\//gi, type: 'INSECURE_CONNECTION', message: 'Insecure HTTP connection' },
            { pattern: /ws:\/\//gi, type: 'INSECURE_WEBSOCKET', message: 'Insecure WebSocket connection' }
        ];

        insecurePatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                // Skip comments, strings, examples, and security-fixed code
                if (!code.includes('//') && !code.includes('example') &&
                    !code.includes('SecurityUtils.createSecure') &&
                    !code.includes('wss://') && !code.includes('https://')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could expose sensitive data in transit',
                        fix: 'Use HTTPS/WSS for all connections'
                    });
                }
            }
        });
    }

    checkDataProductVulnerabilities(content, filePath, lines) {
        Object.entries(this.dataProductPatterns).forEach(([vulnType, config]) => {
            config.patterns.forEach(pattern => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';

                    // Skip false positives
                    if (this.isFalsePositive(code, vulnType, filePath)) {
                        continue;
                    }

                    this.addVulnerability({
                        type: config.category,
                        severity: config.severity,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: config.message,
                        impact: config.impact,
                        fix: this.getDataProductFix(vulnType)
                    });
                }
            });
        });
    }

    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            METADATA_INJECTION: [
                /setMetadata\s*\([^)]*["']/gi,  // String literals
                /SecurityUtils\./gi,  // SecurityUtils functions
                /validation\./gi,  // Validation operations
                /sanitized/gi  // Sanitization operations
            ],
            SCHEMA_MANIPULATION: [
                /validateSchema\s*\([^)]*["']/gi,  // String validation
                /SecurityUtils\./gi,  // SecurityUtils functions
                /schemaType\s*=\s*["']/gi  // String assignments
            ],
            PUBLICATION_BYPASS: [
                /publish\s*=\s*true\s*;?\s*\/\//gi,  // Commented code
                /approved\s*=\s*oData\.approved/gi,  // Data binding
                /SecurityUtils\./gi  // SecurityUtils functions
            ]
        };

        if (falsePositivePatterns[vulnType]) {
            const patterns = falsePositivePatterns[vulnType];
            if (patterns.some(pattern => pattern.test(code))) {
                return true;
            }
        }

        // General false positive checks
        if (filePath.includes('SecurityUtils.js')) {
            // SecurityUtils file contains security functions that may trigger patterns
            return code.includes('sanitized') || code.includes('validation') ||
                   code.includes('SecurityUtils') || code.includes('_sanitize');
        }

        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*')) {
            return true;
        }

        return false;
    }

    getDataProductFix(vulnType) {
        const fixes = {
            METADATA_INJECTION: 'Validate and sanitize all metadata before processing',
            SCHEMA_MANIPULATION: 'Use predefined schema templates and validate all schema changes',
            LINEAGE_TAMPERING: 'Implement immutable lineage tracking with cryptographic signatures',
            QUALITY_MANIPULATION: 'Use automated quality assessment tools and audit trails',
            PUBLICATION_BYPASS: 'Enforce approval workflows at the system level',
            EXPORT_EXPOSURE: 'Implement access controls and data classification for exports'
        };
        return fixes[vulnType] || 'Implement proper validation and security controls';
    }

    checkSAPFioriCompliance(content, filePath, lines) {
        // Check for missing i18n
        if (filePath.includes('.controller.js')) {
            const i18nPatterns = [
                { pattern: /MessageToast\.show\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded message in MessageToast' },
                { pattern: /MessageBox\.\w+\s*\(\s*["'][^"']+["']/gi, message: 'Hardcoded message in MessageBox' },
                { pattern: /setText\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded text in UI element' }
            ];

            i18nPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'SAP_STANDARDS',
                        severity: this.severityLevels.LOW,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: message,
                        impact: 'Reduces internationalization support',
                        fix: 'Use i18n resource bundle for all user-facing text'
                    });
                }
            });
        }

        // Check for missing security headers in manifest
        if (filePath.includes('manifest.json')) {
            const requiredHeaders = [
                'Content-Security-Policy',
                'X-Frame-Options',
                'X-Content-Type-Options'
            ];

            requiredHeaders.forEach(header => {
                if (!content.includes(header)) {
                    this.addVulnerability({
                        type: 'SAP_SECURITY',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: 1,
                        code: 'manifest.json',
                        message: `Missing security header: ${header}`,
                        impact: 'Reduces application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
                }
            });
        }
    }

    getLineNumber(content, index) {
        const lines = content.substring(0, index).split('\n');
        return lines.length;
    }

    addVulnerability(vuln) {
        // Avoid duplicates
        const exists = this.vulnerabilities.some(v =>
            v.file === vuln.file &&
            v.line === vuln.line &&
            v.type === vuln.type
        );

        if (!exists) {
            this.vulnerabilities.push(vuln);
        }
    }

    scanDirectory(dirPath) {
        const files = fs.readdirSync(dirPath);

        files.forEach(file => {
            const fullPath = path.join(dirPath, file);
            const stat = fs.statSync(fullPath);

            if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
                this.scanDirectory(fullPath);
            } else if (stat.isFile() && this.shouldScanFile(file)) {
                this.scanFile(fullPath);
            }
        });
    }

    shouldScanFile(filename) {
        const extensions = ['.js', '.xml', '.json', '.html'];
        return extensions.some(ext => filename.endsWith(ext));
    }

    generateReport() {
        const scanDuration = (Date.now() - this.scanStartTime) / 1000;
        const criticalCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.CRITICAL).length;
        const highCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.HIGH).length;
        const mediumCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.MEDIUM).length;
        const lowCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.LOW).length;
        const warningCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.WARNING).length;

        console.log(`\n${  '='.repeat(80)}`);
        console.log('🔒 AGENT 0 DATA PRODUCT SECURITY SCAN REPORT');
        console.log('='.repeat(80));

        console.log('\n📊 SUMMARY:');
        console.log(`   Files Scanned: ${this.filesScanned}`);
        console.log(`   Scan Duration: ${scanDuration.toFixed(2)}s`);
        console.log(`   Critical Issues: ${criticalCount}`);
        console.log(`   High Issues: ${highCount}`);
        console.log(`   Medium Issues: ${mediumCount}`);
        console.log(`   Low Issues: ${lowCount}`);
        console.log(`   Warnings: ${warningCount}`);

        // Calculate security score
        const totalIssues = criticalCount * 10 + highCount * 5 + mediumCount * 2 + lowCount;
        const maxScore = 100;
        const score = Math.max(0, maxScore - totalIssues);

        console.log(`\n🎯 DATA PRODUCT SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log('   Status: ✅ EXCELLENT - Well secured');
        } else if (score >= 70) {
            console.log('   Status: ⚠️  GOOD - Minor issues to address');
        } else if (score >= 50) {
            console.log('   Status: ⚠️  FAIR - Several issues need attention');
        } else {
            console.log('   Status: ❌ POOR - Significant security improvements needed');
        }

        // Data product-specific findings
        const dataProductIssues = this.vulnerabilities.filter(v => v.type.startsWith('DATA_'));
        if (dataProductIssues.length > 0) {
            console.log('\n🗂️  DATA PRODUCT-SPECIFIC SECURITY FINDINGS:');
            const issueCounts = {};
            dataProductIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });

            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type.replace('DATA_', '')}: ${count} issues`);
            });
        }

        // List vulnerabilities by severity
        if (this.vulnerabilities.length > 0) {
            console.log('\n🚨 VULNERABILITIES FOUND:\n');

            const severityOrder = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING'];
            let issueNumber = 1;

            severityOrder.forEach(severity => {
                const sevVulns = this.vulnerabilities.filter(v => v.severity === severity);
                if (sevVulns.length > 0) {
                    console.log(`${severity} (${sevVulns.length}):\n`);
                    sevVulns.forEach(vuln => {
                        console.log(`${issueNumber}. ${vuln.type} - ${vuln.file}:${vuln.line}`);
                        console.log(`   Description: ${vuln.message}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code.substring(0, 60)}${vuln.code.length > 60 ? '...' : ''}`);
                        console.log(`   Fix: ${vuln.fix}\n`);
                        issueNumber++;
                    });
                }
            });
        }

        // Data product security recommendations
        console.log('💡 AGENT 0 DATA PRODUCT SECURITY RECOMMENDATIONS:\n');
        console.log('1. 🔐 Implement metadata validation');
        console.log('   - Validate all Dublin Core metadata before processing');
        console.log('   - Use schema validation for data product definitions');
        console.log('   - Implement metadata sanitization and encoding');

        console.log('\n2. 🛡️  Secure data lineage tracking');
        console.log('   - Implement cryptographic signatures for lineage data');
        console.log('   - Use immutable audit trails for data provenance');
        console.log('   - Validate lineage data integrity on access');

        console.log('\n3. 🔒 Protect data product workflows');
        console.log('   - Enforce approval workflows for publication');
        console.log('   - Implement access controls for sensitive data products');
        console.log('   - Validate quality metrics and prevent manipulation');

        console.log('\n4. ⚡ Implement export security');
        console.log('   - Apply data classification to control exports');
        console.log('   - Implement audit logging for all export operations');
        console.log('   - Use encryption for sensitive metadata exports');

        console.log(`\n${  '='.repeat(80)}`);
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('='.repeat(80));
    }

    run(targetPath) {
        console.log('🔍 Starting Agent 0 Data Product Security Scan...');
        console.log(`📂 Scanning directory: ${targetPath}\n`);

        if (fs.existsSync(targetPath)) {
            if (fs.statSync(targetPath).isDirectory()) {
                this.scanDirectory(targetPath);
            } else {
                this.scanFile(targetPath);
            }

            this.generateReport();
        } else {
            console.error(`❌ Path not found: ${targetPath}`);
        }
    }
}

// Run the scanner
const scanner = new Agent0SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent0';
scanner.run(targetPath);