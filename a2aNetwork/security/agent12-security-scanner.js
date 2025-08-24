#!/usr/bin/env node

/**
 * Agent 12 (Catalog Manager Agent) Security Scanner
 * 
 * Comprehensive security vulnerability scanner specifically designed for
 * Agent 12's catalog management functionality including service discovery,
 * resource registry, metadata management, and search capabilities.
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

class Agent12SecurityScanner {
    constructor() {
        this.vulnerabilities = [];
        this.warnings = [];
        this.info = [];
        this.scannedFiles = 0;
        this.startTime = performance.now();
        
        // Agent 12 specific security patterns for catalog management
        this.catalogSecurityPatterns = {
            // SQL injection risks in catalog queries
            sqlInjection: [
                /SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*\+.*["']/gi,
                /INSERT\s+INTO\s+.*\s+VALUES\s*\(.*\+.*["']/gi,
                /UPDATE\s+.*\s+SET\s+.*=.*\+.*["']/gi,
                /DELETE\s+FROM\s+.*\s+WHERE\s+.*\+.*["']/gi,
                /oModel\.read\([^)]*\+[^)]*['"]/gi,
                /oModel\.create\([^)]*\+[^)]*['"]/gi
            ],
            
            // Catalog-specific XSS patterns
            catalogXSS: [
                /innerHTML\s*=\s*.*entryName/gi,
                /innerHTML\s*=\s*.*description/gi,
                /innerHTML\s*=\s*.*resourceUrl/gi,
                /innerHTML\s*=\s*.*metadata/gi,
                /innerHTML\s*=\s*.*tags/gi,
                /innerHTML\s*=\s*.*keywords/gi,
                /document\.write\s*\(.*entry\./gi,
                /\$\.html\s*\(.*entry\./gi,
                /setText\s*\(.*entry\..*\)/gi,
                /setHtml\s*\(.*entry\..*\)/gi
            ],
            
            // Insecure WebSocket/EventSource for catalog updates
            insecureConnections: [
                /new\s+WebSocket\s*\(\s*["']ws:\/\//gi,
                /new\s+EventSource\s*\(\s*["']http:\/\//gi,
                /WebSocket\s*\(\s*["']ws:\/\/[^"']*catalog/gi,
                /EventSource\s*\(\s*["']http:\/\/[^"']*catalog/gi,
                /ws:\/\/.*catalog.*updates/gi,
                /http:\/\/.*catalog.*stream/gi
            ],
            
            // Missing CSRF protection in catalog operations
            csrfMissing: [
                /callFunction\s*\(\s*["']\/RegisterResource/gi,
                /callFunction\s*\(\s*["']\/ValidateEntry/gi,
                /callFunction\s*\(\s*["']\/PublishEntry/gi,
                /callFunction\s*\(\s*["']\/IndexResource/gi,
                /callFunction\s*\(\s*["']\/DiscoverDependencies/gi,
                /callFunction\s*\(\s*["']\/SyncRegistryEntry/gi,
                /callFunction\s*\(\s*["']\/StartResourceDiscovery/gi,
                /callFunction\s*\(\s*["']\/ValidateCatalogEntries/gi,
                /callFunction\s*\(\s*["']\/PublishCatalogEntries/gi
            ],
            
            // Catalog URL validation risks
            urlValidation: [
                /resourceUrl.*=.*input/gi,
                /documentationUrl.*=.*input/gi,
                /healthCheckUrl.*=.*input/gi,
                /swaggerUrl.*=.*input/gi,
                /new\s+URL\s*\(\s*[^)]*input/gi,
                /location\.href\s*=.*resourceUrl/gi,
                /window\.open\s*\(.*resourceUrl/gi
            ],
            
            // Metadata injection risks
            metadataInjection: [
                /metadataValue.*innerHTML/gi,
                /metadataKey.*innerHTML/gi,
                /JSON\.parse\s*\(.*metadataValue/gi,
                /eval\s*\(.*metadata/gi,
                /Function\s*\(.*metadata/gi,
                /new\s+Function\s*\(.*metadata/gi
            ],
            
            // Resource discovery security risks
            discoveryRisks: [
                /fetch\s*\(.*resourceUrl/gi,
                /XMLHttpRequest.*resourceUrl/gi,
                /\.load\s*\(.*resourceUrl/gi,
                /import\s*\(.*resourceUrl/gi,
                /require\s*\(.*resourceUrl/gi
            ],
            
            // Search query injection
            searchInjection: [
                /search.*\+.*input/gi,
                /query.*\+.*input/gi,
                /filter.*\+.*input/gi,
                /indexOf\s*\(.*input.*\)/gi,
                /match\s*\(.*input.*\)/gi,
                /RegExp\s*\(.*input/gi
            ],
            
            // Registry synchronization security
            registrySync: [
                /sync.*registry.*http:/gi,
                /registry.*url.*http:/gi,
                /external.*registry.*fetch/gi,
                /registry.*connection.*insecure/gi
            ],
            
            // Catalog export/import security
            catalogDataSecurity: [
                /eval\s*\(.*import/gi,
                /Function\s*\(.*export/gi,
                /document\.write\s*\(.*catalog/gi,
                /innerHTML\s*=.*export/gi,
                /outerHTML\s*=.*import/gi
            ]
        };
        
        this.sensitiveOperations = [
            'RegisterResource', 'ValidateEntry', 'PublishEntry', 'IndexResource',
            'DiscoverDependencies', 'SyncRegistryEntry', 'StartResourceDiscovery',
            'ValidateCatalogEntries', 'PublishCatalogEntries', 'GetCatalogStatistics',
            'GetDiscoveryMethods', 'GetRegistryConfigurations', 'GetServiceCategories',
            'GetSearchIndexes', 'GetMetadataProperties'
        ];
        
        this.catalogPatterns = [
            'entryName', 'resourceUrl', 'metadataValue', 'metadataKey', 'tags',
            'keywords', 'description', 'category', 'provider', 'version',
            'apiEndpoint', 'documentationUrl', 'healthCheckUrl', 'swaggerUrl'
        ];
    }

    async scanDirectory(dirPath) {
        console.log('\n🔍 Starting Agent 12 Catalog Manager Security Scan...');
        console.log(`📂 Scanning directory: ${dirPath}\n`);
        
        try {
            await this.scanFiles(dirPath);
            this.generateReport();
        } catch (error) {
            console.error(`❌ Scan failed: ${error.message}`);
            process.exit(1);
        }
    }

    async scanFiles(dirPath) {
        const files = fs.readdirSync(dirPath, { withFileTypes: true });
        
        for (const file of files) {
            const fullPath = path.join(dirPath, file.name);
            
            if (file.isDirectory()) {
                await this.scanFiles(fullPath);
            } else if (this.isJavaScriptFile(file.name)) {
                await this.scanFile(fullPath);
            }
        }
    }

    isJavaScriptFile(filename) {
        return /\.(js|ts)$/.test(filename) && 
               !filename.includes('.min.') && 
               !filename.includes('test') &&
               !filename.includes('spec');
    }

    async scanFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf8');
        const relativePath = path.relative(process.cwd(), filePath);
        this.scannedFiles++;

        // Skip if not Agent 12 related
        if (!this.isAgent12File(content, filePath)) {
            return;
        }

        console.log(`🔎 Scanning: ${relativePath}`);

        // Catalog-specific security scans
        this.scanCatalogSQLInjection(content, relativePath);
        this.scanCatalogXSS(content, relativePath);
        this.scanInsecureConnections(content, relativePath);
        this.scanCSRFProtection(content, relativePath);
        this.scanURLValidation(content, relativePath);
        this.scanMetadataInjection(content, relativePath);
        this.scanDiscoveryRisks(content, relativePath);
        this.scanSearchInjection(content, relativePath);
        this.scanRegistrySync(content, relativePath);
        this.scanCatalogDataSecurity(content, relativePath);
        
        // General security scans
        this.scanGeneralSecurity(content, relativePath);
        this.scanInputValidation(content, relativePath);
        this.scanAuthenticationChecks(content, relativePath);
        this.scanErrorHandling(content, relativePath);
    }

    isAgent12File(content, filePath) {
        const agent12Indicators = [
            'agent12', 'CatalogEntries', 'RegisterResource', 'ValidateEntry',
            'PublishEntry', 'IndexResource', 'DiscoverDependencies', 'SyncRegistryEntry',
            'ResourceDiscovery', 'CatalogDashboard', 'MetadataEditor', 'ServiceDiscovery',
            'RegistryManager', 'CategoryManager', 'SearchManager', 'catalog/updates',
            'catalog/stream', 'CatalogUtils'
        ];
        
        const checkAgent12Indicator = (indicator) => 
            content.toLowerCase().includes(indicator.toLowerCase()) ||
            filePath.toLowerCase().includes(indicator.toLowerCase());
        
        return agent12Indicators.some(checkAgent12Indicator);
    }

    scanCatalogSQLInjection(content, filePath) {
        const checkSQLInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processSQLInjectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_SQL_INJECTION',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Potential SQL injection vulnerability in catalog query',
                        code: match.trim(),
                        impact: 'Could allow unauthorized access to catalog database',
                        recommendation: 'Use parameterized queries and SecurityUtils.validateSQL()'
                    });
                };
                matches.forEach(processSQLInjectionMatch);
            }
        };
        this.catalogSecurityPatterns.sqlInjection.forEach(checkSQLInjectionPattern);
    }

    scanCatalogXSS(content, filePath) {
        const checkCatalogXSSPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCatalogXSSMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_XSS',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'XSS vulnerability in catalog data display',
                        code: match.trim(),
                        impact: 'Could allow script injection through catalog entries',
                        recommendation: 'Use SecurityUtils.escapeHTML() for catalog data display'
                    });
                };
                matches.forEach(processCatalogXSSMatch);
            }
        };
        this.catalogSecurityPatterns.catalogXSS.forEach(checkCatalogXSSPattern);
    }

    scanInsecureConnections(content, filePath) {
        const checkInsecureConnectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processInsecureConnectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'INSECURE_CATALOG_CONNECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure WebSocket/EventSource for catalog updates',
                        code: match.trim(),
                        impact: 'Catalog update communications not encrypted',
                        recommendation: 'Use WSS/HTTPS for secure catalog communications'
                    });
                };
                matches.forEach(processInsecureConnectionMatch);
            }
        };
        this.catalogSecurityPatterns.insecureConnections.forEach(checkInsecureConnectionPattern);
    }

    scanCSRFProtection(content, filePath) {
        const checkCSRFPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCSRFMatch = (match) => {
                    // Check if CSRF token is present in the context
                    const functionStart = content.indexOf(match);
                    const functionBlock = content.substring(functionStart, functionStart + 500);
                    
                    if (!functionBlock.includes('securityToken') && 
                        !functionBlock.includes('X-CSRF-Token') &&
                        !functionBlock.includes('SecurityUtils.secureCallFunction')) {
                        this.vulnerabilities.push({
                            type: 'CATALOG_CSRF_MISSING',
                            severity: 'HIGH',
                            file: filePath,
                            line: this.getLineNumber(content, match),
                            description: 'Missing CSRF protection for catalog operation',
                            code: match.trim(),
                            impact: 'Catalog operations vulnerable to CSRF attacks',
                            recommendation: 'Use SecurityUtils.secureCallFunction() for catalog operations'
                        });
                    }
                };
                matches.forEach(processCSRFMatch);
            }
        };
        this.catalogSecurityPatterns.csrfMissing.forEach(checkCSRFPattern);
    }

    scanURLValidation(content, filePath) {
        const checkURLValidationPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processURLValidationMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_URL_VALIDATION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insufficient URL validation in catalog resources',
                        code: match.trim(),
                        impact: 'Could allow malicious URLs in catalog entries',
                        recommendation: 'Implement SecurityUtils.validateResourceURL() for all URLs'
                    });
                };
                matches.forEach(processURLValidationMatch);
            }
        };
        this.catalogSecurityPatterns.urlValidation.forEach(checkURLValidationPattern);
    }

    scanMetadataInjection(content, filePath) {
        const checkMetadataInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processMetadataInjectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_METADATA_INJECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Potential code injection through catalog metadata',
                        code: match.trim(),
                        impact: 'Could allow arbitrary code execution via metadata',
                        recommendation: 'Sanitize metadata with SecurityUtils.sanitizeCatalogData()'
                    });
                };
                matches.forEach(processMetadataInjectionMatch);
            }
        };
        this.catalogSecurityPatterns.metadataInjection.forEach(checkMetadataInjectionPattern);
    }

    scanDiscoveryRisks(content, filePath) {
        const checkDiscoveryRisksPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processDiscoveryRisksMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_DISCOVERY_RISK',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe resource discovery operation',
                        code: match.trim(),
                        impact: 'Could expose system to malicious resources',
                        recommendation: 'Validate discovered resources with SecurityUtils.validateDiscoveredResource()'
                    });
                };
                matches.forEach(processDiscoveryRisksMatch);
            }
        };
        this.catalogSecurityPatterns.discoveryRisks.forEach(checkDiscoveryRisksPattern);
    }

    scanSearchInjection(content, filePath) {
        const checkSearchInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processSearchInjectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_SEARCH_INJECTION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Search query injection vulnerability',
                        code: match.trim(),
                        impact: 'Could allow search result manipulation',
                        recommendation: 'Use SecurityUtils.sanitizeSearchQuery() for search inputs'
                    });
                };
                matches.forEach(processSearchInjectionMatch);
            }
        };
        this.catalogSecurityPatterns.searchInjection.forEach(checkSearchInjectionPattern);
    }

    scanRegistrySync(content, filePath) {
        const checkRegistrySyncPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processRegistrySyncMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_REGISTRY_SYNC',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure registry synchronization',
                        code: match.trim(),
                        impact: 'Registry sync operations not secure',
                        recommendation: 'Implement secure registry synchronization with validation'
                    });
                };
                matches.forEach(processRegistrySyncMatch);
            }
        };
        this.catalogSecurityPatterns.registrySync.forEach(checkRegistrySyncPattern);
    }

    scanCatalogDataSecurity(content, filePath) {
        const checkCatalogDataSecurityPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCatalogDataSecurityMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'CATALOG_DATA_SECURITY',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe catalog data handling',
                        code: match.trim(),
                        impact: 'Could allow code injection through catalog import/export',
                        recommendation: 'Sanitize catalog data with SecurityUtils.sanitizeCatalogData()'
                    });
                };
                matches.forEach(processCatalogDataSecurityMatch);
            }
        };
        this.catalogSecurityPatterns.catalogDataSecurity.forEach(checkCatalogDataSecurityPattern);
    }

    scanGeneralSecurity(content, filePath) {
        // Hardcoded credentials
        const credentialPatterns = [
            /password\s*[=:]\s*["'][^"']+["']/gi,
            /api[_-]?key\s*[=:]\s*["'][^"']+["']/gi,
            /secret\s*[=:]\s*["'][^"']+["']/gi,
            /token\s*[=:]\s*["'][^"']+["']/gi
        ];

        const checkCredentialPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCredentialMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'HARDCODED_CREDENTIALS',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Hardcoded credentials detected',
                        code: match.trim(),
                        impact: 'Credentials exposed in source code',
                        recommendation: 'Use secure credential management system'
                    });
                };
                matches.forEach(processCredentialMatch);
            }
        };
        credentialPatterns.forEach(checkCredentialPattern);

        // Unsafe functions
        const unsafeFunctions = [
            /\beval\s*\(/gi,
            /\bnew\s+Function\s*\(/gi,
            /setTimeout\s*\(\s*["'][^"']*["']/gi,
            /setInterval\s*\(\s*["'][^"']*["']/gi
        ];

        const checkUnsafeFunctionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processUnsafeFunctionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'UNSAFE_FUNCTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Use of potentially unsafe function',
                        code: match.trim(),
                        impact: 'Could allow code injection',
                        recommendation: 'Avoid eval() and similar functions'
                    });
                };
                matches.forEach(processUnsafeFunctionMatch);
            }
        };
        unsafeFunctions.forEach(checkUnsafeFunctionPattern);
    }

    scanInputValidation(content, filePath) {
        // Check for catalog-specific input validation
        const checkCatalogPattern = (pattern) => {
            const regex = new RegExp(`${pattern}\\s*[=:]\\s*.*input`, 'gi');
            const matches = content.match(regex);
            if (matches && !content.includes('SecurityUtils.validate') && !content.includes('CatalogUtils.validate')) {
                const processValidationMatch = (match) => {
                    this.warnings.push({
                        type: 'MISSING_CATALOG_VALIDATION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: `Missing validation for catalog field: ${pattern}`,
                        code: match.trim(),
                        recommendation: 'Add SecurityUtils.validateCatalogField() validation'
                    });
                };
                matches.forEach(processValidationMatch);
            }
        };
        this.catalogPatterns.forEach(checkCatalogPattern);
    }

    scanAuthenticationChecks(content, filePath) {
        const checkSensitiveOp = (op) => content.includes(op);
        const sensitiveOps = this.sensitiveOperations.filter(checkSensitiveOp);
        
        const processSensitiveOp = (op) => {
            if (!content.includes('checkAuth') && !content.includes('isAuthenticated') && 
                !content.includes('SecurityUtils.checkCatalogAuth')) {
                this.warnings.push({
                    type: 'MISSING_AUTH_CHECK',
                    severity: 'HIGH',
                    file: filePath,
                    description: `Missing authentication check for catalog operation: ${op}`,
                    recommendation: 'Add SecurityUtils.checkCatalogAuth() before operations'
                });
            }
        };
        sensitiveOps.forEach(processSensitiveOp);
    }

    scanErrorHandling(content, filePath) {
        const errorPatterns = [
            /catch\s*\([^)]*\)\s*\{\s*\}/gi,
            /\.catch\s*\(\s*\)/gi,
            /error[^}]*console\.log/gi
        ];

        const checkErrorPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processErrorMatch = (match) => {
                    this.warnings.push({
                        type: 'POOR_ERROR_HANDLING',
                        severity: 'LOW',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Poor error handling detected',
                        code: match.trim(),
                        recommendation: 'Implement proper error handling with SecurityUtils.logSecureError()'
                    });
                };
                matches.forEach(processErrorMatch);
            }
        };
        errorPatterns.forEach(checkErrorPattern);
    }

    getLineNumber(content, searchString) {
        const lines = content.substring(0, content.indexOf(searchString)).split('\n');
        return lines.length;
    }

    generateReport() {
        const endTime = performance.now();
        const duration = ((endTime - this.startTime) / 1000).toFixed(2);

        console.log(`\n${  '='.repeat(80)}`);
        console.log('🔒 AGENT 12 CATALOG MANAGER SECURITY SCAN REPORT');
        console.log('='.repeat(80));

        // Summary
        const critical = this.vulnerabilities.filter(v => v.severity === 'CRITICAL').length;
        const high = this.vulnerabilities.filter(v => v.severity === 'HIGH').length;
        const medium = this.vulnerabilities.filter(v => v.severity === 'MEDIUM').length;
        const low = this.vulnerabilities.filter(v => v.severity === 'LOW').length;

        console.log('\n📊 SUMMARY:');
        console.log(`   Files Scanned: ${this.scannedFiles}`);
        console.log(`   Scan Duration: ${duration}s`);
        console.log(`   Critical Issues: ${critical}`);
        console.log(`   High Issues: ${high}`);
        console.log(`   Medium Issues: ${medium}`);
        console.log(`   Low Issues: ${low}`);
        console.log(`   Warnings: ${this.warnings.length}`);

        // Risk Score Calculation
        const riskScore = (critical * 10) + (high * 5) + (medium * 2) + (low * 1);
        const maxRisk = this.scannedFiles * 15; // Theoretical max risk per file
        const securityScore = Math.max(0, 100 - Math.min(100, (riskScore / maxRisk) * 100));

        console.log(`\n🎯 CATALOG SECURITY SCORE: ${securityScore.toFixed(1)}/100`);

        if (securityScore >= 90) {
            console.log('   Status: ✅ EXCELLENT - Production Ready');
        } else if (securityScore >= 75) {
            console.log('   Status: ⚠️  GOOD - Minor Issues');
        } else if (securityScore >= 50) {
            console.log('   Status: ⚠️  FAIR - Needs Improvement');
        } else {
            console.log('   Status: ❌ POOR - Significant Issues');
        }

        // Catalog-specific findings
        console.log('\n🗂️  CATALOG-SPECIFIC SECURITY FINDINGS:');
        
        const catalogIssues = {
            'SQL_INJECTION': this.vulnerabilities.filter(v => v.type.includes('SQL_INJECTION')).length,
            'XSS': this.vulnerabilities.filter(v => v.type.includes('XSS')).length,
            'CSRF': this.vulnerabilities.filter(v => v.type.includes('CSRF')).length,
            'URL_VALIDATION': this.vulnerabilities.filter(v => v.type.includes('URL_VALIDATION')).length,
            'METADATA_INJECTION': this.vulnerabilities.filter(v => v.type.includes('METADATA_INJECTION')).length,
            'INSECURE_CONNECTION': this.vulnerabilities.filter(v => v.type.includes('INSECURE')).length
        };

        const logCatalogIssue = ([type, count]) => {
            if (count > 0) {
                console.log(`   ${type.replace('_', ' ')}: ${count} issues`);
            }
        };
        Object.entries(catalogIssues).forEach(logCatalogIssue);

        // Detailed vulnerabilities
        if (this.vulnerabilities.length > 0) {
            console.log('\n🚨 VULNERABILITIES FOUND:');
            
            const processSeverityLevel = (severity) => {
                const issues = this.vulnerabilities.filter(v => v.severity === severity);
                if (issues.length > 0) {
                    console.log(`\n${severity} (${issues.length}):`);
                    const logVulnerability = (vuln, index) => {
                        console.log(`\n${index + 1}. ${vuln.type} - ${vuln.file}:${vuln.line || 'N/A'}`);
                        console.log(`   Description: ${vuln.description}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code}`);
                        console.log(`   Fix: ${vuln.recommendation}`);
                    };
                    issues.forEach(logVulnerability);
                }
            };
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].forEach(processSeverityLevel);
        }

        // Recommendations
        console.log('\n💡 AGENT 12 CATALOG SECURITY RECOMMENDATIONS:');
        console.log('\n1. 🔐 Implement SecurityUtils for catalog operations');
        console.log('   - Add SecurityUtils.validateCatalogEntry() for all entries');
        console.log('   - Use SecurityUtils.secureCallFunction() for catalog operations');
        console.log('   - Implement SecurityUtils.sanitizeCatalogData() for metadata');
        
        console.log('\n2. 🛡️  Enhance catalog-specific security');
        console.log('   - Validate all resource URLs with SecurityUtils.validateResourceURL()');
        console.log('   - Sanitize search queries with SecurityUtils.sanitizeSearchQuery()');
        console.log('   - Secure registry synchronization operations');
        
        console.log('\n3. 🔒 Secure catalog communications');
        console.log('   - Upgrade WebSocket to WSS for catalog updates');
        console.log('   - Use HTTPS for EventSource catalog streams');
        console.log('   - Implement secure catalog discovery protocols');

        console.log('\n4. ⚡ Monitor catalog security');
        console.log('   - Log all catalog operations for audit');
        console.log('   - Monitor resource discovery for malicious URLs');
        console.log('   - Track metadata injection attempts');

        console.log(`\n${  '='.repeat(80)}`);
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('='.repeat(80));

        // Exit with appropriate code
        process.exit(critical > 0 || high > 0 ? 1 : 0);
    }
}

// Run the scanner if called directly
if (require.main === module) {
    const scanner = new Agent12SecurityScanner();
    const targetPath = process.argv[2] || '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent12';
    scanner.scanDirectory(targetPath);
}

module.exports = Agent12SecurityScanner;