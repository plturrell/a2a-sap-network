#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 3 (Vector Processing Agent) Security Scanner
 * Specialized scanner for vector processing, embeddings, similarity search, and vector database vulnerabilities
 */
class Agent3SecurityScanner {
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
        
        // Vector Processing-specific vulnerability patterns
        this.vectorProcessingPatterns = {
            // Vector injection attacks
            VECTOR_INJECTION: {
                patterns: [
                    /vectorQuery\s*=.*eval/gi,
                    /\.searchVectors\s*\([^)]*Function/gi,
                    /embeddingData\s*=.*\+/gi,
                    /\.injectVector\s*\([^)]*\$\{/gi,
                    /similarityQuery\s*=.*user/gi,
                    /\.manipulateVector\s*\(/gi,
                    /unsafeVector\s*=\s*true/gi,
                    /vectorInput\s*=.*input/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'VECTOR_INJECTION',
                message: 'Potential vector injection vulnerability',
                impact: 'Could allow injection of malicious vectors affecting similarity calculations and search results'
            },
            
            // Dimension manipulation vulnerabilities
            DIMENSION_MANIPULATION: {
                patterns: [
                    /dimensions\s*=.*eval/gi,
                    /\.setDimensions\s*\([^)]*Function/gi,
                    /vectorDimensions\s*=.*\+/gi,
                    /\.modifyDimensions\s*\([^)]*\$\{/gi,
                    /customDimensions\s*=.*user/gi,
                    /\.overrideDimensions\s*\(/gi,
                    /dimensionCheck\s*=\s*false/gi,
                    /skipDimensionValidation\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'DIMENSION_MANIPULATION',
                message: 'Vector dimension manipulation vulnerability',
                impact: 'Could allow manipulation of vector dimensions causing calculation errors and system instability'
            },
            
            // Embedding poisoning vulnerabilities
            EMBEDDING_POISONING: {
                patterns: [
                    /embeddings\s*=.*eval/gi,
                    /\.generateEmbeddings\s*\([^)]*Function/gi,
                    /embeddingModel\s*=.*\+/gi,
                    /\.poisonEmbeddings\s*\([^)]*\$\{/gi,
                    /maliciousEmbeddings\s*=.*user/gi,
                    /\.tamperEmbeddings\s*\(/gi,
                    /embeddingValidation\s*=\s*false/gi,
                    /bypassEmbeddingCheck\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'EMBEDDING_POISONING',
                message: 'Embedding poisoning vulnerability',
                impact: 'Could allow poisoning of embedding models affecting all vector operations and similarity searches'
            },
            
            // Vector database security vulnerabilities
            VECTOR_DB_SECURITY: {
                patterns: [
                    /vectorDatabase\s*=.*eval/gi,
                    /\.connectVectorDB\s*\([^)]*Function/gi,
                    /dbConnection\s*=.*\+/gi,
                    /\.executeVectorQuery\s*\([^)]*\$\{/gi,
                    /vectorDbConfig\s*=.*user/gi,
                    /\.bypassVectorAuth\s*\(/gi,
                    /vectorDbAuth\s*=\s*false/gi,
                    /unsafeVectorDB\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'VECTOR_DB_SECURITY',
                message: 'Vector database security vulnerability',
                impact: 'Could allow unauthorized access to vector databases exposing sensitive embedding data'
            },
            
            // Similarity calculation bypass vulnerabilities
            SIMILARITY_BYPASS: {
                patterns: [
                    /similarityThreshold\s*=.*eval/gi,
                    /\.calculateSimilarity\s*\([^)]*Function/gi,
                    /distanceMetric\s*=.*\+/gi,
                    /\.overrideSimilarity\s*\([^)]*\$\{/gi,
                    /similarityCheck\s*=\s*false/gi,
                    /\.bypassSimilarity\s*\(/gi,
                    /forceSimilarity\s*=\s*true/gi,
                    /skipSimilarityValidation\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'SIMILARITY_BYPASS',
                message: 'Similarity calculation bypass vulnerability',
                impact: 'Could allow bypassing similarity calculations returning inaccurate or malicious results'
            },
            
            // Index manipulation vulnerabilities
            INDEX_MANIPULATION: {
                patterns: [
                    /vectorIndex\s*=.*eval/gi,
                    /\.modifyIndex\s*\([^)]*Function/gi,
                    /indexType\s*=.*\+/gi,
                    /\.corruptIndex\s*\([^)]*\$\{/gi,
                    /indexConfig\s*=.*user/gi,
                    /\.tamperlndex\s*\(/gi,
                    /indexValidation\s*=\s*false/gi,
                    /unsafeIndex\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'INDEX_MANIPULATION',
                message: 'Vector index manipulation vulnerability',
                impact: 'Could allow manipulation of vector indexes affecting search performance and accuracy'
            },
            
            // Vector search vulnerabilities
            VECTOR_SEARCH_VULN: {
                patterns: [
                    /vectorSearch\s*=.*eval/gi,
                    /\.searchQuery\s*\([^)]*Function/gi,
                    /searchFilters\s*=.*\+/gi,
                    /\.manipulateSearch\s*\([^)]*\$\{/gi,
                    /searchValidation\s*=\s*false/gi,
                    /\.bypassSearchAuth\s*\(/gi,
                    /unsafeSearch\s*=\s*true/gi,
                    /topK\s*=.*user/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'VECTOR_SEARCH_VULNERABILITY',
                message: 'Vector search vulnerability',
                impact: 'Could allow unauthorized vector search operations or manipulation of search results'
            },
            
            // Clustering manipulation vulnerabilities
            CLUSTERING_MANIPULATION: {
                patterns: [
                    /clusterAnalysis\s*=.*eval/gi,
                    /\.runClustering\s*\([^)]*Function/gi,
                    /clusterConfig\s*=.*\+/gi,
                    /\.manipulateClusters\s*\([^)]*\$\{/gi,
                    /clusterValidation\s*=\s*false/gi,
                    /\.overrideClusters\s*\(/gi,
                    /unsafeClustering\s*=\s*true/gi,
                    /numClusters\s*=.*user/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'CLUSTERING_MANIPULATION',
                message: 'Clustering manipulation vulnerability',
                impact: 'Could allow manipulation of clustering algorithms affecting data analysis and insights'
            },
            
            // 3D Visualization security vulnerabilities
            VISUALIZATION_SECURITY: {
                patterns: [
                    /visualization3D\s*=.*eval/gi,
                    /\.render3D\s*\([^)]*Function/gi,
                    /sceneConfig\s*=.*\+/gi,
                    /\.injectVisualization\s*\([^)]*\$\{/gi,
                    /visualizationData\s*=.*user/gi,
                    /\.manipulateScene\s*\(/gi,
                    /unsafeVisualization\s*=\s*true/gi,
                    /bypassVisualizationAuth\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'VISUALIZATION_SECURITY',
                message: '3D Visualization security vulnerability',
                impact: 'Could allow injection of malicious content into 3D visualization affecting user experience'
            },
            
            // Vector export/import vulnerabilities
            VECTOR_EXPORT_VULN: {
                patterns: [
                    /exportVectors\s*=.*eval/gi,
                    /\.exportEmbeddings\s*\([^)]*Function/gi,
                    /exportConfig\s*=.*\+/gi,
                    /\.exportSensitive\s*\([^)]*\$\{/gi,
                    /includePrivateVectors\s*=\s*true/gi,
                    /\.dumpVectors\s*\(/gi,
                    /exportValidation\s*=\s*false/gi,
                    /unsafeExport\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'VECTOR_EXPORT_VULNERABILITY',
                message: 'Vector export vulnerability',
                impact: 'Could allow unauthorized export of sensitive vector data and embeddings'
            },
            
            // Vector metadata manipulation
            METADATA_MANIPULATION: {
                patterns: [
                    /vectorMetadata\s*=.*eval/gi,
                    /\.setMetadata\s*\([^)]*Function/gi,
                    /metadataSchema\s*=.*\+/gi,
                    /\.injectMetadata\s*\([^)]*\$\{/gi,
                    /customMetadata\s*=.*user/gi,
                    /\.tamperMetadata\s*\(/gi,
                    /metadataValidation\s*=\s*false/gi,
                    /unsafeMetadata\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'METADATA_MANIPULATION',
                message: 'Vector metadata manipulation vulnerability',
                impact: 'Could allow manipulation of vector metadata affecting search and classification results'
            },
            
            // Collection management vulnerabilities
            COLLECTION_MANIPULATION: {
                patterns: [
                    /vectorCollections\s*=.*eval/gi,
                    /\.manageCollections\s*\([^)]*Function/gi,
                    /collectionName\s*=.*\+/gi,
                    /\.deleteCollection\s*\([^)]*\$\{/gi,
                    /collectionAuth\s*=\s*false/gi,
                    /\.overrideCollection\s*\(/gi,
                    /unsafeCollection\s*=\s*true/gi,
                    /bypassCollectionValidation\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'COLLECTION_MANIPULATION',
                message: 'Vector collection manipulation vulnerability',
                impact: 'Could allow unauthorized access or manipulation of vector collections'
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
            
            // Check for vector processing-specific vulnerabilities
            this.checkVectorProcessingVulnerabilities(content, filePath, lines);
            
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
            { pattern: /dangerouslySetInnerHTML/gi, type: 'XSS', message: 'Potential XSS via React dangerouslySetInnerHTML' },
            { pattern: /\.setText\s*\([^)]*\+/gi, type: 'XSS', message: 'Potential XSS via dynamic text setting' }
        ];
        
        xssPatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                
                // Skip if properly sanitized
                if (!code.includes('encodeXML') && !code.includes('sanitizeHTML') && !code.includes('escapeRegExp')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could allow execution of malicious scripts in vector processing interfaces',
                        fix: 'Use proper output encoding and sanitization (encodeXML, sanitizeHTML)'
                    });
                }
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
                // Check if CSRF token is present nearby
                const surroundingCode = content.substring(Math.max(0, match.index - 200), match.index + 200);
                if (!surroundingCode.includes('X-CSRF-Token') && 
                    !surroundingCode.includes('csrf') &&
                    !surroundingCode.includes('_getCSRFToken') &&
                    !surroundingCode.includes('_csrfToken')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection in vector processing requests',
                        impact: 'Could allow unauthorized vector processing operations',
                        fix: 'Add CSRF token using _getCSRFToken() method'
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
                // Skip comments, examples, and already secured code
                if (!code.includes('//') && !code.includes('example') && 
                    !code.includes('wss://') && !code.includes('https://')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could expose sensitive vector data and embeddings in transit',
                        fix: 'Use HTTPS/WSS for all vector processing service connections'
                    });
                }
            }
        });
        
        // Input validation vulnerabilities specific to vector contexts
        const inputValidationPatterns = [
            { pattern: /eval\s*\(/gi, type: 'CODE_INJECTION', message: 'Code injection via eval()' },
            { pattern: /new\s+Function\s*\(/gi, type: 'CODE_INJECTION', message: 'Code injection via Function constructor' },
            { pattern: /setTimeout\s*\([^)]*\+/gi, type: 'CODE_INJECTION', message: 'Potential code injection in setTimeout' },
            { pattern: /setInterval\s*\([^)]*\+/gi, type: 'CODE_INJECTION', message: 'Potential code injection in setInterval' }
        ];
        
        inputValidationPatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                this.addVulnerability({
                    type: type,
                    severity: this.severityLevels.CRITICAL,
                    file: filePath,
                    line: lineNumber,
                    code: lines[lineNumber - 1]?.trim() || '',
                    message: message,
                    impact: 'Could allow arbitrary code execution in vector processing context',
                    fix: 'Remove eval() and Function constructor usage, use safe alternatives'
                });
            }
        });
    }
    
    checkVectorProcessingVulnerabilities(content, filePath, lines) {
        Object.entries(this.vectorProcessingPatterns).forEach(([vulnType, config]) => {
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
                        fix: this.getVectorProcessingFix(vulnType)
                    });
                }
            });
        });
    }
    
    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            VECTOR_INJECTION: [
                /vectorQuery\s*=\s*["']/gi,  // String literals
                /_validateVector\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ],
            DIMENSION_MANIPULATION: [
                /dimensions\s*=\s*\d+/gi,  // Numeric literals
                /_validateDimensions\./gi,  // Validation functions
                /validation\./gi  // Validation operations
            ],
            EMBEDDING_POISONING: [
                /embeddings\s*=\s*["']/gi,  // String assignments
                /_sanitizeEmbedding\./gi,  // Sanitization functions
                /validation\./gi  // Validation operations
            ],
            SIMILARITY_BYPASS: [
                /similarityThreshold\s*=\s*[\d\.]+/gi,  // Numeric literals
                /_validateSimilarity\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ],
            COLLECTION_MANIPULATION: [
                /vectorCollections\s*=\s*\[/gi,  // Array literals
                /_validateCollection\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ]
        };
        
        if (falsePositivePatterns[vulnType]) {
            const patterns = falsePositivePatterns[vulnType];
            if (patterns.some(pattern => pattern.test(code))) {
                return true;
            }
        }
        
        // General false positive checks
        if (filePath.includes('SecurityUtils.js') || filePath.includes('security')) {
            // Security files contain security functions that may trigger patterns
            return code.includes('sanitized') || code.includes('validation') || 
                   code.includes('_sanitize') || code.includes('_validate');
        }
        
        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*')) {
            return true;
        }
        
        // Skip console.log and debug statements
        if (code.includes('console.log') || code.includes('console.error')) {
            return true;
        }
        
        return false;
    }
    
    getVectorProcessingFix(vulnType) {
        const fixes = {
            VECTOR_INJECTION: 'Validate and sanitize all vector input data, use parameterized queries for vector operations',
            DIMENSION_MANIPULATION: 'Implement strict dimension validation and enforce model-specific dimension constraints',
            EMBEDDING_POISONING: 'Validate embedding integrity using cryptographic checksums and anomaly detection',
            VECTOR_DB_SECURITY: 'Implement proper authentication and authorization for vector database access',
            SIMILARITY_BYPASS: 'Enforce similarity calculation validation and implement threshold boundaries',
            INDEX_MANIPULATION: 'Protect vector indexes with integrity checks and access controls',
            VECTOR_SEARCH_VULN: 'Implement proper authorization for vector search operations and validate search parameters',
            CLUSTERING_MANIPULATION: 'Validate clustering parameters and implement secure clustering configurations',
            VISUALIZATION_SECURITY: 'Sanitize all visualization data and implement content security policies',
            VECTOR_EXPORT_VULN: 'Implement access controls and data classification for vector exports',
            METADATA_MANIPULATION: 'Validate all metadata operations and implement schema validation',
            COLLECTION_MANIPULATION: 'Implement proper authorization and validation for collection management operations'
        };
        return fixes[vulnType] || 'Implement proper validation and security controls for vector processing operations';
    }
    
    checkSAPFioriCompliance(content, filePath, lines) {
        // Check for missing i18n
        if (filePath.includes('.controller.js')) {
            const i18nPatterns = [
                { pattern: /MessageToast\.show\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded message in MessageToast' },
                { pattern: /MessageBox\.\w+\s*\(\s*["'][^"']+["']/gi, message: 'Hardcoded message in MessageBox' },
                { pattern: /setText\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded text in UI element' },
                { pattern: /headerText\s*:\s*["'][^"']+["']/gi, message: 'Hardcoded header text' }
            ];
            
            i18nPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';
                    // Skip already internationalized strings
                    if (!code.includes('getResourceBundle()') && !code.includes('i18n')) {
                        this.addVulnerability({
                            type: 'SAP_STANDARDS',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: code,
                            message: message,
                            impact: 'Reduces internationalization support for vector processing interfaces',
                            fix: 'Use i18n resource bundle: this.getResourceBundle().getText("key")'
                        });
                    }
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
                        impact: 'Reduces vector processing application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
                }
            });
        }
        
        // Check for vector-specific accessibility issues
        if (filePath.includes('.controller.js') || filePath.includes('.fragment.xml')) {
            const accessibilityPatterns = [
                { pattern: /VizFrame\s*\([^)]*\)/gi, message: 'Chart accessibility: Missing ARIA labels for vector visualizations' },
                { pattern: /MicroChart\s*\([^)]*\)/gi, message: 'MicroChart accessibility: Missing ARIA labels' },
                { pattern: /setBusy\s*\(\s*true\s*\)/gi, message: 'Loading state accessibility: Consider announcing vector processing status changes' },
                { pattern: /3D.*visualization/gi, message: '3D visualization accessibility: Missing alternative text descriptions' }
            ];
            
            accessibilityPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const surroundingCode = content.substring(Math.max(0, match.index - 100), match.index + 100);
                    if (!surroundingCode.includes('aria') && !surroundingCode.includes('announceForAccessibility')) {
                        this.addVulnerability({
                            type: 'ACCESSIBILITY',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: lines[lineNumber - 1]?.trim() || '',
                            message: message,
                            impact: 'Reduces accessibility for vector processing interfaces',
                            fix: 'Add appropriate ARIA labels and accessibility announcements'
                        });
                    }
                }
            });
        }
        
        // Check for vector processing performance issues
        const performancePatterns = [
            { pattern: /_animationFrameId\s*=\s*requestAnimationFrame/gi, message: 'Animation frame management: Potential memory leak in 3D visualization' },
            { pattern: /WebSocket.*ws:/gi, message: 'WebSocket security: Use secure WebSocket connections (wss://)' },
            { pattern: /setTimeout.*\d{4,}/gi, message: 'Long timeout detected: May affect vector processing responsiveness' }
        ];
        
        performancePatterns.forEach(({ pattern, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                this.addVulnerability({
                    type: 'PERFORMANCE',
                    severity: this.severityLevels.WARNING,
                    file: filePath,
                    line: lineNumber,
                    code: code,
                    message: message,
                    impact: 'May affect vector processing performance and user experience',
                    fix: 'Review implementation for performance optimization and proper resource cleanup'
                });
            }
        });
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
        const extensions = ['.js', '.xml', '.json', '.html', '.ts'];
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
        console.log('🧮 AGENT 3 VECTOR PROCESSING SECURITY SCAN REPORT');
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
        
        console.log(`\n🎯 VECTOR PROCESSING SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log('   Status: ✅ EXCELLENT - Vector processing system is well secured');
        } else if (score >= 70) {
            console.log('   Status: ⚠️  GOOD - Minor vector security issues to address');
        } else if (score >= 50) {
            console.log('   Status: ⚠️  FAIR - Several vector security issues need attention');
        } else {
            console.log('   Status: ❌ POOR - Significant vector processing security improvements needed');
        }
        
        // Vector-specific findings
        const vectorIssues = this.vulnerabilities.filter(v => 
            v.type.includes('VECTOR') || v.type.includes('EMBEDDING') || 
            v.type.includes('SIMILARITY') || v.type.includes('INDEX') ||
            v.type.includes('CLUSTERING') || v.type.includes('VISUALIZATION'));
        if (vectorIssues.length > 0) {
            console.log('\n🧮 VECTOR PROCESSING-SPECIFIC SECURITY FINDINGS:');
            const issueCounts = {};
            vectorIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });
            
            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type}: ${count} issues`);
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
                        console.log(`   Code: ${vuln.code.substring(0, 80)}${vuln.code.length > 80 ? '...' : ''}`);
                        console.log(`   Fix: ${vuln.fix}\n`);
                        issueNumber++;
                    });
                }
            });
        }
        
        // Vector processing security recommendations
        console.log('💡 AGENT 3 VECTOR PROCESSING SECURITY RECOMMENDATIONS:\n');
        console.log('1. 🛡️  Secure Vector Input Validation');
        console.log('   - Validate all vector dimensions against model specifications');
        console.log('   - Implement vector data type validation and bounds checking');
        console.log('   - Sanitize vector metadata and ensure schema compliance');
        console.log('   - Use cryptographic checksums to verify vector data integrity');
        
        console.log('\n2. 🔒 Embedding Security Protection');
        console.log('   - Implement embedding poisoning detection mechanisms');
        console.log('   - Validate embedding models and their configurations');
        console.log('   - Monitor embedding generation for anomalous patterns');
        console.log('   - Implement differential privacy for sensitive embeddings');
        
        console.log('\n3. 🔐 Vector Database Security');
        console.log('   - Use strong authentication for vector database connections');
        console.log('   - Implement proper authorization for vector operations');
        console.log('   - Encrypt vector data at rest and in transit');
        console.log('   - Monitor vector database access for unauthorized operations');
        
        console.log('\n4. ⚡ Similarity Search Protection');
        console.log('   - Validate similarity thresholds and distance metrics');
        console.log('   - Implement rate limiting for similarity search operations');
        console.log('   - Protect against similarity calculation bypass attacks');
        console.log('   - Monitor search patterns for anomalous behavior');
        
        console.log('\n5. 🔍 Vector Index Integrity');
        console.log('   - Implement index integrity validation and checksums');
        console.log('   - Protect index configurations from unauthorized changes');
        console.log('   - Monitor index operations for manipulation attempts');
        console.log('   - Use secure index storage with proper access controls');
        
        console.log('\n6. 📊 Clustering and Analysis Security');
        console.log('   - Validate clustering parameters and algorithm configurations');
        console.log('   - Implement secure cluster analysis with privacy preservation');
        console.log('   - Monitor clustering results for data leakage');
        console.log('   - Use differential privacy in cluster analysis when needed');
        
        console.log('\n7. 🎨 3D Visualization Security');
        console.log('   - Sanitize all 3D visualization data and configurations');
        console.log('   - Implement content security policies for WebGL content');
        console.log('   - Validate visualization parameters and scene configurations');
        console.log('   - Monitor 3D rendering for malicious content injection');
        
        console.log('\n8. 📤 Vector Export Controls');
        console.log('   - Implement access controls for vector data exports');
        console.log('   - Validate export configurations and data inclusion policies');
        console.log('   - Use data classification and labeling for sensitive vectors');
        console.log('   - Audit all vector export operations for compliance');
        
        console.log(`\n${  '='.repeat(80)}`);
        console.log('Vector Processing Security Scan completed. Address critical vector vulnerabilities first.');
        console.log('Focus on vector injection and embedding poisoning prevention.');
        console.log('='.repeat(80));
        
        // Generate JSON report for further processing
        this.saveJSONReport();
    }
    
    saveJSONReport() {
        const reportData = {
            agent: 'Agent3-VectorProcessing',
            timestamp: new Date().toISOString(),
            summary: {
                filesScanned: this.filesScanned,
                totalVulnerabilities: this.vulnerabilities.length,
                critical: this.vulnerabilities.filter(v => v.severity === 'CRITICAL').length,
                high: this.vulnerabilities.filter(v => v.severity === 'HIGH').length,
                medium: this.vulnerabilities.filter(v => v.severity === 'MEDIUM').length,
                low: this.vulnerabilities.filter(v => v.severity === 'LOW').length,
                warnings: this.vulnerabilities.filter(v => v.severity === 'WARNING').length
            },
            vulnerabilities: this.vulnerabilities,
            recommendations: [
                'Implement vector input validation and dimension checking',
                'Protect against embedding poisoning attacks',
                'Secure vector database connections and access',
                'Validate similarity search operations and parameters',
                'Implement vector index integrity protection',
                'Secure clustering and analysis operations',
                'Protect 3D visualization from content injection',
                'Implement proper vector export access controls'
            ]
        };
        
        try {
            fs.writeFileSync('agent3-security-report.json', JSON.stringify(reportData, null, 2));
            console.log('\n📄 Detailed report saved to: agent3-security-report.json');
        } catch (error) {
            console.error('Failed to save JSON report:', error.message);
        }
    }
    
    run(targetPath) {
        console.log('🔍 Starting Agent 3 Vector Processing Security Scan...');
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
const scanner = new Agent3SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent3';
scanner.run(targetPath);