/**
 * @fileoverview SCIP to Glean Fact Transformer
 * @module gleanFactTransformer
 * @since 1.0.0
 *
 * Transforms SCIP (SCIP Code Intelligence Protocol) data into proper Glean facts
 * following the defined Angle schema for comprehensive code analysis
 */

const crypto = require('crypto');
const path = require('path');

class GleanFactTransformer {
    constructor() {
        this.factIdCounter = 0;
        this.symbolTable = new Map();
        this.crossReferences = new Map();
    }

    /**
     * Transform SCIP index to Glean facts
     * @param {Object} scipIndex - Complete SCIP index
     * @returns {Object} Glean fact batches organized by predicate
     */
    transformSCIPToGlean(scipIndex) {
        const factBatches = {
            'src.File': [],
            'src.Symbol': [],
            'src.XRef': [],
            'src.Dependency': [],
            'src.Function': [],
            'src.Class': [],
            'src.Contract': [],
            'src.Import': [],
            'src.Export': [],
            'src.CDSEntity': [],
            'src.CDSService': [],
            'src.SecurityIssue': [],
            'src.PerformanceIssue': [],
            'src.CodeSmell': []
        };

        // Process each document in the SCIP index
        scipIndex.documents.forEach(document => {
            this.processDocument(document, factBatches);
        });

        // Process external symbols
        scipIndex.external_symbols.forEach(symbol => {
            this.processExternalSymbol(symbol, factBatches);
        });

        return factBatches;
    }

    processDocument(document, factBatches) {
        const filePath = document.relative_path;
        const language = this.detectLanguage(filePath);

        // Create File fact
        const fileFact = this.createFileFact(document, language);
        factBatches['src.File'].push(fileFact);

        // Process symbols in the document
        document.symbols.forEach(symbol => {
            this.processSymbol(symbol, document, factBatches);
        });

        // Process occurrences (cross-references)
        document.occurrences.forEach(occurrence => {
            this.processOccurrence(occurrence, document, factBatches);
        });

        // Language-specific processing
        if (language === 'typescript' || language === 'javascript') {
            this.processJavaScriptSpecific(document, factBatches);
        } else if (language === 'python') {
            this.processPythonSpecific(document, factBatches);
        } else if (language === 'solidity') {
            this.processSoliditySpecific(document, factBatches);
        } else if (language === 'cds') {
            this.processCDSSpecific(document, factBatches);
        }
    }

    createFileFact(document, language) {
        const content = this.getDocumentContent(document);
        const lines = content ? content.split('\n').length : 0;

        return {
            id: this.generateFactId(),
            key: {
                file: document.relative_path
            },
            value: {
                file: document.relative_path,
                language: language,
                size: content ? content.length : 0,
                symbols: document.symbols.length,
                lines: lines,
                checksum: this.calculateChecksum(content || '')
            }
        };
    }

    processSymbol(symbol, document, factBatches) {
        const symbolFact = this.createSymbolFact(symbol, document);
        factBatches['src.Symbol'].push(symbolFact);

        // Store in symbol table for cross-referencing
        this.symbolTable.set(symbol.symbol, {
            file: document.relative_path,
            symbolData: symbol
        });

        // Create specific facts based on symbol kind
        if (symbol.definition?.syntax_kind) {
            switch (symbol.definition.syntax_kind) {
                case 'Function':
                case 'Method':
                    this.createFunctionFact(symbol, document, factBatches);
                    break;
                case 'Class':
                    this.createClassFact(symbol, document, factBatches);
                    break;
                case 'Contract':
                    this.createContractFact(symbol, document, factBatches);
                    break;
                case 'Import':
                case 'ImportStatement':
                    this.createImportFact(symbol, document, factBatches);
                    break;
                case 'Export':
                case 'ExportStatement':
                    this.createExportFact(symbol, document, factBatches);
                    break;
            }
        }
    }

    createSymbolFact(symbol, document) {
        const range = symbol.definition?.range;

        return {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                symbol: symbol.symbol,
                line: range?.start?.line || 0
            },
            value: {
                file: document.relative_path,
                symbol: symbol.symbol,
                name: this.extractSymbolName(symbol),
                kind: this.mapSyntaxKindToSymbolKind(symbol.definition?.syntax_kind),
                line: range?.start?.line || 0,
                column: range?.start?.character || 0,
                visibility: this.extractVisibility(symbol),
                range: range ? {
                    start_line: range.start.line,
                    start_column: range.start.character,
                    end_line: range.end.line,
                    end_column: range.end.character
                } : null
            }
        };
    }

    createFunctionFact(symbol, document, factBatches) {
        const functionInfo = this.extractFunctionInfo(symbol, document);

        const functionFact = {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                name: functionInfo.name
            },
            value: {
                file: document.relative_path,
                name: functionInfo.name,
                signature: functionInfo.signature,
                return_type: functionInfo.returnType,
                parameters: functionInfo.parameters,
                line: symbol.definition?.range?.start?.line || 0,
                complexity: this.calculateComplexity(functionInfo.body),
                async: functionInfo.isAsync,
                exported: functionInfo.isExported
            }
        };

        factBatches['src.Function'].push(functionFact);
    }

    createClassFact(symbol, document, factBatches) {
        const classInfo = this.extractClassInfo(symbol, document);

        const classFact = {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                name: classInfo.name
            },
            value: {
                file: document.relative_path,
                name: classInfo.name,
                extends: classInfo.extends,
                implements: classInfo.implements,
                line: symbol.definition?.range?.start?.line || 0,
                abstract: classInfo.isAbstract,
                exported: classInfo.isExported,
                methods: classInfo.methods,
                fields: classInfo.fields
            }
        };

        factBatches['src.Class'].push(classFact);
    }

    createContractFact(symbol, document, factBatches) {
        const contractInfo = this.extractContractInfo(symbol, document);

        const contractFact = {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                name: contractInfo.name
            },
            value: {
                file: document.relative_path,
                name: contractInfo.name,
                extends: contractInfo.extends,
                line: symbol.definition?.range?.start?.line || 0,
                functions: contractInfo.functions,
                events: contractInfo.events,
                modifiers: contractInfo.modifiers,
                state_variables: contractInfo.stateVariables
            }
        };

        factBatches['src.Contract'].push(contractFact);
    }

    createImportFact(symbol, document, factBatches) {
        const importInfo = this.extractImportInfo(symbol, document);

        const importFact = {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                module: importInfo.module,
                line: symbol.definition?.range?.start?.line || 0
            },
            value: {
                file: document.relative_path,
                module: importInfo.module,
                imported_names: importInfo.importedNames,
                default_import: importInfo.defaultImport,
                namespace_import: importInfo.namespaceImport,
                line: symbol.definition?.range?.start?.line || 0
            }
        };

        factBatches['src.Import'].push(importFact);

        // Create dependency fact
        const dependencyFact = {
            id: this.generateFactId(),
            key: {
                source_file: document.relative_path,
                target_file: importInfo.resolvedPath || importInfo.module,
                line: symbol.definition?.range?.start?.line || 0
            },
            value: {
                source_file: document.relative_path,
                target_file: importInfo.resolvedPath || importInfo.module,
                import_name: importInfo.module,
                dependency_type: 'import',
                line: symbol.definition?.range?.start?.line || 0
            }
        };

        factBatches['src.Dependency'].push(dependencyFact);
    }

    createExportFact(symbol, document, factBatches) {
        const exportInfo = this.extractExportInfo(symbol, document);

        const exportFact = {
            id: this.generateFactId(),
            key: {
                file: document.relative_path,
                name: exportInfo.name,
                line: symbol.definition?.range?.start?.line || 0
            },
            value: {
                file: document.relative_path,
                name: exportInfo.name,
                export_type: exportInfo.exportType,
                line: symbol.definition?.range?.start?.line || 0
            }
        };

        factBatches['src.Export'].push(exportFact);
    }

    processOccurrence(occurrence, document, factBatches) {
        const targetSymbol = this.symbolTable.get(occurrence.symbol);

        if (targetSymbol) {
            const xrefFact = {
                id: this.generateFactId(),
                key: {
                    file: document.relative_path,
                    symbol: occurrence.symbol,
                    line: occurrence.range?.start?.line || 0
                },
                value: {
                    file: document.relative_path,
                    symbol: occurrence.symbol,
                    target_file: targetSymbol.file,
                    target_symbol: occurrence.symbol,
                    kind: this.mapSymbolRolesToXRefKind(occurrence.symbol_roles),
                    line: occurrence.range?.start?.line || 0,
                    column: occurrence.range?.start?.character || 0
                }
            };

            factBatches['src.XRef'].push(xrefFact);
        }
    }

    processJavaScriptSpecific(document, factBatches) {
        // Additional JavaScript/TypeScript specific analysis
        const content = this.getDocumentContent(document);
        if (!content) return;

        // Detect security issues
        this.detectJavaScriptSecurityIssues(content, document, factBatches);

        // Detect performance issues
        this.detectJavaScriptPerformanceIssues(content, document, factBatches);

        // Detect code smells
        this.detectJavaScriptCodeSmells(content, document, factBatches);
    }

    processPythonSpecific(document, factBatches) {
        const content = this.getDocumentContent(document);
        if (!content) return;

        // Python-specific analysis
        this.detectPythonSecurityIssues(content, document, factBatches);
        this.detectPythonPerformanceIssues(content, document, factBatches);
    }

    processSoliditySpecific(document, factBatches) {
        const content = this.getDocumentContent(document);
        if (!content) return;

        // Solidity-specific security analysis
        this.detectSoliditySecurityIssues(content, document, factBatches);
        this.detectSolidityPerformanceIssues(content, document, factBatches);
    }

    processCDSSpecific(document, factBatches) {
        const content = this.getDocumentContent(document);
        if (!content) return;

        // Extract CDS entities and services
        this.extractCDSEntities(content, document, factBatches);
        this.extractCDSServices(content, document, factBatches);
    }

    detectJavaScriptSecurityIssues(content, document, factBatches) {
        const securityPatterns = [
            {
                pattern: /eval\s*\(/g,
                type: 'unsafe_eval',
                severity: 'high',
                description: 'Use of eval() can lead to code injection'
            },
            {
                pattern: /innerHTML\s*=/g,
                type: 'xss',
                severity: 'medium',
                description: 'Direct innerHTML assignment may lead to XSS'
            },
            {
                pattern: /document\.write\s*\(/g,
                type: 'xss',
                severity: 'medium',
                description: 'document.write() can lead to XSS vulnerabilities'
            },
            {
                pattern: /(?:password|secret|key|token)\s*[:=]\s*['"][^'"]{8,}['"]/gi,
                type: 'hardcoded_secret',
                severity: 'critical',
                description: 'Hardcoded secrets detected'
            }
        ];

        securityPatterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const securityFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        column: this.getColumnNumber(content, match.index),
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        cwe_id: this.getCWEId(pattern.type)
                    }
                };

                factBatches['src.SecurityIssue'].push(securityFact);
            }
        });
    }

    detectJavaScriptPerformanceIssues(content, document, factBatches) {
        const performancePatterns = [
            {
                pattern: /for\s*\([^)]*\)\s*\{[^}]*await/g,
                type: 'sync_in_async',
                severity: 'medium',
                description: 'Synchronous iteration with async operations',
                suggestion: 'Use Promise.all() or for...of with async/await'
            },
            {
                pattern: /JSON\.parse\(JSON\.stringify\(/g,
                type: 'inefficient_loop',
                severity: 'low',
                description: 'Inefficient deep clone using JSON methods',
                suggestion: 'Use structuredClone() or a proper deep clone library'
            }
        ];

        performancePatterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const perfFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        suggestion: pattern.suggestion
                    }
                };

                factBatches['src.PerformanceIssue'].push(perfFact);
            }
        });
    }

    detectSoliditySecurityIssues(content, document, factBatches) {
        const solidityPatterns = [
            {
                pattern: /\.call\s*\(/g,
                type: 'unsafe_call',
                severity: 'high',
                description: 'Low-level call() usage may be unsafe'
            },
            {
                pattern: /tx\.origin/g,
                type: 'tx_origin',
                severity: 'high',
                description: 'Use of tx.origin for authorization is unsafe'
            },
            {
                pattern: /selfdestruct\s*\(/g,
                type: 'selfdestruct',
                severity: 'critical',
                description: 'selfdestruct() usage requires careful consideration'
            }
        ];

        solidityPatterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const securityFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        column: this.getColumnNumber(content, match.index),
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        cwe_id: null
                    }
                };

                factBatches['src.SecurityIssue'].push(securityFact);
            }
        });
    }

    // Helper methods
    generateFactId() {
        return `fact_${this.factIdCounter++}_${crypto.randomBytes(8).toString('hex')}`;
    }

    detectLanguage(filePath) {
        const ext = path.extname(filePath);
        const langMap = {
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.py': 'python',
            '.sol': 'solidity',
            '.java': 'java',
            '.cds': 'cds'
        };
        return langMap[ext] || 'unknown';
    }

    mapSyntaxKindToSymbolKind(syntaxKind) {
        const kindMap = {
            'Function': 'function',
            'Method': 'method',
            'Class': 'class',
            'Interface': 'interface',
            'Contract': 'contract',
            'Event': 'event',
            'Modifier': 'modifier',
            'ImportStatement': 'import',
            'ExportStatement': 'export',
            'Variable': 'variable',
            'Field': 'field'
        };
        return kindMap[syntaxKind] || 'unknown';
    }

    mapSymbolRolesToXRefKind(symbolRoles) {
        if (!symbolRoles || symbolRoles.length === 0) return 'reference';

        if (symbolRoles.includes('Definition')) return 'definition';
        if (symbolRoles.includes('Reference')) return 'reference';
        if (symbolRoles.includes('Call')) return 'call';

        return 'reference';
    }

    extractSymbolName(symbol) {
        // Extract actual symbol name from SCIP symbol
        if (symbol.name) return symbol.name;

        // Parse from symbol ID if needed
        const parts = symbol.symbol.split(' ');
        return parts[parts.length - 1] || symbol.symbol;
    }

    extractVisibility(symbol) {
        // Extract visibility from symbol or default to public
        return symbol.visibility || 'public';
    }

    calculateChecksum(content) {
        return crypto.createHash('sha256').update(content).digest('hex');
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length - 1;
    }

    getColumnNumber(content, index) {
        const lines = content.substring(0, index).split('\n');
        return lines[lines.length - 1].length;
    }

    getCWEId(issueType) {
        const cweMap = {
            'unsafe_eval': 95,
            'xss': 79,
            'hardcoded_secret': 798,
            'sql_injection': 89,
            'command_injection': 78
        };
        return cweMap[issueType] || null;
    }

    getDocumentContent(document) {
        // Try to read the actual file content for security analysis
        try {
            const fs = require('fs');
            const path = require('path');

            // Construct full file path
            const fullPath = path.resolve(document.relative_path);

            // Check if file exists and read content
            if (fs.existsSync(fullPath)) {
                return await fs.readFile(fullPath, 'utf8');
            }

            // Fallback: try relative to current working directory
            const cwdPath = path.join(process.cwd(), document.relative_path);
            if (fs.existsSync(cwdPath)) {
                return await fs.readFile(cwdPath, 'utf8');
            }

        } catch (error) {
            console.warn(`Could not read file content for ${document.relative_path}: ${error.message}`);
        }

        return null;
    }

    extractFunctionInfo(symbol, document) {
        // Extract function-specific information
        return {
            name: this.extractSymbolName(symbol),
            signature: '',
            returnType: null,
            parameters: [],
            isAsync: false,
            isExported: false,
            body: ''
        };
    }

    extractClassInfo(symbol, document) {
        // Extract class-specific information
        return {
            name: this.extractSymbolName(symbol),
            extends: null,
            implements: [],
            isAbstract: false,
            isExported: false,
            methods: [],
            fields: []
        };
    }

    extractContractInfo(symbol, document) {
        // Extract Solidity contract information
        return {
            name: this.extractSymbolName(symbol),
            extends: null,
            functions: [],
            events: [],
            modifiers: [],
            stateVariables: []
        };
    }

    extractImportInfo(symbol, document) {
        // Extract import information
        return {
            module: '',
            importedNames: [],
            defaultImport: null,
            namespaceImport: null,
            resolvedPath: null
        };
    }

    extractExportInfo(symbol, document) {
        // Extract export information
        return {
            name: this.extractSymbolName(symbol),
            exportType: 'named'
        };
    }

    calculateComplexity(functionBody) {
        // Basic cyclomatic complexity calculation
        if (!functionBody) return 1;

        const complexityPatterns = [
            /\bif\s*\(/g,
            /\belse\s+if\s*\(/g,
            /\bwhile\s*\(/g,
            /\bfor\s*\(/g,
            /\bswitch\s*\(/g,
            /\bcase\s+/g,
            /\bcatch\s*\(/g,
            /\?\s*[^:]+:/g // ternary operators
        ];

        let complexity = 1; // Base complexity
        complexityPatterns.forEach(pattern => {
            const matches = functionBody.match(pattern);
            if (matches) complexity += matches.length;
        });

        return complexity;
    }

    detectJavaScriptCodeSmells(content, document, factBatches) {
        // Detect long functions
        const functionMatches = content.match(/function\s+\w+\s*\([^)]*\)\s*\{[\s\S]*?\}/g) || [];
        functionMatches.forEach((func, index) => {
            const lines = func.split('\n').length;
            if (lines > 50) {
                const lineNumber = this.getLineNumber(content, content.indexOf(func));

                const smellFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        smell_type: 'long_method'
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        smell_type: 'long_method',
                        severity: 'major',
                        metric_value: lines,
                        threshold: 50
                    }
                };

                factBatches['src.CodeSmell'].push(smellFact);
            }
        });
    }

    extractCDSEntities(content, document, factBatches) {
        // Extract CDS entity definitions
        const entityRegex = /entity\s+(\w+)\s*(?::\s*([^{]+))?\s*\{/g;
        let match;

        while ((match = entityRegex.exec(content)) !== null) {
            const entityName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);

            const entityFact = {
                id: this.generateFactId(),
                key: {
                    file: document.relative_path,
                    name: entityName
                },
                value: {
                    file: document.relative_path,
                    name: entityName,
                    namespace: this.extractNamespace(content),
                    line: lineNumber,
                    fields: [],
                    keys: [],
                    associations: []
                }
            };

            factBatches['src.CDSEntity'].push(entityFact);
        }
    }

    extractCDSServices(content, document, factBatches) {
        // Extract CDS service definitions
        const serviceRegex = /service\s+(\w+)\s*\{/g;
        let match;

        while ((match = serviceRegex.exec(content)) !== null) {
            const serviceName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);

            const serviceFact = {
                id: this.generateFactId(),
                key: {
                    file: document.relative_path,
                    name: serviceName
                },
                value: {
                    file: document.relative_path,
                    name: serviceName,
                    namespace: this.extractNamespace(content),
                    line: lineNumber,
                    entities: [],
                    actions: [],
                    functions: []
                }
            };

            factBatches['src.CDSService'].push(serviceFact);
        }
    }

    extractNamespace(content) {
        const nsMatch = content.match(/namespace\s+([\w.]+)/);
        return nsMatch ? nsMatch[1] : '';
    }

    detectPythonSecurityIssues(content, document, factBatches) {
        // Python-specific security patterns
        const patterns = [
            {
                pattern: /exec\s*\(/g,
                type: 'unsafe_eval',
                severity: 'high',
                description: 'Use of exec() can lead to code injection'
            },
            {
                pattern: /subprocess\.call\([^)]*shell\s*=\s*True/g,
                type: 'command_injection',
                severity: 'high',
                description: 'subprocess with shell=True can be dangerous'
            }
        ];

        patterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const securityFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        column: this.getColumnNumber(content, match.index),
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        cwe_id: this.getCWEId(pattern.type)
                    }
                };

                factBatches['src.SecurityIssue'].push(securityFact);
            }
        });
    }

    detectPythonPerformanceIssues(content, document, factBatches) {
        // Python-specific performance patterns
        const patterns = [
            {
                pattern: /\+\s*=.*\[\]/g,
                type: 'inefficient_loop',
                severity: 'low',
                description: 'String concatenation in loop is inefficient',
                suggestion: 'Use list comprehension or join()'
            }
        ];

        patterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const perfFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        suggestion: pattern.suggestion
                    }
                };

                factBatches['src.PerformanceIssue'].push(perfFact);
            }
        });
    }

    detectSolidityPerformanceIssues(content, document, factBatches) {
        // Solidity-specific performance patterns
        const patterns = [
            {
                pattern: /for\s*\([^)]*;\s*\w+\.length;/g,
                type: 'inefficient_loop',
                severity: 'medium',
                description: 'Array length lookup in loop condition is inefficient',
                suggestion: 'Cache array length before loop'
            }
        ];

        patterns.forEach(pattern => {
            let match;
            while ((match = pattern.pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);

                const perfFact = {
                    id: this.generateFactId(),
                    key: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type
                    },
                    value: {
                        file: document.relative_path,
                        line: lineNumber,
                        issue_type: pattern.type,
                        severity: pattern.severity,
                        description: pattern.description,
                        suggestion: pattern.suggestion
                    }
                };

                factBatches['src.PerformanceIssue'].push(perfFact);
            }
        });
    }

    processExternalSymbol(symbol, factBatches) {
        // Process external symbols (from dependencies)
        // These represent symbols defined outside the current codebase
    }
}

module.exports = GleanFactTransformer;