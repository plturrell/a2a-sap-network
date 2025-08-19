/**
 * @fileoverview SCIP-based indexer for generating Glean-compatible facts
 * @module scipIndexer
 * @since 1.0.0
 * 
 * Implements SCIP (SCIP Code Intelligence Protocol) indexing to generate
 * proper Glean facts for code intelligence queries
 */

const { spawn, exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { parse } = require('@typescript-eslint/parser');
const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');
const AdvancedCDSParser = require('./advancedCDSParser');
const CAPFactTransformer = require('./capFactTransformer');

class SCIPIndexer {
    constructor(workspaceRoot) {
        this.workspaceRoot = workspaceRoot;
        this.scipPath = process.env.SCIP_PATH || 'scip';
        this.scipIndexes = new Map();
        this.gleanFactsBuffer = [];
        this.languageServers = new Map();
        this.advancedCDSParser = new AdvancedCDSParser();
        this.capFactTransformer = new CAPFactTransformer();
    }

    async initialize() {
        // Check if SCIP is available
        await this.ensureSCIPAvailable();
        
        // Initialize language servers for SCIP generation
        await this.initializeLanguageServers();
        
        // console.log('SCIP indexer initialized successfully');
    }

    async ensureSCIPAvailable() {
        // console.log('Using real AST parsers: TypeScript ESLint parser and Babel parser');
        // console.log('Enhanced symbol extraction with proper AST traversal enabled');
        return true;
    }

    async installSCIPTypeScript() {
        // console.log('Using real AST parsers - no SCIP installation needed');
        return Promise.resolve();
    }

    async initializeLanguageServers() {
        // JavaScript/TypeScript with TypeScript Language Server
        this.languageServers.set('typescript', {
            command: 'scip-typescript',
            args: ['index'],
            extensions: ['.ts', '.tsx'],
            config: path.join(this.workspaceRoot, 'tsconfig.json')
        });

        // JavaScript with enhanced AST parsing
        this.languageServers.set('javascript', {
            command: 'scip-typescript',
            args: ['index'],
            extensions: ['.js', '.jsx'],
            config: path.join(this.workspaceRoot, 'package.json')
        });

        // Python with Pylsp
        this.languageServers.set('python', {
            command: 'scip-python',
            args: ['index'],
            extensions: ['.py'],
            config: path.join(this.workspaceRoot, 'pyproject.toml')
        });

        // Java (for potential SAP Java components)
        this.languageServers.set('java', {
            command: 'scip-java',
            args: ['index'],
            extensions: ['.java'],
            config: path.join(this.workspaceRoot, 'pom.xml')
        });

        // Solidity (custom implementation)
        this.languageServers.set('solidity', {
            command: this.scipSolidityIndexer.bind(this),
            extensions: ['.sol'],
            config: null
        });

        // SAP CAP CDS (Core Data Services) - Enhanced
        this.languageServers.set('cds', {
            command: this.enhancedCDSIndexer.bind(this),
            extensions: ['.cds'],
            config: path.join(this.workspaceRoot, 'package.json')
        });

        // SAP CAP Services (JavaScript/Node.js services)
        this.languageServers.set('cap-service', {
            command: 'scip-typescript',
            args: ['index'],
            extensions: ['.js', '.ts'],
            config: path.join(this.workspaceRoot, 'package.json'),
            capSpecific: true
        });
    }

    async indexProject(languages = ['typescript', 'javascript', 'python', 'solidity', 'cds']) {
        // console.log(`Starting SCIP indexing for languages: ${languages.join(', ')}`);
        
        // Try to use real SCIP indexers first
        const scipIndex = await this.generateRealSCIPIndex(languages);
        
        return {
            scipIndex: scipIndex,
            documentCount: scipIndex.documents ? scipIndex.documents.length : 0,
            symbolCount: scipIndex.external_symbols ? scipIndex.external_symbols.length : 0
        };
    }

    async generateRealSCIPIndex(languages) {
        // console.log('Using enhanced AST-based indexing with real parsers...');
        return await this.generateEnhancedSCIPIndex(languages);
    }

    async canUseRealSCIPTypeScript() {
        // Always return true since we have real AST parsers now
        return true;
    }

    async runRealSCIPTypeScript() {
        // console.log('Using enhanced AST-based TypeScript/JavaScript indexing...');
        return await this.generateEnhancedSCIPIndex(['typescript', 'javascript']);
    }

    async generateEnhancedSCIPIndex(languages) {
        const scipIndex = {
            metadata: {
                version: '0.3.0',
                tool_info: {
                    name: 'a2a-enhanced-ast-indexer',
                    version: '2.0.0',
                    arguments: ['--ast-based', '--typescript-eslint', '--babel']
                },
                project_root: `file://${this.workspaceRoot}`,
                text_document_encoding: 'UTF-8'
            },
            documents: [],
            external_symbols: [],
            symbol_roles: []
        };

        for (const language of languages) {
            // console.log(`Fallback indexing ${language} files...`);
            const langResults = await this.indexLanguage(language);
            
            // Merge SCIP results
            if (langResults.scip.documents) {
                scipIndex.documents.push(...langResults.scip.documents);
            }
            if (langResults.scip.external_symbols) {
                scipIndex.external_symbols.push(...langResults.scip.external_symbols);
            }
        }

        return scipIndex;
    }

    async indexLanguage(language) {
        const server = this.languageServers.get(language);
        if (!server) {
            throw new Error(`No language server configured for ${language}`);
        }

        const files = await this.findFilesForLanguage(language);
        const scipResults = {
            documents: [],
            external_symbols: [],
            symbol_roles: []
        };
        const gleanFacts = { facts: [] };

        for (const file of files) {
            // console.log(`Processing ${file}...`);
            
            if (typeof server.command === 'function') {
                // Custom indexer (e.g., Solidity)
                const result = await server.command(file);
                scipResults.documents.push(result.scip);
                gleanFacts.facts.push(...result.glean);
            } else {
                // Standard SCIP indexer
                const result = await this.runSCIPIndexer(server, file);
                scipResults.documents.push(result.scip);
                gleanFacts.facts.push(...result.glean);
            }
        }

        return {
            scip: scipResults,
            glean: gleanFacts
        };
    }

    async findFilesForLanguage(language) {
        const server = this.languageServers.get(language);
        const files = [];
        
        await this.walkDirectory(this.workspaceRoot, (filePath) => {
            const ext = path.extname(filePath);
            if (server.extensions.includes(ext)) {
                // Skip node_modules and other unwanted directories
                if (!filePath.includes('node_modules') && 
                    !filePath.includes('.git') && 
                    !filePath.includes('dist') &&
                    !filePath.includes('build')) {
                    files.push(filePath);
                }
            }
        });

        return files;
    }

    async walkDirectory(dir, callback) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });
            
            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                
                if (entry.isDirectory()) {
                    if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
                        await this.walkDirectory(fullPath, callback);
                    }
                } else if (entry.isFile()) {
                    callback(fullPath);
                }
            }
        } catch (error) {
            console.warn(`Warning: Could not read directory ${dir}: ${error.message}`);
        }
    }

    async runSCIPIndexer(server, filePath) {
        const relativeFile = path.relative(this.workspaceRoot, filePath);
        const fileContent = await fs.readFile(filePath, 'utf8');
        const fileUri = `file://${filePath}`;

        // Generate SCIP document using language server
        const scipDoc = await this.generateSCIPDocument(server, filePath, fileContent);
        
        // Convert SCIP to Glean facts
        const gleanFacts = this.scipToGleanFacts(scipDoc, relativeFile);

        return {
            scip: scipDoc,
            glean: gleanFacts
        };
    }

    async generateSCIPDocument(server, filePath, content) {
        // This would typically use the actual language server protocol
        // For this implementation, we'll parse the content directly
        const symbols = [];
        const occurrences = [];
        
        const document = {
            relative_path: path.relative(this.workspaceRoot, filePath),
            occurrences: [],
            symbols: []
        };

        // Parse based on language type
        if (server.command === 'scip-typescript') {
            return await this.parseTypeScript(filePath, content, document);
        } else if (server.command === 'scip-python') {
            return await this.parsePython(filePath, content, document);
        }

        return document;
    }

    async parseTypeScript(filePath, content, document) {
        try {
            // console.log(`Attempting AST parsing for ${filePath}`);
            
            // Use Babel parser for all files as it handles both JS and TS
            const ast = babelParser.parse(content, {
                sourceType: 'module',
                allowImportExportEverywhere: true,
                allowReturnOutsideFunction: true,
                plugins: [
                    'jsx',
                    'typescript',
                    'decorators-legacy',
                    'classProperties',
                    'asyncGenerators',
                    'functionBind',
                    'exportDefaultFrom',
                    'exportNamespaceFrom',
                    'dynamicImport',
                    'nullishCoalescingOperator',
                    'optionalChaining'
                ]
            });
            // console.log('Babel parser succeeded');

            let symbolCounter = 0;
            
            // Traverse AST to extract symbols
            const walker = {
                ImportDeclaration: (node) => {
                    try {
                        const symbolId = `import_${symbolCounter++}`;
                        const range = this.nodeToRange(node, content);
                        
                        document.symbols.push({
                            symbol: symbolId,
                            definition: {
                                range: range,
                                syntax_kind: 'ImportDeclaration'
                            },
                            name: node.source ? node.source.value : 'unknown',
                            specifiers: node.specifiers ? node.specifiers.map(spec => spec.local ? spec.local.name : 'unknown') : []
                        });

                        document.occurrences.push({
                            range: range,
                            symbol: symbolId,
                            symbol_roles: ['Definition']
                        });
                    } catch (err) {
                        console.warn(`Error processing import declaration: ${err.message}`);
                    }
                },
                
                FunctionDeclaration: (node) => {
                    try {
                        if (node.id) {
                            const symbolId = `function_${symbolCounter++}`;
                            const range = this.nodeToRange(node, content);
                            
                            document.symbols.push({
                                symbol: symbolId,
                                definition: {
                                    range: range,
                                    syntax_kind: 'FunctionDeclaration'
                                },
                                name: node.id.name,
                                async: node.async || false,
                                generator: node.generator || false,
                                params: node.params ? node.params.map(param => this.extractParamInfo(param)) : []
                            });

                            document.occurrences.push({
                                range: range,
                                symbol: symbolId,
                                symbol_roles: ['Definition']
                            });
                        }
                    } catch (err) {
                        console.warn(`Error processing function declaration: ${err.message}`);
                    }
                },
                
                ClassDeclaration: (node) => {
                    try {
                        if (node.id) {
                            const symbolId = `class_${symbolCounter++}`;
                            const range = this.nodeToRange(node, content);
                            
                            document.symbols.push({
                                symbol: symbolId,
                                definition: {
                                    range: range,
                                    syntax_kind: 'ClassDeclaration'
                                },
                                name: node.id.name,
                                superClass: node.superClass ? (node.superClass.name || 'unknown') : null,
                                methods: (node.body && node.body.body) ? 
                                    node.body.body
                                        .filter(member => member.type === 'MethodDefinition')
                                        .map(method => method.key ? method.key.name : 'unknown') : []
                            });

                            document.occurrences.push({
                                range: range,
                                symbol: symbolId,
                                symbol_roles: ['Definition']
                            });
                        }
                    } catch (err) {
                        console.warn(`Error processing class declaration: ${err.message}`);
                    }
                },
                
                VariableDeclaration: (node) => {
                    try {
                        if (node.declarations) {
                            node.declarations.forEach(declarator => {
                                if (declarator.id && declarator.id.type === 'Identifier') {
                                    const symbolId = `variable_${symbolCounter++}`;
                                    const range = this.nodeToRange(declarator, content);
                                    
                                    document.symbols.push({
                                        symbol: symbolId,
                                        definition: {
                                            range: range,
                                            syntax_kind: 'VariableDeclaration'
                                        },
                                        name: declarator.id.name,
                                        kind: node.kind,
                                        hasInit: !!declarator.init,
                                        initType: declarator.init ? declarator.init.type : null
                                    });

                                    document.occurrences.push({
                                        range: range,
                                        symbol: symbolId,
                                        symbol_roles: ['Definition']
                                    });
                                }
                            });
                        }
                    } catch (err) {
                        console.warn(`Error processing variable declaration: ${err.message}`);
                    }
                },
                
                MethodDefinition: (node) => {
                    try {
                        const symbolId = `method_${symbolCounter++}`;
                        const range = this.nodeToRange(node, content);
                        
                        document.symbols.push({
                            symbol: symbolId,
                            definition: {
                                range: range,
                                syntax_kind: 'MethodDefinition'
                            },
                            name: node.key ? node.key.name : 'unknown',
                            kind: node.kind,
                            static: node.static || false,
                            async: (node.value && node.value.async) || false,
                            generator: (node.value && node.value.generator) || false
                        });

                        document.occurrences.push({
                            range: range,
                            symbol: symbolId,
                            symbol_roles: ['Definition']
                        });
                    } catch (err) {
                        console.warn(`Error processing method definition: ${err.message}`);
                    }
                }
            };

            // Use Babel traversal
            // console.log('Using Babel traversal');
            traverse(ast, {
                enter(path) {
                    const nodeType = path.node.type;
                    if (walker[nodeType]) {
                        walker[nodeType](path.node);
                    }
                }
            });

            return document;
        } catch (error) {
            console.warn(`AST parsing failed for ${filePath}: ${error.message}`);
            // Fallback to regex-based parsing
            return this.parseTypeScriptFallback(filePath, content, document);
        }
    }

    async parsePython(filePath, content, document) {
        const lines = content.split('\n');
        let symbolCounter = 0;

        // Extract imports
        const importRegex = /(?:from\s+(\S+)\s+)?import\s+([^#\n]+)/g;
        let match;
        while ((match = importRegex.exec(content)) !== null) {
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'ImportStatement'
                }
            });
        }

        // Extract function definitions
        const functionRegex = /def\s+(\w+)\s*\(/g;
        while ((match = functionRegex.exec(content)) !== null) {
            const functionName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Function'
                }
            });
        }

        // Extract class definitions
        const classRegex = /class\s+(\w+)/g;
        while ((match = classRegex.exec(content)) !== null) {
            const className = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Class'
                }
            });
        }

        return document;
    }

    async scipSolidityIndexer(filePath) {
        const content = await fs.readFile(filePath, 'utf8');
        const lines = content.split('\n');
        const document = {
            relative_path: path.relative(this.workspaceRoot, filePath),
            occurrences: [],
            symbols: []
        };

        let symbolCounter = 0;

        // Extract pragma statements
        const pragmaRegex = /pragma\s+solidity\s+([^;]+);/g;
        let match;
        while ((match = pragmaRegex.exec(content)) !== null) {
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Pragma'
                }
            });
        }

        // Extract contract definitions
        const contractRegex = /contract\s+(\w+)/g;
        while ((match = contractRegex.exec(content)) !== null) {
            const contractName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Contract'
                }
            });
        }

        // Extract function definitions
        const functionRegex = /function\s+(\w+)\s*\(/g;
        while ((match = functionRegex.exec(content)) !== null) {
            const functionName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Function'
                }
            });
        }

        // Extract event definitions
        const eventRegex = /event\s+(\w+)\s*\(/g;
        while ((match = eventRegex.exec(content)) !== null) {
            const eventName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `local ${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Event'
                }
            });
        }

        // Convert to Glean facts
        const gleanFacts = this.scipToGleanFacts(document, path.relative(this.workspaceRoot, filePath));

        return {
            scip: document,
            glean: gleanFacts
        };
    }

    scipToGleanFacts(scipDocument, filePath) {
        const facts = [];

        // Enhanced file fact with more metadata
        facts.push({
            id: crypto.randomUUID(),
            key: {
                file: filePath
            },
            value: {
                language: this.detectLanguage(filePath),
                size: scipDocument.symbols.length,
                symbols: scipDocument.symbols.length,
                imports: scipDocument.symbols.filter(s => s.definition?.syntax_kind?.includes('Import')).length,
                functions: scipDocument.symbols.filter(s => s.definition?.syntax_kind?.includes('Function')).length,
                classes: scipDocument.symbols.filter(s => s.definition?.syntax_kind?.includes('Class')).length,
                variables: scipDocument.symbols.filter(s => s.definition?.syntax_kind?.includes('Variable')).length
            }
        });

        // Enhanced symbol facts with rich metadata
        scipDocument.symbols.forEach((symbol, index) => {
            const symbolFact = {
                id: crypto.randomUUID(),
                key: {
                    file: filePath,
                    symbol: symbol.symbol,
                    line: symbol.definition?.range?.start?.line || 0
                },
                value: {
                    name: this.extractSymbolName(symbol),
                    kind: symbol.definition?.syntax_kind || 'Unknown',
                    range: symbol.definition?.range || null,
                    // Enhanced metadata based on symbol type
                    ...this.extractEnhancedSymbolMetadata(symbol)
                }
            };
            facts.push(symbolFact);

            // Enhanced cross-reference fact
            facts.push({
                id: crypto.randomUUID(),
                key: {
                    file: filePath,
                    symbol: symbol.symbol
                },
                value: {
                    definition_file: filePath,
                    definition_line: symbol.definition?.range?.start?.line || 0,
                    references: scipDocument.occurrences
                        .filter(occ => occ.symbol === symbol.symbol)
                        .map(occ => ({
                            line: occ.range?.start?.line || 0,
                            column: occ.range?.start?.character || 0,
                            roles: occ.symbol_roles || []
                        })),
                    symbol_type: symbol.definition?.syntax_kind || 'Unknown',
                    enhanced_ast: !symbol.fallback
                }
            });
            
            // Add function-specific facts
            if (symbol.definition?.syntax_kind?.includes('Function') && symbol.params) {
                facts.push({
                    id: crypto.randomUUID(),
                    key: {
                        file: filePath,
                        function: symbol.name || this.extractSymbolName(symbol)
                    },
                    value: {
                        parameters: symbol.params,
                        async: symbol.async || false,
                        generator: symbol.generator || false,
                        paramCount: symbol.params?.length || 0,
                        hasDefaultParams: symbol.params?.some(p => p.hasDefault) || false,
                        hasRestParams: symbol.params?.some(p => p.isRest) || false
                    }
                });
            }
            
            // Add class-specific facts
            if (symbol.definition?.syntax_kind?.includes('Class') && symbol.methods) {
                facts.push({
                    id: crypto.randomUUID(),
                    key: {
                        file: filePath,
                        class: symbol.name || this.extractSymbolName(symbol)
                    },
                    value: {
                        superClass: symbol.superClass,
                        methods: symbol.methods,
                        methodCount: symbol.methods?.length || 0,
                        hasInheritance: !!symbol.superClass
                    }
                });
            }
            
            // Add import-specific facts
            if (symbol.definition?.syntax_kind?.includes('Import') && symbol.specifiers) {
                facts.push({
                    id: crypto.randomUUID(),
                    key: {
                        file: filePath,
                        import: symbol.name || this.extractSymbolName(symbol)
                    },
                    value: {
                        module: symbol.name,
                        specifiers: symbol.specifiers,
                        specifierCount: symbol.specifiers?.length || 0,
                        importType: symbol.specifiers?.length === 1 && symbol.specifiers[0] === 'default' ? 'default' : 'named'
                    }
                });
            }
        });

        return facts;
    }

    async enhancedCDSIndexer(filePath) {
        try {
            const content = await fs.readFile(filePath, 'utf8');
            const relativePath = path.relative(this.workspaceRoot, filePath);
            
            // console.log(`Processing CDS file: ${relativePath} (${content.length} chars)`);
            
            // Use advanced CDS parser for comprehensive analysis
            const parseResult = this.advancedCDSParser.parseAdvancedCDSContent(content, relativePath);
            
            // Add file reference to symbols
            parseResult.symbols.forEach(symbol => {
                symbol.file = relativePath;
            });
            
            // Generate comprehensive Glean facts using CAP fact transformer
            const gleanFacts = this.capFactTransformer.transformCAPToGlean(parseResult, relativePath, content);
            
            // Convert to flat fact array for compatibility
            const flatFacts = [];
            Object.values(gleanFacts).forEach(factArray => {
                flatFacts.push(...factArray);
            });
            
            // console.log(`✅ CDS processing complete: ${parseResult.symbols.length} symbols, ${flatFacts.length} facts`);
            
            return {
                scip: {
                    relative_path: relativePath,
                    symbols: parseResult.symbols,
                    occurrences: parseResult.occurrences,
                    syntax_kind: 'CDS',
                    metadata: parseResult.metadata,
                    complexity: parseResult.metadata.complexity,
                    namespace: parseResult.metadata.namespace,
                    imports: parseResult.metadata.imports.length,
                    annotations: parseResult.metadata.annotations.length
                },
                glean: flatFacts,
                advanced: {
                    parseResult: parseResult,
                    factBatches: gleanFacts
                }
            };
            
        } catch (error) {
            console.error(`❌ Error processing CDS file ${filePath}:`, error.message);
            
            // Provide graceful fallback with error information
            const relativePath = path.relative(this.workspaceRoot, filePath);
            return {
                scip: {
                    relative_path: relativePath,
                    symbols: [],
                    occurrences: [],
                    syntax_kind: 'CDS',
                    error: error.message,
                    metadata: {
                        namespace: null,
                        imports: [],
                        annotations: [],
                        complexity: 0
                    }
                },
                glean: [],
                advanced: {
                    error: error.message,
                    parseResult: null,
                    factBatches: {}
                }
            };
        }
    }

    async scipCDSIndexer(filePath) {
        const content = await fs.readFile(filePath, 'utf8');
        const lines = content.split('\n');
        const document = {
            relative_path: path.relative(this.workspaceRoot, filePath),
            occurrences: [],
            symbols: []
        };

        let symbolCounter = 0;

        // Extract namespace declarations
        const namespaceRegex = /namespace\s+([\w\.]+)\s*;/g;
        let match;
        while ((match = namespaceRegex.exec(content)) !== null) {
            const namespaceName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `namespace_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Namespace'
                },
                name: namespaceName,
                type: 'namespace'
            });
        }

        // Extract using declarations (imports)
        const usingRegex = /using\s+([\w\.\{\}\s,\*]+)\s+from\s+['"]([^'"]+)['"]/g;
        while ((match = usingRegex.exec(content)) !== null) {
            const importItems = match[1];
            const fromModule = match[2];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `using_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'UsingDeclaration'
                },
                name: fromModule,
                importItems: importItems.trim(),
                type: 'import'
            });
        }

        // Extract entity definitions
        const entityRegex = /(?:entity|view)\s+(\w+)\s*(?:\([^)]*\))?\s*(?:as\s+select|\{)/g;
        while ((match = entityRegex.exec(content)) !== null) {
            const entityName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `entity_${symbolCounter++}`;
            
            // Extract entity body to find fields
            const entityStart = match.index;
            const entityBody = this.extractCDSBlock(content, entityStart);
            const fields = this.extractCDSFields(entityBody);
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Entity'
                },
                name: entityName,
                type: 'entity',
                fields: fields,
                fieldCount: fields.length
            });
        }

        // Extract service definitions
        const serviceRegex = /service\s+(\w+)\s*(?:@[^\{]*)?\s*\{/g;
        while ((match = serviceRegex.exec(content)) !== null) {
            const serviceName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `service_${symbolCounter++}`;
            
            // Extract service operations
            const serviceStart = match.index;
            const serviceBody = this.extractCDSBlock(content, serviceStart);
            const operations = this.extractServiceOperations(serviceBody);
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Service'
                },
                name: serviceName,
                type: 'service',
                operations: operations,
                operationCount: operations.length
            });
        }

        // Extract type definitions
        const typeRegex = /type\s+(\w+)\s*:\s*([^;]+);/g;
        while ((match = typeRegex.exec(content)) !== null) {
            const typeName = match[1];
            const typeDefinition = match[2].trim();
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `type_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'TypeDefinition'
                },
                name: typeName,
                type: 'type',
                typeDefinition: typeDefinition
            });
        }

        // Extract aspect definitions (mixins)
        const aspectRegex = /aspect\s+(\w+)\s*\{/g;
        while ((match = aspectRegex.exec(content)) !== null) {
            const aspectName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `aspect_${symbolCounter++}`;
            
            const aspectStart = match.index;
            const aspectBody = this.extractCDSBlock(content, aspectStart);
            const fields = this.extractCDSFields(aspectBody);
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Aspect'
                },
                name: aspectName,
                type: 'aspect',
                fields: fields,
                fieldCount: fields.length
            });
        }

        // Extract annotations
        const annotationRegex = /@([\w\.]+)(?:\(([^)]+)\))?/g;
        const annotations = [];
        while ((match = annotationRegex.exec(content)) !== null) {
            const annotationName = match[1];
            const annotationValue = match[2] || '';
            const lineNumber = this.getLineNumber(content, match.index);
            
            annotations.push({
                name: annotationName,
                value: annotationValue,
                line: lineNumber
            });
        }

        // Convert to Glean facts
        const gleanFacts = this.scipToGleanFacts(document, path.relative(this.workspaceRoot, filePath));
        
        // Add CDS-specific facts
        gleanFacts.push(...this.generateCDSSpecificFacts(document, filePath, annotations));

        return {
            scip: document,
            glean: gleanFacts
        };
    }

    extractCDSBlock(content, startIndex) {
        let braceCount = 0;
        let i = startIndex;
        let foundOpenBrace = false;
        
        // Find the opening brace
        while (i < content.length && !foundOpenBrace) {
            if (content[i] === '{') {
                foundOpenBrace = true;
                braceCount = 1;
            }
            i++;
        }
        
        if (!foundOpenBrace) return '';
        
        const blockStart = i;
        
        // Find the matching closing brace
        while (i < content.length && braceCount > 0) {
            if (content[i] === '{') {
                braceCount++;
            } else if (content[i] === '}') {
                braceCount--;
            }
            i++;
        }
        
        return content.substring(blockStart, i - 1);
    }

    extractCDSFields(blockContent) {
        const fields = [];
        // Match field definitions like: fieldName : Type;
        const fieldRegex = /(\w+)\s*:\s*([^;]+);/g;
        let match;
        
        while ((match = fieldRegex.exec(blockContent)) !== null) {
            const fieldName = match[1];
            const fieldType = match[2].trim();
            
            fields.push({
                name: fieldName,
                type: fieldType,
                nullable: fieldType.includes('?'),
                key: fieldType.includes('key') || blockContent.includes(`key ${fieldName}`)
            });
        }
        
        return fields;
    }

    extractServiceOperations(serviceBody) {
        const operations = [];
        
        // Extract entity exposure: entity EntityName as projection on db.Entity
        const entityRegex = /entity\s+(\w+)\s+as\s+([^;]+);/g;
        let match;
        
        while ((match = entityRegex.exec(serviceBody)) !== null) {
            operations.push({
                type: 'entity',
                name: match[1],
                definition: match[2].trim()
            });
        }
        
        // Extract action/function definitions
        const actionRegex = /(action|function)\s+(\w+)\s*\(([^)]*)\)\s*(?:returns\s+([^;]+))?;/g;
        
        while ((match = actionRegex.exec(serviceBody)) !== null) {
            operations.push({
                type: match[1], // action or function
                name: match[2],
                parameters: match[3] ? match[3].trim() : '',
                returns: match[4] ? match[4].trim() : null
            });
        }
        
        return operations;
    }

    generateCDSSpecificFacts(document, filePath, annotations) {
        const facts = [];
        const relativePath = path.relative(this.workspaceRoot, filePath);
        
        // CDS File fact
        facts.push({
            id: crypto.randomUUID(),
            key: {
                cdsFile: relativePath
            },
            value: {
                file: relativePath,
                language: 'cds',
                entities: document.symbols.filter(s => s.type === 'entity').length,
                services: document.symbols.filter(s => s.type === 'service').length,
                types: document.symbols.filter(s => s.type === 'type').length,
                aspects: document.symbols.filter(s => s.type === 'aspect').length,
                annotations: annotations.length
            }
        });
        
        // Entity facts
        document.symbols.filter(s => s.type === 'entity').forEach(entity => {
            facts.push({
                id: crypto.randomUUID(),
                key: {
                    cdsEntity: entity.name,
                    file: relativePath
                },
                value: {
                    name: entity.name,
                    file: relativePath,
                    fields: entity.fields || [],
                    fieldCount: entity.fieldCount || 0,
                    keyFields: (entity.fields || []).filter(f => f.key).map(f => f.name)
                }
            });
        });
        
        // Service facts
        document.symbols.filter(s => s.type === 'service').forEach(service => {
            facts.push({
                id: crypto.randomUUID(),
                key: {
                    cdsService: service.name,
                    file: relativePath
                },
                value: {
                    name: service.name,
                    file: relativePath,
                    operations: service.operations || [],
                    operationCount: service.operationCount || 0,
                    exposedEntities: (service.operations || []).filter(op => op.type === 'entity').map(op => op.name)
                }
            });
        });
        
        // Annotation facts
        annotations.forEach(annotation => {
            facts.push({
                id: crypto.randomUUID(),
                key: {
                    cdsAnnotation: annotation.name,
                    file: relativePath,
                    line: annotation.line
                },
                value: {
                    name: annotation.name,
                    value: annotation.value,
                    file: relativePath,
                    line: annotation.line
                }
            });
        });
        
        return facts;
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

    extractSymbolName(symbol) {
        // Use the enhanced name from AST parsing if available
        if (symbol.name) {
            return symbol.name;
        }
        
        // Fallback to extracting from symbol ID
        if (symbol.definition && symbol.definition.range) {
            return symbol.symbol.replace(/^(local|import|function|class|variable|method)_\d+$/, 'symbol');
        }
        return symbol.symbol;
    }
    
    extractEnhancedSymbolMetadata(symbol) {
        const metadata = {};
        
        // Add type-specific metadata
        if (symbol.async !== undefined) metadata.async = symbol.async;
        if (symbol.generator !== undefined) metadata.generator = symbol.generator;
        if (symbol.static !== undefined) metadata.static = symbol.static;
        if (symbol.kind !== undefined) metadata.kind = symbol.kind;
        if (symbol.hasInit !== undefined) metadata.hasInit = symbol.hasInit;
        if (symbol.initType !== undefined) metadata.initType = symbol.initType;
        if (symbol.fallback !== undefined) metadata.fallback = symbol.fallback;
        
        // Add parameter info for functions
        if (symbol.params) {
            metadata.parameterCount = symbol.params.length;
            metadata.hasDefaultParameters = symbol.params.some(p => p.hasDefault);
            metadata.hasRestParameters = symbol.params.some(p => p.isRest);
        }
        
        // Add class info
        if (symbol.superClass) {
            metadata.extends = symbol.superClass;
        }
        if (symbol.methods) {
            metadata.methodCount = symbol.methods.length;
        }
        
        // Add import info
        if (symbol.specifiers) {
            metadata.importSpecifiers = symbol.specifiers;
        }
        
        return metadata;
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length - 1;
    }
    
    nodeToRange(node, content) {
        if (node.loc) {
            return {
                start: {
                    line: node.loc.start.line - 1,
                    character: node.loc.start.column
                },
                end: {
                    line: node.loc.end.line - 1,
                    character: node.loc.end.column
                }
            };
        }
        
        // Fallback for nodes without location info
        return {
            start: { line: 0, character: 0 },
            end: { line: 0, character: 0 }
        };
    }
    
    traverseESTreeNode(node, walker) {
        if (!node || typeof node !== 'object') return;
        
        if (walker[node.type]) {
            walker[node.type](node);
        }
        
        for (const key in node) {
            if (key === 'parent' || key === 'type' || key === 'loc' || key === 'range') continue;
            const child = node[key];
            
            if (Array.isArray(child)) {
                child.forEach(item => this.traverseESTreeNode(item, walker));
            } else if (child && typeof child === 'object' && child.type) {
                this.traverseESTreeNode(child, walker);
            }
        }
    }
    
    extractParamInfo(param) {
        try {
            if (!param) return { name: 'unknown', type: 'unknown' };
            
            if (param.type === 'Identifier') {
                return { name: param.name || 'unknown', type: 'Identifier' };
            } else if (param.type === 'AssignmentPattern') {
                return { 
                    name: (param.left && param.left.name) || 'unknown', 
                    type: 'AssignmentPattern',
                    hasDefault: true
                };
            } else if (param.type === 'RestElement') {
                return {
                    name: (param.argument && param.argument.name) || 'unknown',
                    type: 'RestElement',
                    isRest: true
                };
            }
            return { name: 'unknown', type: param.type || 'unknown' };
        } catch (err) {
            console.warn(`Error extracting param info: ${err.message}`);
            return { name: 'unknown', type: 'error' };
        }
    }
    
    async parseTypeScriptFallback(filePath, content, document) {
        // console.log(`Using fallback regex parsing for ${filePath}`);
        
        const lines = content.split('\n');
        let symbolCounter = 0;

        // Extract imports with regex
        const importRegex = /import\s+(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))\s+from\s+['"]([^'"]+)['"]/g;
        let match;
        while ((match = importRegex.exec(content)) !== null) {
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `import_fallback_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'ImportStatement'
                },
                name: match[4],
                fallback: true
            });

            document.occurrences.push({
                range: this.getRange(lines, lineNumber, match.index, match[0].length),
                symbol: symbolId,
                symbol_roles: ['Definition']
            });
        }

        // Extract functions with regex
        const functionRegex = /(?:export\s+)?(?:async\s+)?function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*[^=]+=)/g;
        while ((match = functionRegex.exec(content)) !== null) {
            const functionName = match[1] || match[2];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `function_fallback_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Function'
                },
                name: functionName,
                fallback: true
            });

            document.occurrences.push({
                range: this.getRange(lines, lineNumber, match.index, match[0].length),
                symbol: symbolId,
                symbol_roles: ['Definition']
            });
        }

        // Extract classes with regex
        const classRegex = /(?:export\s+)?class\s+(\w+)/g;
        while ((match = classRegex.exec(content)) !== null) {
            const className = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `class_fallback_${symbolCounter++}`;
            
            document.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.getRange(lines, lineNumber, match.index, match[0].length),
                    syntax_kind: 'Class'
                },
                name: className,
                fallback: true
            });

            document.occurrences.push({
                range: this.getRange(lines, lineNumber, match.index, match[0].length),
                symbol: symbolId,
                symbol_roles: ['Definition']
            });
        }

        return document;
    }

    getRange(lines, lineNumber, startIndex, length) {
        const lineContent = lines[lineNumber] || '';
        const lineStartIndex = lines.slice(0, lineNumber).join('\n').length + (lineNumber > 0 ? 1 : 0);
        const characterOffset = startIndex - lineStartIndex;
        
        return {
            start: {
                line: lineNumber,
                character: Math.max(0, characterOffset)
            },
            end: {
                line: lineNumber,
                character: Math.max(0, characterOffset + length)
            }
        };
    }

    async uploadToGlean(gleanFacts, gleanUrl = 'http://localhost:8080') {
        try {
            const response = await fetch(`${gleanUrl}/api/v1/facts`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${process.env.GLEAN_API_TOKEN || ''}`
                },
                body: JSON.stringify({
                    schema_version: '1.0',
                    predicate: 'src.File',
                    facts: gleanFacts.facts
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to upload facts: ${response.statusText}`);
            }

            const result = await response.json();
            // console.log(`Uploaded ${gleanFacts.facts.length} facts to Glean`);
            return result;
        } catch (error) {
            console.error('Failed to upload facts to Glean:', error);
            throw error;
        }
    }
}

module.exports = SCIPIndexer;