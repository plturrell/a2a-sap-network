/**
 * @fileoverview CAP Service Implementation Analyzer
 * @module capServiceAnalyzer
 * @since 1.0.0
 * 
 * Analyzes CAP service implementation files (JavaScript/TypeScript) to detect
 * event handlers, middleware, and CAP-specific patterns
 */

const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');

class CAPServiceAnalyzer {
    constructor() {
        this.capPatterns = {
            // CAP service patterns
            serviceHandlers: [
                /\.on\s*\(\s*['"`]([^'"`]+)['"`]/g,
                /\.before\s*\(\s*['"`]([^'"`]+)['"`]/g,
                /\.after\s*\(\s*['"`]([^'"`]+)['"`]/g
            ],
            eventEmitters: [
                /\.emit\s*\(\s*['"`]([^'"`]+)['"`]/g,
                /srv\.emit\s*\(\s*['"`]([^'"`]+)['"`]/g
            ],
            cdsRequire: [
                /require\s*\(\s*['"`]@sap\/cds['"`]\)/g,
                /import.*from\s*['"`]@sap\/cds['"`]/g
            ]
        };
    }

    /**
     * Analyze CAP service implementation file
     */
    analyzeCAPService(content, filePath) {
        const result = {
            serviceHandlers: [],
            eventHandlers: [],
            middleware: [],
            cdsImports: [],
            entities: [],
            customLogic: [],
            metadata: {
                isCAPService: false,
                complexity: 0,
                hasCustomAuth: false,
                hasValidation: false,
                hasBusinessLogic: false
            }
        };

        try {
            // Parse JavaScript/TypeScript content
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

            // Traverse AST to identify CAP patterns
            try {
                traverse(ast, {
                    ImportDeclaration: (path) => {
                        this.analyzeImports(path, result);
                    },
                    VariableDeclarator: (path) => {
                        this.analyzeRequireStatements(path, result);
                    },
                    CallExpression: (path) => {
                        this.analyzeCallExpressions(path, result, content);
                    },
                    FunctionDeclaration: (path) => {
                        this.analyzeFunctionDeclarations(path, result);
                    },
                    ArrowFunctionExpression: {
                        enter(path) {
                            // Properly handle arrow functions with enter method
                            try {
                                if (path.node && path.node.params) {
                                    const firstParam = path.node.params[0];
                                    if (firstParam && firstParam.name && ['req', 'request', 'srv', 'service'].includes(firstParam.name)) {
                                        result.customLogic.push({
                                            type: 'handler_function',
                                            isAsync: path.node.async || false,
                                            parameters: path.node.params.length,
                                            line: path.node.loc ? path.node.loc.start.line : 0
                                        });
                                    }
                                }
                            } catch (error) {
                                // Skip on error
                            }
                        }
                    },
                    MemberExpression: (path) => {
                        this.analyzeMemberExpressions(path, result);
                    }
                });
            } catch (traverseError) {
                // Silently fall back to regex analysis without warning
                this.fallbackRegexAnalysis(content, result);
            }

            // Calculate complexity and detect patterns
            this.calculateServiceComplexity(result);
            this.detectCAPPatterns(result, content);

        } catch (error) {
            console.warn(`Warning: Could not parse ${filePath} as JavaScript:`, error.message);
            // Fallback to regex-based analysis
            this.fallbackRegexAnalysis(content, result);
        }

        return result;
    }

    analyzeImports(path, result) {
        const node = path.node;
        
        if (node.source && node.source.value) {
            const importSource = node.source.value;
            
            // Detect CDS imports
            if (importSource.includes('@sap/cds') || importSource.includes('cds')) {
                result.cdsImports.push({
                    source: importSource,
                    specifiers: node.specifiers.map(spec => ({
                        type: spec.type,
                        name: spec.local ? spec.local.name : 'default',
                        imported: spec.imported ? spec.imported.name : null
                    })),
                    line: node.loc ? node.loc.start.line : 0
                });
                result.metadata.isCAPService = true;
            }
        }
    }

    analyzeRequireStatements(path, result) {
        const node = path.node;
        
        if (node.init && node.init.type === 'CallExpression' && 
            node.init.callee && node.init.callee.name === 'require') {
            
            const argument = node.init.arguments[0];
            if (argument && argument.type === 'StringLiteral') {
                const requireSource = argument.value;
                
                if (requireSource.includes('@sap/cds') || requireSource.includes('cds')) {
                    result.cdsImports.push({
                        source: requireSource,
                        variable: node.id ? node.id.name : 'unknown',
                        line: node.loc ? node.loc.start.line : 0,
                        type: 'require'
                    });
                    result.metadata.isCAPService = true;
                }
            }
        }
    }

    analyzeCallExpressions(path, result, content) {
        const node = path.node;
        
        // Analyze service handler registrations
        if (node.callee && node.callee.type === 'MemberExpression') {
            const object = node.callee.object;
            const property = node.callee.property;
            
            if (property && property.name) {
                const methodName = property.name;
                
                // CAP event handlers (.on, .before, .after)
                if (['on', 'before', 'after'].includes(methodName)) {
                    this.analyzeEventHandler(node, result, methodName, content);
                }
                
                // Service connections (.to, .connect)
                if (['to', 'connect'].includes(methodName)) {
                    this.analyzeServiceConnection(node, result);
                }
                
                // Entity operations (.read, .create, .update, .delete)
                if (['read', 'create', 'update', 'delete'].includes(methodName)) {
                    this.analyzeEntityOperation(node, result, methodName);
                }
                
                // Emit events
                if (methodName === 'emit') {
                    this.analyzeEventEmission(node, result);
                }
            }
        }
        
        // Analyze function calls for CAP patterns
        if (node.callee && node.callee.name) {
            const functionName = node.callee.name;
            
            // CDS.connect(), CDS.serve(), etc.
            if (functionName.startsWith('cds') || functionName.toUpperCase() === 'CDS') {
                result.metadata.isCAPService = true;
            }
        }
    }

    analyzeEventHandler(node, result, handlerType, content) {
        const args = node.arguments;
        
        if (args.length >= 2) {
            const eventArg = args[0];
            const handlerArg = args[1];
            
            let eventName = 'unknown';
            let entityName = null;
            
            // Extract event name
            if (eventArg.type === 'StringLiteral') {
                const eventString = eventArg.value;
                const parts = eventString.split(' ');
                eventName = parts[0];
                entityName = parts.length > 1 ? parts[1] : null;
            }
            
            // Analyze handler function
            let handlerInfo = {
                type: handlerType,
                event: eventName,
                entity: entityName,
                line: node.loc ? node.loc.start.line : 0,
                isAsync: false,
                parameters: [],
                hasValidation: false,
                hasAuthorization: false,
                hasBusinessLogic: false
            };
            
            if (handlerArg.type === 'ArrowFunctionExpression' || 
                handlerArg.type === 'FunctionExpression') {
                handlerInfo.isAsync = handlerArg.async || false;
                handlerInfo.parameters = handlerArg.params.map(param => param.name || 'unknown');
                
                // Analyze handler body for patterns
                this.analyzeHandlerBody(handlerArg, handlerInfo, content);
            }
            
            result.serviceHandlers.push(handlerInfo);
        }
    }

    analyzeHandlerBody(handlerNode, handlerInfo, content) {
        // Look for common CAP patterns in handler body
        traverse(handlerNode, {
            CallExpression: (path) => {
                const node = path.node;
                
                if (node.callee && node.callee.type === 'MemberExpression' && 
                    node.callee.property && node.callee.property.name) {
                    
                    const methodName = node.callee.property.name;
                    
                    // Validation patterns
                    if (['assert', 'validate', 'check'].includes(methodName)) {
                        handlerInfo.hasValidation = true;
                        result.metadata.hasValidation = true;
                    }
                    
                    // Authorization patterns
                    if (['authorize', 'checkAuth', 'hasRole', 'isAuthenticated'].includes(methodName)) {
                        handlerInfo.hasAuthorization = true;
                        result.metadata.hasCustomAuth = true;
                    }
                    
                    // Business logic patterns
                    if (['calculate', 'process', 'transform', 'compute'].includes(methodName)) {
                        handlerInfo.hasBusinessLogic = true;
                        result.metadata.hasBusinessLogic = true;
                    }
                }
            },
            ThrowStatement: (path) => {
                // Error handling indicates validation or business logic
                handlerInfo.hasValidation = true;
            },
            IfStatement: (path) => {
                // Conditional logic indicates business rules
                handlerInfo.hasBusinessLogic = true;
            }
        });
    }

    analyzeServiceConnection(node, result) {
        const args = node.arguments;
        
        if (args.length > 0 && args[0].type === 'StringLiteral') {
            const serviceName = args[0].value;
            
            result.entities.push({
                name: serviceName,
                type: 'external_service',
                line: node.loc ? node.loc.start.line : 0
            });
        }
    }

    analyzeEntityOperation(node, result, operation) {
        // Track entity CRUD operations
        result.customLogic.push({
            type: 'entity_operation',
            operation: operation,
            line: node.loc ? node.loc.start.line : 0
        });
    }

    analyzeEventEmission(node, result) {
        const args = node.arguments;
        
        if (args.length > 0 && args[0].type === 'StringLiteral') {
            const eventName = args[0].value;
            
            result.eventHandlers.push({
                type: 'emit',
                event: eventName,
                line: node.loc ? node.loc.start.line : 0
            });
        }
    }

    analyzeFunctionDeclarations(path, result) {
        const node = path.node;
        
        if (node.id && node.id.name) {
            const functionName = node.id.name;
            
            // Check if it's a CAP service handler
            if (functionName.includes('handler') || functionName.includes('Handler')) {
                result.customLogic.push({
                    type: 'custom_handler',
                    name: functionName,
                    isAsync: node.async || false,
                    parameters: node.params.length,
                    line: node.loc ? node.loc.start.line : 0
                });
            }
        }
    }

    analyzeArrowFunctions(path, result) {
        // Analyze arrow function patterns that might be event handlers
        const node = path.node;
        
        if (node.params && node.params.length > 0) {
            const firstParam = node.params[0];
            
            // Common CAP handler parameter names
            if (firstParam.name && ['req', 'request', 'srv', 'service'].includes(firstParam.name)) {
                result.customLogic.push({
                    type: 'handler_function',
                    isAsync: node.async || false,
                    parameters: node.params.length,
                    line: node.loc ? node.loc.start.line : 0
                });
            }
        }
    }

    analyzeMemberExpressions(path, result) {
        const node = path.node;
        
        // Look for CAP service API usage
        if (node.object && node.property) {
            const objectName = node.object.name;
            const propertyName = node.property.name;
            
            // CDS API patterns
            if (objectName === 'cds' || objectName === 'CDS') {
                result.customLogic.push({
                    type: 'cds_api_usage',
                    api: propertyName,
                    line: node.loc ? node.loc.start.line : 0
                });
            }
        }
    }

    calculateServiceComplexity(result) {
        let complexity = 0;
        
        complexity += result.serviceHandlers.length * 2;
        complexity += result.eventHandlers.length;
        complexity += result.customLogic.length;
        complexity += result.cdsImports.length;
        
        // Add complexity for advanced patterns
        result.serviceHandlers.forEach(handler => {
            if (handler.hasValidation) complexity += 2;
            if (handler.hasAuthorization) complexity += 3;
            if (handler.hasBusinessLogic) complexity += 2;
            if (handler.isAsync) complexity += 1;
        });
        
        result.metadata.complexity = complexity;
    }

    detectCAPPatterns(result, content) {
        // Additional pattern detection
        
        // Check for validation patterns
        if (content.includes('req.error') || content.includes('request.error') || 
            content.includes('throw new Error') || content.includes('validate') ||
            content.includes('.assert') || content.includes('validation')) {
            result.metadata.hasValidation = true;
        }
        
        // Check for authorization patterns
        if (content.includes('req.user') || content.includes('hasRole') || 
            content.includes('isAuthenticated') || content.includes('checkRole') ||
            content.includes('authorization') || content.includes('Unauthorized') ||
            content.includes('auth') || content.includes('permit')) {
            result.metadata.hasCustomAuth = true;
        }
        
        // Check for business logic patterns
        if (content.includes('business') || content.includes('calculate') || 
            content.includes('process') || content.includes('workflow') ||
            content.includes('logic') || content.includes('custom')) {
            result.metadata.hasBusinessLogic = true;
        }
        
        // Check for OData annotations
        if (content.includes('@odata') || content.includes('OData')) {
            result.metadata.hasODataFeatures = true;
        }
        
        // Check for SAP specific imports
        if (content.includes('@sap/') || content.includes('sap.')) {
            result.metadata.usesSAPLibraries = true;
        }
        
        // Check for database operations
        if (content.includes('SELECT') || content.includes('INSERT') || 
            content.includes('UPDATE') || content.includes('DELETE')) {
            result.metadata.hasDirectDatabaseAccess = true;
        }
        
        // Check for transaction handling
        if (content.includes('transaction') || content.includes('commit') || 
            content.includes('rollback')) {
            result.metadata.hasTransactionHandling = true;
        }
    }

    fallbackRegexAnalysis(content, result) {
        // Fallback analysis using regex patterns
        
        // Service handlers
        this.capPatterns.serviceHandlers.forEach(pattern => {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                result.serviceHandlers.push({
                    type: 'regex_detected',
                    event: match[1],
                    line: this.getLineNumber(content, match.index)
                });
            }
        });
        
        // Event emitters
        this.capPatterns.eventEmitters.forEach(pattern => {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                result.eventHandlers.push({
                    type: 'emit',
                    event: match[1],
                    line: this.getLineNumber(content, match.index)
                });
            }
        });
        
        // CDS require statements
        this.capPatterns.cdsRequire.forEach(pattern => {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                result.cdsImports.push({
                    type: 'regex_detected',
                    line: this.getLineNumber(content, match.index)
                });
                result.metadata.isCAPService = true;
            }
        });
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length - 1;
    }
}

module.exports = CAPServiceAnalyzer;