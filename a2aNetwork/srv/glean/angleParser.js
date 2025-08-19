/**
 * @fileoverview Angle Query Language Parser for Glean
 * @module angleParser
 * @since 1.0.0
 * 
 * Implements a subset of Glean's Angle query language for code analysis
 * Parses Angle syntax and converts to executable queries against Glean facts
 */

const crypto = require('crypto');

class AngleParser {
    constructor() {
        this.tokens = [];
        this.currentToken = 0;
        this.predicateDefinitions = new Map();
        this.typeDefinitions = new Map();
        this.derivedPredicates = new Map();
    }

    /**
     * Parse Angle schema definition
     * @param {string} schemaText - Angle schema source
     * @returns {Object} Parsed schema with predicates and types
     */
    parseSchema(schemaText) {
        this.tokens = this.tokenize(schemaText);
        this.currentToken = 0;
        
        const schema = {
            name: null,
            version: null,
            predicates: {},
            types: {},
            derived: {}
        };

        while (!this.isAtEnd()) {
            const statement = this.parseStatement();
            if (statement) {
                switch (statement.type) {
                    case 'schema':
                        schema.name = statement.name;
                        schema.version = statement.version;
                        
                        // Process nested statements from schema
                        if (statement.statements) {
                            statement.statements.forEach(nestedStatement => {
                                switch (nestedStatement.type) {
                                    case 'predicate':
                                        schema.predicates[nestedStatement.name] = nestedStatement;
                                        this.predicateDefinitions.set(nestedStatement.name, nestedStatement);
                                        break;
                                    case 'type':
                                        schema.types[nestedStatement.name] = nestedStatement;
                                        this.typeDefinitions.set(nestedStatement.name, nestedStatement);
                                        break;
                                    case 'derived':
                                        schema.derived[nestedStatement.name] = nestedStatement;
                                        this.derivedPredicates.set(nestedStatement.name, nestedStatement);
                                        break;
                                }
                            });
                        }
                        break;
                    case 'predicate':
                        schema.predicates[statement.name] = statement;
                        this.predicateDefinitions.set(statement.name, statement);
                        break;
                    case 'type':
                        schema.types[statement.name] = statement;
                        this.typeDefinitions.set(statement.name, statement);
                        break;
                    case 'derived':
                        schema.derived[statement.name] = statement;
                        this.derivedPredicates.set(statement.name, statement);
                        break;
                }
            }
        }

        return schema;
    }

    /**
     * Parse Angle query
     * @param {string} queryText - Angle query source
     * @returns {Object} Parsed query AST
     */
    parseQuery(queryText) {
        this.tokens = this.tokenize(queryText);
        this.currentToken = 0;

        return this.parseQueryStatement();
    }

    /**
     * Tokenize Angle source code
     * @param {string} source - Source code
     * @returns {Array} Array of tokens
     */
    tokenize(source) {
        const tokens = [];
        const keywords = new Set([
            'schema', 'predicate', 'type', 'enum', 'query', 'where', 'if', 'else',
            'maybe', 'nat', 'string', 'bool', 'true', 'false', 'and', 'or', 'not'
        ]);

        let i = 0;
        while (i < source.length) {
            const char = source[i];

            // Skip whitespace
            if (/\s/.test(char)) {
                i++;
                continue;
            }

            // Skip comments
            if (char === '#') {
                while (i < source.length && source[i] !== '\n') {
                    i++;
                }
                continue;
            }

            // Multi-character operators
            if (i + 1 < source.length) {
                const twoChar = source.slice(i, i + 2);
                if (['==', '!=', '<=', '>=', '&&', '||', '->', '=>'].includes(twoChar)) {
                    tokens.push({ type: 'OPERATOR', value: twoChar });
                    i += 2;
                    continue;
                }
            }

            // Single character tokens
            if ('{}[](),:;=<>!&|+-*/.'.includes(char)) {
                const tokenType = {
                    '{': 'LBRACE', '}': 'RBRACE',
                    '[': 'LBRACKET', ']': 'RBRACKET',
                    '(': 'LPAREN', ')': 'RPAREN',
                    ',': 'COMMA', ':': 'COLON', ';': 'SEMICOLON',
                    '=': 'EQUALS', '<': 'LT', '>': 'GT',
                    '!': 'NOT', '&': 'AND', '|': 'OR',
                    '+': 'PLUS', '-': 'MINUS', '*': 'STAR',
                    '/': 'SLASH', '.': 'DOT'
                }[char];
                
                tokens.push({ type: tokenType, value: char });
                i++;
                continue;
            }

            // String literals
            if (char === '"' || char === "'") {
                const quote = char;
                let value = '';
                i++; // Skip opening quote
                
                while (i < source.length && source[i] !== quote) {
                    if (source[i] === '\\' && i + 1 < source.length) {
                        i++; // Skip escape character
                        const escaped = {
                            'n': '\n', 't': '\t', 'r': '\r',
                            '\\': '\\', '"': '"', "'": "'"
                        }[source[i]] || source[i];
                        value += escaped;
                    } else {
                        value += source[i];
                    }
                    i++;
                }
                
                if (i < source.length) i++; // Skip closing quote
                tokens.push({ type: 'STRING', value });
                continue;
            }

            // Numbers
            if (/\d/.test(char)) {
                let value = '';
                while (i < source.length && /[\d.]/.test(source[i])) {
                    value += source[i];
                    i++;
                }
                tokens.push({ 
                    type: value.includes('.') ? 'FLOAT' : 'INTEGER', 
                    value: value.includes('.') ? parseFloat(value) : parseInt(value)
                });
                continue;
            }

            // Identifiers and keywords
            if (/[a-zA-Z_]/.test(char)) {
                let value = '';
                while (i < source.length && /[a-zA-Z0-9_]/.test(source[i])) {
                    value += source[i];
                    i++;
                }
                
                const type = keywords.has(value) ? 'KEYWORD' : 'IDENTIFIER';
                tokens.push({ type, value });
                continue;
            }

            // Unknown character
            tokens.push({ type: 'UNKNOWN', value: char });
            i++;
        }

        tokens.push({ type: 'EOF', value: null });
        return tokens;
    }

    /**
     * Parse a single statement
     * @returns {Object|null} Parsed statement or null
     */
    parseStatement() {
        if (this.isAtEnd()) return null;

        const token = this.peek();
        
        if (token.type === 'KEYWORD') {
            switch (token.value) {
                case 'schema':
                    return this.parseSchemaDeclaration();
                case 'predicate':
                    return this.parsePredicateDeclaration();
                case 'type':
                    return this.parseTypeDeclaration();
                case 'query':
                    return this.parseQueryStatement();
                default:
                    // Check for derived predicate (function-style)
                    if (this.peekAhead(1)?.value === '(') {
                        return this.parseDerivedPredicate();
                    }
            }
        }

        if (token.type === 'IDENTIFIER') {
            // Derived predicate definition
            return this.parseDerivedPredicate();
        }

        // Skip unknown statements
        this.advance();
        return null;
    }

    /**
     * Parse schema declaration
     */
    parseSchemaDeclaration() {
        this.consume('KEYWORD', 'schema'); // 'schema'
        const name = this.consume('IDENTIFIER').value;
        
        // Handle schema.version format
        let version = 1;
        if (this.check('DOT')) {
            this.advance(); // consume '.'
            version = this.consume('INTEGER').value;
        }
        
        this.consume('LBRACE'); // '{'
        
        // Parse schema body statements
        const statements = [];
        while (!this.check('RBRACE') && !this.isAtEnd()) {
            const statement = this.parseStatement();
            if (statement) {
                statements.push(statement);
            }
        }
        
        this.consume('RBRACE'); // '}'
        
        return {
            type: 'schema',
            name,
            version,
            statements
        };
    }

    /**
     * Parse predicate declaration
     */
    parsePredicateDeclaration() {
        this.consume('KEYWORD', 'predicate'); // 'predicate'
        const name = this.consume('IDENTIFIER').value;
        this.consume('COLON'); // ':'
        
        const fields = this.parseFieldList();
        
        return {
            type: 'predicate',
            name,
            fields
        };
    }

    /**
     * Parse type declaration
     */
    parseTypeDeclaration() {
        this.consume('KEYWORD', 'type'); // 'type'
        const name = this.consume('IDENTIFIER').value;
        this.consume('COLON'); // ':'
        
        const definition = this.parseTypeDefinition();
        
        return {
            type: 'type',
            name,
            definition
        };
    }

    /**
     * Parse field list for predicates
     */
    parseFieldList() {
        this.consume('LBRACE'); // '{'
        const fields = [];
        
        while (!this.check('RBRACE') && !this.isAtEnd()) {
            const field = this.parseField();
            if (field) {
                fields.push(field);
            }
            
            if (this.check('COMMA')) {
                this.advance();
            }
        }
        
        this.consume('RBRACE'); // '}'
        return fields;
    }

    /**
     * Parse a single field
     */
    parseField() {
        const name = this.consume('IDENTIFIER').value;
        this.consume('COLON'); // ':'
        const fieldType = this.parseTypeExpression();
        
        return {
            name,
            type: fieldType
        };
    }

    /**
     * Parse type expression
     */
    parseTypeExpression() {
        if (this.check('KEYWORD', 'maybe')) {
            this.advance();
            const innerType = this.parseTypeExpression();
            return { type: 'maybe', inner: innerType };
        }
        
        if (this.check('LBRACKET')) {
            this.advance(); // '['
            const elementType = this.parseTypeExpression();
            this.consume('RBRACKET'); // ']'
            return { type: 'array', element: elementType };
        }
        
        if (this.check('KEYWORD', 'enum')) {
            this.advance(); // 'enum'
            this.consume('LBRACE'); // '{'
            const values = [];
            
            while (!this.check('RBRACE') && !this.isAtEnd()) {
                values.push(this.consume('IDENTIFIER').value);
                if (this.check('OR')) {
                    this.advance();
                }
            }
            
            this.consume('RBRACE'); // '}'
            return { type: 'enum', values };
        }
        
        // Primitive types
        if (this.check('KEYWORD')) {
            const primitiveType = this.advance().value;
            return { type: 'primitive', name: primitiveType };
        }
        
        // Custom type reference (could be keyword or identifier)
        let typeName;
        if (this.check('KEYWORD')) {
            typeName = this.advance().value;
        } else {
            typeName = this.consume('IDENTIFIER').value;
        }
        return { type: 'reference', name: typeName };
    }

    /**
     * Parse type definition
     */
    parseTypeDefinition() {
        if (this.check('LBRACE')) {
            // Struct type
            return this.parseFieldList();
        }
        
        return this.parseTypeExpression();
    }

    /**
     * Parse derived predicate
     */
    parseDerivedPredicate() {
        const name = this.consume('IDENTIFIER').value;
        
        // Parameters
        this.consume('LPAREN'); // '('
        const parameters = [];
        
        while (!this.check('RPAREN') && !this.isAtEnd()) {
            const param = this.parseParameter();
            parameters.push(param);
            
            if (this.check('COMMA')) {
                this.advance();
            }
        }
        
        this.consume('RPAREN'); // ')'
        this.consume('COLON'); // ':'
        
        const returnType = this.parseTypeExpression();
        this.consume('EQUALS'); // '='
        
        const body = this.parseExpression();
        
        return {
            type: 'derived',
            name,
            parameters,
            returnType,
            body
        };
    }

    /**
     * Parse parameter
     */
    parseParameter() {
        const name = this.consume('IDENTIFIER').value;
        this.consume('COLON'); // ':'
        const paramType = this.parseTypeExpression();
        
        return { name, type: paramType };
    }

    /**
     * Parse query statement
     */
    parseQueryStatement() {
        this.consume('KEYWORD', 'query'); // 'query'
        const name = this.consume('IDENTIFIER').value;
        
        // Parameters
        this.consume('LPAREN'); // '('
        const parameters = [];
        
        while (!this.check('RPAREN') && !this.isAtEnd()) {
            const param = this.parseParameter();
            parameters.push(param);
            
            if (this.check('COMMA')) {
                this.advance();
            }
        }
        
        this.consume('RPAREN'); // ')'
        this.consume('COLON'); // ':'
        
        const returnType = this.parseTypeExpression();
        this.consume('EQUALS'); // '='
        
        const body = this.parseExpression();
        
        return {
            type: 'query',
            name,
            parameters,
            returnType,
            body
        };
    }

    /**
     * Parse expression
     */
    parseExpression() {
        return this.parseOrExpression();
    }

    /**
     * Parse OR expression
     */
    parseOrExpression() {
        let left = this.parseAndExpression();
        
        while (this.check('OR') || this.check('OPERATOR', '||')) {
            const operator = this.advance().value;
            const right = this.parseAndExpression();
            left = {
                type: 'binary',
                operator: 'or',
                left,
                right
            };
        }
        
        return left;
    }

    /**
     * Parse AND expression
     */
    parseAndExpression() {
        let left = this.parseComparisonExpression();
        
        while (this.check('AND') || this.check('OPERATOR', '&&')) {
            const operator = this.advance().value;
            const right = this.parseComparisonExpression();
            left = {
                type: 'binary',
                operator: 'and',
                left,
                right
            };
        }
        
        return left;
    }

    /**
     * Parse comparison expression
     */
    parseComparisonExpression() {
        let left = this.parsePrimaryExpression();
        
        while (this.checkAny(['EQUALS', 'LT', 'GT', 'OPERATOR'])) {
            const operator = this.advance().value;
            const right = this.parsePrimaryExpression();
            left = {
                type: 'comparison',
                operator,
                left,
                right
            };
        }
        
        return left;
    }

    /**
     * Parse primary expression
     */
    parsePrimaryExpression() {
        if (this.check('IDENTIFIER')) {
            let name = this.advance().value;
            
            // Handle dotted identifiers (e.g., src.File)
            while (this.check('DOT')) {
                this.advance(); // consume '.'
                const nextPart = this.consume('IDENTIFIER').value;
                name += '.' + nextPart;
            }
            
            // Function call or predicate reference
            if (this.check('LBRACE')) {
                return this.parsePredicateReference(name);
            }
            
            // Simple identifier
            return {
                type: 'identifier',
                name
            };
        }
        
        if (this.check('STRING')) {
            return {
                type: 'literal',
                value: this.advance().value,
                dataType: 'string'
            };
        }
        
        if (this.check('INTEGER')) {
            return {
                type: 'literal',
                value: this.advance().value,
                dataType: 'integer'
            };
        }
        
        if (this.check('FLOAT')) {
            return {
                type: 'literal',
                value: this.advance().value,
                dataType: 'float'
            };
        }
        
        if (this.check('KEYWORD', 'true') || this.check('KEYWORD', 'false')) {
            return {
                type: 'literal',
                value: this.advance().value === 'true',
                dataType: 'boolean'
            };
        }
        
        if (this.check('LPAREN')) {
            this.advance(); // '('
            const expr = this.parseExpression();
            this.consume('RPAREN'); // ')'
            return expr;
        }
        
        throw new Error(`Unexpected token: ${JSON.stringify(this.peek())}`);
    }

    /**
     * Parse predicate reference
     */
    parsePredicateReference(predicateName) {
        this.consume('LBRACE'); // '{'
        const bindings = [];
        
        while (!this.check('RBRACE') && !this.isAtEnd()) {
            const binding = this.parseBinding();
            bindings.push(binding);
            
            if (this.check('COMMA')) {
                this.advance();
            }
        }
        
        this.consume('RBRACE'); // '}'
        
        // Check for where clause
        let whereClause = null;
        if (this.check('KEYWORD', 'where')) {
            this.advance(); // 'where'
            whereClause = this.parseExpression();
        }
        
        return {
            type: 'predicate_ref',
            predicate: predicateName,
            bindings,
            where: whereClause
        };
    }

    /**
     * Parse binding
     */
    parseBinding() {
        let field = this.consume('IDENTIFIER').value;
        
        // Handle dotted field names (e.g., value.language)
        while (this.check('DOT')) {
            this.advance(); // consume '.'
            const nextPart = this.consume('IDENTIFIER').value;
            field += '.' + nextPart;
        }
        
        if (this.check('EQUALS')) {
            this.advance(); // '='
            const value = this.parseExpression();
            return { type: 'assignment', field, value };
        }
        
        return { type: 'extraction', field };
    }

    // Utility methods
    peek() {
        return this.tokens[this.currentToken] || { type: 'EOF', value: null };
    }

    peekAhead(offset) {
        return this.tokens[this.currentToken + offset] || { type: 'EOF', value: null };
    }

    advance() {
        if (!this.isAtEnd()) {
            this.currentToken++;
        }
        return this.tokens[this.currentToken - 1];
    }

    check(type, value = null) {
        if (this.isAtEnd()) return false;
        const token = this.peek();
        return token.type === type && (value === null || token.value === value);
    }

    checkAny(types) {
        return types.some(type => this.check(type));
    }

    consume(type, value = null) {
        if (this.check(type, value)) {
            return this.advance();
        }
        
        const expected = value ? `${type}('${value}')` : type;
        throw new Error(`Expected ${expected}, got ${JSON.stringify(this.peek())}`);
    }

    isAtEnd() {
        return this.peek().type === 'EOF';
    }

    /**
     * Execute a parsed query against fact database
     * @param {Object} query - Parsed query AST
     * @param {Object} factDb - Fact database
     * @returns {Array} Query results
     */
    executeQuery(query, factDb) {
        const executor = new AngleQueryExecutor(factDb, this.predicateDefinitions);
        return executor.execute(query);
    }
}

/**
 * Angle Query Executor
 * Executes parsed Angle queries against a fact database
 */
class AngleQueryExecutor {
    constructor(factDb, predicateDefinitions) {
        this.factDb = factDb;
        this.predicateDefinitions = predicateDefinitions;
        this.bindings = new Map();
    }

    execute(query) {
        if (query.type === 'query') {
            return this.executeQueryBody(query.body, query.parameters);
        }
        
        throw new Error(`Cannot execute query of type: ${query.type}`);
    }

    executeQueryBody(body, parameters = []) {
        // Set up parameter bindings
        const paramBindings = new Map();
        parameters.forEach(param => {
            paramBindings.set(param.name, null); // Will be bound during execution
        });

        return this.evaluateExpression(body, paramBindings);
    }

    evaluateExpression(expr, bindings) {
        switch (expr.type) {
            case 'predicate_ref':
                return this.evaluatePredicateReference(expr, bindings);
            
            case 'binary':
                return this.evaluateBinaryExpression(expr, bindings);
            
            case 'comparison':
                return this.evaluateComparison(expr, bindings);
            
            case 'identifier':
                return bindings.get(expr.name) || expr.name;
            
            case 'literal':
                return expr.value;
            
            default:
                throw new Error(`Unknown expression type: ${expr.type}`);
        }
    }

    evaluatePredicateReference(expr, bindings) {
        const predicateName = expr.predicate;
        const facts = this.factDb[predicateName] || [];
        
        const results = [];
        
        for (const fact of facts) {
            const factBindings = new Map(bindings);
            let matches = true;
            
            // Try to match bindings
            for (const binding of expr.bindings) {
                if (binding.type === 'assignment') {
                    const expectedValue = this.evaluateExpression(binding.value, factBindings);
                    const actualValue = this.getFactValue(fact, binding.field);
                    
                    if (actualValue !== expectedValue) {
                        matches = false;
                        break;
                    }
                } else if (binding.type === 'extraction') {
                    const value = this.getFactValue(fact, binding.field);
                    factBindings.set(binding.field, value);
                }
            }
            
            // Apply where clause if present
            if (matches && expr.where) {
                matches = this.evaluateExpression(expr.where, factBindings);
            }
            
            if (matches) {
                results.push(this.createResult(fact, factBindings));
            }
        }
        
        return results;
    }

    evaluateBinaryExpression(expr, bindings) {
        const left = this.evaluateExpression(expr.left, bindings);
        const right = this.evaluateExpression(expr.right, bindings);
        
        switch (expr.operator) {
            case 'and':
                return left && right;
            case 'or':
                return left || right;
            default:
                throw new Error(`Unknown binary operator: ${expr.operator}`);
        }
    }

    evaluateComparison(expr, bindings) {
        const left = this.evaluateExpression(expr.left, bindings);
        const right = this.evaluateExpression(expr.right, bindings);
        
        switch (expr.operator) {
            case '=':
            case '==':
                return left === right;
            case '!=':
                return left !== right;
            case '<':
                return left < right;
            case '>':
                return left > right;
            case '<=':
                return left <= right;
            case '>=':
                return left >= right;
            default:
                throw new Error(`Unknown comparison operator: ${expr.operator}`);
        }
    }

    getFactValue(fact, field) {
        // Navigate nested field paths (e.g., "value.name")
        const path = field.split('.');
        let current = fact;
        
        for (const segment of path) {
            if (current && typeof current === 'object') {
                current = current[segment];
            } else {
                return undefined;
            }
        }
        
        return current;
    }

    createResult(fact, bindings) {
        const result = { ...fact };
        
        // Add bound variables to result
        for (const [key, value] of bindings) {
            if (!result.hasOwnProperty(key)) {
                result[key] = value;
            }
        }
        
        return result;
    }
}

module.exports = { AngleParser, AngleQueryExecutor };