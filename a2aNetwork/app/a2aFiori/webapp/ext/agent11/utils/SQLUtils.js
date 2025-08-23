sap.ui.define([
    "sap/m/MessageToast",
    "sap/base/Log",
    "a2a/network/agent11/ext/utils/SecurityUtils"
], function (MessageToast, Log, SecurityUtils) {
    "use strict";

    return {
        /**
         * Validates SQL query syntax and security
         * @param {string} sql - The SQL query to validate
         * @param {string} dialect - The SQL dialect (HANA, POSTGRESQL, etc.)
         * @returns {object} Validation result
         */
        validateSQL: function (sql, dialect) {
            // Use SecurityUtils for comprehensive validation
            const securityValidation = SecurityUtils.validateSQL(sql);
            
            if (!securityValidation.isValid) {
                return {
                    isValid: false,
                    errors: securityValidation.errors,
                    warnings: [],
                    suggestions: [],
                    sanitized: securityValidation.sanitized
                };
            }

            const errors = [];
            const warnings = [];
            const suggestions = [];
            const sanitizedSQL = securityValidation.sanitized;

            // Basic syntax checks
            const syntaxIssues = this._checkBasicSyntax(sanitizedSQL, dialect);
            if (syntaxIssues.errors.length > 0) {
                errors.push(...syntaxIssues.errors);
            }
            if (syntaxIssues.warnings.length > 0) {
                warnings.push(...syntaxIssues.warnings);
            }

            // Performance suggestions
            const performanceSuggestions = this._getPerformanceSuggestions(sanitizedSQL);
            suggestions.push(...performanceSuggestions);

            // Query complexity validation
            const complexityValidation = SecurityUtils.validateQueryComplexity(sanitizedSQL);
            if (!complexityValidation.isValid) {
                warnings.push(...complexityValidation.issues);
            }

            return {
                isValid: errors.length === 0,
                errors: errors,
                warnings: warnings,
                suggestions: suggestions,
                sanitized: sanitizedSQL
            };
        },

        /**
         * Sanitizes SQL input to prevent injection attacks
         * @param {string} sql - The SQL to sanitize
         * @returns {string} Sanitized SQL
         */
        sanitizeSQL: function (sql) {
            if (!sql) return "";
            
            // Use SecurityUtils for comprehensive sanitization
            const validation = SecurityUtils.validateSQL(sql);
            return validation.sanitized || "";
        },

        /**
         * Security vulnerability checks
         * @private
         */
        _checkSecurityVulnerabilities: function (sql) {
            const issues = [];
            const lowerSQL = sql.toLowerCase();

            // Check for SQL injection patterns
            const injectionPatterns = [
                /union\s+select/i,
                /drop\s+table/i,
                /delete\s+from.*where\s+1\s*=\s*1/i,
                /exec\s*\(/i,
                /xp_cmdshell/i,
                /sp_executesql/i
            ];

            injectionPatterns.forEach(pattern => {
                if (pattern.test(sql)) {
                    issues.push("Potential SQL injection pattern detected");
                }
            });

            // Check for dangerous functions
            const dangerousFunctions = ['exec', 'execute', 'eval', 'system'];
            dangerousFunctions.forEach(func => {
                if (lowerSQL.includes(func + '(')) {
                    issues.push(`Dangerous function '${func}' detected`);
                }
            });

            return issues;
        },

        /**
         * Basic SQL syntax validation
         * @private
         */
        _checkBasicSyntax: function (sql, dialect) {
            const errors = [];
            const warnings = [];
            const lowerSQL = sql.toLowerCase();

            // Check for balanced parentheses
            const openParens = (sql.match(/\(/g) || []).length;
            const closeParens = (sql.match(/\)/g) || []).length;
            if (openParens !== closeParens) {
                errors.push("Unbalanced parentheses in SQL query");
            }

            // Check for SELECT without FROM (except for certain cases)
            if (lowerSQL.includes('select') && !lowerSQL.includes('from') && 
                !lowerSQL.includes('dual') && dialect !== 'HANA') {
                warnings.push("SELECT statement without FROM clause");
            }

            // Check for missing WHERE clause in UPDATE/DELETE
            if ((lowerSQL.includes('update') || lowerSQL.includes('delete')) && 
                !lowerSQL.includes('where')) {
                warnings.push("UPDATE/DELETE without WHERE clause - this affects all rows");
            }

            return { errors, warnings };
        },

        /**
         * Get performance optimization suggestions
         * @private
         */
        _getPerformanceSuggestions: function (sql) {
            const suggestions = [];
            const lowerSQL = sql.toLowerCase();

            // Check for SELECT *
            if (lowerSQL.includes('select *')) {
                suggestions.push({
                    type: "Performance",
                    message: "Consider selecting specific columns instead of using SELECT *",
                    impact: "Medium"
                });
            }

            // Check for LIKE with leading wildcard
            if (lowerSQL.includes('like \'%')) {
                suggestions.push({
                    type: "Performance",
                    message: "Leading wildcards in LIKE clauses prevent index usage",
                    impact: "High"
                });
            }

            // Check for functions in WHERE clause
            if (/where\s+\w+\s*\(/i.test(sql)) {
                suggestions.push({
                    type: "Performance",
                    message: "Functions in WHERE clause may prevent index usage",
                    impact: "Medium"
                });
            }

            return suggestions;
        },

        /**
         * Format SQL query for better readability
         * @param {string} sql - The SQL to format
         * @returns {string} Formatted SQL
         */
        formatSQL: function (sql) {
            if (!sql) return "";

            // Basic SQL formatting
            let formatted = sql
                .replace(/\bSELECT\b/gi, '\nSELECT')
                .replace(/\bFROM\b/gi, '\nFROM')
                .replace(/\bWHERE\b/gi, '\nWHERE')
                .replace(/\bAND\b/gi, '\n  AND')
                .replace(/\bOR\b/gi, '\n  OR')
                .replace(/\bORDER BY\b/gi, '\nORDER BY')
                .replace(/\bGROUP BY\b/gi, '\nGROUP BY')
                .replace(/\bHAVING\b/gi, '\nHAVING')
                .replace(/\bJOIN\b/gi, '\nJOIN')
                .replace(/\bINNER JOIN\b/gi, '\nINNER JOIN')
                .replace(/\bLEFT JOIN\b/gi, '\nLEFT JOIN')
                .replace(/\bRIGHT JOIN\b/gi, '\nRIGHT JOIN')
                .replace(/\bFULL JOIN\b/gi, '\nFULL JOIN')
                .replace(/\bUNION\b/gi, '\nUNION')
                .replace(/\bINSERT INTO\b/gi, '\nINSERT INTO')
                .replace(/\bVALUES\b/gi, '\nVALUES')
                .replace(/\bUPDATE\b/gi, '\nUPDATE')
                .replace(/\bSET\b/gi, '\nSET')
                .replace(/\bDELETE FROM\b/gi, '\nDELETE FROM');

            // Clean up extra whitespace and newlines
            formatted = formatted
                .replace(/\n\s*\n/g, '\n')
                .replace(/^\n+/, '')
                .trim();

            return formatted;
        },

        /**
         * Extract table names from SQL query
         * @param {string} sql - The SQL query
         * @returns {array} Array of table names
         */
        extractTableNames: function (sql) {
            if (!sql) return [];

            const tables = [];
            const lowerSQL = sql.toLowerCase();

            // Extract FROM clause tables
            const fromMatches = sql.match(/from\s+(\w+)/gi);
            if (fromMatches) {
                fromMatches.forEach(match => {
                    const tableName = match.replace(/from\s+/gi, '').trim();
                    if (!tables.includes(tableName)) {
                        tables.push(tableName);
                    }
                });
            }

            // Extract JOIN clause tables
            const joinMatches = sql.match(/join\s+(\w+)/gi);
            if (joinMatches) {
                joinMatches.forEach(match => {
                    const tableName = match.replace(/.*join\s+/gi, '').trim();
                    if (!tables.includes(tableName)) {
                        tables.push(tableName);
                    }
                });
            }

            return tables;
        },

        /**
         * Parse natural language query to extract intent and entities
         * @param {string} naturalLanguage - The natural language query
         * @returns {object} Parsed intent and entities
         */
        parseNaturalLanguage: function (naturalLanguage) {
            if (!naturalLanguage) return { intent: null, entities: [], confidence: 0 };

            // Sanitize the input to prevent injection
            const sanitizedQuery = SecurityUtils.escapeHTML(naturalLanguage.trim());
            const lowerQuery = sanitizedQuery.toLowerCase();
            const entities = [];
            let intent = null;
            let confidence = 0;

            // Simple intent recognition
            if (lowerQuery.includes('show') || lowerQuery.includes('list') || lowerQuery.includes('get')) {
                intent = 'SELECT';
                confidence = 70;
            } else if (lowerQuery.includes('add') || lowerQuery.includes('insert') || lowerQuery.includes('create')) {
                intent = 'INSERT';
                confidence = 70;
            } else if (lowerQuery.includes('update') || lowerQuery.includes('change') || lowerQuery.includes('modify')) {
                intent = 'UPDATE';
                confidence = 70;
            } else if (lowerQuery.includes('delete') || lowerQuery.includes('remove')) {
                intent = 'DELETE';
                confidence = 70;
            } else if (lowerQuery.includes('count') || lowerQuery.includes('how many')) {
                intent = 'COUNT';
                confidence = 80;
            }

            // Simple entity extraction with sanitization
            const tableKeywords = ['users', 'orders', 'products', 'customers', 'employees', 'sales'];
            tableKeywords.forEach(keyword => {
                if (lowerQuery.includes(keyword)) {
                    entities.push({
                        type: 'TABLE',
                        value: SecurityUtils.sanitizeSQLParameter(keyword),
                        confidence: 60
                    });
                }
            });

            return {
                intent: intent,
                entities: entities,
                confidence: confidence,
                sanitizedInput: sanitizedQuery
            };
        },

        /**
         * Generate example questions for different categories
         * @returns {array} Array of example questions
         */
        getExampleQuestions: function () {
            return [
                {
                    text: "Show me all customers from Germany",
                    description: "Simple SELECT with WHERE clause",
                    category: "Basic Query"
                },
                {
                    text: "How many orders were placed last month?",
                    description: "COUNT with date filtering",
                    category: "Aggregation"
                },
                {
                    text: "List the top 10 products by sales revenue",
                    description: "JOIN with ORDER BY and LIMIT",
                    category: "Complex Query"
                },
                {
                    text: "Update customer email for customer ID 123",
                    description: "Simple UPDATE statement",
                    category: "Data Modification"
                },
                {
                    text: "Show sales data grouped by month for this year",
                    description: "GROUP BY with date functions",
                    category: "Analytics"
                }
            ];
        },

        /**
         * Estimate query complexity based on SQL structure
         * @param {string} sql - The SQL query
         * @returns {object} Complexity assessment
         */
        estimateComplexity: function (sql) {
            if (!sql) return { level: 'Unknown', score: 0, factors: [] };

            const lowerSQL = sql.toLowerCase();
            let score = 0;
            const factors = [];

            // Count tables
            const tables = this.extractTableNames(sql);
            score += tables.length * 2;
            if (tables.length > 1) {
                factors.push(`${tables.length} tables involved`);
            }

            // Count JOINs
            const joinCount = (sql.match(/join/gi) || []).length;
            score += joinCount * 3;
            if (joinCount > 0) {
                factors.push(`${joinCount} JOIN operations`);
            }

            // Count subqueries
            const subqueryCount = (sql.match(/\(/g) || []).length;
            score += subqueryCount * 2;
            if (subqueryCount > 2) {
                factors.push(`${subqueryCount} nested expressions`);
            }

            // Check for aggregations
            const aggregations = ['count', 'sum', 'avg', 'min', 'max'];
            const aggCount = aggregations.filter(agg => lowerSQL.includes(agg)).length;
            score += aggCount;
            if (aggCount > 0) {
                factors.push(`${aggCount} aggregation functions`);
            }

            // Check for window functions
            if (lowerSQL.includes('over(') || lowerSQL.includes('partition by')) {
                score += 5;
                factors.push('Window functions used');
            }

            // Determine complexity level
            let level;
            if (score < 5) {
                level = 'Simple';
            } else if (score < 15) {
                level = 'Moderate';
            } else if (score < 25) {
                level = 'Complex';
            } else {
                level = 'Very Complex';
            }

            return {
                level: level,
                score: score,
                factors: factors
            };
        },

        /**
         * Generate SQL from common patterns
         * @param {object} pattern - The pattern configuration
         * @returns {string} Generated SQL
         */
        generateSQLFromPattern: function (pattern) {
            // Sanitize all pattern components before SQL generation
            const sanitizedPattern = {
                type: SecurityUtils.sanitizeSQLParameter(pattern.type),
                table: SecurityUtils.sanitizeSQLParameter(pattern.table),
                columns: pattern.columns ? SecurityUtils.sanitizeSQLParameter(pattern.columns) : '*',
                condition: pattern.condition ? SecurityUtils.sanitizeSQLParameter(pattern.condition) : '',
                values: pattern.values ? SecurityUtils.sanitizeSQLParameter(pattern.values) : '',
                setClause: pattern.setClause ? SecurityUtils.sanitizeSQLParameter(pattern.setClause) : ''
            };
            
            let generatedSQL = "";
            
            switch (sanitizedPattern.type) {
                case 'SELECT_ALL':
                    generatedSQL = "SELECT * FROM ?";
                    break;
                    
                case 'SELECT_WHERE':
                    generatedSQL = "SELECT ? FROM ? WHERE ?";
                    break;
                    
                case 'COUNT':
                    generatedSQL = "SELECT COUNT(*) FROM ?";
                    break;
                    
                case 'INSERT':
                    generatedSQL = "INSERT INTO ? (?) VALUES (?)";
                    break;
                    
                case 'UPDATE':
                    generatedSQL = "UPDATE ? SET ? WHERE ?";
                    break;
                    
                case 'DELETE':
                    generatedSQL = "DELETE FROM ? WHERE ?";
                    break;
                    
                default:
                    return "";
            }
            
            // Validate the generated SQL
            const validation = SecurityUtils.validateSQL(generatedSQL);
            if (!validation.isValid) {
                Log.error("Generated SQL failed validation", validation.errors);
                return "";
            }
            
            return validation.sanitized;
        },

        /**
         * Get SQL dialect-specific syntax information
         * @param {string} dialect - The SQL dialect
         * @returns {object} Dialect-specific information
         */
        getDialectInfo: function (dialect) {
            const dialects = {
                'HANA': {
                    name: 'SAP HANA',
                    features: ['Column store', 'In-memory processing', 'SQL Script'],
                    limitSyntax: 'LIMIT n',
                    stringConcat: '||',
                    dateFormat: 'YYYY-MM-DD'
                },
                'POSTGRESQL': {
                    name: 'PostgreSQL',
                    features: ['JSONB support', 'Arrays', 'Custom types'],
                    limitSyntax: 'LIMIT n',
                    stringConcat: '||',
                    dateFormat: 'YYYY-MM-DD'
                },
                'MYSQL': {
                    name: 'MySQL',
                    features: ['Full-text search', 'JSON support', 'Partitioning'],
                    limitSyntax: 'LIMIT n',
                    stringConcat: 'CONCAT()',
                    dateFormat: 'YYYY-MM-DD'
                },
                'SQLITE': {
                    name: 'SQLite',
                    features: ['Lightweight', 'Serverless', 'Cross-platform'],
                    limitSyntax: 'LIMIT n',
                    stringConcat: '||',
                    dateFormat: 'YYYY-MM-DD'
                },
                'ORACLE': {
                    name: 'Oracle Database',
                    features: ['Advanced analytics', 'Partitioning', 'PL/SQL'],
                    limitSyntax: 'ROWNUM <= n',
                    stringConcat: '||',
                    dateFormat: 'YYYY-MM-DD'
                },
                'SQLSERVER': {
                    name: 'Microsoft SQL Server',
                    features: ['T-SQL', 'Columnstore indexes', 'In-memory OLTP'],
                    limitSyntax: 'TOP n',
                    stringConcat: '+',
                    dateFormat: 'YYYY-MM-DD'
                }
            };

            return dialects[dialect] || {
                name: 'Unknown',
                features: [],
                limitSyntax: 'LIMIT n',
                stringConcat: '||',
                dateFormat: 'YYYY-MM-DD'
            };
        }
    };
});