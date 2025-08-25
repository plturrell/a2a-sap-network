sap.ui.define([
    "sap/m/MessageToast",
    "sap/base/Log",
    "a2a/network/agent11/ext/utils/SecurityUtils"
], (MessageToast, Log, SecurityUtils) => {
    "use strict";

    return {
        /**
         * Comprehensive SQL query validation with enhanced security
         * @param {string} sql - The SQL query to validate
         * @param {string} dialect - The SQL dialect (HANA, POSTGRESQL, etc.)
         * @param {object} options - Validation options
         * @returns {object} Enhanced validation result
         */
        validateSQL(sql, dialect, options) {
            options = options || {};

            // Use SecurityUtils for comprehensive security validation
            const securityValidation = SecurityUtils.validateSQL(sql, options.parameters, {
                allowLiterals: options.allowLiterals || false,
                allowedOperations: options.allowedOperations,
                minSecurityScore: options.minSecurityScore || 70
            });

            if (!securityValidation.isValid) {
                return {
                    isValid: false,
                    errors: securityValidation.errors,
                    warnings: securityValidation.warnings || [],
                    suggestions: [],
                    sanitized: securityValidation.sanitized,
                    securityScore: securityValidation.securityScore,
                    riskLevel: securityValidation.riskLevel,
                    detectedPatterns: securityValidation.detectedPatterns
                };
            }

            const errors = [];
            const warnings = [...(securityValidation.warnings || [])];
            const suggestions = [];
            const sanitizedSQL = securityValidation.sanitized;

            // Enhanced syntax checks with dialect-specific rules
            const syntaxIssues = this._checkEnhancedSyntax(sanitizedSQL, dialect, options);
            if (syntaxIssues.errors.length > 0) {
                errors.push(...syntaxIssues.errors);
            }
            if (syntaxIssues.warnings.length > 0) {
                warnings.push(...syntaxIssues.warnings);
            }

            // Enhanced performance suggestions
            const performanceSuggestions = this._getEnhancedPerformanceSuggestions(sanitizedSQL, dialect);
            suggestions.push(...performanceSuggestions);

            // Advanced query complexity validation with resource limits
            const complexityValidation = SecurityUtils.validateQueryComplexity(sanitizedSQL, options.complexityLimits);
            if (!complexityValidation.isValid) {
                warnings.push(...complexityValidation.issues);
            }

            // Add complexity metrics to result
            const complexityMetrics = {
                complexity: complexityValidation.complexity,
                score: complexityValidation.score,
                riskLevel: complexityValidation.riskLevel,
                estimatedExecutionTime: complexityValidation.estimatedExecutionTime,
                resourceRequirements: complexityValidation.resourceRequirements
            };

            return {
                isValid: errors.length === 0 && securityValidation.securityScore >= (options.minSecurityScore || 70),
                errors,
                warnings,
                suggestions,
                sanitized: sanitizedSQL,
                securityScore: securityValidation.securityScore,
                riskLevel: securityValidation.riskLevel,
                detectedPatterns: securityValidation.detectedPatterns,
                complexity: complexityMetrics,
                dialect,
                queryHash: this._generateQueryFingerprint(sanitizedSQL)
            };
        },

        /**
         * Sanitizes SQL input to prevent injection attacks
         * @param {string} sql - The SQL to sanitize
         * @returns {string} Sanitized SQL
         */
        sanitizeSQL(sql) {
            if (!sql) {return "";}

            // Use SecurityUtils for comprehensive sanitization
            const validation = SecurityUtils.validateSQL(sql);
            return validation.sanitized || "";
        },

        /**
         * Security vulnerability checks
         * @private
         */
        _checkSecurityVulnerabilities(sql) {
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

            const checkInjectionPattern = (pattern) => {
                if (pattern.test(sql)) {
                    issues.push("Potential SQL injection pattern detected");
                }
            };
            injectionPatterns.forEach(checkInjectionPattern);

            // Check for dangerous functions
            const dangerousFunctions = ["exec", "execute", "eval", "system"];
            const checkDangerousFunction = (func) => {
                if (lowerSQL.includes(`${func }(`)) {
                    issues.push(`Dangerous function '${ func }' detected`);
                }
            };
            dangerousFunctions.forEach(checkDangerousFunction);

            return issues;
        },

        /**
         * Basic SQL syntax validation
         * @private
         */
        _checkBasicSyntax(sql, dialect) {
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
            if (lowerSQL.includes("select") && !lowerSQL.includes("from") &&
                !lowerSQL.includes("dual") && dialect !== "HANA") {
                warnings.push("SELECT statement without FROM clause");
            }

            // Check for missing WHERE clause in UPDATE/DELETE
            if ((lowerSQL.includes("update") || lowerSQL.includes("delete")) &&
                !lowerSQL.includes("where")) {
                warnings.push("UPDATE/DELETE without WHERE clause - this affects all rows");
            }

            return { errors, warnings };
        },

        /**
         * Get performance optimization suggestions
         * @private
         */
        _getPerformanceSuggestions(sql) {
            const suggestions = [];
            const lowerSQL = sql.toLowerCase();

            // Check for SELECT *
            if (lowerSQL.includes("select *")) {
                suggestions.push({
                    type: "Performance",
                    message: "Consider selecting specific columns instead of using SELECT *",
                    impact: "Medium"
                });
            }

            // Check for LIKE with leading wildcard
            if (lowerSQL.includes("like '%")) {
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
        formatSQL(sql) {
            if (!sql) {return "";}

            // Basic SQL formatting
            let formatted = sql
                .replace(/\bSELECT\b/gi, "\nSELECT")
                .replace(/\bFROM\b/gi, "\nFROM")
                .replace(/\bWHERE\b/gi, "\nWHERE")
                .replace(/\bAND\b/gi, "\n  AND")
                .replace(/\bOR\b/gi, "\n  OR")
                .replace(/\bORDER BY\b/gi, "\nORDER BY")
                .replace(/\bGROUP BY\b/gi, "\nGROUP BY")
                .replace(/\bHAVING\b/gi, "\nHAVING")
                .replace(/\bJOIN\b/gi, "\nJOIN")
                .replace(/\bINNER JOIN\b/gi, "\nINNER JOIN")
                .replace(/\bLEFT JOIN\b/gi, "\nLEFT JOIN")
                .replace(/\bRIGHT JOIN\b/gi, "\nRIGHT JOIN")
                .replace(/\bFULL JOIN\b/gi, "\nFULL JOIN")
                .replace(/\bUNION\b/gi, "\nUNION")
                .replace(/\bINSERT INTO\b/gi, "\nINSERT INTO")
                .replace(/\bVALUES\b/gi, "\nVALUES")
                .replace(/\bUPDATE\b/gi, "\nUPDATE")
                .replace(/\bSET\b/gi, "\nSET")
                .replace(/\bDELETE FROM\b/gi, "\nDELETE FROM");

            // Clean up extra whitespace and newlines
            formatted = formatted
                .replace(/\n\s*\n/g, "\n")
                .replace(/^\n+/, "")
                .trim();

            return formatted;
        },

        /**
         * Extract table names from SQL query
         * @param {string} sql - The SQL query
         * @returns {array} Array of table names
         */
        extractTableNames(sql) {
            if (!sql) {return [];}

            const tables = [];
            const _lowerSQL = sql.toLowerCase();

            // Extract FROM clause tables
            const fromMatches = sql.match(/from\s+(\w+)/gi);
            if (fromMatches) {
                const processFromMatch = (match) => {
                    const tableName = match.replace(/from\s+/gi, "").trim();
                    if (!tables.includes(tableName)) {
                        tables.push(tableName);
                    }
                };
                fromMatches.forEach(processFromMatch);
            }

            // Extract JOIN clause tables
            const joinMatches = sql.match(/join\s+(\w+)/gi);
            if (joinMatches) {
                const processJoinMatch = (match) => {
                    const tableName = match.replace(/.*join\s+/gi, "").trim();
                    if (!tables.includes(tableName)) {
                        tables.push(tableName);
                    }
                };
                joinMatches.forEach(processJoinMatch);
            }

            return tables;
        },

        /**
         * Parse natural language query to extract intent and entities
         * @param {string} naturalLanguage - The natural language query
         * @returns {object} Parsed intent and entities
         */
        parseNaturalLanguage(naturalLanguage) {
            if (!naturalLanguage) {return { intent: null, entities: [], confidence: 0 };}

            // Sanitize the input to prevent injection
            const sanitizedQuery = SecurityUtils.escapeHTML(naturalLanguage.trim());
            const lowerQuery = sanitizedQuery.toLowerCase();
            const entities = [];
            let intent = null;
            let confidence = 0;

            // Simple intent recognition
            if (lowerQuery.includes("show") || lowerQuery.includes("list") || lowerQuery.includes("get")) {
                intent = "SELECT";
                confidence = 70;
            } else if (lowerQuery.includes("add") || lowerQuery.includes("insert") || lowerQuery.includes("create")) {
                intent = "INSERT";
                confidence = 70;
            } else if (lowerQuery.includes("update") || lowerQuery.includes("change") || lowerQuery.includes("modify")) {
                intent = "UPDATE";
                confidence = 70;
            } else if (lowerQuery.includes("delete") || lowerQuery.includes("remove")) {
                intent = "DELETE";
                confidence = 70;
            } else if (lowerQuery.includes("count") || lowerQuery.includes("how many")) {
                intent = "COUNT";
                confidence = 80;
            }

            // Simple entity extraction with sanitization
            const tableKeywords = ["users", "orders", "products", "customers", "employees", "sales"];
            const checkTableKeyword = (keyword) => {
                if (lowerQuery.includes(keyword)) {
                    entities.push({
                        type: "TABLE",
                        value: SecurityUtils.sanitizeSQLParameter(keyword),
                        confidence: 60
                    });
                }
            };
            tableKeywords.forEach(checkTableKeyword);

            return {
                intent,
                entities,
                confidence,
                sanitizedInput: sanitizedQuery
            };
        },

        /**
         * Generate example questions for different categories
         * @returns {array} Array of example questions
         */
        getExampleQuestions() {
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
        estimateComplexity(sql) {
            if (!sql) {return { level: "Unknown", score: 0, factors: [] };}

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
            const aggregations = ["count", "sum", "avg", "min", "max"];
            const includesAggregation = (agg) => lowerSQL.includes(agg);
            const aggCount = aggregations.filter(includesAggregation).length;
            score += aggCount;
            if (aggCount > 0) {
                factors.push(`${aggCount} aggregation functions`);
            }

            // Check for window functions
            if (lowerSQL.includes("over(") || lowerSQL.includes("partition by")) {
                score += 5;
                factors.push("Window functions used");
            }

            // Determine complexity level
            let level;
            if (score < 5) {
                level = "Simple";
            } else if (score < 15) {
                level = "Moderate";
            } else if (score < 25) {
                level = "Complex";
            } else {
                level = "Very Complex";
            }

            return {
                level,
                score,
                factors
            };
        },

        /**
         * Generate SQL from common patterns
         * @param {object} pattern - The pattern configuration
         * @returns {string} Generated SQL
         */
        generateSQLFromPattern(pattern) {
            // Sanitize all pattern components before SQL generation
            const sanitizedPattern = {
                type: SecurityUtils.sanitizeSQLParameter(pattern.type),
                table: SecurityUtils.sanitizeSQLParameter(pattern.table),
                columns: pattern.columns ? SecurityUtils.sanitizeSQLParameter(pattern.columns) : "*",
                condition: pattern.condition ? SecurityUtils.sanitizeSQLParameter(pattern.condition) : "",
                values: pattern.values ? SecurityUtils.sanitizeSQLParameter(pattern.values) : "",
                setClause: pattern.setClause ? SecurityUtils.sanitizeSQLParameter(pattern.setClause) : ""
            };

            let generatedSQL = "";

            switch (sanitizedPattern.type) {
            case "SELECT_ALL":
                generatedSQL = "SELECT * FROM ?";
                break;

            case "SELECT_WHERE":
                generatedSQL = "SELECT ? FROM ? WHERE ?";
                break;

            case "COUNT":
                generatedSQL = "SELECT COUNT(*) FROM ?";
                break;

            case "INSERT":
                generatedSQL = "INSERT INTO ? (?) VALUES (?)";
                break;

            case "UPDATE":
                generatedSQL = "UPDATE ? SET ? WHERE ?";
                break;

            case "DELETE":
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
        getDialectInfo(dialect) {
            const dialects = {
                "HANA": {
                    name: "SAP HANA",
                    features: ["Column store", "In-memory processing", "SQL Script"],
                    limitSyntax: "LIMIT n",
                    stringConcat: "||",
                    dateFormat: "YYYY-MM-DD"
                },
                "POSTGRESQL": {
                    name: "PostgreSQL",
                    features: ["JSONB support", "Arrays", "Custom types"],
                    limitSyntax: "LIMIT n",
                    stringConcat: "||",
                    dateFormat: "YYYY-MM-DD"
                },
                "MYSQL": {
                    name: "MySQL",
                    features: ["Full-text search", "JSON support", "Partitioning"],
                    limitSyntax: "LIMIT n",
                    stringConcat: "CONCAT()",
                    dateFormat: "YYYY-MM-DD"
                },
                "SQLITE": {
                    name: "SQLite",
                    features: ["Lightweight", "Serverless", "Cross-platform"],
                    limitSyntax: "LIMIT n",
                    stringConcat: "||",
                    dateFormat: "YYYY-MM-DD"
                },
                "ORACLE": {
                    name: "Oracle Database",
                    features: ["Advanced analytics", "Partitioning", "PL/SQL"],
                    limitSyntax: "ROWNUM <= n",
                    stringConcat: "||",
                    dateFormat: "YYYY-MM-DD"
                },
                "SQLSERVER": {
                    name: "Microsoft SQL Server",
                    features: ["T-SQL", "Columnstore indexes", "In-memory OLTP"],
                    limitSyntax: "TOP n",
                    stringConcat: "+",
                    dateFormat: "YYYY-MM-DD"
                }
            };

            return dialects[dialect] || {
                name: "Unknown",
                features: [],
                limitSyntax: "LIMIT n",
                stringConcat: "||",
                dateFormat: "YYYY-MM-DD"
            };
        },

        /**
         * Enhanced syntax checking with dialect-specific rules
         * @private
         */
        _checkEnhancedSyntax(sql, dialect, options) {
            const errors = [];
            const warnings = [];
            const lowerSQL = sql.toLowerCase();

            // Check for balanced parentheses
            const openParens = (sql.match(/\(/g) || []).length;
            const closeParens = (sql.match(/\)/g) || []).length;
            if (openParens !== closeParens) {
                errors.push("Unbalanced parentheses in SQL query");
            }

            // Check for proper quotes
            const singleQuotes = (sql.match(/'/g) || []).length;
            const doubleQuotes = (sql.match(/"/g) || []).length;
            if (singleQuotes % 2 !== 0) {
                errors.push("Unmatched single quotes in SQL query");
            }
            if (doubleQuotes % 2 !== 0) {
                errors.push("Unmatched double quotes in SQL query");
            }

            // Dialect-specific validations
            if (dialect === "HANA") {
                if (lowerSQL.includes("dual") && !lowerSQL.includes("sys.dual")) {
                    warnings.push("Use SYS.DUAL instead of DUAL in SAP HANA");
                }
            } else if (dialect === "POSTGRESQL") {
                if (lowerSQL.includes("limit") && lowerSQL.includes("top")) {
                    errors.push("PostgreSQL uses LIMIT, not TOP");
                }
            } else if (dialect === "SQLSERVER") {
                if (lowerSQL.includes("limit") && !lowerSQL.includes("top")) {
                    errors.push("SQL Server uses TOP, not LIMIT");
                }
            }

            // Check for potentially dangerous patterns that SecurityUtils might miss
            if (lowerSQL.includes("truncate table")) {
                warnings.push("TRUNCATE TABLE operation detected - ensure proper permissions");
            }

            if (lowerSQL.includes("with recursive") && dialect !== "POSTGRESQL") {
                warnings.push("Recursive CTEs may not be supported in all databases");
            }

            return { errors, warnings };
        },

        /**
         * Enhanced performance suggestions
         * @private
         */
        _getEnhancedPerformanceSuggestions(sql, dialect) {
            const suggestions = [];
            const lowerSQL = sql.toLowerCase();

            // Check for SELECT *
            if (lowerSQL.includes("select *")) {
                suggestions.push({
                    type: "Performance",
                    message: "Consider selecting specific columns instead of using SELECT *",
                    impact: "Medium",
                    suggestion: "Replace SELECT * with specific column names"
                });
            }

            // Check for LIKE with leading wildcard
            if (lowerSQL.includes("like '%")) {
                suggestions.push({
                    type: "Performance",
                    message: "Leading wildcards in LIKE clauses prevent index usage",
                    impact: "High",
                    suggestion: "Consider full-text search or restructuring the query"
                });
            }

            // Check for functions in WHERE clause
            if (/where\s+\w*\s*\(/i.test(sql)) {
                suggestions.push({
                    type: "Performance",
                    message: "Functions in WHERE clause may prevent index usage",
                    impact: "Medium",
                    suggestion: "Move functions to SELECT clause or use computed columns"
                });
            }

            // Check for OR conditions
            if ((sql.match(/\bor\b/gi) || []).length > 2) {
                suggestions.push({
                    type: "Performance",
                    message: "Multiple OR conditions can be expensive",
                    impact: "Medium",
                    suggestion: "Consider using UNION or IN clauses"
                });
            }

            // Check for subqueries that could be JOINs
            if (lowerSQL.includes("in (select") && !lowerSQL.includes("exists")) {
                suggestions.push({
                    type: "Performance",
                    message: "IN subqueries can often be optimized with EXISTS or JOINs",
                    impact: "Medium",
                    suggestion: "Consider rewriting with EXISTS or JOIN"
                });
            }

            // Dialect-specific suggestions
            if (dialect === "HANA") {
                if (lowerSQL.includes("group by") && !lowerSQL.includes("order by")) {
                    suggestions.push({
                        type: "Performance",
                        message: "SAP HANA performs better with explicit ORDER BY after GROUP BY",
                        impact: "Low",
                        suggestion: "Add ORDER BY clause after GROUP BY"
                    });
                }
            }

            return suggestions;
        },

        /**
         * Generate query fingerprint for caching and analysis
         * @private
         */
        _generateQueryFingerprint(sql) {
            // Normalize SQL for fingerprinting
            const normalized = sql
                .replace(/\s+/g, " ")
                .replace(/'[^']*'/g, "'?'")
                .replace(/\d+/g, "?")
                .toLowerCase()
                .trim();

            // Simple hash function
            let hash = 0;
            for (let i = 0; i < normalized.length; i++) {
                const char = normalized.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            return Math.abs(hash).toString(16);
        },

        /**
         * Extract enhanced entities from natural language
         * @private
         */
        _extractEnhancedEntities(query, context) {
            const entities = [];
            const _lowerQuery = query.toLowerCase();

            // Enhanced table pattern recognition
            const tablePatterns = [
                /(?:from|in|on|table)\s+([a-z_][a-z0-9_]*)/gi,
                /(?:users?|customers?|orders?|products?|employees?|sales?|invoices?|payments?)/gi
            ];

            const processTablePattern = (pattern) => {
                let match;
                while ((match = pattern.exec(query)) !== null) {
                    const tableName = match[1] || match[0];
                    const findTableEntity = (e) => e.value === tableName && e.type === "TABLE";
                    if (!entities.find(findTableEntity)) {
                        entities.push({
                            type: "TABLE",
                            value: SecurityUtils.sanitizeSQLParameter(tableName),
                            confidence: 75,
                            position: match.index
                        });
                    }
                }
            };
            tablePatterns.forEach(processTablePattern);

            // Column pattern recognition
            const columnPatterns = [
                /(?:select|where|order by)\s+([a-z_][a-z0-9_]*)/gi,
                /(?:name|id|date|time|status|amount|price|quantity)/gi
            ];

            const processColumnPattern = (pattern) => {
                let match;
                while ((match = pattern.exec(query)) !== null) {
                    const columnName = match[1] || match[0];
                    const findColumnEntity = (e) => e.value === columnName && e.type === "COLUMN";
                    if (!entities.find(findColumnEntity)) {
                        entities.push({
                            type: "COLUMN",
                            value: SecurityUtils.sanitizeSQLParameter(columnName),
                            confidence: 60,
                            position: match.index
                        });
                    }
                }
            };
            columnPatterns.forEach(processColumnPattern);

            // Date/time pattern recognition
            const datePattern = /(?:today|yesterday|last week|this month|this year|\d{4}-\d{2}-\d{2})/gi;
            let match;
            while ((match = datePattern.exec(query)) !== null) {
                entities.push({
                    type: "DATE",
                    value: match[0],
                    confidence: 85,
                    position: match.index
                });
            }

            return entities;
        },

        /**
         * Generate secure SQL templates based on intent and entities
         * @private
         */
        _generateSecureSQLTemplates(intent, entities, context) {
            const templates = [];
            const isTableEntity = (e) => e.type === "TABLE";
            const isColumnEntity = (e) => e.type === "COLUMN";
            const extractValue = (e) => e.value;
            const tables = entities.filter(isTableEntity).map(extractValue);
            const columns = entities.filter(isColumnEntity).map(extractValue);

            if (!tables.length) {
                return templates;
            }

            const primaryTable = tables[0];

            switch (intent) {
            case "SELECT":
                templates.push({
                    sql: `SELECT ${ columns.length ? columns.join(", ") : "*" } FROM ${ primaryTable}`,
                    confidence: 80,
                    parameters: {},
                    description: "Basic SELECT query"
                });

                if (columns.length > 0) {
                    templates.push({
                        sql: `SELECT ${ columns[0] } FROM ${ primaryTable } WHERE ${ columns[0] } = ?`,
                        confidence: 75,
                        parameters: { param1: "value" },
                        description: "SELECT with WHERE condition"
                    });
                }
                break;

            case "COUNT":
                templates.push({
                    sql: `SELECT COUNT(*) FROM ${ primaryTable}`,
                    confidence: 90,
                    parameters: {},
                    description: "Count all records"
                });

                if (columns.length > 0) {
                    templates.push({
                        sql: `SELECT COUNT(*) FROM ${ primaryTable } WHERE ${ columns[0] } = ?`,
                        confidence: 85,
                        parameters: { param1: "value" },
                        description: "Count with condition"
                    });
                }
                break;

            case "INSERT":
                if (columns.length > 0) {
                    const createPlaceholder = () => "?";
                    const placeholders = columns.map(createPlaceholder).join(", ");
                    const createParameterEntry = (col, i) => [`param${i + 1}`, "value"];
                    templates.push({
                        sql: `INSERT INTO ${ primaryTable } (${ columns.join(", ") }) VALUES (${ placeholders })`,
                        confidence: 75,
                        parameters: Object.fromEntries(columns.map(createParameterEntry)),
                        description: "Insert new record"
                    });
                }
                break;

            case "UPDATE":
                if (columns.length > 1) {
                    const createSetClause = function(col) { return `${col } = ?`; };
                    const setClause = columns.slice(1).map(createSetClause).join(", ");
                    const createUpdateParameterEntry = (col, i) => [`param${i + 1}`, "value"];
                    templates.push({
                        sql: `UPDATE ${ primaryTable } SET ${ setClause } WHERE ${ columns[0] } = ?`,
                        confidence: 70,
                        parameters: Object.fromEntries(columns.map(createUpdateParameterEntry)),
                        description: "Update existing record"
                    });
                }
                break;
            }

            return templates;
        },

        /**
         * Assess natural language complexity
         * @private
         */
        _assessNLComplexity(query) {
            const indicators = {
                simple: ["show", "get", "list"],
                moderate: ["where", "and", "count", "sum"],
                complex: ["join", "group by", "having", "subquery", "union"],
                veryComplex: ["recursive", "window", "pivot", "case when"]
            };

            const lowerQuery = query.toLowerCase();
            let maxLevel = "simple";

            const checkLevelIndicators = ([level, words]) => {
                const includesWord = (word) => lowerQuery.includes(word);
                if (words.some(includesWord)) {
                    maxLevel = level;
                }
            };
            Object.entries(indicators).forEach(checkLevelIndicators);

            return maxLevel;
        },

        /**
         * Sanitize pattern for SQL generation
         * @private
         */
        _sanitizePattern(pattern) {
            const errors = [];

            if (!pattern || typeof pattern !== "object") {
                return {
                    isValid: false,
                    errors: ["Pattern must be a valid object"]
                };
            }

            const sanitized = {
                type: SecurityUtils.sanitizeSQLParameter(pattern.type || ""),
                table: SecurityUtils.sanitizeSQLParameter(pattern.table || ""),
                columns: pattern.columns ? SecurityUtils.sanitizeSQLParameter(pattern.columns) : "*",
                condition: pattern.condition ? SecurityUtils.sanitizeSQLParameter(pattern.condition) : "",
                values: pattern.values ? SecurityUtils.sanitizeSQLParameter(pattern.values) : "",
                setClause: pattern.setClause ? SecurityUtils.sanitizeSQLParameter(pattern.setClause) : ""
            };

            if (!sanitized.type) {
                errors.push("Pattern type is required");
            }

            if (!sanitized.table) {
                errors.push("Table name is required");
            }

            return {
                isValid: errors.length === 0,
                pattern: sanitized,
                errors
            };
        },

        /**
         * Generate secure SQL template
         * @private
         */
        _generateSecureTemplate(pattern, options) {
            const templates = {
                "SELECT_ALL": {
                    template: "SELECT * FROM :table",
                    parameters: { table: pattern.table }
                },
                "SELECT_WHERE": {
                    template: "SELECT :columns FROM :table WHERE :condition",
                    parameters: {
                        columns: pattern.columns || "*",
                        table: pattern.table,
                        condition: pattern.condition || "1=1"
                    }
                },
                "COUNT": {
                    template: "SELECT COUNT(*) FROM :table",
                    parameters: { table: pattern.table }
                },
                "INSERT": {
                    template: "INSERT INTO :table (:columns) VALUES (:values)",
                    parameters: {
                        table: pattern.table,
                        columns: pattern.columns,
                        values: pattern.values
                    }
                },
                "UPDATE": {
                    template: "UPDATE :table SET :setClause WHERE :condition",
                    parameters: {
                        table: pattern.table,
                        setClause: pattern.setClause,
                        condition: pattern.condition || "1=0" // Safer default
                    }
                },
                "DELETE": {
                    template: "DELETE FROM :table WHERE :condition",
                    parameters: {
                        table: pattern.table,
                        condition: pattern.condition || "1=0" // Safer default
                    }
                }
            };

            return templates[pattern.type] || null;
        }
    };
});