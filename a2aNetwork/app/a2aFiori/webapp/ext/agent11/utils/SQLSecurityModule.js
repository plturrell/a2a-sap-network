sap.ui.define([
    "sap/base/Log"
], (Log) => {
    "use strict";

    /**
     * SQL Security Module - Prevents SQL injection through parameterized queries
     * All SQL operations must go through this module
     */
    return {
        /**
         * Create a parameterized query object
         * @param {string} template - SQL template with ? placeholders
         * @param {array} parameters - Array of parameters to bind
         * @returns {object} Secure query object
         */
        createParameterizedQuery(template, parameters) {
            if (!template || typeof template !== "string") {
                throw new Error("SQL template must be a non-empty string");
            }

            // Count placeholders
            const placeholderCount = (template.match(/\?/g) || []).length;

            // Validate parameter count
            if (!parameters) {
                parameters = [];
            }

            if (placeholderCount !== parameters.length) {
                throw new Error(`Parameter count mismatch: expected ${ placeholderCount }, got ${ parameters.length}`);
            }

            // Validate and sanitize parameters
            const sanitizedParams = parameters.map(this.sanitizeParameter.bind(this));

            return {
                template,
                parameters: sanitizedParams,
                placeholderCount,
                isParameterized: true,
                hash: this._generateQueryHash(template)
            };
        },

        /**
         * Sanitize a single parameter value
         * @param {any} value - Parameter value to sanitize
         * @returns {any} Sanitized value
         */
        sanitizeParameter(value) {
            if (value === null || value === undefined) {
                return null;
            }

            // Handle different types
            if (typeof value === "string") {
                // Remove any SQL meta-characters that could break out of parameterization
                // Note: In a proper parameterized query, these should be safe, but we add extra protection
                return value
                    .replace(/\\/g, "\\\\") // Escape backslashes
                    .replace(/\0/g, "\\0") // Null bytes
                    .replace(/\n/g, "\\n") // Newlines
                    .replace(/\r/g, "\\r") // Carriage returns
                    .replace(/\x1a/g, "\\Z"); // Ctrl-Z
            } else if (typeof value === "number") {
                // Validate number is finite
                if (!isFinite(value)) {
                    throw new Error(`Invalid number parameter: ${ value}`);
                }
                return value;
            } else if (typeof value === "boolean") {
                return value ? 1 : 0;
            } else if (value instanceof Date) {
                // Format date as ISO string
                return value.toISOString();
            }
            // For other types, convert to string and sanitize
            return this.sanitizeParameter(String(value));

        },

        /**
         * Validate SQL identifier (table/column name)
         * @param {string} identifier - Identifier to validate
         * @returns {string} Validated identifier
         */
        validateIdentifier(identifier) {
            if (!identifier || typeof identifier !== "string") {
                throw new Error("Invalid identifier");
            }

            // Only allow alphanumeric, underscore, and dot (for schema.table)
            if (!/^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$/.test(identifier)) {
                throw new Error(`Invalid identifier format: ${ identifier}`);
            }

            // Check against SQL reserved words
            const reserved = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "EXEC", "EXECUTE"];
            if (reserved.includes(identifier.toUpperCase())) {
                throw new Error(`Reserved word used as identifier: ${ identifier}`);
            }

            return identifier;
        },

        /**
         * Build SELECT query safely
         * @param {object} options - Query options
         * @returns {object} Parameterized query
         */
        buildSelectQuery(options) {
            const table = this.validateIdentifier(options.table);
            const columns = options.columns ? options.columns.map(this.validateIdentifier.bind(this)) : ["*"];
            const conditions = options.where || {};
            const orderBy = options.orderBy;
            const limit = options.limit;

            let query = `SELECT ${ columns.join(", ") } FROM ${ table}`;
            const parameters = [];

            // Add WHERE clause
            const whereClause = this._buildWhereClause(conditions, parameters);
            if (whereClause) {
                query += ` WHERE ${ whereClause}`;
            }

            // Add ORDER BY
            if (orderBy) {
                const orderColumns = Array.isArray(orderBy) ? orderBy : [orderBy];
                const orderParts = orderColumns.map(col => {
                    if (typeof col === "string") {
                        return `${this.validateIdentifier(col) } ASC`;
                    }
                    return `${this.validateIdentifier(col.column) } ${ col.desc ? "DESC" : "ASC"}`;

                });
                query += ` ORDER BY ${ orderParts.join(", ")}`;
            }

            // Add LIMIT
            if (limit && typeof limit === "number" && limit > 0) {
                query += ` LIMIT ${ parseInt(limit, 10)}`;
            }

            return this.createParameterizedQuery(query, parameters);
        },

        /**
         * Build INSERT query safely
         * @param {object} options - Query options
         * @returns {object} Parameterized query
         */
        buildInsertQuery(options) {
            const table = this.validateIdentifier(options.table);
            const data = options.data || {};

            const columns = [];
            const placeholders = [];
            const parameters = [];

            Object.keys(data).forEach(column => {
                columns.push(this.validateIdentifier(column));
                placeholders.push("?");
                parameters.push(data[column]);
            });

            if (columns.length === 0) {
                throw new Error("No data provided for INSERT");
            }

            const query = `INSERT INTO ${ table } (${ columns.join(", ") }) VALUES (${ placeholders.join(", ") })`;

            return this.createParameterizedQuery(query, parameters);
        },

        /**
         * Build UPDATE query safely
         * @param {object} options - Query options
         * @returns {object} Parameterized query
         */
        buildUpdateQuery(options) {
            const table = this.validateIdentifier(options.table);
            const data = options.data || {};
            const conditions = options.where || {};

            const setClauses = [];
            const parameters = [];

            // Build SET clause
            Object.keys(data).forEach(column => {
                setClauses.push(`${this.validateIdentifier(column) } = ?`);
                parameters.push(data[column]);
            });

            if (setClauses.length === 0) {
                throw new Error("No data provided for UPDATE");
            }

            let query = `UPDATE ${ table } SET ${ setClauses.join(", ")}`;

            // Add WHERE clause (mandatory for safety)
            const whereClause = this._buildWhereClause(conditions, parameters);
            if (!whereClause) {
                throw new Error("UPDATE without WHERE clause is not allowed");
            }
            query += ` WHERE ${ whereClause}`;

            return this.createParameterizedQuery(query, parameters);
        },

        /**
         * Build DELETE query safely
         * @param {object} options - Query options
         * @returns {object} Parameterized query
         */
        buildDeleteQuery(options) {
            const table = this.validateIdentifier(options.table);
            const conditions = options.where || {};

            let query = `DELETE FROM ${ table}`;
            const parameters = [];

            // Add WHERE clause (mandatory for safety)
            const whereClause = this._buildWhereClause(conditions, parameters);
            if (!whereClause) {
                throw new Error("DELETE without WHERE clause is not allowed");
            }
            query += ` WHERE ${ whereClause}`;

            return this.createParameterizedQuery(query, parameters);
        },

        /**
         * Build WHERE clause from conditions object
         * @private
         */
        _buildWhereClause(conditions, parameters) {
            const clauses = [];

            Object.keys(conditions).forEach(column => {
                const value = conditions[column];
                const validColumn = this.validateIdentifier(column);

                if (value === null) {
                    clauses.push(`${validColumn } IS NULL`);
                } else if (Array.isArray(value)) {
                    // IN clause
                    const placeholders = value.map(() => "?").join(", ");
                    clauses.push(`${validColumn } IN (${ placeholders })`);
                    parameters.push(...value);
                } else if (typeof value === "object" && value.operator) {
                    // Complex condition
                    const op = this._validateOperator(value.operator);
                    if (value.value === null && (op === "=" || op === "!=")) {
                        clauses.push(validColumn + (op === "=" ? " IS NULL" : " IS NOT NULL"));
                    } else {
                        clauses.push(`${validColumn } ${ op } ?`);
                        parameters.push(value.value);
                    }
                } else {
                    // Simple equality
                    clauses.push(`${validColumn } = ?`);
                    parameters.push(value);
                }
            });

            return clauses.join(" AND ");
        },

        /**
         * Validate SQL operator
         * @private
         */
        _validateOperator(operator) {
            const allowed = ["=", "!=", "<", ">", "<=", ">=", "LIKE", "NOT LIKE"];
            const upper = operator.toUpperCase();

            if (!allowed.includes(upper)) {
                throw new Error(`Invalid operator: ${ operator}`);
            }

            return upper;
        },

        /**
         * Generate query hash for caching
         * @private
         */
        _generateQueryHash(query) {
            let hash = 0;
            for (let i = 0; i < query.length; i++) {
                const char = query.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            return Math.abs(hash).toString(16);
        },

        /**
         * Execute parameterized query (mock for UI5)
         * In real implementation, this would send to backend
         */
        executeQuery(parameterizedQuery) {
            if (!parameterizedQuery || !parameterizedQuery.isParameterized) {
                throw new Error("Only parameterized queries can be executed");
            }

            Log.info("Executing parameterized query", {
                template: parameterizedQuery.template,
                paramCount: parameterizedQuery.parameters.length,
                hash: parameterizedQuery.hash
            });

            // Return mock response
            return Promise.resolve({
                success: true,
                rows: [],
                rowCount: 0,
                message: "Query executed successfully"
            });
        }
    };
});