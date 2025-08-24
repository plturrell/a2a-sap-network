sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "a2a/network/agent11/ext/utils/SQLSecurityModule",
    "a2a/network/agent11/ext/utils/SecurityUtils",
    "sap/m/MessageBox",
    "sap/base/Log"
], (Controller, SQLSecurityModule, SecurityUtils, MessageBox, Log) => {
    "use strict";

    /**
     * Secure SQL Controller - All SQL operations must use this controller
     * Prevents SQL injection through mandatory parameterized queries
     */
    return Controller.extend("a2a.network.agent11.ext.controller.SecureSQLController", {

        /**
         * Initialize the controller
         */
        onInit() {
            this._sqlSecurity = SQLSecurityModule;
            this._securityUtils = SecurityUtils;
        },

        /**
         * Execute a SELECT query safely
         * @param {object} queryOptions - Query options
         * @returns {Promise} Query result
         */
        executeSelect(queryOptions) {
            return new Promise((resolve, reject) => {
                try {
                    // Validate user permissions
                    if (!this._securityUtils.hasRole("SQLUser") && !this._securityUtils.hasRole("Admin")) {
                        this._securityUtils.auditLog("SQL_ACCESS_DENIED", { operation: "SELECT" });
                        reject(new Error("Access denied: SQL query requires SQLUser role"));
                        return;
                    }

                    // Validate input
                    if (!queryOptions || !queryOptions.table) {
                        reject(new Error("Table name is required"));
                        return;
                    }

                    // Build secure parameterized query
                    const parameterizedQuery = this._sqlSecurity.buildSelectQuery(queryOptions);

                    // Log the query for audit
                    this._securityUtils.auditLog("SQL_QUERY", {
                        operation: "SELECT",
                        table: queryOptions.table,
                        queryHash: parameterizedQuery.hash
                    });

                    // Execute through secure channel
                    this._executeSecureQuery(parameterizedQuery)
                        .then(resolve)
                        .catch(reject);

                } catch (error) {
                    Log.error("Secure SELECT failed", error);
                    reject(error);
                }
            });
        },

        /**
         * Execute an INSERT query safely
         * @param {object} insertOptions - Insert options
         * @returns {Promise} Insert result
         */
        executeInsert(insertOptions) {
            return new Promise((resolve, reject) => {
                try {
                    // Validate user permissions
                    if (!this._securityUtils.hasRole("SQLWriter") && !this._securityUtils.hasRole("Admin")) {
                        this._securityUtils.auditLog("SQL_ACCESS_DENIED", { operation: "INSERT" });
                        reject(new Error("Access denied: SQL insert requires SQLWriter role"));
                        return;
                    }

                    // Validate input
                    if (!insertOptions || !insertOptions.table || !insertOptions.data) {
                        reject(new Error("Table name and data are required"));
                        return;
                    }

                    // Build secure parameterized query
                    const parameterizedQuery = this._sqlSecurity.buildInsertQuery(insertOptions);

                    // Log the query for audit
                    this._securityUtils.auditLog("SQL_QUERY", {
                        operation: "INSERT",
                        table: insertOptions.table,
                        queryHash: parameterizedQuery.hash
                    });

                    // Execute through secure channel
                    this._executeSecureQuery(parameterizedQuery)
                        .then(resolve)
                        .catch(reject);

                } catch (error) {
                    Log.error("Secure INSERT failed", error);
                    reject(error);
                }
            });
        },

        /**
         * Execute an UPDATE query safely
         * @param {object} updateOptions - Update options
         * @returns {Promise} Update result
         */
        executeUpdate(updateOptions) {
            return new Promise((resolve, reject) => {
                try {
                    // Validate user permissions
                    if (!this._securityUtils.hasRole("SQLWriter") && !this._securityUtils.hasRole("Admin")) {
                        this._securityUtils.auditLog("SQL_ACCESS_DENIED", { operation: "UPDATE" });
                        reject(new Error("Access denied: SQL update requires SQLWriter role"));
                        return;
                    }

                    // Validate input
                    if (!updateOptions || !updateOptions.table || !updateOptions.data || !updateOptions.where) {
                        reject(new Error("Table name, data, and where clause are required"));
                        return;
                    }

                    // Build secure parameterized query
                    const parameterizedQuery = this._sqlSecurity.buildUpdateQuery(updateOptions);

                    // Log the query for audit
                    this._securityUtils.auditLog("SQL_QUERY", {
                        operation: "UPDATE",
                        table: updateOptions.table,
                        queryHash: parameterizedQuery.hash
                    });

                    // Execute through secure channel
                    this._executeSecureQuery(parameterizedQuery)
                        .then(resolve)
                        .catch(reject);

                } catch (error) {
                    Log.error("Secure UPDATE failed", error);
                    reject(error);
                }
            });
        },

        /**
         * Execute a DELETE query safely
         * @param {object} deleteOptions - Delete options
         * @returns {Promise} Delete result
         */
        executeDelete(deleteOptions) {
            return new Promise((resolve, reject) => {
                try {
                    // Validate user permissions
                    if (!this._securityUtils.hasRole("SQLAdmin")) {
                        this._securityUtils.auditLog("SQL_ACCESS_DENIED", { operation: "DELETE" });
                        reject(new Error("Access denied: SQL delete requires SQLAdmin role"));
                        return;
                    }

                    // Validate input
                    if (!deleteOptions || !deleteOptions.table || !deleteOptions.where) {
                        reject(new Error("Table name and where clause are required"));
                        return;
                    }

                    // Require confirmation for delete operations
                    MessageBox.confirm("Are you sure you want to delete records?", {
                        title: "Confirm Delete",
                        onClose: (action) => {
                            if (action === MessageBox.Action.OK) {
                                // Build secure parameterized query
                                const parameterizedQuery = this._sqlSecurity.buildDeleteQuery(deleteOptions);

                                // Log the query for audit
                                this._securityUtils.auditLog("SQL_QUERY", {
                                    operation: "DELETE",
                                    table: deleteOptions.table,
                                    queryHash: parameterizedQuery.hash
                                });

                                // Execute through secure channel
                                this._executeSecureQuery(parameterizedQuery)
                                    .then(resolve)
                                    .catch(reject);
                            } else {
                                reject(new Error("Delete operation cancelled"));
                            }
                        }
                    });

                } catch (error) {
                    Log.error("Secure DELETE failed", error);
                    reject(error);
                }
            });
        },

        /**
         * Translate natural language to SQL (secure)
         * @param {string} naturalLanguage - Natural language query
         * @returns {Promise} SQL query options
         */
        translateToSQL(naturalLanguage) {
            return new Promise((resolve, reject) => {
                try {
                    // Validate user permissions
                    if (!this._securityUtils.hasRole("SQLUser") && !this._securityUtils.hasRole("Admin")) {
                        this._securityUtils.auditLog("SQL_ACCESS_DENIED", { operation: "TRANSLATE" });
                        reject(new Error("Access denied: SQL translation requires SQLUser role"));
                        return;
                    }

                    // Sanitize input
                    const sanitized = this._securityUtils.escapeHTML(naturalLanguage);

                    // Parse natural language (simplified)
                    const parsed = this._parseNaturalLanguage(sanitized);

                    if (!parsed.intent) {
                        reject(new Error("Could not understand the query"));
                        return;
                    }

                    // Build query options based on intent
                    const queryOptions = this._buildQueryOptionsFromParsed(parsed);

                    // Log the translation for audit
                    this._securityUtils.auditLog("SQL_TRANSLATE", {
                        input: sanitized.substring(0, 100), // Log first 100 chars only
                        intent: parsed.intent
                    });

                    resolve(queryOptions);

                } catch (error) {
                    Log.error("Natural language translation failed", error);
                    reject(error);
                }
            });
        },

        /**
         * Execute a secure parameterized query
         * @private
         */
        _executeSecureQuery(parameterizedQuery) {
            // In real implementation, this would call the backend
            // For now, we'll use the mock from SQLSecurityModule
            return this._sqlSecurity.executeQuery(parameterizedQuery);
        },

        /**
         * Parse natural language query
         * @private
         */
        _parseNaturalLanguage(text) {
            const lower = text.toLowerCase();
            let intent = null;
            const entities = [];

            // Simple intent detection
            if (lower.includes("show") || lower.includes("get") || lower.includes("list")) {
                intent = "SELECT";
            } else if (lower.includes("add") || lower.includes("insert") || lower.includes("create")) {
                intent = "INSERT";
            } else if (lower.includes("update") || lower.includes("change") || lower.includes("modify")) {
                intent = "UPDATE";
            } else if (lower.includes("delete") || lower.includes("remove")) {
                intent = "DELETE";
            } else if (lower.includes("count") || lower.includes("how many")) {
                intent = "COUNT";
            }

            // Extract table names (simplified)
            const tableMatches = text.match(/\b(users?|orders?|products?|customers?|employees?)\b/gi);
            if (tableMatches) {
                tableMatches.forEach(table => {
                    entities.push({ type: "TABLE", value: table.toLowerCase() });
                });
            }

            return { intent, entities };
        },

        /**
         * Build query options from parsed natural language
         * @private
         */
        _buildQueryOptionsFromParsed(parsed) {
            const table = parsed.entities.find(e => e.type === "TABLE");
            if (!table) {
                throw new Error("No table identified in query");
            }

            const options = {
                table: table.value
            };

            // Add intent-specific options
            switch (parsed.intent) {
            case "SELECT":
            case "COUNT":
                // For SELECT/COUNT, we can add columns and conditions later
                break;
            case "INSERT":
                options.data = {}; // User needs to provide data
                break;
            case "UPDATE":
                options.data = {}; // User needs to provide data
                options.where = {}; // User needs to provide conditions
                break;
            case "DELETE":
                options.where = {}; // User needs to provide conditions
                break;
            }

            return options;
        }
    });
});