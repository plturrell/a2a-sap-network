sap.ui.define([
    "sap/m/MessageToast",
    "sap/base/Log",
    "a2a/network/agent10/ext/utils/SecurityUtils"
], (MessageToast, Log, SecurityUtils) => {
    "use strict";

    return {
        /**
         * Validates a mathematical formula
         * @param {string} formula - The formula to validate
         * @returns {object} Validation result with isValid flag and errors array
         */
        validateFormula(formula) {
            // Use SecurityUtils for comprehensive validation
            const securityValidation = SecurityUtils.validateFormula(formula);

            if (!securityValidation.isValid) {
                return {
                    isValid: false,
                    errors: securityValidation.errors,
                    sanitized: securityValidation.sanitized
                };
            }

            // Additional mathematical validation
            const errors = [];
            const sanitized = securityValidation.sanitized;

            // Check for valid operators
            const invalidOpSequence = /[\+\-\*\/\^\%]{2,}/;
            if (invalidOpSequence.test(sanitized)) {
                errors.push("Invalid operator sequence");
            }

            // Check for valid function names
            const functionPattern = /\b([a-zA-Z_]\w*)\s*\(/g;
            const validFunctions = [
                "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                "exp", "log", "log10", "log2", "sqrt", "cbrt", "abs", "ceil", "floor",
                "round", "max", "min", "pow", "sum", "avg", "mean", "median", "mode",
                "std", "var", "corr", "cov"
            ];

            let match;
            while ((match = functionPattern.exec(sanitized)) !== null) {
                if (!validFunctions.includes(match[1])) {
                    errors.push(`Unknown function: ${match[1]}`);
                }
            }

            return {
                isValid: errors.length === 0,
                errors,
                warnings: this._getFormulaWarnings(sanitized),
                sanitized
            };
        },

        /**
         * Get warnings for a formula
         * @private
         */
        _getFormulaWarnings(formula) {
            const warnings = [];

            // Check for potential precision issues
            if (formula.includes("/") && !formula.includes("round")) {
                warnings.push("Division operations may result in precision loss");
            }

            // Check for potential overflow
            if (formula.includes("**") || formula.includes("^")) {
                warnings.push("Exponentiation may cause overflow for large values");
            }

            return warnings;
        },

        /**
         * Format calculation result for display
         * @param {any} result - The raw calculation result
         * @param {string} precision - The precision type
         * @returns {string} Formatted result
         */
        formatResult(result, precision) {
            if (result === null || result === undefined) {
                return "N/A";
            }

            if (typeof result === "number") {
                // Validate precision before formatting
                const precisionValidation = SecurityUtils.validatePrecision(result, precision);
                if (!precisionValidation.isValid) {
                    Log.warning("Precision validation failed", precisionValidation.error);
                }

                if (isNaN(result)) {
                    return "NaN";
                }
                if (!isFinite(result)) {
                    return result > 0 ? "∞" : "-∞";
                }

                // Format based on precision
                switch (precision) {
                case "DECIMAL32":
                    return result.toFixed(7);
                case "DECIMAL64":
                    return result.toFixed(15);
                case "DECIMAL128":
                    return result.toFixed(34);
                default:
                    return result.toString();
                }
            }

            if (Array.isArray(result)) {
                return `[${result.map(r => this.formatResult(r, precision)).join(", ")}]`;
            }

            if (typeof result === "object") {
                return SecurityUtils.sanitizeResult(result);
            }

            return SecurityUtils.sanitizeResult(result);
        },

        /**
         * Parse data input for statistical analysis
         * @param {string} input - The input data string
         * @param {string} format - The format type (csv, space, newline)
         * @returns {array} Parsed data array
         */
        parseDataInput(input, format) {
            if (!input || input.trim() === "") {
                return [];
            }

            // Sanitize input first
            const sanitizedInput = SecurityUtils.escapeHTML(input.trim());
            let data = [];

            try {
                // Try to parse as JSON first (with validation)
                data = JSON.parse(sanitizedInput);
                if (Array.isArray(data)) {
                    // Validate each data point
                    return data.map(item => {
                        if (typeof item === "number") {
                            return Number.isFinite(item) ? item : 0;
                        }
                        return SecurityUtils.escapeHTML(String(item));
                    });
                }
            } catch (e) {
                // Not JSON, continue with other formats
            }

            // Parse based on format
            const delimiter = format === "csv" ? "," : format === "space" ? /\s+/ : "\n";
            const values = sanitizedInput.split(delimiter);

            data = values
                .map(v => v.trim())
                .filter(v => v !== "")
                .map(v => {
                    const num = parseFloat(v);
                    if (!isNaN(num) && Number.isFinite(num)) {
                        return num;
                    }
                    return SecurityUtils.escapeHTML(v);
                });

            return data;
        },

        /**
         * Generate visualization data from calculation results
         * @param {object} results - The calculation results
         * @returns {array} Visualization data
         */
        generateVisualizationData(results) {
            if (!results || !results.data) {
                return [];
            }

            // For simple arrays, create index-based visualization
            if (Array.isArray(results.data)) {
                return results.data.map((value, index) => ({
                    category: `Item ${index + 1}`,
                    value
                }));
            }

            // For statistical results, create category-based visualization
            if (results.statistics) {
                return Object.keys(results.statistics).map(key => ({
                    category: this._formatStatKey(key),
                    value: results.statistics[key]
                }));
            }

            return [];
        },

        /**
         * Format statistical key for display
         * @private
         */
        _formatStatKey(key) {
            const keyMap = {
                "mean": "Mean",
                "median": "Median",
                "mode": "Mode",
                "standardDeviation": "Std Dev",
                "variance": "Variance",
                "min": "Minimum",
                "max": "Maximum"
            };

            return keyMap[key] || key;
        },

        /**
         * Get calculation method recommendations based on formula
         * @param {string} formula - The formula expression
         * @returns {object} Recommended methods and reasons
         */
        getMethodRecommendations(formula) {
            const recommendations = [];

            if (formula.includes("matrix") || formula.includes("[]")) {
                recommendations.push({
                    method: "NEURAL_NETWORK",
                    reason: "Optimized for matrix operations"
                });
            }

            if (formula.includes("sum") || formula.includes("product")) {
                recommendations.push({
                    method: "ITERATIVE",
                    reason: "Efficient for iterative calculations"
                });
            }

            if (formula.includes("optimize") || formula.includes("min") || formula.includes("max")) {
                recommendations.push({
                    method: "GENETIC_ALGORITHM",
                    reason: "Suitable for optimization problems"
                });
            }

            if (formula.includes("random") || formula.includes("simulate")) {
                recommendations.push({
                    method: "MONTE_CARLO",
                    reason: "Best for probabilistic simulations"
                });
            }

            if (recommendations.length === 0) {
                recommendations.push({
                    method: "DIRECT",
                    reason: "Standard calculation method"
                });
            }

            return recommendations;
        },

        /**
         * Estimate calculation complexity
         * @param {string} formula - The formula expression
         * @returns {object} Complexity metrics
         */
        estimateComplexity(formula) {
            const complexity = {
                score: 0,
                level: "Simple",
                factors: []
            };

            // Count operations
            const operations = (formula.match(/[\+\-\*\/\^\%]/g) || []).length;
            complexity.score += operations;
            if (operations > 5) {
                complexity.factors.push(`${operations} operations`);
            }

            // Count functions
            const functions = (formula.match(/\b[a-zA-Z_]\w*\s*\(/g) || []).length;
            complexity.score += functions * 2;
            if (functions > 0) {
                complexity.factors.push(`${functions} functions`);
            }

            // Count nesting depth
            let maxDepth = 0, currentDepth = 0;
            for (const char of formula) {
                if (char === "(") {
                    currentDepth++;
                    maxDepth = Math.max(maxDepth, currentDepth);
                } else if (char === ")") {
                    currentDepth--;
                }
            }
            complexity.score += maxDepth * 3;
            if (maxDepth > 2) {
                complexity.factors.push(`Nesting depth: ${maxDepth}`);
            }

            // Determine complexity level
            if (complexity.score < 5) {
                complexity.level = "Simple";
            } else if (complexity.score < 15) {
                complexity.level = "Moderate";
            } else if (complexity.score < 30) {
                complexity.level = "Complex";
            } else {
                complexity.level = "Very Complex";
            }

            return complexity;
        },

        /**
         * Export calculation results in various formats
         * @param {object} results - The calculation results
         * @param {string} format - Export format (json, csv, excel)
         * @returns {object} Export data and metadata
         */
        exportResults(results, format) {
            // Sanitize results before export
            const sanitizedResults = SecurityUtils.sanitizeResult(results);

            switch (format) {
            case "json":
                return {
                    data: JSON.stringify(sanitizedResults, null, 2),
                    mimeType: "application/json",
                    extension: "json"
                };

            case "csv":
                return {
                    data: this._convertToCSV(sanitizedResults),
                    mimeType: "text/csv",
                    extension: "csv"
                };

            case "excel":
                return {
                    data: this._convertToExcel(sanitizedResults),
                    mimeType: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    extension: "xlsx"
                };

            default:
                return {
                    data: SecurityUtils.escapeHTML(sanitizedResults.toString()),
                    mimeType: "text/plain",
                    extension: "txt"
                };
            }
        },

        /**
         * Convert results to CSV format
         * @private
         */
        _convertToCSV(results) {
            const rows = [];

            // Headers
            rows.push(["Property", "Value"].join(","));

            // Basic properties
            rows.push(["Task Name", results.taskName || ""].join(","));
            rows.push(["Formula", `"${results.formula || ""}"`].join(","));
            rows.push(["Result", results.result || ""].join(","));
            rows.push(["Execution Time", results.executionTime || ""].join(","));
            rows.push(["Accuracy", results.accuracy || ""].join(","));

            // Steps if available
            if (results.steps && results.steps.length > 0) {
                rows.push(["", ""]); // Empty row
                rows.push(["Step", "Operation", "Input", "Output", "Duration"].join(","));

                results.steps.forEach((step, index) => {
                    rows.push([
                        index + 1,
                        step.operation || "",
                        `"${step.input || ""}"`,
                        `"${step.output || ""}"`,
                        step.duration || ""
                    ].join(","));
                });
            }

            return rows.join("\n");
        },

        /**
         * Convert results to Excel format (simplified)
         * @private
         */
        _convertToExcel(results) {
            // This is a simplified version - in production, use a library like SheetJS
            Log.warning("Excel export requires additional library implementation");
            return this._convertToCSV(results);
        }
    };
});