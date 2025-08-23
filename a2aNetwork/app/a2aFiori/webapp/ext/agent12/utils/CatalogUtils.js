sap.ui.define([
    "sap/m/MessageToast",
    "sap/base/Log",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], function (MessageToast, Log, SecurityUtils) {
    "use strict";

    return {
        /**
         * Validates catalog entry data
         * @param {object} entryData - The catalog entry data to validate
         * @returns {object} Validation result with isValid flag and errors array
         */
        validateCatalogEntry: function (entryData) {
            // Use SecurityUtils for comprehensive validation
            const securityValidation = SecurityUtils.validateCatalogEntry(entryData);
            
            const errors = [...securityValidation.errors];
            const warnings = [...securityValidation.warnings];
            
            // Additional business logic validations
            if (!entryData.entryName || entryData.entryName.trim() === "") {
                errors.push("Entry name is required");
            }
            
            if (!entryData.category) {
                errors.push("Category is required");
            }
            
            if (!entryData.entryType) {
                errors.push("Entry type is required");
            }
            
            // Format validations
            if (entryData.contactEmail && !this._isValidEmail(entryData.contactEmail)) {
                errors.push("Contact email format is invalid");
            }
            
            if (entryData.documentationUrl && !this._isValidUrl(entryData.documentationUrl)) {
                warnings.push("Documentation URL format may be invalid");
            }
            
            if (entryData.apiEndpoint && !this._isValidUrl(entryData.apiEndpoint)) {
                warnings.push("API endpoint URL format may be invalid");
            }
            
            // Business logic validations
            if (entryData.rating && (entryData.rating < 0 || entryData.rating > 5)) {
                errors.push("Rating must be between 0 and 5");
            }
            
            if (entryData.version && !this._isValidVersion(entryData.version)) {
                warnings.push("Version format should follow semantic versioning (e.g., 1.0.0)");
            }
            
            // Security validations
            if (entryData.tags && this._containsScriptTags(entryData.tags)) {
                errors.push("Tags contain potentially unsafe content");
            }
            
            if (entryData.description && this._containsScriptTags(entryData.description)) {
                errors.push("Description contains potentially unsafe content");
            }
            
            return {
                isValid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },

        /**
         * Validates search query input
         * @param {string} query - The search query
         * @returns {object} Validation result
         */
        validateSearchQuery: function (query) {
            // Use SecurityUtils for secure search query validation
            const sanitizedQuery = SecurityUtils.sanitizeSearchQuery(query);
            
            const errors = [];
            const warnings = [];
            
            if (!query || query.trim() === "") {
                errors.push("Search query cannot be empty");
            }
            
            if (query && query.length > 500) {
                errors.push("Search query is too long (maximum 500 characters)");
            }
            
            // Performance warnings
            if (query && query.includes('*') && query.length < 3) {
                warnings.push("Short wildcard searches may impact performance");
            }
            
            return {
                isValid: errors.length === 0,
                errors: errors,
                warnings: warnings,
                sanitized: sanitizedQuery
            };
        },

        /**
         * Formats catalog entry for display
         * @param {object} entry - The catalog entry
         * @returns {object} Formatted entry
         */
        formatCatalogEntry: function (entry) {
            if (!entry) return null;
            
            return {
                ...entry,
                displayName: this._truncateString(entry.entryName, 50),
                shortDescription: this._truncateString(entry.description, 100),
                formattedRating: entry.rating ? entry.rating.toFixed(1) : "Not rated",
                formattedUsage: this._formatNumber(entry.usageCount),
                formattedDate: this._formatDate(entry.modifiedAt),
                statusBadge: this._getStatusBadge(entry.status),
                categoryIcon: this._getCategoryIcon(entry.category),
                tagsArray: entry.tags ? entry.tags.split(',').map(tag => tag.trim()) : []
            };
        },

        /**
         * Generates search filters based on entry data
         * @param {array} entries - Array of catalog entries
         * @returns {object} Available filter options
         */
        generateSearchFilters: function (entries) {
            if (!entries || !Array.isArray(entries)) return {};
            
            const categories = new Set();
            const providers = new Set();
            const statuses = new Set();
            const tags = new Set();
            const entryTypes = new Set();
            
            entries.forEach(entry => {
                if (entry.category) categories.add(entry.category);
                if (entry.provider) providers.add(entry.provider);
                if (entry.status) statuses.add(entry.status);
                if (entry.entryType) entryTypes.add(entry.entryType);
                
                if (entry.tags) {
                    entry.tags.split(',').forEach(tag => {
                        const cleanTag = tag.trim();
                        if (cleanTag) tags.add(cleanTag);
                    });
                }
            });
            
            return {
                categories: Array.from(categories).sort(),
                providers: Array.from(providers).sort(),
                statuses: Array.from(statuses).sort(),
                tags: Array.from(tags).sort(),
                entryTypes: Array.from(entryTypes).sort()
            };
        },

        /**
         * Calculates search relevance score
         * @param {object} entry - Catalog entry
         * @param {string} query - Search query
         * @returns {number} Relevance score (0-100)
         */
        calculateRelevanceScore: function (entry, query) {
            if (!entry || !query) return 0;
            
            const lowerQuery = query.toLowerCase();
            let score = 0;
            
            // Exact name match gets highest score
            if (entry.entryName && entry.entryName.toLowerCase() === lowerQuery) {
                score += 100;
            } else if (entry.entryName && entry.entryName.toLowerCase().includes(lowerQuery)) {
                score += 50;
            }
            
            // Description match
            if (entry.description && entry.description.toLowerCase().includes(lowerQuery)) {
                score += 30;
            }
            
            // Tags match
            if (entry.tags) {
                const tags = entry.tags.toLowerCase().split(',');
                tags.forEach(tag => {
                    if (tag.trim().includes(lowerQuery)) {
                        score += 20;
                    }
                });
            }
            
            // Keywords match
            if (entry.keywords && entry.keywords.toLowerCase().includes(lowerQuery)) {
                score += 15;
            }
            
            // Provider match
            if (entry.provider && entry.provider.toLowerCase().includes(lowerQuery)) {
                score += 10;
            }
            
            // Boost score based on entry popularity
            if (entry.rating) {
                score += entry.rating * 5;
            }
            
            if (entry.usageCount) {
                score += Math.min(entry.usageCount / 100, 20);
            }
            
            return Math.min(score, 100);
        },

        /**
         * Validates metadata property
         * @param {object} property - Metadata property
         * @returns {object} Validation result
         */
        validateMetadataProperty: function (property) {
            // Use SecurityUtils for secure metadata validation
            const metadataValidation = SecurityUtils.validateMetadata(property);
            
            const errors = [...(metadataValidation.isValid ? [] : ['Invalid metadata format'])];
            const warnings = [];
            
            if (!property.metadataKey || property.metadataKey.trim() === "") {
                errors.push("Property key is required");
            }
            
            if (!property.metadataValue || property.metadataValue.trim() === "") {
                errors.push("Property value is required");
            }
            
            // Type-specific validation
            if (property.valueType && property.metadataValue) {
                switch (property.valueType) {
                    case 'NUMBER':
                        if (isNaN(property.metadataValue)) {
                            errors.push("Value must be a valid number");
                        }
                        break;
                    case 'BOOLEAN':
                        if (property.metadataValue !== 'true' && property.metadataValue !== 'false') {
                            errors.push("Value must be true or false");
                        }
                        break;
                    case 'EMAIL':
                        if (!this._isValidEmail(property.metadataValue)) {
                            errors.push("Value must be a valid email address");
                        }
                        break;
                    case 'URL':
                        if (!this._isValidUrl(property.metadataValue)) {
                            errors.push("Value must be a valid URL");
                        }
                        break;
                    case 'JSON':
                        try {
                            JSON.parse(property.metadataValue);
                        } catch (e) {
                            errors.push("Value must be valid JSON");
                        }
                        break;
                }
            }
            
            // Enhanced security validation using SecurityUtils
            if (property.metadataValue) {
                const sanitizedValue = SecurityUtils.sanitizeCatalogData(property.metadataValue);
                if (sanitizedValue !== property.metadataValue) {
                    errors.push("Property value contains potentially unsafe content");
                }
            }
            
            return {
                isValid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },

        /**
         * Generates metadata schema from properties
         * @param {array} properties - Array of metadata properties
         * @returns {object} JSON Schema
         */
        generateMetadataSchema: function (properties) {
            if (!properties || !Array.isArray(properties)) return {};
            
            const schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "title": "Catalog Entry Metadata",
                "properties": {},
                "required": []
            };
            
            properties.forEach(prop => {
                if (!prop.metadataKey) return;
                
                const propertySchema = {
                    "title": prop.metadataKey,
                    "description": prop.description || ""
                };
                
                // Map value types to JSON Schema types
                switch (prop.valueType) {
                    case 'STRING':
                    case 'EMAIL':
                    case 'URL':
                        propertySchema.type = "string";
                        if (prop.valueType === 'EMAIL') {
                            propertySchema.format = "email";
                        } else if (prop.valueType === 'URL') {
                            propertySchema.format = "uri";
                        }
                        break;
                    case 'NUMBER':
                        propertySchema.type = "number";
                        break;
                    case 'BOOLEAN':
                        propertySchema.type = "boolean";
                        break;
                    case 'DATE':
                        propertySchema.type = "string";
                        propertySchema.format = "date";
                        break;
                    case 'JSON':
                        propertySchema.type = "object";
                        break;
                    default:
                        propertySchema.type = "string";
                }
                
                schema.properties[prop.metadataKey] = propertySchema;
                
                // Add to required if marked as such
                if (prop.required) {
                    schema.required.push(prop.metadataKey);
                }
            });
            
            return schema;
        },

        /**
         * Exports catalog data to various formats
         * @param {array} entries - Catalog entries
         * @param {string} format - Export format (json, csv, xml)
         * @returns {string} Exported data
         */
        exportCatalogData: function (entries, format) {
            if (!entries || !Array.isArray(entries)) return "";
            
            // Sanitize entries before export
            const sanitizedEntries = entries.map(entry => {
                const validation = SecurityUtils.validateCatalogEntry(entry);
                return validation.sanitized || entry;
            });
            
            switch (format.toLowerCase()) {
                case 'json':
                    return JSON.stringify(sanitizedEntries, null, 2);
                case 'csv':
                    return this._convertToCSV(sanitizedEntries);
                case 'xml':
                    return this._convertToXML(sanitizedEntries);
                default:
                    return JSON.stringify(sanitizedEntries, null, 2);
            }
        },

        /**
         * Imports catalog data from various formats
         * @param {string} data - Import data
         * @param {string} format - Import format
         * @returns {array} Parsed entries
         */
        importCatalogData: function (data, format) {
            if (!data) return [];
            
            try {
                let entries = [];
                switch (format.toLowerCase()) {
                    case 'json':
                        const parsed = JSON.parse(data);
                        entries = Array.isArray(parsed) ? parsed : [parsed];
                        break;
                    case 'csv':
                        entries = this._parseCSV(data);
                        break;
                    case 'xml':
                        entries = this._parseXML(data);
                        break;
                    default:
                        entries = JSON.parse(data);
                }
                
                // Validate and sanitize imported entries
                return entries.map(entry => {
                    const validation = SecurityUtils.validateCatalogEntry(entry);
                    if (!validation.isValid) {
                        Log.warning("Invalid catalog entry during import:", validation.errors);
                        return null;
                    }
                    return validation.sanitized || entry;
                }).filter(entry => entry !== null);
                
            } catch (error) {
                Log.error("Failed to import catalog data:", error);
                return [];
            }
        },

        /**
         * Generates catalog statistics
         * @param {array} entries - Catalog entries
         * @returns {object} Statistics
         */
        generateCatalogStatistics: function (entries) {
            if (!entries || !Array.isArray(entries)) return {};
            
            const stats = {
                totalEntries: entries.length,
                byCategory: {},
                byStatus: {},
                byProvider: {},
                averageRating: 0,
                totalUsage: 0,
                recentEntries: 0
            };
            
            let totalRating = 0;
            let ratedEntries = 0;
            const oneMonthAgo = new Date();
            oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
            
            entries.forEach(entry => {
                // Category stats
                stats.byCategory[entry.category] = (stats.byCategory[entry.category] || 0) + 1;
                
                // Status stats
                stats.byStatus[entry.status] = (stats.byStatus[entry.status] || 0) + 1;
                
                // Provider stats
                if (entry.provider) {
                    stats.byProvider[entry.provider] = (stats.byProvider[entry.provider] || 0) + 1;
                }
                
                // Rating calculation
                if (entry.rating) {
                    totalRating += entry.rating;
                    ratedEntries++;
                }
                
                // Usage stats
                stats.totalUsage += entry.usageCount || 0;
                
                // Recent entries
                if (entry.createdAt && new Date(entry.createdAt) > oneMonthAgo) {
                    stats.recentEntries++;
                }
            });
            
            stats.averageRating = ratedEntries > 0 ? (totalRating / ratedEntries).toFixed(1) : 0;
            
            return stats;
        },

        // Private utility methods
        _isValidEmail: function (email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(email);
        },

        _isValidUrl: function (url) {
            try {
                new URL(url);
                return true;
            } catch (e) {
                return false;
            }
        },

        _isValidVersion: function (version) {
            const versionRegex = /^\d+\.\d+\.\d+(-\w+)?$/;
            return versionRegex.test(version);
        },

        _containsScriptTags: function (str) {
            // Use SecurityUtils for consistent security checks
            if (!str) return false;
            const sanitized = SecurityUtils.sanitizeCatalogData(str);
            return sanitized !== str;
        },

        _sanitizeString: function (str) {
            // Delegate to SecurityUtils for consistent sanitization
            return SecurityUtils.sanitizeCatalogData(str);
        },

        _truncateString: function (str, maxLength) {
            if (!str) return "";
            return str.length > maxLength ? str.substring(0, maxLength) + "..." : str;
        },

        _formatNumber: function (num) {
            if (!num) return "0";
            return num.toLocaleString();
        },

        _formatDate: function (dateStr) {
            if (!dateStr) return "";
            const date = new Date(dateStr);
            return date.toLocaleDateString();
        },

        _getStatusBadge: function (status) {
            const badges = {
                'PUBLISHED': { text: 'Published', state: 'Success' },
                'DRAFT': { text: 'Draft', state: 'Warning' },
                'DEPRECATED': { text: 'Deprecated', state: 'Error' },
                'ARCHIVED': { text: 'Archived', state: 'Information' }
            };
            return badges[status] || { text: status, state: 'None' };
        },

        _getCategoryIcon: function (category) {
            const icons = {
                'SERVICE': 'sap-icon://connected',
                'API': 'sap-icon://syntax',
                'DATABASE': 'sap-icon://database',
                'WORKFLOW': 'sap-icon://workflow-tasks',
                'AGENT': 'sap-icon://person-placeholder',
                'RESOURCE': 'sap-icon://documents',
                'TEMPLATE': 'sap-icon://template',
                'CONNECTOR': 'sap-icon://chain-link'
            };
            return icons[category] || 'sap-icon://document';
        },

        _convertToCSV: function (entries) {
            if (!entries.length) return "";
            
            const headers = Object.keys(entries[0]);
            const csvContent = [
                headers.join(','),
                ...entries.map(entry => 
                    headers.map(header => {
                        const value = entry[header] || '';
                        return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
                    }).join(',')
                )
            ].join('\n');
            
            return csvContent;
        },

        _convertToXML: function (entries) {
            let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<catalog>\n';
            
            entries.forEach(entry => {
                xml += '  <entry>\n';
                Object.keys(entry).forEach(key => {
                    const value = entry[key] || '';
                    xml += `    <${key}>${this._escapeXml(value.toString())}</${key}>\n`;
                });
                xml += '  </entry>\n';
            });
            
            xml += '</catalog>';
            return xml;
        },

        _parseCSV: function (csvData) {
            const lines = csvData.split('\n');
            if (lines.length < 2) return [];
            
            const headers = lines[0].split(',');
            const entries = [];
            
            for (let i = 1; i < lines.length; i++) {
                if (lines[i].trim() === '') continue;
                
                const values = lines[i].split(',');
                const entry = {};
                
                headers.forEach((header, index) => {
                    entry[header.trim()] = values[index]?.trim().replace(/^"|"$/g, '') || '';
                });
                
                entries.push(entry);
            }
            
            return entries;
        },

        _parseXML: function (xmlData) {
            // Basic XML parsing - in production, use a proper XML parser
            const entries = [];
            const entryRegex = /<entry>([\s\S]*?)<\/entry>/g;
            let entryMatch;
            
            while ((entryMatch = entryRegex.exec(xmlData)) !== null) {
                const entryContent = entryMatch[1];
                const entry = {};
                
                const fieldRegex = /<(\w+)>([\s\S]*?)<\/\1>/g;
                let fieldMatch;
                
                while ((fieldMatch = fieldRegex.exec(entryContent)) !== null) {
                    entry[fieldMatch[1]] = fieldMatch[2];
                }
                
                entries.push(entry);
            }
            
            return entries;
        },

        _escapeXml: function (str) {
            return str.replace(/[<>&'"]/g, function (c) {
                switch (c) {
                    case '<': return '&lt;';
                    case '>': return '&gt;';
                    case '&': return '&amp;';
                    case "'": return '&apos;';
                    case '"': return '&quot;';
                    default: return c;
                }
            });
        }
    };
});