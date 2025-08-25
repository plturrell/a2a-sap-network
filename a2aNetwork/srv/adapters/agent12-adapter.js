/**
 * Agent 12 Adapter - Catalog Manager
 * Converts between REST API and OData formats for catalog management,
 * service discovery, registry management, and metadata operations
 */

const fetch = require('node-fetch');
// const { v4: uuidv4 } = require('uuid');

class Agent12Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT12_BASE_URL || 'http://localhost:8012';
        this.apiVersion = 'v1';
        this.timeout = 30000; // 30 second timeout for catalog operations
    }

    // ===== CATALOG ENTRY OPERATIONS =====
    async getCatalogEntries(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CatalogEntry');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCatalogEntry(data) {
        try {
            const restData = this._convertODataCatalogEntryToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTCatalogEntryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateCatalogEntry(id, data) {
        try {
            const restData = this._convertODataCatalogEntryToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTCatalogEntryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteCatalogEntry(id) {
        try {
            await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CATALOG ENTRY ACTIONS =====
    async publishEntry(entryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/publish`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                message: data.message,
                entryName: data.entry_name,
                version: data.version
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deprecateEntry(entryId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/deprecate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                reason: data.reason
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success,
                entryName: data.entry_name,
                replacementEntry: data.replacement_entry
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateEntryMetadata(entryId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/metadata`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                metadata: data.metadata
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async archiveEntry(entryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/archive`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async generateDocumentation(entryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/generate-docs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout * 2 // Longer timeout for documentation generation

            });
            const data = await response.json();

            return {
                documentationUrl: data.documentation_url,
                content: data.content
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateEntry(entryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                isValid: data.is_valid,
                errors: data.errors,
                warnings: data.warnings
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async duplicateEntry(entryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/duplicate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return this._convertRESTCatalogEntryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportEntry(entryId, options) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-entries/${entryId}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                format: options.format
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                downloadUrl: data.download_url,
                content: data.content,
                fileName: data.file_name
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== DEPENDENCY OPERATIONS =====
    async getCatalogDependencies(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-dependencies?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CatalogDependency');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCatalogDependency(data) {
        try {
            const restData = this._convertODataDependencyToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-dependencies`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTDependencyToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteCatalogDependency(id) {
        try {
            await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-dependencies/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== REVIEW OPERATIONS =====
    async getCatalogReviews(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-reviews?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CatalogReview');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCatalogReview(data) {
        try {
            const restData = this._convertODataReviewToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-reviews`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTReviewToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async approveReview(reviewId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-reviews/${reviewId}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async rejectReview(reviewId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-reviews/${reviewId}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                reason: data.reason
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async flagReview(reviewId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-reviews/${reviewId}/flag`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                reason: data.reason
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== METADATA OPERATIONS =====
    async getCatalogMetadata(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-metadata?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CatalogMetadata');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCatalogMetadata(data) {
        try {
            const restData = this._convertODataMetadataToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-metadata`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTMetadataToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateCatalogMetadata(id, data) {
        try {
            const restData = this._convertODataMetadataToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-metadata/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTMetadataToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteCatalogMetadata(id) {
        try {
            await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-metadata/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            return true;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== SEARCH OPERATIONS =====
    async getCatalogSearches(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-searches?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'CatalogSearch');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCatalogSearch(data) {
        try {
            const restData = this._convertODataSearchToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/catalog-searches`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTSearchToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== REGISTRY OPERATIONS =====
    async getRegistries(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries?${new URLSearchParams(params)}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTToOData(data, 'Registry');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getRegistry(id) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${id}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });

            return this._convertRESTRegistryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createRegistry(data) {
        try {
            const restData = this._convertODataRegistryToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTRegistryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateRegistry(id, data) {
        try {
            const restData = this._convertODataRegistryToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTRegistryToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async syncRegistry(registryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${registryId}/sync`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout * 3 // Longer timeout for sync operations

            });
            const data = await response.json();

            return {
                message: data.message,
                entriesProcessed: data.entries_processed,
                entriesAdded: data.entries_added,
                entriesUpdated: data.entries_updated,
                errors: data.errors,
                duration: data.duration
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async testRegistryConnection(registryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${registryId}/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message,
                responseTime: data.response_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resetRegistry(registryId) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${registryId}/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportRegistry(registryId, options) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${registryId}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                format: options.format
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                downloadUrl: data.download_url,
                content: data.content,
                fileName: data.file_name
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async importRegistry(registryId, data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/registries/${registryId}/import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                data: data.data
            }),
                timeout: this.timeout * 2

            });
            const data = await response.json();

            return {
                message: data.message,
                entriesImported: data.entries_imported
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== MAIN CATALOG OPERATIONS =====
    async searchCatalog(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                query: data.query,
                category: data.category,
                tags: data.tags,
                filters: data.filters,
                sort_by: data.sortBy,
                limit: data.limit
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                catalogEntries: data.catalog_entries?.map(entry =>
                    this._convertRESTCatalogEntryToOData(entry)) || [],
                totalCount: data.total_count,
                searchTime: data.search_time,
                facets: data.facets
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async discoverServices(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/discover-services`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                registry_type: data.registryType,
                filters: data.filters,
                auto_register: data.autoRegister
            }, {
                timeout: this.timeout * 2
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                message: data.message,
                services: data.services,
                discoveredCount: data.discovered_count
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async registerService(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/register-service`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                service_name: data.serviceName,
                service_url: data.serviceUrl,
                service_type: data.serviceType,
                metadata: data.metadata,
                health_check_url: data.healthCheckUrl
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                serviceId: data.service_id,
                message: data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateServiceHealth(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/update-service-health`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                service_id: data.serviceId,
                health_status: data.healthStatus,
                health_details: data.healthDetails
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                success: data.success,
                serviceName: data.service_name,
                previousHealth: data.previous_health
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async analyzeDependencies(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/analyze-dependencies`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                entry_id: data.entryId,
                depth: data.depth,
                include_indirect: data.includeIndirect
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                analysis: data.analysis,
                dependencies: data.dependencies,
                dependents: data.dependents
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateMetadata(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/validate-metadata`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                entry_id: data.entryId,
                schema_validation: data.schemaValidation,
                quality_checks: data.qualityChecks
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                isValid: data.is_valid,
                errors: data.errors,
                warnings: data.warnings,
                report: data.report
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async generateCatalogReport(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/generate-report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                format: data.format,
                include_stats: data.includeStats,
                include_reviews: data.includeReviews,
                date_range: data.dateRange
            }, {
                timeout: this.timeout * 2
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                reportUrl: data.report_url,
                content: data.content,
                fileName: data.file_name
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async bulkImport(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/bulk-import`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                import_format: data.importFormat,
                data: data.data,
                validate_before_import: data.validateBeforeImport,
                update_existing: data.updateExisting
            }, {
                timeout: this.timeout * 3
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                message: data.message,
                imported: data.imported,
                updated: data.updated,
                errors: data.errors
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async bulkExport(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/bulk-export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                export_format: data.exportFormat,
                categories: data.categories,
                filters: data.filters,
                include_metadata: data.includeMetadata
            }, {
                timeout: this.timeout * 2
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                downloadUrl: data.download_url,
                content: data.content,
                fileName: data.file_name
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async syncExternalCatalog(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/sync-external-catalog`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                catalog_url: data.catalogUrl,
                catalog_type: data.catalogType,
                sync_mode: data.syncMode,
                mapping_rules: data.mappingRules
            }, {
                timeout: this.timeout * 3
            }),
                timeout: this.timeout
            });
            const data = await response.json();

            return {
                message: data.message,
                synced: data.synced,
                errors: data.errors
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeSearchIndex() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/optimize-search-index`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout * 2

            });
            const data = await response.json();

            return {
                success: data.success,
                message: data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async generateRecommendations(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/recommendations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                user_id: data.userId,
                context: data.context,
                limit: data.limit
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                recommendations: data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async rebuildCatalog() {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/rebuild-catalog`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
                timeout: this.timeout * 5

            });
            const data = await response.json();

            return {
                message: data.message,
                entriesProcessed: data.entries_processed
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCategory(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/categories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                category_name: data.categoryName,
                description: data.description,
                parent_category: data.parentCategory,
                icon: data.icon
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                categoryId: data.category_id,
                message: data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async manageVersioning(data) {
        try {
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/versioning`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                entry_id: data.entryId,
                operation: data.operation,
                version: data.version,
                change_log: data.changeLog
            }),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return {
                message: data.message,
                newVersion: data.new_version
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // ===== CONVERSION UTILITIES =====
    _convertODataToREST(query) {
        const params = {};

        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$orderby) params.sort = query.$orderby;
        if (query.$search) params.search = query.$search;

        return params;
    }

    _parseODataFilter(filter) {
        return filter
            .replace(/eq/g, '=')
            .replace(/ne/g, '!=')
            .replace(/gt/g, '>')
            .replace(/lt/g, '<')
            .replace(/ge/g, '>=')
            .replace(/le/g, '<=')
            .replace(/and/g, '&&')
            .replace(/or/g, '||');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        switch (entityType) {
            case 'CatalogEntry':
                return this._convertRESTCatalogEntryToOData(item);
            case 'CatalogDependency':
                return this._convertRESTDependencyToOData(item);
            case 'CatalogReview':
                return this._convertRESTReviewToOData(item);
            case 'CatalogMetadata':
                return this._convertRESTMetadataToOData(item);
            case 'CatalogSearch':
                return this._convertRESTSearchToOData(item);
            case 'Registry':
                return this._convertRESTRegistryToOData(item);
            default:
                return item;
        }
    }

    _convertODataCatalogEntryToREST(data) {
        return {
            entry_name: data.entryName,
            description: data.description,
            category: data.category?.toLowerCase(),
            sub_category: data.subCategory,
            version: data.version,
            status: data.status?.toLowerCase() || 'draft',
            visibility: data.visibility?.toLowerCase() || 'private',
            entry_type: data.entryType?.toLowerCase(),
            provider: data.provider,
            owner: data.owner,
            contact_email: data.contactEmail,
            documentation_url: data.documentationUrl,
            source_url: data.sourceUrl,
            api_endpoint: data.apiEndpoint,
            health_check_url: data.healthCheckUrl,
            tags: data.tags,
            keywords: data.keywords,
            rating: data.rating,
            usage_count: data.usageCount,
            download_count: data.downloadCount,
            is_featured: data.isFeatured,
            is_verified: data.isVerified,
            last_accessed: data.lastAccessed,
            metadata: data.metadata,
            configuration_schema: data.configurationSchema,
            example_usage: data.exampleUsage,
            license: data.license,
            security_level: data.securityLevel?.toLowerCase() || 'internal'
        };
    }

    _convertRESTCatalogEntryToOData(item) {
        return {
            ID: item.id || uuidv4(),
            entryName: item.entry_name,
            description: item.description,
            category: item.category?.toUpperCase() || 'SERVICE',
            subCategory: item.sub_category,
            version: item.version,
            status: item.status?.toUpperCase() || 'DRAFT',
            visibility: item.visibility?.toUpperCase() || 'PRIVATE',
            entryType: item.entry_type?.toUpperCase() || 'MICROSERVICE',
            provider: item.provider,
            owner: item.owner,
            contactEmail: item.contact_email,
            documentationUrl: item.documentation_url,
            sourceUrl: item.source_url,
            apiEndpoint: item.api_endpoint,
            healthCheckUrl: item.health_check_url,
            tags: item.tags,
            keywords: item.keywords,
            rating: item.rating || 0.0,
            usageCount: item.usage_count || 0,
            downloadCount: item.download_count || 0,
            isFeatured: item.is_featured !== false,
            isVerified: item.is_verified !== false,
            lastAccessed: item.last_accessed,
            metadata: item.metadata,
            configurationSchema: item.configuration_schema,
            exampleUsage: item.example_usage,
            license: item.license,
            securityLevel: item.security_level?.toUpperCase() || 'INTERNAL',
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _convertODataDependencyToREST(data) {
        return {
            catalog_entry_id: data.catalogEntry_ID,
            dependent_entry_id: data.dependentEntry_ID,
            dependency_type: data.dependencyType?.toLowerCase(),
            version_range: data.versionRange,
            is_critical: data.isCritical,
            description: data.description
        };
    }

    _convertRESTDependencyToOData(item) {
        return {
            ID: item.id || uuidv4(),
            catalogEntry_ID: item.catalog_entry_id,
            dependentEntry_ID: item.dependent_entry_id,
            dependencyType: item.dependency_type?.toUpperCase() || 'REQUIRES',
            versionRange: item.version_range,
            isCritical: item.is_critical !== false,
            description: item.description,
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _convertODataReviewToREST(data) {
        return {
            catalog_entry_id: data.catalogEntry_ID,
            reviewer: data.reviewer,
            rating: data.rating,
            title: data.title,
            review_text: data.reviewText,
            pros: data.pros,
            cons: data.cons,
            recommended_use_case: data.recommendedUseCase,
            is_verified_review: data.isVerifiedReview,
            helpful_votes: data.helpfulVotes,
            review_status: data.reviewStatus?.toLowerCase() || 'pending'
        };
    }

    _convertRESTReviewToOData(item) {
        return {
            ID: item.id || uuidv4(),
            catalogEntry_ID: item.catalog_entry_id,
            reviewer: item.reviewer,
            rating: item.rating,
            title: item.title,
            reviewText: item.review_text,
            pros: item.pros,
            cons: item.cons,
            recommendedUseCase: item.recommended_use_case,
            isVerifiedReview: item.is_verified_review !== false,
            helpfulVotes: item.helpful_votes || 0,
            reviewStatus: item.review_status?.toUpperCase() || 'PENDING',
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _convertODataMetadataToREST(data) {
        return {
            catalog_entry_id: data.catalogEntry_ID,
            metadata_key: data.metadataKey,
            metadata_value: data.metadataValue,
            value_type: data.valueType?.toLowerCase() || 'string',
            is_searchable: data.isSearchable,
            display_order: data.displayOrder,
            category: data.category?.toLowerCase() || 'technical'
        };
    }

    _convertRESTMetadataToOData(item) {
        return {
            ID: item.id || uuidv4(),
            catalogEntry_ID: item.catalog_entry_id,
            metadataKey: item.metadata_key,
            metadataValue: item.metadata_value,
            valueType: item.value_type?.toUpperCase() || 'STRING',
            isSearchable: item.is_searchable !== false,
            displayOrder: item.display_order || 0,
            category: item.category?.toUpperCase() || 'TECHNICAL',
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _convertODataSearchToREST(data) {
        return {
            search_query: data.searchQuery,
            search_type: data.searchType?.toLowerCase(),
            filters_applied: data.filtersApplied,
            results_count: data.resultsCount,
            search_time: data.searchTime,
            user_agent: data.userAgent,
            ip_address: data.ipAddress,
            session_id: data.sessionId,
            search_results: data.searchResults,
            selected_result_id: data.selectedResult_ID
        };
    }

    _convertRESTSearchToOData(item) {
        return {
            ID: item.id || uuidv4(),
            searchQuery: item.search_query,
            searchType: item.search_type?.toUpperCase() || 'KEYWORD',
            filtersApplied: item.filters_applied,
            resultsCount: item.results_count || 0,
            searchTime: item.search_time,
            userAgent: item.user_agent,
            ipAddress: item.ip_address,
            sessionId: item.session_id,
            searchResults: item.search_results,
            selectedResult_ID: item.selected_result_id,
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _convertODataRegistryToREST(data) {
        return {
            registry_name: data.registryName,
            registry_type: data.registryType?.toLowerCase(),
            registry_url: data.registryUrl,
            status: data.status?.toLowerCase() || 'active',
            last_sync: data.lastSync,
            sync_frequency: data.syncFrequency?.toLowerCase() || 'daily',
            authentication_type: data.authenticationType?.toLowerCase() || 'none',
            configuration: data.configuration,
            health_check_url: data.healthCheckUrl,
            total_entries: data.totalEntries,
            active_entries: data.activeEntries,
            last_error: data.lastError,
            sync_statistics: data.syncStatistics
        };
    }

    _convertRESTRegistryToOData(item) {
        return {
            ID: item.id || uuidv4(),
            registryName: item.registry_name,
            registryType: item.registry_type?.toUpperCase() || 'SERVICE_REGISTRY',
            registryUrl: item.registry_url,
            status: item.status?.toUpperCase() || 'ACTIVE',
            lastSync: item.last_sync,
            syncFrequency: item.sync_frequency?.toUpperCase() || 'DAILY',
            authenticationType: item.authentication_type?.toUpperCase() || 'NONE',
            configuration: item.configuration,
            healthCheckUrl: item.health_check_url,
            totalEntries: item.total_entries || 0,
            activeEntries: item.active_entries || 0,
            lastError: item.last_error,
            syncStatistics: item.sync_statistics,
            createdAt: item.created_at || new Date().toISOString(),
            modifiedAt: item.modified_at || new Date().toISOString()
        };
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.data?.message || error.response.statusText;
            const details = error.data?.details || null;

            return new Error(`Agent 12 Error (${status}): ${message}${details ? ` - ${JSON.stringify(details)}` : ''}`);
        } else if (error.request) {
            return new Error('Agent 12 Connection Error: No response from catalog service');
        } else {
            return new Error(`Agent 12 Error: ${error.message}`);
        }
    }
}

module.exports = Agent12Adapter;