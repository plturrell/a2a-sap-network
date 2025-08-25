/**
 * Agent 0 Adapter - Data Product Agent
 * Converts between REST API and OData formats for data product management operations
 */

const fetch = require('node-fetch');
// const { BlockchainClient } = require('../core/blockchain-client');
// const { v4: uuidv4 } = require('uuid');

// TODO: Replace all blockchainClient.sendMessage calls with proper fetch calls
const blockchainClient = {
    sendMessage: async (url, data, options = {}) => {
        const response = await fetch(url, {
            method: data ? 'POST' : 'GET',
            headers: { 'Content-Type': 'application/json' },
            body: data ? JSON.stringify(data) : undefined,
            timeout: options.timeout || 30000
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return { data: result };
    }
};

class Agent0Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT0_BASE_URL || 'http://localhost:8000';
        this.apiVersion = 'v1';
        this.timeout = 30000;
    }

    // Data Products
    async getDataProducts(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const fetch = require('node-fetch');
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/data-products`, {
                params,
                timeout: this.timeout
            });

            return this._convertRESTToOData(response.data, 'DataProduct');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createDataProduct(data) {
        try {
            const restData = this._convertODataProductToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/data-products`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return this._convertRESTProductToOData(data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateDataProduct(id, data) {
        try {
            const restData = this._convertODataProductToREST(data);
            const response = await fetch(`${this.baseUrl}/api/${this.apiVersion}/data-products/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(restData),
                timeout: this.timeout
            });

            return this._convertRESTProductToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteDataProduct(id) {
        try {
            await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Dublin Core Operations
    async generateDublinCore(productId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/dublin-core/generate`, {}, {
                timeout: this.timeout
            });

            return {
                success: response.data.success,
                message: response.data.message,
                metadata: this._convertRESTDublinCoreToOData(response.data.dublin_core_metadata)
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateDublinCore(productId, dublinCoreData) {
        try {
            const restData = this._convertDublinCoreToREST(dublinCoreData);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/dublin-core`, restData, {
                timeout: this.timeout
            });

            return this._convertRESTDublinCoreToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Metadata Operations
    async validateMetadata(productId, validationOptions = {}) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/validate-metadata`, {
                validation_options: validationOptions
            }, {
                timeout: this.timeout
            });

            return {
                isValid: response.data.is_valid,
                validationResults: response.data.validation_results?.map(result => ({
                    field: result.field,
                    status: result.status?.toUpperCase(),
                    errors: result.errors || [],
                    warnings: result.warnings || []
                })) || [],
                overallScore: response.data.overall_score,
                recommendations: response.data.recommendations || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateSchema(productId, schemaData) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/validate-schema`, {
                schema_data: schemaData
            }, {
                timeout: this.timeout
            });

            return {
                isValid: response.data.is_valid,
                errors: response.data.errors?.map(error => ({
                    path: error.path,
                    message: error.message,
                    severity: error.severity?.toUpperCase()
                })) || [],
                schemaVersion: response.data.schema_version,
                validatedFields: response.data.validated_fields
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Quality Assessment
    async assessQuality(productId, assessmentCriteria = {}) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/assess-quality`, {
                assessment_criteria: this._convertQualityCriteriaToREST(assessmentCriteria)
            }, {
                timeout: this.timeout
            });

            return {
                overallScore: response.data.overall_score,
                qualityDimensions: {
                    completeness: response.data.quality_dimensions?.completeness || 0,
                    accuracy: response.data.quality_dimensions?.accuracy || 0,
                    consistency: response.data.quality_dimensions?.consistency || 0,
                    timeliness: response.data.quality_dimensions?.timeliness || 0,
                    validity: response.data.quality_dimensions?.validity || 0,
                    uniqueness: response.data.quality_dimensions?.uniqueness || 0
                },
                issues: response.data.issues?.map(issue => ({
                    type: issue.type,
                    severity: issue.severity?.toUpperCase(),
                    description: issue.description,
                    recommendation: issue.recommendation,
                    affectedRecords: issue.affected_records
                })) || [],
                recommendations: response.data.recommendations || [],
                timestamp: response.data.timestamp
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Publishing Operations
    async publishProduct(productId, publishOptions = {}) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/publish`, {
                publish_options: {
                    target_catalog: publishOptions.targetCatalog || 'default',
                    visibility: publishOptions.visibility || 'private',
                    approval_required: publishOptions.approvalRequired || false,
                    notification_enabled: publishOptions.notificationEnabled || true
                }
            }, {
                timeout: this.timeout
            });

            return {
                success: response.data.success,
                message: response.data.message,
                publicationId: response.data.publication_id,
                catalogUrl: response.data.catalog_url,
                publishedAt: response.data.published_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async archiveProduct(productId, archiveReason) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/archive`, {
                archive_reason: archiveReason
            }, {
                timeout: this.timeout
            });

            return {
                success: response.data.success,
                message: response.data.message,
                archivedAt: response.data.archived_at,
                archiveLocation: response.data.archive_location
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Data Lineage
    async getDataLineage(productId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/lineage`, {
                timeout: this.timeout
            });

            return {
                productId: response.data.product_id,
                lineageGraph: {
                    sources: response.data.lineage_graph?.sources?.map(source => ({
                        id: source.id,
                        name: source.name,
                        type: source.type,
                        description: source.description,
                        lastUpdated: source.last_updated
                    })) || [],
                    transformations: response.data.lineage_graph?.transformations?.map(transform => ({
                        id: transform.id,
                        name: transform.name,
                        type: transform.type,
                        description: transform.description,
                        rules: transform.rules
                    })) || [],
                    consumers: response.data.lineage_graph?.consumers?.map(consumer => ({
                        id: consumer.id,
                        name: consumer.name,
                        type: consumer.type,
                        description: consumer.description,
                        lastAccessed: consumer.last_accessed
                    })) || []
                },
                impactAnalysis: response.data.impact_analysis
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Version Management
    async createVersion(productId, versionData) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/versions`, {
                version_number: versionData.versionNumber,
                change_description: versionData.changeDescription,
                version_type: versionData.versionType || 'minor',
                auto_increment: versionData.autoIncrement || true
            }, {
                timeout: this.timeout
            });

            return {
                versionId: response.data.version_id,
                versionNumber: response.data.version_number,
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async compareVersions(productId, fromVersion, toVersion) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/${productId}/versions/compare`, {
                params: {
                    from_version: fromVersion,
                    to_version: toVersion
                },
                timeout: this.timeout
            });

            return {
                fromVersion: response.data.from_version,
                toVersion: response.data.to_version,
                differences: response.data.differences?.map(diff => ({
                    field: diff.field,
                    changeType: diff.change_type?.toUpperCase(),
                    oldValue: diff.old_value,
                    newValue: diff.new_value,
                    impact: diff.impact?.toUpperCase()
                })) || [],
                summary: response.data.summary
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Import/Export Operations
    async importMetadata(importData) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/import-metadata`, {
                import_format: importData.format,
                import_data: importData.data,
                import_options: {
                    overwrite_existing: importData.overwriteExisting || false,
                    validate_before_import: importData.validateBeforeImport || true,
                    create_backup: importData.createBackup || true
                }
            }, {
                timeout: this.timeout
            });

            return {
                success: response.data.success,
                message: response.data.message,
                importedCount: response.data.imported_count,
                skippedCount: response.data.skipped_count,
                errorCount: response.data.error_count,
                errors: response.data.errors || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportCatalog(exportFormat = 'json', includePrivate = false) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/export-catalog`, {
                export_format: exportFormat,
                include_private: includePrivate,
                export_options: {
                    include_dublin_core: true,
                    include_quality_metrics: true,
                    include_lineage: false
                }
            }, {
                timeout: this.timeout
            });

            return {
                success: response.data.success,
                downloadUrl: response.data.download_url,
                exportId: response.data.export_id,
                expiresAt: response.data.expires_at
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Bulk Operations
    async bulkUpdateProducts(productIds, updateData) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/bulk-update`, {
                product_ids: productIds,
                update_data: this._convertBulkUpdateToREST(updateData)
            }, {
                timeout: this.timeout * 2 // Longer timeout for bulk operations
            });

            return {
                success: response.data.success,
                message: response.data.message,
                updatedCount: response.data.updated_count,
                failedCount: response.data.failed_count,
                errors: response.data.errors || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async batchValidateProducts(productIds, validationType = 'metadata') {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-products/batch-validate`, {
                product_ids: productIds,
                validation_type: validationType,
                validation_options: {
                    include_quality_check: true,
                    include_schema_validation: true,
                    include_dublin_core_validation: true
                }
            }, {
                timeout: this.timeout * 2
            });

            return {
                validationId: response.data.validation_id,
                results: response.data.results?.map(result => ({
                    productId: result.product_id,
                    isValid: result.is_valid,
                    score: result.score,
                    errors: result.errors || [],
                    warnings: result.warnings || []
                })) || [],
                summary: {
                    totalProducts: response.data.summary?.total_products || 0,
                    validProducts: response.data.summary?.valid_products || 0,
                    invalidProducts: response.data.summary?.invalid_products || 0
                }
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Dashboard Data
    async getDashboardMetrics() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/dashboard/metrics`, {
                timeout: this.timeout
            });

            return {
                totalProducts: response.data.total_products || 0,
                activeProducts: response.data.active_products || 0,
                averageQuality: response.data.average_quality || 0,
                productsByType: response.data.products_by_type || {},
                qualityDistribution: response.data.quality_distribution || {},
                recentActivity: response.data.recent_activity?.map(activity => ({
                    id: activity.id,
                    productName: activity.product_name,
                    action: activity.action,
                    timestamp: activity.timestamp,
                    user: activity.user
                })) || [],
                topContributors: response.data.top_contributors || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Dublin Core Metadata
    async getDublinCoreMetadata(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/dublin-core`, {
                params,
                timeout: this.timeout
            });

            return this._convertRESTToOData(response.data, 'DublinCoreMetadata');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Ingestion Sessions
    async getIngestionSessions(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/ingestion-sessions`, {
                params,
                timeout: this.timeout
            });

            return this._convertRESTToOData(response.data, 'IngestionSession');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Quality Assessments
    async getQualityAssessments(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/quality-assessments`, {
                params,
                timeout: this.timeout
            });

            return this._convertRESTToOData(response.data, 'QualityAssessment');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Product Transformations
    async getProductTransformations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/product-transformations`, {
                params,
                timeout: this.timeout
            });

            return this._convertRESTToOData(response.data, 'ProductTransformation');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Utility methods
    _convertODataToREST(query) {
        const params = {};

        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$orderby) params.sort = query.$orderby.replace(/ desc/gi, '-').replace(/ asc/gi, '');
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$select) params.fields = query.$select;

        return params;
    }

    _parseODataFilter(filter) {
        // Convert OData filter to REST query parameters
        return filter
            .replace(/ eq /g, '=')
            .replace(/ ne /g, '!=')
            .replace(/ gt /g, '>')
            .replace(/ ge /g, '>=')
            .replace(/ lt /g, '<')
            .replace(/ le /g, '<=')
            .replace(/ and /g, '&')
            .replace(/ or /g, '|');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        switch (entityType) {
            case 'DataProduct':
                return this._convertRESTProductToOData(item);
            case 'DublinCoreMetadata':
                return this._convertRESTDublinCoreToOData(item);
            case 'IngestionSession':
                return this._convertRESTIngestionToOData(item);
            case 'QualityAssessment':
                return this._convertRESTQualityToOData(item);
            case 'ProductTransformation':
                return this._convertRESTTransformationToOData(item);
            default:
                return item;
        }
    }

    _convertRESTProductToOData(product) {
        return {
            ID: product.id || `product_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            productName: product.product_name || product.name,
            description: product.description,
            productType: product.product_type?.toUpperCase() || 'DATASET',
            status: product.status?.toUpperCase() || 'DRAFT',
            dataSource: product.data_source,
            dataFormat: product.data_format?.toUpperCase() || 'JSON',
            dataSizeMB: product.data_size_mb,
            recordCount: product.record_count,
            qualityScore: product.quality_score,
            isPublic: product.is_public,
            isEncrypted: product.is_encrypted,
            compressionType: product.compression_type,
            schemaVersion: product.schema_version,
            dataSchema: JSON.stringify(product.data_schema || {}),
            sampleData: JSON.stringify(product.sample_data || {}),
            tags: product.tags,
            category: product.category,
            owner: product.owner,
            dataClassification: product.data_classification?.toUpperCase() || 'INTERNAL',
            retentionDays: product.retention_days,
            expiryDate: product.expiry_date,
            createdBy: product.created_by,
            lastAccessed: product.last_accessed,
            accessCount: product.access_count,
            processingPipeline: product.processing_pipeline,
            nextAgent: product.next_agent,
            createdAt: product.created_at,
            modifiedAt: product.modified_at,
            modifiedBy: product.modified_by
        };
    }

    _convertODataProductToREST(product) {
        const restProduct = {
            product_name: product.productName,
            description: product.description,
            product_type: product.productType?.toLowerCase(),
            status: product.status?.toLowerCase()
        };

        if (product.dataSource) restProduct.data_source = product.dataSource;
        if (product.dataFormat) restProduct.data_format = product.dataFormat.toLowerCase();
        if (product.dataSizeMB !== undefined) restProduct.data_size_mb = product.dataSizeMB;
        if (product.recordCount !== undefined) restProduct.record_count = product.recordCount;
        if (product.qualityScore !== undefined) restProduct.quality_score = product.qualityScore;
        if (product.isPublic !== undefined) restProduct.is_public = product.isPublic;
        if (product.isEncrypted !== undefined) restProduct.is_encrypted = product.isEncrypted;
        if (product.compressionType) restProduct.compression_type = product.compressionType;
        if (product.schemaVersion) restProduct.schema_version = product.schemaVersion;
        if (product.dataSchema) restProduct.data_schema = JSON.parse(product.dataSchema);
        if (product.sampleData) restProduct.sample_data = JSON.parse(product.sampleData);
        if (product.tags) restProduct.tags = product.tags;
        if (product.category) restProduct.category = product.category;
        if (product.owner) restProduct.owner = product.owner;
        if (product.dataClassification) restProduct.data_classification = product.dataClassification.toLowerCase();
        if (product.retentionDays !== undefined) restProduct.retention_days = product.retentionDays;
        if (product.expiryDate) restProduct.expiry_date = product.expiryDate;
        if (product.processingPipeline) restProduct.processing_pipeline = product.processingPipeline;
        if (product.nextAgent) restProduct.next_agent = product.nextAgent;

        return restProduct;
    }

    _convertRESTDublinCoreToOData(dublinCore) {
        return {
            ID: dublinCore.id || `dublin_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            title: dublinCore.title,
            creator: dublinCore.creator,
            subject: dublinCore.subject,
            description: dublinCore.description,
            publisher: dublinCore.publisher,
            contributor: dublinCore.contributor,
            date: dublinCore.date,
            type: dublinCore.type,
            format: dublinCore.format,
            identifier: dublinCore.identifier,
            source: dublinCore.source,
            language: dublinCore.language,
            relation: dublinCore.relation,
            coverage: dublinCore.coverage,
            rights: dublinCore.rights,
            additionalMetadata: JSON.stringify(dublinCore.additional_metadata || {}),
            metadataVersion: dublinCore.metadata_version,
            createdAt: dublinCore.created_at,
            modifiedAt: dublinCore.modified_at
        };
    }

    _convertDublinCoreToREST(dublinCore) {
        return {
            title: dublinCore.title,
            creator: dublinCore.creator,
            subject: dublinCore.subject,
            description: dublinCore.description,
            publisher: dublinCore.publisher,
            contributor: dublinCore.contributor,
            date: dublinCore.date,
            type: dublinCore.type,
            format: dublinCore.format,
            identifier: dublinCore.identifier,
            source: dublinCore.source,
            language: dublinCore.language,
            relation: dublinCore.relation,
            coverage: dublinCore.coverage,
            rights: dublinCore.rights,
            additional_metadata: dublinCore.additionalMetadata ? JSON.parse(dublinCore.additionalMetadata) : {},
            metadata_version: dublinCore.metadataVersion
        };
    }

    _convertRESTIngestionToOData(session) {
        return {
            ID: session.id || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            sessionName: session.session_name,
            status: session.status?.toUpperCase(),
            startTime: session.start_time,
            endTime: session.end_time,
            recordsIngested: session.records_ingested,
            recordsFailed: session.records_failed,
            errorMessages: JSON.stringify(session.error_messages || []),
            ingestionMethod: session.ingestion_method?.toUpperCase(),
            sourceLocation: session.source_location,
            configuration: JSON.stringify(session.configuration || {}),
            createdAt: session.created_at,
            modifiedAt: session.modified_at
        };
    }

    _convertRESTQualityToOData(assessment) {
        return {
            ID: assessment.id || `assessment_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            assessmentType: assessment.assessment_type?.toUpperCase(),
            overallScore: assessment.overall_score,
            completenessScore: assessment.completeness_score,
            accuracyScore: assessment.accuracy_score,
            consistencyScore: assessment.consistency_score,
            timelinessScore: assessment.timeliness_score,
            validityScore: assessment.validity_score,
            uniquenessScore: assessment.uniqueness_score,
            assessmentResults: JSON.stringify(assessment.assessment_results || {}),
            recommendations: JSON.stringify(assessment.recommendations || []),
            assessedAt: assessment.assessed_at,
            assessedBy: assessment.assessed_by,
            createdAt: assessment.created_at
        };
    }

    _convertRESTTransformationToOData(transformation) {
        return {
            ID: transformation.id || `transform_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            transformationName: transformation.transformation_name,
            transformationType: transformation.transformation_type?.toUpperCase(),
            transformationRules: JSON.stringify(transformation.transformation_rules || {}),
            status: transformation.status?.toUpperCase(),
            inputSchema: JSON.stringify(transformation.input_schema || {}),
            outputSchema: JSON.stringify(transformation.output_schema || {}),
            recordsProcessed: transformation.records_processed,
            recordsSuccessful: transformation.records_successful,
            recordsFailed: transformation.records_failed,
            errorRate: transformation.error_rate,
            executionTime: transformation.execution_time,
            createdAt: transformation.created_at,
            modifiedAt: transformation.modified_at
        };
    }

    _convertQualityCriteriaToREST(criteria) {
        if (!criteria) return {};

        return {
            completeness_weight: criteria.completenessWeight || 20,
            accuracy_weight: criteria.accuracyWeight || 20,
            consistency_weight: criteria.consistencyWeight || 20,
            timeliness_weight: criteria.timelinessWeight || 20,
            validity_weight: criteria.validityWeight || 10,
            uniqueness_weight: criteria.uniquenessWeight || 10,
            custom_criteria: criteria.customCriteria
        };
    }

    _convertBulkUpdateToREST(updateData) {
        const restData = {};

        if (updateData.status) restData.status = updateData.status.toLowerCase();
        if (updateData.category) restData.category = updateData.category;
        if (updateData.tags) restData.tags = updateData.tags;
        if (updateData.dataClassification) restData.data_classification = updateData.dataClassification.toLowerCase();
        if (updateData.retentionDays !== undefined) restData.retention_days = updateData.retentionDays;
        if (updateData.nextAgent) restData.next_agent = updateData.nextAgent;

        return restData;
    }

    _handleError(error) {
        if (error.response) {
            const status = error.response.status;
            const message = error.response.data?.message || error.message;

            switch (status) {
                case 400:
                    return new Error(`Bad Request: ${message}`);
                case 401:
                    return new Error(`Unauthorized: ${message}`);
                case 403:
                    return new Error(`Forbidden: ${message}`);
                case 404:
                    return new Error(`Not Found: ${message}`);
                case 409:
                    return new Error(`Conflict: ${message}`);
                case 422:
                    return new Error(`Validation Error: ${message}`);
                case 500:
                    return new Error(`Internal Server Error: ${message}`);
                default:
                    return new Error(`HTTP ${status}: ${message}`);
            }
        } else if (error.request) {
            return new Error(`No response from Agent 0 service: ${error.message}`);
        } else {
            return new Error(`Agent 0 adapter error: ${error.message}`);
        }
    }
}

module.exports = Agent0Adapter;