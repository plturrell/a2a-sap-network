/**
 * Agent 0 Service Implementation - Data Product Agent
 * Implements business logic for data product management with Dublin Core metadata,
 * quality assessment, data lineage tracking, and catalog publishing
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');
const { SELECT, INSERT, UPDATE, DELETE } = cds.ql;
const Agent0Adapter = require('../adapters/agent0-adapter');
const { LoggerFactory } = require('../../shared/logging/structured-logger');

class Agent0Service extends cds.ApplicationService {
    async init() {
        const db = await cds.connect.to('db');
        this.adapter = new Agent0Adapter();
        this.logger = LoggerFactory.createAgentLogger('0', process.env.NODE_ENV || 'development');

        // Entity references
        const {
            DataProducts,
            DublinCoreMetadata,
            IngestionSessions,
            QualityAssessments,
            ProductTransformations
        } = db.entities;

        // CRUD Operations for DataProducts
        const handleReadDataProducts = async (req) => {
            try {
                const products = await this.adapter.getDataProducts(req.query);
                return products;
            } catch (error) {
                req.error(500, `Failed to read data products: ${error.message}`);
            }
        };
        this.on('READ', 'DataProducts', handleReadDataProducts);

        const handleCreateDataProducts = async (req) => {
            try {
                // Validate required Dublin Core fields
                if (!req.data.productName || !req.data.description) {
                    req.error(400, 'Product name and description are required');
                    return;
                }

                const product = await this.adapter.createDataProduct(req.data);

                // Auto-generate Dublin Core metadata if not provided
                if (!req.data.dublinCoreMetadata) {
                    try {
                        const dublinCore = await this.adapter.generateDublinCore(product.ID);
                        await INSERT.into(DublinCoreMetadata).entries({
                            ID: uuidv4(),
                            dataProduct_ID: product.ID,
                            ...dublinCore.metadata
                        });
                    } catch (dcError) {
                        this.logger.warn('Failed to auto-generate Dublin Core metadata', {
                            productId: product.ID,
                            error: dcError.message,
                            stack: dcError.stack,
                            operation: 'createDataProduct',
                            agent: 'agent-0'
                        });
                    }
                }

                // Emit event
                await this.emit('DataProductCreated', {
                    productId: product.ID,
                    productName: product.productName,
                    timestamp: new Date(),
                    createdBy: req.data.createdBy || req.user.id
                });

                return product;
            } catch (error) {
                req.error(500, `Failed to create data product: ${error.message}`);
            }
        };
        this.on('CREATE', 'DataProducts', handleCreateDataProducts);

        const handleUpdateDataProducts = async (req) => {
            try {
                const product = await this.adapter.updateDataProduct(req.params[0], req.data);

                // Emit event
                await this.emit('DataProductUpdated', {
                    productId: req.params[0],
                    timestamp: new Date(),
                    modifiedBy: req.user.id
                });

                return product;
            } catch (error) {
                req.error(500, `Failed to update data product: ${error.message}`);
            }
        };
        this.on('UPDATE', 'DataProducts', handleUpdateDataProducts);

        const handleDeleteDataProducts = async (req) => {
            try {
                await this.adapter.deleteDataProduct(req.params[0]);

                // Emit event
                await this.emit('DataProductDeleted', {
                    productId: req.params[0],
                    timestamp: new Date(),
                    deletedBy: req.user.id
                });
            } catch (error) {
                req.error(500, `Failed to delete data product: ${error.message}`);
            }
        };
        this.on('DELETE', 'DataProducts', handleDeleteDataProducts);

        // Dublin Core Operations
        const handleGenerateDublinCore = async (req) => {
            try {
                const { ID } = req.params[0];
                const result = await this.adapter.generateDublinCore(ID);

                // Update or create Dublin Core metadata in database
                const existingDC = await SELECT.one.from(DublinCoreMetadata).where({ dataProduct_ID: ID });

                if (existingDC) {
                    await UPDATE(DublinCoreMetadata)
                        .set(result.metadata)
                        .where({ dataProduct_ID: ID });
                } else {
                    await INSERT.into(DublinCoreMetadata).entries({
                        ID: uuidv4(),
                        dataProduct_ID: ID,
                        ...result.metadata
                    });
                }

                // Update product's last modified timestamp
                await UPDATE(DataProducts)
                    .set({
                        modifiedAt: new Date(),
                        modifiedBy: req.user.id
                    })
                    .where({ ID });

                // Emit event
                await this.emit('DublinCoreGenerated', {
                    productId: ID,
                    timestamp: new Date(),
                    generatedBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to generate Dublin Core metadata: ${error.message}`);
            }
        };
        this.on('generateDublinCore', 'DataProducts', handleGenerateDublinCore);

        const handleUpdateDublinCore = async (req) => {
            try {
                const { ID } = req.params[0];
                const dublinCoreData = req.data;

                const result = await this.adapter.updateDublinCore(ID, dublinCoreData);

                // Update database
                await UPDATE(DublinCoreMetadata)
                    .set(result)
                    .where({ dataProduct_ID: ID });

                // Emit event
                await this.emit('DublinCoreUpdated', {
                    productId: ID,
                    timestamp: new Date(),
                    updatedBy: req.user.id
                });

                return 'Dublin Core metadata updated successfully';
            } catch (error) {
                req.error(500, `Failed to update Dublin Core metadata: ${error.message}`);
            }
        };
        this.on('updateDublinCore', 'DataProducts', handleUpdateDublinCore);

        // Validation Operations
        const handleValidateMetadata = async (req) => {
            try {
                const { ID } = req.params[0];
                const validationOptions = req.data || {};

                const result = await this.adapter.validateMetadata(ID, validationOptions);

                // Update product validation status
                await UPDATE(DataProducts)
                    .set({
                        qualityScore: result.overallScore,
                        modifiedAt: new Date()
                    })
                    .where({ ID });

                // Emit event
                await this.emit('MetadataValidated', {
                    productId: ID,
                    isValid: result.isValid,
                    score: result.overallScore,
                    timestamp: new Date()
                });

                return `Metadata validation completed. Overall score: ${result.overallScore}`;
            } catch (error) {
                req.error(500, `Failed to validate metadata: ${error.message}`);
            }
        };
        this.on('validateMetadata', 'DataProducts', handleValidateMetadata);

        const handleValidateSchema = async (req) => {
            try {
                const { ID } = req.params[0];
                const schemaData = req.data;

                const result = await this.adapter.validateSchema(ID, schemaData);

                // Emit event
                await this.emit('SchemaValidated', {
                    productId: ID,
                    isValid: result.isValid,
                    errors: result.errors,
                    timestamp: new Date()
                });

                return result.isValid ? 'Schema validation passed' : 'Schema validation failed';
            } catch (error) {
                req.error(500, `Failed to validate schema: ${error.message}`);
            }
        };
        this.on('validateSchema', 'DataProducts', handleValidateSchema);

        // Quality Assessment
        const handleAssessQuality = async (req) => {
            try {
                const { ID } = req.params[0];
                const assessmentCriteria = req.data || {};

                const result = await this.adapter.assessQuality(ID, assessmentCriteria);

                // Create quality assessment record
                await INSERT.into(QualityAssessments).entries({
                    ID: uuidv4(),
                    dataProduct_ID: ID,
                    assessmentType: 'COMPREHENSIVE',
                    overallScore: result.overallScore,
                    completenessScore: result.qualityDimensions.completeness,
                    accuracyScore: result.qualityDimensions.accuracy,
                    consistencyScore: result.qualityDimensions.consistency,
                    timelinessScore: result.qualityDimensions.timeliness,
                    validityScore: result.qualityDimensions.validity,
                    uniquenessScore: result.qualityDimensions.uniqueness,
                    assessmentResults: JSON.stringify(result),
                    recommendations: JSON.stringify(result.recommendations),
                    assessedAt: new Date(),
                    assessedBy: req.user.id
                });

                // Update product quality score
                await UPDATE(DataProducts)
                    .set({
                        qualityScore: result.overallScore,
                        modifiedAt: new Date()
                    })
                    .where({ ID });

                // Emit event
                await this.emit('QualityAssessed', {
                    productId: ID,
                    overallScore: result.overallScore,
                    qualityDimensions: result.qualityDimensions,
                    timestamp: new Date()
                });

                return `Quality assessment completed. Overall score: ${result.overallScore}`;
            } catch (error) {
                req.error(500, `Failed to assess quality: ${error.message}`);
            }
        };
        this.on('assessQuality', 'DataProducts', handleAssessQuality);

        // Publishing Operations
        const handlePublish = async (req) => {
            try {
                const { ID } = req.params[0];
                const publishOptions = req.data || {};

                // Check if Dublin Core metadata exists and is complete
                const dublinCore = await SELECT.one.from(DublinCoreMetadata).where({ dataProduct_ID: ID });
                if (!dublinCore || !dublinCore.title || !dublinCore.creator) {
                    req.error(400, 'Complete Dublin Core metadata is required for publishing');
                    return;
                }

                const result = await this.adapter.publishProduct(ID, publishOptions);

                // Update product status
                await UPDATE(DataProducts)
                    .set({
                        status: 'ACTIVE',
                        modifiedAt: new Date()
                    })
                    .where({ ID });

                // Emit event
                await this.emit('DataProductPublished', {
                    productId: ID,
                    publicationId: result.publicationId,
                    catalogUrl: result.catalogUrl,
                    timestamp: new Date(),
                    publishedBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to publish product: ${error.message}`);
            }
        };
        this.on('publish', 'DataProducts', handlePublish);

        this.on('archive', 'DataProducts', async (req) => {
            try {
                const { ID } = req.params[0];
                const { reason } = req.data;

                const result = await this.adapter.archiveProduct(ID, reason);

                // Update product status
                await UPDATE(DataProducts)
                    .set({
                        status: 'ARCHIVED',
                        modifiedAt: new Date()
                    })
                    .where({ ID });

                // Emit event
                await this.emit('DataProductArchived', {
                    productId: ID,
                    reason,
                    timestamp: new Date(),
                    archivedBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to archive product: ${error.message}`);
            }
        });

        // Data Lineage
        this.on('getLineage', 'DataProducts', async (req) => {
            try {
                const { ID } = req.params[0];
                const lineage = await this.adapter.getDataLineage(ID);
                return lineage;
            } catch (error) {
                req.error(500, `Failed to get data lineage: ${error.message}`);
            }
        });

        // Version Management
        this.on('createVersion', 'DataProducts', async (req) => {
            try {
                const { ID } = req.params[0];
                const versionData = req.data;

                const result = await this.adapter.createVersion(ID, versionData);

                // Emit event
                await this.emit('DataProductVersionCreated', {
                    productId: ID,
                    versionId: result.versionId,
                    versionNumber: result.versionNumber,
                    timestamp: new Date(),
                    createdBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to create version: ${error.message}`);
            }
        });

        this.on('compareVersions', 'DataProducts', async (req) => {
            try {
                const { ID } = req.params[0];
                const { fromVersion, toVersion } = req.data;

                const comparison = await this.adapter.compareVersions(ID, fromVersion, toVersion);
                return comparison;
            } catch (error) {
                req.error(500, `Failed to compare versions: ${error.message}`);
            }
        });

        // Function implementations
        this.on('getDashboardMetrics', async (req) => {
            try {
                const metrics = await this.adapter.getDashboardMetrics();
                return metrics;
            } catch (error) {
                req.error(500, `Failed to get dashboard metrics: ${error.message}`);
            }
        });

        this.on('importMetadata', async (req) => {
            try {
                const importData = req.data;
                const result = await this.adapter.importMetadata(importData);

                // Emit event
                await this.emit('MetadataImported', {
                    importedCount: result.importedCount,
                    skippedCount: result.skippedCount,
                    errorCount: result.errorCount,
                    timestamp: new Date(),
                    importedBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to import metadata: ${error.message}`);
            }
        });

        this.on('exportCatalog', async (req) => {
            try {
                const { format, includePrivate } = req.data;
                const result = await this.adapter.exportCatalog(format, includePrivate);

                // Emit event
                await this.emit('CatalogExported', {
                    exportId: result.exportId,
                    format,
                    timestamp: new Date(),
                    exportedBy: req.user.id
                });

                return result;
            } catch (error) {
                req.error(500, `Failed to export catalog: ${error.message}`);
            }
        });

        this.on('bulkUpdateProducts', async (req) => {
            try {
                const { productIds, updateData } = req.data;
                const result = await this.adapter.bulkUpdateProducts(productIds, updateData);

                // Update modified timestamps in database
                await UPDATE(DataProducts)
                    .set({ modifiedAt: new Date() })
                    .where({ ID: { in: productIds } });

                // Emit event
                await this.emit('ProductsBulkUpdated', {
                    productIds,
                    updatedCount: result.updatedCount,
                    failedCount: result.failedCount,
                    timestamp: new Date(),
                    updatedBy: req.user.id
                });

                return result.message;
            } catch (error) {
                req.error(500, `Failed to bulk update products: ${error.message}`);
            }
        });

        this.on('batchValidateProducts', async (req) => {
            try {
                const { productIds, validationType } = req.data;
                const result = await this.adapter.batchValidateProducts(productIds, validationType);

                // Update quality scores in database for valid products
                for (const validationResult of result.results) {
                    if (validationResult.isValid && validationResult.score !== undefined) {
                        await UPDATE(DataProducts)
                            .set({
                                qualityScore: validationResult.score,
                                modifiedAt: new Date()
                            })
                            .where({ ID: validationResult.productId });
                    }
                }

                // Emit event
                await this.emit('ProductsBatchValidated', {
                    validationId: result.validationId,
                    totalProducts: result.summary.totalProducts,
                    validProducts: result.summary.validProducts,
                    invalidProducts: result.summary.invalidProducts,
                    timestamp: new Date()
                });

                return `Batch validation completed. ${result.summary.validProducts}/${result.summary.totalProducts} products passed validation.`;
            } catch (error) {
                req.error(500, `Failed to batch validate products: ${error.message}`);
            }
        });

        // CRUD for DublinCoreMetadata
        this.on('READ', 'DublinCoreMetadata', async (req) => {
            try {
                const metadata = await this.adapter.getDublinCoreMetadata(req.query);
                return metadata;
            } catch (error) {
                req.error(500, `Failed to read Dublin Core metadata: ${error.message}`);
            }
        });

        // CRUD for IngestionSessions
        this.on('READ', 'IngestionSessions', async (req) => {
            try {
                const sessions = await this.adapter.getIngestionSessions(req.query);
                return sessions;
            } catch (error) {
                req.error(500, `Failed to read ingestion sessions: ${error.message}`);
            }
        });

        // CRUD for QualityAssessments
        this.on('READ', 'QualityAssessments', async (req) => {
            try {
                const assessments = await this.adapter.getQualityAssessments(req.query);
                return assessments;
            } catch (error) {
                req.error(500, `Failed to read quality assessments: ${error.message}`);
            }
        });

        // CRUD for ProductTransformations
        this.on('READ', 'ProductTransformations', async (req) => {
            try {
                const transformations = await this.adapter.getProductTransformations(req.query);
                return transformations;
            } catch (error) {
                req.error(500, `Failed to read product transformations: ${error.message}`);
            }
        });

        // Stream handler for real-time data product updates
        this.on('streamProductUpdates', async function* (req) {
            const { productId } = req.data;

            // Simulate real-time updates - in production, this would connect to actual streams
            const updateInterval = 5000; // 5 seconds
            let counter = 0;

            while (counter < 10) { // Limit for demo purposes
                try {
                    // Get latest product data
                    const product = await SELECT.one.from(DataProducts).where({ ID: productId });

                    if (product) {
                        yield {
                            productId,
                            timestamp: new Date(),
                            status: product.status,
                            qualityScore: product.qualityScore,
                            lastAccessed: product.lastAccessed,
                            accessCount: product.accessCount,
                            updateNumber: counter + 1
                        };
                    }

                    await new Promise(resolve => setTimeout(resolve, updateInterval));
                    counter++;
                } catch (error) {
                    yield {
                        productId,
                        timestamp: new Date(),
                        error: error.message,
                        updateNumber: counter + 1
                    };
                    break;
                }
            }
        });

        await super.init();
    }
}

module.exports = Agent0Service;