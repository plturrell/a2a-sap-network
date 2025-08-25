/**
 * @fileoverview Enterprise API Documentation Generator with Version Support
 * @description Generates comprehensive API documentation with version-specific changes,
 * deprecation notices, migration guides, and SAP enterprise standards
 * @module api-documentation-generator
 * @since 1.0.0
 * @author A2A Network Team
 */

const fs = require('fs').promises;

const { LoggerFactory } = require('../../shared/logging/structured-logger');
const logger = LoggerFactory.createLogger('apiDocumentationGenerator');
const path = require('path');
const { API_CONFIG, BREAKING_CHANGES } = require('../middleware/apiVersioning');

/**
 * API Documentation Generator
 */
class APIDocumentationGenerator {
    constructor() {
        this.apiRoutes = new Map();
        this.versionedRoutes = new Map();
        this.endpoints = [];
    }

    /**
     * Register API route for documentation
     */
    registerRoute(method, path, version, description, params = {}, responses = {}) {
        const routeKey = `${method}:${path}`;

        if (!this.versionedRoutes.has(routeKey)) {
            this.versionedRoutes.set(routeKey, new Map());
        }

        this.versionedRoutes.get(routeKey).set(version, {
            method,
            path,
            version,
            description,
            parameters: params,
            responses,
            deprecated: API_CONFIG.deprecationSchedule[version]?.status === 'deprecated',
            sunset: API_CONFIG.deprecationSchedule[version]?.sunsetDate
        });
    }

    /**
     * Generate complete API documentation
     */
    async generateDocumentation() {
        const docs = {
            openapi: '3.0.3',
            info: {
                title: 'A2A Network API',
                description: 'Enterprise SAP-integrated Autonomous Agent Orchestration Platform API',
                version: API_CONFIG.currentVersion,
                contact: {
                    name: 'A2A Network Team',
                    email: 'support@a2a-network.com'
                },
                license: {
                    name: 'Apache 2.0',
                    url: 'https://www.apache.org/licenses/LICENSE-2.0.html'
                }
            },
            servers: [
                {
                    url: 'https://api.a2a-network.com',
                    description: 'Production server'
                },
                {
                    url: 'https://staging-api.a2a-network.com',
                    description: 'Staging server'
                },
                {
                    url: 'http://localhost:4004',
                    description: 'Development server'
                }
            ],
            paths: await this.generatePaths(),
            components: {
                securitySchemes: {
                    BearerAuth: {
                        type: 'http',
                        scheme: 'bearer',
                        bearerFormat: 'JWT'
                    },
                    ApiKeyAuth: {
                        type: 'apiKey',
                        in: 'header',
                        name: 'X-API-Key'
                    }
                },
                schemas: this.generateSchemas(),
                parameters: this.generateCommonParameters()
            },
            security: [
                { BearerAuth: [] },
                { ApiKeyAuth: [] }
            ],
            'x-api-versions': this.generateVersionInfo(),
            'x-deprecation-policy': this.generateDeprecationPolicy(),
            'x-migration-guides': await this.generateMigrationGuides()
        };

        return docs;
    }

    /**
     * Generate API paths with version support
     */
    async generatePaths() {
        const paths = {};

        // Version management endpoints
        paths['/api/versions'] = {
            get: {
                summary: 'Get supported API versions',
                description: 'Returns all supported API versions with their status and features',
                tags: ['Version Management'],
                responses: {
                    200: {
                        description: 'Supported versions information',
                        content: {
                            'application/json': {
                                schema: {
                                    type: 'object',
                                    properties: {
                                        supportedVersions: {
                                            type: 'array',
                                            items: { type: 'string' },
                                            example: ['1.0.0', '1.1.0', '2.0.0']
                                        },
                                        currentVersion: {
                                            type: 'string',
                                            example: '2.0.0'
                                        },
                                        defaultVersion: {
                                            type: 'string',
                                            example: '1.0.0'
                                        },
                                        versionsInfo: {
                                            type: 'array',
                                            items: { $ref: '#/components/schemas/VersionInfo' }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        paths['/api/version/{version}'] = {
            get: {
                summary: 'Get specific version information',
                description: 'Returns detailed information about a specific API version',
                tags: ['Version Management'],
                parameters: [{
                    name: 'version',
                    in: 'path',
                    required: true,
                    schema: { type: 'string' },
                    example: '2.0.0'
                }],
                responses: {
                    200: {
                        description: 'Version information',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/DetailedVersionInfo' }
                            }
                        }
                    },
                    400: {
                        description: 'Invalid version specified'
                    }
                }
            }
        };

        paths['/api/migration/{from}/{to}'] = {
            get: {
                summary: 'Get migration guide',
                description: 'Returns detailed migration guide between API versions',
                tags: ['Version Management'],
                parameters: [
                    {
                        name: 'from',
                        in: 'path',
                        required: true,
                        schema: { type: 'string' },
                        example: '1.0.0'
                    },
                    {
                        name: 'to',
                        in: 'path',
                        required: false,
                        schema: { type: 'string' },
                        example: '2.0.0'
                    }
                ],
                responses: {
                    200: {
                        description: 'Migration guide',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/MigrationGuide' }
                            }
                        }
                    }
                }
            }
        };

        // Settings endpoints (versioned)
        paths['/api/v1/settings/network'] = {
            get: {
                summary: 'Get network settings',
                description: 'Retrieve current network configuration settings',
                tags: ['Settings'],
                parameters: [{ $ref: '#/components/parameters/ApiVersion' }],
                responses: {
                    200: {
                        description: 'Network settings',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/NetworkSettings' }
                            }
                        }
                    }
                }
            },
            put: {
                summary: 'Update network settings',
                description: 'Update network configuration settings',
                tags: ['Settings'],
                parameters: [{ $ref: '#/components/parameters/ApiVersion' }],
                requestBody: {
                    required: true,
                    content: {
                        'application/json': {
                            schema: { $ref: '#/components/schemas/NetworkSettingsUpdate' }
                        }
                    }
                },
                responses: {
                    200: {
                        description: 'Settings updated successfully'
                    },
                    400: {
                        description: 'Invalid settings data'
                    }
                }
            }
        };

        // Metrics endpoints
        paths['/api/v1/metrics/current'] = {
            get: {
                summary: 'Get current metrics',
                description: 'Retrieve real-time system metrics',
                tags: ['Metrics'],
                parameters: [{ $ref: '#/components/parameters/ApiVersion' }],
                responses: {
                    200: {
                        description: 'Current system metrics',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/SystemMetrics' }
                            }
                        }
                    }
                }
            }
        };

        // Operations endpoints
        paths['/api/v1/operations/status'] = {
            get: {
                summary: 'Get operations status',
                description: 'Retrieve system operational status',
                tags: ['Operations'],
                parameters: [{ $ref: '#/components/parameters/ApiVersion' }],
                responses: {
                    200: {
                        description: 'Operations status',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/OperationsStatus' }
                            }
                        }
                    }
                }
            }
        };

        // Blockchain endpoints
        paths['/api/v1/blockchain/status'] = {
            get: {
                summary: 'Get blockchain status',
                description: 'Retrieve blockchain network status and statistics',
                tags: ['Blockchain'],
                parameters: [{ $ref: '#/components/parameters/ApiVersion' }],
                responses: {
                    200: {
                        description: 'Blockchain status',
                        content: {
                            'application/json': {
                                schema: { $ref: '#/components/schemas/BlockchainStatus' }
                            }
                        }
                    }
                }
            }
        };

        return paths;
    }

    /**
     * Generate common schemas
     */
    generateSchemas() {
        return {
            VersionInfo: {
                type: 'object',
                properties: {
                    version: { type: 'string', example: '2.0.0' },
                    status: {
                        type: 'string',
                        enum: ['current', 'supported', 'deprecated', 'sunset'],
                        example: 'current'
                    },
                    features: {
                        type: 'object',
                        additionalProperties: true
                    }
                }
            },
            DetailedVersionInfo: {
                type: 'object',
                properties: {
                    version: { type: 'string' },
                    status: { type: 'string' },
                    features: { type: 'object' },
                    deprecation: {
                        type: 'object',
                        properties: {
                            isDeprecated: { type: 'boolean' },
                            deprecatedDate: { type: 'string', format: 'date-time' },
                            sunsetDate: { type: 'string', format: 'date-time' },
                            migrationPath: { type: 'string' }
                        }
                    },
                    breakingChanges: {
                        type: 'array',
                        items: {
                            type: 'object',
                            properties: {
                                type: { type: 'string' },
                                endpoint: { type: 'string' },
                                field: { type: 'string' },
                                description: { type: 'string' }
                            }
                        }
                    }
                }
            },
            MigrationGuide: {
                type: 'object',
                properties: {
                    migrationPath: {
                        type: 'object',
                        properties: {
                            from: { type: 'string' },
                            to: { type: 'string' }
                        }
                    },
                    compatibility: { type: 'boolean' },
                    breakingChanges: {
                        type: 'array',
                        items: { type: 'object' }
                    },
                    featureComparison: {
                        type: 'object',
                        properties: {
                            removed: { type: 'array' },
                            added: { type: 'array' },
                            changed: { type: 'array' }
                        }
                    },
                    estimatedEffort: {
                        type: 'object',
                        properties: {
                            level: { type: 'string', enum: ['low', 'medium', 'high'] },
                            breakingChangesCount: { type: 'integer' },
                            majorVersionChange: { type: 'boolean' },
                            estimatedHours: { type: 'integer' }
                        }
                    },
                    recommendedSteps: {
                        type: 'array',
                        items: { type: 'string' }
                    }
                }
            },
            NetworkSettings: {
                type: 'object',
                properties: {
                    network: { type: 'string', example: 'mainnet' },
                    rpcUrl: { type: 'string', example: 'https://mainnet.infura.io/v3/your-project-id' },
                    chainId: { type: 'integer', example: 1 },
                    contractAddress: { type: 'string', example: '0x...' }
                }
            },
            NetworkSettingsUpdate: {
                type: 'object',
                properties: {
                    network: { type: 'string' },
                    rpcUrl: { type: 'string' },
                    chainId: { type: 'integer' },
                    contractAddress: { type: 'string' }
                }
            },
            SystemMetrics: {
                type: 'object',
                properties: {
                    cpuUsage: { type: 'number', example: 45.2 },
                    memoryUsagePercent: { type: 'number', example: 67.8 },
                    diskUsagePercent: { type: 'number', example: 23.4 },
                    networkLatencyMs: { type: 'number', example: 12.5 },
                    requestsPerSecond: { type: 'number', example: 150.3 },
                    errorsPerMinute: { type: 'number', example: 0.2 },
                    timestamp: { type: 'string', format: 'date-time' }
                }
            },
            OperationsStatus: {
                type: 'object',
                properties: {
                    status: { type: 'string', enum: ['healthy', 'degraded', 'unhealthy'] },
                    services: {
                        type: 'object',
                        properties: {
                            api: { type: 'string' },
                            blockchain: { type: 'string' },
                            messaging: { type: 'string' },
                            database: { type: 'string' },
                            cache: { type: 'string' }
                        }
                    },
                    lastCheck: { type: 'string', format: 'date-time' },
                    uptime: { type: 'number' },
                    version: { type: 'string' }
                }
            },
            BlockchainStatus: {
                type: 'object',
                properties: {
                    connected: { type: 'boolean' },
                    network: { type: 'string' },
                    blockNumber: { type: 'integer' },
                    gasPrice: { type: 'string' },
                    chainId: { type: 'integer' },
                    contracts: { type: 'object' },
                    lastSync: { type: 'string', format: 'date-time' }
                }
            }
        };
    }

    /**
     * Generate common parameters
     */
    generateCommonParameters() {
        return {
            ApiVersion: {
                name: 'API-Version',
                in: 'header',
                description: 'API version to use for the request',
                required: false,
                schema: {
                    type: 'string',
                    enum: API_CONFIG.supportedVersions,
                    default: API_CONFIG.defaultVersion
                },
                example: API_CONFIG.currentVersion
            },
            AcceptVersion: {
                name: 'Accept-Version',
                in: 'header',
                description: 'Preferred API version (alternative to API-Version header)',
                required: false,
                schema: {
                    type: 'string',
                    enum: API_CONFIG.supportedVersions
                }
            }
        };
    }

    /**
     * Generate version information section
     */
    generateVersionInfo() {
        return {
            supported: API_CONFIG.supportedVersions,
            current: API_CONFIG.currentVersion,
            default: API_CONFIG.defaultVersion,
            deprecation: API_CONFIG.deprecationSchedule,
            features: API_CONFIG.featureFlags
        };
    }

    /**
     * Generate deprecation policy
     */
    generateDeprecationPolicy() {
        return {
            policy: 'Enterprise SAP API Deprecation Policy',
            timeline: {
                announcement: '6 months before deprecation',
                deprecation: '12 months support after deprecation announcement',
                sunset: 'No longer supported after sunset date'
            },
            notifications: [
                'Deprecation warnings in API responses',
                'Email notifications to registered developers',
                'Migration guides and tools provided',
                'Breaking changes documented with examples'
            ],
            support: {
                deprecated: 'Bug fixes only, no new features',
                sunset: 'No support, immediate migration required'
            }
        };
    }

    /**
     * Generate migration guides
     */
    async generateMigrationGuides() {
        const guides = {};

        for (const [version, changes] of Object.entries(BREAKING_CHANGES)) {
            guides[version] = {
                targetVersion: version,
                sourceVersion: changes.from,
                breakingChanges: changes.changes,
                migrationSteps: this.generateMigrationSteps(changes.from, version),
                automatedTools: [
                    'API version header updater',
                    'Response format converter',
                    'Breaking change detector'
                ],
                testingGuidance: [
                    'Update integration tests for new version',
                    'Validate all endpoint responses',
                    'Check error handling for new error formats',
                    'Performance test with new features'
                ],
                rollbackPlan: [
                    'Keep old version integration as backup',
                    'Monitor error rates after migration',
                    'Prepare rollback procedure',
                    'Document rollback decision criteria'
                ]
            };
        }

        return guides;
    }

    /**
     * Generate migration steps
     */
    generateMigrationSteps(fromVersion, toVersion) {
        const steps = [
            {
                step: 1,
                title: 'Review Migration Guide',
                description: `Study breaking changes from ${fromVersion} to ${toVersion}`,
                estimated: '1-2 hours'
            },
            {
                step: 2,
                title: 'Update Dependencies',
                description: 'Update client libraries and SDKs to support new version',
                estimated: '30 minutes'
            },
            {
                step: 3,
                title: 'Update API Headers',
                description: 'Change API-Version header in all requests',
                estimated: '15 minutes'
            },
            {
                step: 4,
                title: 'Handle Breaking Changes',
                description: 'Implement changes for removed/modified fields and endpoints',
                estimated: '2-8 hours'
            },
            {
                step: 5,
                title: 'Test Integration',
                description: 'Comprehensive testing in staging environment',
                estimated: '2-4 hours'
            },
            {
                step: 6,
                title: 'Deploy to Production',
                description: 'Deploy changes with monitoring and rollback plan',
                estimated: '1 hour'
            },
            {
                step: 7,
                title: 'Monitor and Validate',
                description: 'Monitor application health and API responses',
                estimated: '1-2 days'
            }
        ];

        return steps;
    }

    /**
     * Generate HTML documentation
     */
    async generateHTMLDocumentation() {
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A2A Network API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        .swagger-ui .topbar { display: none; }
        .version-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .deprecation-notice {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .migration-guide {
            background: #e7f3ff;
            border: 1px solid #b8daff;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script>
        SwaggerUIBundle({
            url: '/api/docs/openapi.json',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.presets.standalone
            ],
            plugins: [
                SwaggerUIBundle.plugins.DownloadUrl
            ],
            layout: "StandaloneLayout",
            defaultModelsExpandDepth: 1,
            defaultModelExpandDepth: 1,
            docExpansion: 'list',
            filter: true,
            showExtensions: true,
            showCommonExtensions: true,
            tryItOutEnabled: true
        });
    </script>
</body>
</html>`;

        return html;
    }

    /**
     * Save documentation files
     */
    async saveDocumentation(outputDir = './docs/api') {
        try {
            await fs.mkdir(outputDir, { recursive: true });

            // Generate and save OpenAPI spec
            const openApiSpec = await this.generateDocumentation();
            await fs.writeFile(
                path.join(outputDir, 'openapi.json'),
                JSON.stringify(openApiSpec, null, 2)
            );

            // Generate and save HTML documentation
            const htmlDoc = await this.generateHTMLDocumentation();
            await fs.writeFile(
                path.join(outputDir, 'index.html'),
                htmlDoc
            );

            // Generate version-specific docs
            for (const version of API_CONFIG.supportedVersions) {
                const versionDoc = await this.generateVersionSpecificDocs(version);
                await fs.writeFile(
                    path.join(outputDir, `v${version}.json`),
                    JSON.stringify(versionDoc, null, 2)
                );
            }

            logger.info(`✅ API documentation generated in ${outputDir}`);
            return outputDir;

        } catch (error) {
            logger.error('❌ Failed to save documentation:', { error: error });
            throw error;
        }
    }

    /**
     * Generate version-specific documentation
     */
    async generateVersionSpecificDocs(version) {
        const baseDoc = await this.generateDocumentation();
        const versionInfo = API_CONFIG.deprecationSchedule[version];
        const features = API_CONFIG.featureFlags[version];

        return {
            ...baseDoc,
            info: {
                ...baseDoc.info,
                version,
                description: `${baseDoc.info.description} - Version ${version}`,
                'x-version-status': versionInfo?.status || 'unknown',
                'x-deprecation-date': versionInfo?.deprecatedDate,
                'x-sunset-date': versionInfo?.sunsetDate,
                'x-migration-path': versionInfo?.migrationPath
            },
            'x-version-features': features,
            'x-breaking-changes': BREAKING_CHANGES[version] || null
        };
    }
}

module.exports = {
    APIDocumentationGenerator
};