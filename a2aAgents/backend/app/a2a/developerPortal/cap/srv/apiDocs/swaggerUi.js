"use strict";
/* global SwaggerUIBundle, SwaggerUIStandalonePreset */

/**
 * Swagger UI Integration for SAP A2A Developer Portal
 * Provides interactive API documentation
 */

const express = require('express');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const path = require('path');

class SwaggerUIService {
    constructor() {
        this.router = express.Router();
        this.swaggerDocument = null;
    }

    /**
     * Initialize Swagger UI with custom SAP styling
     */
    initialize() {
        // Load OpenAPI specification
        this.swaggerDocument = YAML.load(path.join(__dirname, 'openapi.yaml'));

        // Custom CSS for SAP Fiori styling
        const customCss = `
            .swagger-ui .topbar { 
                display: none; 
            }
            .swagger-ui {
                font-family: "72", "72full", Arial, Helvetica, sans-serif;
            }
            .swagger-ui .info .title {
                color: #0a6ed1;
            }
            .swagger-ui .btn.authorize {
                background-color: #0a6ed1;
                border-color: #0a6ed1;
            }
            .swagger-ui .btn.authorize:hover {
                background-color: #0854a0;
                border-color: #0854a0;
            }
            .swagger-ui .opblock.opblock-post {
                border-color: #107e3e;
                background: rgba(16, 126, 62, 0.1);
            }
            .swagger-ui .opblock.opblock-post .opblock-summary {
                border-color: #107e3e;
            }
            .swagger-ui .opblock.opblock-get {
                border-color: #0a6ed1;
                background: rgba(10, 110, 209, 0.1);
            }
            .swagger-ui .opblock.opblock-get .opblock-summary {
                border-color: #0a6ed1;
            }
            .swagger-ui .opblock.opblock-put {
                border-color: #e9730c;
                background: rgba(233, 115, 12, 0.1);
            }
            .swagger-ui .opblock.opblock-put .opblock-summary {
                border-color: #e9730c;
            }
            .swagger-ui .opblock.opblock-delete {
                border-color: #bb0000;
                background: rgba(187, 0, 0, 0.1);
            }
            .swagger-ui .opblock.opblock-delete .opblock-summary {
                border-color: #bb0000;
            }
            .swagger-ui select, .swagger-ui input[type=text], .swagger-ui textarea {
                border: 1px solid #89919a;
                border-radius: 4px;
            }
            .swagger-ui select:focus, .swagger-ui input[type=text]:focus, .swagger-ui textarea:focus {
                border-color: #0a6ed1;
                outline: none;
                box-shadow: 0 0 0 1px #0a6ed1;
            }
            .info-container {
                background-color: #f7f7f7;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .servers-container {
                background-color: #fff;
                padding: 15px;
                border: 1px solid #e5e5e5;
                border-radius: 4px;
                margin-bottom: 20px;
            }
        `;

        // Custom favicon
        const customfavIcon = '/assets/favicon.ico';

        // Custom site title
        const customSiteTitle = 'A2A Developer Portal - API Documentation';

        // Swagger UI options
        const swaggerOptions = {
            customCss,
            customfavIcon,
            customSiteTitle,
            swaggerOptions: {
                docExpansion: 'none',
                filter: true,
                showRequestHeaders: true,
                showCommonExtensions: true,
                showExtensions: true,
                tryItOutEnabled: true,
                requestInterceptor: (request) => {
                    // Add correlation ID to all requests
                    request.headers['X-Correlation-Id'] = this._generateCorrelationId();
                    request.headers['X-Request-Timestamp'] = new Date().toISOString();
                    return request;
                },
                responseInterceptor: (response) => {
                    // Log API calls for monitoring
                    // eslint-disable-next-line no-console
                    console.log({
                        type: 'api_documentation_request',
                        method: response.method,
                        url: response.url,
                        status: response.status,
                        duration: response.duration,
                        timestamp: new Date().toISOString()
                    });
                    return response;
                },
                onComplete: () => {
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('Swagger UI loaded successfully');
                },
                plugins: [
                    () => {
                        return {
                            wrapComponents: {
                                InfoContainer: (Original, system) => (props) => {
                                    return system.React.createElement(
                                        'div',
                                        { className: 'info-container' },
                                        system.React.createElement(Original, props)
                                    );
                                }
                            }
                        };
                    }
                ],
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ].filter(Boolean),
                deepLinking: true,
                displayOperationId: true,
                defaultModelsExpandDepth: 1,
                defaultModelExpandDepth: 1,
                showAlternativeSchemaExample: true,
                syntaxHighlight: {
                    activate: true,
                    theme: 'agate'
                },
                validatorUrl: null, // Disable validator
                supportedSubmitMethods: ['get', 'post', 'put', 'delete', 'patch'],
                onFailure: (error) => {
                    console.error('Swagger UI initialization failed:', error);
                }
            }
        };

        // Setup routes
        this.router.use('/', swaggerUi.serve);
        this.router.get('/', swaggerUi.setup(this.swaggerDocument, swaggerOptions));

        // Serve OpenAPI spec as JSON
        this.router.get('/openapi.json', (req, res) => {
            res.json(this.swaggerDocument);
        });

        // Serve OpenAPI spec as YAML
        this.router.get('/openapi.yaml', (req, res) => {
            res.type('text/yaml');
            res.sendFile(path.join(__dirname, 'openapi.yaml'));
        });

        // API documentation home page
        this.router.get('/home', (req, res) => {
            res.send(this._getHomePage());
        });

        return this.router;
    }

    /**
     * Generate correlation ID for request tracking
     */
    _generateCorrelationId() {
        return `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Generate API documentation home page
     */
    _getHomePage() {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A2A Developer Portal - API Documentation</title>
    <style>
        body {
            font-family: "72", "72full", Arial, Helvetica, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f7f7f7;
        }
        .header {
            background-color: #354a5f;
            color: white;
            padding: 20px 40px;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: normal;
        }
        .container {
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 40px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 20px;
        }
        .card h2 {
            color: #0a6ed1;
            margin-top: 0;
        }
        .button {
            display: inline-block;
            background-color: #0a6ed1;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            margin-right: 10px;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #0854a0;
        }
        .button.secondary {
            background-color: #89919a;
        }
        .button.secondary:hover {
            background-color: #6c7680;
        }
        .endpoint-list {
            list-style: none;
            padding: 0;
        }
        .endpoint-list li {
            padding: 10px;
            border-bottom: 1px solid #e5e5e5;
        }
        .endpoint-list li:last-child {
            border-bottom: none;
        }
        .method {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
            min-width: 60px;
            text-align: center;
        }
        .method.get { background-color: #0a6ed1; color: white; }
        .method.post { background-color: #107e3e; color: white; }
        .method.put { background-color: #e9730c; color: white; }
        .method.delete { background-color: #bb0000; color: white; }
        .method.patch { background-color: #594300; color: white; }
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: Monaco, Consolas, monospace;
            font-size: 14px;
        }
        .info-box {
            background-color: #e9f4ff;
            border-left: 4px solid #0a6ed1;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>A2A Developer Portal - API Documentation</h1>
    </div>
    <div class="container">
        <div class="card">
            <h2>Welcome to the A2A Developer Portal API</h2>
            <p>The A2A Developer Portal provides a comprehensive REST API for managing multi-agent systems, workflows, and deployments on the SAP Business Technology Platform.</p>
            <div class="info-box">
                <strong>Base URL:</strong> <code>https://a2a-portal.cfapps.sap.hana.ondemand.com/api/v1</code><br>
                <strong>API Version:</strong> 2.1.0<br>
                <strong>Authentication:</strong> SAP XSUAA OAuth 2.0
            </div>
            <a href="/api-docs" class="button">Interactive API Documentation</a>
            <a href="/api-docs/openapi.json" class="button secondary">Download OpenAPI (JSON)</a>
            <a href="/api-docs/openapi.yaml" class="button secondary">Download OpenAPI (YAML)</a>
        </div>

        <div class="card">
            <h2>Key API Endpoints</h2>
            <ul class="endpoint-list">
                <li>
                    <span class="method get">GET</span>
                    <code>/projects</code> - List all projects
                </li>
                <li>
                    <span class="method post">POST</span>
                    <code>/projects</code> - Create a new project
                </li>
                <li>
                    <span class="method get">GET</span>
                    <code>/projects/{projectId}/agents</code> - List project agents
                </li>
                <li>
                    <span class="method post">POST</span>
                    <code>/agents/{agentId}/execute</code> - Execute an agent
                </li>
                <li>
                    <span class="method post">POST</span>
                    <code>/workflows/{workflowId}/execute</code> - Execute a workflow
                </li>
                <li>
                    <span class="method get">GET</span>
                    <code>/monitoring/metrics</code> - Get system metrics
                </li>
            </ul>
        </div>

        <div class="card">
            <h2>Authentication</h2>
            <p>All API requests must include a valid OAuth 2.0 Bearer token in the Authorization header:</p>
            <code>Authorization: Bearer &lt;your-access-token&gt;</code>
            <p style="margin-top: 15px;">To obtain an access token, use the SAP XSUAA OAuth 2.0 authorization flow with the following endpoints:</p>
            <ul>
                <li><strong>Authorization URL:</strong> <code>https://auth.sap.com/oauth/authorize</code></li>
                <li><strong>Token URL:</strong> <code>https://auth.sap.com/oauth/token</code></li>
            </ul>
        </div>

        <div class="card">
            <h2>Rate Limiting</h2>
            <p>API calls are limited to <strong>100 requests per 15 minutes</strong> per IP address. Rate limit information is included in response headers:</p>
            <ul>
                <li><code>X-RateLimit-Limit</code> - Request limit per window</li>
                <li><code>X-RateLimit-Remaining</code> - Remaining requests in current window</li>
                <li><code>X-RateLimit-Reset</code> - Time when the rate limit resets</li>
            </ul>
        </div>

        <div class="card">
            <h2>SDK & Code Examples</h2>
            <p>We provide SDKs and code examples in multiple languages:</p>
            <ul>
                <li><a href="https://github.com/sap/a2a-portal-sdk-js">JavaScript/TypeScript SDK</a></li>
                <li><a href="https://github.com/sap/a2a-portal-sdk-python">Python SDK</a></li>
                <li><a href="https://github.com/sap/a2a-portal-sdk-java">Java SDK</a></li>
            </ul>
        </div>

        <div class="card">
            <h2>Support & Resources</h2>
            <ul>
                <li><a href="https://answers.sap.com/tags/a2a-developer-portal">SAP Community Q&A</a></li>
                <li><a href="https://support.sap.com/a2a-portal">Support Portal</a></li>
                <li><a href="mailto:a2a-support@sap.com">Contact Support</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
        `;
    }
}

module.exports = new SwaggerUIService();