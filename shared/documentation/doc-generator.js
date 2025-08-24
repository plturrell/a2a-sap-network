#!/usr/bin/env node
/**
 * Automated Documentation Generator for A2A Platform
 * Generates documentation from code comments, schemas, and API definitions
 */

const fs = require('fs').promises;
const path = require('path');
const glob = require('glob');
const { promisify } = require('util');

const globAsync = promisify(glob);

class A2ADocumentationGenerator {
    constructor(options = {}) {
        this.config = {
            rootDir: options.rootDir || process.cwd(),
            outputDir: options.outputDir || './docs/generated',
            includePatterns: options.includePatterns || [
                '**/*.py',
                '**/*.js',
                '**/*.ts',
                '**/*.json',
                '**/*.yml',
                '**/*.yaml'
            ],
            excludePatterns: options.excludePatterns || [
                '**/node_modules/**',
                '**/dist/**',
                '**/build/**',
                '**/.git/**',
                '**/coverage/**'
            ],
            generateApiDocs: options.generateApiDocs ?? true,
            generateAgentDocs: options.generateAgentDocs ?? true,
            generateSchemaDocs: options.generateSchemaDocs ?? true,
            generateConfigDocs: options.generateConfigDocs ?? true,
            ...options
        };
        
        this.logger = options.logger || console;
        this.docs = {
            api: [],
            agents: [],
            schemas: [],
            configs: [],
            overview: {}
        };
    }

    async generateAll() {
        this.logger.info('Starting A2A Platform documentation generation...');
        
        try {
            // Ensure output directory exists
            await fs.mkdir(this.config.outputDir, { recursive: true });

            // Collect all files
            const files = await this.collectFiles();
            this.logger.info(`Found ${files.length} files to process`);

            // Process files by type
            await this.processFiles(files);

            // Generate documentation files
            await this.generateDocumentationFiles();

            // Generate navigation index
            await this.generateNavigationIndex();

            this.logger.info(`Documentation generated successfully in: ${this.config.outputDir}`);
            return {
                success: true,
                outputDir: this.config.outputDir,
                filesProcessed: files.length,
                docsGenerated: this.getDocumentationStats()
            };

        } catch (error) {
            this.logger.error('Documentation generation failed:', error);
            throw error;
        }
    }

    async collectFiles() {
        const allFiles = [];
        
        for (const pattern of this.config.includePatterns) {
            const files = await globAsync(pattern, {
                cwd: this.config.rootDir,
                ignore: this.config.excludePatterns
            });
            allFiles.push(...files.map(f => path.resolve(this.config.rootDir, f)));
        }

        // Remove duplicates
        return [...new Set(allFiles)];
    }

    async processFiles(files) {
        for (const file of files) {
            try {
                await this.processFile(file);
            } catch (error) {
                this.logger.warn(`Failed to process file ${file}:`, error.message);
            }
        }
    }

    async processFile(filePath) {
        const content = await fs.readFile(filePath, 'utf8');
        const relativePath = path.relative(this.config.rootDir, filePath);
        const ext = path.extname(filePath);
        
        // Process based on file type and content
        if (this.isApiFile(filePath, content)) {
            await this.extractApiDocumentation(filePath, content, relativePath);
        }

        if (this.isAgentFile(filePath, content)) {
            await this.extractAgentDocumentation(filePath, content, relativePath);
        }

        if (this.isSchemaFile(filePath, content)) {
            await this.extractSchemaDocumentation(filePath, content, relativePath);
        }

        if (this.isConfigFile(filePath, content)) {
            await this.extractConfigDocumentation(filePath, content, relativePath);
        }
    }

    isApiFile(filePath, content) {
        return filePath.includes('/api/') ||
               content.includes('@api') ||
               content.includes('openapi') ||
               content.includes('swagger') ||
               filePath.includes('openapi') ||
               filePath.includes('service');
    }

    isAgentFile(filePath, content) {
        return filePath.includes('/agents/') ||
               content.includes('class.*Agent') ||
               content.includes('agent_id') ||
               filePath.includes('Agent');
    }

    isSchemaFile(filePath, content) {
        return filePath.includes('/schemas/') ||
               filePath.includes('/models/') ||
               content.includes('"$schema"') ||
               (filePath.endsWith('.json') && (content.includes('"type"') || content.includes('"properties"')));
    }

    isConfigFile(filePath, content) {
        return filePath.includes('/config/') ||
               filePath.endsWith('.yml') ||
               filePath.endsWith('.yaml') ||
               filePath.includes('docker-compose') ||
               filePath.includes('prometheus') ||
               filePath.includes('package.json') ||
               filePath.includes('mta.yaml');
    }

    async extractApiDocumentation(filePath, content, relativePath) {
        if (!this.config.generateApiDocs) return;

        const apiDoc = {
            file: relativePath,
            path: filePath,
            type: 'api',
            title: this.extractTitle(content, path.basename(filePath)),
            description: this.extractDescription(content),
            endpoints: [],
            schemas: [],
            lastModified: await this.getFileModifiedDate(filePath)
        };

        // Extract endpoints from different formats
        if (content.includes('openapi') || content.includes('swagger')) {
            apiDoc.endpoints = this.extractOpenApiEndpoints(content);
            apiDoc.schemas = this.extractOpenApiSchemas(content);
        } else if (filePath.endsWith('.js') || filePath.endsWith('.ts')) {
            apiDoc.endpoints = this.extractJSEndpoints(content);
        } else if (filePath.endsWith('.py')) {
            apiDoc.endpoints = this.extractPythonEndpoints(content);
        }

        this.docs.api.push(apiDoc);
    }

    async extractAgentDocumentation(filePath, content, relativePath) {
        if (!this.config.generateAgentDocs) return;

        const agentDoc = {
            file: relativePath,
            path: filePath,
            type: 'agent',
            title: this.extractTitle(content, path.basename(filePath)),
            description: this.extractDescription(content),
            agentId: this.extractAgentId(content),
            capabilities: this.extractCapabilities(content),
            methods: this.extractMethods(content),
            configuration: this.extractConfiguration(content),
            lastModified: await this.getFileModifiedDate(filePath)
        };

        this.docs.agents.push(agentDoc);
    }

    async extractSchemaDocumentation(filePath, content, relativePath) {
        if (!this.config.generateSchemaDocs) return;

        const schemaDoc = {
            file: relativePath,
            path: filePath,
            type: 'schema',
            title: this.extractTitle(content, path.basename(filePath)),
            description: this.extractDescription(content),
            schema: null,
            properties: [],
            lastModified: await this.getFileModifiedDate(filePath)
        };

        try {
            if (filePath.endsWith('.json')) {
                const parsed = JSON.parse(content);
                schemaDoc.schema = parsed;
                schemaDoc.properties = this.extractJsonSchemaProperties(parsed);
            } else {
                // Extract from code comments
                schemaDoc.properties = this.extractSchemaFromComments(content);
            }
        } catch (error) {
            this.logger.warn(`Failed to parse schema in ${filePath}:`, error.message);
        }

        this.docs.schemas.push(schemaDoc);
    }

    async extractConfigDocumentation(filePath, content, relativePath) {
        if (!this.config.generateConfigDocs) return;

        const configDoc = {
            file: relativePath,
            path: filePath,
            type: 'config',
            title: this.extractTitle(content, path.basename(filePath)),
            description: this.extractDescription(content),
            configuration: {},
            parameters: [],
            lastModified: await this.getFileModifiedDate(filePath)
        };

        try {
            if (filePath.endsWith('.json')) {
                configDoc.configuration = JSON.parse(content);
            } else if (filePath.endsWith('.yml') || filePath.endsWith('.yaml')) {
                // Simple YAML parsing for documentation purposes
                configDoc.parameters = this.extractYamlParameters(content);
            }
        } catch (error) {
            this.logger.warn(`Failed to parse config in ${filePath}:`, error.message);
        }

        this.docs.configs.push(configDoc);
    }

    // Extraction helper methods
    extractTitle(content, filename) {
        // Look for title in comments or docstrings
        const titlePatterns = [
            /(?:^|\n)\s*#\s*([A-Z][^#\n]*)/m,
            /(?:^|\n)\s*\/\*\*?\s*\n\s*\*\s*([A-Z][^*\n]*)/m,
            /(?:^|\n)\s*"""\s*\n\s*([A-Z][^"\n]*)/m,
            /title:\s*["']([^"']+)["']/i
        ];

        for (const pattern of titlePatterns) {
            const match = content.match(pattern);
            if (match) {
                return match[1].trim();
            }
        }

        // Fallback to filename
        return filename.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ');
    }

    extractDescription(content) {
        // Look for description in comments or docstrings
        const descPatterns = [
            /(?:^|\n)\s*"""([\s\S]*?)"""/m,
            /(?:^|\n)\s*\/\*\*([\s\S]*?)\*\//m,
            /description:\s*["']([^"']+)["']/i
        ];

        for (const pattern of descPatterns) {
            const match = content.match(pattern);
            if (match) {
                return match[1].trim().replace(/^\s*[*#]\s?/gm, '').trim();
            }
        }

        return '';
    }

    extractAgentId(content) {
        const patterns = [
            /agent_id\s*[=:]\s*["']([^"']+)["']/i,
            /AGENT_ID\s*=\s*["']([^"']+)["']/i,
            /class\s+(\w*Agent)/i
        ];

        for (const pattern of patterns) {
            const match = content.match(pattern);
            if (match) {
                return match[1];
            }
        }

        return 'unknown';
    }

    extractCapabilities(content) {
        const capabilities = [];
        const lines = content.split('\n');
        
        for (const line of lines) {
            if (line.includes('capability') || line.includes('@capability')) {
                const capability = line.trim().replace(/[#*\/]*\s*@?capability:?\s*/i, '');
                if (capability) capabilities.push(capability);
            }
        }

        return capabilities;
    }

    extractMethods(content) {
        const methods = [];
        const patterns = [
            /(?:async\s+)?def\s+(\w+)\s*\([^)]*\):\s*\n\s*"""([^"]*)/g,
            /(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*{[\s\S]*?\/\*\*([^*]*)/g,
            /(\w+)\s*:\s*(?:async\s+)?function\s*\([^)]*\)\s*{[\s\S]*?\/\*\*([^*]*)/g
        ];

        for (const pattern of patterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                methods.push({
                    name: match[1],
                    description: match[2] ? match[2].trim() : '',
                    type: 'method'
                });
            }
        }

        return methods;
    }

    extractConfiguration(content) {
        const config = {};
        const lines = content.split('\n');
        
        for (const line of lines) {
            const configMatch = line.match(/(\w+)\s*[=:]\s*(.+)/);
            if (configMatch && !line.trim().startsWith('#') && !line.trim().startsWith('//')) {
                config[configMatch[1]] = configMatch[2].trim();
            }
        }

        return config;
    }

    // Generate documentation files
    async generateDocumentationFiles() {
        if (this.docs.api.length > 0) {
            await this.generateApiDocs();
        }

        if (this.docs.agents.length > 0) {
            await this.generateAgentDocs();
        }

        if (this.docs.schemas.length > 0) {
            await this.generateSchemaDocs();
        }

        if (this.docs.configs.length > 0) {
            await this.generateConfigDocs();
        }
    }

    async generateApiDocs() {
        const content = this.generateMarkdownContent('API Documentation', this.docs.api, (doc) => `
## ${doc.title}
**File:** \`${doc.file}\`

${doc.description}

### Endpoints
${doc.endpoints.map(endpoint => `
#### ${endpoint.method?.toUpperCase() || 'GET'} ${endpoint.path || endpoint.route}
${endpoint.description || ''}

${endpoint.parameters ? `**Parameters:**
${endpoint.parameters.map(p => `- \`${p.name}\` (${p.type}): ${p.description}`).join('\n')}` : ''}

${endpoint.responses ? `**Responses:**
${endpoint.responses.map(r => `- ${r.status}: ${r.description}`).join('\n')}` : ''}
`).join('\n')}

### Schemas
${doc.schemas.map(schema => `
#### ${schema.name}
${schema.description || ''}
\`\`\`json
${JSON.stringify(schema.definition, null, 2)}
\`\`\`
`).join('\n')}

---
        `);

        await fs.writeFile(path.join(this.config.outputDir, 'api-documentation.md'), content);
    }

    async generateAgentDocs() {
        const content = this.generateMarkdownContent('Agent Documentation', this.docs.agents, (doc) => `
## ${doc.title}
**File:** \`${doc.file}\`  
**Agent ID:** \`${doc.agentId}\`

${doc.description}

### Capabilities
${doc.capabilities.map(cap => `- ${cap}`).join('\n')}

### Methods
${doc.methods.map(method => `
#### ${method.name}()
${method.description}
`).join('\n')}

### Configuration
\`\`\`yaml
${Object.entries(doc.configuration).map(([key, value]) => `${key}: ${value}`).join('\n')}
\`\`\`

---
        `);

        await fs.writeFile(path.join(this.config.outputDir, 'agent-documentation.md'), content);
    }

    async generateSchemaDocs() {
        const content = this.generateMarkdownContent('Schema Documentation', this.docs.schemas, (doc) => `
## ${doc.title}
**File:** \`${doc.file}\`

${doc.description}

### Properties
${doc.properties.map(prop => `
#### ${prop.name}
- **Type:** ${prop.type}
- **Required:** ${prop.required ? 'Yes' : 'No'}
${prop.description ? `- **Description:** ${prop.description}` : ''}
${prop.default ? `- **Default:** \`${prop.default}\`` : ''}
`).join('\n')}

${doc.schema ? `### Full Schema
\`\`\`json
${JSON.stringify(doc.schema, null, 2)}
\`\`\`` : ''}

---
        `);

        await fs.writeFile(path.join(this.config.outputDir, 'schema-documentation.md'), content);
    }

    async generateConfigDocs() {
        const content = this.generateMarkdownContent('Configuration Documentation', this.docs.configs, (doc) => `
## ${doc.title}
**File:** \`${doc.file}\`

${doc.description}

### Parameters
${doc.parameters.map(param => `
#### ${param.name}
${param.description ? `${param.description}` : ''}
${param.type ? `**Type:** ${param.type}` : ''}
${param.default ? `**Default:** \`${param.default}\`` : ''}
${param.required ? '**Required:** Yes' : ''}
`).join('\n')}

${Object.keys(doc.configuration).length > 0 ? `### Configuration Example
\`\`\`yaml
${JSON.stringify(doc.configuration, null, 2)}
\`\`\`` : ''}

---
        `);

        await fs.writeFile(path.join(this.config.outputDir, 'config-documentation.md'), content);
    }

    generateMarkdownContent(title, docs, formatDoc) {
        const timestamp = new Date().toISOString();
        return `# ${title}

> Generated automatically on ${timestamp}

${docs.map(formatDoc).join('\n')}

## Summary
- **Total ${title.toLowerCase()}:** ${docs.length}
- **Last generated:** ${timestamp}
`;
    }

    async generateNavigationIndex() {
        const indexContent = `# A2A Platform Documentation

> Auto-generated documentation for the A2A Platform

## Documentation Sections

### 📚 API Documentation
- [API Documentation](./api-documentation.md) - ${this.docs.api.length} APIs documented

### 🤖 Agent Documentation  
- [Agent Documentation](./agent-documentation.md) - ${this.docs.agents.length} agents documented

### 📋 Schema Documentation
- [Schema Documentation](./schema-documentation.md) - ${this.docs.schemas.length} schemas documented

### ⚙️ Configuration Documentation
- [Configuration Documentation](./config-documentation.md) - ${this.docs.configs.length} configurations documented

## Generation Statistics
- **Total files processed:** ${this.getDocumentationStats().totalFiles}
- **Generated on:** ${new Date().toISOString()}
- **Documentation types:** ${Object.keys(this.docs).filter(key => Array.isArray(this.docs[key]) && this.docs[key].length > 0).length}

## File Coverage
${Object.entries(this.getDocumentationStats()).map(([key, value]) => `- **${key}:** ${value} files`).join('\n')}
`;

        await fs.writeFile(path.join(this.config.outputDir, 'README.md'), indexContent);
    }

    getDocumentationStats() {
        return {
            totalFiles: Object.values(this.docs).reduce((sum, docs) => sum + (Array.isArray(docs) ? docs.length : 0), 0),
            api: this.docs.api.length,
            agents: this.docs.agents.length,
            schemas: this.docs.schemas.length,
            configs: this.docs.configs.length
        };
    }

    async getFileModifiedDate(filePath) {
        try {
            const stats = await fs.stat(filePath);
            return stats.mtime.toISOString();
        } catch {
            return new Date().toISOString();
        }
    }

    // Additional helper methods for different formats
    extractOpenApiEndpoints(content) {
        try {
            const spec = JSON.parse(content);
            const endpoints = [];
            
            if (spec.paths) {
                for (const [path, methods] of Object.entries(spec.paths)) {
                    for (const [method, details] of Object.entries(methods)) {
                        endpoints.push({
                            path,
                            method,
                            description: details.description || details.summary,
                            parameters: details.parameters || [],
                            responses: Object.entries(details.responses || {}).map(([status, resp]) => ({
                                status,
                                description: resp.description
                            }))
                        });
                    }
                }
            }
            
            return endpoints;
        } catch {
            return [];
        }
    }

    extractOpenApiSchemas(content) {
        try {
            const spec = JSON.parse(content);
            const schemas = [];
            
            if (spec.components && spec.components.schemas) {
                for (const [name, definition] of Object.entries(spec.components.schemas)) {
                    schemas.push({
                        name,
                        description: definition.description,
                        definition
                    });
                }
            }
            
            return schemas;
        } catch {
            return [];
        }
    }

    extractJSEndpoints(content) {
        const endpoints = [];
        const patterns = [
            /(?:router|app)\.(get|post|put|delete|patch)\s*\(\s*['"`]([^'"`]+)['"`]/g,
            /@(Get|Post|Put|Delete|Patch)\s*\(\s*['"`]([^'"`]+)['"`]/g
        ];

        for (const pattern of patterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                endpoints.push({
                    method: match[1].toLowerCase(),
                    path: match[2],
                    description: ''
                });
            }
        }

        return endpoints;
    }

    extractPythonEndpoints(content) {
        const endpoints = [];
        const patterns = [
            /@app\.route\s*\(\s*['"`]([^'"`]+)['"`].*?methods\s*=\s*\[['"`]([^'"`]+)['"`]\]/g,
            /@(get|post|put|delete|patch)\s*\(\s*['"`]([^'"`]+)['"`]/g
        ];

        for (const pattern of patterns) {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                if (match[2]) {
                    endpoints.push({
                        method: match[2].toLowerCase(),
                        path: match[1],
                        description: ''
                    });
                } else {
                    endpoints.push({
                        method: match[1].toLowerCase(),
                        path: match[2] || match[1],
                        description: ''
                    });
                }
            }
        }

        return endpoints;
    }

    extractJsonSchemaProperties(schema) {
        const properties = [];
        
        if (schema.properties) {
            for (const [name, prop] of Object.entries(schema.properties)) {
                properties.push({
                    name,
                    type: prop.type || 'unknown',
                    description: prop.description || '',
                    required: schema.required ? schema.required.includes(name) : false,
                    default: prop.default
                });
            }
        }

        return properties;
    }

    extractSchemaFromComments(content) {
        const properties = [];
        const lines = content.split('\n');
        
        for (const line of lines) {
            const propMatch = line.match(/@property\s+(\w+)\s+(\w+)\s*(.*)/);
            if (propMatch) {
                properties.push({
                    name: propMatch[2],
                    type: propMatch[1],
                    description: propMatch[3].trim(),
                    required: line.includes('required'),
                    default: null
                });
            }
        }

        return properties;
    }

    extractYamlParameters(content) {
        const parameters = [];
        const lines = content.split('\n');
        let currentParam = null;
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('#')) continue;
            
            const keyMatch = trimmed.match(/^([^:]+):\s*(.*)/);
            if (keyMatch) {
                if (currentParam) {
                    parameters.push(currentParam);
                }
                currentParam = {
                    name: keyMatch[1],
                    type: this.inferYamlType(keyMatch[2]),
                    default: keyMatch[2] || null,
                    description: '',
                    required: false
                };
            }
        }
        
        if (currentParam) {
            parameters.push(currentParam);
        }

        return parameters;
    }

    inferYamlType(value) {
        if (!value) return 'string';
        if (value === 'true' || value === 'false') return 'boolean';
        if (/^\d+$/.test(value)) return 'integer';
        if (/^\d*\.\d+$/.test(value)) return 'number';
        if (value.startsWith('[') && value.endsWith(']')) return 'array';
        if (value.startsWith('{') && value.endsWith('}')) return 'object';
        return 'string';
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    const options = {
        rootDir: args[0] || process.cwd(),
        outputDir: args[1] || './docs/generated'
    };

    try {
        const generator = new A2ADocumentationGenerator(options);
        const result = await generator.generateAll();
        
        console.log('\n✅ Documentation generation completed successfully!');
        console.log(`📁 Output directory: ${result.outputDir}`);
        console.log(`📄 Files processed: ${result.filesProcessed}`);
        console.log(`📚 Documentation generated: ${JSON.stringify(result.docsGenerated, null, 2)}`);
        
        process.exit(0);
    } catch (error) {
        console.error('\n❌ Documentation generation failed:', error.message);
        process.exit(1);
    }
}

// Export for use as module
module.exports = A2ADocumentationGenerator;

// Run as CLI if called directly
if (require.main === module) {
    main();
}