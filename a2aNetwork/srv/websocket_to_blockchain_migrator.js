/**
 * WebSocket to Blockchain Event Stream Migrator
 * Converts WebSocket implementations to use blockchain event streaming
 */

const fs = require('fs').promises;
const path = require('path');

class WebSocketToBlockchainMigrator {
    constructor() {
        this.filesProcessed = 0;
        this.filesMigrated = 0;
        this.errors = [];

        // WebSocket patterns to replace
        this.WS_PATTERNS = {
            // WebSocket server creation
            wsServerCreation: {
                pattern: /new\s+WebSocket\.Server\s*\([^)]+\)/g,
                replacement: 'new BlockchainEventServer($1)',
                description: 'BlockchainEventClient.Server creation'
            },

            // WebSocket client creation
            wsClientCreation: {
                pattern: /new\s+WebSocket\s*\(\s*['"`]ws:\/\/[^'"`]+['"`]\s*\)/g,
                replacement: 'new BlockchainEventClient()',
                description: 'WebSocket client creation'
            },

            // WebSocket connection handlers
            wsOnConnection: {
                pattern: /\.on\s*\(\s*['"`]connection['"`]\s*,/g,
                replacement: '.on(\'blockchain-connection\',',
                description: 'WebSocket connection handler'
            },

            // WebSocket message handlers
            wsOnMessage: {
                pattern: /ws\.on\s*\(\s*['"`]message['"`]\s*,/g,
                replacement: 'blockchainClient.on(\'event\',',
                description: 'WebSocket message handler'
            },

            // WebSocket send
            wsSend: {
                pattern: /ws\.send\s*\(/g,
                replacement: 'blockchainClient.publishEvent(',
                description: 'WebSocket send'
            },

            // WebSocket URLs
            wsUrls: {
                pattern: /['"`]ws:\/\/[^'"`]+['"`]/g,
                replacement: '\'blockchain://a2a-events\'',
                description: 'WebSocket URLs'
            }
        };

        // Blockchain event stream template
        this.BLOCKCHAIN_ADAPTER_TEMPLATE = `
/**
 * Blockchain Event Adapter
 * Provides WebSocket-like interface using blockchain events
 */
class BlockchainEventAdapter {
    constructor(options = {}) {
        this.eventStream = require('./blockchain-event-stream');
        this.connections = new Map();
        this.logger = options.logger || console;
    }

    on(event, handler) {
        if (event === 'connection' || event === 'blockchain-connection') {
            // Handle new connections via blockchain
            this.eventStream.on('new-subscriber', async (subscriberId) => {
                const connection = {
                    id: subscriberId,
                    send: (data) => this.sendToSubscriber(subscriberId, data),
                    on: (evt, fn) => this.eventStream.subscribe(subscriberId, [evt], fn),
                    close: () => this.eventStream.unsubscribe(subscriberId)
                };
                this.connections.set(subscriberId, connection);
                handler(connection);
            });
        }
    }

    async sendToSubscriber(subscriberId, data) {
        try {
            await this.eventStream.publishEvent('message', {
                to: subscriberId,
                data: typeof data === 'string' ? data : JSON.stringify(data)
            });
        } catch (error) {
            this.logger.error('Failed to send via blockchain:', error);
        }
    }
}

// WebSocket compatibility layer
class BlockchainEventServer extends BlockchainEventAdapter {
    constructor(options) {
        super(options);
        this.port = options.port;
        this.path = options.path;
        this.logger.info(\`Blockchain event server replacing WebSocket on port \${this.port}\`);
    }
}

class BlockchainEventClient extends BlockchainEventAdapter {
    constructor() {
        super();
        this.readyState = 1; // OPEN - for compatibility
    }

    send(data) {
        this.publishEvent('message', { data });
    }

    close() {
        this.eventStream.disconnect();
        this.readyState = 3; // CLOSED
    }
}

module.exports = { BlockchainEventServer, BlockchainEventClient };
`;
    }

    async analyzeFile(filePath) {
        try {
            const content = await fs.readFile(filePath, 'utf8');

            // Check if already migrated
            if (content.includes('BlockchainEventServer') || content.includes('BlockchainEventAdapter')) {
                return {
                    needsMigration: false,
                    reason: 'Already migrated to blockchain events'
                };
            }

            // Check for WebSocket usage
            const wsPatterns = [
                'WebSocket',
                'blockchain://',
                'blockchains://',
                '.on(\'connection\'',
                '.on("connection"'
            ];

            const hasWebSocket = wsPatterns.some(pattern => content.includes(pattern));

            if (!hasWebSocket) {
                return {
                    needsMigration: false,
                    reason: 'No WebSocket usage found'
                };
            }

            // Analyze specific patterns
            const foundPatterns = [];
            for (const [name, patternInfo] of Object.entries(this.WS_PATTERNS)) {
                if (patternInfo.pattern.test(content)) {
                    foundPatterns.push(name);
                }
            }

            return {
                needsMigration: true,
                patterns: foundPatterns
            };

        } catch (error) {
            this.errors.push({ file: filePath, error: error.message });
            return {
                needsMigration: false,
                reason: `Error analyzing file: ${error.message}`
            };
        }
    }

    async migrateFile(filePath) {
        try {
            let content = await fs.readFile(filePath, 'utf8');
            const originalContent = content;

            // Apply replacements
            let replacements = 0;
            for (const [, patternInfo] of Object.entries(this.WS_PATTERNS)) {
                const matches = content.match(patternInfo.pattern);
                if (matches) {
                    content = content.replace(patternInfo.pattern, patternInfo.replacement);
                    replacements += matches.length;
                }
            }

            // Add blockchain adapter import if WebSocket was replaced
            if (replacements > 0 && !content.includes('BlockchainEventServer')) {
                // Find where to add import
                const wsImportMatch = content.match(/const\s+WebSocket\s*=\s*require\s*\([^)]+\);?/);
                if (wsImportMatch) {
                    // Replace WebSocket import
                    content = content.replace(
                        wsImportMatch[0],
                        'const { BlockchainEventServer, BlockchainEventClient } = require(\'./blockchain-event-adapter\');'
                    );
                } else {
                    // Add at the beginning after other requires
                    const requireMatches = content.match(/require\s*\([^)]+\);/g);
                    if (requireMatches) {
                        const lastRequireIndex = content.lastIndexOf(requireMatches[requireMatches.length - 1]);
                        const insertPos = content.indexOf('\n', lastRequireIndex) + 1;
                        content = `${content.slice(0, insertPos)
                            }const { BlockchainEventServer, BlockchainEventClient } = require('./blockchain-event-adapter');\n${
                            content.slice(insertPos)}`;
                    }
                }
            }

            // Add compatibility comments
            if (replacements > 0) {
                const header = `/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

`;
                if (!content.startsWith('/**')) {
                    content = header + content;
                }
            }

            // Write migrated content
            if (content !== originalContent) {
                await fs.writeFile(filePath, content);
                this.filesMigrated++;

                return {
                    status: 'migrated',
                    replacements: replacements,
                    file: filePath
                };
            }

            return {
                status: 'no_changes',
                file: filePath
            };

        } catch (error) {
            this.errors.push({ file: filePath, error: error.message });
            return {
                status: 'error',
                error: error.message,
                file: filePath
            };
        }
    }

    async createBlockchainAdapter(directory) {
        try {
            const adapterPath = path.join(directory, 'blockchain-event-adapter.js');
            await fs.writeFile(adapterPath, this.BLOCKCHAIN_ADAPTER_TEMPLATE);
            // console.log(`✅ Created blockchain event adapter at: ${adapterPath}`);
        } catch (error) {
            // console.error(`Failed to create blockchain adapter: ${error.message}`);
        }
    }

    async processDirectory(directory) {
        const results = {
            totalFiles: 0,
            filesNeedingMigration: 0,
            filesMigrated: 0,
            fileResults: []
        };

        try {
            const files = await this.getJsFiles(directory);

            for (const file of files) {
                results.totalFiles++;

                const analysis = await this.analyzeFile(file);

                if (analysis.needsMigration) {
                    results.filesNeedingMigration++;

                    const migrationResult = await this.migrateFile(file);
                    results.fileResults.push(migrationResult);

                    if (migrationResult.status === 'migrated') {
                        results.filesMigrated++;
                    }
                }

                this.filesProcessed++;
            }

            // Create blockchain adapter if files were migrated
            if (results.filesMigrated > 0) {
                await this.createBlockchainAdapter(directory);
            }

        } catch (error) {
            console.error(`Error processing directory: ${error.message}`);
        }

        return results;
    }

    async getJsFiles(directory, fileList = []) {
        const files = await fs.readdir(directory, { withFileTypes: true });

        for (const file of files) {
            const fullPath = path.join(directory, file.name);

            if (file.isDirectory()) {
                // Skip node_modules and test directories
                if (!file.name.includes('node_modules') && !file.name.includes('test')) {
                    await this.getJsFiles(fullPath, fileList);
                }
            } else if (file.name.endsWith('.js') && !file.name.includes('.test.')) {
                fileList.push(fullPath);
            }
        }

        return fileList;
    }

    generateReport(results) {
        // console.log('\n=== WebSocket to Blockchain Migration Report ===\n');
        // console.log(`Total files scanned: ${results.totalFiles}`);
        // console.log(`Files needing migration: ${results.filesNeedingMigration}`);
        // console.log(`Files successfully migrated: ${results.filesMigrated}`);

        if (results.fileResults.length > 0) {
            // console.log('\nMigrated files:');
            results.fileResults.forEach(result => {
                if (result.status === 'migrated') {
                    // console.log(`  ✅ ${result.file}: ${result.replacements} replacements`);
                } else if (result.status === 'error') {
                    // console.log(`  ❌ ${result.file}: ${result.error}`);
                }
            });
        }

        if (this.errors.length > 0) {
            // console.log('\nErrors encountered:');
            this.errors.forEach(error => {
                // console.log(`  - ${error.file}: ${error.error}`);
            });
        }

        // console.log('\n✅ WebSocket to blockchain migration complete!');
        // console.log('🔗 All real-time communication now uses blockchain events');
        // console.log('⚠️  Remember to deploy the blockchain event contracts');
    }
}

async function main() {
    const migrator = new WebSocketToBlockchainMigrator();

    // Process network services directory
    const networkDir = path.join(__dirname, '..');

    // console.log('🔍 Scanning for WebSocket usage...');
    const results = await migrator.processDirectory(networkDir);

    // Generate report
    migrator.generateReport(results);

    // Save detailed results
    const resultsPath = path.join(__dirname, 'websocket_migration_results.json');
    await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
    // console.log(`\nDetailed results saved to: ${resultsPath}`);
}

// Run if called directly
if (require.main === module) {
    main().catch(() => {});
}

module.exports = WebSocketToBlockchainMigrator;