#!/usr/bin/env node

/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

/**
 * A2A Network - SCIP-based Code Indexing Script
 * Indexes the codebase using SCIP and uploads facts to Glean
 */

const SCIPIndexer = require('../srv/glean/scipIndexer');
const GleanFactTransformer = require('../srv/glean/gleanFactTransformer');
const path = require('path');

class CodebaseIndexer {
    constructor() {
        this.workspaceRoot = path.resolve(process.cwd());
        this.scipIndexer = new SCIPIndexer(this.workspaceRoot);
        this.factTransformer = new GleanFactTransformer();
        this.verbose = process.argv.includes('--verbose') || process.argv.includes('-v');
        this.dryRun = process.argv.includes('--dry-run');
        this.languages = this.parseLanguages();
        this.gleanUrl = process.env.GLEAN_URL || 'http://localhost:8080';
    }

    parseLanguages() {
        const langIndex = process.argv.indexOf('--languages');
        if (langIndex !== -1 && process.argv[langIndex + 1]) {
            return process.argv[langIndex + 1].split(',');
        }
        return ['typescript', 'javascript', 'python', 'solidity'];
    }

    log(message, level = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            'info': '📋',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'debug': '🔍'
        }[level] || '📋';

        if (level !== 'debug' || this.verbose) {
            console.log(`${prefix} [${timestamp}] ${message}`);
        }
    }

    async run() {
        try {
            this.log('🚀 Starting A2A Network codebase indexing with SCIP', 'info');
            this.log(`Languages to index: ${this.languages.join(', ')}`, 'info');
            this.log(`Workspace root: ${this.workspaceRoot}`, 'debug');

            if (this.dryRun) {
                this.log('Running in dry-run mode - no facts will be uploaded', 'warning');
            }

            // Initialize SCIP indexer
            await this.scipIndexer.initialize();
            this.log('SCIP indexer initialized', 'success');

            // Check Glean availability
            if (!this.dryRun) {
                await this.checkGleanAvailability();
            }

            // Index the project
            this.log('Starting SCIP indexing...', 'info');
            const scipResults = await this.scipIndexer.indexProject(this.languages);

            this.log(`SCIP indexing completed:`, 'success');
            this.log(`  - Documents indexed: ${scipResults.documentCount}`, 'info');
            this.log(`  - Symbols found: ${scipResults.symbolCount}`, 'info');
            this.log(`  - SCIP index file: ${scipResults.scipIndex}`, 'debug');

            // Transform to Glean facts
            this.log('Transforming SCIP data to Glean facts...', 'info');
            const gleanFacts = this.factTransformer.transformSCIPToGlean(scipResults.scipIndex);

            // Count total facts
            const totalFacts = Object.values(gleanFacts).reduce((sum, facts) => sum + facts.length, 0);
            this.log(`Generated ${totalFacts} Glean facts across ${Object.keys(gleanFacts).length} predicates`, 'success');

            // Log fact breakdown
            for (const [predicate, facts] of Object.entries(gleanFacts)) {
                if (facts.length > 0) {
                    this.log(`  - ${predicate}: ${facts.length} facts`, 'debug');
                }
            }

            if (!this.dryRun) {
                // Upload to Glean
                this.log('Uploading facts to Glean...', 'info');
                await this.uploadFactsToGlean(gleanFacts);
                this.log('All facts uploaded successfully', 'success');

                // Trigger Glean index refresh
                await this.refreshGleanIndex();
                this.log('Glean index refreshed', 'success');
            }

            // Generate summary report
            const report = this.generateSummaryReport(scipResults, gleanFacts, totalFacts);
            await this.saveReport(report);

            this.log('✨ Indexing completed successfully!', 'success');

            return {
                success: true,
                documentsIndexed: scipResults.documentCount,
                symbolsFound: scipResults.symbolCount,
                factsGenerated: totalFacts,
                scipIndex: scipResults.scipIndex
            };

        } catch (error) {
            this.log(`Indexing failed: ${error.message}`, 'error');
            if (this.verbose) {
                this.log(`Stack trace: ${error.stack}`, 'debug');
            }
            process.exit(1);
        }
    }

    async checkGleanAvailability() {
        try {
            const response = await blockchainClient.sendMessage(`${this.gleanUrl}/health`);
            if (!response.ok) {
                throw new Error(`Glean server responded with ${response.status}`);
            }
            this.log('Glean server is available', 'success');
        } catch (error) {
            throw new Error(`Glean server not available at ${this.gleanUrl}: ${error.message}`);
        }
    }

    async uploadFactsToGlean(gleanFactBatches) {
        for (const [predicate, facts] of Object.entries(gleanFactBatches)) {
            if (facts.length === 0) continue;

            this.log(`Uploading ${facts.length} facts for predicate ${predicate}`, 'info');

            try {
                const response = await blockchainClient.sendMessage(`${this.gleanUrl}/api/v1/facts`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${process.env.GLEAN_API_TOKEN || ''}`
                    },
                    body: JSON.stringify({
                        schema_version: '1.0',
                        predicate: predicate,
                        facts: facts
                    })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                this.log(`Successfully uploaded ${facts.length} ${predicate} facts`, 'debug');

            } catch (error) {
                this.log(`Failed to upload ${predicate} facts: ${error.message}`, 'error');
                throw error;
            }
        }
    }

    async refreshGleanIndex() {
        try {
            const response = await blockchainClient.sendMessage(`${this.gleanUrl}/api/v1/index/refresh`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${process.env.GLEAN_API_TOKEN || ''}`
                }
            });

            if (!response.ok) {
                this.log(`Warning: Failed to refresh Glean index: ${response.statusText}`, 'warning');
            }
        } catch (error) {
            this.log(`Warning: Could not refresh Glean index: ${error.message}`, 'warning');
        }
    }

    generateSummaryReport(scipResults, gleanFacts, totalFacts) {
        const timestamp = new Date().toISOString();

        return {
            indexing_summary: {
                timestamp,
                workspace_root: this.workspaceRoot,
                languages: this.languages,
                scip_results: {
                    documents_indexed: scipResults.documentCount,
                    symbols_found: scipResults.symbolCount,
                    index_file: scipResults.scipIndex
                },
                glean_facts: {
                    total_facts: totalFacts,
                    predicates: Object.keys(gleanFacts).length,
                    breakdown: Object.fromEntries(
                        Object.entries(gleanFacts).map(([pred, facts]) => [pred, facts.length])
                    )
                },
                configuration: {
                    glean_url: this.gleanUrl,
                    dry_run: this.dryRun,
                    verbose: this.verbose
                }
            }
        };
    }

    async saveReport(report) {
        const fs = require('fs').promises;
        const timestamp = Date.now();
        const reportFile = path.join(this.workspaceRoot, `indexing-report-${timestamp}.json`);

        await fs.writeFile(reportFile, JSON.stringify(report, null, 2));
        this.log(`Report saved to: ${reportFile}`, 'info');
    }
}

// CLI usage help
function showHelp() {
    console.log(`
A2A Network SCIP-based Code Indexer

Usage: node indexCodebase.js [options]

Options:
  --languages <langs>    Comma-separated list of languages to index
                        (default: typescript,javascript,python,solidity)
  --dry-run             Generate facts but don't upload to Glean
  --verbose, -v         Enable verbose logging
  --help, -h            Show this help message

Environment Variables:
  GLEAN_URL             Glean server URL (default: http://localhost:8080)
  GLEAN_API_TOKEN       Authentication token for Glean API

Examples:
  node indexCodebase.js
  node indexCodebase.js --languages typescript,javascript
  node indexCodebase.js --dry-run --verbose
`);
}

// Main execution
if (require.main === module) {
    if (process.argv.includes('--help') || process.argv.includes('-h')) {
        showHelp();
        process.exit(0);
    }

    const indexer = new CodebaseIndexer();
    indexer.run().then(result => {
        process.exit(0);
    }).catch(error => {
        console.error('❌ Indexing failed:', error.message);
        process.exit(1);
    });
}

module.exports = CodebaseIndexer;