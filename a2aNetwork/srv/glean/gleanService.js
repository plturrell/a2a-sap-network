const cds = require('@sap/cds');
const GraphAlgorithms = require('../algorithms/graphAlgorithms');
const TreeAlgorithms = require('../algorithms/treeAlgorithms');

/**
 * Enhanced Glean Service with CLRS algorithms and Tree operations
 * Extends the existing Glean service with advanced code analysis capabilities
 */
class EnhancedGleanService extends cds.ApplicationService {
    constructor(...args) {
        super(...args);
        this.graphAlgorithms = new GraphAlgorithms();
        this.treeAlgorithms = new TreeAlgorithms();
        this.logger = cds.log('enhanced-glean-service');
    }

    /**
     * Initialize enhanced service with additional capabilities
     */
    async initializeService() {
        await super.initializeService();
        
        this.logger.info('Initializing Enhanced Glean Service');
        
        // Initialize algorithm services
        await this.graphAlgorithms.initializeService();
        await this.treeAlgorithms.initializeService();
        
        // Register enhanced actions
        this._registerEnhancedActions();
    }

    /**
     * Error handling wrapper for service methods
     * @private
     */
    async _withErrorHandling(methodName, operation) {
        try {
            return await operation();
        } catch (error) {
            this.logger.error(`${methodName} failed:`, error);
            return {
                success: false,
                error: error.message,
                methodName,
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Register enhanced SAP CAP actions
     */
    _registerEnhancedActions() {
        // Advanced dependency analysis
        this.on('analyzeDependencyCriticalPaths', async (req) => {
            return await this._withErrorHandling('analyzeDependencyCriticalPaths', async () => {
                const { projectId, targetFile } = req.data;
                return await this.analyzeDependencyCriticalPaths(projectId, targetFile);
            });
        });

        // Code similarity detection
        this.on('findSimilarCode', async (req) => {
            return await this._withErrorHandling('findSimilarCode', async () => {
                const { codeSnippet, threshold = 0.8 } = req.data;
                return await this.findSimilarCode(codeSnippet, threshold);
            });
        });

        // Hierarchical code navigation
        this.on('navigateCodeHierarchy', async (req) => {
            return await this._withErrorHandling('navigateCodeHierarchy', async () => {
                const { rootPath, query } = req.data;
                return await this.navigateCodeHierarchy(rootPath, query);
            });
        });

        // Intelligent refactoring suggestions
        this.on('suggestRefactorings', async (req) => {
            return await this._withErrorHandling('suggestRefactorings', async () => {
                const { filePath, analysisDepth = 2 } = req.data;
                return await this.suggestRefactorings(filePath, analysisDepth);
            });
        });
    }

    /**
     * Analyze critical dependency paths using Dijkstra's algorithm
     */
    async analyzeDependencyCriticalPaths(projectId, targetFile) {
        this.logger.info(`Analyzing critical paths for ${targetFile} in project ${projectId}`);
        
        // Build dependency graph from Glean facts
        const graph = await this._buildDependencyGraph(projectId);
        
        // Find entry points (files with no incoming dependencies)
        const entryPoints = await this._findEntryPoints(graph);
        
        const results = {
            criticalPaths: [],
            circularDependencies: [],
            buildOrder: [],
            dependencyMetrics: {}
        };
        
        // Analyze paths from each entry point to target
        for (const entryPoint of entryPoints) {
            const pathAnalysis = this.graphAlgorithms.dijkstra(graph, entryPoint);
            
            if (pathAnalysis.hasPath(targetFile)) {
                results.criticalPaths.push({
                    from: entryPoint,
                    to: targetFile,
                    path: pathAnalysis.getPath(targetFile),
                    distance: pathAnalysis.getDistance(targetFile),
                    isMainPath: entryPoint === 'index.js' || entryPoint === 'main.js'
                });
            }
        }
        
        // Find circular dependencies
        const cycles = this.graphAlgorithms.detectCycles(graph);
        results.circularDependencies = cycles.map(cycle => ({
            files: cycle,
            severity: this._calculateCycleSeverity(cycle, graph)
        }));
        
        // Calculate optimal build order
        try {
            results.buildOrder = this.graphAlgorithms.topologicalSort(graph);
        } catch (error) {
            this.logger.warn('Cannot determine build order due to circular dependencies');
        }
        
        // Calculate dependency metrics
        results.dependencyMetrics = await this._calculateDependencyMetrics(graph, targetFile);
        
        return results;
    }

    /**
     * Find similar code using longest common subsequence
     */
    async findSimilarCode(codeSnippet, threshold) {
        this.logger.info(`Finding code similar to snippet with threshold ${threshold}`);
        
        // Get all indexed code files
        const indexedFiles = await this._getIndexedFiles();
        const similarities = [];
        
        // Normalize the input snippet
        const normalizedSnippet = this._normalizeCode(codeSnippet);
        
        for (const file of indexedFiles) {
            const fileContent = await this._readFileContent(file.path);
            const normalizedContent = this._normalizeCode(fileContent);
            
            // Use sliding window for large files
            const windowSize = normalizedSnippet.length * 2;
            const windows = this._createSlidingWindows(normalizedContent, windowSize);
            
            for (const window of windows) {
                const similarity = this._calculateSimilarity(normalizedSnippet, window.content);
                
                if (similarity >= threshold) {
                    similarities.push({
                        file: file.path,
                        startLine: window.startLine,
                        endLine: window.endLine,
                        similarity: similarity,
                        matchedCode: window.originalContent,
                        suggestions: this._generateRefactoringSuggestions(codeSnippet, window.originalContent)
                    });
                }
            }
        }
        
        // Sort by similarity score
        similarities.sort((a, b) => b.similarity - a.similarity);
        
        return {
            query: codeSnippet,
            matches: similarities.slice(0, 50), // Limit results
            statistics: {
                totalFilesScanned: indexedFiles.length,
                matchesFound: similarities.length,
                averageSimilarity: similarities.reduce((sum, s) => sum + s.similarity, 0) / similarities.length || 0
            }
        };
    }

    /**
     * Navigate code hierarchy using tree algorithms
     */
    async navigateCodeHierarchy(rootPath, query) {
        this.logger.info(`Navigating code hierarchy from ${rootPath} with query: ${query}`);
        
        // Build hierarchical code structure
        const codeTree = await this._buildCodeTree(rootPath);
        
        // Apply query filters
        const queryResult = this._applyHierarchicalQuery(codeTree, query);
        
        // Calculate metrics
        const metrics = {
            totalNodes: this.treeAlgorithms.getNodeCount(codeTree),
            totalLeaves: this.treeAlgorithms.getLeafCount(codeTree),
            maxDepth: this.treeAlgorithms.getDepth(codeTree),
            matchedNodes: this.treeAlgorithms.getNodeCount(queryResult)
        };
        
        // Get all paths for matched elements
        const matchedPaths = this.treeAlgorithms.getAllPaths(queryResult);
        
        return {
            rootPath,
            query,
            hierarchy: queryResult,
            matchedPaths: matchedPaths.map(path => ({
                path: path.keys.join('/'),
                element: path.value,
                depth: path.keys.length
            })),
            metrics
        };
    }

    /**
     * Suggest refactorings using combined graph and tree analysis
     */
    async suggestRefactorings(filePath, analysisDepth) {
        this.logger.info(`Analyzing ${filePath} for refactoring suggestions with depth ${analysisDepth}`);
        
        const suggestions = [];
        
        // Parse file AST
        const ast = await this._parseFileAST(filePath);
        const codeStructure = this._astToTreeStructure(ast);
        
        // Analyze code patterns
        const patterns = await this._analyzeCodePatterns(codeStructure);
        
        // 1. Detect complex functions using cyclomatic complexity
        const complexFunctions = this._findComplexFunctions(codeStructure);
        for (const func of complexFunctions) {
            suggestions.push({
                type: 'EXTRACT_METHOD',
                severity: 'high',
                location: func.path,
                description: `Function ${func.name} has cyclomatic complexity of ${func.complexity}. Consider breaking it into smaller functions.`,
                automated: true,
                refactoring: this._generateExtractMethodRefactoring(func)
            });
        }
        
        // 2. Find duplicate code patterns
        const duplicates = this._findDuplicatePatterns(codeStructure);
        for (const duplicate of duplicates) {
            suggestions.push({
                type: 'EXTRACT_COMMON',
                severity: 'medium',
                locations: duplicate.locations,
                description: `Found ${duplicate.count} instances of similar code pattern. Consider extracting to a common function.`,
                automated: true,
                refactoring: this._generateExtractCommonRefactoring(duplicate)
            });
        }
        
        // 3. Analyze dependencies for better structure
        const dependencyIssues = await this._analyzeDependencyStructure(filePath, analysisDepth);
        for (const issue of dependencyIssues) {
            suggestions.push({
                type: 'RESTRUCTURE_DEPENDENCIES',
                severity: issue.severity,
                description: issue.description,
                automated: false,
                recommendation: issue.recommendation
            });
        }
        
        // 4. Find inefficient patterns
        const inefficiencies = this._findInefficiencies(codeStructure);
        for (const inefficiency of inefficiencies) {
            suggestions.push({
                type: 'OPTIMIZE_PATTERN',
                severity: 'low',
                location: inefficiency.path,
                description: inefficiency.description,
                automated: true,
                refactoring: inefficiency.optimization
            });
        }
        
        // Sort suggestions by severity and feasibility
        suggestions.sort((a, b) => {
            const severityOrder = { high: 3, medium: 2, low: 1 };
            const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
            if (severityDiff !== 0) return severityDiff;
            return b.automated ? 1 : -1;
        });
        
        return {
            file: filePath,
            totalSuggestions: suggestions.length,
            suggestions: suggestions.slice(0, 20), // Limit to top 20
            codeQualityScore: this._calculateCodeQualityScore(patterns, suggestions),
            executionPlan: this._generateRefactoringExecutionPlan(suggestions)
        };
    }

    /**
     * Build dependency graph from Glean facts
     * @private
     */
    async _buildDependencyGraph(projectId) {
        const graph = this.graphAlgorithms.createGraph();
        
        // Query Glean for all file dependencies
        const dependencies = await this.query({
            predicate: 'code.Dependency',
            projectId
        });
        
        for (const dep of dependencies) {
            const fromFile = dep.facts.find(f => f.predicate === 'code.FileDefines')?.key;
            const toFile = dep.facts.find(f => f.predicate === 'code.FileUses')?.key;
            
            if (fromFile && toFile) {
                this.graphAlgorithms.addEdge(graph, fromFile, toFile, 1);
            }
        }
        
        return graph;
    }

    /**
     * Calculate code similarity using enhanced algorithm
     * @private
     */
    _calculateSimilarity(code1, code2) {
        // Token-based similarity for better accuracy
        const tokens1 = this._tokenizeCode(code1);
        const tokens2 = this._tokenizeCode(code2);
        
        // Calculate longest common subsequence
        const lcs = this._longestCommonSubsequence(tokens1, tokens2);
        
        // Similarity score based on LCS length relative to both sequences
        const similarity = (2 * lcs.length) / (tokens1.length + tokens2.length);
        
        return similarity;
    }

    /**
     * Longest Common Subsequence implementation
     * @private
     */
    _longestCommonSubsequence(arr1, arr2) {
        const m = arr1.length;
        const n = arr2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (arr1[i - 1] === arr2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        // Reconstruct the LCS
        const lcs = [];
        let i = m, j = n;
        while (i > 0 && j > 0) {
            if (arr1[i - 1] === arr2[j - 1]) {
                lcs.unshift(arr1[i - 1]);
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }
        
        return lcs;
    }

    /**
     * Build hierarchical code tree from file system
     * @private
     */
    async _buildCodeTree(rootPath) {
        const files = await this._getFilesRecursively(rootPath);
        const tree = {};
        
        for (const file of files) {
            const pathParts = file.relativePath.split('/');
            let current = tree;
            
            for (let i = 0; i < pathParts.length - 1; i++) {
                if (!current[pathParts[i]]) {
                    current[pathParts[i]] = {};
                }
                current = current[pathParts[i]];
            }
            
            // Add file with metadata
            const fileName = pathParts[pathParts.length - 1];
            current[fileName] = await this._getFileMetadata(file.absolutePath);
        }
        
        return tree;
    }

    /**
     * Apply hierarchical query to code tree
     * @private
     */
    _applyHierarchicalQuery(codeTree, query) {
        // Parse query (supports patterns like "*.controller.js", "function:onClick*")
        const queryParts = query.split(':');
        const pattern = queryParts[0];
        const filter = queryParts[1];
        
        return this.treeAlgorithms.filterStructure((item, path) => {
            const pathString = path.join('/');
            
            // Check pattern match
            if (pattern !== '*' && !this._matchesPattern(pathString, pattern)) {
                return false;
            }
            
            // Check filter if provided
            if (filter && item.type === 'file') {
                return this._matchesFilter(item, filter);
            }
            
            return true;
        }, codeTree);
    }

    /**
     * Generate refactoring execution plan
     * @private
     */
    _generateRefactoringExecutionPlan(suggestions) {
        const automatedSuggestions = suggestions.filter(s => s.automated);
        
        // Build dependency graph for refactorings
        const graph = this.graphAlgorithms.createGraph();
        
        // Add nodes for each refactoring
        automatedSuggestions.forEach((suggestion, index) => {
            this.graphAlgorithms.addNode(graph, `refactoring-${index}`, {
                suggestion,
                estimatedTime: this._estimateRefactoringTime(suggestion)
            });
        });
        
        // Add dependencies between refactorings
        for (let i = 0; i < automatedSuggestions.length; i++) {
            for (let j = i + 1; j < automatedSuggestions.length; j++) {
                if (this._refactoringsConflict(automatedSuggestions[i], automatedSuggestions[j])) {
                    // Add edge to indicate dependency
                    this.graphAlgorithms.addEdge(graph, `refactoring-${i}`, `refactoring-${j}`);
                }
            }
        }
        
        // Calculate execution order
        try {
            const executionOrder = this.graphAlgorithms.topologicalSort(graph);
            return {
                canExecuteAll: true,
                executionOrder: executionOrder.map(id => {
                    const index = parseInt(id.split('-')[1]);
                    return automatedSuggestions[index];
                }),
                estimatedTotalTime: automatedSuggestions.reduce((sum, s) => 
                    sum + this._estimateRefactoringTime(s), 0
                )
            };
        } catch (error) {
            // Has circular dependencies
            return {
                canExecuteAll: false,
                conflictingRefactorings: this.graphAlgorithms.detectCycles(graph),
                recommendation: 'Manual intervention required to resolve conflicts'
            };
        }
    }
}

module.exports = EnhancedGleanService;