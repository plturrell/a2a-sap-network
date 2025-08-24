sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "../model/formatter",
    "../utils/CodeAnalysisService",
    "sap/ui/core/format/NumberFormat"
], (BaseController, MessageToast, MessageBox, JSONModel, formatter, CodeAnalysisService, NumberFormat) => {
    "use strict";

    /**
     * Code Intelligence Dashboard Controller
     * Manages advanced code analysis and visualization features
     * Following SAP Fiori design guidelines and enterprise patterns
     */
    return BaseController.extend("a2a.network.fiori.controller.CodeIntelligence", {

        formatter,

        /**
         * Initialize the code intelligence view
         */
        onInit() {
            this.getRouter().getRoute("codeIntelligence").attachPatternMatched(this._onRouteMatched, this);

            // Initialize view models
            this._initializeViewModels();

            // Initialize code analysis service
            this.oCodeAnalysisService = new CodeAnalysisService();

            // Set up real-time updates
            this._setupRealTimeUpdates();

            // Load initial dashboard data
            this._loadDashboardData();
        },

        /**
         * Initialize all view models
         * @private
         */
        _initializeViewModels() {
            // Main dashboard model
            const oDashboardModel = new JSONModel({
                overview: {
                    totalFiles: 0,
                    codeQualityScore: 0,
                    technicalDebt: 0,
                    lastUpdated: null
                },
                metrics: {
                    complexity: {
                        averageCyclomatic: 0,
                        highComplexityFunctions: 0,
                        totalFunctions: 0
                    },
                    dependencies: {
                        circularDependencies: 0,
                        dependencyDepth: 0,
                        criticalPaths: []
                    },
                    duplicates: {
                        duplicateBlocks: 0,
                        duplicatePercentage: 0,
                        suggestions: []
                    }
                },
                charts: {
                    complexityTrend: [],
                    dependencyGraph: {
                        nodes: [],
                        edges: []
                    },
                    qualityTimeline: []
                },
                recommendations: [],
                alerts: [],
                isLoading: false,
                hasError: false,
                errorMessage: ""
            });
            this.setModel(oDashboardModel, "dashboard");

            // Analysis parameters model
            const oAnalysisModel = new JSONModel({
                selectedProject: "",
                analysisType: "full",
                includeMetrics: ["complexity", "dependencies", "duplicates"],
                maxDepth: 5,
                fuzzyThreshold: 0.8,
                filters: {
                    filePattern: "*",
                    severity: "medium"
                }
            });
            this.setModel(oAnalysisModel, "analysis");

            // Search model
            const oSearchModel = new JSONModel({
                searchPattern: "",
                searchType: "semantic",
                searchResults: [],
                totalResults: 0,
                isSearching: false
            });
            this.setModel(oSearchModel, "search");
        },

        /**
         * Handle route pattern matched
         * @private
         */
        _onRouteMatched(oEvent) {
            const oArgs = oEvent.getParameter("arguments");
            if (oArgs.projectId) {
                this.getModel("analysis").setProperty("/selectedProject", oArgs.projectId);
                this._refreshAnalysis();
            }
        },

        /**
         * Set up real-time updates for dashboard
         * @private
         */
        _setupRealTimeUpdates() {
            // Update dashboard every 30 seconds
            this._updateTimer = setInterval(() => {
                this._updateMetrics();
            }, 30000);
        },

        /**
         * Load initial dashboard data
         * @private
         */
        _loadDashboardData() {
            this._showDashboardLoading(true);

            Promise.all([
                this._loadOverviewMetrics(),
                this._loadComplexityMetrics(),
                this._loadDependencyMetrics(),
                this._loadDuplicateMetrics(),
                this._loadRecommendations(),
                this._loadQualityAlerts()
            ]).then((results) => {
                this._updateDashboardData(results);
                this._showDashboardLoading(false);
            }).catch((error) => {
                this._handleDashboardError(error);
                this._showDashboardLoading(false);
            });
        },

        /**
         * Refresh analysis for selected project
         * @private
         */
        _refreshAnalysis() {
            const sProjectId = this.getModel("analysis").getProperty("/selectedProject");
            if (!sProjectId) {
                return;
            }

            this._showDashboardLoading(true);

            this.oCodeAnalysisService.analyzeProject(sProjectId, {
                analysisType: this.getModel("analysis").getProperty("/analysisType"),
                includeMetrics: this.getModel("analysis").getProperty("/includeMetrics"),
                maxDepth: this.getModel("analysis").getProperty("/maxDepth")
            }).then((results) => {
                this._updateAnalysisResults(results);
                this._showDashboardLoading(false);
                MessageToast.show(this.getResourceBundle().getText("analysisCompleted"));
            }).catch((error) => {
                this._handleAnalysisError(error);
                this._showDashboardLoading(false);
            });
        },

        /**
         * Handle analysis type selection change
         */
        onAnalysisTypeChange(oEvent) {
            const sSelectedType = oEvent.getParameter("selectedItem").getKey();
            this.getModel("analysis").setProperty("/analysisType", sSelectedType);
            this._refreshAnalysis();
        },

        /**
         * Handle metric selection change
         */
        onMetricSelectionChange(oEvent) {
            const aSelectedItems = oEvent.getParameter("selectedItems");
            const aMetrics = aSelectedItems.map((oItem) => {
                return oItem.getKey();
            });
            this.getModel("analysis").setProperty("/includeMetrics", aMetrics);
        },

        /**
         * Perform smart code search
         */
        onSmartSearch() {
            const oSearchModel = this.getModel("search");
            const sPattern = oSearchModel.getProperty("/searchPattern");
            const sSearchType = oSearchModel.getProperty("/searchType");

            if (!sPattern) {
                MessageToast.show(this.getResourceBundle().getText("enterSearchPattern"));
                return;
            }

            oSearchModel.setProperty("/isSearching", true);

            this.oCodeAnalysisService.smartSearch({
                pattern: sPattern,
                searchType: sSearchType,
                fuzzyThreshold: this.getModel("analysis").getProperty("/fuzzyThreshold"),
                maxResults: 50
            }).then((results) => {
                oSearchModel.setProperty("/searchResults", results.results);
                oSearchModel.setProperty("/totalResults", results.resultCount);
                oSearchModel.setProperty("/isSearching", false);

                if (results.resultCount === 0) {
                    MessageToast.show(this.getResourceBundle().getText("noSearchResults"));
                }
            }).catch((error) => {
                this._handleSearchError(error);
                oSearchModel.setProperty("/isSearching", false);
            });
        },

        /**
         * Navigate to specific code location
         */
        onNavigateToCode(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("search");
            const oResult = oContext.getObject();

            // Open file in code editor (implement based on your setup)
            this._openCodeEditor(oResult.file, oResult.startLine);
        },

        /**
         * Execute automated refactoring
         */
        onExecuteRefactoring(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("dashboard");
            const oRefactoring = oContext.getObject();

            if (!oRefactoring.automated) {
                MessageBox.information(
                    this.getResourceBundle().getText("manualRefactoringRequired"),
                    {
                        title: this.getResourceBundle().getText("refactoringInfo")
                    }
                );
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("confirmRefactoring", [oRefactoring.description]),
                {
                    title: this.getResourceBundle().getText("executeRefactoring"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._executeAutomatedRefactoring(oRefactoring);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * View dependency graph
         */
        onViewDependencyGraph() {
            const oDependencyData = this.getModel("dashboard").getProperty("/metrics/dependencies");

            if (!oDependencyData.criticalPaths || oDependencyData.criticalPaths.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("noDependencyData"));
                return;
            }

            // Open dependency graph dialog
            this._openDependencyGraphDialog(oDependencyData);
        },

        /**
         * Export analysis report
         */
        onExportReport() {
            const oDashboardData = this.getModel("dashboard").getData();

            // Generate comprehensive report
            const _oReport = this._generateAnalysisReport(oDashboardData);

            // Download as JSON file
            this._downloadReport(oReport, "code-analysis-report.json");
        },

        /**
         * Open dependency graph dialog
         * @private
         */
        _openDependencyGraphDialog(oDependencyData) {
            if (!this._dependencyGraphDialog) {
                this._dependencyGraphDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.view.fragments.DependencyGraphDialog",
                    this
                );
                this.getView().addDependent(this._dependencyGraphDialog);
            }

            // Set graph data
            const oGraphModel = new JSONModel({
                nodes: this._prepareGraphNodes(oDependencyData),
                edges: this._prepareGraphEdges(oDependencyData),
                layout: "hierarchical"
            });
            this._dependencyGraphDialog.setModel(oGraphModel, "graph");

            this._dependencyGraphDialog.open();
        },

        /**
         * Close dependency graph dialog
         */
        onCloseDependencyGraph() {
            this._dependencyGraphDialog.close();
        },

        /**
         * Handle complexity drill-down
         */
        onComplexityDrillDown(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("dashboard");
            const sFilePath = oContext.getProperty("filePath");

            // Navigate to detailed complexity view
            this.getRouter().navTo("complexityDetail", {
                filePath: encodeURIComponent(sFilePath)
            });
        },

        /**
         * Show/hide loading state for dashboard
         * @private
         */
        _showDashboardLoading(bLoading) {
            this.getModel("dashboard").setProperty("/isLoading", bLoading);
        },

        /**
         * Handle dashboard errors
         * @private
         */
        _handleDashboardError(error) {
            this.getModel("dashboard").setProperty("/hasError", true);
            this.getModel("dashboard").setProperty("/errorMessage", error.message);
            this.logger.error("Dashboard error:", error);
            MessageToast.show(this.getResourceBundle().getText("dashboardLoadError"));
        },

        /**
         * Handle analysis errors
         * @private
         */
        _handleAnalysisError(error) {
            this.logger.error("Analysis error:", error);
            MessageBox.error(
                this.getResourceBundle().getText("analysisError", [error.message]),
                {
                    title: this.getResourceBundle().getText("analysisErrorTitle")
                }
            );
        },

        /**
         * Handle search errors
         * @private
         */
        _handleSearchError(error) {
            this.logger.error("Search error:", error);
            MessageToast.show(this.getResourceBundle().getText("searchError"));
        },

        /**
         * Load overview metrics
         * @private
         */
        _loadOverviewMetrics() {
            return this.oCodeAnalysisService.getOverviewMetrics();
        },

        /**
         * Load complexity metrics
         * @private
         */
        _loadComplexityMetrics() {
            return this.oCodeAnalysisService.getComplexityMetrics();
        },

        /**
         * Load dependency metrics
         * @private
         */
        _loadDependencyMetrics() {
            return this.oCodeAnalysisService.getDependencyMetrics();
        },

        /**
         * Load duplicate metrics
         * @private
         */
        _loadDuplicateMetrics() {
            return this.oCodeAnalysisService.getDuplicateMetrics();
        },

        /**
         * Load recommendations
         * @private
         */
        _loadRecommendations() {
            return this.oCodeAnalysisService.getRecommendations();
        },

        /**
         * Load quality alerts
         * @private
         */
        _loadQualityAlerts() {
            return this.oCodeAnalysisService.getQualityAlerts();
        },

        /**
         * Update dashboard with loaded data
         * @private
         */
        _updateDashboardData(results) {
            const oDashboardModel = this.getModel("dashboard");

            oDashboardModel.setProperty("/overview", results[0]);
            oDashboardModel.setProperty("/metrics/complexity", results[1]);
            oDashboardModel.setProperty("/metrics/dependencies", results[2]);
            oDashboardModel.setProperty("/metrics/duplicates", results[3]);
            oDashboardModel.setProperty("/recommendations", results[4]);
            oDashboardModel.setProperty("/alerts", results[5]);
            oDashboardModel.setProperty("/overview/lastUpdated", new Date());
        },

        /**
         * Update metrics (for real-time updates)
         * @private
         */
        _updateMetrics() {
            // Only update if dashboard is visible and not loading
            if (this.getModel("dashboard").getProperty("/isLoading")) {
                return;
            }

            this.oCodeAnalysisService.getQuickMetrics().then((metrics) => {
                const oDashboardModel = this.getModel("dashboard");
                oDashboardModel.setProperty("/overview/codeQualityScore", metrics.qualityScore);
                oDashboardModel.setProperty("/overview/technicalDebt", metrics.technicalDebt);
                oDashboardModel.setProperty("/overview/lastUpdated", new Date());
            }).catch((error) => {
                this.logger.error("Quick metrics update failed:", error);
            });
        },

        /**
         * Execute automated refactoring
         * @private
         */
        _executeAutomatedRefactoring(oRefactoring) {
            this.showLoadingDialog(this.getResourceBundle().getText("executingRefactoring"));

            this.oCodeAnalysisService.executeRefactoring(oRefactoring.id).then((result) => {
                this.hideLoadingDialog();

                if (result.success) {
                    MessageToast.show(this.getResourceBundle().getText("refactoringCompleted"));
                    this._refreshAnalysis(); // Refresh to show updated metrics
                } else {
                    MessageBox.error(
                        this.getResourceBundle().getText("refactoringFailed", [result.error]),
                        {
                            title: this.getResourceBundle().getText("refactoringError")
                        }
                    );
                }
            }).catch((error) => {
                this.hideLoadingDialog();
                this._handleAnalysisError(error);
            });
        },

        /**
         * Generate analysis report
         * @private
         */
        _generateAnalysisReport(oDashboardData) {
            return {
                generatedAt: new Date().toISOString(),
                projectId: this.getModel("analysis").getProperty("/selectedProject"),
                overview: oDashboardData.overview,
                metrics: oDashboardData.metrics,
                recommendations: oDashboardData.recommendations,
                alerts: oDashboardData.alerts,
                summary: {
                    qualityGrade: this._calculateQualityGrade(oDashboardData.overview.codeQualityScore),
                    priorityRecommendations: oDashboardData.recommendations.filter(r => r.severity === "high").length,
                    criticalIssues: oDashboardData.alerts.filter(a => a.severity === "critical").length
                }
            };
        },

        /**
         * Download report file
         * @private
         */
        _downloadReport(oReport, sFilename) {
            const sData = JSON.stringify(oReport, null, 2);
            const oBlob = new Blob([sData], { type: "application/json" });
            const sUrl = URL.createObjectURL(oBlob);

            const oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFilename;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
        },

        /**
         * Calculate quality grade from score
         * @private
         */
        _calculateQualityGrade(nScore) {
            if (nScore >= 90) {
                return "A";
            }
            if (nScore >= 80) {
                return "B";
            }
            if (nScore >= 70) {
                return "C";
            }
            if (nScore >= 60) {
                return "D";
            }
            return "F";
        },

        /**
         * Clean up resources
         */
        onExit() {
            if (this._updateTimer) {
                clearInterval(this._updateTimer);
            }

            if (this._dependencyGraphDialog) {
                this._dependencyGraphDialog.destroy();
            }
        }
    });
});