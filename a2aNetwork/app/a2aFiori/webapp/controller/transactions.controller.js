sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log",
    "sap/viz/ui5/controls/VizFrame",
    "sap/viz/ui5/data/FlattenedDataset",
    "sap/viz/ui5/format/ChartFormatter",
    "sap/viz/ui5/api/env/Format",
    "sap/ui/export/library",
    "sap/ui/export/Spreadsheet",
    "sap/ui/core/format/NumberFormat",
    "sap/ui/core/format/DateFormat"
], function(BaseController, MessageToast, MessageBox, JSONModel, Filter, FilterOperator, Log,
    VizFrame, FlattenedDataset, ChartFormatter, Format, exportLibrary, Spreadsheet, NumberFormat, DateFormat) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Transactions", {

        onInit() {
            // Initialize comprehensive transactions model with professional data structure
            const _oTransactionsModel = new JSONModel({
                totalCount: 0,
                successCount: 0,
                failedCount: 0,
                pendingCount: 0,
                displayedCount: 0,
                totalVolume: 0,
                averageGasFee: 0,
                transactionsPerSecond: 0,
                networkLatency: 0,
                blockHeight: 0,
                gasPrice: 0,
                items: [],
                // Enhanced analytics data
                hourlyVolume: [],
                transactionTypes: [],
                gasUsageData: [],
                performanceHistory: []
            });
            this.getView().setModel(oTransactionsModel, "transactions");

            // Initialize UI state model for professional loading states
            const oUIModel = new JSONModel({
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                progressValue: 0,
                progressText: "",
                progressTitle: "",
                loadingMessage: "",
                errorMessage: "",
                selectedTab: "overview",
                timeRange: "24h",
                autoRefreshEnabled: true,
                selectedFilters: {
                    type: "all",
                    status: "all",
                    dateFrom: null,
                    dateTo: null
                },
                sortBy: "timestamp",
                sortDescending: true,
                searchQuery: "",
                pageSize: 50,
                currentPage: 1,
                validationErrors: {},
                lastRefresh: null,
                isRealTimeEnabled: true
            });
            this.getView().setModel(oUIModel, "ui");

            // Initialize performance metrics model
            this._initializePerformanceMetrics();

            // Load initial data with professional loading states
            this._loadTransactions();

            // Set up real-time monitoring and auto-refresh
            this._setupRealTimeMonitoring();

            // Initialize formatters for charts
            Format.numericFormatter(ChartFormatter.getInstance());

            // Setup keyboard shortcuts for power users
            this._setupKeyboardShortcuts();

            Log.info("Professional Transactions controller initialized with advanced features");
        },

        _initializePerformanceMetrics() {
            // Initialize comprehensive performance metrics for transaction analytics
            const oPerformanceModel = new JSONModel({
                networkStats: {
                    currentTPS: 0,
                    peakTPS: 0,
                    averageConfirmationTime: 0,
                    currentGasPrice: 0,
                    networkUtilization: 0
                },
                volumeMetrics: {
                    last24h: 0,
                    last7d: 0,
                    last30d: 0,
                    totalVolume: 0
                },
                feeAnalytics: {
                    averageFee: 0,
                    medianFee: 0,
                    totalFees: 0,
                    feeDistribution: []
                },
                transactionDistribution: {
                    byType: [],
                    byStatus: [],
                    byHour: [],
                    byValue: []
                },
                realTimeData: {
                    recentTransactions: [],
                    pendingPool: 0,
                    mempoolSize: 0,
                    blockTime: 0
                }
            });
            this.getView().setModel(oPerformanceModel, "performance");

            // Initialize charts after model is set
            setTimeout(() => {
                this._initializeCharts();
            }, 100);
        },

        async _loadTransactions(sTimeRange = "24h", oFilters = {}) {
            this._showLoadingState("progress", {
                title: this.getResourceBundle().getText("loadingTransactions"),
                message: this.getResourceBundle().getText("fetchingBlockchainData"),
                value: 10
            });

            try {
                this._updateProgress(20, this.getResourceBundle().getText("connectingToBlockchain"));

                // Get blockchain service
                const blockchainService = await this.getOwnerComponent().getBlockchainService();

                this._updateProgress(40, this.getResourceBundle().getText("queryingTransactions"));

                // Build comprehensive query parameters
                const queryParams = {
                    timeRange: sTimeRange,
                    includeAnalytics: true,
                    includeGasMetrics: true,
                    includePerformanceData: true,
                    ...oFilters
                };

                // Load transactions with analytics
                const result = await blockchainService.getTransactions(queryParams);

                this._updateProgress(60, this.getResourceBundle().getText("processingTransactionData"));

                // Load additional analytics data
                const [volumeData, feeAnalytics, networkStats] = await Promise.all([
                    blockchainService.getVolumeMetrics(sTimeRange),
                    blockchainService.getFeeAnalytics(sTimeRange),
                    blockchainService.getNetworkStats()
                ]);

                this._updateProgress(80, this.getResourceBundle().getText("updatingCharts"));

                // Update comprehensive model with all data
                const oModel = this.getView().getModel("transactions");
                const oPerformanceModel = this.getView().getModel("performance");

                // Enhanced transaction data with calculated metrics
                const processedItems = this._processTransactionData(result.items || []);
                const analytics = this._calculateAnalytics(processedItems);

                oModel.setData({
                    ...oModel.getData(),
                    totalCount: result.totalCount || 0,
                    successCount: result.successCount || 0,
                    failedCount: result.failedCount || 0,
                    pendingCount: result.pendingCount || 0,
                    displayedCount: processedItems.length,
                    totalVolume: analytics.totalVolume,
                    averageGasFee: analytics.averageGasFee,
                    transactionsPerSecond: analytics.transactionsPerSecond,
                    networkLatency: networkStats.latency || 0,
                    blockHeight: networkStats.blockHeight || 0,
                    gasPrice: networkStats.gasPrice || 0,
                    items: processedItems,
                    hourlyVolume: analytics.hourlyVolume,
                    transactionTypes: analytics.transactionTypes,
                    gasUsageData: analytics.gasUsageData
                });

                // Update performance metrics
                oPerformanceModel.setProperty("/networkStats", {
                    currentTPS: networkStats.tps || 0,
                    peakTPS: networkStats.peakTps || 0,
                    averageConfirmationTime: networkStats.avgConfirmationTime || 0,
                    currentGasPrice: networkStats.gasPrice || 0,
                    networkUtilization: networkStats.utilization || 0
                });

                oPerformanceModel.setProperty("/volumeMetrics", volumeData);
                oPerformanceModel.setProperty("/feeAnalytics", feeAnalytics);

                this._updateProgress(100, this.getResourceBundle().getText("transactionsLoaded"));

                // Update charts with new data
                this._updateChartData(analytics);

                // Update UI state
                const oUIModel = this.getView().getModel("ui");
                oUIModel.setProperty("/lastRefresh", new Date());
                oUIModel.setProperty("/hasNoData", processedItems.length === 0);

                this._hideLoadingState();

                Log.info(`Loaded ${processedItems.length} transactions with comprehensive analytics`);

            } catch (error) {
                Log.error("Failed to load transactions", error);
                this._showErrorState(this.getResourceBundle().getText("transactionsLoadError"));
            }
        },

        _processTransactionData(aItems) {
            return aItems.map(item => ({
                ...item,
                formattedValue: this._formatCurrency(item.value),
                formattedGasFee: this._formatCurrency(item.gasFee),
                formattedTimestamp: this._formatDateTime(item.timestamp),
                shortHash: this._formatAddress(item.hash),
                shortFrom: this._formatAddress(item.from),
                shortTo: this._formatAddress(item.to),
                confirmationTime: item.confirmationTime || 0,
                gasEfficiency: this._calculateGasEfficiency(item),
                valueCategory: this._categorizeValue(item.value)
            }));
        },

        _calculateAnalytics(aItems) {
            const totalVolume = aItems.reduce((sum, item) => sum + (parseFloat(item.value) || 0), 0);
            const totalGasFees = aItems.reduce((sum, item) => sum + (parseFloat(item.gasFee) || 0), 0);
            const averageGasFee = aItems.length > 0 ? totalGasFees / aItems.length : 0;

            // Calculate TPS (assuming data represents recent period)
            const timeSpan = aItems.length > 0 ?
                (new Date(aItems[0].timestamp).getTime() -
                 new Date(aItems[aItems.length - 1].timestamp).getTime()) / 1000 : 1;
            const transactionsPerSecond = timeSpan > 0 ? aItems.length / timeSpan : 0;

            // Calculate hourly volume distribution
            const hourlyVolume = this._calculateHourlyDistribution(aItems, "value");

            // Calculate transaction type distribution
            const typeCount = {};
            aItems.forEach(item => {
                typeCount[item.type] = (typeCount[item.type] || 0) + 1;
            });
            const transactionTypes = Object.keys(typeCount).map(type => ({
                type,
                count: typeCount[type],
                percentage: (typeCount[type] / aItems.length * 100).toFixed(1)
            }));

            // Calculate gas usage distribution
            const gasUsageData = this._calculateGasUsageDistribution(aItems);

            return {
                totalVolume,
                averageGasFee,
                transactionsPerSecond,
                hourlyVolume,
                transactionTypes,
                gasUsageData
            };
        },

        _setupRealTimeMonitoring() {
            // Enhanced real-time monitoring with multiple intervals
            this._realTimeInterval = setInterval(async() => {
                const oUIModel = this.getView().getModel("ui");
                if (!oUIModel.getProperty("/isRealTimeEnabled")) {
                    return;
                }

                await this._refreshRealTimeData();
            }, 10000); // Every 10 seconds

            // Pending transactions refresh
            this._pendingRefreshInterval = setInterval(async() => {
                const oModel = this.getView().getModel("transactions");
                if (oModel.getProperty("/pendingCount") > 0) {
                    await this._refreshPendingTransactions();
                }
            }, 5000); // Every 5 seconds for pending transactions

            // Network stats refresh
            this._networkStatsInterval = setInterval(async() => {
                await this._refreshNetworkStats();
            }, 15000); // Every 15 seconds
        },

        async _refreshRealTimeData() {
            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const recentTxs = await blockchainService.getRecentTransactions({ limit: 10 });

                // Update real-time data in performance model
                const oPerformanceModel = this.getView().getModel("performance");
                oPerformanceModel.setProperty("/realTimeData/recentTransactions", recentTxs);

                // Update any new transactions in the main model
                this._mergeNewTransactions(recentTxs);

            } catch (error) {
                Log.warning("Failed to refresh real-time data", error);
            }
        },

        async _refreshPendingTransactions() {
            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const oModel = this.getView().getModel("transactions");
                const aItems = oModel.getProperty("/items");

                // Find pending transactions
                const aPendingTxs = aItems.filter(tx => tx.status === "pending");

                if (aPendingTxs.length === 0) {
                    return;
                }

                let bUpdated = false;

                // Check status of each pending transaction with batching for efficiency
                const updates = await Promise.all(
                    aPendingTxs.map(tx => blockchainService.getTransactionStatus(tx.hash))
                );

                updates.forEach((updatedTx, index) => {
                    if (updatedTx.status !== "pending") {
                        const originalTx = aPendingTxs[index];
                        const iIndex = aItems.findIndex(item => item.hash === originalTx.hash);
                        if (iIndex >= 0) {
                            aItems[iIndex] = { ...aItems[iIndex], ...updatedTx };
                            bUpdated = true;

                            // Show notification for completed transactions
                            if (updatedTx.status === "success") {
                                MessageToast.show(
                                    this.getResourceBundle().getText("transactionConfirmed", [this._formatAddress(updatedTx.hash)])
                                );
                            }
                        }
                    }
                });

                if (bUpdated) {
                    // Update model and recalculate analytics
                    oModel.setProperty("/items", aItems);
                    this._updateCounts();
                    this._updateAnalytics(aItems);
                }

            } catch (error) {
                Log.error("Failed to refresh pending transactions", error);
            }
        },

        async _refreshNetworkStats() {
            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const networkStats = await blockchainService.getNetworkStats();

                const oPerformanceModel = this.getView().getModel("performance");
                oPerformanceModel.setProperty("/networkStats", {
                    currentTPS: networkStats.tps || 0,
                    peakTPS: networkStats.peakTps || 0,
                    averageConfirmationTime: networkStats.avgConfirmationTime || 0,
                    currentGasPrice: networkStats.gasPrice || 0,
                    networkUtilization: networkStats.utilization || 0
                });

                // Update blockchain-specific loading states
                const oUIModel = this.getView().getModel("ui");
                oUIModel.setProperty("/blockchainStep", `Block Height: ${networkStats.blockHeight || 0}`);

            } catch (error) {
                Log.warning("Failed to refresh network stats", error);
            }
        },

        _updateCounts() {
            const oModel = this.getView().getModel("transactions");
            const aItems = oModel.getProperty("/items");

            const counts = {
                totalCount: aItems.length,
                successCount: aItems.filter(tx => tx.status === "success").length,
                failedCount: aItems.filter(tx => tx.status === "failed").length,
                pendingCount: aItems.filter(tx => tx.status === "pending").length,
                displayedCount: aItems.length
            };

            Object.keys(counts).forEach(key => {
                oModel.setProperty(`/${key}`, counts[key]);
            });
        },

        _updateAnalytics(aItems) {
            const analytics = this._calculateAnalytics(aItems);
            const oModel = this.getView().getModel("transactions");

            oModel.setProperty("/totalVolume", analytics.totalVolume);
            oModel.setProperty("/averageGasFee", analytics.averageGasFee);
            oModel.setProperty("/transactionsPerSecond", analytics.transactionsPerSecond);
            oModel.setProperty("/hourlyVolume", analytics.hourlyVolume);
            oModel.setProperty("/transactionTypes", analytics.transactionTypes);
            oModel.setProperty("/gasUsageData", analytics.gasUsageData);

            // Update charts
            this._updateChartData(analytics);
        },

        _mergeNewTransactions(aNewTransactions) {
            const oModel = this.getView().getModel("transactions");
            const aExistingItems = oModel.getProperty("/items") || [];
            const aExistingHashes = aExistingItems.map(item => item.hash);

            // Filter out transactions that already exist
            const aUniqueNewTxs = aNewTransactions.filter(tx => !aExistingHashes.includes(tx.hash));

            if (aUniqueNewTxs.length > 0) {
                const aProcessedNewTxs = this._processTransactionData(aUniqueNewTxs);
                const aUpdatedItems = [...aProcessedNewTxs, ...aExistingItems].slice(0, 1000); // Limit to 1000 items

                oModel.setProperty("/items", aUpdatedItems);
                this._updateCounts();
                this._updateAnalytics(aUpdatedItems);

                Log.debug(`Added ${aUniqueNewTxs.length} new transactions`);
            }
        },

        // Professional loading state management
        _showLoadingState(sType, oOptions = {}) {
            const oUIModel = this.getView().getModel("ui");

            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: sType === "skeleton",
                isLoadingSpinner: sType === "spinner",
                isLoadingProgress: sType === "progress",
                isLoadingBlockchain: sType === "blockchain",
                hasError: false,
                hasNoData: false,
                loadingMessage: oOptions.message || "",
                progressValue: oOptions.value || 0,
                progressText: oOptions.text || `${oOptions.value || 0}%`,
                progressTitle: oOptions.title || ""
            });
        },

        _updateProgress(iValue, sMessage) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/progressValue", iValue);
            oUIModel.setProperty("/progressText", `${iValue}%`);
            oUIModel.setProperty("/loadingMessage", sMessage);
        },

        _hideLoadingState() {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false
            });
        },

        _showErrorState(sMessage) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: true,
                hasNoData: false,
                errorMessage: sMessage
            });
        },

        // Utility functions for data formatting
        _formatCurrency(value) {
            const formatter = NumberFormat.getCurrencyInstance({ currencyCode: false });
            return formatter.format(parseFloat(value) || 0, "ETH");
        },

        _formatDateTime(timestamp) {
            const formatter = DateFormat.getDateTimeInstance({ style: "medium" });
            return formatter.format(new Date(timestamp));
        },

        _formatAddress(address) {
            if (!address || typeof address !== "string") {
                return "";
            }
            if (!/^0x[a-fA-F0-9]{40,64}$/.test(address)) {
                return address;
            }
            return `${address.substring(0, 6) }...${ address.substring(address.length - 4)}`;
        },

        _calculateGasEfficiency(transaction) {
            const gasUsed = parseFloat(transaction.gasUsed) || 0;
            const gasLimit = parseFloat(transaction.gasLimit) || 1;
            return gasLimit > 0 ? (gasUsed / gasLimit * 100).toFixed(1) : 0;
        },

        _categorizeValue(value) {
            const val = parseFloat(value) || 0;
            if (val === 0) {
                return "zero";
            }
            if (val < 0.01) {
                return "micro";
            }
            if (val < 0.1) {
                return "small";
            }
            if (val < 1) {
                return "medium";
            }
            if (val < 10) {
                return "large";
            }
            return "whale";
        },

        _calculateHourlyDistribution(aItems, sProperty) {
            const hourlyData = {};

            aItems.forEach(item => {
                const hour = new Date(item.timestamp).getHours();
                const value = parseFloat(item[sProperty]) || 0;
                hourlyData[hour] = (hourlyData[hour] || 0) + value;
            });

            return Array.from({ length: 24 }, (_, hour) => ({
                hour: `${hour.toString().padStart(2, "0") }:00`,
                value: hourlyData[hour] || 0
            }));
        },

        _calculateGasUsageDistribution(aItems) {
            const ranges = [
                { label: "Low (0-21K)", min: 0, max: 21000, count: 0 },
                { label: "Normal (21K-50K)", min: 21001, max: 50000, count: 0 },
                { label: "High (50K-100K)", min: 50001, max: 100000, count: 0 },
                { label: "Very High (100K+)", min: 100001, max: Infinity, count: 0 }
            ];

            aItems.forEach(item => {
                const gasUsed = parseFloat(item.gasUsed) || 0;
                ranges.forEach(range => {
                    if (gasUsed >= range.min && gasUsed <= range.max) {
                        range.count++;
                    }
                });
            });

            return ranges.map(range => ({
                category: range.label,
                count: range.count,
                percentage: aItems.length > 0 ? (range.count / aItems.length * 100).toFixed(1) : 0
            }));
        },

        _setupKeyboardShortcuts() {
            // Setup keyboard shortcuts for power users
            document.addEventListener("keydown", (event) => {
                if (event.ctrlKey || event.metaKey) {
                    switch (event.key) {
                    case "r":
                        event.preventDefault();
                        this.onRefresh();
                        break;
                    case "e":
                        event.preventDefault();
                        this.onExport();
                        break;
                    case "f":
                        event.preventDefault();
                        const searchField = this.byId("searchField");
                        if (searchField) {
                            searchField.focus();
                        }
                        break;
                    }
                }
            });
        },

        onRefresh() {
            const oUIModel = this.getView().getModel("ui");
            const sTimeRange = oUIModel.getProperty("/timeRange");
            this._loadTransactions(sTimeRange);
        },

        onTimeRangeSelect(oEvent) {
            const sTimeRange = oEvent.getParameter("key") || oEvent.getSource().getSelectedKey();
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/timeRange", sTimeRange);

            this._showLoadingState("spinner", {
                message: this.getResourceBundle().getText("loadingTimeRange", [sTimeRange])
            });

            this._loadTransactions(sTimeRange);
        },

        onSearch(oEvent) {
            const sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/searchQuery", sQuery);

            // Debounce search to avoid too many API calls
            if (this._searchTimeout) {
                clearTimeout(this._searchTimeout);
            }

            this._searchTimeout = setTimeout(() => {
                this._applyFilters({ search: sQuery });
            }, 300);
        },

        onTabSelect(oEvent) {
            const sKey = oEvent.getParameter("key");
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/selectedTab", sKey);

            Log.debug("Transactions tab selected", sKey);

            // Load tab-specific data
            switch (sKey) {
            case "analytics":
                this._refreshAnalyticsData();
                break;
            case "realtime":
                this._refreshRealTimeData();
                break;
            case "overview":
                this._refreshOverviewData();
                break;
            }
        },

        onTypeFilterChange(oEvent) {
            const sType = oEvent.getSource().getSelectedKey();
            this._applyFilters({ type: sType === "all" ? null : sType });
        },

        onStatusFilterChange(oEvent) {
            const sStatus = oEvent.getSource().getSelectedKey();
            this._applyFilters({ status: sStatus === "all" ? null : sStatus });
        },

        onDateRangeChange(oEvent) {
            const oDateRange = oEvent.getSource();
            const dFrom = oDateRange.getDateValue();
            const dTo = oDateRange.getSecondDateValue();

            if (dFrom && dTo) {
                this._applyFilters({
                    dateFrom: dFrom.toISOString(),
                    dateTo: dTo.toISOString()
                });
            }
        },

        _applyFilters(oFilters = {}) {
            const oTable = this.byId("transactionsTable");
            if (!oTable) {
                return;
            }

            const _oBinding = oTable.getBinding("items");
            if (!oBinding) {
                return;
            }

            const oUIModel = this.getView().getModel("ui");
            const currentFilters = oUIModel.getProperty("/selectedFilters");

            // Update filters state
            const updatedFilters = { ...currentFilters, ...oFilters };
            oUIModel.setProperty("/selectedFilters", updatedFilters);

            const aFilters = [];

            // Apply search filter with enhanced search across multiple fields
            if (updatedFilters.search && updatedFilters.search.length > 0) {
                aFilters.push(new Filter([
                    new Filter("hash", FilterOperator.Contains, updatedFilters.search),
                    new Filter("shortHash", FilterOperator.Contains, updatedFilters.search),
                    new Filter("from", FilterOperator.Contains, updatedFilters.search),
                    new Filter("to", FilterOperator.Contains, updatedFilters.search),
                    new Filter("shortFrom", FilterOperator.Contains, updatedFilters.search),
                    new Filter("shortTo", FilterOperator.Contains, updatedFilters.search),
                    new Filter("type", FilterOperator.Contains, updatedFilters.search),
                    new Filter("value", FilterOperator.Contains, updatedFilters.search)
                ], false));
            }

            // Apply type filter
            if (updatedFilters.type && updatedFilters.type !== "all") {
                aFilters.push(new Filter("type", FilterOperator.EQ, updatedFilters.type));
            }

            // Apply status filter
            if (updatedFilters.status && updatedFilters.status !== "all") {
                aFilters.push(new Filter("status", FilterOperator.EQ, updatedFilters.status));
            }

            // Apply date range filter
            if (updatedFilters.dateFrom && updatedFilters.dateTo) {
                aFilters.push(new Filter("timestamp", FilterOperator.BT,
                    updatedFilters.dateFrom, updatedFilters.dateTo));
            }

            oBinding.filter(aFilters);

            Log.debug("Applied filters", { filters: updatedFilters, filterCount: aFilters.length });
        },

        onTransactionPress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("transactions");
            const _sHash = oContext.getProperty("hash");

            // Navigate to transaction detail (if implemented) or show details dialog
            this._showTransactionDetails(oContext.getObject());
        },

        onHashPress(oEvent) {
            const _sHash = oEvent.getSource().getText();
            const sExplorerUrl = `${this.getOwnerComponent().getBlockExplorerUrl() }/tx/${ sHash}`;
            window.open(sExplorerUrl, "_blank");
        },

        onCopyHash(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("transactions");
            const _sHash = oContext.getProperty("hash");

            navigator.clipboard.writeText(sHash).then(() => {
                MessageToast.show(this.getResourceBundle().getText("hashCopied"));
            });
        },

        onAddressPress(oEvent) {
            const _sAddress = oEvent.getSource().getText();

            // Try to navigate to agent detail if it's an agent address
            this.getRouter().navTo("agentDetail", {
                agentId: sAddress
            });
        },

        onViewDetails(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("transactions");
            this._showTransactionDetails(oContext.getObject());
        },

        _showTransactionDetails(oTransaction) {
            if (!this._oTransactionDialog) {
                this._oTransactionDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.TransactionDetails",
                    this
                );
                this.getView().addDependent(this._oTransactionDialog);
            }

            // Set transaction data with enhanced information
            const oDialogModel = new JSONModel({
                transaction: {
                    ...oTransaction,
                    formattedValue: this._formatCurrency(oTransaction.value),
                    formattedGasFee: this._formatCurrency(oTransaction.gasFee),
                    formattedTimestamp: this._formatDateTime(oTransaction.timestamp),
                    gasEfficiency: this._calculateGasEfficiency(oTransaction),
                    confirmationStatus: this._getConfirmationStatus(oTransaction),
                    explorerUrl: `${this.getOwnerComponent().getBlockExplorerUrl() }/tx/${ oTransaction.hash}`
                }
            });
            this._oTransactionDialog.setModel(oDialogModel, "dialog");

            this._oTransactionDialog.open();
        },

        _getConfirmationStatus(oTransaction) {
            if (oTransaction.status === "pending") {
                return { state: "Warning", text: "Pending Confirmation" };
            } else if (oTransaction.status === "success") {
                const confirmations = oTransaction.confirmations || 0;
                if (confirmations >= 6) {
                    return { state: "Success", text: `Confirmed (${confirmations} blocks)` };
                }
                return { state: "Warning", text: `${confirmations}/6 Confirmations` };

            }
            return { state: "Error", text: "Failed" };

        },

        onViewOnExplorer(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("transactions");
            const _sHash = oContext.getProperty("hash");
            const sExplorerUrl = `${this.getOwnerComponent().getBlockExplorerUrl() }/tx/${ sHash}`;
            window.open(sExplorerUrl, "_blank");
        },

        async onCheckStatus(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("transactions");
            const _sHash = oContext.getProperty("hash");

            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const updatedTx = await blockchainService.getTransactionStatus(sHash);

                // Update the transaction in the model
                const oModel = this.getView().getModel("transactions");
                const aItems = oModel.getProperty("/items");
                const iIndex = aItems.findIndex(item => item.hash === sHash);

                if (iIndex >= 0) {
                    aItems[iIndex] = { ...aItems[iIndex], ...updatedTx };
                    oModel.setProperty("/items", aItems);
                    this._updateCounts();
                }

                MessageToast.show(this.getResourceBundle().getText("statusUpdated"));

            } catch (error) {
                Log.error("Failed to check transaction status", error);
                MessageBox.error(this.getResourceBundle().getText("statusCheckError"));
            }
        },

        onExport() {
            this._showLoadingState("spinner", {
                message: this.getResourceBundle().getText("preparingExport")
            });

            const oModel = this.getView().getModel("transactions");
            const aItems = oModel.getProperty("/items");

            if (!aItems || aItems.length === 0) {
                MessageBox.warning(this.getResourceBundle().getText("noDataToExport"));
                this._hideLoadingState();
                return;
            }

            // Enhanced export with comprehensive data
            const _aColumns = [
                { label: "Transaction Hash", property: "hash", type: "string" },
                { label: "Type", property: "type", type: "string" },
                { label: "Status", property: "status", type: "string" },
                { label: "From Address", property: "from", type: "string" },
                { label: "To Address", property: "to", type: "string" },
                { label: "Value (ETH)", property: "value", type: "number" },
                { label: "Gas Used", property: "gasUsed", type: "number" },
                { label: "Gas Fee (ETH)", property: "gasFee", type: "number" },
                { label: "Gas Efficiency (%)", property: "gasEfficiency", type: "number" },
                { label: "Block Number", property: "blockNumber", type: "number" },
                { label: "Confirmation Time (s)", property: "confirmationTime", type: "number" },
                { label: "Timestamp", property: "timestamp", type: "date" },
                { label: "Value Category", property: "valueCategory", type: "string" }
            ];

            // Create Excel export using SAP UI5 Spreadsheet
            const oSpreadsheet = new Spreadsheet({
                workbook: {
                    columns: aColumns,
                    context: {
                        sheetName: "A2A Transactions Export",
                        metaInfo: [
                            { name: "Export Date", value: new Date().toLocaleString() },
                            { name: "Total Transactions", value: aItems.length },
                            { name: "Export Filter", value: this._getActiveFiltersText() }
                        ]
                    }
                },
                dataSource: aItems,
                fileName: `a2a-transactions-${new Date().toISOString().split("T")[0]}.xlsx`
            });

            oSpreadsheet.build().finally(() => {
                this._hideLoadingState();
                MessageToast.show(this.getResourceBundle().getText("transactionsExported"));
            });
        },

        _getActiveFiltersText() {
            const oUIModel = this.getView().getModel("ui");
            const filters = oUIModel.getProperty("/selectedFilters");
            const activeFilters = [];

            if (filters.search) {
                activeFilters.push(`Search: ${filters.search}`);
            }
            if (filters.type && filters.type !== "all") {
                activeFilters.push(`Type: ${filters.type}`);
            }
            if (filters.status && filters.status !== "all") {
                activeFilters.push(`Status: ${filters.status}`);
            }
            if (filters.dateFrom && filters.dateTo) {
                activeFilters.push(`Date Range: ${filters.dateFrom} - ${filters.dateTo}`);
            }

            return activeFilters.length > 0 ? activeFilters.join(", ") : "No filters applied";
        },

        onSettings() {
            this.getRouter().navTo("settings");
        },

        onTableSettings() {
            if (!this._oTableSettingsDialog) {
                this._oTableSettingsDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.TransactionTableSettings",
                    this
                );
                this.getView().addDependent(this._oTableSettingsDialog);
            }
            this._oTableSettingsDialog.open();
        },

        // Analytics refresh handlers
        async _refreshAnalyticsData() {
            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const [volumeMetrics, feeAnalytics] = await Promise.all([
                    blockchainService.getVolumeMetrics("7d"),
                    blockchainService.getFeeAnalytics("7d")
                ]);

                const oPerformanceModel = this.getView().getModel("performance");
                oPerformanceModel.setProperty("/volumeMetrics", volumeMetrics);
                oPerformanceModel.setProperty("/feeAnalytics", feeAnalytics);

                this._updateAnalyticsCharts();

            } catch (error) {
                Log.warning("Failed to refresh analytics data", error);
            }
        },

        _refreshOverviewData() {
            // Refresh overview tab data
            this._loadTransactions();
        },

        // Chart initialization and management
        _initializeCharts() {
            try {
                this._initializeVolumeChart();
                this._initializeTypeDistributionChart();
                this._initializeGasUsageChart();
                this._initializePerformanceChart();
            } catch (error) {
                Log.error("Failed to initialize charts", error);
            }
        },

        _initializeVolumeChart() {
            const oVizFrame = this.byId("volumeChart");
            if (!oVizFrame) {
                return;
            }

            oVizFrame.setVizType("line");
            oVizFrame.setUiConfig({ "applicationSet": "fiori" });

            const oDataset = new FlattenedDataset({
                dimensions: [{ name: "Hour", value: "{hour}" }],
                measures: [{ name: "Volume", value: "{value}" }],
                data: { path: "transactions>/hourlyVolume" }
            });

            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(this.getView().getModel("transactions"));

            // Configure feeds
            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis", type: "Measure", values: ["Volume"]
            }));
            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis", type: "Dimension", values: ["Hour"]
            }));
        },

        _initializeTypeDistributionChart() {
            const oVizFrame = this.byId("typeDistributionChart");
            if (!oVizFrame) {
                return;
            }

            oVizFrame.setVizType("donut");
            oVizFrame.setUiConfig({ "applicationSet": "fiori" });

            const oDataset = new FlattenedDataset({
                dimensions: [{ name: "Type", value: "{type}" }],
                measures: [{ name: "Count", value: "{count}" }],
                data: { path: "transactions>/transactionTypes" }
            });

            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(this.getView().getModel("transactions"));

            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "size", type: "Measure", values: ["Count"]
            }));
            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "color", type: "Dimension", values: ["Type"]
            }));
        },

        _initializeGasUsageChart() {
            const oVizFrame = this.byId("gasUsageChart");
            if (!oVizFrame) {
                return;
            }

            oVizFrame.setVizType("column");
            oVizFrame.setUiConfig({ "applicationSet": "fiori" });

            const oDataset = new FlattenedDataset({
                dimensions: [{ name: "Category", value: "{category}" }],
                measures: [{ name: "Count", value: "{count}" }],
                data: { path: "transactions>/gasUsageData" }
            });

            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(this.getView().getModel("transactions"));

            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis", type: "Measure", values: ["Count"]
            }));
            oVizFrame.addFeed(new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis", type: "Dimension", values: ["Category"]
            }));
        },

        _initializePerformanceChart() {
            const oVizFrame = this.byId("performanceChart");
            if (!oVizFrame) {
                return;
            }

            // Performance chart showing TPS and confirmation times
            oVizFrame.setVizType("combination");
            oVizFrame.setUiConfig({ "applicationSet": "fiori" });

            // This would use performance history data from the model
            Log.info("Performance chart initialized");
        },

        _updateChartData(analytics) {
            // Update all charts with new analytics data
            const _oTransactionsModel = this.getView().getModel("transactions");

            // Refresh chart bindings
            ["volumeChart", "typeDistributionChart", "gasUsageChart", "performanceChart"].forEach(chartId => {
                const oChart = this.byId(chartId);
                if (oChart && oChart.getDataset()) {
                    oChart.getDataset().getBinding("data").refresh();
                }
            });
        },

        _updateAnalyticsCharts() {
            // Update analytics-specific charts
            this._updateChartData(this._calculateAnalytics(this.getView().getModel("transactions").getProperty("/items")));
        },

        onToggleRealTime(oEvent) {
            const bEnabled = oEvent.getParameter("state");
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/isRealTimeEnabled", bEnabled);

            if (bEnabled) {
                MessageToast.show(this.getResourceBundle().getText("realTimeEnabled"));
            } else {
                MessageToast.show(this.getResourceBundle().getText("realTimeDisabled"));
            }
        },

        onRetryLoad() {
            this.onRefresh();
        },

        onCloseError() {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/hasError", false);
        },

        onExit() {
            // Clean up all intervals and timeouts
            [this._realTimeInterval, this._pendingRefreshInterval,
                this._networkStatsInterval, this._searchTimeout].forEach(id => {
                if (id) {
                    clearInterval(id);
                }
            });

            // Clean up dialogs
            [this._oTransactionDialog].forEach(dialog => {
                if (dialog) {
                    dialog.destroy();
                }
            });

            Log.info("Transactions controller cleanup completed");
        }
    });
});