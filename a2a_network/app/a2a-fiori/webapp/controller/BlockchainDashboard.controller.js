sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log"
], function(BaseController, MessageToast, MessageBox, formatter, JSONModel, Log) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.BlockchainDashboard", {
        formatter: formatter,
        _iRefreshInterval: null,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit: function() {
            // Call base controller initialization
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Initialize blockchain model
            var oBlockchainModel = new JSONModel({
                networkStatus: "connected",
                blockHeight: 0,
                gasPrice: 20,
                consensusType: "Proof of Authority",
                nodeCount: 4,
                pendingTxCount: 0,
                contracts: [],
                recentBlocks: [],
                agents: [],
                pendingTransactions: [],
                gasPriceHistory: [],
                txCount24h: 0,
                successRate: 98.5
            });
            this.setModel(oBlockchainModel, "blockchain");

            // Attach route matched handler
            this.getRouter().getRoute("blockchain").attachPatternMatched(this._onRouteMatched, this);

            // Set up automatic refresh every 10 seconds
            this._iRefreshInterval = setInterval(function() {
                this._refreshBlockchainData();
            }.bind(this), 10000);

            // Register for cleanup
            this._registerForCleanup(function() {
                if (this._iRefreshInterval) {
                    clearInterval(this._iRefreshInterval);
                    this._iRefreshInterval = null;
                }
            }.bind(this));

            Log.info("Blockchain Dashboard controller initialized");
        },

        /* =========================================================== */
        /* event handlers                                              */
        /* =========================================================== */

        /**
         * Event handler for blockchain sync button press.
         * @public
         */
        onSyncBlockchain: function() {
            var oModel = this.getModel();
            var oBlockchainModel = this.getModel("blockchain");

            Log.info("Blockchain sync initiated");

            // Show blockchain loading with educational progress
            var syncSteps = [
                "Connecting to blockchain network...",
                "Fetching latest blocks...",
                "Synchronizing agent registry...",
                "Updating contract states...",
                "Refreshing transaction pool..."
            ];
            
            this.simulateBlockchainOperation("Blockchain Sync", syncSteps, function() {
                MessageToast.show("Sync completed successfully");
                this._refreshBlockchainData();
            }.bind(this));

            // Show blockchain address education
            this.setBlockchainAddress("0x5fbdb2315678afecb367f032d93f642f64180aa3");
            this.setContractInfo("AgentRegistry");

            // Call blockchain sync
            oModel.callFunction("/syncBlockchain", {
                method: "POST",
                success: function(oData) {
                    var oResult = oData.syncBlockchain;
                    if (oResult && oResult.transactionHash) {
                        // Show transaction hash for education
                        this.setTransactionHash(oResult.transactionHash, "https://etherscan.io");
                        this.setGasInfo(oResult.gasUsed || 21000, "Success");
                        
                        if (oResult.confirmations !== undefined) {
                            this.setTransactionStatus(oResult.confirmations, 12, "Success");
                        }
                    }
                    
                    Log.info("Blockchain sync completed", oResult);
                }.bind(this),
                error: function(oError) {
                    var sMessage = this._createErrorMessage ? this._createErrorMessage(oError) : "Sync failed";
                    this.showError(sMessage);
                    Log.error("Blockchain sync failed", sMessage);
                }.bind(this)
            });
        },

        /**
         * Event handler for deploy contract button press.
         * @public
         */
        onDeployContract: function() {
            Log.info("Deploy contract dialog requested");
            
            if (!this._oDeployDialog) {
                this._createDeployDialog();
            }
            this._oDeployDialog.open();
        },

        /**
         * Event handler for refresh blocks button press.
         * @public
         */
        onRefreshBlocks: function() {
            this._loadRecentBlocks();
        },

        /**
         * Event handler for block press.
         * @param {sap.ui.base.Event} oEvent the press event
         * @public
         */
        onBlockPress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var iBlockNumber = oContext.getProperty("number");
            
            Log.debug("Block details requested", { blockNumber: iBlockNumber });
            
            // Show block details in a dialog
            this._showBlockDetails(iBlockNumber);
        },

        /**
         * Event handler for contract press.
         * @param {sap.ui.base.Event} oEvent the press event
         * @public
         */
        onContractPress: function(oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext("blockchain");
            var sAddress = oContext.getProperty("address");
            
            Log.debug("Contract details requested", { address: sAddress });
            
            // Navigate to contract details
            this.getRouter().navTo("contractDetail", {
                address: sAddress
            });
        },

        /**
         * Event handler for search agents.
         * @param {sap.ui.base.Event} oEvent the search event
         * @public
         */
        onSearchAgents: function(oEvent) {
            var sQuery = oEvent.getParameter("newValue");
            var oTable = this.byId("blockchainAgentsTable");
            var oBinding = oTable.getBinding("items");
            
            var aFilters = [];
            if (sQuery) {
                aFilters.push(new sap.ui.model.Filter({
                    filters: [
                        new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                        new sap.ui.model.Filter("address", sap.ui.model.FilterOperator.Contains, sQuery)
                    ],
                    and: false
                }));
            }
            
            // Always include onChain filter
            aFilters.push(new sap.ui.model.Filter("onChain", sap.ui.model.FilterOperator.EQ, true));
            
            oBinding.filter(aFilters, "Application");
        },

        /**
         * Event handler for speed up transaction button.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onSpeedUpTransaction: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sTxHash = oContext.getProperty("hash");
            
            Log.info("Speed up transaction requested", { txHash: sTxHash });
            
            MessageBox.confirm(
                this.getResourceBundle().getText("confirmSpeedUpTransaction"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._speedUpTransaction(sTxHash);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * Event handler for cancel transaction button.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onCancelTransaction: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sTxHash = oContext.getProperty("hash");
            
            Log.info("Cancel transaction requested", { txHash: sTxHash });
            
            MessageBox.warning(
                this.getResourceBundle().getText("confirmCancelTransaction"),
                {
                    actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.YES) {
                            this._cancelTransaction(sTxHash);
                        }
                    }.bind(this)
                }
            );
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Binds the view to the object path.
         * @function
         * @param {sap.ui.base.Event} oEvent pattern match event
         * @private
         */
        _onRouteMatched: function(oEvent) {
            Log.debug("Blockchain dashboard route matched");
            
            // Refresh blockchain data
            this._refreshBlockchainData();
        },

        /**
         * Refreshes all blockchain data.
         * @private
         */
        _refreshBlockchainData: function() {
            this._loadBlockchainStatus();
            this._loadContracts();
            this._loadRecentBlocks();
            this._loadOnChainAgents();
            this._loadPendingTransactions();
            this._loadGasPriceHistory();
        },

        /**
         * Loads blockchain status.
         * @private
         */
        _loadBlockchainStatus: function() {
            var oModel = this.getModel();
            var oBlockchainModel = this.getModel("blockchain");
            
            // In real implementation, this would call blockchain service
            // For now, simulate with sample data
            oBlockchainModel.setProperty("/blockHeight", Math.floor(Math.random() * 1000000) + 15000000);
            oBlockchainModel.setProperty("/gasPrice", Math.floor(Math.random() * 30) + 15);
            oBlockchainModel.setProperty("/pendingTxCount", Math.floor(Math.random() * 50));
            oBlockchainModel.setProperty("/txCount24h", Math.floor(Math.random() * 10000) + 5000);
        },

        /**
         * Loads deployed contracts.
         * @private
         */
        _loadContracts: function() {
            var oBlockchainModel = this.getModel("blockchain");
            
            // Sample contract data
            var aContracts = [
                {
                    name: "AgentRegistry",
                    address: "0x5fbdb2315678afecb367f032d93f642f64180aa3",
                    status: "Active",
                    type: "Registry"
                },
                {
                    name: "MessageRouter", 
                    address: "0xe7f1725e7734ce288f8367e1bb143e90bb3f0512",
                    status: "Active",
                    type: "Communication"
                },
                {
                    name: "ReputationSystem",
                    address: "0x9fe46736679d2d9a65f0992f2272de9f3c7fa6e0",
                    status: "Active",
                    type: "Governance"
                }
            ];
            
            oBlockchainModel.setProperty("/contracts", aContracts);
        },

        /**
         * Loads recent blocks.
         * @private
         */
        _loadRecentBlocks: function() {
            var oBlockchainModel = this.getModel("blockchain");
            var iCurrentBlock = oBlockchainModel.getProperty("/blockHeight");
            
            var aBlocks = [];
            for (var i = 0; i < 10; i++) {
                aBlocks.push({
                    number: iCurrentBlock - i,
                    hash: "0x" + Math.random().toString(36).substring(2, 15),
                    transactionCount: Math.floor(Math.random() * 200),
                    timestamp: new Date(Date.now() - i * 15000) // 15 seconds per block
                });
            }
            
            oBlockchainModel.setProperty("/recentBlocks", aBlocks);
        },

        /**
         * Loads on-chain agents.
         * @private
         */
        _loadOnChainAgents: function() {
            var oModel = this.getModel();
            var oBlockchainModel = this.getModel("blockchain");
            
            // Get agents from OData model and filter on-chain ones
            oModel.read("/Agents", {
                filters: [
                    new sap.ui.model.Filter("address", sap.ui.model.FilterOperator.NE, null)
                ],
                success: function(oData) {
                    var aAgents = oData.results.map(function(agent) {
                        return {
                            name: agent.name,
                            address: agent.address,
                            reputation: agent.reputation,
                            type: agent.type || "General",
                            onChain: true
                        };
                    });
                    oBlockchainModel.setProperty("/agents", aAgents);
                },
                error: function(oError) {
                    Log.error("Failed to load agents", oError);
                }
            });
        },

        /**
         * Loads pending transactions.
         * @private
         */
        _loadPendingTransactions: function() {
            var oBlockchainModel = this.getModel("blockchain");
            
            // Sample pending transactions
            var aPendingTx = [];
            var iTxCount = Math.floor(Math.random() * 5);
            
            for (var i = 0; i < iTxCount; i++) {
                aPendingTx.push({
                    hash: "0x" + Math.random().toString(36).substring(2, 15),
                    type: ["Agent Registration", "Service Call", "Reputation Update"][Math.floor(Math.random() * 3)],
                    from: "0x" + Math.random().toString(36).substring(2, 10),
                    to: "0x" + Math.random().toString(36).substring(2, 10),
                    gasPrice: Math.floor(Math.random() * 50) + 20,
                    submittedAt: new Date(Date.now() - Math.random() * 300000) // Last 5 minutes
                });
            }
            
            oBlockchainModel.setProperty("/pendingTransactions", aPendingTx);
        },

        /**
         * Loads gas price history.
         * @private
         */
        _loadGasPriceHistory: function() {
            var oBlockchainModel = this.getModel("blockchain");
            
            var aHistory = [];
            var basePrice = 20;
            
            for (var i = 168; i >= 0; i -= 24) { // Last 7 days, every 24 hours
                aHistory.push({
                    timestamp: new Date(Date.now() - i * 3600000).toISOString(),
                    price: basePrice + Math.random() * 20 - 10
                });
            }
            
            oBlockchainModel.setProperty("/gasPriceHistory", aHistory);
        },
        
        /**
         * Simulate blockchain operation with progress
         */
        simulateBlockchainOperation: function (operation, steps, callback) {
            let currentStep = 0;
            const totalSteps = steps.length;
            
            this.showProgressLoading(
                operation,
                0,
                "0%",
                steps[0],
                "None"
            );

            const progressTimer = setInterval(() => {
                currentStep++;
                const progress = (currentStep / totalSteps) * 100;
                
                this.showProgressLoading(
                    operation,
                    progress,
                    Math.round(progress) + "%",
                    currentStep < totalSteps ? steps[currentStep] : "Completed",
                    progress === 100 ? "Success" : "None"
                );

                if (currentStep >= totalSteps) {
                    clearInterval(progressTimer);
                    setTimeout(() => {
                        this.hideLoading();
                        if (callback) {
                            callback();
                        }
                    }, 1000);
                }
            }, 1500);
        },
        
        /**
         * Set blockchain address for education
         */
        setBlockchainAddress: function (address) {
            this.oUIModel.setProperty("/showBlockchainAddress", !!address);
            this.oUIModel.setProperty("/address", address);
        },
        
        /**
         * Set contract info for education
         */
        setContractInfo: function (contractName) {
            this.oUIModel.setProperty("/showContractInfo", !!contractName);
            this.oUIModel.setProperty("/contractName", contractName);
        },
        
        /**
         * Copy block hash to clipboard
         */
        onCopyBlockHash: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sHash = oContext.getProperty("hash");
            
            if (sHash && navigator.clipboard) {
                navigator.clipboard.writeText(sHash).then(() => {
                    MessageToast.show("Block hash copied to clipboard");
                }).catch(() => {
                    MessageToast.show("Could not copy hash");
                });
            }
        },
        
        /**
         * Copy agent address to clipboard
         */
        onCopyAgentAddress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sAddress = oContext.getProperty("address");
            
            if (sAddress && navigator.clipboard) {
                navigator.clipboard.writeText(sAddress).then(() => {
                    MessageToast.show("Address copied to clipboard");
                }).catch(() => {
                    MessageToast.show("Could not copy address");
                });
            }
        },
        
        /**
         * View address on block explorer
         */
        onViewOnExplorer: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sAddress = oContext.getProperty("address");
            
            if (sAddress) {
                var sUrl = "https://etherscan.io/address/" + sAddress;
                window.open(sUrl, "_blank");
            }
        },
        
        /**
         * Sync individual agent
         */
        onSyncAgent: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sAgentName = oContext.getProperty("name");
            
            this.showSpinnerLoading(
                "Syncing agent...",
                "Updating " + sAgentName + " data from blockchain"
            );
            
            // Simulate sync operation
            setTimeout(() => {
                this.hideLoading();
                MessageToast.show("Agent " + sAgentName + " synchronized successfully");
            }, 2000);
        },
        
        /**
         * View agent details
         */
        onViewAgentDetails: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("blockchain");
            var sAgentId = oContext.getProperty("id");
            
            if (sAgentId) {
                this.getRouter().navTo("agentDetail", {
                    agentId: sAgentId
                });
            }
        }
    });
});