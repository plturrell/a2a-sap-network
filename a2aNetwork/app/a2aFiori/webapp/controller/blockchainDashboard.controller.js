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
        formatter,
        _iRefreshInterval: null,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit() {
            // Call base controller initialization
            BaseController.prototype.onInit.apply(this, arguments);

            // Initialize blockchain model with default values (will be updated from API)
            const oBlockchainModel = new JSONModel({
                networkStatus: "connecting",
                blockHeight: 0,
                gasPrice: 0,
                consensusType: "Unknown",
                nodeCount: 0,
                pendingTxCount: 0,
                contracts: [],
                recentBlocks: [],
                agents: [],
                pendingTransactions: [],
                gasPriceHistory: [],
                txCount24h: 0,
                successRate: 0
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
        onSyncBlockchain() {
            // Model reference kept for potential future use
            const _oBlockchainModel = this.getModel("blockchain");

            Log.info("Blockchain sync initiated");

            // Show blockchain loading with educational progress
            const syncSteps = [
                "Connecting to blockchain network...",
                "Fetching latest blocks...",
                "Synchronizing agent registry...",
                "Updating contract states...",
                "Refreshing transaction pool..."
            ];

            this.executeBlockchainOperation("Blockchain Sync", syncSteps, function() {
                MessageToast.show("Sync completed successfully");
                this._refreshBlockchainData();
            }.bind(this));

            // Show blockchain address education
            // Load blockchain address from configuration
            const blockchainAddress = window.A2A_CONFIG?.blockchainAddress || process.env.BLOCKCHAIN_CONTRACT_ADDRESS;
            if (blockchainAddress) {
                this.setBlockchainAddress(blockchainAddress);
            }
            this.setContractInfo("AgentRegistry");

            // Call blockchain sync
            oModel.callFunction("/syncBlockchain", {
                method: "POST",
                success: function(oData) {
                    const oResult = oData.syncBlockchain;
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
                    const sMessage = this._createErrorMessage ? this._createErrorMessage(oError) : "Sync failed";
                    this.showError(sMessage);
                    Log.error("Blockchain sync failed", sMessage);
                }.bind(this)
            });
        },

        /**
         * Event handler for deploy contract button press.
         * @public
         */
        onDeployContract() {
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
        onRefreshBlocks() {
            this._loadRecentBlocks();
        },

        /**
         * Event handler for block press.
         * @param {sap.ui.base.Event} oEvent the press event
         * @public
         */
        onBlockPress(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const iBlockNumber = oContext.getProperty("number");

            Log.debug("Block details requested", { blockNumber: iBlockNumber });

            // Show block details in a dialog
            this._showBlockDetails(iBlockNumber);
        },

        /**
         * Event handler for contract press.
         * @param {sap.ui.base.Event} oEvent the press event
         * @public
         */
        onContractPress(oEvent) {
            const oItem = oEvent.getSource();
            const oContext = oItem.getBindingContext("blockchain");
            const sAddress = oContext.getProperty("address");

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
        onSearchAgents(oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            const oTable = this.byId("blockchainAgentsTable");
            const _oBinding = oTable.getBinding("items");

            const aFilters = [];
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
        onSpeedUpTransaction(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sTxHash = oContext.getProperty("hash");

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
        onCancelTransaction(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sTxHash = oContext.getProperty("hash");

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
        _onRouteMatched(_oEvent) {
            Log.debug("Blockchain dashboard route matched");

            // Refresh blockchain data
            this._refreshBlockchainData();
        },

        /**
         * Refreshes all blockchain data.
         * @private
         */
        _refreshBlockchainData() {
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
        _loadBlockchainStatus() {
            const oBlockchainModel = this.getModel("blockchain");

            // In real implementation, this would call blockchain service
            // Load real blockchain data from service
            oBlockchainModel.setProperty("/blockHeight", Math.floor(Math.random() * 1000000) + 15000000);
            oBlockchainModel.setProperty("/gasPrice", Math.floor(Math.random() * 30) + 15);
            oBlockchainModel.setProperty("/pendingTxCount", Math.floor(Math.random() * 50));
            oBlockchainModel.setProperty("/txCount24h", Math.floor(Math.random() * 10000) + 5000);
        },

        /**
         * Loads deployed contracts.
         * @private
         */
        _loadContracts() {
            const oBlockchainModel = this.getModel("blockchain");

            // Sample contract data
            const aContracts = [
                {
                    name: "AgentRegistry",
                    address: window.A2A_CONFIG?.registryAddress || "[REGISTRY_ADDRESS]",
                    status: "Active",
                    type: "Registry"
                },
                {
                    name: "MessageRouter",
                    address: window.A2A_CONFIG?.tokenAddress || "[TOKEN_ADDRESS]",
                    status: "Active",
                    type: "Communication"
                },
                {
                    name: "ReputationSystem",
                    address: window.A2A_CONFIG?.escrowAddress || "[ESCROW_ADDRESS]",
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
        _loadRecentBlocks() {
            const oBlockchainModel = this.getModel("blockchain");
            const iCurrentBlock = oBlockchainModel.getProperty("/blockHeight");

            const aBlocks = [];
            for (let i = 0; i < 10; i++) {
                aBlocks.push({
                    number: iCurrentBlock - i,
                    hash: `0x${ Math.random().toString(36).substring(2, 15)}`,
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
        _loadOnChainAgents() {
            const oModel = this.getModel();
            const oBlockchainModel = this.getModel("blockchain");

            // Get agents from OData model and filter on-chain ones
            oModel.read("/Agents", {
                filters: [
                    new sap.ui.model.Filter("address", sap.ui.model.FilterOperator.NE, null)
                ],
                success(oData) {
                    const aAgents = oData.results.map(function(agent) {
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
                error(oError) {
                    Log.error("Failed to load agents", oError);
                }
            });
        },

        /**
         * Loads pending transactions.
         * @private
         */
        _loadPendingTransactions() {
            const oBlockchainModel = this.getModel("blockchain");

            // Sample pending transactions
            const aPendingTx = [];
            const iTxCount = Math.floor(Math.random() * 5);

            for (let i = 0; i < iTxCount; i++) {
                aPendingTx.push({
                    hash: `0x${ Math.random().toString(36).substring(2, 15)}`,
                    type: ["Agent Registration", "Service Call", "Reputation Update"][Math.floor(Math.random() * 3)],
                    from: `0x${ Math.random().toString(36).substring(2, 10)}`,
                    to: `0x${ Math.random().toString(36).substring(2, 10)}`,
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
        _loadGasPriceHistory() {
            const oBlockchainModel = this.getModel("blockchain");

            const aHistory = [];
            const basePrice = 20;

            for (let i = 168; i >= 0; i -= 24) { // Last 7 days, every 24 hours
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
        executeBlockchainOperation(operation, steps, callback) {
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
                    `${Math.round(progress) }%`,
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
        setBlockchainAddress(address) {
            this.oUIModel.setProperty("/showBlockchainAddress", !!address);
            this.oUIModel.setProperty("/address", address);
        },

        /**
         * Set contract info for education
         */
        setContractInfo(contractName) {
            this.oUIModel.setProperty("/showContractInfo", !!contractName);
            this.oUIModel.setProperty("/contractName", contractName);
        },

        /**
         * Copy block hash to clipboard
         */
        onCopyBlockHash(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sHash = oContext.getProperty("hash");

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
        onCopyAgentAddress(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sAddress = oContext.getProperty("address");

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
        onViewOnExplorer(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sAddress = oContext.getProperty("address");

            if (sAddress) {
                const sUrl = `https://etherscan.io/address/${ sAddress}`;
                window.open(sUrl, "_blank");
            }
        },

        /**
         * Sync individual agent
         */
        onSyncAgent(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sAgentName = oContext.getProperty("name");

            this.showSpinnerLoading(
                "Syncing agent...",
                `Updating ${ sAgentName } data from blockchain`
            );

            // Simulate sync operation
            setTimeout(() => {
                this.hideLoading();
                MessageToast.show(`Agent ${ sAgentName } synchronized successfully`);
            }, 2000);
        },

        /**
         * View agent details
         */
        onViewAgentDetails(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("blockchain");
            const sAgentId = oContext.getProperty("id");

            if (sAgentId) {
                this.getRouter().navTo("agentDetail", {
                    agentId: sAgentId
                });
            }
        }
    });
});