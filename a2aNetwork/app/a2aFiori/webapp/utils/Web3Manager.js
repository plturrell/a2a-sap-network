sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/base/EventProvider",
    "sap/base/Log"
], (BaseObject, EventProvider, Log) => {
    "use strict";

    /**
     * Web3 Manager for handling blockchain connections and operations
     * Integrates with SAP authentication and enterprise security
     */
    return BaseObject.extend("a2a.network.fiori.utils.Web3Manager", {

        metadata: {
            events: {
                "connectionChanged": {
                    parameters: {
                        connected: { type: "boolean" },
                        account: { type: "string" },
                        network: { type: "string" }
                    }
                },
                "accountChanged": {
                    parameters: {
                        newAccount: { type: "string" },
                        oldAccount: { type: "string" }
                    }
                },
                "networkChanged": {
                    parameters: {
                        newNetwork: { type: "string" },
                        oldNetwork: { type: "string" }
                    }
                }
            }
        },

        constructor() {
            BaseObject.apply(this, arguments);
            EventProvider.apply(this, arguments);

            this._web3 = null;
            this._provider = null;
            this._currentAccount = null;
            this._currentNetwork = null;
            this._isConnected = false;
            this._contractAddresses = {};
            this._contracts = {};

            this._initialize();
        },

        /**
         * Initialize Web3 connection
         * @private
         */
        _initialize() {
            // Check if running in SAP BTP environment
            this._isSAPEnvironment = this._detectSAPEnvironment();

            // Load contract addresses from configuration
            this._loadContractAddresses();

            // Initialize Web3 provider
            this._initializeProvider();

            Log.info("Web3Manager initialized", {
                isSAPEnvironment: this._isSAPEnvironment,
                hasProvider: !!this._provider
            });
        },

        /**
         * Detect if running in SAP environment
         * @private
         * @returns {boolean} True if in SAP environment
         */
        _detectSAPEnvironment() {
            const sHostname = window.location.hostname;
            return sHostname.includes("cfapps") ||
                   sHostname.includes("hana.ondemand.com") ||
                   sHostname.includes("cloud.sap") ||
                   !!sap.ushell;
        },

        /**
         * Load contract addresses from SAP configuration service
         * @private
         */
        _loadContractAddresses() {
            // In SAP BTP, these would come from destination service or configuration
            this._contractAddresses = {
                governanceToken: process.env.GOVERNANCE_TOKEN_ADDRESS || "0x...",
                governor: process.env.GOVERNOR_ADDRESS || "0x...",
                timelock: process.env.TIMELOCK_ADDRESS || "0x...",
                agentMarketplace: process.env.AGENT_MARKETPLACE_ADDRESS || "0x...",
                reputation: process.env.REPUTATION_ADDRESS || "0x...",
                multisig: process.env.MULTISIG_ADDRESS || "0x..."
            };

            // In production, load from SAP destination service
            if (this._isSAPEnvironment) {
                this._loadContractAddressesFromSAP();
            }
        },

        /**
         * Load contract addresses from SAP destination service
         * @private
         */
        _loadContractAddressesFromSAP() {
            // This would integrate with SAP BTP destination service
            jQuery.ajax({
                url: "/destinations/blockchain-config",
                type: "GET",
                success: function(oData) {
                    if (oData && oData.contractAddresses) {
                        this._contractAddresses = Object.assign(this._contractAddresses, oData.contractAddresses);
                    }
                }.bind(this),
                error(oError) {
                    Log.warning("Could not load contract addresses from SAP destination", oError);
                }
            });
        },

        /**
         * Initialize Web3 provider
         * @private
         */
        _initializeProvider() {
            if (typeof window.ethereum !== "undefined") {
                this._provider = window.ethereum;
                this._web3 = new Web3(this._provider);

                // Setup event listeners
                this._setupProviderEventListeners();

                // Auto-connect if previously connected
                this._checkPreviousConnection();

            } else if (this._isSAPEnvironment) {
                // In SAP environment, might use enterprise wallet or custodial service
                this._initializeEnterpriseProvider();
            } else {
                Log.warning("No Web3 provider detected");
            }
        },

        /**
         * Initialize enterprise provider for SAP environment
         * @private
         */
        _initializeEnterpriseProvider() {
            // This would integrate with enterprise wallet solutions
            // or SAP's blockchain service offerings
            Log.info("Initializing enterprise blockchain provider");

            // Mock enterprise provider for now
            this._provider = {
                isEnterprise: true,
                request(params) {
                    return new Promise((resolve, reject) => {
                        // Handle enterprise wallet operations
                        Log.info("Enterprise wallet request", params);
                        resolve(null);
                    });
                }
            };
        },

        /**
         * Setup provider event listeners
         * @private
         */
        _setupProviderEventListeners() {
            if (!this._provider) {
                return;
            }

            // Account changed
            this._provider.on("accountsChanged", (accounts) => {
                const sOldAccount = this._currentAccount;
                this._currentAccount = accounts[0] || null;

                this.fireEvent("accountChanged", {
                    newAccount: this._currentAccount,
                    oldAccount: sOldAccount
                });

                this._updateConnectionState();

                Log.info("Account changed", {
                    from: sOldAccount,
                    to: this._currentAccount
                });
            });

            // Network changed
            this._provider.on("chainChanged", (chainId) => {
                const sOldNetwork = this._currentNetwork;
                this._currentNetwork = parseInt(chainId, 16).toString();

                this.fireEvent("networkChanged", {
                    newNetwork: this._currentNetwork,
                    oldNetwork: sOldNetwork
                });

                this._updateConnectionState();

                Log.info("Network changed", {
                    from: sOldNetwork,
                    to: this._currentNetwork
                });
            });

            // Disconnect
            this._provider.on("disconnect", () => {
                this._currentAccount = null;
                this._currentNetwork = null;
                this._updateConnectionState();

                Log.info("Web3 provider disconnected");
            });
        },

        /**
         * Check for previous connection
         * @private
         */
        _checkPreviousConnection() {
            if (this._provider && this._provider.selectedAddress) {
                this._getCurrentAccount().then((sAccount) => {
                    if (sAccount) {
                        this._currentAccount = sAccount;
                        this._getCurrentNetwork().then((sNetwork) => {
                            this._currentNetwork = sNetwork;
                            this._updateConnectionState();
                        });
                    }
                });
            }
        },

        /**
         * Update connection state and fire events
         * @private
         */
        _updateConnectionState() {
            const bWasConnected = this._isConnected;
            this._isConnected = !!(this._currentAccount && this._currentNetwork);

            if (bWasConnected !== this._isConnected) {
                this.fireEvent("connectionChanged", {
                    connected: this._isConnected,
                    account: this._currentAccount,
                    network: this._currentNetwork
                });
            }
        },

        /**
         * Connect to Web3 provider
         * @public
         * @returns {Promise} Promise that resolves when connected
         */
        connect() {
            if (!this._provider) {
                return Promise.reject(new Error("No Web3 provider available"));
            }

            return this._provider.request({ method: "eth_requestAccounts" })
                .then((accounts) => {
                    this._currentAccount = accounts[0];
                    return this._getCurrentNetwork();
                })
                .then((sNetwork) => {
                    this._currentNetwork = sNetwork;
                    this._updateConnectionState();

                    Log.info("Web3 connected", {
                        account: this._currentAccount,
                        network: this._currentNetwork
                    });

                    return {
                        account: this._currentAccount,
                        network: this._currentNetwork
                    };
                })
                .catch((oError) => {
                    Log.error("Web3 connection failed", oError);
                    throw oError;
                });
        },

        /**
         * Disconnect from Web3 provider
         * @public
         */
        disconnect() {
            this._currentAccount = null;
            this._currentNetwork = null;
            this._updateConnectionState();

            Log.info("Web3 disconnected");
        },

        /**
         * Get current account
         * @private
         * @returns {Promise<string>} Current account address
         */
        _getCurrentAccount() {
            if (!this._web3) {
                return Promise.resolve(null);
            }

            return this._web3.eth.getAccounts()
                .then((accounts) => {
                    return accounts[0] || null;
                });
        },

        /**
         * Get current network
         * @private
         * @returns {Promise<string>} Current network ID
         */
        _getCurrentNetwork() {
            if (!this._web3) {
                return Promise.resolve(null);
            }

            return this._web3.eth.net.getId()
                .then((networkId) => {
                    return networkId.toString();
                });
        },

        /**
         * Get contract instance
         * @public
         * @param {string} contractName Contract name
         * @param {Object} abi Contract ABI
         * @returns {Object} Contract instance
         */
        getContract(contractName, abi) {
            if (!this._web3) {
                throw new Error("Web3 not initialized");
            }

            const sAddress = this._contractAddresses[contractName];
            if (!sAddress) {
                throw new Error(`Contract address not found: ${ contractName}`);
            }

            if (!this._contracts[contractName]) {
                this._contracts[contractName] = new this._web3.eth.Contract(abi, sAddress);
            }

            return this._contracts[contractName];
        },

        /**
         * Send transaction with SAP audit logging
         * @public
         * @param {Object} oTransaction Transaction object
         * @returns {Promise} Transaction promise
         */
        sendTransaction(oTransaction) {
            if (!this._isConnected) {
                return Promise.reject(new Error("Web3 not connected"));
            }

            // Add SAP audit trail
            const sCorrelationId = this._generateCorrelationId();

            // Log transaction attempt
            this._logTransactionAttempt(oTransaction, sCorrelationId);

            return this._web3.eth.sendTransaction({
                from: this._currentAccount,
                ...oTransaction
            }).then((oReceipt) => {
                // Log successful transaction
                this._logTransactionSuccess(oReceipt, sCorrelationId);
                return oReceipt;
            }).catch((oError) => {
                // Log failed transaction
                this._logTransactionFailure(oError, sCorrelationId);
                throw oError;
            });
        },

        /**
         * Generate correlation ID for transaction tracking
         * @private
         * @returns {string} Correlation ID
         */
        _generateCorrelationId() {
            return `tx-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Log transaction attempt
         * @private
         * @param {Object} oTransaction Transaction object
         * @param {string} sCorrelationId Correlation ID
         */
        _logTransactionAttempt(oTransaction, sCorrelationId) {
            Log.info("Transaction attempt", {
                correlationId: sCorrelationId,
                from: this._currentAccount,
                to: oTransaction.to,
                value: oTransaction.value,
                data: oTransaction.data ? `${oTransaction.data.substring(0, 10) }...` : null
            });

            // In SAP environment, send to audit service
            if (this._isSAPEnvironment) {
                this._sendToSAPAuditService("TRANSACTION_ATTEMPT", {
                    correlationId: sCorrelationId,
                    transaction: oTransaction,
                    account: this._currentAccount,
                    network: this._currentNetwork
                });
            }
        },

        /**
         * Log transaction success
         * @private
         * @param {Object} oReceipt Transaction receipt
         * @param {string} sCorrelationId Correlation ID
         */
        _logTransactionSuccess(oReceipt, sCorrelationId) {
            Log.info("Transaction successful", {
                correlationId: sCorrelationId,
                txHash: oReceipt.transactionHash,
                blockNumber: oReceipt.blockNumber,
                gasUsed: oReceipt.gasUsed
            });

            // In SAP environment, send to audit service
            if (this._isSAPEnvironment) {
                this._sendToSAPAuditService("TRANSACTION_SUCCESS", {
                    correlationId: sCorrelationId,
                    receipt: oReceipt
                });
            }
        },

        /**
         * Log transaction failure
         * @private
         * @param {Error} oError Error object
         * @param {string} sCorrelationId Correlation ID
         */
        _logTransactionFailure(oError, sCorrelationId) {
            Log.error("Transaction failed", {
                correlationId: sCorrelationId,
                error: oError.message,
                code: oError.code
            });

            // In SAP environment, send to audit service
            if (this._isSAPEnvironment) {
                this._sendToSAPAuditService("TRANSACTION_FAILURE", {
                    correlationId: sCorrelationId,
                    error: {
                        message: oError.message,
                        code: oError.code
                    }
                });
            }
        },

        /**
         * Send audit data to SAP audit service
         * @private
         * @param {string} sEventType Event type
         * @param {Object} oData Event data
         */
        _sendToSAPAuditService(sEventType, oData) {
            jQuery.ajax({
                url: "/api/v1/audit/blockchain",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    eventType: sEventType,
                    timestamp: new Date().toISOString(),
                    data: oData,
                    user: sap.ushell.Container.getUser().getId(),
                    session: sap.ushell.Container.getUser().getId() // In real SAP, get session ID
                }),
                success() {
                    Log.debug("Audit event sent to SAP service");
                },
                error(oError) {
                    Log.warning("Failed to send audit event to SAP service", oError);
                }
            });
        },

        /* =========================================================== */
        /* Public API                                                 */
        /* =========================================================== */

        /**
         * Check if Web3 is connected
         * @public
         * @returns {boolean} True if connected
         */
        isConnected() {
            return this._isConnected;
        },

        /**
         * Get current account
         * @public
         * @returns {string} Current account address
         */
        getCurrentAccount() {
            return this._currentAccount;
        },

        /**
         * Get current network
         * @public
         * @returns {string} Current network ID
         */
        getCurrentNetwork() {
            return this._currentNetwork;
        },

        /**
         * Get Web3 instance
         * @public
         * @returns {Object} Web3 instance
         */
        getWeb3() {
            return this._web3;
        },

        /**
         * Get contract addresses
         * @public
         * @returns {Object} Contract addresses
         */
        getContractAddresses() {
            return this._contractAddresses;
        },

        /**
         * Check if in SAP environment
         * @public
         * @returns {boolean} True if in SAP environment
         */
        isSAPEnvironment() {
            return this._isSAPEnvironment;
        }
    });
});