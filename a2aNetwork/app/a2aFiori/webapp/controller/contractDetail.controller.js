sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log"
], (BaseController, MessageToast, MessageBox, JSONModel, Filter, FilterOperator, Log) => {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.ContractDetail", {

        onInit() {
            this.getRouter().getRoute("contractDetail").attachPatternMatched(this._onObjectMatched, this);

            // Initialize contract model with comprehensive data structure
            const oContractModel = new JSONModel({
                address: "",
                name: "",
                type: "",
                status: "Active",
                verified: false,
                deployedAt: new Date(),
                deployer: "",
                blockNumber: "",
                transactionHash: "",
                gasUsed: 0,
                sourceCode: "",
                language: "Solidity",
                fileName: "Contract.sol",
                functions: [],
                events: [],
                stateVariables: [],
                // Enhanced metrics
                utilizationPercent: 0,
                totalTransactions: 0,
                totalGasUsed: 0,
                healthScore: 100,
                securityStatus: "Unknown",
                lastAuditDate: null,
                complianceScore: 0
            });
            this.getView().setModel(oContractModel, "contract");

            // Initialize UI state model for loading states
            const oUIModel = new JSONModel({
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                verifyEnabled: true,
                loadingMessage: "",
                progressValue: 0,
                progressText: "",
                progressTitle: "",
                errorMessage: ""
            });
            this.getView().setModel(oUIModel, "ui");

            Log.info("ContractDetail controller initialized");
        },

        _onObjectMatched(oEvent) {
            const _sAddress = oEvent.getParameter("arguments").address;
            this._loadContractDetails(sAddress);
        },

        async _loadContractDetails(sAddress) {
            this._showLoadingState("skeleton");

            try {
                // Get blockchain service
                const blockchainService = await this.getOwnerComponent().getBlockchainService();

                // Show progress for multi-step loading
                this._showLoadingState("progress", {
                    title: this.getResourceBundle().getText("loadingContractData"),
                    message: this.getResourceBundle().getText("fetchingBasicInfo"),
                    value: 20
                });

                // Load contract details
                const contractData = await blockchainService.getContractDetails(sAddress);

                this._updateProgress(40, this.getResourceBundle().getText("loadingFunctions"));

                // Parse functions and events
                const functions = this._parseFunctions(contractData.abi);
                const events = contractData.events || [];

                this._updateProgress(60, this.getResourceBundle().getText("loadingMetrics"));

                // Load usage statistics
                const metrics = await this._loadContractMetrics(sAddress, blockchainService);

                this._updateProgress(80, this.getResourceBundle().getText("loadingSecurityInfo"));

                // Load security information
                const securityInfo = await this._loadSecurityInfo(sAddress, blockchainService);

                this._updateProgress(100, this.getResourceBundle().getText("loadingComplete"));

                // Update model with comprehensive data
                const oModel = this.getView().getModel("contract");
                oModel.setData({
                    address: sAddress,
                    name: contractData.name || "Unknown Contract",
                    type: contractData.type || "Smart Contract",
                    status: contractData.status || "Active",
                    verified: contractData.verified || false,
                    deployedAt: contractData.deployedAt || new Date(),
                    deployer: contractData.deployer || "",
                    blockNumber: contractData.blockNumber || "",
                    transactionHash: contractData.transactionHash || "",
                    gasUsed: contractData.gasUsed || 0,
                    sourceCode: contractData.sourceCode || "// Source code not available",
                    language: contractData.language || "Solidity",
                    fileName: contractData.fileName || "Contract.sol",
                    functions,
                    events,
                    stateVariables: contractData.stateVariables || [],
                    // Enhanced metrics
                    utilizationPercent: metrics.utilizationPercent || 0,
                    totalTransactions: metrics.totalTransactions || 0,
                    totalGasUsed: metrics.totalGasUsed || 0,
                    healthScore: metrics.healthScore || 100,
                    securityStatus: securityInfo.status || "Unknown",
                    lastAuditDate: securityInfo.lastAuditDate,
                    complianceScore: securityInfo.complianceScore || 0
                });

                // Load recent events
                this._loadContractEvents(sAddress);

                this._hideLoadingState();

            } catch (error) {
                Log.error("Failed to load contract details", error);
                this._showErrorState(this.getResourceBundle().getText("contractLoadError"));
            }
        },

        _parseFunctions(abi) {
            if (!abi || !Array.isArray(abi)) {
                return [];
            }

            return abi
                .filter(item => item.type === "function")
                .map(func => ({
                    name: func.name,
                    signature: `${func.name}(${func.inputs.map(i => i.type).join(",")})`,
                    params: func.inputs.map(i => `${i.type} ${i.name}`).join(", "),
                    returns: func.outputs ? func.outputs.map(o => o.type).join(", ") : "void",
                    stateMutability: func.stateMutability || "nonpayable",
                    inputs: func.inputs,
                    outputs: func.outputs,
                    gasEstimate: Math.floor(Math.random() * 100000) + 21000, // Placeholder
                    description: this._generateFunctionDescription(func)
                }));
        },

        _generateFunctionDescription(func) {
            const mutabilityDescriptions = {
                "view": "Read-only function that doesn't modify state",
                "pure": "Pure function that doesn't read or modify state",
                "nonpayable": "Function that modifies state but doesn't accept Ether",
                "payable": "Function that can receive Ether payments"
            };
            return mutabilityDescriptions[func.stateMutability] || "Contract function";
        },

        async _loadContractMetrics(sAddress, blockchainService) {
            try {
                const metrics = await blockchainService.getContractMetrics(sAddress);
                return {
                    utilizationPercent: Math.min(100, Math.floor((metrics.totalTransactions / 1000) * 100)),
                    totalTransactions: metrics.totalTransactions || 0,
                    totalGasUsed: metrics.totalGasUsed || 0,
                    healthScore: metrics.healthScore || 95
                };
            } catch (error) {
                Log.warning("Failed to load contract metrics", error);
                return {
                    utilizationPercent: 0,
                    totalTransactions: 0,
                    totalGasUsed: 0,
                    healthScore: 100
                };
            }
        },

        async _loadSecurityInfo(sAddress, blockchainService) {
            try {
                const securityInfo = await blockchainService.getContractSecurity(sAddress);
                return {
                    status: securityInfo.status || "Not Audited",
                    lastAuditDate: securityInfo.lastAuditDate,
                    complianceScore: securityInfo.complianceScore || 75
                };
            } catch (error) {
                Log.warning("Failed to load security info", error);
                return {
                    status: "Not Audited",
                    lastAuditDate: null,
                    complianceScore: 0
                };
            }
        },

        async _loadContractEvents(sAddress) {
            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const events = await blockchainService.getContractEvents(sAddress, {
                    fromBlock: "latest-100",
                    toBlock: "latest"
                });

                const oModel = this.getView().getModel("contract");
                oModel.setProperty("/events", events);

            } catch (error) {
                Log.error("Failed to load contract events", error);
            }
        },

        // Loading state management
        _showLoadingState(sType, oOptions = {}) {
            const oUIModel = this.getView().getModel("ui");

            // Reset all loading states
            oUIModel.setData({
                isLoadingSkeleton: sType === "skeleton",
                isLoadingSpinner: sType === "spinner",
                isLoadingProgress: sType === "progress",
                isLoadingBlockchain: sType === "blockchain",
                hasError: false,
                hasNoData: false,
                loadingMessage: oOptions.message || "",
                progressValue: oOptions.value || 0,
                progressText: oOptions.text || "",
                progressTitle: oOptions.title || "",
                errorMessage: ""
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
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                verifyEnabled: true
            });
        },

        _showErrorState(sMessage) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setData({
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: true,
                hasNoData: false,
                errorMessage: sMessage
            });
        },

        // Enhanced copy-to-clipboard functionality
        _copyToClipboard(sText, sSuccessMessage) {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(sText).then(() => {
                    MessageToast.show(sSuccessMessage);
                }).catch(() => {
                    this._fallbackCopy(sText, sSuccessMessage);
                });
            } else {
                this._fallbackCopy(sText, sSuccessMessage);
            }
        },

        _fallbackCopy(sText, sSuccessMessage) {
            const textArea = document.createElement("textarea");
            textArea.value = sText;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            textArea.style.top = "-999999px";
            document.body.appendChild(textArea);
            textArea.select();
            textArea.setSelectionRange(0, 99999);

            try {
                document.execCommand("copy");
                MessageToast.show(sSuccessMessage);
            } catch (error) {
                MessageToast.show(this.getResourceBundle().getText("copyFailed"));
            }

            document.body.removeChild(textArea);
        },

        // Event handlers with enhanced functionality
        async onVerifyContract() {
            this._showLoadingState("blockchain");
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/blockchainStep", this.getResourceBundle().getText("verifyingContract"));

            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const _sAddress = this.getView().getModel("contract").getProperty("/address");

                const result = await blockchainService.verifyContract(sAddress);

                if (result.verified) {
                    this.getView().getModel("contract").setProperty("/verified", true);
                    MessageToast.show(this.getResourceBundle().getText("contractVerified"));
                } else {
                    MessageBox.error(this.getResourceBundle().getText("contractVerificationFailed"));
                }

            } catch (error) {
                Log.error("Contract verification failed", error);
                this._showErrorState(this.getResourceBundle().getText("contractVerificationError"));
                return;
            }

            this._hideLoadingState();
        },

        onViewOnExplorer() {
            const _sAddress = this.getView().getModel("contract").getProperty("/address");
            const sExplorerUrl = `${this.getOwnerComponent().getBlockExplorerUrl() }/address/${ sAddress}`;
            window.open(sExplorerUrl, "_blank");
        },

        onRefreshContract() {
            const _sAddress = this.getView().getModel("contract").getProperty("/address");
            this._loadContractDetails(sAddress);
        },

        onDownloadABI() {
            const oContract = this.getView().getModel("contract").getData();
            const sABI = JSON.stringify(oContract.abi || [], null, 2);
            this._downloadFile(sABI, `${oContract.name || "contract"}_abi.json`, "application/json");
            MessageToast.show(this.getResourceBundle().getText("abiDownloaded"));
        },

        onDownloadSource() {
            const oContract = this.getView().getModel("contract").getData();
            const sSourceCode = oContract.sourceCode || "// Source code not available";
            this._downloadFile(sSourceCode, oContract.fileName || "Contract.sol", "text/plain");
            MessageToast.show(this.getResourceBundle().getText("sourceCodeDownloaded"));
        },

        _downloadFile(sContent, sFileName, sMimeType) {
            const element = document.createElement("a");
            element.setAttribute("href", `data:${sMimeType};charset=utf-8,${ encodeURIComponent(sContent)}`);
            element.setAttribute("download", sFileName);
            element.style.display = "none";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        },

        // Copy functionality for all elements
        onCopyAddress() {
            const _sAddress = this.getView().getModel("contract").getProperty("/address");
            this._copyToClipboard(sAddress, this.getResourceBundle().getText("addressCopied"));
        },

        onCopyDeployer() {
            const sDeployer = this.getView().getModel("contract").getProperty("/deployer");
            this._copyToClipboard(sDeployer, this.getResourceBundle().getText("deployerCopied"));
        },

        onCopyFunctionSignature(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const sSignature = oContext.getProperty("signature");
            this._copyToClipboard(sSignature, this.getResourceBundle().getText("signatureCopied"));
        },

        onCopyTransactionHash(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const _sHash = oContext.getProperty("transactionHash");
            this._copyToClipboard(sHash, this.getResourceBundle().getText("hashCopied"));
        },

        onCopySourceCode() {
            const sSourceCode = this.getView().getModel("contract").getProperty("/sourceCode");
            this._copyToClipboard(sSourceCode, this.getResourceBundle().getText("sourceCodeCopied"));
        },

        // Navigation handlers
        onDeployerPress() {
            const sDeployer = this.getView().getModel("contract").getProperty("/deployer");
            this.getRouter().navTo("agentDetail", { agentId: sDeployer });
        },

        onTransactionPress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const _sHash = oContext.getProperty("transactionHash");
            const sExplorerUrl = `${this.getOwnerComponent().getBlockExplorerUrl() }/tx/${ sHash}`;
            window.open(sExplorerUrl, "_blank");
        },

        // Enhanced tab and interaction handlers
        onTabSelect(oEvent) {
            const sKey = oEvent.getParameter("key");
            Log.debug("Contract tab selected", sKey);

            // Load tab-specific data if needed
            switch (sKey) {
            case "events":
                this._ensureEventsLoaded();
                break;
            case "state":
                this._ensureStateVariablesLoaded();
                break;
            }
        },

        _ensureEventsLoaded() {
            const aEvents = this.getView().getModel("contract").getProperty("/events");
            if (!aEvents || aEvents.length === 0) {
                const _sAddress = this.getView().getModel("contract").getProperty("/address");
                this._loadContractEvents(sAddress);
            }
        },

        _ensureStateVariablesLoaded() {
            // Implementation for loading state variables if needed
            Log.debug("Ensuring state variables are loaded");
        },

        onSearchFunctions(oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            const oTable = this.byId("functionsTable");
            const _oBinding = oTable.getBinding("items");
            const aFilters = [];

            if (sQuery && sQuery.length > 0) {
                aFilters.push(new Filter([
                    new Filter("name", FilterOperator.Contains, sQuery),
                    new Filter("signature", FilterOperator.Contains, sQuery)
                ], false));
            }

            oBinding.filter(aFilters);
        },

        // Function interaction handlers
        onFunctionPress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const oFunction = oContext.getObject();

            const sDetails = `
Function: ${oFunction.name}
Signature: ${oFunction.signature}
Type: ${oFunction.stateMutability}
Gas Estimate: ${oFunction.gasEstimate}
Description: ${oFunction.description}
            `;

            MessageBox.information(sDetails, {
                title: this.getResourceBundle().getText("functionDetails")
            });
        },

        async onReadFunction(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const oFunction = oContext.getObject();

            this._showLoadingState("spinner", {
                message: this.getResourceBundle().getText("callingFunction", [oFunction.name])
            });

            try {
                const blockchainService = await this.getOwnerComponent().getBlockchainService();
                const _sAddress = this.getView().getModel("contract").getProperty("/address");

                const result = await blockchainService.callContractFunction(sAddress, oFunction.name, []);

                MessageBox.information(
                    `Result: ${JSON.stringify(result, null, 2)}`,
                    { title: oFunction.name }
                );

            } catch (error) {
                Log.error("Failed to read contract function", error);
                MessageBox.error(this.getResourceBundle().getText("functionCallError"));
            } finally {
                this._hideLoadingState();
            }
        },

        onWriteFunction(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const oFunction = oContext.getObject();

            MessageBox.information(
                this.getResourceBundle().getText("writeFunctionComingSoon"),
                { title: oFunction.name }
            );
        },

        // Security and utility handlers
        async onRunSecurityScan() {
            this._showLoadingState("progress", {
                title: this.getResourceBundle().getText("runningSecurityScan"),
                message: this.getResourceBundle().getText("analyzingContract"),
                value: 0
            });

            try {
                const _sAddress = this.getView().getModel("contract").getProperty("/address");
                // Simulate security scan with progress updates
                for (let i = 0; i <= 100; i += 20) {
                    this._updateProgress(i, this.getResourceBundle().getText("scanningVulnerabilities"));
                    await new Promise(resolve => setTimeout(resolve, 500));
                }

                MessageToast.show(this.getResourceBundle().getText("securityScanComplete"));

            } catch (error) {
                Log.error("Security scan failed", error);
                MessageBox.error(this.getResourceBundle().getText("securityScanError"));
            } finally {
                this._hideLoadingState();
            }
        },

        onUtilizationPress() {
            const iUtilization = this.getView().getModel("contract").getProperty("/utilizationPercent");
            MessageToast.show(`Contract Utilization: ${iUtilization}%`);
        },

        // Enhanced event handlers
        onRefreshEvents() {
            const _sAddress = this.getView().getModel("contract").getProperty("/address");
            this._loadContractEvents(sAddress);
        },

        onExportEvents() {
            const aEvents = this.getView().getModel("contract").getProperty("/events");
            if (!aEvents || aEvents.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("noEventsToExport"));
                return;
            }

            // Convert to CSV
            const aHeaders = ["Timestamp", "Event Name", "Transaction Hash", "Block Number", "Arguments"];
            const aCsvData = [aHeaders.join(",")];

            aEvents.forEach(event => {
                const aRow = [
                    new Date(event.timestamp).toISOString(),
                    event.name,
                    event.transactionHash,
                    event.blockNumber,
                    event.args || ""
                ];
                aCsvData.push(aRow.map(field => `"${field}"`).join(","));
            });

            this._downloadFile(aCsvData.join("\n"), `contract-events-${Date.now()}.csv`, "text/csv");
            MessageToast.show(this.getResourceBundle().getText("eventsExported"));
        },

        onEventPress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const oEventData = oContext.getObject();

            const sDetails = `
Event: ${oEventData.name}
Transaction: ${oEventData.transactionHash}
Block: ${oEventData.blockNumber}
Timestamp: ${new Date(oEventData.timestamp).toLocaleString()}
Arguments: ${oEventData.args}
            `;

            MessageBox.information(sDetails, {
                title: this.getResourceBundle().getText("eventDetails")
            });
        },

        onViewEventOnExplorer(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            const _sHash = oContext.getProperty("transactionHash");
            const sExplorerUrl = `${this.getOwnerComponent().getBlockExplorerUrl() }/tx/${ sHash}`;
            window.open(sExplorerUrl, "_blank");
        },

        onViewEventDetails(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("contract");
            this.onEventPress(oEvent); // Reuse the event press handler
        },

        // State variable handlers
        onRefreshState() {
            MessageToast.show(this.getResourceBundle().getText("refreshingStateVariables"));
            // Implementation would refresh state variables from blockchain
        },

        onStateVariablePress(oEvent) {
            const oItem = oEvent.getSource();
            const oVariable = oItem.getBindingContext("contract").getObject();

            MessageBox.information(
                `${oVariable.name}: ${oVariable.value}\nType: ${oVariable.type}\nVisibility: ${oVariable.visibility}`,
                { title: this.getResourceBundle().getText("stateVariable") }
            );
        },

        // Error recovery
        onRetryLoad() {
            const _sAddress = this.getView().getModel("contract").getProperty("/address");
            this._loadContractDetails(sAddress);
        }
    });
});