sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "a2a/network/fiori/utils/Web3Manager",
    "a2a/network/fiori/utils/GovernanceService",
    "sap/base/Log",
    "sap/ui/fl/variants/VariantManagement",
    "sap/m/p13n/Engine",
    "sap/m/p13n/SelectionController",
    "sap/m/p13n/SortController",
    "sap/m/p13n/FilterController",
    "sap/m/p13n/GroupController"
], function(Controller, JSONModel, MessageBox, MessageToast, Fragment, Filter, FilterOperator,
    Web3Manager, GovernanceService, Log, VariantManagement, Engine, SelectionController,
    SortController, FilterController, GroupController) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.governance", {

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        onInit() {
            this._oRouter = this.getOwnerComponent().getRouter();
            this._oRouter.getRoute("governance").attachPatternMatched(this._onRouteMatched, this);

            // Initialize models
            this._initializeModels();

            // Initialize Web3 and governance services
            this._initializeServices();

            // Initialize P13n and Variant Management
            this._initializeP13n();
            this._initializeVariantManagement();

            Log.info("Governance controller initialized");
        },

        onBeforeRendering() {
            this._loadGovernanceData();
        },

        /* =========================================================== */
        /* event handlers                                             */
        /* =========================================================== */

        onNavBack() {
            this._oRouter.navTo("home");
        },

        onCreateProposal() {
            this._openCreateProposalDialog();
        },

        onProposalSelect(oEvent) {
            const oSelectedItem = oEvent.getParameter("listItem");
            const oContext = oSelectedItem.getBindingContext("governance");
            const sProposalId = oContext.getProperty("id");

            this._oRouter.navTo("proposalDetail", {
                proposalId: sProposalId
            });
        },

        onProposalPress(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const sProposalId = oContext.getProperty("id");

            this._oRouter.navTo("proposalDetail", {
                proposalId: sProposalId
            });
        },

        onVoteFor(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const oProposal = oContext.getObject();

            this._castVote(oProposal.id, 1, "Supporting this proposal");
        },

        onVoteAgainst(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const oProposal = oContext.getObject();

            this._castVote(oProposal.id, 0, "Opposing this proposal");
        },

        onSearchProposals(oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            const aFilters = [];

            if (sQuery && sQuery.length > 0) {
                aFilters.push(new Filter("title", FilterOperator.Contains, sQuery));
            }

            const oTable = this.byId("proposalsTable");
            const oBinding = oTable.getBinding("items");
            oBinding.filter(aFilters);
        },

        onCategoryFilter(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const aFilters = [];

            if (sSelectedKey && sSelectedKey !== "all") {
                aFilters.push(new Filter("category", FilterOperator.EQ, sSelectedKey));
            }

            const oTable = this.byId("proposalsTable");
            const oBinding = oTable.getBinding("items");
            oBinding.filter(aFilters);
        },

        onManageStaking() {
            const oTabBar = this.byId("governanceTabBar");
            oTabBar.setSelectedKey("staking");
        },

        onStakeTokens() {
            const oModel = this.getView().getModel("governance");
            const oStakeForm = oModel.getProperty("/stakeForm");

            if (!oStakeForm.valid) {
                MessageToast.show(this._getResourceBundle().getText("invalidStakeForm"));
                return;
            }

            this._stakeTokens(oStakeForm.amount, oStakeForm.period);
        },

        onUnstake(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const oStake = oContext.getObject();

            this._unstakeTokens(oStake.id);
        },

        onStakeAmountChange() {
            this._validateStakeForm();
            this._calculateStakingRewards();
        },

        onDelegateVotes() {
            const oModel = this.getView().getModel("governance");
            const sDelegateAddress = oModel.getProperty("/delegateForm/address");

            if (!this._isValidAddress(sDelegateAddress)) {
                MessageToast.show(this._getResourceBundle().getText("invalidAddress"));
                return;
            }

            this._delegateVotes(sDelegateAddress);
        },

        onRemoveDelegation() {
            this._delegateVotes(null); // Self-delegate
        },

        onDelegateToUser(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const sAddress = oContext.getProperty("address");

            const oModel = this.getView().getModel("governance");
            oModel.setProperty("/delegateForm/address", sAddress);

            this.onDelegateVotes();
        },

        onDelegateAddressChange() {
            this._validateDelegateForm();
        },

        onCreateProposalConfirm() {
            this._createProposal();
        },

        onCreateProposalCancel() {
            this._closeCreateProposalDialog();
        },

        onFilterHistory() {
            const oDateRange = this.byId("historyDateRange");
            const oFrom = oDateRange.getFrom();
            const oTo = oDateRange.getTo();

            this._filterHistory(oFrom, oTo);
        },

        onHistoryProposalPress(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("governance");
            const sProposalId = oContext.getProperty("proposalId");

            this._oRouter.navTo("proposalDetail", {
                proposalId: sProposalId
            });
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        _onRouteMatched() {
            this._loadGovernanceData();
        },

        _initializeModels() {
            const oGovernanceModel = new JSONModel({
                tokenStats: {
                    totalSupply: "1,000,000,000",
                    circulatingSupply: "750,000,000",
                    stakedTokens: "250,000,000",
                    stakingAPR: "12.5"
                },
                userStats: {
                    tokens: "10,000",
                    votingPower: "15,000",
                    delegatedTo: "Self",
                    availableBalance: "5,000",
                    currentDelegate: "Self",
                    hasDelegated: false
                },
                stats: {
                    activeProposals: 3,
                    totalProposals: 47,
                    participationRate: 68.5,
                    averageQuorum: 15.2
                },
                proposals: [],
                categories: [
                    { key: "all", text: "All Categories" },
                    { key: "PROTOCOL_UPGRADE", text: "Protocol Upgrade" },
                    { key: "PARAMETER_CHANGE", text: "Parameter Change" },
                    { key: "TREASURY_ALLOCATION", text: "Treasury Allocation" },
                    { key: "EMERGENCY_ACTION", text: "Emergency Action" },
                    { key: "AGENT_MANAGEMENT", text: "Agent Management" },
                    { key: "REPUTATION_SYSTEM", text: "Reputation System" }
                ],
                stakingPeriods: [
                    { key: "30", text: "30 Days (5% APR)" },
                    { key: "90", text: "90 Days (8% APR)" },
                    { key: "180", text: "180 Days (12% APR)" },
                    { key: "365", text: "365 Days (15% APR)" }
                ],
                executionDelays: [
                    { key: "1", text: "1 Day" },
                    { key: "3", text: "3 Days" },
                    { key: "7", text: "7 Days" },
                    { key: "14", text: "14 Days" }
                ],
                stakeForm: {
                    amount: "",
                    period: "30",
                    estimatedRewards: "0",
                    valid: false
                },
                delegateForm: {
                    address: "",
                    valid: false
                },
                newProposal: {
                    title: "",
                    category: "",
                    description: "",
                    delay: "3",
                    ipfsHash: "",
                    valid: false
                },
                userStakes: [],
                topDelegates: [],
                userHistory: [],
                selectedCategory: "all"
            });

            this.getView().setModel(oGovernanceModel, "governance");
        },

        _initializeServices() {
            this._oWeb3Manager = new Web3Manager();
            this._oGovernanceService = new GovernanceService(this._oWeb3Manager);

            // Listen for Web3 connection changes
            this._oWeb3Manager.attachConnectionChanged(this._onWeb3ConnectionChanged, this);
        },

        _loadGovernanceData() {
            this.getView().getModel("governance").setProperty("/busy", true);

            Promise.all([
                this._loadProposals(),
                this._loadUserStats(),
                this._loadTokenStats(),
                this._loadUserStakes(),
                this._loadTopDelegates(),
                this._loadUserHistory()
            ]).then(function() {
                this.getView().getModel("governance").setProperty("/busy", false);
                Log.info("Governance data loaded successfully");
            }.bind(this)).catch(function(oError) {
                Log.error("Failed to load governance data", oError);
                this.getView().getModel("governance").setProperty("/busy", false);
                MessageBox.error(this._getResourceBundle().getText("loadDataError"));
            }.bind(this));
        },

        _loadProposals() {
            return this._oGovernanceService.getActiveProposals().then(function(aProposals) {
                this.getView().getModel("governance").setProperty("/proposals", aProposals);
            }.bind(this));
        },

        _loadUserStats() {
            if (!this._oWeb3Manager.isConnected()) {
                return Promise.resolve();
            }

            return this._oGovernanceService.getUserStats().then(function(oStats) {
                this.getView().getModel("governance").setProperty("/userStats", oStats);
            }.bind(this));
        },

        _loadTokenStats() {
            return this._oGovernanceService.getTokenStats().then(function(oStats) {
                this.getView().getModel("governance").setProperty("/tokenStats", oStats);
            }.bind(this));
        },

        _loadUserStakes() {
            if (!this._oWeb3Manager.isConnected()) {
                return Promise.resolve();
            }

            return this._oGovernanceService.getUserStakes().then(function(aStakes) {
                this.getView().getModel("governance").setProperty("/userStakes", aStakes);
            }.bind(this));
        },

        _loadTopDelegates() {
            return this._oGovernanceService.getTopDelegates().then(function(aDelegates) {
                this.getView().getModel("governance").setProperty("/topDelegates", aDelegates);
            }.bind(this));
        },

        _loadUserHistory() {
            if (!this._oWeb3Manager.isConnected()) {
                return Promise.resolve();
            }

            return this._oGovernanceService.getUserHistory().then(function(aHistory) {
                this.getView().getModel("governance").setProperty("/userHistory", aHistory);
            }.bind(this));
        },

        _castVote(sProposalId, iSupport, sReason) {
            if (!this._oWeb3Manager.isConnected()) {
                MessageBox.error(this._getResourceBundle().getText("walletNotConnected"));
                return;
            }

            MessageBox.confirm(
                this._getResourceBundle().getText("confirmVote", [iSupport ? "FOR" : "AGAINST"]),
                {
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._oGovernanceService.castVote(sProposalId, iSupport, sReason)
                                .then(function(sTxHash) {
                                    MessageToast.show(this._getResourceBundle().getText("voteSubmitted"));
                                    this._loadProposals();
                                }.bind(this))
                                .catch(function(oError) {
                                    Log.error("Vote failed", oError);
                                    MessageBox.error(this._getResourceBundle().getText("voteFailed"));
                                }.bind(this));
                        }
                    }.bind(this)
                }
            );
        },

        _stakeTokens(sAmount, sPeriod) {
            if (!this._oWeb3Manager.isConnected()) {
                MessageBox.error(this._getResourceBundle().getText("walletNotConnected"));
                return;
            }

            this._oGovernanceService.stakeTokens(sAmount, sPeriod)
                .then(function(sTxHash) {
                    MessageToast.show(this._getResourceBundle().getText("tokensStaked"));
                    this._loadUserStats();
                    this._loadUserStakes();
                }.bind(this))
                .catch(function(oError) {
                    Log.error("Staking failed", oError);
                    MessageBox.error(this._getResourceBundle().getText("stakingFailed"));
                }.bind(this));
        },

        _unstakeTokens(sStakeId) {
            if (!this._oWeb3Manager.isConnected()) {
                MessageBox.error(this._getResourceBundle().getText("walletNotConnected"));
                return;
            }

            MessageBox.confirm(
                this._getResourceBundle().getText("confirmUnstake"),
                {
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._oGovernanceService.unstakeTokens(sStakeId)
                                .then(function(sTxHash) {
                                    MessageToast.show(this._getResourceBundle().getText("tokensUnstaked"));
                                    this._loadUserStats();
                                    this._loadUserStakes();
                                }.bind(this))
                                .catch(function(oError) {
                                    Log.error("Unstaking failed", oError);
                                    MessageBox.error(this._getResourceBundle().getText("unstakingFailed"));
                                }.bind(this));
                        }
                    }.bind(this)
                }
            );
        },

        _delegateVotes(sDelegateAddress) {
            if (!this._oWeb3Manager.isConnected()) {
                MessageBox.error(this._getResourceBundle().getText("walletNotConnected"));
                return;
            }

            this._oGovernanceService.delegateVotes(sDelegateAddress)
                .then(function(sTxHash) {
                    MessageToast.show(this._getResourceBundle().getText("votesDelegated"));
                    this._loadUserStats();
                }.bind(this))
                .catch(function(oError) {
                    Log.error("Delegation failed", oError);
                    MessageBox.error(this._getResourceBundle().getText("delegationFailed"));
                }.bind(this));
        },

        _createProposal() {
            const oModel = this.getView().getModel("governance");
            const oProposal = oModel.getProperty("/newProposal");

            if (!oProposal.valid) {
                MessageToast.show(this._getResourceBundle().getText("invalidProposalForm"));
                return;
            }

            if (!this._oWeb3Manager.isConnected()) {
                MessageBox.error(this._getResourceBundle().getText("walletNotConnected"));
                return;
            }

            this._oGovernanceService.createProposal(oProposal)
                .then(function(sTxHash) {
                    MessageToast.show(this._getResourceBundle().getText("proposalCreated"));
                    this._closeCreateProposalDialog();
                    this._loadProposals();
                }.bind(this))
                .catch(function(oError) {
                    Log.error("Proposal creation failed", oError);
                    MessageBox.error(this._getResourceBundle().getText("proposalCreationFailed"));
                }.bind(this));
        },

        _openCreateProposalDialog() {
            if (!this._oCreateProposalDialog) {
                this._oCreateProposalDialog = Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.fiori.view.fragments.CreateProposalDialog",
                    controller: this
                }).then(function(oDialog) {
                    this.getView().addDependent(oDialog);
                    return oDialog;
                }.bind(this));
            }

            this._oCreateProposalDialog.then(function(oDialog) {
                this._resetProposalForm();
                oDialog.open();
            }.bind(this));
        },

        _closeCreateProposalDialog() {
            this.byId("createProposalDialog").close();
        },

        _resetProposalForm() {
            const oModel = this.getView().getModel("governance");
            oModel.setProperty("/newProposal", {
                title: "",
                category: "",
                description: "",
                delay: "3",
                ipfsHash: "",
                valid: false
            });
        },

        _validateStakeForm() {
            const oModel = this.getView().getModel("governance");
            const oStakeForm = oModel.getProperty("/stakeForm");
            const bValid = oStakeForm.amount && parseFloat(oStakeForm.amount) > 0;

            oModel.setProperty("/stakeForm/valid", bValid);
        },

        _validateDelegateForm() {
            const oModel = this.getView().getModel("governance");
            const sDelegateAddress = oModel.getProperty("/delegateForm/address");
            const bValid = this._isValidAddress(sDelegateAddress);

            oModel.setProperty("/delegateForm/valid", bValid);
        },

        _calculateStakingRewards() {
            const oModel = this.getView().getModel("governance");
            const oStakeForm = oModel.getProperty("/stakeForm");

            if (oStakeForm.amount && oStakeForm.period) {
                const fAmount = parseFloat(oStakeForm.amount);
                const iPeriod = parseInt(oStakeForm.period, 10);
                const fAPR = this._getAPRForPeriod(iPeriod);
                const fRewards = (fAmount * fAPR * iPeriod) / (365 * 100);

                oModel.setProperty("/stakeForm/estimatedRewards", fRewards.toFixed(2));
            }
        },

        _getAPRForPeriod(iPeriod) {
            const mAPRs = {
                30: 5,
                90: 8,
                180: 12,
                365: 15
            };
            return mAPRs[iPeriod] || 5;
        },

        _isValidAddress(sAddress) {
            return sAddress && /^0x[a-fA-F0-9]{40}$/.test(sAddress);
        },

        _filterHistory(oFrom, oTo) {
            const aFilters = [];

            if (oFrom) {
                aFilters.push(new Filter("voteDate", FilterOperator.GE, oFrom));
            }

            if (oTo) {
                aFilters.push(new Filter("voteDate", FilterOperator.LE, oTo));
            }

            const oTable = this.byId("historyTable");
            const oBinding = oTable.getBinding("items");
            oBinding.filter(aFilters);
        },

        _onWeb3ConnectionChanged(oEvent) {
            const bConnected = oEvent.getParameter("connected");
            if (bConnected) {
                this._loadUserStats();
                this._loadUserStakes();
                this._loadUserHistory();
            }
        },

        _getResourceBundle() {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },

        /* =========================================================== */
        /* formatters                                                  */
        /* =========================================================== */

        formatCategoryState(sCategory) {
            const mStates = {
                "PROTOCOL_UPGRADE": "Error",
                "PARAMETER_CHANGE": "Warning",
                "TREASURY_ALLOCATION": "Success",
                "EMERGENCY_ACTION": "Error",
                "AGENT_MANAGEMENT": "Information",
                "REPUTATION_SYSTEM": "Warning"
            };
            return mStates[sCategory] || "None";
        },

        formatStatusState(sStatus) {
            const mStates = {
                "Active": "Success",
                "Pending": "Warning",
                "Succeeded": "Success",
                "Defeated": "Error",
                "Cancelled": "None",
                "Executed": "Information"
            };
            return mStates[sStatus] || "None";
        },

        formatVoteState(sVote) {
            const mStates = {
                "For": "Success",
                "Against": "Error",
                "Abstain": "Warning"
            };
            return mStates[sVote] || "None";
        },

        formatResultState(sResult) {
            const mStates = {
                "Passed": "Success",
                "Failed": "Error",
                "Pending": "Warning"
            };
            return mStates[sResult] || "None";
        },

        isVotingActive(sStatus) {
            return sStatus === "Active";
        },

        /* =========================================================== */
        /* Variant Management and P13n                                */
        /* =========================================================== */

        onVariantSelect(oEvent) {
            const sVariantKey = oEvent.getParameter("key");
            this._applyVariant(sVariantKey);
        },

        onVariantSave(oEvent) {
            const sVariantName = oEvent.getParameter("name");
            const _bDefault = oEvent.getParameter("def");
            const _bPublic = oEvent.getParameter("public");
            const bOverwrite = oEvent.getParameter("overwrite");

            this._saveVariant(sVariantName, bDefault, bPublic, bOverwrite);
        },

        onVariantManage(oEvent) {
            // Handle variant management
        },

        onTableSettings(oEvent) {
            const oTable = this.byId("proposalsTable");
            this._openP13nDialog(oTable);
        },

        onStakeTableSettings(oEvent) {
            const oTable = this.byId("stakesTable");
            this._openP13nDialog(oTable);
        },

        onHistoryTableSettings(oEvent) {
            const oTable = this.byId("historyTable");
            this._openP13nDialog(oTable);
        },

        _initializeP13n() {
            // Initialize P13n Engine for governance tables
            const oProposalsTable = this.byId("proposalsTable");
            if (oProposalsTable) {
                Engine.getInstance().register(oProposalsTable, {
                    helper: {
                        name: "sap.m.p13n.TableHelper",
                        payload: {
                            column: this._getProposalsColumnConfiguration()
                        }
                    },
                    controller: {
                        Columns: new SelectionController({
                            targetAggregation: "columns"
                        }),
                        Sorter: new SortController({
                            targetAggregation: "sorter"
                        }),
                        Filter: new FilterController({
                            targetAggregation: "filter"
                        })
                    }
                });
            }
        },

        _initializeVariantManagement() {
            // Initialize variant management for governance tables
            const oProposalsVariant = this.byId("proposalsVariant");
            if (oProposalsVariant) {
                this._setupVariantManagement(oProposalsVariant, "proposalsTable");
            }
        },

        _setupVariantManagement(oVariantManagement, sTableId) {
            if (!oVariantManagement) {
                return;
            }

            // Load saved variants from backend or local storage
            const aVariants = this._loadVariants(sTableId);
            oVariantManagement.setModel(new JSONModel({
                variants: aVariants
            }), "variants");
        },

        _getProposalsColumnConfiguration() {
            return [
                { key: "id", label: "Proposal ID", dataType: "sap.ui.model.type.Integer" },
                { key: "title", label: "Title", dataType: "sap.ui.model.type.String" },
                { key: "category", label: "Category", dataType: "sap.ui.model.type.String" },
                { key: "status", label: "Status", dataType: "sap.ui.model.type.String" },
                { key: "forVotes", label: "For Votes", dataType: "sap.ui.model.type.String" },
                { key: "againstVotes", label: "Against Votes", dataType: "sap.ui.model.type.String" },
                { key: "endDate", label: "End Date", dataType: "sap.ui.model.odata.type.DateTimeOffset" }
            ];
        },

        _openP13nDialog(oTable) {
            // Would open P13n dialog for table personalization
            MessageToast.show("Table personalization dialog would open");
        },

        _loadVariants(sTableId) {
            // Load saved variants from backend or local storage
            return [
                {
                    key: "default",
                    text: "Default View",
                    isDefault: true
                },
                {
                    key: "activeProposals",
                    text: "Active Proposals Only",
                    isDefault: false
                }
            ];
        },

        _applyVariant(sVariantKey) {
            // Apply selected variant
            MessageToast.show(`Variant applied: ${ sVariantKey}`);
        },

        _saveVariant(sName, bDefault, bPublic, bOverwrite) {
            // Save current state as variant
            MessageToast.show(`Variant saved: ${ sName}`);
        }
    });
});