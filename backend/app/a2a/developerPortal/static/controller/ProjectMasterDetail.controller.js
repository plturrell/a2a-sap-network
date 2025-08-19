sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/f/library",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, Filter, FilterOperator, fioriLibrary, MessageToast, MessageBox) {
    "use strict";

    var LayoutType = fioriLibrary.LayoutType;

    return Controller.extend("com.sap.a2a.developerportal.controller.ProjectMasterDetail", {

        onInit: function () {
            this.oRouter = this.getOwnerComponent().getRouter();
            this.oModel = this.getOwnerComponent().getModel();
            
            // Initialize layout model
            this.oLayoutModel = new JSONModel({
                layout: LayoutType.OneColumn,
                actionButtonsInfo: {
                    midColumn: {
                        fullScreen: false
                    }
                },
                // Column width configuration
                columnWidths: {
                    beginColumn: "30%",
                    midColumn: "70%",
                    endColumn: "0%"
                },
                // Resize settings
                resizeMode: "manual",
                minColumnWidth: 320,
                maxBeginColumnWidth: 600
            });
            this.getView().setModel(this.oLayoutModel);
            
            // Initialize master model
            this.oMasterModel = new JSONModel({
                projects: []
            });
            this.getView().setModel(this.oMasterModel, "masterModel");
            
            // Initialize detail model with loading state
            this.oDetailModel = new JSONModel({
                loading: false,
                updateCount: 0,
                lastUpdate: null
            });
            this.getView().setModel(this.oDetailModel, "detailModel");
            
            // Initialize loading state model
            this.oLoadingModel = new JSONModel({
                masterLoading: false,
                detailLoading: false,
                operationLoading: false,
                searchLoading: false,
                deployLoading: false,
                refreshLoading: false,
                currentOperation: null,
                operationProgress: 0
            });
            this.getView().setModel(this.oLoadingModel, "loading");
            
            // Track detail updates
            this._detailUpdateHistory = [];
            
            // Initialize error handling
            this._initializeErrorHandling();
            
            // Load initial data
            this._loadProjects();
            
            // Register for route matched
            this.oRouter.getRoute("projectMasterDetail").attachPatternMatched(this._onRouteMatched, this);
            
            // Load column width preferences
            this._loadColumnWidthPreference();
            
            // Initialize resize handler
            this._initializeResizeHandler();
            
            // Initialize responsive handler
            this._initializeResponsiveHandler();
            
            // Initialize keyboard navigation
            this._initializeKeyboardNavigation();
        },
        
        _initializeResizeHandler: function () {
            // Add resize handle to view after rendering
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    // Add resize handle between columns
                    var oFCL = this.byId("fcl");
                    if (oFCL && !this._resizeHandleAdded) {
                        var $resizeHandle = $('<div class="a2a-fcl-resize-handle"></div>');
                        $resizeHandle.on("mousedown touchstart", this.onBeginColumnResize.bind(this));
                        oFCL.$().append($resizeHandle);
                        this._resizeHandleAdded = true;
                    }
                }.bind(this)
            });
        },

        _onRouteMatched: function (oEvent) {
            var sProjectId = oEvent.getParameter("arguments").projectId;
            
            if (sProjectId) {
                this._showDetail(sProjectId);
            } else {
                this._showMaster();
            }
        },

        _loadProjects: function () {
            var that = this;
            
            // Set master loading state
            this.oLoadingModel.setProperty("/masterLoading", true);
            this.oLoadingModel.setProperty("/currentOperation", "Loading projects...");
            
            jQuery.ajax({
                url: "/api/v2/projects",
                method: "GET",
                timeout: 10000, // 10 second timeout
                success: function (oData) {
                    // Simulate processing time for demo
                    setTimeout(function () {
                        that.oMasterModel.setProperty("/projects", oData.projects || []);
                        that.oLoadingModel.setProperty("/masterLoading", false);
                        that.oLoadingModel.setProperty("/currentOperation", null);
                        MessageToast.show("Projects loaded successfully");
                    }, 500);
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    that.oLoadingModel.setProperty("/masterLoading", false);
                    that.oLoadingModel.setProperty("/currentOperation", null);
                    
                    that._handleError("Failed to load projects", {
                        type: "ajax",
                        url: "/api/v2/projects",
                        status: jqXHR.status,
                        statusText: jqXHR.statusText,
                        textStatus: textStatus,
                        errorThrown: errorThrown
                    });
                }
            });
        },

        _showMaster: function () {
            this.oLayoutModel.setProperty("/layout", LayoutType.OneColumn);
        },

        _loadProjectDetails: function (sProjectId) {
            var that = this;
            
            // Track update start
            var updateStartTime = Date.now();
            
            // Set loading state and increment update count
            this.oDetailModel.setProperty("/loading", true);
            this.oLoadingModel.setProperty("/detailLoading", true);
            this.oLoadingModel.setProperty("/currentOperation", "Loading project details...");
            var iUpdateCount = this.oDetailModel.getProperty("/updateCount") || 0;
            this.oDetailModel.setProperty("/updateCount", iUpdateCount + 1);
            
            // Add fade out effect to detail content
            var oDetailPage = this.byId("detailPage");
            if (oDetailPage) {
                oDetailPage.addStyleClass("a2a-detail-updating");
            }
            
            jQuery.ajax({
                url: "/api/v2/projects/" + sProjectId,
                method: "GET",
                success: function (oData) {
                    // Add mock data for demonstration
                    oData.metrics = {
                        success_rate: "98.5%",
                        avg_response_time: "245ms",
                        total_requests: "12.4K",
                        error_rate: "1.5%"
                    };
                    
                    oData.agents = oData.agents || [
                        {
                            name: "Data Processor Agent",
                            type: "Processing",
                            status: "active",
                            description: "Processes incoming data streams",
                            last_run: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
                        },
                        {
                            name: "Workflow Orchestrator",
                            type: "Orchestration",
                            status: "active",
                            description: "Manages workflow execution",
                            last_run: new Date(Date.now() - 30 * 60 * 1000).toISOString()
                        }
                    ];
                    
                    oData.recent_activity = [
                        {
                            user: "System",
                            description: "Project deployed successfully",
                            timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
                            type: "deployment"
                        },
                        {
                            user: "Developer",
                            description: "Agent configuration updated",
                            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
                            type: "configuration"
                        }
                    ];
                    
                    // Track update completion
                    var updateDuration = Date.now() - updateStartTime;
                    that._detailUpdateHistory.push({
                        projectId: sProjectId,
                        timestamp: new Date(),
                        duration: updateDuration,
                        success: true
                    });
                    
                    // Keep only last 10 updates
                    if (that._detailUpdateHistory.length > 10) {
                        that._detailUpdateHistory.shift();
                    }
                    
                    // Update detail model with animation
                    setTimeout(function () {
                        that.oDetailModel.setData(oData);
                        that.oDetailModel.setProperty("/loading", false);
                        that.oDetailModel.setProperty("/lastUpdate", new Date());
                        
                        // Clear loading states
                        that.oLoadingModel.setProperty("/detailLoading", false);
                        that.oLoadingModel.setProperty("/currentOperation", null);
                        
                        // Remove updating class with fade in effect
                        if (oDetailPage) {
                            oDetailPage.removeStyleClass("a2a-detail-updating");
                        }
                        
                        // Sync selection after loading
                        that._syncMasterSelection(sProjectId);
                        
                        // Show update notification
                        MessageToast.show("Project details updated");
                    }, 300); // Small delay for smooth transition
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    // Track failed update
                    that._detailUpdateHistory.push({
                        projectId: sProjectId,
                        timestamp: new Date(),
                        duration: Date.now() - updateStartTime,
                        success: false,
                        error: jqXHR.statusText || errorThrown || "Unknown error"
                    });
                    
                    that.oDetailModel.setProperty("/loading", false);
                    that.oDetailModel.setProperty("/error", true);
                    that.oDetailModel.setProperty("/errorMessage", "Failed to load project details");
                    
                    // Clear loading states
                    that.oLoadingModel.setProperty("/detailLoading", false);
                    that.oLoadingModel.setProperty("/currentOperation", null);
                    
                    // Remove updating class
                    if (oDetailPage) {
                        oDetailPage.removeStyleClass("a2a-detail-updating");
                    }
                    
                    that._handleError("Failed to load project details", {
                        type: "ajax",
                        url: "/api/v2/projects/" + sProjectId,
                        status: jqXHR.status,
                        statusText: jqXHR.statusText,
                        textStatus: textStatus,
                        errorThrown: errorThrown
                    });
                }
            });
        },

        // Master List Events
        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("newValue");
            var oList = this.byId("masterList");
            var oBinding = oList.getBinding("items");
            
            // Set search loading state for longer searches
            this.oLoadingModel.setProperty("/searchLoading", true);
            this.oLoadingModel.setProperty("/currentOperation", "Searching projects...");
            
            // Simulate search delay
            setTimeout(function () {
                if (sQuery && sQuery.length > 0) {
                    var aFilters = [
                        new Filter("name", FilterOperator.Contains, sQuery),
                        new Filter("description", FilterOperator.Contains, sQuery),
                        new Filter("type", FilterOperator.Contains, sQuery)
                    ];
                    var oFilter = new Filter({
                        filters: aFilters,
                        and: false
                    });
                    oBinding.filter([oFilter]);
                } else {
                    oBinding.filter([]);
                }
                
                // Clear search loading state
                this.oLoadingModel.setProperty("/searchLoading", false);
                this.oLoadingModel.setProperty("/currentOperation", null);
                
                var iResultCount = oList.getItems().length;
                MessageToast.show("Search complete - " + iResultCount + " results found");
            }.bind(this), 200);
        },

        onSelectionChange: function (oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            if (oSelectedItem) {
                var sProjectId = oSelectedItem.getCustomData()[0].getValue();
                this.oRouter.navTo("projectMasterDetail", {
                    projectId: sProjectId
                });
            }
        },

        onItemPress: function (oEvent) {
            var oItem = oEvent.getSource();
            var sProjectId = oItem.getCustomData()[0].getValue();
            
            this.oRouter.navTo("projectMasterDetail", {
                projectId: sProjectId
            });
        },

        // Layout Events
        onStateChanged: function (oEvent) {
            var bIsNavigationArrow = oEvent.getParameter("isNavigationArrow");
            var sLayout = oEvent.getParameter("layout");
            
            this.oLayoutModel.setProperty("/layout", sLayout);
            
            // Update action buttons info
            var oActionButtonsInfo = {
                midColumn: {
                    fullScreen: sLayout === LayoutType.MidColumnFullScreen
                }
            };
            this.oLayoutModel.setProperty("/actionButtonsInfo", oActionButtonsInfo);
        },

        onDetailNavButtonPress: function () {
            this.oLayoutModel.setProperty("/layout", LayoutType.OneColumn);
            this.oRouter.navTo("projectMasterDetail");
        },

        onEndNavButtonPress: function () {
            this.oLayoutModel.setProperty("/layout", LayoutType.TwoColumnsMidExpanded);
        },

        // Action Events
        onCreateProject: function () {
            this.oRouter.navTo("projectCreate");
        },

        onRefresh: function () {
            this.oLoadingModel.setProperty("/refreshLoading", true);
            this.oLoadingModel.setProperty("/currentOperation", "Refreshing projects...");
            
            this._loadProjects();
            
            // Clear refresh loading after projects load
            setTimeout(function () {
                this.oLoadingModel.setProperty("/refreshLoading", false);
            }.bind(this), 1000);
        },

        onEditProject: function () {
            var oProject = this.oDetailModel.getData();
            this.oRouter.navTo("projectEdit", {
                projectId: oProject.project_id
            });
        },

        onDeployProject: function () {
            var oProject = this.oDetailModel.getData();
            
            MessageBox.confirm(
                "Are you sure you want to deploy project '" + oProject.name + "'?",
                {
                    title: "Deploy Project",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._deployProject(oProject.project_id);
                        }
                    }.bind(this)
                }
            );
        },

        _deployProject: function (sProjectId) {
            var that = this;
            
            // Set deployment loading state
            this.oLoadingModel.setProperty("/deployLoading", true);
            this.oLoadingModel.setProperty("/operationLoading", true);
            this.oLoadingModel.setProperty("/currentOperation", "Deploying project...");
            this.oLoadingModel.setProperty("/operationProgress", 0);
            
            // Simulate deployment progress
            var iProgress = 0;
            var oProgressInterval = setInterval(function () {
                iProgress += 10;
                that.oLoadingModel.setProperty("/operationProgress", iProgress);
                if (iProgress >= 100) {
                    clearInterval(oProgressInterval);
                }
            }, 300);
            
            jQuery.ajax({
                url: "/api/v2/projects/" + sProjectId + "/deploy",
                method: "POST",
                timeout: 30000, // 30 second timeout
                success: function () {
                    setTimeout(function () {
                        clearInterval(oProgressInterval);
                        that.oLoadingModel.setProperty("/deployLoading", false);
                        that.oLoadingModel.setProperty("/operationLoading", false);
                        that.oLoadingModel.setProperty("/currentOperation", null);
                        that.oLoadingModel.setProperty("/operationProgress", 0);
                        
                        MessageToast.show("Project deployment completed successfully");
                        that._loadProjectDetails(sProjectId);
                    }, 2000);
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    clearInterval(oProgressInterval);
                    that.oLoadingModel.setProperty("/deployLoading", false);
                    that.oLoadingModel.setProperty("/operationLoading", false);
                    that.oLoadingModel.setProperty("/currentOperation", null);
                    that.oLoadingModel.setProperty("/operationProgress", 0);
                    
                    that._handleError("Failed to deploy project", {
                        type: "ajax",
                        url: "/api/v2/projects/" + sProjectId + "/deploy",
                        status: jqXHR.status,
                        statusText: jqXHR.statusText,
                        textStatus: textStatus,
                        errorThrown: errorThrown
                    });
                }
            });
        },

        onMoreActions: function () {
            MessageToast.show("More actions menu");
        },

        onToggleFullScreen: function () {
            var sCurrentLayout = this.oLayoutModel.getProperty("/layout");
            var sNewLayout = sCurrentLayout === LayoutType.MidColumnFullScreen 
                ? LayoutType.TwoColumnsMidExpanded 
                : LayoutType.MidColumnFullScreen;
            
            this.oLayoutModel.setProperty("/layout", sNewLayout);
        },

        // Agent Actions
        onAddAgent: function () {
            var oProject = this.oDetailModel.getData();
            this.oRouter.navTo("agentBuilder", {
                projectId: oProject.project_id
            });
        },

        onAgentPress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("detailModel");
            var oAgent = oBindingContext.getObject();
            
            // Show agent details in end column
            this.oLayoutModel.setProperty("/layout", LayoutType.ThreeColumnsMidExpanded);
            
            // Load agent details (placeholder)
            MessageToast.show("Loading agent: " + oAgent.name);
        },

        onRunAgent: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("detailModel");
            var oAgent = oBindingContext.getObject();
            
            MessageToast.show("Running agent: " + oAgent.name);
        },

        onEditAgent: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("detailModel");
            var oAgent = oBindingContext.getObject();
            
            MessageToast.show("Editing agent: " + oAgent.name);
        },

        onViewAllActivity: function () {
            MessageToast.show("View all activity");
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) return "";
            
            var oDate = new Date(sDate);
            return oDate.toLocaleDateString();
        },

        formatRelativeTime: function (sDate) {
            if (!sDate) return "";
            
            var oDate = new Date(sDate);
            var oNow = new Date();
            var iDiff = oNow.getTime() - oDate.getTime();
            var iMinutes = Math.floor(iDiff / (1000 * 60));
            var iHours = Math.floor(iMinutes / 60);
            var iDays = Math.floor(iHours / 24);
            
            if (iMinutes < 1) {
                return "Just now";
            } else if (iMinutes < 60) {
                return iMinutes + " minutes ago";
            } else if (iHours < 24) {
                return iHours + " hours ago";
            } else if (iDays === 1) {
                return "Yesterday";
            } else if (iDays < 7) {
                return iDays + " days ago";
            } else {
                return oDate.toLocaleDateString();
            }
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Success";
                case "deployed":
                    return "Success";
                case "inactive":
                    return "Warning";
                case "error":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatStatusIcon: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "sap-icon://status-positive";
                case "deployed":
                    return "sap-icon://status-positive";
                case "inactive":
                    return "sap-icon://status-inactive";
                case "error":
                    return "sap-icon://status-negative";
                default:
                    return "";
            }
        },

        formatAgentsState: function (iCount) {
            if (iCount > 10) {
                return "Success";
            } else if (iCount > 5) {
                return "Warning";
            } else {
                return "None";
            }
        },

        formatDeploymentState: function (sStatus) {
            switch (sStatus) {
                case "deployed":
                    return "Success";
                case "deploying":
                    return "Warning";
                case "failed":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatAgentStatusState: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Success";
                case "inactive":
                    return "Warning";
                case "error":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatActivityIcon: function (sType) {
            switch (sType) {
                case "deployment":
                    return "sap-icon://cloud";
                case "configuration":
                    return "sap-icon://settings";
                case "execution":
                    return "sap-icon://play";
                default:
                    return "sap-icon://information";
            }
        },
        
        // Detail Update Methods
        
        getUpdateStatistics: function () {
            if (!this._detailUpdateHistory || this._detailUpdateHistory.length === 0) {
                return {
                    totalUpdates: 0,
                    successfulUpdates: 0,
                    failedUpdates: 0,
                    averageDuration: 0,
                    lastUpdate: null
                };
            }
            
            var iTotal = this._detailUpdateHistory.length;
            var iSuccessful = this._detailUpdateHistory.filter(function (update) {
                return update.success;
            }).length;
            var iDurationSum = this._detailUpdateHistory.reduce(function (sum, update) {
                return sum + update.duration;
            }, 0);
            
            return {
                totalUpdates: iTotal,
                successfulUpdates: iSuccessful,
                failedUpdates: iTotal - iSuccessful,
                averageDuration: Math.round(iDurationSum / iTotal),
                lastUpdate: this._detailUpdateHistory[iTotal - 1].timestamp
            };
        },
        
        forceDetailRefresh: function () {
            var sCurrentProjectId = this.oDetailModel.getProperty("/project_id");
            if (sCurrentProjectId) {
                this._loadProjectDetails(sCurrentProjectId);
            }
        },
        
        // Keyboard Navigation
        
        _initializeKeyboardNavigation: function () {
            // Add keyboard event listener to the view
            this.getView().addEventDelegate({
                onAfterRendering: function () {
                    // Add keyboard support to master list
                    var oMasterList = this.byId("masterList");
                    if (oMasterList) {
                        oMasterList.attachEvent("_focusChanged", this._onMasterListFocusChanged.bind(this));
                    }
                    
                    // Add global keyboard shortcuts
                    jQuery(document).on("keydown.masterDetail", this._onGlobalKeyDown.bind(this));
                }.bind(this),
                
                onExit: function () {
                    jQuery(document).off("keydown.masterDetail");
                }
            });
        },
        
        _onGlobalKeyDown: function (oEvent) {
            var iKeyCode = oEvent.which || oEvent.keyCode;
            var bCtrlKey = oEvent.ctrlKey || oEvent.metaKey;
            var bAltKey = oEvent.altKey;
            
            // Handle global shortcuts when focus is in master-detail area
            var $target = jQuery(oEvent.target);
            if (!$target.closest(".a2a-tablet-page").length) {
                return; // Not in our component
            }
            
            switch (iKeyCode) {
                case 70: // F key
                    if (bCtrlKey) {
                        oEvent.preventDefault();
                        this._focusSearchField();
                    }
                    break;
                    
                case 82: // R key
                    if (bCtrlKey) {
                        oEvent.preventDefault();
                        this._refreshCurrentView();
                    }
                    break;
                    
                case 78: // N key
                    if (bCtrlKey) {
                        oEvent.preventDefault();
                        this.onCreateProject();
                    }
                    break;
                    
                case 27: // Escape key
                    this._handleEscapeKey();
                    break;
                    
                case 37: // Left arrow
                    if (bAltKey) {
                        oEvent.preventDefault();
                        this._navigateToMaster();
                    }
                    break;
                    
                case 39: // Right arrow
                    if (bAltKey) {
                        oEvent.preventDefault();
                        this._navigateToDetail();
                    }
                    break;
                    
                case 38: // Up arrow
                case 40: // Down arrow
                    if (bCtrlKey) {
                        oEvent.preventDefault();
                        this._navigateList(iKeyCode === 38 ? "up" : "down");
                    }
                    break;
            }
        },
        
        _onMasterListFocusChanged: function () {
            // Handle focus changes in master list
            var oList = this.byId("masterList");
            if (!oList) return;
            
            var oFocusedItem = oList.getItems().find(function (oItem) {
                return oItem.$().hasClass("sapMLIBFocused");
            });
            
            if (oFocusedItem) {
                var sProjectId = oFocusedItem.getCustomData()[0].getValue();
                this._preloadProjectDetails(sProjectId);
            }
        },
        
        _focusSearchField: function () {
            var oSearchField = this.getView().byId("searchField");
            if (oSearchField) {
                oSearchField.focus();
            }
        },
        
        _refreshCurrentView: function () {
            var sCurrentLayout = this.oLayoutModel.getProperty("/layout");
            if (sCurrentLayout === LayoutType.OneColumn) {
                this._loadProjects();
            } else {
                this.forceDetailRefresh();
            }
            MessageToast.show("View refreshed");
        },
        
        _handleEscapeKey: function () {
            var sCurrentLayout = this.oLayoutModel.getProperty("/layout");
            
            // Close detail view and go back to master
            if (sCurrentLayout !== LayoutType.OneColumn) {
                this.oLayoutModel.setProperty("/layout", LayoutType.OneColumn);
                this.oRouter.navTo("projectMasterDetail");
            }
        },
        
        _navigateToMaster: function () {
            var oList = this.byId("masterList");
            if (oList) {
                var oFirstItem = oList.getItems()[0];
                if (oFirstItem) {
                    oFirstItem.focus();
                }
            }
        },
        
        _navigateToDetail: function () {
            var sCurrentLayout = this.oLayoutModel.getProperty("/layout");
            if (sCurrentLayout !== LayoutType.OneColumn) {
                // Focus first focusable element in detail area
                var oDetailPage = this.byId("detailPage");
                if (oDetailPage) {
                    var $firstButton = oDetailPage.$().find("button:visible:first");
                    if ($firstButton.length) {
                        $firstButton.focus();
                    }
                }
            }
        },
        
        _navigateList: function (sDirection) {
            var oList = this.byId("masterList");
            if (!oList) return;
            
            var aItems = oList.getItems();
            var oSelectedItem = oList.getSelectedItem();
            var iCurrentIndex = oSelectedItem ? aItems.indexOf(oSelectedItem) : -1;
            
            var iNewIndex;
            if (sDirection === "up") {
                iNewIndex = Math.max(0, iCurrentIndex - 1);
            } else {
                iNewIndex = Math.min(aItems.length - 1, iCurrentIndex + 1);
            }
            
            if (iNewIndex !== iCurrentIndex && aItems[iNewIndex]) {
                var sProjectId = aItems[iNewIndex].getCustomData()[0].getValue();
                this.oRouter.navTo("projectMasterDetail", {
                    projectId: sProjectId
                });
            }
        },
        
        _preloadProjectDetails: function (sProjectId) {
            // Preload project details for smoother keyboard navigation
            // This could be implemented to cache project details
            var sCacheKey = "project_" + sProjectId;
            if (!this._projectCache) {
                this._projectCache = {};
            }
            
            // Simple cache implementation
            if (!this._projectCache[sCacheKey]) {
                jQuery.ajax({
                    url: "/api/v2/projects/" + sProjectId,
                    method: "GET",
                    success: function (oData) {
                        this._projectCache[sCacheKey] = {
                            data: oData,
                            timestamp: Date.now()
                        };
                    }.bind(this),
                    error: function () {
                        // Ignore preload errors
                    }
                });
            }
        },
        
        // Error Handling System
        
        _initializeErrorHandling: function () {
            // Initialize error tracking
            this._errorHistory = [];
            
            // Initialize error model
            this.oErrorModel = new JSONModel({
                hasErrors: false,
                errorCount: 0,
                lastError: null,
                networkError: false,
                serverError: false,
                timeoutError: false,
                validationError: false
            });
            this.getView().setModel(this.oErrorModel, "errors");
            
            // Global error handler for unhandled AJAX errors
            jQuery(document).ajaxError(this._onGlobalAjaxError.bind(this));
        },
        
        _onGlobalAjaxError: function (event, jqXHR, ajaxSettings, thrownError) {
            // Only handle errors from our API endpoints
            if (ajaxSettings.url && ajaxSettings.url.includes("/api/v2/projects")) {
                this._handleError(thrownError || "Network error", {
                    type: "network",
                    url: ajaxSettings.url,
                    status: jqXHR.status,
                    statusText: jqXHR.statusText,
                    responseText: jqXHR.responseText
                });
            }
        },
        
        _handleError: function (sMessage, oContext) {
            var oError = {
                message: sMessage,
                timestamp: new Date(),
                context: oContext || {},
                id: this._generateErrorId()
            };
            
            // Add to error history
            this._errorHistory.push(oError);
            
            // Keep only last 20 errors
            if (this._errorHistory.length > 20) {
                this._errorHistory.shift();
            }
            
            // Update error model
            this._updateErrorModel();
            
            // Categorize error type
            this._categorizeError(oError);
            
            // Show user-friendly error message
            this._showErrorToUser(oError);
            
            // Log error for debugging
            console.error("ProjectMasterDetail Error:", oError);
        },
        
        _generateErrorId: function () {
            return "ERR_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        },
        
        _updateErrorModel: function () {
            var iErrorCount = this._errorHistory.length;
            var oLastError = iErrorCount > 0 ? this._errorHistory[iErrorCount - 1] : null;
            
            this.oErrorModel.setProperty("/hasErrors", iErrorCount > 0);
            this.oErrorModel.setProperty("/errorCount", iErrorCount);
            this.oErrorModel.setProperty("/lastError", oLastError);
        },
        
        _categorizeError: function (oError) {
            var oContext = oError.context;
            var iStatus = oContext.status;
            
            // Reset error type flags
            this.oErrorModel.setProperty("/networkError", false);
            this.oErrorModel.setProperty("/serverError", false);
            this.oErrorModel.setProperty("/timeoutError", false);
            this.oErrorModel.setProperty("/validationError", false);
            
            // Categorize based on status code and context
            if (iStatus === 0 || oContext.type === "timeout") {
                this.oErrorModel.setProperty("/networkError", true);
                oError.category = "network";
            } else if (iStatus >= 500) {
                this.oErrorModel.setProperty("/serverError", true);
                oError.category = "server";
            } else if (iStatus >= 400 && iStatus < 500) {
                this.oErrorModel.setProperty("/validationError", true);
                oError.category = "validation";
            } else if (oError.message.toLowerCase().includes("timeout")) {
                this.oErrorModel.setProperty("/timeoutError", true);
                oError.category = "timeout";
            } else {
                oError.category = "unknown";
            }
        },
        
        _showErrorToUser: function (oError) {
            var sUserMessage = this._getUserFriendlyMessage(oError);
            var sTitle = this._getErrorTitle(oError);
            
            // For critical errors, show dialog
            if (this._isCriticalError(oError)) {
                MessageBox.error(sUserMessage, {
                    title: sTitle,
                    details: this._getErrorDetails(oError),
                    actions: [MessageBox.Action.OK, "Retry", "Report"],
                    onClose: function (sAction) {
                        if (sAction === "Retry") {
                            this._retryLastOperation(oError);
                        } else if (sAction === "Report") {
                            this._reportError(oError);
                        }
                    }.bind(this)
                });
            } else {
                // For non-critical errors, show toast
                MessageToast.show(sUserMessage, {
                    duration: 5000,
                    width: "25em"
                });
            }
        },
        
        _getUserFriendlyMessage: function (oError) {
            switch (oError.category) {
                case "network":
                    return "Network connection failed. Please check your internet connection and try again.";
                case "server":
                    return "The server encountered an error. Please try again later or contact support.";
                case "validation":
                    return "Invalid request. Please check your input and try again.";
                case "timeout":
                    return "The operation timed out. Please try again.";
                default:
                    return "An unexpected error occurred: " + oError.message;
            }
        },
        
        _getErrorTitle: function (oError) {
            switch (oError.category) {
                case "network":
                    return "Connection Error";
                case "server":
                    return "Server Error";
                case "validation":
                    return "Validation Error";
                case "timeout":
                    return "Timeout Error";
                default:
                    return "Application Error";
            }
        },
        
        _getErrorDetails: function (oError) {
            var sDetails = "Error ID: " + oError.id + "\n";
            sDetails += "Time: " + oError.timestamp.toLocaleString() + "\n";
            if (oError.context.url) {
                sDetails += "URL: " + oError.context.url + "\n";
            }
            if (oError.context.status) {
                sDetails += "Status: " + oError.context.status + " " + oError.context.statusText + "\n";
            }
            return sDetails;
        },
        
        _isCriticalError: function (oError) {
            // Critical errors require user attention
            return oError.category === "server" || 
                   oError.category === "network" || 
                   (oError.context.status >= 500);
        },
        
        _retryLastOperation: function (oError) {
            // Attempt to retry the failed operation
            var oContext = oError.context;
            if (oContext.url && oContext.url.includes("/projects")) {
                if (oContext.url.includes("/deploy")) {
                    // Retry deployment
                    var sProjectId = this._extractProjectIdFromUrl(oContext.url);
                    this._deployProject(sProjectId);
                } else if (oContext.url.endsWith("/projects")) {
                    // Retry loading projects
                    this._loadProjects();
                } else {
                    // Retry loading project details
                    var sProjectId = this._extractProjectIdFromUrl(oContext.url);
                    this._loadProjectDetails(sProjectId);
                }
            }
        },
        
        _extractProjectIdFromUrl: function (sUrl) {
            var aMatches = sUrl.match(/\/projects\/([^\/]+)/);
            return aMatches ? aMatches[1] : null;
        },
        
        _reportError: function (oError) {
            // Prepare error report
            var oReport = {
                errorId: oError.id,
                message: oError.message,
                timestamp: oError.timestamp,
                context: oError.context,
                userAgent: navigator.userAgent,
                url: window.location.href
            };
            
            // Send error report (mock implementation)
            jQuery.ajax({
                url: "/api/v2/error-reports",
                method: "POST",
                data: JSON.stringify(oReport),
                contentType: "application/json",
                success: function () {
                    MessageToast.show("Error report sent successfully");
                },
                error: function () {
                    MessageToast.show("Failed to send error report");
                }
            });
        },
        
        getErrorStatistics: function () {
            if (!this._errorHistory || this._errorHistory.length === 0) {
                return {
                    totalErrors: 0,
                    networkErrors: 0,
                    serverErrors: 0,
                    validationErrors: 0,
                    timeoutErrors: 0,
                    lastError: null
                };
            }
            
            var oStats = {
                totalErrors: this._errorHistory.length,
                networkErrors: 0,
                serverErrors: 0,
                validationErrors: 0,
                timeoutErrors: 0,
                lastError: this._errorHistory[this._errorHistory.length - 1]
            };
            
            this._errorHistory.forEach(function (oError) {
                switch (oError.category) {
                    case "network":
                        oStats.networkErrors++;
                        break;
                    case "server":
                        oStats.serverErrors++;
                        break;
                    case "validation":
                        oStats.validationErrors++;
                        break;
                    case "timeout":
                        oStats.timeoutErrors++;
                        break;
                }
            });
            
            return oStats;
        },
        
        clearErrors: function () {
            this._errorHistory = [];
            this._updateErrorModel();
            MessageToast.show("Error history cleared");
        },
        
        onShowErrorDetails: function () {
            if (!this._pErrorDialog) {
                this._pErrorDialog = this._createErrorDialog();
            }
            this._pErrorDialog.then(function (oDialog) {
                this._updateErrorDialogContent();
                oDialog.open();
            }.bind(this));
        },
        
        _createErrorDialog: function () {
            return this.loadFragment({
                name: "com.sap.a2a.developerportal.fragment.ErrorDialog",
                controller: this
            });
        },
        
        _updateErrorDialogContent: function () {
            // Update error dialog with current error information
            var oErrorModel = new JSONModel({
                errors: this._errorHistory,
                statistics: this.getErrorStatistics()
            });
            
            this._pErrorDialog.then(function (oDialog) {
                oDialog.setModel(oErrorModel, "errorDialog");
            });
        },
        
        onCloseErrorDialog: function () {
            this._pErrorDialog.then(function (oDialog) {
                oDialog.close();
            });
        },
        
        onClearAllErrors: function () {
            this.clearErrors();
            this.onCloseErrorDialog();
        },
        
        onRetryError: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("errorDialog");
            var oError = oBindingContext.getObject();
            this._retryLastOperation(oError);
        },
        
        onReportError: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("errorDialog");
            var oError = oBindingContext.getObject();
            this._reportError(oError);
        },
        
        // Error Dialog Formatters
        
        formatErrorTime: function (oTimestamp) {
            if (!oTimestamp) return "";
            var oDate = new Date(oTimestamp);
            return oDate.toLocaleTimeString();
        },
        
        formatErrorState: function (sCategory) {
            switch (sCategory) {
                case "network":
                    return "Warning";
                case "server":
                    return "Error";
                case "validation":
                    return "Warning";
                case "timeout":
                    return "Warning";
                default:
                    return "None";
            }
        },
        
        formatErrorDetails: function (oError) {
            if (!oError) return "";
            
            var sDetails = "<strong>Error Message:</strong><br/>" + oError.message + "<br/><br/>";
            sDetails += "<strong>Category:</strong> " + (oError.category || "Unknown") + "<br/>";
            sDetails += "<strong>Time:</strong> " + oError.timestamp.toLocaleString() + "<br/>";
            sDetails += "<strong>Error ID:</strong> " + oError.id + "<br/>";
            
            if (oError.context) {
                sDetails += "<br/><strong>Technical Details:</strong><br/>";
                if (oError.context.url) {
                    sDetails += "URL: " + oError.context.url + "<br/>";
                }
                if (oError.context.status) {
                    sDetails += "HTTP Status: " + oError.context.status + " " + oError.context.statusText + "<br/>";
                }
                if (oError.context.textStatus) {
                    sDetails += "Request Status: " + oError.context.textStatus + "<br/>";
                }
            }
            
            return sDetails;
        },
        
        // Resize functionality
        
        onBeginColumnResize: function (oEvent) {
            var oFCL = this.byId("fcl");
            
            // Initialize resize
            this._bResizing = true;
            this._iStartX = oEvent.pageX || oEvent.touches[0].pageX;
            this._iStartWidth = oFCL.$().find(".sapFFCLColumn--begin").width();
            
            // Add event listeners
            $(document).on("mousemove.resize touchmove.resize", this._onResize.bind(this));
            $(document).on("mouseup.resize touchend.resize", this._onResizeEnd.bind(this));
            
            // Prevent text selection during resize
            $("body").addClass("a2a-resizing");
        },
        
        _onResize: function (oEvent) {
            if (!this._bResizing) return;
            
            var iCurrentX = oEvent.pageX || oEvent.touches[0].pageX;
            var iDelta = iCurrentX - this._iStartX;
            var iNewWidth = this._iStartWidth + iDelta;
            
            // Apply constraints
            var iMinWidth = this.oLayoutModel.getProperty("/minColumnWidth");
            var iMaxWidth = this.oLayoutModel.getProperty("/maxBeginColumnWidth");
            
            iNewWidth = Math.max(iMinWidth, Math.min(iMaxWidth, iNewWidth));
            
            // Calculate percentage
            var iTotalWidth = this.byId("fcl").$().width();
            var iBeginPercent = (iNewWidth / iTotalWidth) * 100;
            var iMidPercent = 100 - iBeginPercent;
            
            // Update model
            this.oLayoutModel.setProperty("/columnWidths/beginColumn", iBeginPercent + "%");
            this.oLayoutModel.setProperty("/columnWidths/midColumn", iMidPercent + "%");
            
            // Apply to FCL
            this._applyColumnWidths();
        },
        
        _onResizeEnd: function () {
            this._bResizing = false;
            
            // Remove event listeners
            $(document).off("mousemove.resize touchmove.resize");
            $(document).off("mouseup.resize touchend.resize");
            
            // Remove resizing class
            $("body").removeClass("a2a-resizing");
            
            // Save user preference
            this._saveColumnWidthPreference();
        },
        
        _applyColumnWidths: function () {
            var oFCL = this.byId("fcl");
            var oWidths = this.oLayoutModel.getProperty("/columnWidths");
            
            // Apply custom widths via CSS
            oFCL.$().find(".sapFFCLColumn--begin").css("width", oWidths.beginColumn);
            oFCL.$().find(".sapFFCLColumn--mid").css("width", oWidths.midColumn);
        },
        
        _saveColumnWidthPreference: function () {
            var oWidths = this.oLayoutModel.getProperty("/columnWidths");
            
            // Save to local storage
            localStorage.setItem("a2a.fcl.columnWidths", JSON.stringify(oWidths));
        },
        
        _loadColumnWidthPreference: function () {
            var sWidths = localStorage.getItem("a2a.fcl.columnWidths");
            if (sWidths) {
                try {
                    var oWidths = JSON.parse(sWidths);
                    this.oLayoutModel.setProperty("/columnWidths", oWidths);
                    this._applyColumnWidths();
                } catch (e) {
                    console.error("Failed to load column width preferences", e);
                }
            }
        },
        
        onResetColumnWidths: function () {
            // Reset to default widths
            this.oLayoutModel.setProperty("/columnWidths/beginColumn", "30%");
            this.oLayoutModel.setProperty("/columnWidths/midColumn", "70%");
            this._applyColumnWidths();
            this._saveColumnWidthPreference();
            
            MessageToast.show("Column widths reset to default");
        },
        
        // Responsive behavior
        
        _initializeResponsiveHandler: function () {
            // Set up responsive breakpoints
            this._oResponsiveConfig = {
                phone: {
                    maxWidth: 599,
                    layout: LayoutType.OneColumn,
                    beginColumnWidth: "100%",
                    midColumnWidth: "100%"
                },
                tablet: {
                    minWidth: 600,
                    maxWidth: 1023,
                    layout: LayoutType.TwoColumnsMidExpanded,
                    beginColumnWidth: "40%",
                    midColumnWidth: "60%",
                    minBeginColumnWidth: 320,
                    maxBeginColumnWidth: 450
                },
                desktop: {
                    minWidth: 1024,
                    layout: LayoutType.TwoColumnsMidExpanded,
                    beginColumnWidth: "30%",
                    midColumnWidth: "70%",
                    minBeginColumnWidth: 320,
                    maxBeginColumnWidth: 600
                }
            };
            
            // Initialize device model
            var oDeviceModel = new JSONModel({
                isPhone: false,
                isTablet: false,
                isDesktop: true,
                currentBreakpoint: "desktop"
            });
            this.getView().setModel(oDeviceModel, "device");
            
            // Check initial screen size
            this._checkResponsiveBreakpoint();
            
            // Set up resize listener with debouncing
            this._iResizeTimer = null;
            $(window).on("resize.responsive", function () {
                clearTimeout(this._iResizeTimer);
                this._iResizeTimer = setTimeout(function () {
                    this._checkResponsiveBreakpoint();
                }.bind(this), 200);
            }.bind(this));
            
            // Clean up on exit
            this.getView().addEventDelegate({
                onExit: function () {
                    $(window).off("resize.responsive");
                    clearTimeout(this._iResizeTimer);
                }.bind(this)
            });
        },
        
        _checkResponsiveBreakpoint: function () {
            var iWindowWidth = $(window).width();
            var oDeviceModel = this.getView().getModel("device");
            var oCurrentBreakpoint = null;
            var sBreakpointName = "";
            
            // Determine current breakpoint
            if (iWindowWidth <= this._oResponsiveConfig.phone.maxWidth) {
                oCurrentBreakpoint = this._oResponsiveConfig.phone;
                sBreakpointName = "phone";
            } else if (iWindowWidth >= this._oResponsiveConfig.tablet.minWidth && 
                      iWindowWidth <= this._oResponsiveConfig.tablet.maxWidth) {
                oCurrentBreakpoint = this._oResponsiveConfig.tablet;
                sBreakpointName = "tablet";
            } else {
                oCurrentBreakpoint = this._oResponsiveConfig.desktop;
                sBreakpointName = "desktop";
            }
            
            // Update device model
            oDeviceModel.setProperty("/isPhone", sBreakpointName === "phone");
            oDeviceModel.setProperty("/isTablet", sBreakpointName === "tablet");
            oDeviceModel.setProperty("/isDesktop", sBreakpointName === "desktop");
            oDeviceModel.setProperty("/currentBreakpoint", sBreakpointName);
            
            // Apply responsive layout changes
            this._applyResponsiveLayout(oCurrentBreakpoint, sBreakpointName);
        },
        
        _applyResponsiveLayout: function (oBreakpointConfig, sBreakpointName) {
            var sCurrentLayout = this.oLayoutModel.getProperty("/layout");
            
            // For phones, always use OneColumn layout
            if (sBreakpointName === "phone") {
                if (sCurrentLayout !== LayoutType.OneColumn) {
                    this.oLayoutModel.setProperty("/layout", LayoutType.OneColumn);
                }
                
                // Hide resize handle on phones
                $(".a2a-fcl-resize-handle").hide();
            } else {
                // Show resize handle on tablet/desktop
                $(".a2a-fcl-resize-handle").show();
                
                // Update column constraints based on breakpoint
                this.oLayoutModel.setProperty("/minColumnWidth", 
                    oBreakpointConfig.minBeginColumnWidth || 320);
                this.oLayoutModel.setProperty("/maxBeginColumnWidth", 
                    oBreakpointConfig.maxBeginColumnWidth || 600);
                
                // If coming from phone view, restore two-column layout
                if (sCurrentLayout === LayoutType.OneColumn && this._sLastSelectedProjectId) {
                    this.oLayoutModel.setProperty("/layout", LayoutType.TwoColumnsMidExpanded);
                }
                
                // Apply default column widths for this breakpoint
                if (!this._bUserResized) {
                    this.oLayoutModel.setProperty("/columnWidths/beginColumn", 
                        oBreakpointConfig.beginColumnWidth);
                    this.oLayoutModel.setProperty("/columnWidths/midColumn", 
                        oBreakpointConfig.midColumnWidth);
                    this._applyColumnWidths();
                }
            }
            
            // Add CSS class for breakpoint-specific styling
            var oFCL = this.byId("fcl");
            if (oFCL) {
                oFCL.removeStyleClass("a2a-fcl-phone a2a-fcl-tablet a2a-fcl-desktop");
                oFCL.addStyleClass("a2a-fcl-" + sBreakpointName);
            }
        },
        
        _showDetail: function (sProjectId) {
            this._sLastSelectedProjectId = sProjectId;
            var oDeviceModel = this.getView().getModel("device");
            
            // On phone, use OneColumn layout
            if (oDeviceModel.getProperty("/isPhone")) {
                this.oLayoutModel.setProperty("/layout", LayoutType.OneColumn);
                // Navigate to detail view on phone
                this._loadProjectDetails(sProjectId);
                // Show back button
                this.oLayoutModel.setProperty("/actionButtonsInfo/midColumn/fullScreen", false);
            } else {
                // On tablet/desktop, use TwoColumnsMidExpanded
                this.oLayoutModel.setProperty("/layout", LayoutType.TwoColumnsMidExpanded);
                this._loadProjectDetails(sProjectId);
            }
            
            // Synchronize selection in master list
            this._syncMasterSelection(sProjectId);
        },
        
        _syncMasterSelection: function (sProjectId) {
            var oList = this.byId("masterList");
            if (!oList) return;
            
            var aItems = oList.getItems();
            var bFound = false;
            
            aItems.forEach(function (oItem) {
                var sItemProjectId = oItem.getCustomData()[0].getValue();
                if (sItemProjectId === sProjectId) {
                    oList.setSelectedItem(oItem);
                    bFound = true;
                    
                    // Ensure item is visible in viewport
                    this._ensureItemVisible(oItem);
                }
            }.bind(this));
            
            // If not found, clear selection
            if (!bFound) {
                oList.removeSelections();
            }
            
            // Update selection model
            this.oMasterModel.setProperty("/selectedProjectId", sProjectId);
        },
        
        _ensureItemVisible: function (oItem) {
            // Scroll item into view if needed
            setTimeout(function () {
                var oItemDom = oItem.getDomRef();
                if (oItemDom) {
                    oItemDom.scrollIntoView({
                        behavior: "smooth",
                        block: "nearest"
                    });
                }
            }, 100);
        },
        
        _onResize: function (oEvent) {
            if (!this._bResizing) return;
            
            // Mark that user has manually resized
            this._bUserResized = true;
            
            var iCurrentX = oEvent.pageX || oEvent.touches[0].pageX;
            var iDelta = iCurrentX - this._iStartX;
            var iNewWidth = this._iStartWidth + iDelta;
            
            // Apply constraints
            var iMinWidth = this.oLayoutModel.getProperty("/minColumnWidth");
            var iMaxWidth = this.oLayoutModel.getProperty("/maxBeginColumnWidth");
            
            iNewWidth = Math.max(iMinWidth, Math.min(iMaxWidth, iNewWidth));
            
            // Calculate percentage
            var iTotalWidth = this.byId("fcl").$().width();
            var iBeginPercent = (iNewWidth / iTotalWidth) * 100;
            var iMidPercent = 100 - iBeginPercent;
            
            // Update model
            this.oLayoutModel.setProperty("/columnWidths/beginColumn", iBeginPercent + "%");
            this.oLayoutModel.setProperty("/columnWidths/midColumn", iMidPercent + "%");
            
            // Apply to FCL
            this._applyColumnWidths();
        }
    });
});
