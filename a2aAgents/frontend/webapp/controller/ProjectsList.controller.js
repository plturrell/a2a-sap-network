sap.ui.define([
    'com/sap/a2a/portal/controller/BaseController',
    'sap/ui/model/json/JSONModel',
    'sap/m/MessageToast',
    'sap/ui/core/Fragment'
], (BaseController, JSONModel, MessageToast, Fragment) => {
    'use strict';

    return BaseController.extend('com.sap.a2a.portal.controller.ProjectsList', {
        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                busy: false,
                delay: 0,
                selectedIndices: []
            });
            this.setModel(oViewModel, 'projectsListView');

            // Initialize scroll position management
            this._initScrollPositionManager();

            // Load projects with virtual scrolling support
            this._loadProjects();
            
            // Setup scroll event listener for lazy loading
            this._setupVirtualScrolling();
        },
        
        _initScrollPositionManager: function () {
            // Storage key for this view's scroll position
            this._sScrollPositionKey = 'ProjectsList_ScrollPosition';
            this._iSavedScrollPosition = 0;
            this._bScrollPositionRestored = false;
            
            // Listen for route changes to save scroll position
            const oRouter = this.getRouter();
            if (oRouter) {
                oRouter.attachBeforeRouteMatched(this._onBeforeRouteMatched, this);
            }
        },
        
        _onBeforeRouteMatched: function (oEvent) {
            // Save current scroll position before navigation
            this._saveScrollPosition();
        },
        
        _saveScrollPosition: function () {
            const oTable = this.byId('projectsTable');
            if (oTable) {
                const iFirstVisibleRow = oTable.getFirstVisibleRow();
                sessionStorage.setItem(this._sScrollPositionKey, iFirstVisibleRow.toString());
            }
        },
        
        _restoreScrollPosition: function () {
            if (this._bScrollPositionRestored) {
                return;
            }
            
            const sSavedPosition = sessionStorage.getItem(this._sScrollPositionKey);
            if (sSavedPosition) {
                const iPosition = parseInt(sSavedPosition, 10);
                const oTable = this.byId('projectsTable');
                
                if (oTable && iPosition > 0) {
                    // Restore scroll position after data is loaded
                    setTimeout(() => {
                        oTable.setFirstVisibleRow(iPosition);
                        this._bScrollPositionRestored = true;
                    }, 100);
                }
            }
        },

        _loadProjects: function () {
            const oViewModel = this.getModel('projectsListView');
            oViewModel.setProperty('/busy', true);
            
            jQuery.ajax({
                url: '/api/projects',
                method: 'GET',
                data: {
                    skip: 0,
                    limit: 100
                },
                success: function(data) {
                    const oModel = new JSONModel({
                        Projects: data.items || [],
                        totalCount: data.totalCount || 0,
                        loadedCount: data.items ? data.items.length : 0
                    });
                    
                    this.getView().setModel(oModel);
                    oViewModel.setProperty('/busy', false);
                    
                    this._iTotalCount = data.totalCount || 0;
                    this._iLoadedCount = data.items ? data.items.length : 0;
                    
                    // Restore scroll position after data is loaded
                    this._restoreScrollPosition();
                }.bind(this),
                error: function(xhr) {
                    oViewModel.setProperty('/busy', false);
                    MessageToast.show('Failed to load projects');
                }.bind(this)
            });
        },

        _setupVirtualScrolling: function () {
            const oTable = this.byId('projectsTable');
            
            // Enable virtual scrolling through table properties
            oTable.setThreshold(50); // Optimized for smooth scrolling
            oTable.setEnableBusyIndicator(true);
            
            // Initialize lazy loading properties
            this._bLoadingMore = false;
            this._iLoadedCount = 0;
            this._iTotalCount = 0;
            this._iScrollDebounceTimer = null;
            
            // Handle scroll events for lazy loading with throttling
            oTable.attachFirstVisibleRowChanged((oEvent) => {
                const iFirstVisibleRow = oEvent.getParameter('firstVisibleRow');
                const iVisibleRowCount = oTable.getVisibleRowCount();
                const iTotalRows = oTable.getBinding('rows').getLength();
                
                // Throttle scroll events for better performance
                clearTimeout(this._iScrollDebounceTimer);
                this._iScrollDebounceTimer = setTimeout(() => {
                    // Check if we need to load more data
                    if ((iFirstVisibleRow + iVisibleRowCount + 30) >= iTotalRows && 
                        !this._bLoadingMore && 
                        this._iLoadedCount < this._iTotalCount) {
                        this._loadMoreProjects();
                    }
                }, 100); // 100ms debounce
                
                // Performance monitoring in development only
                if (window.location.hostname === 'localhost') {
                    console.log('Scroll position:', iFirstVisibleRow, 'of', iTotalRows, 'rows');
                }
            });
            
            // Optimize table rendering for smooth scrolling
            this._optimizeTableRendering(oTable);
        },
        
        _optimizeTableRendering: function (oTable) {
            // Set fixed row heights for consistent scrolling
            oTable.setRowHeight(48);
            
            // Enable row virtualization for large datasets
            if (oTable.setEnableGrouping) {
                oTable.setEnableGrouping(false);
            }
            
            // Optimize column resizing
            oTable.setColumnHeaderHeight(40);
            
            // Add smooth scrolling CSS class
            oTable.addStyleClass('a2a-smooth-scroll');
        },
        
        _loadMoreProjects: function () {
            // Prevent multiple simultaneous loads
            if (this._bLoadingMore || this._iLoadedCount >= this._iTotalCount) {
                return;
            }
            
            this._bLoadingMore = true;
            const oModel = this.getView().getModel();
            const aCurrentProjects = oModel.getProperty('/Projects') || [];
            
            jQuery.ajax({
                url: '/api/projects',
                method: 'GET',
                data: {
                    skip: this._iLoadedCount,
                    limit: 100
                },
                success: function(data) {
                    if (data.items && data.items.length > 0) {
                        // Append new projects to existing array
                        const aUpdatedProjects = aCurrentProjects.concat(data.items);
                        oModel.setProperty('/Projects', aUpdatedProjects);
                        
                        this._iLoadedCount = aUpdatedProjects.length;
                        oModel.setProperty('/loadedCount', this._iLoadedCount);
                    }
                    
                    this._bLoadingMore = false;
                }.bind(this),
                error: function(xhr) {
                    this._bLoadingMore = false;
                    MessageToast.show('Failed to load more projects');
                }.bind(this)
            });
        },

        onSearch: function (oEvent) {
            const sQuery = oEvent.getParameter('query');
            this._applyFilters();
        },

        onSearchLiveChange: function (oEvent) {
            const sQuery = oEvent.getParameter('newValue');
            // Debounced search for performance
            clearTimeout(this._searchDelayTimer);
            this._searchDelayTimer = setTimeout(() => {
                this._applyFilters();
            }, 300);
        },

        _applyFilters: function () {
            // Apply filters with virtual scrolling consideration
            const oTable = this.byId('projectsTable');
            const oBinding = oTable.getBinding('rows');
            
            // Filters would be applied here
            // Virtual scrolling handles large filtered datasets automatically
        },

        onCreateProject: function () {
            MessageToast.show('Create project functionality');
        },

        onImportProject: function () {
            MessageToast.show('Import project functionality');
        },

        onRowSelectionChange: function (oEvent) {
            const oTable = this.byId('projectsTable');
            const aIndices = oTable.getSelectedIndices();
            this.getModel('projectsListView').setProperty('/selectedIndices', aIndices);
        },

        onDeleteSelected: function () {
            const oTable = this.byId('projectsTable');
            const aIndices = oTable.getSelectedIndices();
            MessageToast.show(`${aIndices.length  } projects selected for deletion`);
        },

        onExportToExcel: function () {
            MessageToast.show('Export to Excel functionality');
        },
        
        onExit: function () {
            // Save scroll position before destroying the controller
            this._saveScrollPosition();
            
            // Cleanup router event handler
            const oRouter = this.getRouter();
            if (oRouter) {
                oRouter.detachBeforeRouteMatched(this._onBeforeRouteMatched, this);
            }
            
            // Clear any pending timers
            if (this._iScrollDebounceTimer) {
                clearTimeout(this._iScrollDebounceTimer);
            }
        },
        
        _clearScrollPosition: function () {
            // Method to clear saved scroll position (e.g., after data refresh)
            sessionStorage.removeItem(this._sScrollPositionKey);
            this._bScrollPositionRestored = false;
        },
        
        _updateScrollPositionForNewData: function (iNewDataCount) {
            // Adjust saved scroll position when data changes
            const sSavedPosition = sessionStorage.getItem(this._sScrollPositionKey);
            if (sSavedPosition) {
                const iPosition = parseInt(sSavedPosition, 10);
                // If saved position is beyond new data range, adjust it
                if (iPosition >= iNewDataCount) {
                    const iNewPosition = Math.max(0, iNewDataCount - 15); // Keep some visible rows
                    sessionStorage.setItem(this._sScrollPositionKey, iNewPosition.toString());
                }
            }
        }
    });
});