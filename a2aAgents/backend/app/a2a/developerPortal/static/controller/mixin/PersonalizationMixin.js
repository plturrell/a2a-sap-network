sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/fl/variants/VariantManagement',
  'sap/ui/fl/apply/api/ControlVariantApplyAPI',
  'sap/ui/fl/write/api/ControlPersonalizationWriteAPI',
  'sap/m/MessageToast'
], (Controller, VariantManagement, ControlVariantApplyAPI, ControlPersonalizationWriteAPI, MessageToast) => {
  'use strict';
  /* global localStorage */

  /**
     * PersonalizationMixin - Adds UI adaptation and personalization capabilities
     * @namespace com.sap.a2a.controller.mixin.PersonalizationMixin
     */
  return {
    /**
         * Initialize personalization for the view
         * @public
         */
    initPersonalization: function () {
      this._oPersonalizationPromise = this._loadPersonalizationData();
      this._registerPersonalizationHandlers();
      this._initializeVariantManagement();
    },

    /**
         * Load user-specific personalization data
         * @private
         * @returns {Promise} Promise resolving with personalization data
         */
    _loadPersonalizationData: function () {
      const sUserId = this.getOwnerComponent().getModel('user').getProperty('/id');
      const sViewId = this.getView().getId();

      return new Promise((resolve, reject) => {
        // Load from backend or local storage
        const sStorageKey = `a2a-personalization-${sUserId}-${sViewId}`;
        const oPersonalizationData = localStorage.getItem(sStorageKey);

        if (oPersonalizationData) {
          try {
            const oParsedData = JSON.parse(oPersonalizationData);
            this._applyPersonalization(oParsedData);
            resolve(oParsedData);
          } catch (e) {
            reject(e);
          }
        } else {
          // Load default personalization
          resolve(this._getDefaultPersonalization());
        }
      });
    },

    /**
         * Get default personalization settings
         * @private
         * @returns {object} Default personalization object
         */
    _getDefaultPersonalization: function () {
      return {
        theme: 'sap_horizon',
        density: 'cozy',
        language: sap.ui.getCore().getConfiguration().getLanguage(),
        tableSettings: {
          visibleColumns: [],
          columnOrder: [],
          sortOrder: [],
          filterSettings: []
        },
        dashboardLayout: {
          tiles: [],
          layout: 'default'
        },
        preferences: {
          notifications: true,
          autoSave: true,
          tooltips: true
        }
      };
    },

    /**
         * Apply personalization settings to the view
         * @private
         * @param {object} oPersonalizationData Personalization data
         */
    _applyPersonalization: function (oPersonalizationData) {
      // Apply theme
      if (oPersonalizationData.theme) {
        sap.ui.getCore().applyTheme(oPersonalizationData.theme);
      }

      // Apply content density
      if (oPersonalizationData.density) {
        this.getView().addStyleClass(`sapUiSize${oPersonalizationData.density}`);
      }

      // Apply table settings
      if (oPersonalizationData.tableSettings) {
        this._applyTablePersonalization(oPersonalizationData.tableSettings);
      }

      // Apply dashboard layout
      if (oPersonalizationData.dashboardLayout) {
        this._applyDashboardPersonalization(oPersonalizationData.dashboardLayout);
      }

      // Store in component model for easy access
      this.getOwnerComponent().getModel('personalization').setData(oPersonalizationData);
    },

    /**
         * Apply table-specific personalization
         * @private
         * @param {object} oTableSettings Table personalization settings
         */
    _applyTablePersonalization: function (oTableSettings) {
      const aTables = this._findControlsByType('sap.m.Table');
            
      aTables.forEach(oTable => {
        const sTableId = oTable.getId();
        const oSettings = oTableSettings[sTableId];

        if (oSettings) {
          // Apply column visibility
          if (oSettings.visibleColumns) {
            oTable.getColumns().forEach((oColumn, iIndex) => {
              oColumn.setVisible(oSettings.visibleColumns.includes(iIndex));
            });
          }

          // Apply sort settings
          if (oSettings.sortOrder && oTable.getBinding('items')) {
            const aSorters = oSettings.sortOrder.map(oSort => 
              new sap.ui.model.Sorter(oSort.path, oSort.descending)
            );
            oTable.getBinding('items').sort(aSorters);
          }
        }
      });
    },

    /**
         * Apply dashboard-specific personalization
         * @private
         * @param {object} oDashboardLayout Dashboard layout settings
         */
    _applyDashboardPersonalization: function (oDashboardLayout) {
      const oDashboard = this.byId('dashboard');
      if (oDashboard && oDashboard.setLayout) {
        oDashboard.setLayout(oDashboardLayout.layout);
                
        // Reorder tiles based on saved preferences
        if (oDashboardLayout.tiles && Array.isArray(oDashboardLayout.tiles)) {
          this._reorderDashboardTiles(oDashboardLayout.tiles);
        }
      }
    },

    /**
         * Initialize variant management for the view
         * @private
         */
    _initializeVariantManagement: function () {
      const oView = this.getView();
            
      // Create variant management control
      this._oVariantManagement = new VariantManagement({
        standardItemText: 'Standard',
        showExecuteOnSelection: true,
        showShare: true,
        showSetAsDefault: true,
        save: this.onSaveVariant.bind(this),
        select: this.onSelectVariant.bind(this)
      });

      // Add to view
      const oPage = oView.byId('page');
      if (oPage && oPage.setCustomHeader) {
        const oBar = oPage.getCustomHeader() || new sap.m.Bar();
        oBar.addContentRight(this._oVariantManagement);
        oPage.setCustomHeader(oBar);
      }

      // Load available variants
      this._loadVariants();
    },

    /**
         * Load available variants for the current view
         * @private
         */
    _loadVariants: function () {
      ControlVariantApplyAPI.loadVariants({
        control: this.getView(),
        standardVariant: {}
      }).then((aVariants) => {
        // Add variants to variant management control
        aVariants.forEach(oVariant => {
          this._oVariantManagement.addVariantItem(new sap.ui.fl.variants.VariantItem({
            key: oVariant.fileName,
            title: oVariant.title,
            favorite: oVariant.favorite,
            executeOnSelect: oVariant.executeOnSelect
          }));
        });
      });
    },

    /**
         * Register personalization event handlers
         * @private
         */
    _registerPersonalizationHandlers: function () {
      // Enable drag and drop for dashboard tiles
      this._enableDashboardDragDrop();

      // Add personalization toolbar
      this._addPersonalizationToolbar();

      // Register keyboard shortcuts
      this._registerPersonalizationShortcuts();
    },

    /**
         * Enable drag and drop functionality for dashboard tiles
         * @private
         */
    _enableDashboardDragDrop: function () {
      const oDashboard = this.byId('dashboard');
      if (oDashboard) {
        oDashboard.addEventDelegate({
          onAfterRendering: function () {
            this._initializeDragDrop(oDashboard);
          }.bind(this)
        });
      }
    },

    /**
         * Initialize drag and drop for a container
         * @private
         * @param {sap.ui.core.Control} oContainer Container control
         */
    _initializeDragDrop: function (oContainer) {
      const aDraggables = oContainer.findAggregatedObjects(true, (oControl) => {
        return oControl.isA('sap.m.GenericTile');
      });

      aDraggables.forEach(oDraggable => {
        oDraggable.addDragDropConfig(new sap.ui.core.dnd.DragInfo({
          sourceAggregation: 'tiles',
          dragStart: this.onDragStart.bind(this)
        }));

        oDraggable.addDragDropConfig(new sap.ui.core.dnd.DropInfo({
          targetAggregation: 'tiles',
          dropPosition: 'Between',
          drop: this.onDrop.bind(this)
        }));
      });
    },

    /**
         * Add personalization toolbar to the view
         * @private
         */
    _addPersonalizationToolbar: function () {
      const oPage = this.byId('page');
      if (!oPage) {
        return;
      }

      const oToolbar = new sap.m.Toolbar({
        visible: false,
        content: [
          new sap.m.Button({
            icon: 'sap-icon://user-settings',
            text: 'Personalize',
            press: this.onOpenPersonalizationDialog.bind(this)
          }),
          new sap.m.Button({
            icon: 'sap-icon://reset',
            text: 'Reset',
            press: this.onResetPersonalization.bind(this)
          }),
          new sap.m.ToolbarSpacer(),
          new sap.m.SegmentedButton({
            selectedKey: '{personalization>/density}',
            items: [
              new sap.m.SegmentedButtonItem({
                key: 'Cozy',
                text: 'Cozy'
              }),
              new sap.m.SegmentedButtonItem({
                key: 'Compact',
                text: 'Compact'
              })
            ],
            selectionChange: this.onDensityChange.bind(this)
          })
        ]
      });

      oPage.setSubHeader(oToolbar);
            
      // Show toolbar in personalization mode
      this._oPersonalizationToolbar = oToolbar;
    },

    /**
         * Register keyboard shortcuts for personalization
         * @private
         */
    _registerPersonalizationShortcuts: function () {
      document.addEventListener('keydown', (oEvent) => {
        // Ctrl/Cmd + Shift + P: Open personalization
        if ((oEvent.ctrlKey || oEvent.metaKey) && oEvent.shiftKey && oEvent.key === 'P') {
          oEvent.preventDefault();
          this.onOpenPersonalizationDialog();
        }
                
        // Ctrl/Cmd + Shift + R: Reset personalization
        if ((oEvent.ctrlKey || oEvent.metaKey) && oEvent.shiftKey && oEvent.key === 'R') {
          oEvent.preventDefault();
          this.onResetPersonalization();
        }
      });
    },

    /**
         * Open personalization dialog
         * @public
         */
    onOpenPersonalizationDialog: function () {
      if (!this._oPersonalizationDialog) {
        this._oPersonalizationDialog = sap.ui.xmlfragment(
          'com.sap.a2a.fragment.PersonalizationDialog',
          this
        );
        this.getView().addDependent(this._oPersonalizationDialog);
      }

      // Load current settings
      const oPersonalizationModel = this.getOwnerComponent().getModel('personalization');
      this._oPersonalizationDialog.setModel(oPersonalizationModel, 'settings');

      this._oPersonalizationDialog.open();
    },

    /**
         * Save personalization settings
         * @public
         */
    onSavePersonalization: function () {
      const oPersonalizationData = this.getOwnerComponent().getModel('personalization').getData();
      const sUserId = this.getOwnerComponent().getModel('user').getProperty('/id');
      const sViewId = this.getView().getId();
      const sStorageKey = `a2a-personalization-${sUserId}-${sViewId}`;

      // Save to local storage
      localStorage.setItem(sStorageKey, JSON.stringify(oPersonalizationData));

      // Save to backend
      this._savePersonalizationToBackend(oPersonalizationData);

      MessageToast.show('Personalization saved successfully');
      this._oPersonalizationDialog.close();
    },

    /**
         * Save personalization to backend
         * @private
         * @param {object} oPersonalizationData Personalization data
         */
    _savePersonalizationToBackend: function (oPersonalizationData) {
      const oModel = this.getOwnerComponent().getModel();
      oModel.create('/UserPersonalizations', {
        userId: this.getOwnerComponent().getModel('user').getProperty('/id'),
        viewId: this.getView().getId(),
        settings: JSON.stringify(oPersonalizationData)
      });
    },

    /**
         * Reset personalization to defaults
         * @public
         */
    onResetPersonalization: function () {
      sap.m.MessageBox.confirm('Are you sure you want to reset all personalizations?', {
        onClose: function (oAction) {
          if (oAction === sap.m.MessageBox.Action.OK) {
            const oDefaultPersonalization = this._getDefaultPersonalization();
            this._applyPersonalization(oDefaultPersonalization);
            this.onSavePersonalization();
          }
        }.bind(this)
      });
    },

    /**
         * Handle density change
         * @public
         * @param {sap.ui.base.Event} oEvent Selection change event
         */
    onDensityChange: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('item').getKey();
      const oView = this.getView();

      // Remove existing density classes
      oView.removeStyleClass('sapUiSizeCozy');
      oView.removeStyleClass('sapUiSizeCompact');

      // Add new density class
      oView.addStyleClass(`sapUiSize${sSelectedKey}`);

      // Update model
      this.getOwnerComponent().getModel('personalization').setProperty('/density', sSelectedKey);
    },

    /**
         * Handle variant save
         * @public
         * @param {sap.ui.base.Event} oEvent Save event
         */
    onSaveVariant: function (oEvent) {
      const sVariantName = oEvent.getParameter('name');
      const _bDefault = oEvent.getParameter('def');
      const _bPublic = oEvent.getParameter('public');
      const bExecuteOnSelect = oEvent.getParameter('execute');

      // Get current personalization state
      const oPersonalizationData = this.getOwnerComponent().getModel('personalization').getData();

      // Save variant
      ControlPersonalizationWriteAPI.save({
        selector: this.getView(),
        changes: [{
          fileName: `variant_${Date.now()}`,
          fileType: 'variant',
          changeType: 'addFavorite',
          layer: 'USER',
          content: {
            title: sVariantName,
            favorite: true,
            visible: true,
            executeOnSelect: bExecuteOnSelect,
            contexts: {},
            content: oPersonalizationData
          }
        }]
      }).then(() => {
        MessageToast.show(`Variant "${sVariantName}" saved successfully`);
        this._loadVariants();
      });
    },

    /**
         * Handle variant selection
         * @public  
         * @param {sap.ui.base.Event} oEvent Selection event
         */
    onSelectVariant: function (oEvent) {
      const sVariantKey = oEvent.getParameter('key');

      if (sVariantKey === '*standard*') {
        this.onResetPersonalization();
      } else {
        // Load and apply variant
        ControlVariantApplyAPI.activateVariant({
          element: this.getView(),
          variantReference: sVariantKey
        }).then(() => {
          MessageToast.show('Variant applied successfully');
        });
      }
    },

    /**
         * Handle drag start
         * @public
         * @param {sap.ui.base.Event} oEvent Drag event
         */
    onDragStart: function (oEvent) {
      const oDraggedControl = oEvent.getParameter('draggedControl');
      const oDragSession = oEvent.getParameter('dragSession');

      oDragSession.setComplexData('draggedControl', {
        id: oDraggedControl.getId(),
        index: oDraggedControl.getParent().indexOfAggregation('tiles', oDraggedControl)
      });
    },

    /**
         * Handle drop
         * @public
         * @param {sap.ui.base.Event} oEvent Drop event
         */
    onDrop: function (oEvent) {
      const _oDragSession = oEvent.getParameter('dragSession');
      const oDraggedControl = oEvent.getParameter('draggedControl');
      const oDroppedControl = oEvent.getParameter('droppedControl');
      const sDropPosition = oEvent.getParameter('dropPosition');

      // Reorder tiles
      const oParent = oDraggedControl.getParent();
      const iDraggedIndex = oParent.indexOfAggregation('tiles', oDraggedControl);
      const iDroppedIndex = oParent.indexOfAggregation('tiles', oDroppedControl);

      oParent.removeAggregation('tiles', oDraggedControl);

      let iNewIndex = iDroppedIndex;
      if (sDropPosition === 'After') {
        iNewIndex = iDroppedIndex + 1;
      }
      if (iDraggedIndex < iDroppedIndex) {
        iNewIndex--;
      }

      oParent.insertAggregation('tiles', oDraggedControl, iNewIndex);

      // Save new order
      this._saveTileOrder();
    },

    /**
         * Save tile order to personalization
         * @private
         */
    _saveTileOrder: function () {
      const oDashboard = this.byId('dashboard');
      if (!oDashboard) {
        return;
      }

      const aTiles = oDashboard.getAggregation('tiles') || [];
      const aTileOrder = aTiles.map(oTile => oTile.getId());

      const oPersonalizationModel = this.getOwnerComponent().getModel('personalization');
      oPersonalizationModel.setProperty('/dashboardLayout/tiles', aTileOrder);

      // Auto-save
      if (oPersonalizationModel.getProperty('/preferences/autoSave')) {
        this.onSavePersonalization();
      }
    },

    /**
         * Find controls by type in the view
         * @private
         * @param {string} sType Control type
         * @returns {array} Array of controls
         */
    _findControlsByType: function (sType) {
      const aControls = [];
            
      function findControls(oControl) {
        if (oControl.isA(sType)) {
          aControls.push(oControl);
        }
                
        const aAggregations = oControl.getMetadata().getAllAggregations();
        for (const sAggregation in aAggregations) {
          const aChildren = oControl.getAggregation(sAggregation);
          if (Array.isArray(aChildren)) {
            aChildren.forEach(findControls);
          } else if (aChildren) {
            findControls(aChildren);
          }
        }
      }

      findControls(this.getView());
      return aControls;
    },

    /**
         * Clean up personalization on exit
         * @public
         */
    cleanupPersonalization: function () {
      if (this._oPersonalizationDialog) {
        this._oPersonalizationDialog.destroy();
        this._oPersonalizationDialog = null;
      }

      if (this._oVariantManagement) {
        this._oVariantManagement.destroy();
        this._oVariantManagement = null;
      }
    }
  };
});