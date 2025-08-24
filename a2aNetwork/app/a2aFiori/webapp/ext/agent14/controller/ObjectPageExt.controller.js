sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent14/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeSecurity();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize dialog cache
                this._dialogCache = {};
                
                // Initialize real-time monitoring
                this._initializeRealtimeMonitoring();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onRestoreData
         * @description Restores data from backup with comprehensive options.
         * @public
         */
        onRestoreData: function() {
            if (!this._hasRole("BackupAdmin")) {
                MessageBox.error("Access denied. Backup Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "RestoreData", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if backup is in valid state for restore
            if (oData.status !== "COMPLETED") {
                MessageBox.warning(this.getResourceBundle().getText("msg.backupNotCompleted"));
                return;
            }
            
            this._auditLogger.log("RESTORE_DATA", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("restoreData", "a2a.network.agent14.ext.fragment.RestoreData")
                .then(function(oDialog) {
                    var oRestoreModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        backupId: oData.backupId,
                        restoreType: "FULL",
                        targetLocation: "ORIGINAL",
                        customLocation: "",
                        restoreOptions: {
                            verifyBeforeRestore: true,
                            createBackupBeforeRestore: true,
                            preservePermissions: true,
                            overwriteExisting: true,
                            skipCorruptedFiles: false,
                            continueOnError: false
                        },
                        restoreFilters: {
                            includePatterns: [],
                            excludePatterns: [],
                            fileTypeFilter: "ALL",
                            dateRangeFilter: false,
                            sizeFilter: false
                        },
                        estimatedSize: oData.dataSize || 0,
                        estimatedDuration: oData.estimatedRestoreDuration || 0,
                        availableVersions: [],
                        selectedVersion: oData.latestVersion || "latest"
                    });
                    oDialog.setModel(oRestoreModel, "restore");
                    oDialog.open();
                    this._loadRestoreOptions(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Restore Data dialog: " + error.message);
                });
        },

        /**
         * @function onVerifyBackup
         * @description Verifies backup integrity and consistency.
         * @public
         */
        onVerifyBackup: function() {
            if (!this._hasRole("BackupUser")) {
                MessageBox.error("Access denied. Backup User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "VerifyBackup", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("VERIFY_BACKUP", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("verifyBackup", "a2a.network.agent14.ext.fragment.VerifyBackup")
                .then(function(oDialog) {
                    var oVerifyModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        backupId: oData.backupId,
                        verificationType: "COMPREHENSIVE",
                        verificationChecks: {
                            checksumValidation: true,
                            fileStructureValidation: true,
                            metadataValidation: true,
                            encryptionValidation: true,
                            compressionValidation: true,
                            crossReferenceValidation: true
                        },
                        verificationScope: "ALL",
                        sampleSize: 100,
                        detailedLogging: true,
                        repairMode: false,
                        verificationResults: {
                            status: "NOT_STARTED",
                            progress: 0,
                            filesChecked: 0,
                            totalFiles: oData.fileCount || 0,
                            errorsFound: 0,
                            warningsFound: 0,
                            details: []
                        }
                    });
                    oDialog.setModel(oVerifyModel, "verify");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Verify Backup dialog: " + error.message);
                });
        },

        /**
         * @function onDownloadBackup
         * @description Downloads backup with customizable options.
         * @public
         */
        onDownloadBackup: function() {
            if (!this._hasRole("BackupUser")) {
                MessageBox.error("Access denied. Backup User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "DownloadBackup", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if backup is downloadable
            if (oData.status !== "COMPLETED") {
                MessageBox.warning(this.getResourceBundle().getText("msg.backupNotAvailable"));
                return;
            }
            
            this._auditLogger.log("DOWNLOAD_BACKUP", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("downloadBackup", "a2a.network.agent14.ext.fragment.DownloadBackup")
                .then(function(oDialog) {
                    var oDownloadModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        backupId: oData.backupId,
                        downloadFormat: "ARCHIVE",
                        compressionLevel: "STANDARD",
                        encryptDownload: true,
                        includeMetadata: true,
                        downloadOptions: {
                            splitArchive: false,
                            maxArchiveSize: 2048,
                            generateManifest: true,
                            includeChecksum: true,
                            preserveStructure: true
                        },
                        availableFormats: [
                            { key: "ARCHIVE", text: "Compressed Archive (.tar.gz)" },
                            { key: "ZIP", text: "ZIP Archive (.zip)" },
                            { key: "RAW", text: "Raw Files (uncompressed)" },
                            { key: "IMAGE", text: "Disk Image (.img)" }
                        ],
                        estimatedSize: oData.compressedSize || oData.dataSize || 0,
                        estimatedDuration: this._calculateDownloadDuration(oData.dataSize),
                        downloadUrl: "",
                        downloadStatus: "READY"
                    });
                    oDialog.setModel(oDownloadModel, "download");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Download Backup dialog: " + error.message);
                });
        },

        /**
         * @function onDeleteBackup
         * @description Deletes backup with confirmation and audit trail.
         * @public
         */
        onDeleteBackup: function() {
            if (!this._hasRole("BackupAdmin")) {
                MessageBox.error("Access denied. Backup Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "DeleteBackup", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("DELETE_BACKUP_REQUESTED", { taskId: sTaskId, taskName: sTaskName });
            
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.deleteBackupConfirm", [sTaskName]),
                {
                    title: this.getResourceBundle().getText("title.confirmDeletion"),
                    emphasizedAction: MessageBox.Action.CANCEL,
                    initialFocus: MessageBox.Action.CANCEL,
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeBackupDeletion(sTaskId, sTaskName, oData);
                        } else {
                            this._auditLogger.log("DELETE_BACKUP_CANCELLED", { taskId: sTaskId, taskName: sTaskName });
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function _loadRestoreOptions
         * @description Loads restore configuration options.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadRestoreOptions: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetRestoreOptions", {
                urlParameters: {
                    taskId: sTaskId,
                    includeVersions: true
                },
                success: function(data) {
                    var oRestoreModel = oDialog.getModel("restore");
                    if (oRestoreModel) {
                        var oCurrentData = oRestoreModel.getData();
                        oCurrentData.availableVersions = data.versions || [];
                        oCurrentData.targetLocations = data.targetLocations || [];
                        oCurrentData.compatibilityInfo = data.compatibility || {};
                        oCurrentData.dependencyInfo = data.dependencies || {};
                        oRestoreModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load restore options: " + error.message);
                }
            });
        },

        /**
         * @function _calculateDownloadDuration
         * @description Calculates estimated download duration based on size.
         * @param {number} dataSize - Data size in bytes
         * @returns {number} Estimated duration in seconds
         * @private
         */
        _calculateDownloadDuration: function(dataSize) {
            // Assume average download speed of 10 MB/s
            const avgSpeed = 10 * 1024 * 1024; // 10 MB/s in bytes
            return Math.ceil(dataSize / avgSpeed);
        },

        /**
         * @function _executeBackupDeletion
         * @description Executes backup deletion with audit logging.
         * @param {string} sTaskId - Task ID
         * @param {string} sTaskName - Task name
         * @param {Object} oData - Backup data
         * @private
         */
        _executeBackupDeletion: function(sTaskId, sTaskName, oData) {
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/DeleteBackup", {
                urlParameters: {
                    taskId: sTaskId,
                    backupId: oData.backupId,
                    force: false,
                    createAuditLog: true
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.backupDeleted"));
                    this._auditLogger.log("BACKUP_DELETED", { 
                        taskId: sTaskId, 
                        taskName: sTaskName,
                        backupId: oData.backupId,
                        success: true 
                    });
                    
                    // Navigate back or refresh
                    this.base.getExtensionAPI().getRouting().navigateToTarget("BackupList");
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.backupDeletionFailed"));
                    this._auditLogger.log("BACKUP_DELETION_FAILED", { 
                        taskId: sTaskId, 
                        taskName: sTaskName,
                        error: error.message 
                    });
                }.bind(this)
            });
        },

        /**
         * @function _initializeRealtimeMonitoring
         * @description Initializes real-time monitoring for backup operations.
         * @private
         */
        _initializeRealtimeMonitoring: function() {
            // WebSocket for real-time backup updates
            this._initializeBackupWebSocket();
        },

        /**
         * @function _initializeBackupWebSocket
         * @description Initializes WebSocket for backup updates.
         * @private
         */
        _initializeBackupWebSocket: function() {
            if (this._backupWs) return;

            try {
                this._backupWs = SecurityUtils.createSecureWebSocket('ws://localhost:8014/backup/task-updates', {
                    onMessage: function(data) {
                        this._handleBackupTaskUpdate(data);
                    }.bind(this)
                });

                this._backupWs.onclose = function() {
                    console.info("Backup WebSocket closed, will reconnect...");
                    setTimeout(() => this._initializeBackupWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("Backup WebSocket not available");
            }
        },

        /**
         * @function _handleBackupTaskUpdate
         * @description Handles real-time backup task updates.
         * @param {Object} data - Update data
         * @private
         */
        _handleBackupTaskUpdate: function(data) {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;
            
            const oCurrentData = oContext.getObject();
            
            // Check if update is for current task
            if (data.taskId === oCurrentData.taskId) {
                switch (data.type) {
                    case 'STATUS_UPDATE':
                        // Refresh the binding to get latest status
                        oContext.refresh();
                        break;
                    case 'VERIFICATION_COMPLETED':
                        MessageToast.show("Backup verification completed");
                        this._updateVerificationResults(data);
                        break;
                    case 'RESTORE_PROGRESS':
                        this._updateRestoreProgress(data);
                        break;
                    case 'DOWNLOAD_READY':
                        MessageToast.show("Backup download is ready");
                        this._updateDownloadStatus(data);
                        break;
                }
            }
        },

        /**
         * @function _updateVerificationResults
         * @description Updates verification results in open dialog.
         * @param {Object} data - Verification data
         * @private
         */
        _updateVerificationResults: function(data) {
            if (this._dialogCache.verifyBackup && this._dialogCache.verifyBackup.isOpen()) {
                const oVerifyModel = this._dialogCache.verifyBackup.getModel("verify");
                if (oVerifyModel) {
                    const oCurrentData = oVerifyModel.getData();
                    oCurrentData.verificationResults = data.results;
                    oVerifyModel.setData(oCurrentData);
                }
            }
        },

        /**
         * @function _updateRestoreProgress
         * @description Updates restore progress in open dialog.
         * @param {Object} data - Progress data
         * @private
         */
        _updateRestoreProgress: function(data) {
            if (this._dialogCache.restoreData && this._dialogCache.restoreData.isOpen()) {
                const oRestoreModel = this._dialogCache.restoreData.getModel("restore");
                if (oRestoreModel) {
                    const oCurrentData = oRestoreModel.getData();
                    oCurrentData.restoreProgress = data.progress;
                    oCurrentData.restoreStatus = data.status;
                    oRestoreModel.setData(oCurrentData);
                }
            }
        },

        /**
         * @function _updateDownloadStatus
         * @description Updates download status in open dialog.
         * @param {Object} data - Download data
         * @private
         */
        _updateDownloadStatus: function(data) {
            if (this._dialogCache.downloadBackup && this._dialogCache.downloadBackup.isOpen()) {
                const oDownloadModel = this._dialogCache.downloadBackup.getModel("download");
                if (oDownloadModel) {
                    const oCurrentData = oDownloadModel.getData();
                    oCurrentData.downloadUrl = data.downloadUrl;
                    oCurrentData.downloadStatus = data.status;
                    oDownloadModel.setData(oCurrentData);
                }
            }
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache && this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            
            if (!this._dialogCache) {
                this._dialogCache = {};
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then(function(oDialog) {
                that._dialogCache[sDialogId] = oDialog;
                that.base.getView().addDependent(oDialog);
                
                // Enable accessibility
                that._enableDialogAccessibility(oDialog);
                
                // Optimize for mobile
                that._optimizeDialogForDevice(oDialog);
                
                return oDialog;
            });
        },
        
        /**
         * @function _enableDialogAccessibility
         * @description Adds accessibility features to dialog.
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @private
         */
        _enableDialogAccessibility: function(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering: function() {
                    var $dialog = oDialog.$();
                    
                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");
                    
                    // Handle escape key
                    $dialog.on("keydown", function(e) {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });
                    
                    // Focus first input on open
                    setTimeout(function() {
                        $dialog.find("input:visible:first").focus();
                    }, 100);
                }
            });
        },
        
        /**
         * @function _optimizeDialogForDevice
         * @description Optimizes dialog for current device.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         * @private
         */
        _optimizeDialogForDevice: function(oDialog) {
            if (sap.ui.Device.system.phone) {
                oDialog.setStretch(true);
                oDialog.setContentWidth("100%");
                oDialog.setContentHeight("100%");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("95%");
                oDialog.setContentHeight("90%");
            }
        },

        /**
         * @function _initializeSecurity
         * @description Initializes security features and audit logging.
         * @private
         */
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    var user = this._getCurrentUser();
                    var timestamp = new Date().toISOString();
                    var logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent14_Backup",
                        action: action,
                        details: details || {}
                    };
                    console.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user ID for audit logging.
         * @returns {string} User ID or "anonymous"
         * @private
         */
        _getCurrentUser: function() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        /**
         * @function _hasRole
         * @description Checks if current user has specified role.
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole: function(role) {
            const user = sap.ushell?.Container?.getUser();
            if (user && user.hasRole) {
                return user.hasRole(role);
            }
            // Mock role validation for development/testing
            const mockRoles = ["BackupAdmin", "BackupUser", "BackupOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up WebSocket connections
            if (this._backupWs) {
                this._backupWs.close();
                this._backupWs = null;
            }
            
            // Clean up cached dialogs
            if (this._dialogCache) {
                Object.keys(this._dialogCache).forEach(function(key) {
                    if (this._dialogCache[key]) {
                        this._dialogCache[key].destroy();
                    }
                }.bind(this));
                this._dialogCache = {};
            }
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @public
         */
        getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});