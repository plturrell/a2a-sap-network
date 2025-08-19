sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, JSONModel, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("a2a.portal.controller.ProjectDetail", {

        onInit: function () {
            // Get router
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("projectDetail").attachPatternMatched(this._onPatternMatched, this);
            
            // Initialize terminal
            this._initializeTerminal();
        },

        _onPatternMatched: function (oEvent) {
            var sProjectId = oEvent.getParameter("arguments").projectId;
            this._loadProjectDetails(sProjectId);
        },

        _loadProjectDetails: function (sProjectId) {
            var oView = this.getView();
            oView.setBusy(true);
            this._sCurrentProjectId = sProjectId;

            jQuery.ajax({
                url: "/api/projects/" + sProjectId,
                method: "GET",
                success: function (data) {
                    oView.setModel(new JSONModel(data));
                    
                    // Initialize file explorer model
                    this._initializeFileExplorer();
                    
                    // Load project files
                    this._loadProjectFiles(sProjectId);
                    
                    oView.setBusy(false);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to load project details: " + error);
                    oView.setBusy(false);
                }.bind(this)
            });
        },
        
        _initializeFileExplorer: function () {
            var oFileModel = new JSONModel({
                files: [],
                selectedFiles: [],
                expandedFolders: {},
                clipboardFiles: [],
                clipboardOperation: null // 'copy' or 'cut'
            });
            this.getView().setModel(oFileModel, "files");
        },
        
        _loadProjectFiles: function (sProjectId) {
            jQuery.ajax({
                url: "/api/projects/" + sProjectId + "/files",
                method: "GET",
                success: function (data) {
                    var oFileModel = this.getView().getModel("files");
                    oFileModel.setProperty("/files", this._buildFileTree(data));
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to load project files: " + error);
                }.bind(this)
            });
        },
        
        _buildFileTree: function (aFiles) {
            // Convert flat file list to hierarchical tree structure
            var oFileMap = {};
            var aRootFiles = [];
            
            // First pass: create file objects and map by path
            aFiles.forEach(function (oFile) {
                oFileMap[oFile.path] = {
                    path: oFile.path,
                    name: oFile.name,
                    type: oFile.type, // 'file' or 'folder'
                    size: oFile.size,
                    lastModified: oFile.lastModified,
                    extension: oFile.extension,
                    children: [],
                    expanded: false
                };
            });
            
            // Second pass: build hierarchy
            aFiles.forEach(function (oFile) {
                var sParentPath = oFile.path.substring(0, oFile.path.lastIndexOf('/'));
                
                if (sParentPath === '' || !oFileMap[sParentPath]) {
                    // Root level file
                    aRootFiles.push(oFileMap[oFile.path]);
                } else {
                    // Add to parent folder
                    oFileMap[sParentPath].children.push(oFileMap[oFile.path]);
                }
            });
            
            return aRootFiles;
        },

        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("projects");
        },

        onEditProject: function () {
            MessageToast.show("Edit project functionality coming soon");
        },

        onDeployProject: function () {
            var sProjectId = this._sCurrentProjectId;
            var oView = this.getView();
            
            // Show deployment configuration dialog
            if (!this._oDeployConfigDialog) {
                this._oDeployConfigDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.DeployConfigDialog",
                    this
                );
                this.getView().addDependent(this._oDeployConfigDialog);
            }
            
            // Set deployment configuration model
            var oDeployConfig = new JSONModel({
                targetEnvironment: "staging",
                deploymentStrategy: "rolling",
                replicas: 2,
                healthCheckPath: "/health",
                enableAutoScaling: true,
                minInstances: 1,
                maxInstances: 5,
                cpuThreshold: 70,
                memoryLimit: "512Mi",
                environmentVariables: [],
                preDeploymentScript: "",
                postDeploymentScript: "",
                rollbackOnFailure: true
            });
            
            this._oDeployConfigDialog.setModel(oDeployConfig, "deployConfig");
            this._oDeployConfigDialog.open();
        },
        
        onConfirmDeployProject: function () {
            var oDeployConfig = this._oDeployConfigDialog.getModel("deployConfig").getData();
            var sProjectId = this._sCurrentProjectId;
            var oView = this.getView();
            
            this._oDeployConfigDialog.close();
            
            // Show deployment progress dialog
            if (!this._oDeployProgressDialog) {
                this._oDeployProgressDialog = new sap.m.Dialog({
                    title: "Deploying Project",
                    type: "Message",
                    content: [
                        new sap.m.VBox({
                            items: [
                                new sap.m.Text({ 
                                    text: "Deploying to " + oDeployConfig.targetEnvironment + " environment..." 
                                }),
                                new sap.m.ProgressIndicator({
                                    id: "deployProgressIndicator",
                                    width: "100%",
                                    showValue: true,
                                    state: "Information"
                                }),
                                new sap.m.Text({
                                    id: "deployStatusText",
                                    text: "Preparing deployment package..."
                                }),
                                new sap.m.VBox({
                                    id: "deploymentStepsBox",
                                    items: []
                                })
                            ]
                        })
                    ],
                    beginButton: new sap.m.Button({
                        text: "View Logs",
                        press: function () {
                            this._viewDeploymentLogs();
                        }.bind(this)
                    }),
                    endButton: new sap.m.Button({
                        text: "Close",
                        enabled: false,
                        id: "deployCloseButton",
                        press: function () {
                            this._oDeployProgressDialog.close();
                        }.bind(this)
                    })
                });
                
                oView.addDependent(this._oDeployProgressDialog);
            }
            
            this._oDeployProgressDialog.open();
            
            // Start deployment
            jQuery.ajax({
                url: "/api/projects/" + sProjectId + "/deploy",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oDeployConfig),
                success: function (data) {
                    this._deploymentId = data.deploymentId;
                    this._updateDeploymentProgress(data.deploymentId);
                }.bind(this),
                error: function (xhr, status, error) {
                    this._oDeployProgressDialog.close();
                    MessageToast.show("Failed to start deployment: " + error);
                }.bind(this)
            });
        },
        
        _updateDeploymentProgress: function (sDeploymentId) {
            var oProgressIndicator = sap.ui.getCore().byId("deployProgressIndicator");
            var oStatusText = sap.ui.getCore().byId("deployStatusText");
            var oStepsBox = sap.ui.getCore().byId("deploymentStepsBox");
            var oCloseButton = sap.ui.getCore().byId("deployCloseButton");
            
            var fnCheckProgress = function () {
                jQuery.ajax({
                    url: "/api/deployments/" + sDeploymentId + "/status",
                    method: "GET",
                    success: function (data) {
                        oProgressIndicator.setPercentValue(data.progress);
                        oStatusText.setText(data.currentStep);
                        
                        // Update deployment steps
                        oStepsBox.removeAllItems();
                        data.steps.forEach(function (step) {
                            var oStepItem = new sap.m.HBox({
                                items: [
                                    new sap.ui.core.Icon({
                                        src: step.status === "completed" ? "sap-icon://accept" : 
                                             step.status === "failed" ? "sap-icon://error" : 
                                             step.status === "running" ? "sap-icon://synchronize" : 
                                             "sap-icon://pending",
                                        color: step.status === "completed" ? "Positive" : 
                                               step.status === "failed" ? "Negative" : 
                                               step.status === "running" ? "Critical" : 
                                               "Neutral",
                                        size: "1rem"
                                    }).addStyleClass("sapUiTinyMarginEnd"),
                                    new sap.m.Text({ text: step.name })
                                ]
                            }).addStyleClass("sapUiTinyMarginBottom");
                            oStepsBox.addItem(oStepItem);
                        });
                        
                        if (data.status === "completed") {
                            oProgressIndicator.setState("Success");
                            oStatusText.setText("Deployment completed successfully!");
                            oCloseButton.setEnabled(true);
                            MessageToast.show("Project deployed successfully to " + data.environment);
                            
                            // Update project status
                            this.getView().getModel().setProperty("/deployment/lastDeployment", new Date().toISOString());
                            this.getView().getModel().setProperty("/deployment/environment", data.environment);
                            
                            // Open deployment URL if available
                            if (data.deploymentUrl) {
                                sap.m.MessageBox.confirm(
                                    "Deployment successful! Would you like to open the deployed application?",
                                    {
                                        title: "Deployment Complete",
                                        actions: [sap.m.MessageBox.Action.YES, sap.m.MessageBox.Action.NO],
                                        onClose: function (oAction) {
                                            if (oAction === sap.m.MessageBox.Action.YES) {
                                                window.open(data.deploymentUrl, "_blank");
                                            }
                                        }
                                    }
                                );
                            }
                        } else if (data.status === "failed") {
                            oProgressIndicator.setState("Error");
                            oStatusText.setText("Deployment failed: " + data.error);
                            oCloseButton.setEnabled(true);
                            
                            // Handle rollback if enabled
                            if (data.rollbackInitiated) {
                                MessageToast.show("Rollback initiated due to deployment failure");
                            }
                        } else {
                            // Continue checking
                            setTimeout(fnCheckProgress, 2000);
                        }
                    }.bind(this),
                    error: function () {
                        oProgressIndicator.setState("Error");
                        oStatusText.setText("Failed to get deployment status");
                        oCloseButton.setEnabled(true);
                    }
                });
            }.bind(this);
            
            fnCheckProgress();
        },
        
        onCancelDeployConfig: function () {
            this._oDeployConfigDialog.close();
        },
        
        _viewDeploymentLogs: function () {
            if (this._deploymentId) {
                window.open("/api/deployments/" + this._deploymentId + "/logs", "_blank");
            }
        },
        
        // File Explorer Event Handlers
        
        onToggleFolder: function (oEvent) {
            var oTreeItem = oEvent.getParameter("item");
            var oContext = oTreeItem.getBindingContext();
            var oFileData = oContext.getObject();
            
            if (oFileData.type === "folder") {
                var bExpanded = oEvent.getParameter("expanded");
                oContext.getModel().setProperty(oContext.getPath() + "/expanded", bExpanded);
                
                // Load folder contents if expanding
                if (bExpanded && (!oFileData.children || oFileData.children.length === 0)) {
                    this._loadFolderContents(oFileData.path);
                }
            }
        },
        
        onFilePress: function (oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext();
            var oFileData = oContext.getObject();
            
            if (oFileData.type === "file") {
                this._openFile(oFileData);
            }
        },
        
        onTreeItemPress: function (oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext();
            var oFileData = oContext.getObject();
            
            // Handle single file selection for actions
            var oFileModel = this.getView().getModel("files");
            oFileModel.setProperty("/selectedFiles", [oFileData]);
        },
        
        onFileSelectionChange: function (oEvent) {
            var aSelectedItems = oEvent.getParameter("selectedItems");
            var aSelectedFiles = aSelectedItems.map(function (oItem) {
                return oItem.getBindingContext().getObject();
            });
            
            var oFileModel = this.getView().getModel("files");
            oFileModel.setProperty("/selectedFiles", aSelectedFiles);
        },
        
        onCreateFile: function () {
            if (!this._oCreateFileDialog) {
                this._oCreateFileDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.CreateFileDialog",
                    this
                );
                this.getView().addDependent(this._oCreateFileDialog);
            }
            
            // Reset dialog
            var oCreateModel = new JSONModel({
                fileName: "",
                fileType: "txt",
                parentPath: "/",
                fileContent: ""
            });
            this._oCreateFileDialog.setModel(oCreateModel, "createFile");
            this._oCreateFileDialog.open();
        },
        
        onCreateFolder: function () {
            if (!this._oCreateFolderDialog) {
                this._oCreateFolderDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.CreateFolderDialog",
                    this
                );
                this.getView().addDependent(this._oCreateFolderDialog);
            }
            
            var oCreateModel = new JSONModel({
                folderName: "",
                parentPath: "/"
            });
            this._oCreateFolderDialog.setModel(oCreateModel, "createFolder");
            this._oCreateFolderDialog.open();
        },
        
        onConfirmCreateFile: function () {
            var oCreateModel = this._oCreateFileDialog.getModel("createFile");
            var sFileName = oCreateModel.getProperty("/fileName");
            var sFileType = oCreateModel.getProperty("/fileType");
            var sParentPath = oCreateModel.getProperty("/parentPath");
            var sFileContent = oCreateModel.getProperty("/fileContent");
            
            if (!sFileName.trim()) {
                MessageToast.show("Please enter a file name");
                return;
            }
            
            var sFullPath = sParentPath + (sParentPath.endsWith("/") ? "" : "/") + sFileName;
            
            this._createFile(sFullPath, sFileType, sFileContent);
            this._oCreateFileDialog.close();
        },
        
        onConfirmCreateFolder: function () {
            var oCreateModel = this._oCreateFolderDialog.getModel("createFolder");
            var sFolderName = oCreateModel.getProperty("/folderName");
            var sParentPath = oCreateModel.getProperty("/parentPath");
            
            if (!sFolderName.trim()) {
                MessageToast.show("Please enter a folder name");
                return;
            }
            
            var sFullPath = sParentPath + (sParentPath.endsWith("/") ? "" : "/") + sFolderName;
            
            this._createFolder(sFullPath);
            this._oCreateFolderDialog.close();
        },
        
        onCancelCreateFile: function () {
            this._oCreateFileDialog.close();
        },
        
        onCancelCreateFolder: function () {
            this._oCreateFolderDialog.close();
        },
        
        _createFile: function (sFilePath, sFileType, sContent) {
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    path: sFilePath,
                    type: "file",
                    extension: sFileType,
                    content: sContent || ""
                }),
                success: function () {
                    MessageToast.show("File created successfully");
                    this._loadProjectFiles(this._sCurrentProjectId);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to create file: " + error);
                }.bind(this)
            });
        },
        
        _createFolder: function (sFolderPath) {
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    path: sFolderPath,
                    type: "folder"
                }),
                success: function () {
                    MessageToast.show("Folder created successfully");
                    this._loadProjectFiles(this._sCurrentProjectId);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to create folder: " + error);
                }.bind(this)
            });
        },
        
        onUploadFiles: function () {
            // Create file input dynamically
            var oFileUpload = document.createElement("input");
            oFileUpload.type = "file";
            oFileUpload.multiple = true;
            oFileUpload.style.display = "none";
            
            oFileUpload.onchange = function (oEvent) {
                var aFiles = oEvent.target.files;
                if (aFiles.length > 0) {
                    this._handleFileUpload(aFiles);
                }
                document.body.removeChild(oFileUpload);
            }.bind(this);
            
            document.body.appendChild(oFileUpload);
            oFileUpload.click();
        },
        
        _handleFileUpload: function (aFiles) {
            var oFormData = new FormData();
            
            for (var i = 0; i < aFiles.length; i++) {
                oFormData.append("files", aFiles[i]);
            }
            
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files/upload",
                method: "POST",
                data: oFormData,
                processData: false,
                contentType: false,
                success: function (data) {
                    MessageToast.show("Files uploaded successfully: " + data.uploadedCount + " files");
                    this._loadProjectFiles(this._sCurrentProjectId);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("File upload failed: " + error);
                }.bind(this)
            });
        },
        
        onRefreshFileTree: function () {
            this._loadProjectFiles(this._sCurrentProjectId);
            MessageToast.show("File tree refreshed");
        },
        
        onOpenSelectedFile: function () {
            var oFileModel = this.getView().getModel("files");
            var aSelectedFiles = oFileModel.getProperty("/selectedFiles");
            
            if (aSelectedFiles.length === 1) {
                this._openFile(aSelectedFiles[0]);
            }
        },
        
        _openFile: function (oFileData) {
            if (oFileData.type === "file") {
                // For demo purposes, show file content in a dialog
                this._showFileContentDialog(oFileData);
            }
        },
        
        _showFileContentDialog: function (oFileData) {
            // Load file content and show in dialog
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files/" + encodeURIComponent(oFileData.path),
                method: "GET",
                success: function (sContent) {
                    if (!this._oFileContentDialog) {
                        this._oFileContentDialog = sap.ui.xmlfragment(
                            "a2a.portal.fragment.FileContentDialog",
                            this
                        );
                        this.getView().addDependent(this._oFileContentDialog);
                    }
                    
                    var oContentModel = new JSONModel({
                        fileName: oFileData.name,
                        filePath: oFileData.path,
                        fileContent: sContent,
                        isEditable: this._isEditableFile(oFileData.extension)
                    });
                    
                    this._oFileContentDialog.setModel(oContentModel, "fileContent");
                    this._oFileContentDialog.open();
                }.bind(this),
                error: function () {
                    MessageToast.show("Failed to load file content");
                }.bind(this)
            });
        },
        
        _isEditableFile: function (sExtension) {
            var aEditableExtensions = ["txt", "js", "json", "xml", "html", "css", "md", "yaml", "yml"];
            return aEditableExtensions.includes(sExtension.toLowerCase());
        },
        
        onRenameFile: function () {
            var oFileModel = this.getView().getModel("files");
            var aSelectedFiles = oFileModel.getProperty("/selectedFiles");
            
            if (aSelectedFiles.length === 1) {
                this._showRenameDialog(aSelectedFiles[0]);
            }
        },
        
        _showRenameDialog: function (oFileData) {
            sap.m.MessageBox.prompt("Rename " + oFileData.type, {
                initialText: oFileData.name,
                onClose: function (oAction, sValue) {
                    if (oAction === sap.m.MessageBox.Action.OK && sValue.trim()) {
                        this._renameFile(oFileData.path, sValue.trim());
                    }
                }.bind(this)
            });
        },
        
        _renameFile: function (sOldPath, sNewName) {
            var sParentPath = sOldPath.substring(0, sOldPath.lastIndexOf('/'));
            var sNewPath = sParentPath + (sParentPath ? "/" : "") + sNewName;
            
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files/" + encodeURIComponent(sOldPath) + "/rename",
                method: "PUT",
                contentType: "application/json",
                data: JSON.stringify({ newPath: sNewPath }),
                success: function () {
                    MessageToast.show("File renamed successfully");
                    this._loadProjectFiles(this._sCurrentProjectId);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to rename file: " + error);
                }.bind(this)
            });
        },
        
        onCopyFiles: function () {
            var oFileModel = this.getView().getModel("files");
            var aSelectedFiles = oFileModel.getProperty("/selectedFiles");
            
            if (aSelectedFiles.length > 0) {
                oFileModel.setProperty("/clipboardFiles", aSelectedFiles.slice());
                oFileModel.setProperty("/clipboardOperation", "copy");
                MessageToast.show(aSelectedFiles.length + " file(s) copied to clipboard");
            }
        },
        
        onMoveFiles: function () {
            var oFileModel = this.getView().getModel("files");
            var aSelectedFiles = oFileModel.getProperty("/selectedFiles");
            
            if (aSelectedFiles.length > 0) {
                oFileModel.setProperty("/clipboardFiles", aSelectedFiles.slice());
                oFileModel.setProperty("/clipboardOperation", "cut");
                MessageToast.show(aSelectedFiles.length + " file(s) moved to clipboard");
            }
        },
        
        onDeleteFiles: function () {
            var oFileModel = this.getView().getModel("files");
            var aSelectedFiles = oFileModel.getProperty("/selectedFiles");
            
            if (aSelectedFiles.length === 0) {
                return;
            }
            
            var sMessage = "Are you sure you want to delete " + aSelectedFiles.length + " selected file(s)?\n\nThis action cannot be undone.";
            
            sap.m.MessageBox.confirm(sMessage, {
                title: "Delete Files",
                actions: [sap.m.MessageBox.Action.DELETE, sap.m.MessageBox.Action.CANCEL],
                emphasizedAction: sap.m.MessageBox.Action.DELETE,
                onClose: function (oAction) {
                    if (oAction === sap.m.MessageBox.Action.DELETE) {
                        this._deleteSelectedFiles(aSelectedFiles);
                    }
                }.bind(this)
            });
        },
        
        _deleteSelectedFiles: function (aFiles) {
            var aFilePaths = aFiles.map(function (oFile) {
                return oFile.path;
            });
            
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files/delete",
                method: "DELETE",
                contentType: "application/json",
                data: JSON.stringify({ filePaths: aFilePaths }),
                success: function (data) {
                    MessageToast.show("Successfully deleted " + data.deletedCount + " file(s)");
                    this._loadProjectFiles(this._sCurrentProjectId);
                    
                    // Clear selection
                    var oFileModel = this.getView().getModel("files");
                    oFileModel.setProperty("/selectedFiles", []);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to delete files: " + error);
                }.bind(this)
            });
        },
        
        _loadFolderContents: function (sFolderPath) {
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/files",
                method: "GET",
                data: { path: sFolderPath },
                success: function (data) {
                    // Update the folder's children in the file tree
                    this._updateFolderContents(sFolderPath, data);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to load folder contents: " + error);
                }.bind(this)
            });
        },
        
        _updateFolderContents: function (sFolderPath, aFiles) {
            var oFileModel = this.getView().getModel("files");
            var aFileTree = oFileModel.getProperty("/files");
            
            // Find and update the folder in the tree
            var oFolder = this._findFileInTree(aFileTree, sFolderPath);
            if (oFolder) {
                oFolder.children = this._buildFileTree(aFiles);
                oFileModel.refresh();
            }
        },
        
        _findFileInTree: function (aFiles, sPath) {
            for (var i = 0; i < aFiles.length; i++) {
                if (aFiles[i].path === sPath) {
                    return aFiles[i];
                }
                if (aFiles[i].children && aFiles[i].children.length > 0) {
                    var oFound = this._findFileInTree(aFiles[i].children, sPath);
                    if (oFound) {
                        return oFound;
                    }
                }
            }
            return null;
        },
        
        // Formatter functions for the view
        getFileIcon: function (sType, sExtension) {
            if (sType === "folder") {
                return "sap-icon://folder";
            }
            
            switch (sExtension) {
                case "js": return "sap-icon://syntax";
                case "json": return "sap-icon://document-text";
                case "xml": return "sap-icon://document";
                case "html": return "sap-icon://web";
                case "css": return "sap-icon://palette";
                case "txt": return "sap-icon://text";
                case "md": return "sap-icon://notes";
                case "png":
                case "jpg":
                case "jpeg":
                case "gif": return "sap-icon://image-viewer";
                default: return "sap-icon://document";
            }
        },
        
        getFileItemType: function (sType) {
            return sType === "folder" ? "Navigation" : "Active";
        },
        
        formatDate: function (sDate) {
            if (!sDate) return "";
            var oDateFormat = DateFormat.getDateTimeInstance({
                pattern: "yyyy-MM-dd HH:mm:ss"
            });
            return oDateFormat.format(new Date(sDate));
        },

        onMoreActions: function (oEvent) {
            MessageToast.show("More actions menu coming soon");
        },

        onTabSelect: function (oEvent) {
            var sKey = oEvent.getParameter("key");
            console.log("Tab selected:", sKey);
        },

        onCreateAgent: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sProjectId = this.getView().getModel().getProperty("/project_id");
            oRouter.navTo("agentBuilder", {
                projectId: sProjectId
            });
        },

        onEditAgent: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sAgentId = oContext.getProperty("id");
            MessageToast.show("Edit agent: " + sAgentId);
        },

        onDeleteAgent: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sAgentName = oContext.getProperty("name");
            MessageToast.show("Delete agent: " + sAgentName);
        },

        onCreateWorkflow: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sProjectId = this.getView().getModel().getProperty("/project_id");
            oRouter.navTo("bpmnDesigner", {
                projectId: sProjectId
            });
        },

        onEditWorkflow: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sWorkflowId = oContext.getProperty("id");
            MessageToast.show("Edit workflow: " + sWorkflowId);
        },

        onRunWorkflow: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sWorkflowName = oContext.getProperty("name");
            MessageToast.show("Run workflow: " + sWorkflowName);
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) {
                return "";
            }
            
            var oDateFormat = DateFormat.getDateTimeInstance({
                style: "medium"
            });
            
            return oDateFormat.format(new Date(sDate));
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active": return "Success";
                case "inactive": return "Warning";
                case "error": return "Error";
                default: return "None";
            }
        },
        
        // Project Operations
        
        onSaveProject: function () {
            var oView = this.getView();
            oView.setBusy(true);
            
            var oProjectData = oView.getModel().getData();
            var sProjectId = this._sCurrentProjectId;
            
            // Prepare save data
            var oSaveData = {
                name: oProjectData.name,
                description: oProjectData.description,
                status: oProjectData.status,
                deployment: oProjectData.deployment,
                last_modified: new Date().toISOString()
            };
            
            jQuery.ajax({
                url: "/api/projects/" + sProjectId,
                method: "PUT",
                contentType: "application/json",
                data: JSON.stringify(oSaveData),
                success: function (data) {
                    MessageToast.show("Project saved successfully");
                    
                    // Update model with saved data
                    oView.getModel().setData(data);
                    
                    // Trigger save event for other components
                    var oEventBus = sap.ui.getCore().getEventBus();
                    oEventBus.publish("project", "saved", {
                        projectId: sProjectId,
                        projectData: data
                    });
                    
                    oView.setBusy(false);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to save project: " + error);
                    oView.setBusy(false);
                }.bind(this)
            });
        },
        
        onBuildProject: function () {
            var sProjectId = this._sCurrentProjectId;
            var oView = this.getView();
            
            // Show build progress dialog
            if (!this._oBuildProgressDialog) {
                this._oBuildProgressDialog = new sap.m.Dialog({
                    title: "Building Project",
                    type: "Message",
                    content: [
                        new sap.m.VBox({
                            items: [
                                new sap.m.Text({ text: "Building project components..." }),
                                new sap.m.ProgressIndicator({
                                    id: "buildProgressIndicator",
                                    width: "100%",
                                    showValue: true,
                                    state: "Information"
                                }),
                                new sap.m.Text({
                                    id: "buildStatusText",
                                    text: "Initializing build process..."
                                })
                            ]
                        })
                    ],
                    beginButton: new sap.m.Button({
                        text: "Run in Background",
                        press: function () {
                            this._oBuildProgressDialog.close();
                            MessageToast.show("Build continues in background");
                        }.bind(this)
                    }),
                    endButton: new sap.m.Button({
                        text: "Cancel",
                        press: function () {
                            this._cancelBuild();
                            this._oBuildProgressDialog.close();
                        }.bind(this)
                    })
                });
                
                oView.addDependent(this._oBuildProgressDialog);
            }
            
            this._oBuildProgressDialog.open();
            
            // Start build process
            jQuery.ajax({
                url: "/api/projects/" + sProjectId + "/build",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    configuration: "production",
                    clean: true,
                    optimize: true
                }),
                success: function (data) {
                    // Update progress
                    this._updateBuildProgress(data.buildId);
                }.bind(this),
                error: function (xhr, status, error) {
                    this._oBuildProgressDialog.close();
                    MessageToast.show("Failed to start build: " + error);
                }.bind(this)
            });
        },
        
        _updateBuildProgress: function (sBuildId) {
            var oProgressIndicator = sap.ui.getCore().byId("buildProgressIndicator");
            var oStatusText = sap.ui.getCore().byId("buildStatusText");
            
            var fnCheckProgress = function () {
                jQuery.ajax({
                    url: "/api/builds/" + sBuildId + "/status",
                    method: "GET",
                    success: function (data) {
                        oProgressIndicator.setPercentValue(data.progress);
                        oStatusText.setText(data.message);
                        
                        if (data.status === "completed") {
                            oProgressIndicator.setState("Success");
                            MessageToast.show("Build completed successfully");
                            setTimeout(function () {
                                this._oBuildProgressDialog.close();
                            }.bind(this), 2000);
                        } else if (data.status === "failed") {
                            oProgressIndicator.setState("Error");
                            oStatusText.setText("Build failed: " + data.error);
                        } else {
                            // Continue checking
                            setTimeout(fnCheckProgress, 1000);
                        }
                    }.bind(this),
                    error: function () {
                        oProgressIndicator.setState("Error");
                        oStatusText.setText("Failed to get build status");
                    }
                });
            }.bind(this);
            
            fnCheckProgress();
        },
        
        _cancelBuild: function () {
            // Cancel build operation
            if (this._sBuildId) {
                jQuery.ajax({
                    url: "/api/builds/" + this._sBuildId + "/cancel",
                    method: "POST",
                    success: function () {
                        MessageToast.show("Build cancelled");
                    },
                    error: function () {
                        MessageToast.show("Failed to cancel build");
                    }
                });
            }
        },
        
        onRunProject: function () {
            var sProjectId = this._sCurrentProjectId;
            var oProjectData = this.getView().getModel().getData();
            
            // Show run configuration dialog
            if (!this._oRunConfigDialog) {
                this._oRunConfigDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.RunConfigDialog",
                    this
                );
                this.getView().addDependent(this._oRunConfigDialog);
            }
            
            // Set run configuration model
            var oRunConfig = new JSONModel({
                environment: "development",
                port: 3000,
                debugMode: true,
                watchFiles: true,
                envVariables: []
            });
            
            this._oRunConfigDialog.setModel(oRunConfig, "runConfig");
            this._oRunConfigDialog.open();
        },
        
        onConfirmRunProject: function () {
            var oRunConfig = this._oRunConfigDialog.getModel("runConfig").getData();
            var sProjectId = this._sCurrentProjectId;
            
            this._oRunConfigDialog.close();
            
            // Start project
            jQuery.ajax({
                url: "/api/projects/" + sProjectId + "/run",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oRunConfig),
                success: function (data) {
                    MessageToast.show("Project started successfully");
                    
                    // Open terminal or output panel
                    if (data.terminalUrl) {
                        window.open(data.terminalUrl, "_blank");
                    }
                    
                    // Update project status
                    this.getView().getModel().setProperty("/status", "running");
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to run project: " + error);
                }.bind(this)
            });
        },
        
        onCancelRunProject: function () {
            this._oRunConfigDialog.close();
        },
        
        // Terminal Integration
        
        _initializeTerminal: function () {
            var oTerminalModel = new JSONModel({
                sessions: [],
                activeSessionId: null,
                currentCommand: "",
                commandHistory: [],
                historyIndex: -1,
                settings: {
                    fontSize: 14,
                    fontFamily: "Consolas, Monaco, monospace",
                    theme: "dark",
                    scrollback: 1000
                }
            });
            this.getView().setModel(oTerminalModel, "terminal");
            
            // Create initial terminal session
            this._createTerminalSession();
        },
        
        _createTerminalSession: function () {
            var oTerminalModel = this.getView().getModel("terminal");
            var aSessions = oTerminalModel.getProperty("/sessions");
            var sSessionId = "terminal_" + Date.now();
            
            var oNewSession = {
                id: sSessionId,
                name: "Terminal " + (aSessions.length + 1),
                output: [],
                active: true,
                cwd: "/projects/" + this._sCurrentProjectId,
                websocket: null
            };
            
            aSessions.push(oNewSession);
            oTerminalModel.setProperty("/sessions", aSessions);
            oTerminalModel.setProperty("/activeSessionId", sSessionId);
            
            // Connect WebSocket for terminal
            this._connectTerminalWebSocket(sSessionId);
            
            // Add initial welcome message
            this._appendTerminalOutput(sSessionId, "Connected to project terminal...", "system");
            this._appendTerminalOutput(sSessionId, "Working directory: " + oNewSession.cwd, "system");
        },
        
        _connectTerminalWebSocket: function (sSessionId) {
            var sProjectId = this._sCurrentProjectId;
            var wsUrl = window.location.protocol.replace("http", "ws") + "//" + 
                       window.location.host + "/api/projects/" + sProjectId + "/terminal";
            
            var oWebSocket = new WebSocket(wsUrl);
            
            oWebSocket.onopen = function () {
                this._updateSessionWebSocket(sSessionId, oWebSocket);
                this._appendTerminalOutput(sSessionId, "Terminal connected", "success");
            }.bind(this);
            
            oWebSocket.onmessage = function (event) {
                var data = JSON.parse(event.data);
                if (data.type === "output") {
                    this._appendTerminalOutput(sSessionId, data.content, data.stream || "stdout");
                } else if (data.type === "exit") {
                    this._appendTerminalOutput(sSessionId, "Process exited with code " + data.code, "system");
                }
            }.bind(this);
            
            oWebSocket.onerror = function (error) {
                this._appendTerminalOutput(sSessionId, "Terminal connection error", "error");
            }.bind(this);
            
            oWebSocket.onclose = function () {
                this._appendTerminalOutput(sSessionId, "Terminal disconnected", "system");
                this._updateSessionWebSocket(sSessionId, null);
            }.bind(this);
        },
        
        _updateSessionWebSocket: function (sSessionId, oWebSocket) {
            var oTerminalModel = this.getView().getModel("terminal");
            var aSessions = oTerminalModel.getProperty("/sessions");
            var oSession = aSessions.find(function (s) { return s.id === sSessionId; });
            if (oSession) {
                oSession.websocket = oWebSocket;
            }
        },
        
        _appendTerminalOutput: function (sSessionId, sContent, sType) {
            var oTerminalModel = this.getView().getModel("terminal");
            var aSessions = oTerminalModel.getProperty("/sessions");
            var oSession = aSessions.find(function (s) { return s.id === sSessionId; });
            
            if (oSession) {
                var oOutput = {
                    content: sContent,
                    type: sType || "stdout",
                    timestamp: new Date().toISOString()
                };
                oSession.output.push(oOutput);
                
                // Limit scrollback
                var iMaxLines = oTerminalModel.getProperty("/settings/scrollback");
                if (oSession.output.length > iMaxLines) {
                    oSession.output = oSession.output.slice(-iMaxLines);
                }
                
                oTerminalModel.refresh();
                
                // Auto-scroll to bottom
                setTimeout(function () {
                    var oContainer = this.byId("terminalContainer");
                    if (oContainer) {
                        oContainer.scrollToElement(oContainer.getContent()[0], 0);
                    }
                }.bind(this), 100);
            }
        },
        
        onNewTerminal: function () {
            this._createTerminalSession();
            MessageToast.show("New terminal session created");
        },
        
        onTerminalCommandSubmit: function (oEvent) {
            var oTerminalModel = this.getView().getModel("terminal");
            var sCommand = oTerminalModel.getProperty("/currentCommand");
            var sActiveSessionId = oTerminalModel.getProperty("/activeSessionId");
            
            if (!sCommand.trim()) {
                return;
            }
            
            // Add to command history
            var aHistory = oTerminalModel.getProperty("/commandHistory");
            aHistory.push(sCommand);
            oTerminalModel.setProperty("/commandHistory", aHistory);
            oTerminalModel.setProperty("/historyIndex", aHistory.length);
            
            // Display command in terminal
            this._appendTerminalOutput(sActiveSessionId, "$ " + sCommand, "command");
            
            // Send command via WebSocket
            var aSessions = oTerminalModel.getProperty("/sessions");
            var oSession = aSessions.find(function (s) { return s.id === sActiveSessionId; });
            
            if (oSession && oSession.websocket && oSession.websocket.readyState === WebSocket.OPEN) {
                oSession.websocket.send(JSON.stringify({
                    type: "command",
                    command: sCommand
                }));
            } else {
                // Fallback to REST API
                this._executeTerminalCommand(sActiveSessionId, sCommand);
            }
            
            // Clear input
            oTerminalModel.setProperty("/currentCommand", "");
        },
        
        _executeTerminalCommand: function (sSessionId, sCommand) {
            jQuery.ajax({
                url: "/api/projects/" + this._sCurrentProjectId + "/terminal/execute",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    sessionId: sSessionId,
                    command: sCommand
                }),
                success: function (data) {
                    if (data.output) {
                        this._appendTerminalOutput(sSessionId, data.output, "stdout");
                    }
                    if (data.error) {
                        this._appendTerminalOutput(sSessionId, data.error, "stderr");
                    }
                }.bind(this),
                error: function (xhr, status, error) {
                    this._appendTerminalOutput(sSessionId, "Error executing command: " + error, "error");
                }.bind(this)
            });
        },
        
        onClearTerminal: function () {
            var oTerminalModel = this.getView().getModel("terminal");
            var sActiveSessionId = oTerminalModel.getProperty("/activeSessionId");
            var aSessions = oTerminalModel.getProperty("/sessions");
            
            var oSession = aSessions.find(function (s) { return s.id === sActiveSessionId; });
            if (oSession) {
                oSession.output = [];
                oTerminalModel.refresh();
                this._appendTerminalOutput(sActiveSessionId, "Terminal cleared", "system");
            }
        },
        
        onExportTerminalOutput: function () {
            var oTerminalModel = this.getView().getModel("terminal");
            var sActiveSessionId = oTerminalModel.getProperty("/activeSessionId");
            var aSessions = oTerminalModel.getProperty("/sessions");
            
            var oSession = aSessions.find(function (s) { return s.id === sActiveSessionId; });
            if (oSession) {
                var sOutput = oSession.output.map(function (o) {
                    return "[" + new Date(o.timestamp).toLocaleTimeString() + "] " + o.content;
                }).join("\n");
                
                var blob = new Blob([sOutput], { type: "text/plain" });
                var url = URL.createObjectURL(blob);
                var a = document.createElement("a");
                a.href = url;
                a.download = "terminal_output_" + new Date().toISOString() + ".txt";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                MessageToast.show("Terminal output exported");
            }
        },
        
        onTerminalSettings: function () {
            if (!this._oTerminalSettingsDialog) {
                this._oTerminalSettingsDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.TerminalSettingsDialog",
                    this
                );
                this.getView().addDependent(this._oTerminalSettingsDialog);
            }
            
            this._oTerminalSettingsDialog.open();
        },
        
        onSaveTerminalSettings: function () {
            this._oTerminalSettingsDialog.close();
            MessageToast.show("Terminal settings saved");
            
            // Apply settings to terminal display
            this._applyTerminalSettings();
        },
        
        onCancelTerminalSettings: function () {
            this._oTerminalSettingsDialog.close();
        },
        
        _applyTerminalSettings: function () {
            var oTerminalModel = this.getView().getModel("terminal");
            var oSettings = oTerminalModel.getProperty("/settings");
            
            // Apply settings via CSS custom properties or direct styling
            var oTerminalContent = this.byId("terminalContent");
            if (oTerminalContent) {
                oTerminalContent.$().css({
                    "font-size": oSettings.fontSize + "px",
                    "font-family": oSettings.fontFamily
                });
            }
        }
    });
});