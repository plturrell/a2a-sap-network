/* global sap */
sap.ui.define([
    'sap/m/MessageToast',
    'sap/m/MessageBox'
], (MessageToast, MessageBox) => {
    'use strict';

    /**
     * Alert Actions Controller Extension
     * SAP Fiori Elements extension for Security Alert management
     * Provides action handlers for acknowledge, resolve, and escalate operations
     */
    return {
        /**
         * Acknowledge a security alert
         * Updates alert status and adds acknowledgment metadata
         */
        onAcknowledgeAlert: function (oEvent) {
            const that = this;
            const oModel = this.getModel();
            const aSelectedContexts = this._getSelectedContexts(oEvent);

            if (!aSelectedContexts || aSelectedContexts.length === 0) {
                MessageToast.show('Please select at least one alert to acknowledge');
                return;
            }

            // Show confirmation dialog
            MessageBox.confirm(
                'Are you sure you want to acknowledge the selected alert(s)?',
                {
                    title: 'Acknowledge Security Alert',
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            that._performBatchAcknowledge(aSelectedContexts, oModel);
                        }
                    }
                }
            );
        },

        /**
         * Resolve a security alert
         * Updates alert status to resolved with resolution details
         */
        onResolveAlert: function (oEvent) {
            const that = this;
            const oModel = this.getModel();
            const aSelectedContexts = this._getSelectedContexts(oEvent);

            if (!aSelectedContexts || aSelectedContexts.length === 0) {
                MessageToast.show('Please select at least one alert to resolve');
                return;
            }

            // Show resolution dialog with input field
            MessageBox.show(
                'Please provide resolution details for the selected alert(s):',
                {
                    icon: MessageBox.Icon.QUESTION,
                    title: 'Resolve Security Alert',
                    actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
                    emphasizedAction: MessageBox.Action.OK,
                    initialFocus: 'OK',
                    onClose: function (sAction, sValue) {
                        if (sAction === MessageBox.Action.OK) {
                            const sResolution = sValue || 'Alert resolved by security team';
                            that._performBatchResolve(aSelectedContexts, oModel, sResolution);
                        }
                    }
                }
            );
        },

        /**
         * Escalate a security alert
         * Increases alert priority and notifies escalation team
         */
        onEscalateAlert: function (oEvent) {
            const that = this;
            const oModel = this.getModel();
            const aSelectedContexts = this._getSelectedContexts(oEvent);

            if (!aSelectedContexts || aSelectedContexts.length === 0) {
                MessageToast.show('Please select at least one alert to escalate');
                return;
            }

            // Validate escalation eligibility
            const bCanEscalate = aSelectedContexts.every(oContext => {
                const oData = oContext.getObject();
                return oData.priority > 1 && oData.status !== 'RESOLVED';
            });

            if (!bCanEscalate) {
                MessageBox.error('Selected alerts cannot be escalated (already at highest priority or resolved)');
                return;
            }

            // Show escalation reason dialog
            MessageBox.show(
                'Please provide escalation reason:',
                {
                    icon: MessageBox.Icon.WARNING,
                    title: 'Escalate Security Alert',
                    actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
                    emphasizedAction: MessageBox.Action.OK,
                    onClose: function (sAction, sValue) {
                        if (sAction === MessageBox.Action.OK) {
                            const sEscalationReason = sValue || 'Escalated due to severity';
                            that._performBatchEscalate(aSelectedContexts, oModel, sEscalationReason);
                        }
                    }
                }
            );
        },

        /**
         * Get selected contexts from event or table
         */
        _getSelectedContexts: function (oEvent) {
            let aContexts = [];

            // Try to get from event source (button press)
            if (oEvent.getSource) {
                const oSource = oEvent.getSource();
                const oBindingContext = oSource.getBindingContext();
                if (oBindingContext) {
                    aContexts = [oBindingContext];
                }
            }

            // Fallback to table selection
            if (aContexts.length === 0) {
                const oTable = this.getView().byId('fe::table::SecurityAlerts::LineItem');
                if (oTable && oTable.getSelectedContexts) {
                    aContexts = oTable.getSelectedContexts();
                }
            }

            return aContexts;
        },

        /**
         * Perform batch acknowledge operation
         */
        _performBatchAcknowledge: function (aContexts, oModel) {
            const that = this;
            const aPromises = [];

            aContexts.forEach(oContext => {
                const sPath = oContext.getPath();
                const oData = oContext.getObject();

                // Skip if already acknowledged or resolved
                if (oData.status === 'ACKNOWLEDGED' || oData.status === 'RESOLVED') {
                    return;
                }

                // Prepare update data
                const oUpdateData = {
                    status: 'ACKNOWLEDGED',
                    acknowledgedBy: this._getCurrentUser(),
                    acknowledgedAt: new Date().toISOString()
                };

                // Create promise for OData update
                const oPromise = new Promise((resolve, reject) => {
                    oModel.update(sPath, oUpdateData, {
                        success: function () {
                            resolve(oData.title);
                        },
                        error: function (oError) {
                            reject(oError);
                        }
                    });
                });

                aPromises.push(oPromise);
            });

            // Execute batch operation
            Promise.allSettled(aPromises).then(results => {
                const successCount = results.filter(r => r.status === 'fulfilled').length;
                const failCount = results.filter(r => r.status === 'rejected').length;

                if (successCount > 0) {
                    MessageToast.show(`${successCount} alert(s) acknowledged successfully`);
                    oModel.refresh();
                }

                if (failCount > 0) {
                    MessageBox.error(`Failed to acknowledge ${failCount} alert(s). Please try again.`);
                }
            });
        },

        /**
         * Perform batch resolve operation
         */
        _performBatchResolve: function (aContexts, oModel, sResolution) {
            const that = this;
            const aPromises = [];

            aContexts.forEach(oContext => {
                const sPath = oContext.getPath();
                const oData = oContext.getObject();

                // Skip if already resolved
                if (oData.status === 'RESOLVED') {
                    return;
                }

                // Prepare update data
                const oUpdateData = {
                    status: 'RESOLVED',
                    resolvedBy: this._getCurrentUser(),
                    resolvedAt: new Date().toISOString(),
                    resolution: sResolution
                };

                // Auto-acknowledge if not already acknowledged
                if (oData.status === 'ACTIVE') {
                    oUpdateData.acknowledgedBy = this._getCurrentUser();
                    oUpdateData.acknowledgedAt = new Date().toISOString();
                }

                // Create promise for OData update
                const oPromise = new Promise((resolve, reject) => {
                    oModel.update(sPath, oUpdateData, {
                        success: function () {
                            resolve(oData.title);
                        },
                        error: function (oError) {
                            reject(oError);
                        }
                    });
                });

                aPromises.push(oPromise);
            });

            // Execute batch operation
            Promise.allSettled(aPromises).then(results => {
                const successCount = results.filter(r => r.status === 'fulfilled').length;
                const failCount = results.filter(r => r.status === 'rejected').length;

                if (successCount > 0) {
                    MessageToast.show(`${successCount} alert(s) resolved successfully`);
                    oModel.refresh();
                }

                if (failCount > 0) {
                    MessageBox.error(`Failed to resolve ${failCount} alert(s). Please try again.`);
                }
            });
        },

        /**
         * Perform batch escalate operation
         */
        _performBatchEscalate: function (aContexts, oModel, sEscalationReason) {
            const that = this;
            const aPromises = [];

            aContexts.forEach(oContext => {
                const sPath = oContext.getPath();
                const oData = oContext.getObject();

                // Skip if already at highest priority or resolved
                if (oData.priority <= 1 || oData.status === 'RESOLVED') {
                    return;
                }

                // Prepare update data - increase priority (lower number = higher priority)
                const oUpdateData = {
                    priority: Math.max(1, oData.priority - 1),
                    modifiedAt: new Date().toISOString(),
                    modifiedBy: this._getCurrentUser()
                };

                // Add escalation to recommended actions
                let aRecommendedActions = [];
                try {
                    aRecommendedActions = JSON.parse(oData.recommendedActions || '[]');
                } catch (e) {
                    aRecommendedActions = [];
                }

                aRecommendedActions.unshift({
                    action: 'ESCALATED',
                    timestamp: new Date().toISOString(),
                    user: this._getCurrentUser(),
                    reason: sEscalationReason
                });

                oUpdateData.recommendedActions = JSON.stringify(aRecommendedActions);

                // Create promise for OData update
                const oPromise = new Promise((resolve, reject) => {
                    oModel.update(sPath, oUpdateData, {
                        success: function () {
                            resolve(oData.title);
                        },
                        error: function (oError) {
                            reject(oError);
                        }
                    });
                });

                aPromises.push(oPromise);
            });

            // Execute batch operation
            Promise.allSettled(aPromises).then(results => {
                const successCount = results.filter(r => r.status === 'fulfilled').length;
                const failCount = results.filter(r => r.status === 'rejected').length;

                if (successCount > 0) {
                    MessageToast.show(`${successCount} alert(s) escalated successfully`);
                    oModel.refresh();
                }

                if (failCount > 0) {
                    MessageBox.error(`Failed to escalate ${failCount} alert(s). Please try again.`);
                }
            });
        },

        /**
         * Get current user ID
         */
        _getCurrentUser: function () {
            // Try to get from user info service
            try {
                const oUserInfoService = sap.ushell.Container.getService('UserInfo');
                return oUserInfoService.getId() || 'SYSTEM';
            } catch (e) {
                // Fallback for non-Fiori Launchpad environments
                return 'SECURITY_ADMIN';
            }
        }
    };
});