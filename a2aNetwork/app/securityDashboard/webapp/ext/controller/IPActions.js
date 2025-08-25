/* global sap */
sap.ui.define([
    'sap/m/MessageToast',
    'sap/m/MessageBox',
    'sap/m/Dialog',
    'sap/m/Button',
    'sap/m/Text',
    'sap/m/Input',
    'sap/m/Label',
    'sap/m/VBox',
    'sap/ui/core/ValueState'
], (MessageToast, MessageBox, Dialog, Button, Text, Input, Label, VBox, ValueState) => {
    'use strict';

    /**
     * IP Actions Controller Extension
     * SAP Fiori Elements extension for Blocked IP management
     * Provides action handlers for unblock and extend block operations
     */
    return {
        /**
         * Unblock selected IP addresses
         * Removes IP addresses from blocked list with reason tracking
         */
        onUnblockIP: function (oEvent) {
            const that = this;
            const oModel = this.getModel();
            const aSelectedContexts = this._getSelectedContexts(oEvent);

            if (!aSelectedContexts || aSelectedContexts.length === 0) {
                MessageToast.show('Please select at least one IP address to unblock');
                return;
            }

            // Validate unblock eligibility
            const aValidIPs = aSelectedContexts.filter(oContext => {
                const oData = oContext.getObject();
                return oData.isActive === true;
            });

            if (aValidIPs.length === 0) {
                MessageBox.error('Selected IP addresses are not currently blocked');
                return;
            }

            // Show unblock reason dialog
            this._showUnblockDialog(aValidIPs, oModel);
        },

        /**
         * Extend block duration for selected IP addresses
         * Increases block expiration time for temporary blocks
         */
        onExtendBlock: function (oEvent) {
            const that = this;
            const oModel = this.getModel();
            const aSelectedContexts = this._getSelectedContexts(oEvent);

            if (!aSelectedContexts || aSelectedContexts.length === 0) {
                MessageToast.show('Please select at least one IP address to extend block');
                return;
            }

            // Validate extend eligibility
            const aValidIPs = aSelectedContexts.filter(oContext => {
                const oData = oContext.getObject();
                return oData.isActive === true && oData.blockType !== 'PERMANENT';
            });

            if (aValidIPs.length === 0) {
                MessageBox.error('Selected IP addresses cannot have their block extended (permanent blocks or inactive)');
                return;
            }

            // Show extend block dialog
            this._showExtendBlockDialog(aValidIPs, oModel);
        },

        /**
         * View IP address details and security history
         * Opens detailed view with threat intelligence
         */
        onViewIPDetails: function (oEvent) {
            const oBindingContext = this._getSelectedContexts(oEvent)[0];

            if (!oBindingContext) {
                MessageToast.show('Please select an IP address to view details');
                return;
            }

            const oData = oBindingContext.getObject();
            this._showIPDetailsDialog(oData);
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
                const oTable = this.getView().byId('fe::table::BlockedIPs::LineItem');
                if (oTable && oTable.getSelectedContexts) {
                    aContexts = oTable.getSelectedContexts();
                }
            }

            return aContexts;
        },

        /**
         * Show unblock confirmation dialog with reason input
         */
        _showUnblockDialog: function (aContexts, oModel) {
            const that = this;

            // Create reason input
            const oReasonInput = new Input({
                placeholder: 'Enter reason for unblocking IP address(es)',
                required: true,
                maxLength: 500
            });

            // Create dialog content
            const oDialogContent = new VBox({
                items: [
                    new Text({
                        text: `You are about to unblock ${aContexts.length} IP address(es). This will immediately allow traffic from these addresses.`
                    }).addStyleClass('sapUiMediumMarginBottom'),
                    new Label({
                        text: 'Unblock Reason:',
                        required: true
                    }),
                    oReasonInput
                ]
            });

            // Create dialog
            const oDialog = new Dialog({
                title: 'Unblock IP Addresses',
                type: 'Message',
                state: 'Warning',
                content: oDialogContent,
                beginButton: new Button({
                    text: 'Unblock',
                    type: 'Emphasized',
                    press: function () {
                        const sReason = oReasonInput.getValue().trim();
                        if (!sReason) {
                            oReasonInput.setValueState(ValueState.Error);
                            oReasonInput.setValueStateText('Please provide a reason for unblocking');
                            return;
                        }

                        oDialog.close();
                        that._performBatchUnblock(aContexts, oModel, sReason);
                    }
                }),
                endButton: new Button({
                    text: 'Cancel',
                    press: function () {
                        oDialog.close();
                    }
                }),
                afterClose: function () {
                    oDialog.destroy();
                }
            });

            oDialog.open();
        },

        /**
         * Show extend block dialog with duration input
         */
        _showExtendBlockDialog: function (aContexts, oModel) {
            const that = this;

            // Create duration input
            const oHoursInput = new Input({
                placeholder: '24',
                type: 'Number',
                required: true,
                value: '24'
            });

            // Create dialog content
            const oDialogContent = new VBox({
                items: [
                    new Text({
                        text: `You are about to extend the block duration for ${aContexts.length} IP address(es).`
                    }).addStyleClass('sapUiMediumMarginBottom'),
                    new Label({
                        text: 'Additional Hours:',
                        required: true
                    }),
                    oHoursInput,
                    new Text({
                        text: 'Note: This will extend the existing block duration by the specified hours.'
                    }).addStyleClass('sapUiSmallMarginTop')
                ]
            });

            // Create dialog
            const oDialog = new Dialog({
                title: 'Extend Block Duration',
                type: 'Message',
                state: 'Information',
                content: oDialogContent,
                beginButton: new Button({
                    text: 'Extend Block',
                    type: 'Emphasized',
                    press: function () {
                        const iHours = parseInt(oHoursInput.getValue());
                        if (!iHours || iHours < 1 || iHours > 8760) { // Max 1 year
                            oHoursInput.setValueState(ValueState.Error);
                            oHoursInput.setValueStateText('Please enter a valid number between 1 and 8760 hours');
                            return;
                        }

                        oDialog.close();
                        that._performBatchExtendBlock(aContexts, oModel, iHours);
                    }
                }),
                endButton: new Button({
                    text: 'Cancel',
                    press: function () {
                        oDialog.close();
                    }
                }),
                afterClose: function () {
                    oDialog.destroy();
                }
            });

            oDialog.open();
        },

        /**
         * Show IP details dialog with threat intelligence
         */
        _showIPDetailsDialog: function (oIPData) {
            // Create dialog content with IP information
            const oDialogContent = new VBox({
                items: [
                    new Label({ text: 'IP Address:' }),
                    new Text({ text: oIPData.ipAddress }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Block Type:' }),
                    new Text({ text: oIPData.blockType }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Reason:' }),
                    new Text({ text: oIPData.reason }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Country:' }),
                    new Text({ text: oIPData.country || 'Unknown' }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Organization:' }),
                    new Text({ text: oIPData.organization || 'Unknown' }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Blocked Since:' }),
                    new Text({ text: new Date(oIPData.blockedAt).toLocaleString() }).addStyleClass('sapUiMediumMarginBottom'),

                    new Label({ text: 'Attempt Count:' }),
                    new Text({ text: oIPData.attemptCount.toString() }).addStyleClass('sapUiMediumMarginBottom')
                ]
            });

            // Add expiration info if temporary block
            if (oIPData.expiresAt) {
                oDialogContent.addItem(new Label({ text: 'Expires At:' }));
                oDialogContent.addItem(new Text({ text: new Date(oIPData.expiresAt).toLocaleString() }).addStyleClass('sapUiMediumMarginBottom'));
            }

            // Create dialog
            const oDialog = new Dialog({
                title: 'IP Address Details',
                content: oDialogContent,
                endButton: new Button({
                    text: 'Close',
                    press: function () {
                        oDialog.close();
                    }
                }),
                afterClose: function () {
                    oDialog.destroy();
                }
            });

            oDialog.open();
        },

        /**
         * Perform batch unblock operation
         */
        _performBatchUnblock: function (aContexts, oModel, sReason) {
            // const that = this;
            const aPromises = [];

            aContexts.forEach(oContext => {
                const sPath = oContext.getPath();
                const oData = oContext.getObject();

                // Prepare update data
                const oUpdateData = {
                    isActive: false,
                    unblockReason: sReason,
                    reviewedBy: this._getCurrentUser(),
                    reviewedAt: new Date().toISOString(),
                    modifiedAt: new Date().toISOString(),
                    modifiedBy: this._getCurrentUser()
                };

                // Create promise for OData update
                const oPromise = new Promise((resolve, reject) => {
                    oModel.update(sPath, oUpdateData, {
                        success: function () {
                            resolve(oData.ipAddress);
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
                    MessageToast.show(`${successCount} IP address(es) unblocked successfully`);
                    oModel.refresh();
                }

                if (failCount > 0) {
                    MessageBox.error(`Failed to unblock ${failCount} IP address(es). Please try again.`);
                }
            });
        },

        /**
         * Perform batch extend block operation
         */
        _performBatchExtendBlock: function (aContexts, oModel, iAdditionalHours) {
            // const that = this;
            const aPromises = [];

            aContexts.forEach(oContext => {
                const sPath = oContext.getPath();
                const oData = oContext.getObject();

                // Calculate new expiration time
                const dCurrentExpiration = oData.expiresAt ? new Date(oData.expiresAt) : new Date();
                const dNewExpiration = new Date(dCurrentExpiration.getTime() + (iAdditionalHours * 60 * 60 * 1000));

                // Prepare update data
                const oUpdateData = {
                    expiresAt: dNewExpiration.toISOString(),
                    modifiedAt: new Date().toISOString(),
                    modifiedBy: this._getCurrentUser()
                };

                // Create promise for OData update
                const oPromise = new Promise((resolve, reject) => {
                    oModel.update(sPath, oUpdateData, {
                        success: function () {
                            resolve(oData.ipAddress);
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
                    MessageToast.show(`Block duration extended for ${successCount} IP address(es) by ${iAdditionalHours} hours`);
                    oModel.refresh();
                }

                if (failCount > 0) {
                    MessageBox.error(`Failed to extend block for ${failCount} IP address(es). Please try again.`);
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