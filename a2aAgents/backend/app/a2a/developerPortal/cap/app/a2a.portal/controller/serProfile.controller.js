sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/Fragment",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (Controller, Fragment, MessageToast, MessageBox) => {
    "use strict";
/* global localStorage, btoa, sessionStorage, atob */

    return Controller.extend("a2a.portal.controller.UserProfile", {

        onInit: function () {
            this.oRouter = this.getOwnerComponent().getRouter();
            this.oModel = this.getOwnerComponent().getModel();
            
            // Initialize user profile model
            this._initializeUserProfile();
            
            // Load user sessions
            this._loadUserSessions();
        },

        _initializeUserProfile: function () {
            // Get user info from XSUAA token
            const oUserInfo = this._getUserInfoFromToken();
            
            if (oUserInfo) {
                const oUserModel = new sap.ui.model.json.JSONModel({
                    userInfo: oUserInfo,
                    sessions: [],
                    statistics: {},
                    preferences: {
                        theme: "sap_fiori_3",
                        language: "en",
                        timezone: "UTC",
                        notifications: {
                            email: true,
                            push: true,
                            deployment: true,
                            security: true
                        }
                    }
                });
                
                this.getView().setModel(oUserModel, "user");
            } else {
                // Redirect to login if no user info
                this._redirectToLogin();
            }
        },

        _getUserInfoFromToken: function () {
            try {
                // In a real SAP BTP environment, this would be handled by the Application Router
                // For now, we'll simulate user info
                const sToken = this._getJWTToken();
                
                if (sToken) {
                    const oDecodedToken = this._decodeJWT(sToken);
                    
                    return {
                        userId: oDecodedToken.user_id || oDecodedToken.sub,
                        userName: oDecodedToken.user_name || oDecodedToken.name,
                        email: oDecodedToken.email,
                        firstName: oDecodedToken.given_name,
                        lastName: oDecodedToken.family_name,
                        roles: oDecodedToken.scope || [],
                        tenant: oDecodedToken.zid,
                        loginTime: new Date(oDecodedToken.iat * 1000),
                        tokenExpiry: new Date(oDecodedToken.exp * 1000)
                    };
                }
            } catch (oError) {
                console.error("Failed to decode JWT token:", oError);
            }
            
            return null;
        },

        _getJWTToken: function () {
            // In SAP BTP, this would be automatically provided by the Application Router
            // For development, we can simulate or get from session storage
            return sessionStorage.getItem("jwt_token") || 
                   localStorage.getItem("jwt_token") ||
                   this._simulateJWTToken();
        },

        _simulateJWTToken: function () {
            // Simulate JWT token for development
            const oPayload = {
                sub: "developer@company.com",
                user_id: "DEV001",
                user_name: "Developer User",
                name: "Developer User",
                given_name: "Developer",
                family_name: "User",
                email: "developer@company.com",
                scope: ["Developer", "ProjectManager", "Admin"],
                zid: "tenant-123",
                iat: Math.floor(Date.now() / 1000),
                exp: Math.floor(Date.now() / 1000) + 3600
            };
            
            // Simple base64 encoding (not secure, just for simulation)
            return `header.${  btoa(JSON.stringify(oPayload))  }.signature`;
        },

        _decodeJWT: function (sToken) {
            try {
                const aParts = sToken.split('.');
                if (aParts.length !== 3) {
                    throw new Error("Invalid JWT token format");
                }
                
                let sPayload = aParts[1];
                // Add padding if needed
                sPayload += '='.repeat((4 - sPayload.length % 4) % 4);
                
                return JSON.parse(atob(sPayload));
            } catch (oError) {
                throw new Error(`Failed to decode JWT token: ${  oError.message}`);
            }
        },

        _loadUserSessions: function () {
            const that = this;
            
            jQuery.ajax({
                url: "/api/auth/sessions",
                method: "GET",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`
                },
                success: function (oData) {
                    const oUserModel = that.getView().getModel("user");
                    oUserModel.setProperty("/sessions", oData.sessions || []);
                    oUserModel.setProperty("/statistics", oData.statistics || {});
                },
                error: function (oError) {
                    console.error("Failed to load user sessions:", oError);
                    MessageToast.show("Failed to load session information");
                }
            });
        },

        onTerminateSession: function (oEvent) {
            const that = this;
            const sSessionId = oEvent.getSource().data("sessionId");
            
            MessageBox.confirm("Are you sure you want to terminate this session?", {
                title: "Terminate Session",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._terminateSession(sSessionId);
                    }
                }
            });
        },

        _terminateSession: function (sSessionId) {
            const that = this;
            
            jQuery.ajax({
                url: `/api/auth/sessions/${  sSessionId}`,
                method: "DELETE",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`
                },
                success: function () {
                    MessageToast.show("Session terminated successfully");
                    that._loadUserSessions(); // Reload sessions
                },
                error: function (oError) {
                    console.error("Failed to terminate session:", oError);
                    MessageToast.show("Failed to terminate session");
                }
            });
        },

        onTerminateAllSessions: function () {
            const that = this;
            
            MessageBox.confirm("Are you sure you want to terminate all other sessions? This will log you out from all other devices.", {
                title: "Terminate All Sessions",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._terminateAllSessions();
                    }
                }
            });
        },

        _terminateAllSessions: function () {
            const that = this;
            
            jQuery.ajax({
                url: "/api/auth/sessions/terminate-all",
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`
                },
                success: function (oData) {
                    MessageToast.show(`Terminated ${  oData.terminated_count  } sessions`);
                    that._loadUserSessions(); // Reload sessions
                },
                error: function (oError) {
                    console.error("Failed to terminate all sessions:", oError);
                    MessageToast.show("Failed to terminate sessions");
                }
            });
        },

        onSavePreferences: function () {
            const _that = this;
            const oUserModel = this.getView().getModel("user");
            const oPreferences = oUserModel.getProperty("/preferences");
            
            jQuery.ajax({
                url: "/api/auth/preferences",
                method: "PUT",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`,
                    "Content-Type": "application/json"
                },
                data: JSON.stringify(oPreferences),
                success: function () {
                    MessageToast.show("Preferences saved successfully");
                },
                error: function (oError) {
                    console.error("Failed to save preferences:", oError);
                    MessageToast.show("Failed to save preferences");
                }
            });
        },

        onChangePassword: function () {
            const that = this;
            
            if (!this._oChangePasswordDialog) {
                Fragment.load({
                    name: "com.sap.a2a.developerportal.view.fragments.ChangePasswordDialog",
                    controller: this
                }).then((oDialog) => {
                    that._oChangePasswordDialog = oDialog;
                    that.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._oChangePasswordDialog.open();
            }
        },

        onChangePasswordConfirm: function () {
            const oDialog = this._oChangePasswordDialog;
            const sCurrentPassword = oDialog.getContent()[0].getItems()[1].getValue();
            const sNewPassword = oDialog.getContent()[0].getItems()[3].getValue();
            const sConfirmPassword = oDialog.getContent()[0].getItems()[5].getValue();
            
            // Validate passwords
            if (!sCurrentPassword || !sNewPassword || !sConfirmPassword) {
                MessageToast.show("Please fill in all password fields");
                return;
            }
            
            if (sNewPassword !== sConfirmPassword) {
                MessageToast.show("New passwords do not match");
                return;
            }
            
            if (sNewPassword.length < 8) {
                MessageToast.show("New password must be at least 8 characters long");
                return;
            }
            
            // Call password change API
            this._changePassword(sCurrentPassword, sNewPassword);
        },

        _changePassword: function (sCurrentPassword, sNewPassword) {
            const that = this;
            
            jQuery.ajax({
                url: "/api/auth/change-password",
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`,
                    "Content-Type": "application/json"
                },
                data: JSON.stringify({
                    current_password: sCurrentPassword,
                    new_password: sNewPassword
                }),
                success: function () {
                    MessageToast.show("Password changed successfully");
                    that._oChangePasswordDialog.close();
                    that._clearPasswordFields();
                },
                error: function (oError) {
                    console.error("Failed to change password:", oError);
                    let sMessage = "Failed to change password";
                    
                    if (oError.responseJSON && oError.responseJSON.detail) {
                        sMessage = oError.responseJSON.detail;
                    }
                    
                    MessageToast.show(sMessage);
                }
            });
        },

        onChangePasswordCancel: function () {
            this._oChangePasswordDialog.close();
            this._clearPasswordFields();
        },

        _clearPasswordFields: function () {
            if (this._oChangePasswordDialog) {
                const aItems = this._oChangePasswordDialog.getContent()[0].getItems();
                aItems[1].setValue(""); // Current password
                aItems[3].setValue(""); // New password
                aItems[5].setValue(""); // Confirm password
            }
        },

        onLogout: function () {
            const that = this;
            
            MessageBox.confirm("Are you sure you want to log out?", {
                title: "Logout",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._logout();
                    }
                }
            });
        },

        _logout: function () {
            const _that = this;
            
            jQuery.ajax({
                url: "/api/auth/logout",
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${  this._getJWTToken()}`
                },
                success: function () {
                    // Clear local storage
                    sessionStorage.removeItem("jwt_token");
                    localStorage.removeItem("jwt_token");
                    
                    // Redirect to logout URL (handled by Application Router in SAP BTP)
                    window.location.href = "/logout";
                },
                error: function (oError) {
                    console.error("Logout error:", oError);
                    // Force logout even if API call fails
                    sessionStorage.removeItem("jwt_token");
                    localStorage.removeItem("jwt_token");
                    window.location.href = "/logout";
                }
            });
        },

        _redirectToLogin: function () {
            // In SAP BTP, this would redirect to the XSUAA login page
            window.location.href = "/login";
        },

        onNavBack: function () {
            this.oRouter.navTo("home");
        },

        formatDate: function (oDate) {
            if (!oDate) {
return "";
}
            
            if (typeof oDate === "string") {
                oDate = new Date(oDate);
            }
            
            return oDate.toLocaleString();
        },

        formatSessionStatus: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Success";
                case "expired":
                    return "Warning";
                case "terminated":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatRoles: function (aRoles) {
            if (!aRoles || !Array.isArray(aRoles)) {
                return "";
            }
            
            return aRoles.join(", ");
        }
    });
});
