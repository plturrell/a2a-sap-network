sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], function(Controller, MessageToast, MessageBox, JSONModel) {
    "use strict";

    /**
     * Base Controller for A2A Network Application
     *
     * Provides common functionality and enterprise patterns for all controllers in the A2A Network application.
     * This controller implements SAP standard patterns for loading states, error handling, navigation,
     * and resource bundle management.
     *
     * @namespace a2a.network.fiori.controller
     * @class
     * @extends sap.ui.core.mvc.Controller
     * @public
     * @author SAP SE
     * @since 1.0.0
     * @version 1.0.0
     *
     * @example
     * // Extending BaseController in your controller
     * sap.ui.define([
     *     "./BaseController"
     * ], function(BaseController) {
     *     "use strict";
     *     return BaseController.extend("a2a.network.fiori.controller.MyController", {
     *         onInit: function() {
     *             BaseController.prototype.onInit.apply(this, arguments);
     *             // Your initialization code here
     *         }
     *     });
     * });
     */

    return Controller.extend("a2a.network.fiori.controller.BaseController", {

        /**
         * Called when a controller is instantiated and its View controls have been created.
         * Initializes the UI model for loading states and common controller functionality.
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @since 1.0.0
         */
        onInit() {
            // Initialize UI state model for loading states
            this.oUIModel = new JSONModel({
                // Loading states
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                loadingMessage: "",
                loadingSubMessage: "",
                progressTitle: "",
                progressValue: 0,
                progressText: "",
                progressState: "None",
                progressDescription: "",
                blockchainStep: "",

                // Error states
                hasError: false,
                errorMessage: "",
                errorTitle: "",

                // Data states
                hasNoData: false,
                noDataMessage: "",
                noDataIcon: "sap-icon://product",

                // UI states
                busy: false,
                editable: false,
                hasChanges: false,
                showDetails: false,
                selectedItems: [],

                // Security states
                csrfToken: null,
                sessionId: null,
                correlationId: null,
                securityInitialized: false
            });

            this.getView().setModel(this.oUIModel, "ui");

            // Initialize security
            this._initializeSecurity();
        },

        /**
         * Gets the resource bundle for internationalization
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @returns {sap.base.i18n.ResourceBundle} The resource bundle
         * @since 1.0.0
         */
        getResourceBundle() {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },

        /**
         * Gets the router instance
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @returns {sap.ui.core.routing.Router} The router instance
         * @since 1.0.0
         */
        getRouter() {
            return this.getOwnerComponent().getRouter();
        },

        /**
         * Gets the model by name, or the default model if no name is provided
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sModelName] The name of the model
         * @returns {sap.ui.model.Model} The model instance
         * @since 1.0.0
         */
        getModel(sModelName) {
            return this.getView().getModel(sModelName);
        },

        /**
         * Sets a model on the view
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {sap.ui.model.Model} oModel The model to set
         * @param {string} [sModelName] The name of the model
         * @since 1.0.0
         */
        setModel(oModel, sModelName) {
            this.getView().setModel(oModel, sModelName);
        },

        /**
         * Shows skeleton loading state for lists and tables
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sMessage] Loading message to display
         * @since 1.0.0
         */
        showSkeletonLoading(sMessage) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: true,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                loadingMessage: sMessage || this.getResourceBundle().getText("common.loading")
            });
        },

        /**
         * Shows spinner loading state for actions
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sMessage] Loading message to display
         * @param {string} [sSubMessage] Additional loading context
         * @since 1.0.0
         */
        showSpinnerLoading(sMessage, sSubMessage) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: true,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                loadingMessage: sMessage || this.getResourceBundle().getText("common.processing"),
                loadingSubMessage: sSubMessage || ""
            });
        },

        /**
         * Shows progress loading state for multi-step operations
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {object} oOptions Progress configuration options
         * @param {string} [oOptions.title] Progress title
         * @param {string} [oOptions.message] Progress message
         * @param {number} [oOptions.value] Progress percentage (0-100)
         * @param {string} [oOptions.state] Progress state (None|Success|Warning|Error)
         * @since 1.0.0
         */
        showProgressLoading(oOptions = {}) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: true,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: false,
                progressTitle: oOptions.title || this.getResourceBundle().getText("common.processing"),
                loadingMessage: oOptions.message || "",
                progressValue: oOptions.value || 0,
                progressText: `${oOptions.value || 0}%`,
                progressState: oOptions.state || "None",
                progressDescription: oOptions.description || ""
            });
        },

        /**
         * Shows blockchain-specific loading state
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sStep] Current blockchain operation step
         * @since 1.0.0
         */
        showBlockchainLoading(sStep) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: true,
                hasError: false,
                hasNoData: false,
                blockchainStep: sStep || this.getResourceBundle().getText("blockchain.processing")
            });
        },

        /**
         * Hides all loading states
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @since 1.0.0
         */
        hideLoading() {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false
            });
        },

        /**
         * Shows error state with message
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} sMessage Error message
         * @param {string} [sTitle] Error title
         * @since 1.0.0
         */
        showError(sMessage, sTitle) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: true,
                hasNoData: false,
                errorMessage: sMessage,
                errorTitle: sTitle || this.getResourceBundle().getText("common.error")
            });
        },

        /**
         * Shows no data state
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sMessage] No data message
         * @param {string} [sIcon] No data icon
         * @since 1.0.0
         */
        showNoData(sMessage, sIcon) {
            this.oUIModel.setData({
                ...this.oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                hasError: false,
                hasNoData: true,
                noDataMessage: sMessage || this.getResourceBundle().getText("common.noDataAvailable"),
                noDataIcon: sIcon || "sap-icon://product"
            });
        },

        /**
         * Navigates back in browser history or to a specific route
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} [sDefaultRoute] Default route if no history
         * @since 1.0.0
         */
        onNavBack(sDefaultRoute) {
            const sPreviousHash = this.getRouter().getHashChanger().getPreviousHash();
            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getRouter().navTo(sDefaultRoute || "home", {}, true);
            }
        },

        /**
         * Shows a message toast
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} sMessage Message to display
         * @param {object} [oOptions] Toast options
         * @since 1.0.0
         */
        showMessageToast(sMessage, oOptions = {}) {
            MessageToast.show(sMessage, {
                duration: oOptions.duration || 3000,
                at: oOptions.at || MessageToast.BOTTOM_CENTER,
                ...oOptions
            });
        },

        /**
         * Shows a message box
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @public
         * @param {string} sMessage Message to display
         * @param {object} [oOptions] MessageBox options
         * @since 1.0.0
         */
        showMessageBox(sMessage, oOptions = {}) {
            MessageBox.show(sMessage, {
                icon: oOptions.icon || MessageBox.Icon.INFORMATION,
                title: oOptions.title || this.getResourceBundle().getText("common.information"),
                actions: oOptions.actions || [MessageBox.Action.OK],
                onClose: oOptions.onClose,
                ...oOptions
            });
        },

        /* =========================================================== */
        /* Security Methods                                            */
        /* =========================================================== */

        /**
         * Initializes security features including CSRF token retrieval
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @since 1.0.0
         */
        _initializeSecurity() {
            // Generate correlation ID for request tracking
            this.oUIModel.setProperty("/correlationId", this._generateCorrelationId());

            // Fetch CSRF token for secure operations
            this._fetchCSRFToken();
        },

        /**
         * Generates a unique correlation ID for request tracking
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @returns {string} Unique correlation ID
         * @since 1.0.0
         */
        _generateCorrelationId() {
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c === "x" ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        },

        /**
         * Fetches CSRF token from the server
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @since 1.0.0
         */
        async _fetchCSRFToken() {
            try {
                const response = await fetch("/api/v1/csrf-token", {
                    method: "GET",
                    credentials: "same-origin",
                    headers: {
                        "X-Correlation-ID": this.oUIModel.getProperty("/correlationId")
                    }
                });

                if (response.ok) {
                    const data = await response.json();
                    this.oUIModel.setProperty("/csrfToken", data.csrfToken);
                    this.oUIModel.setProperty("/sessionId", data.sessionId);
                    this.oUIModel.setProperty("/securityInitialized", true);
                }
            } catch (error) {
                console.warn("Failed to fetch CSRF token:", error);
                // Continue without CSRF in development
                this.oUIModel.setProperty("/securityInitialized", true);
            }
        },

        /**
         * Makes a secure AJAX request with CSRF protection and input sanitization
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @protected
         * @param {string} sUrl Request URL
         * @param {object} [oOptions] Request options
         * @param {string} [oOptions.method='GET'] HTTP method
         * @param {object} [oOptions.data] Request data
         * @param {object} [oOptions.headers] Additional headers
         * @param {boolean} [oOptions.sanitize=true] Whether to sanitize input
         * @returns {Promise} Fetch promise
         * @since 1.0.0
         */
        async secureRequest(sUrl, oOptions = {}) {
            const {
                method = "GET",
                data = null,
                headers = {},
                sanitize = true
            } = oOptions;

            // Prepare headers
            const requestHeaders = {
                "Content-Type": "application/json",
                "X-Correlation-ID": this.oUIModel.getProperty("/correlationId"),
                ...headers
            };

            // Add CSRF token for write operations
            if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
                const csrfToken = this.oUIModel.getProperty("/csrfToken");
                if (csrfToken) {
                    requestHeaders["X-CSRF-Token"] = csrfToken;
                }
            }

            // Sanitize request data
            let sanitizedData = data;
            if (sanitize && data) {
                sanitizedData = this._sanitizeRequestData(data);
            }

            const requestOptions = {
                method: method.toUpperCase(),
                credentials: "same-origin",
                headers: requestHeaders
            };

            if (sanitizedData && method.toUpperCase() !== "GET") {
                requestOptions.body = JSON.stringify(sanitizedData);
            }

            try {
                const response = await fetch(sUrl, requestOptions);
                return await this._handleSecureResponse(response);
            } catch (error) {
                return this._handleRequestError(error, sUrl, method);
            }
        },

        /**
         * Sanitizes request data to prevent XSS and injection attacks
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @param {*} data Data to sanitize
         * @returns {*} Sanitized data
         * @since 1.0.0
         */
        _sanitizeRequestData(data) {
            if (typeof data === "string") {
                return this._sanitizeString(data);
            }

            if (Array.isArray(data)) {
                return data.map(item => this._sanitizeRequestData(item));
            }

            if (data && typeof data === "object") {
                const sanitized = {};
                Object.keys(data).forEach(key => {
                    const sanitizedKey = this._sanitizeString(key);
                    sanitized[sanitizedKey] = this._sanitizeRequestData(data[key]);
                });
                return sanitized;
            }

            return data;
        },

        /**
         * Sanitizes a string to prevent XSS attacks
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @param {string} sInput String to sanitize
         * @returns {string} Sanitized string
         * @since 1.0.0
         */
        _sanitizeString(sInput) {
            if (typeof sInput !== "string") {
                return sInput;
            }

            return sInput
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#x27;")
                .replace(/\//g, "&#x2F;")
                .replace(/\\/g, "&#x5C;")
                .replace(/&/g, "&amp;");
        },

        /**
         * Handles secure response processing
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @param {Response} response Fetch response
         * @returns {Promise} Response data or error
         * @since 1.0.0
         */
        async _handleSecureResponse(response) {
            const correlationId = response.headers.get("X-Correlation-ID");

            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    errorData = { error: "Network error", status: response.status };
                }

                // Log security-related errors
                if (response.status === 403 || response.status === 401) {
                    console.warn(`[SECURITY] ${response.status} response for correlation ID: ${correlationId}`);
                }

                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            try {
                const data = await response.json();
                return data;
            } catch (error) {
                // Handle non-JSON responses
                return await response.text();
            }
        },

        /**
         * Handles request errors with proper logging
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @private
         * @param {Error} error Request error
         * @param {string} sUrl Request URL
         * @param {string} sMethod HTTP method
         * @returns {Promise} Rejected promise with sanitized error
         * @since 1.0.0
         */
        _handleRequestError(error, sUrl, sMethod) {
            const correlationId = this.oUIModel.getProperty("/correlationId");

            // Log error with correlation ID
            console.error(`[REQUEST ERROR] ${sMethod} ${sUrl} - Correlation ID: ${correlationId}`, error);

            // Return sanitized error message
            const sanitizedMessage = error.message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            return Promise.reject(new Error(sanitizedMessage));
        },

        /**
         * Validates and sanitizes user input
         *
         * @function
         * @memberOf a2a.network.fiori.controller.BaseController
         * @protected
         * @param {string} sInput Input to validate
         * @param {string} sType Validation type (email, url, number, etc.)
         * @param {object} [oOptions] Validation options
         * @returns {boolean|*} Validation result or sanitized value
         * @since 1.0.0
         */
        validateInput(sInput, sType, oOptions = {}) {
            if (!sInput && !oOptions.required) {
                return oOptions.returnValue ? sInput : true;
            }

            if (!sInput && oOptions.required) {
                return oOptions.returnValue ? null : false;
            }

            const sanitized = this._sanitizeString(sInput);

            switch (sType) {
            case "email":
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                return oOptions.returnValue ?
                    (emailRegex.test(sanitized) ? sanitized : null) :
                    emailRegex.test(sanitized);

            case "url":
                try {
                    const url = new URL(sanitized);
                    return oOptions.returnValue ? url.href : true;
                } catch (e) {
                    return oOptions.returnValue ? null : false;
                }

            case "number":
                const num = parseFloat(sanitized);
                const isValid = !isNaN(num) && isFinite(num);
                if (oOptions.min !== undefined && num < oOptions.min) {
                    return oOptions.returnValue ? null : false;
                }
                if (oOptions.max !== undefined && num > oOptions.max) {
                    return oOptions.returnValue ? null : false;
                }
                return oOptions.returnValue ? (isValid ? num : null) : isValid;

            case "text":
                if (oOptions.maxLength && sanitized.length > oOptions.maxLength) {
                    return oOptions.returnValue ? null : false;
                }
                if (oOptions.minLength && sanitized.length < oOptions.minLength) {
                    return oOptions.returnValue ? null : false;
                }
                return oOptions.returnValue ? sanitized : true;

            default:
                return oOptions.returnValue ? sanitized : true;
            }
        }
    });
});