sap.ui.define([
    "a2a/network/agent8/ext/utils/SecurityUtils",
    "a2a/network/agent8/ext/utils/SecurityConfig"
], (SecurityUtils, SecurityConfig) => {
    "use strict";

    /**
     * Authentication and Authorization Handler for Agent 8
     * Manages user authentication, session handling, and access control
     */
    return {

        _currentUser: null,
        _sessionToken: null,
        _lastActivity: null,
        _authCheckInterval: null,

        /**
         * Initialize authentication handler
         */
        init() {
            this._setupSessionMonitoring();
            this._setupAuthenticationCheck();
            this._bindAuthenticationEvents();

            // Check if user is already authenticated
            this._checkExistingSession();
        },

        /**
         * Authenticate user with credentials
         * @param {string} username - Username
         * @param {string} password - Password
         * @param {string} authMethod - Authentication method (SAML, OAuth2, BasicAuth)
         * @returns {Promise} Authentication result
         */
        authenticate(username, password, authMethod = "BasicAuth") {
            const authenticateUser = (resolve, reject) => {
                // Validate input parameters
                const usernameValidation = SecurityUtils.validateInput(username, "text", {
                    required: true,
                    minLength: 3,
                    maxLength: 50
                });

                if (!usernameValidation.isValid) {
                    SecurityUtils.auditLog("AUTHENTICATION_FAILED", {
                        reason: "Invalid username format",
                        username: SecurityUtils.sanitizeInput(username)
                    });
                    reject(new Error("Invalid username format"));
                    return;
                }

                if (!password || password.length < SecurityConfig.authentication.passwordPolicy.minLength) {
                    SecurityUtils.auditLog("AUTHENTICATION_FAILED", {
                        reason: "Password too short",
                        username: SecurityUtils.sanitizeInput(username)
                    });
                    reject(new Error("Password does not meet minimum requirements"));
                    return;
                }

                const sanitizedUsername = SecurityUtils.sanitizeInput(username);

                // Create authentication request
                const authRequest = {
                    username: sanitizedUsername,
                    password, // Don't log password
                    authMethod: SecurityUtils.sanitizeInput(authMethod),
                    timestamp: new Date().toISOString(),
                    clientInfo: {
                        userAgent: navigator.userAgent,
                        ipAddress: this._getClientIP(),
                        sessionId: this._generateSessionId()
                    }
                };

                // Make authentication request to backend
                jQuery.ajax({
                    url: "/a2a/agent8/v1/auth/authenticate",
                    type: "POST",
                    contentType: "application/json",
                    data: JSON.stringify(authRequest),
                    timeout: 10000, // 10 second timeout
                    success: function handleAuthSuccess(response) {
                        if (response.success && response.token) {
                            this._handleSuccessfulAuthentication(response, sanitizedUsername);
                            SecurityUtils.auditLog("AUTHENTICATION_SUCCESS", {
                                username: sanitizedUsername,
                                authMethod,
                                sessionId: response.sessionId
                            });
                            resolve(response);
                        } else {
                            SecurityUtils.auditLog("AUTHENTICATION_FAILED", {
                                reason: "Invalid credentials",
                                username: sanitizedUsername
                            });
                            reject(new Error("Authentication failed"));
                        }
                    },
                    error: function handleAuthError(xhr) {
                        const errorMsg = SecurityUtils.sanitizeErrorMessage(xhr.responseText);
                        SecurityUtils.auditLog("AUTHENTICATION_ERROR", {
                            error: errorMsg,
                            username: sanitizedUsername
                        });
                        reject(new Error("Authentication service unavailable"));
                    }
                });
            };
            return new Promise(authenticateUser);
        },

        /**
         * Check if user is currently authenticated
         * @returns {boolean} True if authenticated
         */
        isAuthenticated() {
            if (!this._currentUser || !this._sessionToken) {
                return false;
            }

            // Check session expiry
            if (this._isSessionExpired()) {
                this.logout("Session expired");
                return false;
            }

            // Check idle timeout
            if (this._isIdleTimeoutExceeded()) {
                this.logout("Idle timeout exceeded");
                return false;
            }

            return true;
        },

        /**
         * Check if user has specific permission
         * @param {string} permission - Permission to check
         * @param {string} resource - Resource identifier
         * @returns {boolean} True if authorized
         */
        hasPermission(permission, resource = "") {
            if (!this.isAuthenticated()) {
                SecurityUtils.auditLog("AUTHORIZATION_FAILED", {
                    reason: "Not authenticated",
                    permission,
                    resource
                });
                return false;
            }

            const user = this._currentUser;
            if (!user.permissions) {
                return false;
            }

            // Check specific permission
            const hasPermission = user.permissions.includes(permission) ||
                                user.permissions.includes("admin") ||
                                user.permissions.includes("*");

            if (!hasPermission) {
                SecurityUtils.auditLog("AUTHORIZATION_FAILED", {
                    username: user.username,
                    permission,
                    resource,
                    userPermissions: user.permissions
                });
            }

            return hasPermission;
        },

        /**
         * Check if user has specific role
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         */
        hasRole(role) {
            if (!this.isAuthenticated()) {
                return false;
            }

            const user = this._currentUser;
            return user.roles && user.roles.includes(role);
        },

        /**
         * Get current authenticated user
         * @returns {Object} Current user object
         */
        getCurrentUser() {
            if (!this.isAuthenticated()) {
                return null;
            }

            // Return sanitized user info (no sensitive data)
            return {
                username: this._currentUser.username,
                displayName: this._currentUser.displayName,
                email: this._currentUser.email,
                roles: this._currentUser.roles || [],
                permissions: this._currentUser.permissions || [],
                lastLogin: this._currentUser.lastLogin,
                sessionId: this._currentUser.sessionId
            };
        },

        /**
         * Logout current user
         * @param {string} reason - Reason for logout
         */
        logout(reason = "User initiated") {
            if (this._currentUser) {
                SecurityUtils.auditLog("USER_LOGOUT", {
                    username: this._currentUser.username,
                    sessionId: this._currentUser.sessionId,
                    reason,
                    sessionDuration: Date.now() - this._currentUser.loginTime
                });

                // Notify backend about logout
                this._notifyBackendLogout();
            }

            this._clearSessionData();
            this._redirectToLogin();
        },

        /**
         * Refresh authentication token
         * @returns {Promise} Token refresh result
         */
        refreshToken() {
            const refreshAuthToken = (resolve, reject) => {
                if (!this._sessionToken) {
                    reject(new Error("No session token to refresh"));
                    return;
                }

                jQuery.ajax({
                    url: "/a2a/agent8/v1/auth/refresh",
                    type: "POST",
                    contentType: "application/json",
                    headers: {
                        "Authorization": `Bearer ${ this._sessionToken}`,
                        "X-CSRF-Token": SecurityUtils.getCSRFToken()
                    },
                    data: JSON.stringify({
                        sessionId: this._currentUser.sessionId,
                        refreshToken: this._currentUser.refreshToken
                    }),
                    success: function handleRefreshSuccess(response) {
                        if (response.success && response.token) {
                            this._sessionToken = response.token;
                            this._lastActivity = Date.now();
                            SecurityUtils.auditLog("TOKEN_REFRESHED", {
                                username: this._currentUser.username,
                                sessionId: this._currentUser.sessionId
                            });
                            resolve(response);
                        } else {
                            reject(new Error("Token refresh failed"));
                        }
                    },
                    error: function handleRefreshError(xhr) {
                        const errorMsg = SecurityUtils.sanitizeErrorMessage(xhr.responseText);
                        SecurityUtils.auditLog("TOKEN_REFRESH_FAILED", {
                            error: errorMsg,
                            sessionId: this._currentUser ? this._currentUser.sessionId : "unknown"
                        });
                        reject(new Error("Token refresh service unavailable"));
                    }
                });
            };
            return new Promise(refreshAuthToken);
        },

        /**
         * Get authorization header for API requests
         * @returns {Object} Authorization headers
         */
        getAuthHeaders() {
            if (!this.isAuthenticated()) {
                return {};
            }

            return {
                "Authorization": `Bearer ${ this._sessionToken}`,
                "X-User-ID": this._currentUser.userId,
                "X-Session-ID": this._currentUser.sessionId,
                "X-CSRF-Token": SecurityUtils.getCSRFToken()
            };
        },

        /**
         * Record user activity to prevent idle timeout
         */
        recordActivity() {
            this._lastActivity = Date.now();
        },

        // Private methods

        _setupSessionMonitoring() {
            // Monitor user activity
            const recordClickActivity = () => this.recordActivity();
            const recordKeypressActivity = () => this.recordActivity();
            const recordScrollActivity = () => this.recordActivity();
            const recordMouseActivity = () => this.recordActivity();

            document.addEventListener("click", recordClickActivity);
            document.addEventListener("keypress", recordKeypressActivity);
            document.addEventListener("scroll", recordScrollActivity);
            document.addEventListener("mousemove", recordMouseActivity);

            // Set up periodic session validation
            const performAuthCheck = () => {
                if (this.isAuthenticated()) {
                    this._validateSession();
                }
            };
            this._authCheckInterval = setInterval(performAuthCheck, 60000); // Check every minute
        },

        _setupAuthenticationCheck() {
            // Intercept AJAX requests to add authentication headers
            const handleAjaxPrefilter = (options, originalOptions, jqXHR) => {
                if (this.isAuthenticated() && options.url.startsWith("/a2a/agent8/")) {
                    const authHeaders = this.getAuthHeaders();
                    options.headers = Object.assign(options.headers || {}, authHeaders);
                }
            };
            jQuery.ajaxPrefilter(handleAjaxPrefilter);

            // Handle authentication errors globally
            const handleAjaxError = (event, jqXHR, ajaxSettings, thrownError) => {
                if (jqXHR.status === 401) {
                    this.logout("Authentication expired");
                } else if (jqXHR.status === 403) {
                    SecurityUtils.auditLog("AUTHORIZATION_DENIED", {
                        url: ajaxSettings.url,
                        method: ajaxSettings.type,
                        status: jqXHR.status
                    });
                }
            };
            jQuery(document).ajaxError(handleAjaxError);
        },

        _bindAuthenticationEvents() {
            // Listen for storage events (logout from other tabs)
            const handleStorageEvent = (e) => {
                if (e.key === "a2a_agent8_logout" && e.newValue) {
                    this._clearSessionData();
                    location.reload();
                }
            };
            window.addEventListener("storage", handleStorageEvent);

            // Handle page unload
            const handleBeforeUnload = () => {
                if (this._currentUser) {
                    SecurityUtils.auditLog("SESSION_ENDED", {
                        username: this._currentUser.username,
                        sessionId: this._currentUser.sessionId,
                        reason: "Page unload"
                    });
                }
            };
            window.addEventListener("beforeunload", handleBeforeUnload);
        },

        _checkExistingSession() {
            const storedToken = localStorage.getItem("a2a_agent8_token");
            const storedUser = localStorage.getItem("a2a_agent8_user");

            if (storedToken && storedUser) {
                try {
                    this._sessionToken = storedToken;
                    this._currentUser = JSON.parse(storedUser);
                    this._lastActivity = Date.now();

                    // Validate session with backend
                    this._validateSession();
                } catch (e) {
                    this._clearSessionData();
                }
            }
        },

        _handleSuccessfulAuthentication(response, username) {
            this._sessionToken = response.token;
            this._currentUser = {
                username,
                displayName: response.displayName,
                email: response.email,
                userId: response.userId,
                roles: response.roles || [],
                permissions: response.permissions || [],
                sessionId: response.sessionId,
                refreshToken: response.refreshToken,
                loginTime: Date.now(),
                lastLogin: new Date().toISOString()
            };
            this._lastActivity = Date.now();

            // Store session data (encrypted in production)
            localStorage.setItem("a2a_agent8_token", this._sessionToken);
            localStorage.setItem("a2a_agent8_user", JSON.stringify(this._currentUser));
        },

        _validateSession() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/auth/validate",
                type: "GET",
                headers: this.getAuthHeaders(),
                timeout: 5000,
                success: function handleValidationSuccess(response) {
                    if (!response.valid) {
                        this.logout("Session invalidated by server");
                    }
                },
                error: function handleValidationError(xhr) {
                    if (xhr.status === 401) {
                        this.logout("Session validation failed");
                    }
                }
            });
        },

        _isSessionExpired() {
            if (!this._currentUser || !this._currentUser.loginTime) {
                return true;
            }

            const sessionAge = Date.now() - this._currentUser.loginTime;
            const maxSessionAge = SecurityConfig.sessionManagement.absoluteTimeout * 1000;

            return sessionAge > maxSessionAge;
        },

        _isIdleTimeoutExceeded() {
            if (!this._lastActivity) {
                return true;
            }

            const idleDuration = Date.now() - this._lastActivity;
            const maxIdleDuration = SecurityConfig.sessionManagement.idleTimeout * 1000;

            return idleDuration > maxIdleDuration;
        },

        _clearSessionData() {
            this._currentUser = null;
            this._sessionToken = null;
            this._lastActivity = null;

            localStorage.removeItem("a2a_agent8_token");
            localStorage.removeItem("a2a_agent8_user");

            // Clear any cached data
            if (sap.ui.getCore().getModel) {
                const model = sap.ui.getCore().getModel();
                if (model && model.resetChanges) {
                    model.resetChanges();
                }
            }

            // Signal other tabs
            localStorage.setItem("a2a_agent8_logout", Date.now());
            localStorage.removeItem("a2a_agent8_logout");
        },

        _notifyBackendLogout() {
            if (this._sessionToken && this._currentUser) {
                navigator.sendBeacon("/a2a/agent8/v1/auth/logout", JSON.stringify({
                    sessionId: this._currentUser.sessionId,
                    reason: "User logout"
                }));
            }
        },

        _redirectToLogin() {
            // In SAP Fiori launchpad context, redirect differently
            if (window.sap && window.sap.ushell) {
                // Redirect using ushell navigation
                window.sap.ushell.Container.getService("CrossApplicationNavigation").toExternal({
                    target: { semanticObject: "Login", action: "display" }
                });
            } else {
                // Standard redirect
                window.location.href = "/login";
            }
        },

        _generateSessionId() {
            // Generate cryptographically secure session ID
            const array = new Uint8Array(32);
            window.crypto.getRandomValues(array);
            return Array.from(array, byte => byte.toString(16).padStart(2, "0")).join("");
        },

        _getClientIP() {
            // This would typically be provided by the server
            // Client-side IP detection is not reliable for security purposes
            return "client-detected";
        },

        /**
         * Cleanup authentication handler
         */
        destroy() {
            if (this._authCheckInterval) {
                clearInterval(this._authCheckInterval);
                this._authCheckInterval = null;
            }

            this._clearSessionData();
        }
    };
});