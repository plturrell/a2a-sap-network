sap.ui.define([
    'sap/base/Log',
    'sap/ui/base/Object'
], (Log, BaseObject) => {
    'use strict';

    /**
     * Enterprise Security Service for SAP Fiori Launchpad
     * Handles CSRF tokens, authentication, and authorization
     */
    return BaseObject.extend('a2a.network.launchpad.services.SecurityService', {

        _csrfToken: null,
        _csrfTokenExpiry: null,
        _authToken: null,
        _userRoles: [],
        _authorizationCache: new Map(),

        constructor: function() {
            BaseObject.prototype.constructor.apply(this, arguments);
            this._initializeSecurityContext();
        },

        /**
         * Initialize security context on startup
         */
        _initializeSecurityContext: function() {
            // Fetch CSRF token immediately
            this.fetchCSRFToken();

            // Setup token refresh interval (every 25 minutes for 30-minute tokens)
            setInterval(this.fetchCSRFToken.bind(this), 25 * 60 * 1000);

            // Initialize user context
            this._initializeUserContext();
        },

        /**
         * Fetch CSRF token from server
         * @returns {Promise<string>} CSRF token
         */
        fetchCSRFToken: function() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: '/sap/bc/ui2/start_up',
                    method: 'HEAD',
                    headers: {
                        'X-CSRF-Token': 'Fetch',
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    success: (data, textStatus, jqXHR) => {
                        this._csrfToken = jqXHR.getResponseHeader('X-CSRF-Token');
                        this._csrfTokenExpiry = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes
                        Log.info('CSRF token fetched successfully');
                        resolve(this._csrfToken);
                    },
                    error: (jqXHR, textStatus, errorThrown) => {
                        // Fallback for non-SAP environments
                        if (jqXHR.status === 404) {
                            // Try alternative endpoint
                            this._fetchCSRFTokenFallback().then(resolve).catch(reject);
                        } else {
                            Log.error('Failed to fetch CSRF token', errorThrown);
                            reject(errorThrown);
                        }
                    }
                });
            });
        },

        /**
         * Fallback CSRF token fetch for non-NetWeaver environments
         */
        _fetchCSRFTokenFallback: function() {
            return jQuery.ajax({
                url: '/api/v1/security/csrf',
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            }).then(response => {
                this._csrfToken = response.token;
                this._csrfTokenExpiry = new Date(Date.now() + 30 * 60 * 1000);
                return this._csrfToken;
            });
        },

        /**
         * Get current CSRF token, fetching new one if expired
         * @returns {Promise<string>} Valid CSRF token
         */
        getCSRFToken: function() {
            if (this._csrfToken && this._csrfTokenExpiry > new Date()) {
                return Promise.resolve(this._csrfToken);
            }
            return this.fetchCSRFToken();
        },

        /**
         * Initialize user context and roles
         */
        _initializeUserContext: function() {
            if (sap.ushell && sap.ushell.Container) {
                try {
                    const userInfoService = sap.ushell.Container.getService('UserInfo');
                    const user = userInfoService.getUser();

                    // Get user ID and roles
                    this._userId = user.getId();
                    this._userFullName = user.getFullName();
                    this._userEmail = user.getEmail();

                    // Fetch user roles from backend
                    this._fetchUserRoles();

                    // Get authentication token
                    if (user.getAccessToken) {
                        this._authToken = user.getAccessToken();
                    }
                } catch (error) {
                    Log.warning('Could not initialize user context from shell', error);
                    this._initializeUserContextFallback();
                }
            } else {
                this._initializeUserContextFallback();
            }
        },

        /**
         * Fallback user context initialization
         */
        _initializeUserContextFallback: function() {
            // Try to get user info from session
            jQuery.ajax({
                url: '/api/v1/user/current',
                method: 'GET',
                headers: {
                    'X-CSRF-Token': this._csrfToken
                },
                success: (userData) => {
                    this._userId = userData.id;
                    this._userFullName = userData.name;
                    this._userEmail = userData.email;
                    this._userRoles = userData.roles || [];
                    this._authToken = userData.token;
                },
                error: (error) => {
                    Log.error('Failed to fetch user context', error);
                }
            });
        },

        /**
         * Fetch user roles from backend
         */
        _fetchUserRoles: function() {
            jQuery.ajax({
                url: '/sap/bc/ui2/start_up',
                method: 'GET',
                headers: {
                    'X-CSRF-Token': this._csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                success: (data) => {
                    if (data && data.userRoles) {
                        this._userRoles = data.userRoles;
                        // Check for standard SAP roles
                        this._hasSAPUI2User = this._userRoles.includes('SAP_UI2_USER_700') ||
                                             this._userRoles.includes('SAP_UI2_USER_750');
                        this._hasSAPUI2Admin = this._userRoles.includes('SAP_UI2_ADMIN_700') ||
                                              this._userRoles.includes('SAP_UI2_ADMIN_750');
                    }
                },
                error: (error) => {
                    Log.warning('Could not fetch user roles', error);
                    // Fallback to custom endpoint
                    this._fetchUserRolesFallback();
                }
            });
        },

        /**
         * Fallback role fetching
         */
        _fetchUserRolesFallback: function() {
            jQuery.ajax({
                url: '/api/v1/user/roles',
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${  this._authToken}`
                },
                success: (roles) => {
                    this._userRoles = roles;
                }
            });
        },

        /**
         * Check authorization for specific object
         * @param {string} authObject - Authorization object (e.g., "S_SERVICE")
         * @param {string} authField - Authorization field
         * @param {string} authValue - Value to check
         * @returns {Promise<boolean>} Authorization result
         */
        checkAuthorization: function(authObject, authField, authValue) {
            const cacheKey = `${authObject}:${authField}:${authValue}`;

            // Check cache first
            if (this._authorizationCache.has(cacheKey)) {
                return Promise.resolve(this._authorizationCache.get(cacheKey));
            }

            // For NetWeaver environments
            if (window.location.pathname.indexOf('/sap/bc/') !== -1) {
                return new Promise((resolve) => {
                    jQuery.ajax({
                        url: '/sap/bc/ui2/check_auth',
                        method: 'POST',
                        headers: {
                            'X-CSRF-Token': this._csrfToken,
                            'Content-Type': 'application/json'
                        },
                        data: JSON.stringify({
                            authObject: authObject,
                            authField: authField,
                            authValue: authValue
                        }),
                        success: (result) => {
                            const authorized = result.authorized === true;
                            this._authorizationCache.set(cacheKey, authorized);
                            resolve(authorized);
                        },
                        error: () => {
                            // On error, check local roles
                            const authorized = this._checkLocalAuthorization(authObject, authField, authValue);
                            resolve(authorized);
                        }
                    });
                });
            } else {
                // For non-NetWeaver, use local authorization
                const authorized = this._checkLocalAuthorization(authObject, authField, authValue);
                return Promise.resolve(authorized);
            }
        },

        /**
         * Local authorization check based on roles
         */
        _checkLocalAuthorization: function(authObject, authField, authValue) {
            // Map common authorization objects to roles
            const authMap = {
                'S_SERVICE': {
                    'SRV_NAME': {
                        'ZFIORI_LAUNCHPAD': this._hasSAPUI2User || this._userRoles.includes('FLP_USER'),
                        'ZFIORI_ADMIN': this._hasSAPUI2Admin || this._userRoles.includes('FLP_ADMIN')
                    }
                },
                '/UI2/CHIP': {
                    'CHIP_ID': {
                        '*': this._hasSAPUI2User || this._userRoles.includes('FLP_USER')
                    }
                }
            };

            if (authMap[authObject] && authMap[authObject][authField]) {
                return authMap[authObject][authField][authValue] ||
                       authMap[authObject][authField]['*'] || false;
            }

            // Default allow for development
            return true;
        },

        /**
         * Get secure headers for AJAX requests
         * @returns {Promise<Object>} Headers object with security tokens
         */
        getSecureHeaders: function() {
            return this.getCSRFToken().then(csrfToken => {
                const headers = {
                    'X-CSRF-Token': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                };

                if (this._authToken) {
                    headers['Authorization'] = `Bearer ${  this._authToken}`;
                }

                return headers;
            });
        },

        /**
         * Secure AJAX wrapper with automatic CSRF token handling
         * @param {Object} settings - jQuery AJAX settings
         * @returns {Promise} jQuery promise
         */
        secureAjax: function(settings) {
            return this.getSecureHeaders().then(headers => {
                settings.headers = Object.assign({}, settings.headers, headers);

                // Add error handler for 403 CSRF token errors
                const originalError = settings.error;
                settings.error = (jqXHR, textStatus, errorThrown) => {
                    if (jqXHR.status === 403 && jqXHR.getResponseHeader('X-CSRF-Token') === 'Required') {
                        // CSRF token expired, fetch new one and retry
                        Log.info('CSRF token expired, fetching new token and retrying');
                        return this.fetchCSRFToken().then(() => {
                            return this.secureAjax(settings);
                        });
                    }

                    if (originalError) {
                        originalError(jqXHR, textStatus, errorThrown);
                    }
                };

                return jQuery.ajax(settings);
            });
        },

        /**
         * Check if user has specific role
         * @param {string} role - Role to check
         * @returns {boolean} Has role
         */
        hasRole: function(role) {
            return this._userRoles.includes(role);
        },

        /**
         * Get user information
         * @returns {Object} User info
         */
        getUserInfo: function() {
            return {
                id: this._userId,
                name: this._userFullName,
                email: this._userEmail,
                roles: this._userRoles,
                isAuthenticated: !!this._authToken
            };
        },

        /**
         * Get authentication token for WebSocket connections
         * @returns {string} Auth token
         */
        getAuthToken: function() {
            return this._authToken;
        }
    });
});