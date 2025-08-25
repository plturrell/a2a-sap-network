/**
 * Unified Navigation Manager for A2A Platform
 * Implements cross-application navigation with context preservation
 * Test Case: TC-COM-LPD-002
 */

class UnifiedNavigation {
    constructor(config) {
        this.config = {
            applications: {
                launchpad: { url: '/launchpad', name: 'A2A Launchpad', icon: 'sap-icon://home' },
                network: { url: '/network', name: 'A2A Network', icon: 'sap-icon://network' },
                agents: { url: '/agents', name: 'A2A Agents', icon: 'sap-icon://group' }
            },
            contextStorage: config.contextStorage || 'sessionStorage',
            navigationTimeout: config.navigationTimeout || 2000,
            deepLinkPrefix: config.deepLinkPrefix || '/app',
            ...config
        };

        this.navigationHistory = [];
        this.currentContext = {};
        this.breadcrumbs = [];
        this.navigationListeners = new Map();

        this.initializeNavigation();
    }

    /**
     * Initialize navigation system
     */
    initializeNavigation() {
        // Set up browser history management
        if (typeof window !== 'undefined') {
            window.addEventListener('popstate', this.handlePopState.bind(this));

            // Override browser navigation
            const originalPushState = history.pushState;
            history.pushState = (...args) => {
                originalPushState.apply(history, args);
                this.onNavigationChange();
            };
        }

        // Load saved context
        this.loadContext();
    }

    /**
     * Navigate to a different application with context preservation
     * @param {string} targetApp - Target application ID
     * @param {Object} context - Context data to preserve
     * @param {Object} options - Navigation options
     * @returns {Promise<void>}
     */
    async navigateToApplication(targetApp, context = {}, options = {}) {
        const startTime = Date.now();

        try {
            // Validate target application
            if (!this.config.applications[targetApp]) {
                throw new Error(`Unknown application: ${targetApp}`);
            }

            // Save current context
            await this.preserveContext({
                ...this.currentContext,
                ...context,
                sourceApp: this.getCurrentApp(),
                targetApp,
                timestamp: new Date().toISOString()
            });

            // Update navigation history
            this.navigationHistory.push({
                from: this.getCurrentApp(),
                to: targetApp,
                context,
                timestamp: Date.now(),
                options
            });

            // Trigger pre-navigation hooks
            await this.triggerNavigationHooks('beforeNavigate', { targetApp, context });

            // Perform navigation
            const targetUrl = this.buildTargetUrl(targetApp, context, options);

            if (options.newWindow) {
                window.open(targetUrl, '_blank');
            } else {
                // Smooth transition
                await this.performSmoothTransition(targetUrl, targetApp);
            }

            // Update breadcrumbs
            this.updateBreadcrumb(targetApp, context);

            // Log navigation performance
            const navigationTime = Date.now() - startTime;
            this.logNavigationMetrics(targetApp, navigationTime);

            // Trigger post-navigation hooks
            await this.triggerNavigationHooks('afterNavigate', { targetApp, context });

        } catch (error) {
            console.error('Navigation failed:', error);
            this.handleNavigationError(error, targetApp);
            throw error;
        }
    }

    /**
     * Preserve navigation context across applications
     * @param {Object} context - Context to preserve
     * @returns {Promise<void>}
     */
    async preserveContext(context) {
        try {
            this.currentContext = {
                ...this.currentContext,
                ...context,
                lastUpdated: new Date().toISOString()
            };

            // Store in configured storage
            const storage = this.getContextStorage();
            storage.setItem('a2a_navigation_context', JSON.stringify(this.currentContext));

            // Sync to server if configured
            if (this.config.syncContextToServer) {
                await this.syncContextToServer(this.currentContext);
            }

            return this.currentContext;
        } catch (error) {
            console.error('Failed to preserve context:', error);
        }
    }

    /**
     * Update breadcrumb navigation trail
     * @param {string} appId - Application ID
     * @param {Object} context - Navigation context
     */
    updateBreadcrumb(appId, context) {
        const app = this.config.applications[appId];
        const breadcrumb = {
            id: `${appId}_${Date.now()}`,
            appId,
            title: context.title || app.name,
            url: this.buildTargetUrl(appId, context),
            icon: app.icon,
            timestamp: Date.now()
        };

        // Add to breadcrumb trail
        this.breadcrumbs.push(breadcrumb);

        // Limit breadcrumb history
        if (this.breadcrumbs.length > 10) {
            this.breadcrumbs.shift();
        }

        // Notify breadcrumb listeners
        this.notifyBreadcrumbChange();
    }

    /**
     * Build target URL with context parameters and comprehensive validation
     * @param {string} appId - Target application ID
     * @param {Object} context - Context data
     * @param {Object} options - Navigation options
     * @returns {string} Target URL
     */
    buildTargetUrl(appId, context = {}, options = {}) {
        const app = this.config.applications[appId];
        if (!app) {
            throw new Error(`Unknown application: ${appId}`);
        }

        // Validate and normalize base URL
        let url = this.validateAndNormalizeUrl(app.url);

        // Handle deep linking with validation
        if (context.deepLink) {
            const sanitizedDeepLink = this.sanitizeDeepLink(context.deepLink);
            // Ensure proper URL concatenation
            if (!url.endsWith('/') && !sanitizedDeepLink.startsWith('/')) {
                url += '/';
            }
            url += sanitizedDeepLink;
        }

        // Add context parameters with validation
        if (context.params && Object.keys(context.params).length > 0) {
            const validatedParams = this.validateParams(context.params);
            if (Object.keys(validatedParams).length > 0) {
                const params = new URLSearchParams(validatedParams);
                url = `${url}${url.includes('?') ? '&' : '?'}${params.toString()}`;
            }
        }

        // Add navigation token for context retrieval
        if (!options.skipToken) {
            const token = this.generateNavigationToken();
            url = `${url}${url.includes('?') ? '&' : '?'}navToken=${token}`;
        }

        // Final URL validation
        return this.validateFinalUrl(url);
    }

    /**
     * Validate and normalize URL
     * @param {string} url - URL to validate
     * @returns {string} Normalized URL
     */
    validateAndNormalizeUrl(url) {
        if (!url || typeof url !== 'string') {
            throw new Error('Invalid URL: URL must be a non-empty string');
        }

        // Handle relative URLs
        if (url.startsWith('/')) {
            return url;
        }

        // Handle absolute URLs
        if (url.startsWith('http://') || url.startsWith('https://')) {
            try {
                const urlObj = new URL(url);
                return urlObj.pathname;
            } catch (error) {
                throw new Error(`Invalid absolute URL: ${url}`);
            }
        }

        // Default to relative path
        return `/${url}`;
    }

    /**
     * Sanitize deep link to prevent injection
     * @param {string} deepLink - Deep link path
     * @returns {string} Sanitized deep link
     */
    sanitizeDeepLink(deepLink) {
        if (!deepLink || typeof deepLink !== 'string') {
            return '';
        }

        // Remove dangerous characters and patterns
        const sanitized = deepLink
            .replace(/[<>"\\']/g, '') // Remove HTML/script injection chars
            .replace(/javascript:/gi, '') // Remove javascript: protocol
            .replace(/data:/gi, '') // Remove data: protocol
            .replace(/vbscript:/gi, '') // Remove vbscript: protocol
            .replace(/file:/gi, '') // Remove file: protocol
            .replace(/\.\./g, '') // Remove directory traversal
            .replace(/\/+/g, '/') // Normalize multiple slashes
            .substring(0, 500); // Limit length

        return sanitized;
    }

    /**
     * Validate navigation parameters
     * @param {Object} params - Parameters to validate
     * @returns {Object} Validated parameters
     */
    validateParams(params) {
        const validated = {};

        for (const [key, value] of Object.entries(params)) {
            // Validate key
            if (typeof key !== 'string' || key.length === 0 || key.length > 50) {
                console.warn(`Invalid parameter key: ${key}`);
                continue;
            }

            // Sanitize key
            const sanitizedKey = key.replace(/[^a-zA-Z0-9_-]/g, '');
            if (sanitizedKey !== key) {
                console.warn(`Parameter key sanitized: ${key} -> ${sanitizedKey}`);
            }

            // Validate and sanitize value
            if (value !== null && value !== undefined) {
                const stringValue = String(value);
                if (stringValue.length > 200) {
                    console.warn(`Parameter value truncated: ${key}`);
                    validated[sanitizedKey] = stringValue.substring(0, 200);
                } else {
                    validated[sanitizedKey] = stringValue;
                }
            }
        }

        return validated;
    }

    /**
     * Validate final constructed URL
     * @param {string} url - Final URL to validate
     * @returns {string} Validated URL
     */
    validateFinalUrl(url) {
        // Check for common URL issues
        if (!url || typeof url !== 'string') {
            throw new Error('Invalid final URL: empty or non-string');
        }

        // Check length
        if (url.length > 2000) {
            throw new Error('URL too long: exceeds 2000 characters');
        }

        // In test environment, handle relative URLs gracefully
        if (typeof window !== 'undefined' && window.location && global.URL) {
            try {
                // Try to create a valid URL object for validation
                if (url.startsWith('/')) {
                    // Relative URL - construct with current origin
                    new URL(url, window.location.origin || 'http://localhost:4004');
                } else if (url.startsWith('http')) {
                    // Absolute URL
                    new URL(url);
                }
            } catch (error) {
                console.warn(`URL validation warning: ${error.message}, using as-is: ${url}`);
            }
        }

        return url;
    }

    /**
     * Perform smooth application transition
     * @param {string} targetUrl - Target URL
     * @param {string} targetApp - Target application
     * @returns {Promise<void>}
     */
    async performSmoothTransition(targetUrl, targetApp) {
        // Show loading indicator
        this.showTransitionLoader(targetApp);

        // Pre-load target application resources if possible
        if (this.config.preloadResources) {
            await this.preloadApplicationResources(targetApp);
        }

        // Navigate
        if (this.config.useSPA) {
            // Single Page Application navigation
            this.navigateSPA(targetUrl);
        } else {
            // Traditional navigation
            window.location.href = targetUrl;
        }
    }

    /**
     * Handle browser back/forward navigation
     * @param {PopStateEvent} event
     */
    handlePopState(event) {
        if (event.state && event.state.context) {
            this.currentContext = event.state.context;
            this.notifyContextChange();
        }
    }

    /**
     * Get current application from URL
     * @returns {string} Current application ID
     */
    getCurrentApp() {
        if (typeof window === 'undefined') return 'unknown';

        const path = window.location.pathname;
        for (const [appId, config] of Object.entries(this.config.applications)) {
            if (path.startsWith(config.url)) {
                return appId;
            }
        }
        return 'launchpad';
    }

    /**
     * Generate navigation token for context passing
     * @returns {string} Navigation token
     */
    generateNavigationToken() {
        return `nav_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get context storage based on configuration
     * @returns {Storage} Storage object
     */
    getContextStorage() {
        if (typeof window === 'undefined') {
            return {
                setItem: () => {},
                getItem: () => null,
                removeItem: () => {}
            };
        }

        switch (this.config.contextStorage) {
            case 'localStorage':
                return window.localStorage;
            case 'sessionStorage':
                return window.sessionStorage;
            default:
                return window.sessionStorage;
        }
    }

    /**
     * Load saved context from storage
     */
    loadContext() {
        try {
            const storage = this.getContextStorage();
            const savedContext = storage.getItem('a2a_navigation_context');

            if (savedContext) {
                this.currentContext = JSON.parse(savedContext);
            }
        } catch (error) {
            console.error('Failed to load context:', error);
        }
    }

    /**
     * Register navigation event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     * @returns {Function} Unsubscribe function
     */
    on(event, callback) {
        if (!this.navigationListeners.has(event)) {
            this.navigationListeners.set(event, new Set());
        }

        this.navigationListeners.get(event).add(callback);

        // Return unsubscribe function
        return () => {
            const listeners = this.navigationListeners.get(event);
            if (listeners) {
                listeners.delete(callback);
            }
        };
    }

    /**
     * Trigger navigation hooks
     * @param {string} hookName - Hook name
     * @param {Object} data - Hook data
     */
    async triggerNavigationHooks(hookName, data) {
        const hooks = this.navigationListeners.get(hookName);
        if (hooks) {
            for (const hook of hooks) {
                try {
                    await hook(data);
                } catch (error) {
                    console.error(`Navigation hook error (${hookName}):`, error);
                }
            }
        }
    }

    /**
     * Show transition loader
     * @param {string} targetApp - Target application
     */
    showTransitionLoader(targetApp) {
        if (typeof document === 'undefined') return;

        // Create or update loader element
        let loader = document.getElementById('a2a-navigation-loader');
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'a2a-navigation-loader';
            loader.className = 'a2a-navigation-loader';
            document.body.appendChild(loader);
        }

        loader.innerHTML = `
            <div class="loader-content">
                <div class="spinner"></div>
                <p>Navigating to ${this.config.applications[targetApp].name}...</p>
            </div>
        `;
        loader.style.display = 'flex';
    }

    /**
     * Hide transition loader
     */
    hideTransitionLoader() {
        if (typeof document === 'undefined') return;

        const loader = document.getElementById('a2a-navigation-loader');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    /**
     * Preload application resources
     * @param {string} appId - Application ID
     */
    async preloadApplicationResources(appId) {
        // Implementation would preload CSS, JS, and other resources
        // This is a placeholder for actual implementation
        return new Promise(resolve => setTimeout(resolve, 100));
    }

    /**
     * Navigate in SPA mode
     * @param {string} url - Target URL
     */
    navigateSPA(url) {
        // Implementation for SPA navigation
        // Would integrate with routing framework
        history.pushState({ context: this.currentContext }, '', url);
        this.onNavigationChange();
    }

    /**
     * Handle navigation change
     */
    onNavigationChange() {
        this.hideTransitionLoader();
        this.notifyContextChange();
    }

    /**
     * Notify context change listeners
     */
    notifyContextChange() {
        this.triggerNavigationHooks('contextChange', this.currentContext);
    }

    /**
     * Notify breadcrumb change listeners
     */
    notifyBreadcrumbChange() {
        this.triggerNavigationHooks('breadcrumbChange', this.breadcrumbs);
    }

    /**
     * Log navigation metrics
     * @param {string} targetApp - Target application
     * @param {number} navigationTime - Navigation time in ms
     */
    logNavigationMetrics(targetApp, navigationTime) {
        const metric = {
            sourceApp: this.getCurrentApp(),
            targetApp,
            navigationTime,
            timestamp: new Date().toISOString(),
            contextSize: JSON.stringify(this.currentContext).length
        };

        // Send to monitoring system
        if (this.config.metricsEndpoint) {
            this.sendMetrics(metric);
        }
    }

    /**
     * Send metrics to monitoring endpoint
     * @param {Object} metric - Metric data
     */
    async sendMetrics(metric) {
        try {
            await blockchainClient.sendMessage(this.config.metricsEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(metric)
            });
        } catch (error) {
            console.error('Failed to send metrics:', error);
        }
    }

    /**
     * Handle navigation errors
     * @param {Error} error - Navigation error
     * @param {string} targetApp - Target application
     */
    handleNavigationError(error, targetApp) {
        // Log error
        console.error(`Navigation to ${targetApp} failed:`, error);

        // Show user-friendly error
        if (typeof window !== 'undefined') {
            this.showNavigationError(error.message);
        }

        // Trigger error hooks
        this.triggerNavigationHooks('navigationError', { error, targetApp });
    }

    /**
     * Show navigation error to user
     * @param {string} message - Error message
     */
    showNavigationError(message) {
        // Use console.error in test environment, alert in browser
        if (typeof window !== 'undefined' && window.alert && typeof window.alert === 'function') {
            window.alert(`Navigation failed: ${message}`);
        } else {
            console.error(`Navigation failed: ${message}`);
        }
    }

    /**
     * Get navigation history
     * @param {number} limit - Maximum number of entries
     * @returns {Array} Navigation history
     */
    getNavigationHistory(limit = 10) {
        return this.navigationHistory.slice(-limit);
    }

    /**
     * Get current breadcrumbs
     * @returns {Array} Breadcrumb trail
     */
    getBreadcrumbs() {
        return [...this.breadcrumbs];
    }

    /**
     * Clear navigation context
     */
    clearContext() {
        this.currentContext = {};
        const storage = this.getContextStorage();
        storage.removeItem('a2a_navigation_context');
    }

    /**
     * Sync context to server
     * @param {Object} context - Context to sync
     */
    async syncContextToServer(context) {
        if (!this.config.contextSyncEndpoint) return;

        try {
            await blockchainClient.sendMessage(this.config.contextSyncEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId: context.userId,
                    sessionId: context.sessionId,
                    context: context
                })
            });
        } catch (error) {
            console.error('Failed to sync context:', error);
        }
    }
}

module.exports = UnifiedNavigation;