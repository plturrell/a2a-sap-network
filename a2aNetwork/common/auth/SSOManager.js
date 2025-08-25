/**
 * Single Sign-On Manager for A2A Platform
 * Handles authentication integration with SAP Identity Services
 */

window.A2A_SSO_Manager = {
    // Current authentication state
    isAuthenticated: false,
    currentUser: null,
    authToken: null,

    /**
     * Initialize SSO - Development mode bypass
     */
    init: function() {
        console.log('🔐 A2A SSO Manager initialized (Development Mode)');

        // Development mode - bypass authentication
        if (window.A2A_CONFIG && window.A2A_CONFIG.development && window.A2A_CONFIG.development.enableDebug) {
            this.isAuthenticated = true;
            this.currentUser = {
                id: 'developer',
                name: 'Development User',
                email: 'dev@a2a-network.com',
                roles: ['admin', 'developer']
            };
            console.log('🚀 Development authentication active');
            return Promise.resolve(this.currentUser);
        }

        // Production mode - check for existing session
        return this.checkSession();
    },

    /**
     * Check existing session
     */
    checkSession: function() {
        return new Promise((resolve, reject) => {
            // Check for stored auth token
            const storedToken = localStorage.getItem('a2a_auth_token');
            if (storedToken) {
                try {
                    const payload = JSON.parse(atob(storedToken.split('.')[1]));
                    if (payload.exp > Date.now() / 1000) {
                        this.authToken = storedToken;
                        this.isAuthenticated = true;
                        this.currentUser = {
                            id: payload.sub,
                            name: payload.name || payload.sub,
                            email: payload.email,
                            roles: payload.roles || ['user']
                        };
                        resolve(this.currentUser);
                        return;
                    }
                } catch (error) {
                    console.warn('Invalid auth token found', error);
                }
            }

            // No valid session found
            this.logout();
            reject(new Error('No valid session'));
        });
    },

    /**
     * Login with credentials
     */
    login: function(credentials) {
        return new Promise((resolve, reject) => {
            // Development mode - always succeed
            if (window.A2A_CONFIG && window.A2A_CONFIG.development) {
                setTimeout(() => {
                    this.isAuthenticated = true;
                    this.currentUser = {
                        id: credentials.username || 'testuser',
                        name: credentials.name || 'Test User',
                        email: credentials.email || 'test@a2a-network.com',
                        roles: ['user']
                    };
                    resolve(this.currentUser);
                }, 500); // Simulate network delay
                return;
            }

            // Production login logic would go here
            // For now, reject to prevent unauthorized access
            reject(new Error('Production login not implemented'));
        });
    },

    /**
     * Logout current user
     */
    logout: function() {
        this.isAuthenticated = false;
        this.currentUser = null;
        this.authToken = null;
        localStorage.removeItem('a2a_auth_token');
        console.log('🚪 User logged out');

        // Redirect to login if not in development mode
        if (!window.A2A_CONFIG || !window.A2A_CONFIG.development) {
            window.location.href = '/login.html';
        }
    },

    /**
     * Get current user info
     */
    getCurrentUser: function() {
        return this.currentUser;
    },

    /**
     * Check if user has specific role
     */
    hasRole: function(role) {
        if (!this.isAuthenticated || !this.currentUser) {
            return false;
        }
        return this.currentUser.roles && this.currentUser.roles.includes(role);
    },

    /**
     * Get authorization header for API requests
     */
    getAuthHeader: function() {
        if (this.authToken) {
            return `Bearer ${this.authToken}`;
        }
        return null;
    }
};

// Auto-initialize when script loads
document.addEventListener('DOMContentLoaded', () => {
    if (window.A2A_SSO_Manager) {
        window.A2A_SSO_Manager.init().catch((error) => {
            console.warn('SSO initialization failed:', error.message);
        });
    }
});