/**
 * A2A Platform Configuration
 * This file should be populated with environment-specific values
 * DO NOT commit production secrets to version control
 */

// Configuration for development environment
// Environment variables are not available in browser - use development defaults
window.A2A_CONFIG = {
    // Authentication - development mode
    jwtSecret: '', // Not used in frontend
    apiBaseUrl: '/api/auth',
    
    // SAML Configuration - development defaults
    saml: {
        entityId: window.location.origin,
        idpUrl: '',
        trustedIssuers: []
    },
    
    // OAuth2 Configuration - development defaults  
    oauth2: {
        clientId: '',
        clientSecret: '', // Never expose in frontend
        authorizationUrl: '',
        tokenUrl: '',
        userinfoUrl: ''
    },
    
    // API Endpoints
    api: {
        base: '/api',
        version: 'v1',
        endpoints: {
            agents: '/api/v1/network/agents',
            services: '/api/v1/network/services',
            blockchain: '/api/v1/blockchain',
            analytics: '/api/v1/analytics',
            notifications: '/api/v1/notifications'
        }
    },
    
    // WebSocket Configuration
    websocket: {
        url: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
        reconnectInterval: 5000,
        maxReconnectAttempts: 10
    },
    
    // UI Configuration
    ui: {
        defaultTheme: 'sap_horizon',
        dateFormat: 'MM/dd/yyyy',
        timezone: 'UTC',
        pageSize: 20,
        refreshInterval: 30000
    },
    
    // Development flags
    development: {
        enableDebug: true,
        mockData: false,
        logLevel: 'info'
    },
    
    // Feature flags
    features: {
        realtimeNotifications: true,
        advancedAnalytics: true,
        blockchainIntegration: true,
        agentVisualization: true,
        serviceMarketplace: true
    }
};