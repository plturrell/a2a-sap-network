/**
 * Test-specific configuration overrides for A2A Launchpad Common Components
 * Provides controlled test environments with appropriate mocking and validation
 */

class TestConfiguration {
    constructor() {
        this.configs = {
            sso: this.getTestSSOConfig(),
            navigation: this.getTestNavigationConfig(),
            resources: this.getTestResourcesConfig(),
            monitoring: this.getTestMonitoringConfig(),
            disasterRecovery: this.getTestDisasterRecoveryConfig()
        };
    }

    /**
     * Get SSO configuration for testing
     */
    getTestSSOConfig() {
        return {
            jwtSecret: 'test-secret-key-for-unit-testing-only',
            tokenExpiry: '1h',
            refreshTokenExpiry: '24h',
            sessionTimeout: 1800000, // 30 minutes
            
            // Test-specific settings
            enableLogging: true,
            useInMemoryStorage: true,
            mockExternalServices: true,
            
            // SAML test configuration
            samlConfig: {
                entryPoint: 'http://localhost:4004/test/saml/login',
                issuer: 'test-a2a-network',
                cert: 'test-certificate',
                acceptedClockSkewMs: -1,
                disableRequestedAuthnContext: true
            },
            
            // OAuth2 test configuration
            oauth2Config: {
                clientId: 'test-client-id',
                clientSecret: 'test-client-secret',
                redirectUri: 'http://localhost:4004/test/oauth/callback',
                scope: ['read', 'write'],
                authorizationURL: 'http://localhost:4004/test/oauth/authorize',
                tokenURL: 'http://localhost:4004/test/oauth/token'
            },

            // Test user profiles
            testUsers: [
                {
                    id: 'test-user-1',
                    email: 'test.user@example.com',
                    name: 'Test User',
                    roles: ['NetworkAdmin'],
                    permissions: ['network_admin', 'agent_view', 'agent_edit']
                },
                {
                    id: 'test-user-2',
                    email: 'admin@example.com',
                    name: 'System Admin',
                    roles: ['SystemAdmin'],
                    permissions: ['system_admin', 'user_management', 'security_admin'],
                    mfaEnabled: true
                }
            ]
        };
    }

    /**
     * Get Navigation configuration for testing
     */
    getTestNavigationConfig() {
        return {
            applications: {
                launchpad: { 
                    url: '/test/launchpad', 
                    name: 'Test A2A Launchpad',
                    icon: 'sap-icon://home'
                },
                network: { 
                    url: '/test/network', 
                    name: 'Test A2A Network',
                    icon: 'sap-icon://network'
                },
                agents: { 
                    url: '/test/agents', 
                    name: 'Test A2A Agents',
                    icon: 'sap-icon://group'
                }
            },
            
            // Test-specific settings
            contextStorage: 'memory', // Use in-memory storage for tests
            navigationTimeout: 1000, // Shorter timeout for faster tests
            deepLinkPrefix: '/test/app',
            enableValidation: true,
            strictUrlValidation: true,
            
            // Mock external dependencies
            mockBrowserAPIs: true,
            mockHistoryAPI: true,
            mockLocationAPI: true,
            
            // Test navigation scenarios
            testScenarios: {
                crossAppNavigation: {
                    from: 'launchpad',
                    to: 'network',
                    context: { agentId: '123', tab: 'details' }
                },
                deepLinkNavigation: {
                    from: 'network',
                    to: 'agents',
                    deepLink: '/project/456/file/main.js',
                    params: { line: 42 }
                }
            }
        };
    }

    /**
     * Get Resources configuration for testing
     */
    getTestResourcesConfig() {
        return {
            syncInterval: 1000, // Fast sync for tests (1 second)
            conflictResolution: 'last-writer-wins',
            
            // Use in-memory storage for tests
            storage: {
                type: 'memory',
                options: {
                    maxSize: 1000,
                    ttl: 300000 // 5 minutes
                }
            },
            
            // Mock backends for testing
            backends: {
                filesystem: {
                    enabled: false, // Disable filesystem for tests
                    path: '/tmp/test-a2a-resources'
                },
                s3: {
                    enabled: false, // Mock S3 for tests
                    mock: true
                },
                azure: {
                    enabled: false, // Mock Azure for tests
                    mock: true
                }
            },
            
            // Test configurations
            testConfigurations: {
                'theme.settings': {
                    darkMode: true,
                    primaryColor: '#1976d2',
                    fontSize: 'medium'
                },
                'feature.flags': {
                    newUIEnabled: true,
                    betaFeaturesEnabled: false,
                    experimentalFeatures: true
                }
            },
            
            // Test assets
            testAssets: {
                'logo.png': {
                    type: 'image',
                    size: 1024,
                    checksum: 'test-checksum-123'
                }
            }
        };
    }

    /**
     * Get Monitoring configuration for testing
     */
    getTestMonitoringConfig() {
        return {
            metricsInterval: 100, // Fast collection for tests (100ms)
            
            // Alert thresholds for testing
            alertThresholds: {
                responseTime: 1000, // Lower threshold for test detection
                errorRate: 0.1,
                resourceUsage: 0.8
            },
            
            // Mock monitoring backends
            backends: {
                prometheus: {
                    enabled: false,
                    mock: true,
                    url: 'http://localhost:9090'
                },
                grafana: {
                    enabled: false,
                    mock: true,
                    url: 'http://localhost:3000'
                }
            },
            
            // Test metrics
            testMetrics: {
                'response_time': [100, 150, 200, 180, 220],
                'error_rate': [0.01, 0.02, 0.015, 0.03, 0.025],
                'active_users': [10, 12, 15, 11, 14]
            },
            
            // Mock alert scenarios
            testAlerts: [
                {
                    name: 'High Response Time',
                    severity: 'warning',
                    source: 'a2a-network',
                    value: 1200,
                    threshold: 1000
                },
                {
                    name: 'Service Unavailable',
                    severity: 'critical',
                    source: 'a2a-agents',
                    message: 'Service health check failed'
                }
            ]
        };
    }

    /**
     * Get Disaster Recovery configuration for testing
     */
    getTestDisasterRecoveryConfig() {
        return {
            backupInterval: 5000, // Fast backup for tests (5 seconds)
            
            // Test RTO/RPO objectives (shortened for testing)
            rto: {
                database: 300000, // 5 minutes
                application: 180000, // 3 minutes
                network: 60000, // 1 minute
                complete: 600000 // 10 minutes
            },
            
            rpo: {
                database: 60000, // 1 minute
                application: 30000, // 30 seconds
                network: 0, // real-time
                complete: 300000 // 5 minutes
            },
            
            // Mock backup storage
            backupStorage: {
                type: 'memory',
                options: {
                    maxBackups: 5,
                    compressionEnabled: false // Disable for faster tests
                }
            },
            
            // Test backup scenarios
            testBackups: [
                {
                    id: 'test-backup-1',
                    timestamp: Date.now() - 3600000, // 1 hour ago
                    size: 1024,
                    type: 'full',
                    status: 'completed'
                },
                {
                    id: 'test-backup-2',
                    timestamp: Date.now() - 1800000, // 30 minutes ago
                    size: 512,
                    type: 'incremental',
                    status: 'completed'
                }
            ],
            
            // Mock failover scenarios
            testFailoverScenarios: {
                databaseFailure: {
                    enabled: true,
                    duration: 10000, // 10 seconds
                    expectedRTO: 300000
                },
                networkPartition: {
                    enabled: true,
                    duration: 5000, // 5 seconds
                    expectedRTO: 60000
                }
            }
        };
    }

    /**
     * Get configuration for specific component
     * @param {string} component - Component name
     * @param {Object} overrides - Additional overrides
     * @returns {Object} Component configuration
     */
    getConfig(component, overrides = {}) {
        const baseConfig = this.configs[component];
        if (!baseConfig) {
            throw new Error(`Unknown test component: ${component}`);
        }

        return this.deepMerge(baseConfig, overrides);
    }

    /**
     * Get all configurations
     * @param {Object} globalOverrides - Global overrides to apply
     * @returns {Object} All configurations
     */
    getAllConfigs(globalOverrides = {}) {
        const allConfigs = {};
        
        for (const [component, config] of Object.entries(this.configs)) {
            allConfigs[component] = this.deepMerge(config, globalOverrides);
        }
        
        return allConfigs;
    }

    /**
     * Deep merge two objects
     * @param {Object} target - Target object
     * @param {Object} source - Source object
     * @returns {Object} Merged object
     */
    deepMerge(target, source) {
        const result = { ...target };
        
        for (const [key, value] of Object.entries(source)) {
            if (value && typeof value === 'object' && !Array.isArray(value)) {
                result[key] = this.deepMerge(result[key] || {}, value);
            } else {
                result[key] = value;
            }
        }
        
        return result;
    }

    /**
     * Reset configurations to defaults
     */
    reset() {
        this.configs = {
            sso: this.getTestSSOConfig(),
            navigation: this.getTestNavigationConfig(),
            resources: this.getTestResourcesConfig(),
            monitoring: this.getTestMonitoringConfig(),
            disasterRecovery: this.getTestDisasterRecoveryConfig()
        };
    }

    /**
     * Create test environment configuration
     * @param {string} environment - Environment type (unit, integration, e2e)
     * @returns {Object} Environment-specific configuration
     */
    createEnvironmentConfig(environment = 'unit') {
        const baseConfigs = this.getAllConfigs();
        
        switch (environment) {
            case 'unit':
                return this.deepMerge(baseConfigs, {
                    sso: { mockExternalServices: true },
                    navigation: { mockBrowserAPIs: true },
                    resources: { storage: { type: 'memory' } },
                    monitoring: { backends: { prometheus: { mock: true } } },
                    disasterRecovery: { backupStorage: { type: 'memory' } }
                });
                
            case 'integration':
                return this.deepMerge(baseConfigs, {
                    sso: { mockExternalServices: false, useInMemoryStorage: false },
                    resources: { storage: { type: 'filesystem' } },
                    monitoring: { backends: { prometheus: { enabled: true } } }
                });
                
            case 'e2e':
                return this.deepMerge(baseConfigs, {
                    sso: { mockExternalServices: false },
                    navigation: { mockBrowserAPIs: false },
                    resources: { backends: { filesystem: { enabled: true } } },
                    monitoring: { backends: { prometheus: { enabled: true } } }
                });
                
            default:
                return baseConfigs;
        }
    }
}

module.exports = TestConfiguration;