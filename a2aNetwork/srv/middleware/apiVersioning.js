/**
 * @fileoverview Enterprise API Versioning and Deprecation System
 * @description Comprehensive API versioning middleware with SAP enterprise standards,
 * deprecation management, version routing, backward compatibility, and migration guidance
 * @module api-versioning
 * @since 1.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const semver = require('semver');
const crypto = require('crypto');

// OpenTelemetry integration for version tracking
let opentelemetry, trace, SpanStatusCode;
try {
    opentelemetry = require('@opentelemetry/api');
    trace = opentelemetry.trace;
    SpanStatusCode = opentelemetry.SpanStatusCode;
} catch (error) {
    // OpenTelemetry not available
}

/**
 * API versioning configuration
 */
const API_CONFIG = {
    // Current supported versions
    supportedVersions: ['1.0.0', '1.1.0', '2.0.0'],
    currentVersion: '2.0.0',
    defaultVersion: '1.0.0',
    
    // Version deprecation timeline
    deprecationSchedule: {
        '1.0.0': {
            deprecatedDate: '2024-01-01T00:00:00Z',
            sunsetDate: '2024-06-01T00:00:00Z',
            status: 'deprecated',
            migrationPath: '2.0.0'
        },
        '1.1.0': {
            deprecatedDate: null,
            sunsetDate: '2024-12-01T00:00:00Z',
            status: 'supported',
            migrationPath: '2.0.0'
        },
        '2.0.0': {
            deprecatedDate: null,
            sunsetDate: null,
            status: 'current',
            migrationPath: null
        }
    },
    
    // Version-specific feature flags
    featureFlags: {
        '1.0.0': {
            enableAdvancedMetrics: false,
            enableBulkOperations: false,
            enableEventStreaming: false,
            maxRequestSize: '1MB',
            rateLimit: 1000
        },
        '1.1.0': {
            enableAdvancedMetrics: true,
            enableBulkOperations: false,
            enableEventStreaming: false,
            maxRequestSize: '5MB',
            rateLimit: 2000
        },
        '2.0.0': {
            enableAdvancedMetrics: true,
            enableBulkOperations: true,
            enableEventStreaming: true,
            maxRequestSize: '10MB',
            rateLimit: 5000
        }
    }
};

/**
 * Breaking changes tracking between versions
 */
const BREAKING_CHANGES = {
    '2.0.0': {
        from: '1.x',
        changes: [
            {
                type: 'field_removed',
                endpoint: '/api/*/settings/network',
                field: 'legacyMode',
                description: 'Legacy mode field removed - use new configuration API'
            },
            {
                type: 'response_format',
                endpoint: '/api/*/metrics/*',
                description: 'Metrics response format changed - timestamps now in ISO 8601'
            },
            {
                type: 'authentication',
                endpoint: '/api/*/operations/*',
                description: 'Operations endpoints now require admin role'
            }
        ]
    },
    '1.1.0': {
        from: '1.0.0',
        changes: [
            {
                type: 'field_added',
                endpoint: '/api/*/settings/security',
                field: 'mfaEnabled',
                description: 'Added multi-factor authentication support'
            }
        ]
    }
};

/**
 * Enterprise API Versioning Manager
 */
class APIVersionManager {
    constructor() {
        this.log = cds.log('api-versioning');
        this.tracer = trace ? trace.getTracer('api-versioning', '1.0.0') : null;
        this.versionMetrics = new Map();
        this.deprecationWarnings = new Map();
        
        // Initialize metrics tracking
        this.initializeMetrics();
        
        this.intervals = new Map(); // Track intervals for cleanup
        
        // Start cleanup and reporting intervals
        this.startMaintenanceTasks();
    }

    /**
     * Initialize version usage metrics
     */
    initializeMetrics() {
        for (const version of API_CONFIG.supportedVersions) {
            this.versionMetrics.set(version, {
                requests: 0,
                errors: 0,
                avgResponseTime: 0,
                lastUsed: null,
                uniqueClients: new Set(),
                endpoints: new Map()
            });
        }
    }

    /**
     * Extract API version from request
     */
    extractVersion(req) {
        // Priority order for version detection:
        // 1. Header: API-Version
        // 2. Header: Accept-Version  
        // 3. Query parameter: version
        // 4. URL path: /api/v{version}/
        // 5. Content-Type: application/vnd.a2a.v{version}+json
        // 6. Default version
        
        let version = null;
        
        // Method 1: API-Version header (recommended)
        version = req.headers['api-version'] || req.headers['x-api-version'];
        if (version && this.isValidVersion(version)) {
            return this.normalizeVersion(version);
        }
        
        // Method 2: Accept-Version header
        version = req.headers['accept-version'];
        if (version && this.isValidVersion(version)) {
            return this.normalizeVersion(version);
        }
        
        // Method 3: Query parameter
        version = req.query.version;
        if (version && this.isValidVersion(version)) {
            return this.normalizeVersion(version);
        }
        
        // Method 4: URL path extraction
        const pathMatch = req.path.match(/\/api\/v?(\d+\.?\d*\.?\d*)\//);
        if (pathMatch) {
            version = pathMatch[1];
            if (this.isValidVersion(version)) {
                return this.normalizeVersion(version);
            }
        }
        
        // Method 5: Content-Type header
        const contentType = req.headers['content-type'];
        if (contentType) {
            const ctMatch = contentType.match(/application\/vnd\.a2a\.v(\d+\.?\d*\.?\d*)\+json/);
            if (ctMatch) {
                version = ctMatch[1];
                if (this.isValidVersion(version)) {
                    return this.normalizeVersion(version);
                }
            }
        }
        
        // Default version
        return API_CONFIG.defaultVersion;
    }

    /**
     * Validate if version is supported
     */
    isValidVersion(version) {
        if (!version) return false;
        const normalized = this.normalizeVersion(version);
        return API_CONFIG.supportedVersions.includes(normalized);
    }

    /**
     * Normalize version string to semver format
     */
    normalizeVersion(version) {
        if (!version) return API_CONFIG.defaultVersion;
        
        // Handle short versions like "1" or "1.1"
        const parts = version.split('.');
        while (parts.length < 3) {
            parts.push('0');
        }
        
        return parts.join('.');
    }

    /**
     * Get version information
     */
    getVersionInfo(version) {
        const deprecation = API_CONFIG.deprecationSchedule[version];
        const features = API_CONFIG.featureFlags[version];
        
        return {
            version,
            status: deprecation?.status || 'unknown',
            deprecatedDate: deprecation?.deprecatedDate,
            sunsetDate: deprecation?.sunsetDate,
            migrationPath: deprecation?.migrationPath,
            features,
            isSupported: API_CONFIG.supportedVersions.includes(version),
            isCurrent: version === API_CONFIG.currentVersion,
            isDeprecated: deprecation?.status === 'deprecated'
        };
    }

    /**
     * Check if version is deprecated and add appropriate warnings
     */
    checkDeprecation(version, req, res) {
        const versionInfo = this.getVersionInfo(version);
        
        if (versionInfo.isDeprecated || versionInfo.sunsetDate) {
            const warningKey = `${version}-${req.ip}-${req.headers['user-agent'] || 'unknown'}`;
            const warningHash = crypto.createHash('md5').update(warningKey).digest('hex');
            
            // Add deprecation headers
            res.set({
                'API-Deprecated': 'true',
                'API-Sunset-Date': versionInfo.sunsetDate || '',
                'API-Migration-Path': versionInfo.migrationPath || '',
                'Warning': `299 - "API version ${version} is deprecated. Migrate to ${versionInfo.migrationPath || API_CONFIG.currentVersion}"`
            });
            
            // Log deprecation warning (rate limited per client)
            if (!this.deprecationWarnings.has(warningHash)) {
                this.deprecationWarnings.set(warningHash, Date.now());
                
                this.log.warn('Deprecated API version used', {
                    version,
                    endpoint: req.path,
                    userAgent: req.headers['user-agent'],
                    ip: req.ip,
                    sunsetDate: versionInfo.sunsetDate,
                    migrationPath: versionInfo.migrationPath
                });
                
                // Emit metric event for monitoring
                if (this.tracer) {
                    const span = this.tracer.startSpan('api.deprecation.warning');
                    span.setAttributes({
                        'api.version': version,
                        'api.endpoint': req.path,
                        'api.migration_path': versionInfo.migrationPath || ''
                    });
                    span.end();
                }
            }
        }
        
        return versionInfo;
    }

    /**
     * Apply version-specific transformations to request
     */
    transformRequest(req, version) {
        const versionInfo = this.getVersionInfo(version);
        
        // Apply version-specific request transformations
        if (version === '1.0.0') {
            // Legacy compatibility transformations
            this.applyLegacyRequestTransforms(req);
        } else if (version === '1.1.0') {
            // V1.1 specific transformations
            this.applyV11RequestTransforms(req);
        }
        
        // Add version context
        req.apiVersion = version;
        req.apiFeatures = versionInfo.features;
        req.apiVersionInfo = versionInfo;
        
        return req;
    }

    /**
     * Apply version-specific transformations to response
     */
    transformResponse(data, version, req) {
        if (!data) return data;
        
        if (version === '1.0.0') {
            return this.applyLegacyResponseTransforms(data, req);
        } else if (version === '1.1.0') {
            return this.applyV11ResponseTransforms(data, req);
        }
        
        // Default: no transformation for current version
        return data;
    }

    /**
     * Apply legacy (v1.0.0) request transformations
     */
    applyLegacyRequestTransforms(req) {
        // Convert new field names to legacy equivalents
        if (req.body) {
            // Example: map new field names to legacy ones
            if (req.body.networkSettings && req.body.networkSettings.rpcEndpoint) {
                req.body.networkSettings.rpcUrl = req.body.networkSettings.rpcEndpoint;
                delete req.body.networkSettings.rpcEndpoint;
            }
            
            // Legacy date format handling
            this.convertDateFormatsForLegacy(req.body);
        }
    }

    /**
     * Apply v1.1.0 request transformations
     */
    applyV11RequestTransforms(req) {
        // V1.1 specific transformations
        if (req.body) {
            // Handle enhanced security fields
            if (req.body.securitySettings && !req.body.securitySettings.mfaEnabled) {
                req.body.securitySettings.mfaEnabled = false;
            }
        }
    }

    /**
     * Apply legacy (v1.0.0) response transformations
     */
    applyLegacyResponseTransforms(data, req) {
        if (typeof data !== 'object' || data === null) return data;
        
        // Remove fields not available in v1.0.0
        const transformed = { ...data };
        
        // Remove v2.0+ specific fields
        if (transformed.advancedMetrics) {
            delete transformed.advancedMetrics;
        }
        
        if (transformed.eventStream) {
            delete transformed.eventStream;
        }
        
        // Convert timestamp formats for legacy compatibility
        this.convertTimestampsForLegacy(transformed);
        
        // Add legacy compatibility fields
        if (req.path.includes('/settings/network')) {
            transformed.legacyMode = true;
        }
        
        return transformed;
    }

    /**
     * Apply v1.1.0 response transformations
     */
    applyV11ResponseTransforms(data, req) {
        if (typeof data !== 'object' || data === null) return data;
        
        const transformed = { ...data };
        
        // Remove v2.0+ specific fields
        if (transformed.bulkOperations) {
            delete transformed.bulkOperations;
        }
        
        if (transformed.eventStream) {
            delete transformed.eventStream;
        }
        
        return transformed;
    }

    /**
     * Convert date formats for legacy compatibility
     */
    convertDateFormatsForLegacy(obj) {
        if (!obj || typeof obj !== 'object') return;
        
        for (const [key, value] of Object.entries(obj)) {
            if (typeof value === 'string' && this.isISODateString(value)) {
                // Convert ISO 8601 to legacy format (Unix timestamp)
                obj[key] = Math.floor(new Date(value).getTime() / 1000);
            } else if (typeof value === 'object' && value !== null) {
                this.convertDateFormatsForLegacy(value);
            }
        }
    }

    /**
     * Convert timestamps for legacy compatibility
     */
    convertTimestampsForLegacy(obj) {
        if (!obj || typeof obj !== 'object') return;
        
        for (const [key, value] of Object.entries(obj)) {
            if (key.includes('timestamp') || key.includes('Date') || key.includes('Time')) {
                if (typeof value === 'string' && this.isISODateString(value)) {
                    obj[key] = Math.floor(new Date(value).getTime() / 1000);
                }
            } else if (typeof value === 'object' && value !== null) {
                this.convertTimestampsForLegacy(value);
            }
        }
    }

    /**
     * Check if string is ISO date format
     */
    isISODateString(str) {
        return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/.test(str);
    }

    /**
     * Record version usage metrics
     */
    recordVersionUsage(version, req, responseTime, error = null) {
        const metrics = this.versionMetrics.get(version);
        if (!metrics) return;
        
        metrics.requests++;
        metrics.lastUsed = new Date();
        
        // Track unique clients
        const clientId = `${req.ip  }-${  req.headers['user-agent'] || 'unknown'}`;
        metrics.uniqueClients.add(clientId);
        
        // Track endpoint usage
        const endpoint = req.path;
        if (!metrics.endpoints.has(endpoint)) {
            metrics.endpoints.set(endpoint, { requests: 0, errors: 0 });
        }
        metrics.endpoints.get(endpoint).requests++;
        
        // Record errors
        if (error) {
            metrics.errors++;
            metrics.endpoints.get(endpoint).errors++;
        }
        
        // Update average response time
        if (responseTime) {
            const alpha = 0.1;
            metrics.avgResponseTime = metrics.avgResponseTime === 0 
                ? responseTime 
                : (alpha * responseTime) + ((1 - alpha) * metrics.avgResponseTime);
        }
    }

    /**
     * Get version usage analytics
     */
    getVersionAnalytics() {
        const analytics = {};
        
        for (const [version, metrics] of this.versionMetrics.entries()) {
            const versionInfo = this.getVersionInfo(version);
            
            analytics[version] = {
                version,
                status: versionInfo.status,
                requests: metrics.requests,
                errors: metrics.errors,
                errorRate: metrics.requests > 0 ? (metrics.errors / metrics.requests * 100).toFixed(2) : 0,
                avgResponseTime: metrics.avgResponseTime.toFixed(2),
                uniqueClients: metrics.uniqueClients.size,
                lastUsed: metrics.lastUsed,
                topEndpoints: this.getTopEndpoints(metrics.endpoints),
                deprecationInfo: {
                    isDeprecated: versionInfo.isDeprecated,
                    deprecatedDate: versionInfo.deprecatedDate,
                    sunsetDate: versionInfo.sunsetDate,
                    migrationPath: versionInfo.migrationPath
                }
            };
        }
        
        return {
            versions: analytics,
            summary: {
                totalVersions: API_CONFIG.supportedVersions.length,
                currentVersion: API_CONFIG.currentVersion,
                deprecatedVersions: Object.values(analytics).filter(v => v.deprecationInfo.isDeprecated).length,
                totalRequests: Object.values(analytics).reduce((sum, v) => sum + v.requests, 0),
                mostUsedVersion: this.getMostUsedVersion(analytics)
            },
            breakingChanges: BREAKING_CHANGES
        };
    }

    /**
     * Get top endpoints for a version
     */
    getTopEndpoints(endpointsMap) {
        return Array.from(endpointsMap.entries())
            .map(([endpoint, stats]) => ({
                endpoint,
                requests: stats.requests,
                errors: stats.errors,
                errorRate: stats.requests > 0 ? (stats.errors / stats.requests * 100).toFixed(2) : 0
            }))
            .sort((a, b) => b.requests - a.requests)
            .slice(0, 10);
    }

    /**
     * Get most used version
     */
    getMostUsedVersion(analytics) {
        let maxRequests = 0;
        let mostUsed = null;
        
        for (const [version, stats] of Object.entries(analytics)) {
            if (stats.requests > maxRequests) {
                maxRequests = stats.requests;
                mostUsed = version;
            }
        }
        
        return mostUsed;
    }

    /**
     * Generate migration recommendations
     */
    generateMigrationRecommendations() {
        const recommendations = [];
        const analytics = this.getVersionAnalytics();
        
        for (const [version, stats] of Object.entries(analytics.versions)) {
            if (stats.deprecationInfo.isDeprecated && stats.requests > 0) {
                const daysUntilSunset = stats.deprecationInfo.sunsetDate 
                    ? Math.ceil((new Date(stats.deprecationInfo.sunsetDate) - new Date()) / (1000 * 60 * 60 * 24))
                    : null;
                
                recommendations.push({
                    priority: daysUntilSunset && daysUntilSunset < 90 ? 'high' : 'medium',
                    version,
                    migrationPath: stats.deprecationInfo.migrationPath,
                    affectedRequests: stats.requests,
                    uniqueClients: stats.uniqueClients,
                    daysUntilSunset,
                    topEndpoints: stats.topEndpoints.slice(0, 5),
                    breakingChanges: BREAKING_CHANGES[stats.deprecationInfo.migrationPath]?.changes || []
                });
            }
        }
        
        return recommendations.sort((a, b) => {
            if (a.priority === 'high' && b.priority !== 'high') return -1;
            if (b.priority === 'high' && a.priority !== 'high') return 1;
            return b.affectedRequests - a.affectedRequests;
        });
    }

    /**
     * Start maintenance tasks
     */
    startMaintenanceTasks() {
        // Clean up deprecation warnings cache every hour
        const cleanupInterval = setInterval(() => {
            const oneHourAgo = Date.now() - (60 * 60 * 1000);
            for (const [key, timestamp] of this.deprecationWarnings.entries()) {
                if (timestamp < oneHourAgo) {
                    this.deprecationWarnings.delete(key);
                }
            }
        }, 60 * 60 * 1000);
        this.intervals.set('deprecation_cleanup', cleanupInterval);
        
        // Generate daily analytics report
        const reportInterval = setInterval(() => {
            this.generateDailyReport();
        }, 24 * 60 * 60 * 1000);
        this.intervals.set('daily_report', reportInterval);
    }

    /**
     * Generate daily analytics report
     */
    generateDailyReport() {
        const analytics = this.getVersionAnalytics();
        const recommendations = this.generateMigrationRecommendations();
        
        this.log.info('Daily API Version Analytics Report', {
            summary: analytics.summary,
            deprecatedVersionUsage: Object.values(analytics.versions)
                .filter(v => v.deprecationInfo.isDeprecated)
                .map(v => ({
                    version: v.version,
                    requests: v.requests,
                    uniqueClients: v.uniqueClients,
                    sunsetDate: v.deprecationInfo.sunsetDate
                })),
            migrationRecommendations: recommendations.length,
            highPriorityMigrations: recommendations.filter(r => r.priority === 'high').length
        });
    }
}

// Global version manager instance
const versionManager = new APIVersionManager();

/**
 * Main API versioning middleware
 */
function apiVersioningMiddleware() {
    return (req, res, next) => {
        const startTime = Date.now();
        
        // Extract and validate API version
        const version = versionManager.extractVersion(req);
        
        if (!versionManager.isValidVersion(version)) {
            return res.status(400).json({
                error: 'Unsupported API version',
                message: `Version ${version} is not supported`,
                supportedVersions: API_CONFIG.supportedVersions,
                currentVersion: API_CONFIG.currentVersion
            });
        }
        
        // Check for deprecation and add warnings
        const versionInfo = versionManager.checkDeprecation(version, req, res);
        
        // Check if version is sunset (no longer supported)
        if (versionInfo.sunsetDate && new Date() > new Date(versionInfo.sunsetDate)) {
            return res.status(410).json({
                error: 'API version no longer supported',
                message: `Version ${version} was sunset on ${versionInfo.sunsetDate}`,
                migrationPath: versionInfo.migrationPath,
                currentVersion: API_CONFIG.currentVersion
            });
        }
        
        // Transform request based on version
        versionManager.transformRequest(req, version);
        
        // Add version headers to response
        res.set({
            'API-Version': version,
            'API-Current-Version': API_CONFIG.currentVersion,
            'API-Supported-Versions': API_CONFIG.supportedVersions.join(', ')
        });
        
        // Intercept response to apply version-specific transformations
        const originalJson = res.json;
        res.json = function(data) {
            const transformedData = versionManager.transformResponse(data, version, req);
            return originalJson.call(this, transformedData);
        };
        
        // Record metrics when response finishes
        res.on('finish', () => {
            const responseTime = Date.now() - startTime;
            const error = res.statusCode >= 400 ? { status: res.statusCode } : null;
            versionManager.recordVersionUsage(version, req, responseTime, error);
        });
        
        next();
    };
}

/**
 * Version-specific route handlers
 */
const versionRoutes = {
    /**
     * Get version information
     */
    getVersionInfo: (req, res) => {
        const version = req.params.version || versionManager.extractVersion(req);
        const versionInfo = versionManager.getVersionInfo(version);
        
        res.json({
            version: versionInfo.version,
            status: versionInfo.status,
            features: versionInfo.features,
            deprecation: {
                isDeprecated: versionInfo.isDeprecated,
                deprecatedDate: versionInfo.deprecatedDate,
                sunsetDate: versionInfo.sunsetDate,
                migrationPath: versionInfo.migrationPath
            },
            breakingChanges: BREAKING_CHANGES[version] || null
        });
    },
    
    /**
     * Get all supported versions
     */
    getSupportedVersions: (req, res) => {
        res.json({
            supportedVersions: API_CONFIG.supportedVersions,
            currentVersion: API_CONFIG.currentVersion,
            defaultVersion: API_CONFIG.defaultVersion,
            versionsInfo: API_CONFIG.supportedVersions.map(v => versionManager.getVersionInfo(v))
        });
    },
    
    /**
     * Get version analytics (admin only)
     */
    getVersionAnalytics: (req, res) => {
        if (!req.user || !req.user.is('admin')) {
            return res.status(403).json({ error: 'Admin access required' });
        }
        
        const analytics = versionManager.getVersionAnalytics();
        const recommendations = versionManager.generateMigrationRecommendations();
        
        res.json({
            ...analytics,
            recommendations
        });
    },
    
    /**
     * Get migration guide for specific version
     */
    getMigrationGuide: (req, res) => {
        const fromVersion = req.params.from;
        const toVersion = req.params.to || API_CONFIG.currentVersion;
        
        if (!versionManager.isValidVersion(fromVersion)) {
            return res.status(400).json({ error: 'Invalid source version' });
        }
        
        if (!versionManager.isValidVersion(toVersion)) {
            return res.status(400).json({ error: 'Invalid target version' });
        }
        
        const breakingChanges = BREAKING_CHANGES[toVersion];
        const fromVersionInfo = versionManager.getVersionInfo(fromVersion);
        const toVersionInfo = versionManager.getVersionInfo(toVersion);
        
        res.json({
            migrationPath: {
                from: fromVersion,
                to: toVersion
            },
            compatibility: semver.gte(toVersion, fromVersion),
            breakingChanges: breakingChanges?.changes || [],
            featureComparison: {
                removed: this.getFeatureDifferences(fromVersionInfo.features, toVersionInfo.features, 'removed'),
                added: this.getFeatureDifferences(fromVersionInfo.features, toVersionInfo.features, 'added'),
                changed: this.getFeatureDifferences(fromVersionInfo.features, toVersionInfo.features, 'changed')
            },
            estimatedEffort: this.calculateMigrationEffort(fromVersion, toVersion),
            recommendedSteps: this.generateMigrationSteps(fromVersion, toVersion)
        });
    },
    
    /**
     * Calculate migration effort
     */
    calculateMigrationEffort(fromVersion, toVersion) {
        const breakingChanges = BREAKING_CHANGES[toVersion]?.changes || [];
        const majorVersionDiff = semver.major(toVersion) - semver.major(fromVersion);
        
        let effort = 'low';
        if (majorVersionDiff > 0 || breakingChanges.length > 5) {
            effort = 'high';
        } else if (breakingChanges.length > 2) {
            effort = 'medium';
        }
        
        return {
            level: effort,
            breakingChangesCount: breakingChanges.length,
            majorVersionChange: majorVersionDiff > 0,
            estimatedHours: breakingChanges.length * 2 + (majorVersionDiff * 8)
        };
    },
    
    /**
     * Generate migration steps
     */
    generateMigrationSteps(fromVersion, toVersion) {
        const steps = [
            'Review breaking changes documentation',
            'Update API version headers in client applications',
            'Test endpoint compatibility in staging environment'
        ];
        
        const breakingChanges = BREAKING_CHANGES[toVersion];
        if (breakingChanges) {
            breakingChanges.changes.forEach(change => {
                if (change.type === 'field_removed') {
                    steps.push(`Remove usage of deprecated field: ${change.field}`);
                } else if (change.type === 'response_format') {
                    steps.push(`Update response parsing for: ${change.endpoint}`);
                } else if (change.type === 'authentication') {
                    steps.push(`Update authentication for: ${change.endpoint}`);
                }
            });
        }
        
        steps.push('Deploy changes to production');
        steps.push('Monitor for errors and performance impacts');
        
        return steps;
    },
    
    /**
     * Get feature differences between versions
     */
    getFeatureDifferences(oldFeatures, newFeatures, type) {
        const differences = [];
        
        if (type === 'removed') {
            for (const [key, value] of Object.entries(oldFeatures)) {
                if (!(key in newFeatures) || (!newFeatures[key] && value)) {
                    differences.push({ feature: key, oldValue: value });
                }
            }
        } else if (type === 'added') {
            for (const [key, value] of Object.entries(newFeatures)) {
                if (!(key in oldFeatures) || (!oldFeatures[key] && value)) {
                    differences.push({ feature: key, newValue: value });
                }
            }
        } else if (type === 'changed') {
            for (const [key, value] of Object.entries(newFeatures)) {
                if (key in oldFeatures && oldFeatures[key] !== value) {
                    differences.push({ feature: key, oldValue: oldFeatures[key], newValue: value });
                }
            }
        }
        
        return differences;
    }
};

module.exports = {
    APIVersionManager,
    apiVersioningMiddleware,
    versionRoutes,
    versionManager,
    API_CONFIG,
    BREAKING_CHANGES
};