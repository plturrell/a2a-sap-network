/**
 * Shared Resource Manager for A2A Platform
 * Implements resource management and configuration synchronization
 * Test Case: TC-COM-LPD-003
 */

const crypto = require('crypto');
const EventEmitter = require('events');

class SharedResourceManager extends EventEmitter {
    constructor(config) {
        super();

        this.config = {
            syncInterval: config.syncInterval || 60000, // 60 seconds
            cacheExpiry: config.cacheExpiry || 300000, // 5 minutes
            conflictResolution: config.conflictResolution || 'last-writer-wins',
            redisClient: config.redisClient,
            storageBackend: config.storageBackend || 'filesystem',
            cdnUrl: config.cdnUrl,
            ...config
        };

        this.resources = new Map();
        this.configurations = new Map();
        this.featureFlags = new Map();
        this.cache = new Map();
        this.syncTimers = new Map();
        this.conflictQueue = [];

        this.initializeResourceManager();
    }

    /**
     * Initialize resource management system
     */
    initializeResourceManager() {
        // Start synchronization intervals
        this.startSynchronization();

        // Set up event handlers
        this.on('resource-updated', this.handleResourceUpdate.bind(this));
        this.on('conflict-detected', this.handleConflict.bind(this));
        this.on('sync-complete', this.handleSyncComplete.bind(this));
    }

    /**
     * Synchronize configuration across applications
     * @param {string} configKey - Configuration key
     * @param {Object} configValue - Configuration value
     * @param {Object} metadata - Metadata for the configuration
     * @returns {Promise<Object>}
     */
    async syncConfiguration(configKey, configValue, metadata = {}) {
        try {
            const timestamp = Date.now();
            const configEntry = {
                key: configKey,
                value: configValue,
                metadata: {
                    ...metadata,
                    lastModified: timestamp,
                    version: this.generateVersion(),
                    checksum: this.calculateChecksum(configValue)
                }
            };

            // Check for conflicts
            const existingConfig = await this.getConfiguration(configKey);
            if (existingConfig && existingConfig.metadata.lastModified > metadata.lastModified) {
                return this.resolveConflict(existingConfig, configEntry);
            }

            // Store configuration
            await this.storeConfiguration(configEntry);

            // Propagate to all applications
            await this.propagateConfiguration(configEntry);

            // Update cache
            this.updateCache(`config:${configKey}`, configEntry);

            // Emit update event
            this.emit('configuration-synced', configEntry);

            return {
                success: true,
                key: configKey,
                version: configEntry.metadata.version,
                syncTime: Date.now() - timestamp
            };

        } catch (error) {
            console.error('Configuration sync failed:', error);
            throw error;
        }
    }

    /**
     * Manage shared assets across applications
     * @param {string} assetId - Asset identifier
     * @param {Buffer|string} assetData - Asset content
     * @param {Object} metadata - Asset metadata
     * @returns {Promise<Object>}
     */
    async manageSharedAssets(assetId, assetData, metadata = {}) {
        try {
            const assetEntry = {
                id: assetId,
                type: metadata.type || this.detectAssetType(assetId),
                size: Buffer.byteLength(assetData),
                checksum: this.calculateChecksum(assetData),
                metadata: {
                    ...metadata,
                    uploadedAt: new Date().toISOString(),
                    version: this.generateVersion()
                }
            };

            // Store asset
            const storageUrl = await this.storeAsset(assetId, assetData, assetEntry);
            assetEntry.url = storageUrl;

            // Distribute to CDN if configured
            if (this.config.cdnUrl) {
                assetEntry.cdnUrl = await this.distributeToCDN(assetId, assetData);
            }

            // Register asset
            this.resources.set(assetId, assetEntry);

            // Notify all applications
            await this.notifyAssetUpdate(assetEntry);

            return {
                success: true,
                assetId,
                url: assetEntry.url,
                cdnUrl: assetEntry.cdnUrl,
                version: assetEntry.metadata.version
            };

        } catch (error) {
            console.error('Asset management failed:', error);
            throw error;
        }
    }

    /**
     * Validate consistency across all shared resources
     * @returns {Promise<Object>}
     */
    async validateConsistency() {
        const report = {
            timestamp: new Date().toISOString(),
            configurations: { total: 0, valid: 0, invalid: [] },
            assets: { total: 0, valid: 0, invalid: [] },
            featureFlags: { total: 0, valid: 0, invalid: [] }
        };

        try {
            // Validate configurations
            for (const [key, config] of this.configurations) {
                report.configurations.total++;
                const isValid = await this.validateConfiguration(config);
                if (isValid) {
                    report.configurations.valid++;
                } else {
                    report.configurations.invalid.push(key);
                }
            }

            // Validate assets
            for (const [id, asset] of this.resources) {
                report.assets.total++;
                const isValid = await this.validateAsset(asset);
                if (isValid) {
                    report.assets.valid++;
                } else {
                    report.assets.invalid.push(id);
                }
            }

            // Validate feature flags
            for (const [flag, value] of this.featureFlags) {
                report.featureFlags.total++;
                const isValid = this.validateFeatureFlag(flag, value);
                if (isValid) {
                    report.featureFlags.valid++;
                } else {
                    report.featureFlags.invalid.push(flag);
                }
            }

            report.overallHealth = this.calculateOverallHealth(report);

            return report;

        } catch (error) {
            console.error('Consistency validation failed:', error);
            throw error;
        }
    }

    /**
     * Get configuration by key
     * @param {string} key - Configuration key
     * @returns {Promise<Object>}
     */
    async getConfiguration(key) {
        // Check cache first
        const cached = this.getFromCache(`config:${key}`);
        if (cached) return cached;

        // Check local store
        if (this.configurations.has(key)) {
            return this.configurations.get(key);
        }

        // Check persistent store
        if (this.config.redisClient) {
            const data = await this.config.redisClient.get(`config:${key}`);
            if (data) {
                const config = JSON.parse(data);
                this.configurations.set(key, config);
                return config;
            }
        }

        return null;
    }

    /**
     * Store configuration in persistent storage
     * @param {Object} configEntry - Configuration entry
     */
    async storeConfiguration(configEntry) {
        // Store locally
        this.configurations.set(configEntry.key, configEntry);

        // Store in Redis if available
        if (this.config.redisClient) {
            await this.config.redisClient.setex(
                `config:${configEntry.key}`,
                this.config.cacheExpiry / 1000,
                JSON.stringify(configEntry)
            );
        }

        // Store in database
        if (this.config.database) {
            await this.config.database.saveConfiguration(configEntry);
        }
    }

    /**
     * Propagate configuration to all applications
     * @param {Object} configEntry - Configuration to propagate
     */
    async propagateConfiguration(configEntry) {
        const applications = ['network', 'agents', 'launchpad'];
        const propagationTasks = [];

        for (const app of applications) {
            propagationTasks.push(
                this.sendConfigurationUpdate(app, configEntry)
            );
        }

        await Promise.allSettled(propagationTasks);
    }

    /**
     * Send configuration update to specific application
     * @param {string} appId - Application ID
     * @param {Object} configEntry - Configuration entry
     */
    async sendConfigurationUpdate(appId, configEntry) {
        const endpoint = this.config.applicationEndpoints?.[appId];
        if (!endpoint) return;

        try {
            await blockchainClient.sendMessage(`${endpoint}/api/config/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config: configEntry,
                    source: 'shared-resource-manager'
                })
            });
        } catch (error) {
            console.error(`Failed to update ${appId}:`, error);
        }
    }

    /**
     * Handle resource update events
     * @param {Object} resource - Updated resource
     */
    handleResourceUpdate(resource) {
        // Clear related cache entries
        this.clearCache(resource.id);

        // Trigger dependent updates
        this.updateDependentResources(resource);
    }

    /**
     * Resolve configuration conflicts
     * @param {Object} existing - Existing configuration
     * @param {Object} incoming - Incoming configuration
     * @returns {Object} Resolved configuration
     */
    async resolveConflict(existing, incoming) {
        const conflictEntry = {
            id: crypto.randomBytes(16).toString('hex'),
            type: 'configuration',
            existing,
            incoming,
            timestamp: Date.now()
        };

        this.conflictQueue.push(conflictEntry);
        this.emit('conflict-detected', conflictEntry);

        switch (this.config.conflictResolution) {
            case 'last-writer-wins':
                return incoming;

            case 'merge':
                return this.mergeConfigurations(existing, incoming);

            case 'manual':
                // Queue for manual resolution
                await this.queueForManualResolution(conflictEntry);
                return existing; // Keep existing until resolved

            default:
                return incoming;
        }
    }

    /**
     * Merge two configurations
     * @param {Object} existing - Existing configuration
     * @param {Object} incoming - Incoming configuration
     * @returns {Object} Merged configuration
     */
    mergeConfigurations(existing, incoming) {
        const merged = {
            key: existing.key,
            value: this.deepMerge(existing.value, incoming.value),
            metadata: {
                ...incoming.metadata,
                mergedFrom: [existing.metadata.version, incoming.metadata.version],
                mergedAt: Date.now()
            }
        };

        return merged;
    }

    /**
     * Deep merge two objects
     * @param {Object} target - Target object
     * @param {Object} source - Source object
     * @returns {Object} Merged object
     */
    deepMerge(target, source) {
        const output = { ...target };

        if (this.isObject(target) && this.isObject(source)) {
            Object.keys(source).forEach(key => {
                if (this.isObject(source[key])) {
                    if (!(key in target)) {
                        output[key] = source[key];
                    } else {
                        output[key] = this.deepMerge(target[key], source[key]);
                    }
                } else {
                    output[key] = source[key];
                }
            });
        }

        return output;
    }

    /**
     * Check if value is an object
     * @param {*} item - Item to check
     * @returns {boolean}
     */
    isObject(item) {
        return item && typeof item === 'object' && !Array.isArray(item);
    }

    /**
     * Store asset in configured storage backend
     * @param {string} assetId - Asset ID
     * @param {Buffer|string} assetData - Asset data
     * @param {Object} metadata - Asset metadata
     * @returns {Promise<string>} Storage URL
     */
    async storeAsset(assetId, assetData, metadata) {
        switch (this.config.storageBackend) {
            case 'filesystem':
                return this.storeAssetFilesystem(assetId, assetData, metadata);

            case 's3':
                return this.storeAssetS3(assetId, assetData, metadata);

            case 'azure':
                return this.storeAssetAzure(assetId, assetData, metadata);

            default:
                throw new Error(`Unknown storage backend: ${this.config.storageBackend}`);
        }
    }

    /**
     * Store asset in filesystem
     * @param {string} assetId - Asset ID
     * @param {Buffer|string} assetData - Asset data
     * @param {Object} metadata - Asset metadata
     * @returns {Promise<string>} File path
     */
    async storeAssetFilesystem(assetId, assetData, metadata) {
        const fs = require('fs').promises;
        const path = require('path');

        const assetPath = path.join(
            this.config.assetStoragePath || './assets',
            metadata.type || 'misc',
            assetId
        );

        await fs.mkdir(path.dirname(assetPath), { recursive: true });
        await fs.writeFile(assetPath, assetData);

        return `file://${assetPath}`;
    }

    /**
     * Distribute asset to CDN
     * @param {string} assetId - Asset ID
     * @param {Buffer|string} assetData - Asset data
     * @returns {Promise<string>} CDN URL
     */
    async distributeToCDN(assetId, assetData) {
        // Placeholder for CDN distribution
        // Would integrate with actual CDN service
        return `${this.config.cdnUrl}/assets/${assetId}`;
    }

    /**
     * Notify applications about asset update
     * @param {Object} assetEntry - Asset entry
     */
    async notifyAssetUpdate(assetEntry) {
        const notification = {
            type: 'asset-update',
            asset: assetEntry,
            timestamp: Date.now()
        };

        // Broadcast to all connected applications
        this.emit('asset-updated', notification);

        // Send to message queue if configured
        if (this.config.messageQueue) {
            await this.config.messageQueue.publish('asset-updates', notification);
        }
    }

    /**
     * Manage feature flags
     * @param {string} flagName - Feature flag name
     * @param {boolean|Object} flagValue - Flag value or configuration
     * @returns {Object} Feature flag status
     */
    async setFeatureFlag(flagName, flagValue) {
        const flagEntry = {
            name: flagName,
            value: flagValue,
            updatedAt: Date.now(),
            updatedBy: this.config.systemId || 'system'
        };

        // Store flag
        this.featureFlags.set(flagName, flagEntry);

        // Propagate to all applications
        await this.propagateFeatureFlag(flagEntry);

        return {
            success: true,
            flag: flagName,
            value: flagValue,
            propagated: true
        };
    }

    /**
     * Get feature flag value
     * @param {string} flagName - Feature flag name
     * @returns {boolean|Object} Flag value
     */
    getFeatureFlag(flagName) {
        const flag = this.featureFlags.get(flagName);
        return flag ? flag.value : false;
    }

    /**
     * Propagate feature flag to applications
     * @param {Object} flagEntry - Feature flag entry
     */
    async propagateFeatureFlag(flagEntry) {
        const propagationTasks = [];
        const applications = ['network', 'agents', 'launchpad'];

        for (const app of applications) {
            propagationTasks.push(
                this.sendFeatureFlagUpdate(app, flagEntry)
            );
        }

        await Promise.allSettled(propagationTasks);
    }

    /**
     * Send feature flag update to application
     * @param {string} appId - Application ID
     * @param {Object} flagEntry - Flag entry
     */
    async sendFeatureFlagUpdate(appId, flagEntry) {
        const endpoint = this.config.applicationEndpoints?.[appId];
        if (!endpoint) return;

        try {
            await blockchainClient.sendMessage(`${endpoint}/api/feature-flags/update`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    flag: flagEntry,
                    source: 'shared-resource-manager'
                })
            });
        } catch (error) {
            console.error(`Failed to update feature flag in ${appId}:`, error);
        }
    }

    /**
     * Start synchronization processes
     */
    startSynchronization() {
        // Configuration sync
        const configSyncTimer = setInterval(() => {
            this.syncAllConfigurations();
        }, this.config.syncInterval);
        this.syncTimers.set('config', configSyncTimer);

        // Asset sync
        const assetSyncTimer = setInterval(() => {
            this.syncAllAssets();
        }, this.config.syncInterval * 2);
        this.syncTimers.set('assets', assetSyncTimer);

        // Feature flag sync
        const flagSyncTimer = setInterval(() => {
            this.syncAllFeatureFlags();
        }, this.config.syncInterval / 2);
        this.syncTimers.set('flags', flagSyncTimer);
    }

    /**
     * Sync all configurations
     */
    async syncAllConfigurations() {
        for (const [key, config] of this.configurations) {
            await this.syncConfiguration(key, config.value, config.metadata);
        }
        this.emit('sync-complete', { type: 'configurations' });
    }

    /**
     * Sync all assets
     */
    async syncAllAssets() {
        for (const [id, asset] of this.resources) {
            await this.validateAsset(asset);
        }
        this.emit('sync-complete', { type: 'assets' });
    }

    /**
     * Sync all feature flags
     */
    async syncAllFeatureFlags() {
        for (const [name, flag] of this.featureFlags) {
            await this.propagateFeatureFlag(flag);
        }
        this.emit('sync-complete', { type: 'feature-flags' });
    }

    /**
     * Calculate checksum for data
     * @param {*} data - Data to checksum
     * @returns {string} Checksum
     */
    calculateChecksum(data) {
        const content = typeof data === 'object' ? JSON.stringify(data) : String(data);
        return crypto.createHash('md5').update(content).digest('hex');
    }

    /**
     * Generate version identifier
     * @returns {string} Version ID
     */
    generateVersion() {
        return `v${Date.now()}.${Math.random().toString(36).substr(2, 4)}`;
    }

    /**
     * Detect asset type from ID/filename
     * @param {string} assetId - Asset ID
     * @returns {string} Asset type
     */
    detectAssetType(assetId) {
        const ext = assetId.split('.').pop().toLowerCase();
        const typeMap = {
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image',
            'gif': 'image',
            'svg': 'image',
            'css': 'stylesheet',
            'js': 'script',
            'json': 'data',
            'xml': 'data'
        };
        return typeMap[ext] || 'other';
    }

    /**
     * Cache management
     */
    updateCache(key, value) {
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    getFromCache(key) {
        const cached = this.cache.get(key);
        if (cached && (Date.now() - cached.timestamp) < this.config.cacheExpiry) {
            return cached.value;
        }
        this.cache.delete(key);
        return null;
    }

    clearCache(pattern) {
        for (const key of this.cache.keys()) {
            if (key.includes(pattern)) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Validate configuration entry
     * @param {Object} config - Configuration to validate
     * @returns {Promise<boolean>} Validation result
     */
    async validateConfiguration(config) {
        if (!config.key || !config.value) return false;
        if (!config.metadata?.checksum) return false;

        const currentChecksum = this.calculateChecksum(config.value);
        return currentChecksum === config.metadata.checksum;
    }

    /**
     * Validate asset entry
     * @param {Object} asset - Asset to validate
     * @returns {Promise<boolean>} Validation result
     */
    async validateAsset(asset) {
        if (!asset.id || !asset.url) return false;

        // Check if asset is accessible
        try {
            if (asset.url.startsWith('file://')) {
                const fs = require('fs').promises;
                await fs.access(asset.url.replace('file://', ''));
                return true;
            } else {
                const response = await blockchainClient.sendMessage(asset.url, { method: 'HEAD' });
                return response.ok;
            }
        } catch {
            return false;
        }
    }

    /**
     * Validate feature flag
     * @param {string} flag - Flag name
     * @param {Object} value - Flag value
     * @returns {boolean} Validation result
     */
    validateFeatureFlag(flag, value) {
        return flag && value !== undefined && value !== null;
    }

    /**
     * Calculate overall health score
     * @param {Object} report - Validation report
     * @returns {number} Health score (0-100)
     */
    calculateOverallHealth(report) {
        const configHealth = (report.configurations.valid / report.configurations.total) || 0;
        const assetHealth = (report.assets.valid / report.assets.total) || 0;
        const flagHealth = (report.featureFlags.valid / report.featureFlags.total) || 0;

        return Math.round((configHealth + assetHealth + flagHealth) / 3 * 100);
    }

    /**
     * Update dependent resources
     * @param {Object} resource - Updated resource
     */
    updateDependentResources(resource) {
        // Implementation would track and update dependent resources
        this.emit('dependent-update-required', resource);
    }

    /**
     * Queue conflict for manual resolution
     * @param {Object} conflict - Conflict entry
     */
    async queueForManualResolution(conflict) {
        if (this.config.conflictQueue) {
            await this.config.conflictQueue.publish('conflicts', conflict);
        }
    }

    /**
     * Handle sync completion
     * @param {Object} event - Sync event
     */
    handleSyncComplete(event) {
        console.log(`Sync completed for ${event.type}`);
    }

    /**
     * Handle conflict detection
     * @param {Object} conflict - Conflict details
     */
    handleConflict(conflict) {
        console.warn('Conflict detected:', conflict);
    }

    /**
     * Stop all synchronization processes
     */
    stopSynchronization() {
        for (const [key, timer] of this.syncTimers) {
            clearInterval(timer);
            this.syncTimers.delete(key);
        }
    }

    /**
     * Get resource statistics
     * @returns {Object} Resource statistics
     */
    getStatistics() {
        return {
            configurations: this.configurations.size,
            assets: this.resources.size,
            featureFlags: this.featureFlags.size,
            cacheEntries: this.cache.size,
            pendingConflicts: this.conflictQueue.length,
            activeSyncProcesses: this.syncTimers.size
        };
    }
}

module.exports = SharedResourceManager;