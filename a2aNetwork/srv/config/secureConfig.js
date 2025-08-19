/**
 * @fileoverview Secure Configuration Management
 * @description Secure handling of configuration with encryption and validation
 * @module secure-config
 */

const cds = require('@sap/cds');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

/**
 * Configuration security levels
 */
const SecurityLevels = {
    PUBLIC: 'PUBLIC',           // No encryption needed
    INTERNAL: 'INTERNAL',       // Internal use, basic encryption
    CONFIDENTIAL: 'CONFIDENTIAL', // Sensitive data, strong encryption
    SECRET: 'SECRET'            // Critical secrets, highest encryption
};

/**
 * Secure configuration manager
 */
class SecureConfigManager {
    constructor() {
        this.encryptionKey = this.getOrCreateEncryptionKey();
        this.configCache = new Map();
        this.configMetadata = new Map();
        this.validators = new Map();
        this.changeListeners = new Set();
        
        // Register default validators
        this.registerDefaultValidators();
    }

    /**
     * Get or create encryption key for configuration
     */
    getOrCreateEncryptionKey() {
        const keyEnv = process.env.CONFIG_ENCRYPTION_KEY;
        
        if (keyEnv) {
            return Buffer.from(keyEnv, 'hex');
        }
        
        if (process.env.NODE_ENV === 'production') {
            throw new Error('CONFIG_ENCRYPTION_KEY environment variable required in production');
        }
        
        // Development only - generate temporary key
        cds.log('secure-config').warn('Using temporary encryption key for development');
        return crypto.randomBytes(32);
    }

    /**
     * Register configuration value with security level
     */
    registerConfig(key, defaultValue, securityLevel = SecurityLevels.INTERNAL, validator = null) {
        this.configMetadata.set(key, {
            securityLevel,
            validator: validator || this.getDefaultValidator(key),
            defaultValue,
            lastUpdated: new Date(),
            accessCount: 0
        });

        // Set default value if not exists
        if (!this.configCache.has(key)) {
            this.setConfig(key, defaultValue, false);
        }
    }

    /**
     * Get configuration value with decryption
     */
    getConfig(key, defaultValue = null) {
        const metadata = this.configMetadata.get(key);
        
        if (metadata) {
            metadata.accessCount++;
        }

        // Check cache first
        if (this.configCache.has(key)) {
            const cachedValue = this.configCache.get(key);
            return this.decryptValue(cachedValue, metadata?.securityLevel);
        }

        // Try environment variable
        const envValue = process.env[key];
        if (envValue !== undefined) {
            const processedValue = this.processEnvValue(envValue, metadata?.securityLevel);
            this.configCache.set(key, processedValue);
            return this.decryptValue(processedValue, metadata?.securityLevel);
        }

        // Return default
        return defaultValue || metadata?.defaultValue;
    }

    /**
     * Set configuration value with encryption
     */
    setConfig(key, value, persistent = true) {
        const metadata = this.configMetadata.get(key);
        
        // Validate value
        if (metadata?.validator) {
            const validation = metadata.validator(value);
            if (!validation.valid) {
                throw new Error(`Invalid configuration value for ${key}: ${validation.error}`);
            }
        }

        // Encrypt based on security level
        const encryptedValue = this.encryptValue(value, metadata?.securityLevel);
        
        // Update cache
        this.configCache.set(key, encryptedValue);
        
        // Update metadata
        if (metadata) {
            metadata.lastUpdated = new Date();
        }

        // Notify listeners
        this.notifyConfigChange(key, value);

        // Persist if requested and not in test environment
        if (persistent && process.env.NODE_ENV !== 'test') {
            this.persistConfig(key, encryptedValue);
        }

        cds.log('secure-config').info('Configuration updated', { 
            key, 
            securityLevel: metadata?.securityLevel 
        });
    }

    /**
     * Encrypt value based on security level
     */
    encryptValue(value, securityLevel) {
        if (!securityLevel || securityLevel === SecurityLevels.PUBLIC) {
            return value;
        }

        const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
        
        if (securityLevel === SecurityLevels.INTERNAL) {
            // Basic encryption for internal values
            return this.simpleEncrypt(stringValue);
        } else {
            // Strong encryption for confidential and secret values
            return this.strongEncrypt(stringValue);
        }
    }

    /**
     * Decrypt value based on security level
     */
    decryptValue(encryptedValue, securityLevel) {
        if (!securityLevel || securityLevel === SecurityLevels.PUBLIC) {
            return encryptedValue;
        }

        try {
            if (securityLevel === SecurityLevels.INTERNAL) {
                return this.simpleDecrypt(encryptedValue);
            } else {
                return this.strongDecrypt(encryptedValue);
            }
        } catch (error) {
            cds.log('secure-config').error('Failed to decrypt configuration value', error);
            return null;
        }
    }

    /**
     * Simple encryption for internal values
     * SECURITY FIX: Use modern crypto functions instead of deprecated createCipher
     */
    simpleEncrypt(value) {
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipheriv('aes-256-cbc', this.encryptionKey, iv);
        let encrypted = cipher.update(value, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        return `simple:${iv.toString('hex')}:${encrypted}`;
    }

    /**
     * Simple decryption for internal values
     * SECURITY FIX: Use modern crypto functions instead of deprecated createDecipher
     */
    simpleDecrypt(encryptedValue) {
        if (!encryptedValue.startsWith('simple:')) {
            return encryptedValue; // Not encrypted
        }
        
        const parts = encryptedValue.substring(7).split(':');
        if (parts.length !== 2) {
            throw new Error('Invalid encrypted value format');
        }
        
        const iv = Buffer.from(parts[0], 'hex');
        const encrypted = parts[1];
        const decipher = crypto.createDecipheriv('aes-256-cbc', this.encryptionKey, iv);
        let decrypted = decipher.update(encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    }

    /**
     * Strong encryption for confidential/secret values
     */
    strongEncrypt(value) {
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipherGCM('aes-256-gcm', this.encryptionKey, iv);
        
        let encrypted = cipher.update(value, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        
        const authTag = cipher.getAuthTag();
        
        return `strong:${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
    }

    /**
     * Strong decryption for confidential/secret values
     */
    strongDecrypt(encryptedValue) {
        if (!encryptedValue.startsWith('strong:')) {
            return encryptedValue; // Not encrypted
        }
        
        const parts = encryptedValue.substring(7).split(':');
        if (parts.length !== 3) {
            throw new Error('Invalid encrypted value format');
        }
        
        const [ivHex, authTagHex, encrypted] = parts;
        const iv = Buffer.from(ivHex, 'hex');
        const authTag = Buffer.from(authTagHex, 'hex');
        
        const decipher = crypto.createDecipherGCM('aes-256-gcm', this.encryptionKey, iv);
        decipher.setAuthTag(authTag);
        
        let decrypted = decipher.update(encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        return decrypted;
    }

    /**
     * Process environment variable value
     */
    processEnvValue(value, securityLevel) {
        // Handle different data types
        if (value === 'true') return true;
        if (value === 'false') return false;
        if (/^\d+$/.test(value)) return parseInt(value);
        if (/^\d+\.\d+$/.test(value)) return parseFloat(value);
        
        // Try to parse as JSON
        try {
            return JSON.parse(value);
        } catch {
            // Return as string
            return value;
        }
    }

    /**
     * Register default validators for common configuration types
     */
    registerDefaultValidators() {
        // URL validator
        this.validators.set('url', (value) => {
            try {
                new URL(value);
                return { valid: true };
            } catch {
                return { valid: false, error: 'Invalid URL format' };
            }
        });

        // Port validator
        this.validators.set('port', (value) => {
            const port = parseInt(value);
            if (isNaN(port) || port < 1 || port > 65535) {
                return { valid: false, error: 'Port must be between 1 and 65535' };
            }
            return { valid: true };
        });

        // Ethereum address validator
        this.validators.set('ethereum_address', (value) => {
            if (!/^0x[a-fA-F0-9]{40}$/.test(value)) {
                return { valid: false, error: 'Invalid Ethereum address format' };
            }
            return { valid: true };
        });

        // Private key validator
        this.validators.set('private_key', (value) => {
            if (!/^0x[a-fA-F0-9]{64}$/.test(value)) {
                return { valid: false, error: 'Invalid private key format' };
            }
            return { valid: true };
        });

        // Non-empty string validator
        this.validators.set('non_empty_string', (value) => {
            if (typeof value !== 'string' || value.trim().length === 0) {
                return { valid: false, error: 'Value must be a non-empty string' };
            }
            return { valid: true };
        });
    }

    /**
     * Get default validator for configuration key
     */
    getDefaultValidator(key) {
        const lowerKey = key.toLowerCase();
        
        if (lowerKey.includes('url')) return this.validators.get('url');
        if (lowerKey.includes('port')) return this.validators.get('port');
        if (lowerKey.includes('address') && lowerKey.includes('eth')) return this.validators.get('ethereum_address');
        if (lowerKey.includes('private_key') || lowerKey.includes('privatekey')) return this.validators.get('private_key');
        
        return this.validators.get('non_empty_string');
    }

    /**
     * Persist configuration to secure storage
     */
    async persistConfig(key, encryptedValue) {
        try {
            const configDir = path.join(process.cwd(), '.secure-config');
            await fs.mkdir(configDir, { recursive: true });
            
            const configFile = path.join(configDir, `${key}.enc`);
            await fs.writeFile(configFile, encryptedValue, 'utf8');
            
        } catch (error) {
            cds.log('secure-config').error('Failed to persist configuration', error);
        }
    }

    /**
     * Load persisted configuration
     */
    async loadPersistedConfig() {
        try {
            const configDir = path.join(process.cwd(), '.secure-config');
            const files = await fs.readdir(configDir);
            
            for (const file of files) {
                if (file.endsWith('.enc')) {
                    const key = file.replace('.enc', '');
                    const filePath = path.join(configDir, file);
                    const encryptedValue = await fs.readFile(filePath, 'utf8');
                    
                    if (!this.configCache.has(key)) {
                        this.configCache.set(key, encryptedValue);
                    }
                }
            }
            
        } catch (error) {
            // Config directory doesn't exist or other error - this is fine
            cds.log('secure-config').debug('No persisted configuration found');
        }
    }

    /**
     * Add configuration change listener
     */
    addChangeListener(callback) {
        this.changeListeners.add(callback);
    }

    /**
     * Remove configuration change listener
     */
    removeChangeListener(callback) {
        this.changeListeners.delete(callback);
    }

    /**
     * Notify listeners of configuration changes
     */
    notifyConfigChange(key, value) {
        for (const listener of this.changeListeners) {
            try {
                listener(key, value);
            } catch (error) {
                cds.log('secure-config').error('Error in config change listener', error);
            }
        }
    }

    /**
     * Get configuration metadata
     */
    getConfigMetadata(key) {
        return this.configMetadata.get(key);
    }

    /**
     * Get all configuration keys (for administration)
     */
    getAllConfigKeys() {
        return Array.from(this.configMetadata.keys());
    }

    /**
     * Validate all configurations
     */
    validateAllConfigs() {
        const results = {};
        
        for (const [key, metadata] of this.configMetadata.entries()) {
            if (metadata.validator) {
                const value = this.getConfig(key);
                results[key] = metadata.validator(value);
            }
        }
        
        return results;
    }

    /**
     * Get configuration statistics
     */
    getConfigStats() {
        const stats = {
            totalConfigs: this.configMetadata.size,
            configsBySecurityLevel: {},
            configsWithValidators: 0,
            totalAccesses: 0
        };
        
        for (const [key, metadata] of this.configMetadata.entries()) {
            // Count by security level
            const level = metadata.securityLevel;
            stats.configsBySecurityLevel[level] = (stats.configsBySecurityLevel[level] || 0) + 1;
            
            // Count validators
            if (metadata.validator) {
                stats.configsWithValidators++;
            }
            
            // Sum access counts
            stats.totalAccesses += metadata.accessCount;
        }
        
        return stats;
    }
}

// Global secure config manager
const globalConfigManager = new SecureConfigManager();

// Initialize common blockchain configurations
globalConfigManager.registerConfig(
    'BLOCKCHAIN_RPC_URL', 
    'http://localhost:8545', 
    SecurityLevels.CONFIDENTIAL,
    globalConfigManager.validators.get('url')
);

globalConfigManager.registerConfig(
    'BLOCKCHAIN_PRIVATE_KEY', 
    null, 
    SecurityLevels.SECRET,
    globalConfigManager.validators.get('private_key')
);

globalConfigManager.registerConfig(
    'BLOCKCHAIN_GAS_LIMIT', 
    500000, 
    SecurityLevels.INTERNAL
);

globalConfigManager.registerConfig(
    'BLOCKCHAIN_GAS_PRICE_GWEI', 
    20, 
    SecurityLevels.INTERNAL
);

// Load persisted configurations on startup
globalConfigManager.loadPersistedConfig().catch(error => {
    cds.log('secure-config').warn('Failed to load persisted configurations', error);
});

module.exports = {
    SecureConfigManager,
    SecurityLevels,
    getConfig: (key, defaultValue) => globalConfigManager.getConfig(key, defaultValue),
    setConfig: (key, value, persistent) => globalConfigManager.setConfig(key, value, persistent),
    registerConfig: (key, defaultValue, securityLevel, validator) => 
        globalConfigManager.registerConfig(key, defaultValue, securityLevel, validator),
    validateAllConfigs: () => globalConfigManager.validateAllConfigs(),
    getConfigStats: () => globalConfigManager.getConfigStats(),
    addChangeListener: (callback) => globalConfigManager.addChangeListener(callback)
};