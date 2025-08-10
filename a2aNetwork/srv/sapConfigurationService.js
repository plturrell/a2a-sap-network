/**
 * @fileoverview SAP Configuration Service Implementation
 * @description Manages application configuration, network settings, and security parameters
 * @module sapConfigurationService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;

/**
 * Configuration Service Implementation
 * Handles all configuration management operations
 */
module.exports = cds.service.impl(async function () {
    
    const { NetworkSettings, SecuritySettings, ApplicationSettings, SettingsAuditLog, AutoSavedSettings } = this.entities;

    /**
     * Get current network settings
     */
    this.on('getNetworkSettings', async (req) => {
        try {
            // Get the most recent active network settings
            const settings = await SELECT.one.from(NetworkSettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            if (settings) {
                return settings;
            }

            // Return default settings if none exist
            const defaultSettings = {
                ID: cds.utils.uuid(),
                network: process.env.DEFAULT_NETWORK || 'localhost',
                rpcUrl: process.env.RPC_URL || 'http://localhost:8545',
                chainId: parseInt(process.env.CHAIN_ID || '31337'),
                contractAddress: process.env.CONTRACT_ADDRESS || '0x5FbDB2315678afecb367f032d93F642f64180aa3',
                isActive: true,
                version: 1,
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            // Save default settings to database
            await INSERT.into(NetworkSettings).entries(defaultSettings);
            cds.log('config').info('Created default network settings');

            return defaultSettings;

        } catch (error) {
            cds.log('config').error('Failed to get network settings:', error);
            req.error(500, 'Failed to retrieve network settings', error.message);
        }
    });

    /**
     * Update network settings
     */
    this.on('updateNetworkSettings', async (req) => {
        try {
            const { settings } = req.data;
            const userId = req.user?.id || 'system';

            // Get current settings for audit log
            const currentSettings = await SELECT.one.from(NetworkSettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            // Deactivate current settings
            if (currentSettings) {
                await UPDATE(NetworkSettings)
                    .set({ isActive: false })
                    .where({ ID: currentSettings.ID });

                // Create audit log entry
                await this._createAuditLog('NETWORK_SETTINGS', 'UPDATE', 
                    JSON.stringify(currentSettings), JSON.stringify(settings), userId);
            }

            // Create new settings version
            const newSettings = {
                ID: cds.utils.uuid(),
                network: settings.network,
                rpcUrl: settings.rpcUrl,
                chainId: settings.chainId,
                contractAddress: settings.contractAddress,
                isActive: true,
                version: (currentSettings?.version || 0) + 1,
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            await INSERT.into(NetworkSettings).entries(newSettings);
            
            cds.log('config').info('Network settings updated', { 
                version: newSettings.version, 
                userId 
            });

            return 'Network settings updated successfully';

        } catch (error) {
            cds.log('config').error('Failed to update network settings:', error);
            req.error(500, 'Failed to update network settings', error.message);
        }
    });

    /**
     * Get current security settings
     */
    this.on('getSecuritySettings', async (req) => {
        try {
            const settings = await SELECT.one.from(SecuritySettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            if (settings) {
                return settings;
            }

            // Return default security settings
            const defaultSettings = {
                ID: cds.utils.uuid(),
                encryptionEnabled: true,
                authRequired: process.env.NODE_ENV === 'production',
                twoFactorEnabled: false,
                sessionTimeout: 30,
                maxLoginAttempts: 5,
                passwordMinLength: 8,
                isActive: true,
                version: 1,
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            await INSERT.into(SecuritySettings).entries(defaultSettings);
            cds.log('config').info('Created default security settings');

            return defaultSettings;

        } catch (error) {
            cds.log('config').error('Failed to get security settings:', error);
            req.error(500, 'Failed to retrieve security settings', error.message);
        }
    });

    /**
     * Update security settings
     */
    this.on('updateSecuritySettings', async (req) => {
        try {
            const { settings } = req.data;
            const userId = req.user?.id || 'system';

            // Get current settings for audit log
            const currentSettings = await SELECT.one.from(SecuritySettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            // Deactivate current settings
            if (currentSettings) {
                await UPDATE(SecuritySettings)
                    .set({ isActive: false })
                    .where({ ID: currentSettings.ID });

                await this._createAuditLog('SECURITY_SETTINGS', 'UPDATE',
                    JSON.stringify(currentSettings), JSON.stringify(settings), userId);
            }

            // Create new settings version
            const newSettings = {
                ID: cds.utils.uuid(),
                ...settings,
                isActive: true,
                version: (currentSettings?.version || 0) + 1,
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            await INSERT.into(SecuritySettings).entries(newSettings);
            
            cds.log('config').info('Security settings updated', { 
                version: newSettings.version, 
                userId 
            });

            return 'Security settings updated successfully';

        } catch (error) {
            cds.log('config').error('Failed to update security settings:', error);
            req.error(500, 'Failed to update security settings', error.message);
        }
    });

    /**
     * Auto-save settings backup
     */
    this.on('autoSaveSettings', async (req) => {
        try {
            const { settings, timestamp, userId } = req.data;

            // Mark previous auto-saves as not latest
            await UPDATE(AutoSavedSettings)
                .set({ isLatest: false })
                .where({ userId: userId, isLatest: true });

            // Create new auto-save entry
            const autoSave = {
                ID: cds.utils.uuid(),
                settingsData: typeof settings === 'string' ? settings : JSON.stringify(settings),
                settingsType: 'AUTO_SAVE',
                userId: userId || 'system',
                timestamp: new Date(timestamp || Date.now()),
                version: Date.now(), // Use timestamp as version
                isLatest: true,
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            await INSERT.into(AutoSavedSettings).entries(autoSave);
            
            cds.log('config').info('Settings auto-saved', { userId, version: autoSave.version });

            return 'Settings auto-saved successfully';

        } catch (error) {
            cds.log('config').error('Failed to auto-save settings:', error);
            req.error(500, 'Failed to auto-save settings', error.message);
        }
    });

    /**
     * Get current settings version
     */
    this.on('getSettingsVersion', async (req) => {
        try {
            const networkVersion = await SELECT.one`version`.from(NetworkSettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            const securityVersion = await SELECT.one`version`.from(SecuritySettings)
                .where({ isActive: true })
                .orderBy({ version: 'desc' });

            // Return the highest version number
            const maxVersion = Math.max(
                networkVersion?.version || 1,
                securityVersion?.version || 1
            );

            return maxVersion;

        } catch (error) {
            cds.log('config').error('Failed to get settings version:', error);
            return 1;
        }
    });

    /**
     * Get settings history
     */
    this.on('getSettingsHistory', async (req) => {
        try {
            const { settingType } = req.data;

            const history = await SELECT.from(SettingsAuditLog)
                .where({ settingType })
                .orderBy({ timestamp: 'desc' })
                .limit(50);

            return history;

        } catch (error) {
            cds.log('config').error('Failed to get settings history:', error);
            return [];
        }
    });

    /**
     * Create audit log entry
     * @private
     */
    this._createAuditLog = async (settingType, action, oldValue, newValue, userId) => {
        try {
            const auditEntry = {
                ID: cds.utils.uuid(),
                settingType,
                settingKey: action,
                oldValue: oldValue || '',
                newValue: newValue || '',
                changedBy: userId || 'system',
                changeReason: `${action} operation`,
                timestamp: new Date(),
                version: Date.now(),
                createdAt: new Date(),
                modifiedAt: new Date()
            };

            await INSERT.into(SettingsAuditLog).entries(auditEntry);

        } catch (error) {
            cds.log('config').error('Failed to create audit log:', error);
            // Don't throw error as this shouldn't break the main operation
        }
    };

    // Initialize default settings on service startup
    this.on('served', async () => {
        try {
            // Check if default settings exist
            const networkExists = await SELECT.one.from(NetworkSettings).where({ isActive: true });
            const securityExists = await SELECT.one.from(SecuritySettings).where({ isActive: true });

            if (!networkExists) {
                cds.log('config').info('Initializing default network settings');
                // This will be handled by the getNetworkSettings action when first called
            }

            if (!securityExists) {
                cds.log('config').info('Initializing default security settings');
                // This will be handled by the getSecuritySettings action when first called
            }

            cds.log('config').info('Configuration service initialized');

        } catch (error) {
            cds.log('config').error('Failed to initialize configuration service:', error);
        }
    });

});