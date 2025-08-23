/**
 * Real Tile Personalization with Database Persistence
 * Stores user tile preferences, positions, and visibility
 */
const path = require('path');
const fs = require('fs');

class TilePersonalizationService {
    constructor(db, isBTP = false) {
        this.db = db;
        this.isBTP = isBTP;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            if (this.isBTP) {
                // CAP/HANA: Use proper CDS queries
                await this.initializeHANATables();
            } else {
                // SQLite: Create tables directly
                await this.initializeSQLiteTables();
            }
            
            this.initialized = true;
            // console.log('✅ Tile personalization service initialized');
        } catch (error) {
            console.error('❌ Failed to initialize tile personalization:', error.message);
        }
    }

    async initializeSQLiteTables() {
        const handleInitializeSQLiteTables = (resolve, reject) => {
            const serializeInitialization = () => {
                // User tile configurations table
                this.db.run(`CREATE TABLE IF NOT EXISTS user_tile_config (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tile_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    position INTEGER DEFAULT 0,
                    is_visible BOOLEAN DEFAULT 1,
                    size TEXT DEFAULT '1x1',
                    custom_title TEXT,
                    refresh_interval INTEGER DEFAULT 30,
                    settings TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, tile_id)
                )`);

                // User group configurations table
                this.db.run(`CREATE TABLE IF NOT EXISTS user_group_config (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_title TEXT,
                    position INTEGER DEFAULT 0,
                    is_visible BOOLEAN DEFAULT 1,
                    is_collapsed BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, group_id)
                )`);

                // User preferences table
                this.db.run(`CREATE TABLE IF NOT EXISTS user_preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    theme TEXT DEFAULT 'sap_horizon',
                    layout_mode TEXT DEFAULT 'grid',
                    tile_size_preference TEXT DEFAULT 'medium',
                    auto_refresh BOOLEAN DEFAULT 1,
                    show_notifications BOOLEAN DEFAULT 1,
                    compact_mode BOOLEAN DEFAULT 0,
                    preferences_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )`);

                // Insert default tile configurations
                this.insertDefaultTileConfigurations();
                
                resolve();
            };
            this.db.serialize(serializeInitialization);
        };
        
        return new Promise(handleInitializeSQLiteTables);
    }

    async initializeHANATables() {
        // For HANA, we would use CDS entities
        const createTileConfigEntity = `
            CREATE TABLE IF NOT EXISTS A2A_USER_TILE_CONFIG (
                ID NVARCHAR(36) PRIMARY KEY,
                USER_ID NVARCHAR(100) NOT NULL,
                TILE_ID NVARCHAR(100) NOT NULL,
                GROUP_ID NVARCHAR(100) NOT NULL,
                POSITION INTEGER DEFAULT 0,
                IS_VISIBLE BOOLEAN DEFAULT TRUE,
                SIZE NVARCHAR(10) DEFAULT '1x1',
                CUSTOM_TITLE NVARCHAR(200),
                REFRESH_INTERVAL INTEGER DEFAULT 30,
                SETTINGS NCLOB,
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        `;
        
        try {
            await this.db.run(createTileConfigEntity);
        } catch (error) {
            console.warn('HANA table creation handled by CDS deployment');
        }
    }

    insertDefaultTileConfigurations() {
        const defaultConfigs = [
            // Home Group Defaults
            { groupId: 'a2a_home_group', tileId: 'overview_tile', position: 0, size: '2x2' },
            { groupId: 'a2a_home_group', tileId: 'agents_tile', position: 1, size: '1x1' },
            { groupId: 'a2a_home_group', tileId: 'blockchain_tile', position: 2, size: '1x1' },
            { groupId: 'a2a_home_group', tileId: 'marketplace_tile', position: 3, size: '1x1' },
            
            // Operations Group Defaults
            { groupId: 'a2a_operations_group', tileId: 'operations_tile', position: 0, size: '1x1' },
            { groupId: 'a2a_operations_group', tileId: 'analytics_tile', position: 1, size: '2x1' },
            { groupId: 'a2a_operations_group', tileId: 'alerts_tile', position: 2, size: '1x1' },
            { groupId: 'a2a_operations_group', tileId: 'logs_tile', position: 3, size: '1x1' },
        ];

        const insertDefaultConfig = (config) => {
            const id = `default-${config.groupId}-${config.tileId}`;
            this.db.run(`INSERT OR REPLACE INTO user_tile_config 
                (id, user_id, tile_id, group_id, position, size) 
                VALUES (?, 'default', ?, ?, ?, ?)`,
                [id, config.tileId, config.groupId, config.position, config.size]);
        };
        defaultConfigs.forEach(insertDefaultConfig);
    }

    // Get user's tile configuration
    async getUserTileConfig(userId) {
        if (!this.initialized) await this.initialize();
        
        const handleGetUserTileConfig = (resolve, reject) => {
            if (this.isBTP) {
                // HANA query
                const handleHanaResults = (results) => resolve(results);
                const handleHanaError = (error) => reject(error);
                this.db.run(`SELECT * FROM A2A_USER_TILE_CONFIG WHERE USER_ID = ?`, [userId])
                    .then(handleHanaResults)
                    .catch(handleHanaError);
            } else {
                // SQLite query
                const handleSQLiteResults = (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows || []);
                };
                this.db.all(`SELECT * FROM user_tile_config WHERE user_id = ? ORDER BY group_id, position`, 
                    [userId], handleSQLiteResults);
            }
        };
        
        return new Promise(handleGetUserTileConfig);
    }

    // Save user's tile configuration
    async saveUserTileConfig(userId, tileId, groupId, config) {
        if (!this.initialized) await this.initialize();
        
        const id = `${userId}-${tileId}`;
        const now = new Date().toISOString();
        
        const handleSaveUserTileConfig = (resolve, reject) => {
            const query = `INSERT OR REPLACE INTO user_tile_config 
                (id, user_id, tile_id, group_id, position, is_visible, size, custom_title, refresh_interval, settings, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
            
            const values = [
                id, userId, tileId, groupId,
                config.position || 0,
                config.isVisible !== false,
                config.size || '1x1',
                config.customTitle || null,
                config.refreshInterval || 30,
                JSON.stringify(config.settings || {}),
                now
            ];

            if (this.isBTP) {
                // HANA
                const handleHanaSaveSuccess = () => resolve({ success: true });
                const handleHanaSaveError = (error) => reject(error);
                this.db.run(query, values)
                    .then(handleHanaSaveSuccess)
                    .catch(handleHanaSaveError);
            } else {
                // SQLite
                const handleSQLiteSave = function(err) {
                    if (err) reject(err);
                    else resolve({ success: true, changes: this.changes });
                };
                this.db.run(query, values, handleSQLiteSave);
            }
        };
        
        return new Promise(handleSaveUserTileConfig);
    }

    // Get user's group configuration
    async getUserGroupConfig(userId) {
        if (!this.initialized) await this.initialize();
        
        const handleGetUserGroupConfig = (resolve, reject) => {
            if (this.isBTP) {
                const handleHanaGroupResults = (results) => resolve(results);
                const handleHanaGroupError = (error) => reject(error);
                this.db.run(`SELECT * FROM A2A_USER_GROUP_CONFIG WHERE USER_ID = ?`, [userId])
                    .then(handleHanaGroupResults).catch(handleHanaGroupError);
            } else {
                const handleSQLiteGroupResults = (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows || []);
                };
                this.db.all(`SELECT * FROM user_group_config WHERE user_id = ? ORDER BY position`, 
                    [userId], handleSQLiteGroupResults);
            }
        };
        
        return new Promise(handleGetUserGroupConfig);
    }

    // Save user's group configuration
    async saveUserGroupConfig(userId, groupId, config) {
        if (!this.initialized) await this.initialize();
        
        const id = `${userId}-${groupId}`;
        const now = new Date().toISOString();
        
        const handleSaveUserGroupConfig = (resolve, reject) => {
            const query = `INSERT OR REPLACE INTO user_group_config 
                (id, user_id, group_id, group_title, position, is_visible, is_collapsed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)`;
            
            const values = [
                id, userId, groupId,
                config.title || null,
                config.position || 0,
                config.isVisible !== false,
                config.isCollapsed || false,
                now
            ];

            if (this.isBTP) {
                this.db.run(query, values).then(() => resolve({ success: true })).catch(reject);
            } else {
                this.db.run(query, values, function(err) {
                    if (err) reject(err);
                    else resolve({ success: true, changes: this.changes });
                });
            }
        });
    }

    // Get user preferences
    async getUserPreferences(userId) {
        if (!this.initialized) await this.initialize();
        
        return new Promise((resolve, reject) => {
            if (this.isBTP) {
                this.db.run(`SELECT * FROM A2A_USER_PREFERENCES WHERE USER_ID = ?`, [userId])
                    .then(results => resolve(results[0] || null))
                    .catch(reject);
            } else {
                this.db.get(`SELECT * FROM user_preferences WHERE user_id = ?`, [userId], (err, row) => {
                    if (err) reject(err);
                    else {
                        if (row && row.preferences_json) {
                            try {
                                row.preferences = JSON.parse(row.preferences_json);
                            } catch (e) {
                                console.warn('Failed to parse user preferences JSON');
                            }
                        }
                        resolve(row || null);
                    }
                });
            }
        });
    }

    // Save user preferences
    async saveUserPreferences(userId, preferences) {
        if (!this.initialized) await this.initialize();
        
        const id = `pref-${userId}`;
        const now = new Date().toISOString();
        
        return new Promise((resolve, reject) => {
            const query = `INSERT OR REPLACE INTO user_preferences 
                (id, user_id, theme, layout_mode, tile_size_preference, auto_refresh, show_notifications, compact_mode, preferences_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;
            
            const values = [
                id, userId,
                preferences.theme || 'sap_horizon',
                preferences.layoutMode || 'grid',
                preferences.tileSizePreference || 'medium',
                preferences.autoRefresh !== false,
                preferences.showNotifications !== false,
                preferences.compactMode || false,
                JSON.stringify(preferences),
                now
            ];

            if (this.isBTP) {
                this.db.run(query, values).then(() => resolve({ success: true })).catch(reject);
            } else {
                this.db.run(query, values, function(err) {
                    if (err) reject(err);
                    else resolve({ success: true, changes: this.changes });
                });
            }
        });
    }

    // Reset user configuration to defaults
    async resetUserConfiguration(userId) {
        if (!this.initialized) await this.initialize();
        
        return new Promise((resolve, reject) => {
            if (this.isBTP) {
                Promise.all([
                    this.db.run(`DELETE FROM A2A_USER_TILE_CONFIG WHERE USER_ID = ?`, [userId]),
                    this.db.run(`DELETE FROM A2A_USER_GROUP_CONFIG WHERE USER_ID = ?`, [userId]),
                    this.db.run(`DELETE FROM A2A_USER_PREFERENCES WHERE USER_ID = ?`, [userId])
                ]).then(() => resolve({ success: true })).catch(reject);
            } else {
                this.db.serialize(() => {
                    this.db.run(`DELETE FROM user_tile_config WHERE user_id = ?`, [userId]);
                    this.db.run(`DELETE FROM user_group_config WHERE user_id = ?`, [userId]);
                    this.db.run(`DELETE FROM user_preferences WHERE user_id = ?`, [userId], function(err) {
                        if (err) reject(err);
                        else resolve({ success: true });
                    });
                });
            }
        });
    }

    // Export user configuration
    async exportUserConfiguration(userId) {
        if (!this.initialized) await this.initialize();
        
        try {
            const [tileConfig, groupConfig, preferences] = await Promise.all([
                this.getUserTileConfig(userId),
                this.getUserGroupConfig(userId),
                this.getUserPreferences(userId)
            ]);

            return {
                userId,
                exportDate: new Date().toISOString(),
                tileConfiguration: tileConfig,
                groupConfiguration: groupConfig,
                userPreferences: preferences
            };
        } catch (error) {
            throw new Error(`Failed to export user configuration: ${error.message}`);
        }
    }

    // Import user configuration
    async importUserConfiguration(userId, configData) {
        if (!this.initialized) await this.initialize();
        
        try {
            // Validate configuration data
            if (!configData.tileConfiguration && !configData.groupConfiguration && !configData.userPreferences) {
                throw new Error('Invalid configuration data');
            }

            // Import tile configurations
            if (configData.tileConfiguration && Array.isArray(configData.tileConfiguration)) {
                for (const tileConfig of configData.tileConfiguration) {
                    await this.saveUserTileConfig(userId, tileConfig.tile_id, tileConfig.group_id, {
                        position: tileConfig.position,
                        isVisible: tileConfig.is_visible,
                        size: tileConfig.size,
                        customTitle: tileConfig.custom_title,
                        refreshInterval: tileConfig.refresh_interval,
                        settings: JSON.parse(tileConfig.settings || '{}')
                    });
                }
            }

            // Import group configurations
            if (configData.groupConfiguration && Array.isArray(configData.groupConfiguration)) {
                for (const groupConfig of configData.groupConfiguration) {
                    await this.saveUserGroupConfig(userId, groupConfig.group_id, {
                        title: groupConfig.group_title,
                        position: groupConfig.position,
                        isVisible: groupConfig.is_visible,
                        isCollapsed: groupConfig.is_collapsed
                    });
                }
            }

            // Import user preferences
            if (configData.userPreferences) {
                const prefs = configData.userPreferences;
                await this.saveUserPreferences(userId, {
                    theme: prefs.theme,
                    layoutMode: prefs.layout_mode,
                    tileSizePreference: prefs.tile_size_preference,
                    autoRefresh: prefs.auto_refresh,
                    showNotifications: prefs.show_notifications,
                    compactMode: prefs.compact_mode,
                    ...JSON.parse(prefs.preferences_json || '{}')
                });
            }

            return { success: true, message: 'Configuration imported successfully' };
        } catch (error) {
            throw new Error(`Failed to import user configuration: ${error.message}`);
        }
    }

    // Get analytics on tile usage
    async getTileUsageAnalytics() {
        if (!this.initialized) await this.initialize();
        
        return new Promise((resolve, reject) => {
            const query = `
                SELECT 
                    tile_id,
                    COUNT(*) as user_count,
                    AVG(position) as avg_position,
                    COUNT(CASE WHEN is_visible = 1 THEN 1 END) as visible_count,
                    COUNT(CASE WHEN is_visible = 0 THEN 1 END) as hidden_count
                FROM user_tile_config 
                WHERE user_id != 'default'
                GROUP BY tile_id 
                ORDER BY user_count DESC
            `;

            if (this.isBTP) {
                this.db.run(query).then(resolve).catch(reject);
            } else {
                this.db.all(query, (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows || []);
                });
            }
        });
    }
}

module.exports = TilePersonalizationService;