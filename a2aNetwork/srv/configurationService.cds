using { managed } from '@sap/cds/common';

/**
 * Configuration Service for A2A Network
 * Manages application settings, network configuration, and security settings
 */
service ConfigurationService @(path: '/api/v1/config') {

    /**
     * Network Configuration Settings
     */
    entity NetworkSettings : managed {
        key ID          : UUID;
        network         : String(50)  @title: 'Network Name';
        rpcUrl          : String(200) @title: 'RPC URL';
        chainId         : Integer     @title: 'Chain ID';
        contractAddress : String(42)  @title: 'Contract Address';
        isActive        : Boolean     @title: 'Is Active' default true;
        version         : Integer     @title: 'Version' default 1;
    }

    /**
     * Security Configuration Settings
     */
    entity SecuritySettings : managed {
        key ID                : UUID;
        encryptionEnabled     : Boolean @title: 'Encryption Enabled' default true;
        authRequired          : Boolean @title: 'Authentication Required' default true;
        twoFactorEnabled      : Boolean @title: 'Two Factor Authentication' default false;
        sessionTimeout       : Integer @title: 'Session Timeout (minutes)' default 30;
        maxLoginAttempts     : Integer @title: 'Max Login Attempts' default 5;
        passwordMinLength    : Integer @title: 'Password Min Length' default 8;
        isActive             : Boolean @title: 'Is Active' default true;
        version              : Integer @title: 'Version' default 1;
    }

    /**
     * Application Configuration Settings
     */
    entity ApplicationSettings : managed {
        key ID               : UUID;
        environment          : String(20)  @title: 'Environment';
        logLevel            : String(10)   @title: 'Log Level' default 'info';
        enableMetrics       : Boolean      @title: 'Enable Metrics' default true;
        enableTracing       : Boolean      @title: 'Enable Tracing' default true;
        maintenanceMode     : Boolean      @title: 'Maintenance Mode' default false;
        maxConcurrentUsers  : Integer      @title: 'Max Concurrent Users' default 1000;
        cacheEnabled        : Boolean      @title: 'Cache Enabled' default true;
        cacheTTL            : Integer      @title: 'Cache TTL (seconds)' default 300;
        isActive            : Boolean      @title: 'Is Active' default true;
        version             : Integer      @title: 'Version' default 1;
    }

    /**
     * Settings Audit Log
     */
    entity SettingsAuditLog : managed {
        key ID          : UUID;
        settingType     : String(50)  @title: 'Setting Type';
        settingKey      : String(100) @title: 'Setting Key';
        oldValue        : String(500) @title: 'Old Value';
        newValue        : String(500) @title: 'New Value';
        changedBy       : String(100) @title: 'Changed By';
        changeReason    : String(200) @title: 'Change Reason';
        timestamp       : DateTime    @title: 'Timestamp';
        version         : Integer     @title: 'Version';
    }

    /**
     * Auto-saved Settings Backup
     */
    entity AutoSavedSettings : managed {
        key ID          : UUID;
        settingsData    : LargeString @title: 'Settings JSON Data';
        settingsType    : String(50)  @title: 'Settings Type';
        userId          : String(100) @title: 'User ID';
        timestamp       : DateTime    @title: 'Timestamp';
        version         : Integer     @title: 'Version';
        isLatest        : Boolean     @title: 'Is Latest Version' default false;
    }

    // Actions and Functions
    action getNetworkSettings() returns NetworkSettings;
    action updateNetworkSettings(settings: NetworkSettings) returns String;
    
    action getSecuritySettings() returns SecuritySettings;
    action updateSecuritySettings(settings: SecuritySettings) returns String;
    
    action getApplicationSettings() returns ApplicationSettings;
    action updateApplicationSettings(settings: ApplicationSettings) returns String;
    
    action autoSaveSettings(data: {
        settings: LargeString;
        timestamp: DateTime;
        userId: String;
    }) returns String;
    
    function getSettingsVersion() returns Integer;
    function getSettingsHistory(settingType: String) returns array of SettingsAuditLog;
    function restoreSettings(version: Integer, settingType: String) returns String;
}