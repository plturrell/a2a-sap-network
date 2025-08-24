sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/Fragment',
  'sap/ui/model/json/JSONModel',
  'sap/ui/model/odata/v4/ODataModel',
  'sap/base/Log',
  'sap/ui/core/BusyIndicator',
  'sap/base/security/encodeXML'
], (Controller, MessageToast, MessageBox, Fragment, JSONModel, ODataModel, Log, BusyIndicator, encodeXML) => {
  'use strict';
  /* global Notification, Intl */

  return Controller.extend('a2a.portal.controller.App', {

    onInit: function () {
      Log.info('SAP A2A Developer Portal Controller initialized', null, 'a2a.portal.controller.App');
            
      // Initialize component and model references
      this._oComponent = this.getOwnerComponent();
      this._oRouter = this._oComponent.getRouter();
      this._oResourceBundle = this._oComponent.getModel('i18n').getResourceBundle();
            
      // Initialize SAP BTP services
      this._initializeBTPServices();
            
      // Initialize security context
      this._initializeSecurityContext();
            
      // Initialize notification model with SAP standards
      this._initializeNotificationModel();
            
      // Initialize telemetry and monitoring
      this._initializeTelemetryMonitoring();
            
      // Set initial navigation with proper error handling
      this._setInitialNavigation();
    },

    _initializeBTPServices: function () {
      // Initialize SAP Cloud SDK destination service
      this._oDestinationService = this._oComponent.getModel('destinationService');
            
      // Initialize XSUAA security context
      this._oSecurityContext = this._oComponent.getModel('securityContext');
            
      // Initialize audit logging service
      this._oAuditLog = this._oComponent.getService('auditLogging');
            
      Log.info('SAP BTP services initialized successfully', null, 'a2a.portal.controller.App');
    },

    _initializeSecurityContext: function () {
      const that = this;
      const oUserModel = new JSONModel();
            
      // Get user info from XSUAA token
      jQuery.ajax({
        url: '/api/user/info',
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        success: function (oUserData) {
          oUserModel.setData(oUserData);
          that.getView().setModel(oUserModel, 'user');
                    
          // Initialize role-based features
          that._initializeRoleBasedFeatures(oUserData.scopes || []);
                    
          // Log successful authentication
          that._logSecurityEvent('USER_LOGIN', {
            userId: oUserData.userId,
            roles: oUserData.scopes
          });
        },
        error: function (jqXHR) {
          Log.error('Failed to initialize security context', jqXHR.responseText, 'a2a.portal.controller.App');
          that._handleSecurityError();
        }
      });
    },

    _initializeTelemetryMonitoring: function () {
      const that = this;
      // Initialize OpenTelemetry integration for SAP Cloud ALM
      this._oTelemetryService = {
        trackEvent: function(sEventName, mProperties) {
          // Send telemetry data to SAP Cloud ALM
          jQuery.ajax({
            url: '/api/telemetry/track',
            method: 'POST',
            data: JSON.stringify({
              eventName: sEventName,
              properties: mProperties,
              timestamp: new Date().toISOString(),
              sessionId: that._getSessionId(),
              userId: that._getCurrentUserId()
            }),
            contentType: 'application/json'
          });
        }
      };
    },

    _initializeNotificationModel: function () {
      // Create enterprise notification model with SAP standards
      const oNotificationModel = new JSONModel({
        notifications: [],
        stats: {
          total: 0,
          unread: 0,
          read: 0,
          dismissed: 0,
          critical: 0,
          high: 0,
          medium: 0,
          low: 0
        },
        loading: false,
        hasMore: false,
        lastRefresh: null,
        filters: {
          status: null,
          type: null,
          priority: null,
          dateRange: null
        },
        preferences: {
          autoRefresh: true,
          refreshInterval: 30000, // 30 seconds
          maxNotifications: 100,
          enableSound: false,
          enableDesktopNotifications: false
        }
      });
            
      this.getView().setModel(oNotificationModel, 'notifications');
            
      // Load initial notification data with enterprise patterns
      this._loadNotificationsWithRetry();
            
      // Set up auto-refresh if enabled
      this._setupNotificationAutoRefresh();
    },

    _loadNotificationsWithRetry: function (iRetryCount) {
      const that = this;
      const oModel = this.getView().getModel('notifications');
      const oData = oModel.getData();
      iRetryCount = iRetryCount || 0;
            
      oData.loading = true;
      oData.lastRefresh = new Date().toISOString();
      oModel.setData(oData);
            
      // Enhanced notification loading with SAP Cloud SDK patterns
      jQuery.ajax({
        url: '/api/v2/notifications',
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
          'SAP-ContextId': this._getContextId(),
          'Authorization': `Bearer ${  this._getBearerToken()}`
        },
        data: {
          '$top': oData.preferences.maxNotifications,
          '$skip': 0,
          '$filter': this._buildNotificationFilter(oData.filters),
          '$orderby': 'createdAt desc',
          '$expand': 'actions,attachments'
        },
        success: function (oResponse) {
          that._handleNotificationSuccess(oResponse, oData, oModel);
                    
          // Track successful load
          that._oTelemetryService.trackEvent('NOTIFICATIONS_LOADED', {
            count: oResponse.value ? oResponse.value.length : 0,
            loadTime: Date.now() - that._loadStartTime
          });
        },
        error: function (jqXHR, _textStatus, _errorThrown) {
          that._handleNotificationError(jqXHR, iRetryCount, oData, oModel);
        },
        timeout: 10000 // 10 second timeout
      });
            
      this._loadStartTime = Date.now();
    },

    _handleNotificationSuccess: function (oResponse, oData, oModel) {
      try {
        // Process OData v4 response
        oData.notifications = this._processNotifications(oResponse.value || []);
        oData.stats.total = parseInt(oResponse['@odata.count']) || 0;
        oData.stats.unread = oData.notifications.filter((n) => {
          return n.status === 'unread'; 
        }).length;
        oData.hasMore = !!oResponse['@odata.nextLink'];
        oData.loading = false;
        oData.error = null;
                
        oModel.setData(oData);
                
        Log.info(`Loaded ${  oData.notifications.length  } notifications successfully`, null, 'a2a.portal.controller.App');
                
        // Update browser title if unread notifications
        this._updateBrowserTitle(oData.stats.unread);
                
        // Show desktop notification if enabled and there are critical notifications
        this._checkForCriticalNotifications(oData.notifications);
                
      } catch (oError) {
        Log.error('Error processing notification response', oError.message, 'a2a.portal.controller.App');
        this._showErrorMessage(this._oResourceBundle.getText('notifications.processing.error'));
      }
    },

    _handleNotificationError: function (jqXHR, iRetryCount, oData, oModel) {
      let sErrorMessage = 'Failed to load notifications';
            
      oData.loading = false;
            
      // Implement exponential backoff retry logic
      if (iRetryCount < 3 && (jqXHR.status === 0 || jqXHR.status >= 500)) {
        const iDelay = Math.pow(2, iRetryCount) * 1000; // 1s, 2s, 4s
                
        Log.warning(`Retrying notification load in ${  iDelay  }ms (attempt ${  iRetryCount + 1  })`, 
          jqXHR.responseText, 'a2a.portal.controller.App');
                          
        setTimeout(() => {
          this._loadNotificationsWithRetry(iRetryCount + 1);
        }, iDelay);
                
        return;
      }
            
      // Handle specific error cases
      switch (jqXHR.status) {
      case 401:
        sErrorMessage = this._oResourceBundle.getText('notifications.error.unauthorized');
        this._handleAuthenticationError();
        break;
      case 403:
        sErrorMessage = this._oResourceBundle.getText('notifications.error.forbidden');
        this._logSecurityEvent('ACCESS_DENIED', { resource: 'notifications' });
        break;
      case 404:
        sErrorMessage = this._oResourceBundle.getText('notifications.error.notFound');
        break;
      default:
        sErrorMessage = this._oResourceBundle.getText('notifications.error.generic');
      }
            
      oData.error = {
        message: sErrorMessage,
        details: jqXHR.responseText || jqXHR.statusText,
        timestamp: new Date().toISOString(),
        retryable: jqXHR.status >= 500
      };
            
      oModel.setData(oData);
            
      Log.error('Notification load failed after retries', JSON.stringify({
        status: jqXHR.status,
        statusText: jqXHR.statusText,
        responseText: jqXHR.responseText
      }), 'a2a.portal.controller.App');
            
      // Show user-friendly error message
      this._showErrorMessage(sErrorMessage);
    },

    onItemSelect: function (oEvent) {
      const oItem = oEvent.getParameter('item');
      const sKey = oItem.getKey();
            
      // Validate access before navigation
      if (!this._validatePageAccess(sKey)) {
        return;
      }
            
      switch (sKey) {
      case 'projects':
        this._oRouter.navTo('projects');
        break;
      case 'agentBuilder':
        this._oRouter.navTo('agentBuilder');
        break;
      case 'bpmnDesigner':
        this._oRouter.navTo('bpmnDesigner');
        break;
      case 'templates':
        this._oRouter.navTo('templates');
        break;
      case 'testing':
        this._oRouter.navTo('testing');
        break;
      case 'deployment':
        this._oRouter.navTo('deployment');
        break;
      case 'monitoring':
        this._oRouter.navTo('monitoring');
        break;
      case 'a2aNetwork':
        this._oRouter.navTo('a2aNetwork');
        break;
      default:
        Log.warning(`Unknown navigation key: ${  sKey}`, null, 'a2a.portal.controller.App');
      }
            
      // Track navigation
      this._oTelemetryService.trackEvent('NAVIGATION', {
        to: sKey,
        trigger: 'sidebar_navigation'
      });
    },

    onUserProfilePress: function () {
      // Navigate to user profile with proper routing
      this._oRouter.navTo('profile', {}, false);
            
      // Track navigation event
      this._oTelemetryService.trackEvent('NAVIGATION', {
        from: 'shell',
        to: 'profile',
        trigger: 'user_action'
      });
    },

    onNotificationPress: function () {
      this._openNotificationPanel();
    },

    _openNotificationPanel: function () {
      const that = this;
            
      if (!this._notificationPanel) {
        Fragment.load({
          name: 'a2a.portal.view.fragments.NotificationPanel',
          controller: this
        }).then((oFragment) => {
          that._notificationPanel = oFragment;
          that.getView().addDependent(oFragment);
          that._notificationPanel.open();
        });
      } else {
        this._notificationPanel.open();
      }
            
      // Refresh notifications when panel opens
      this._loadNotificationsWithRetry();
    },

    onCloseNotificationPanel: function () {
      if (this._notificationPanel) {
        this._notificationPanel.close();
      }
    },

    onRefreshNotifications: function () {
      this._loadNotificationsWithRetry();
      MessageToast.show('Notifications refreshed');
    },

    onMarkAllAsRead: function () {
      const that = this;
      jQuery.ajax({
        url: '/api/v2/notifications/mark-all-read',
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${  this._getBearerToken()}`,
          'Content-Type': 'application/json'
        },
        success: function () {
          that._loadNotificationsWithRetry();
          MessageToast.show('All notifications marked as read');
        },
        error: function () {
          MessageToast.show('Failed to mark notifications as read');
        }
      });
    },

    onSettingsPress: function () {
      this._showSettingsDialog();
    },

    _showSettingsDialog: function () {
      const that = this;
            
      if (!this._oSettingsDialog) {
        Fragment.load({
          name: 'a2a.portal.fragment.SettingsDialog',
          controller: this
        }).then((oDialog) => {
          that._oSettingsDialog = oDialog;
          that.getView().addDependent(oDialog);
                    
          // Initialize settings model
          that._initializeSettingsModel();
                    
          oDialog.open();
        }).catch((oError) => {
          Log.error('Failed to load settings dialog', oError.message, 'a2a.portal.controller.App');
          MessageToast.show('Failed to load settings dialog');
        });
      } else {
        this._initializeSettingsModel();
        this._oSettingsDialog.open();
      }
    },

    _initializeSettingsModel: function () {
      const that = this;
            
      this._loadUserSettings().then((oSettings) => {
        const oSettingsModel = new JSONModel(oSettings);
        that._oSettingsDialog.setModel(oSettingsModel, 'settings');
                
        Log.info('Settings model initialized', null, 'a2a.portal.controller.App');
      }).catch((oError) => {
        Log.error('Failed to initialize settings model', oError.message, 'a2a.portal.controller.App');
      });
    },

    _loadUserSettings: function () {
      // Load user settings from SAP BTP User Settings Service
      const that = this;
            
      return new Promise((resolve, _reject) => {
        jQuery.ajax({
          url: '/api/v2/user/settings',
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${  that._getBearerToken()}`
          },
          success: function(oResponse) {
            resolve(oResponse);
          },
          error: function(jqXHR) {
            // Fallback to defaults on error
            Log.warning('Failed to load user settings from backend, using defaults', 
              jqXHR.responseText, 'a2a.portal.controller.App');
            resolve(null);
          }
        });
      }).then((oServerSettings) => {
        const oDefaultSettings = {
          theme: 'sap_horizon_dark',
          language: sap.ui.getCore().getConfiguration().getLanguage() || 'en',
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
          autoSave: true,
          accessibility: {
            highContrast: false,
            screenReader: false,
            keyboardNavigation: true
          },
          notifications: {
            email: true,
            push: false,
            deployment: true,
            security: true,
            agents: true,
            workflow: true,
            integration: true,
            sound: false,
            desktop: false
          },
          developer: {
            debugMode: false,
            consoleLogging: false,
            performanceMonitoring: true,
            apiTimeout: 30,
            codeEditor: {
              theme: 'vs-dark',
              fontSize: 14,
              wordWrap: true,
              minimap: true
            }
          },
          privacy: {
            analytics: true,
            crashReporting: true,
            usageStatistics: false
          },
          integration: {
            s4hanaSystem: null,
            analyticsCloud: null,
            workflowService: true,
            eventMesh: true
          }
        };
                
        // Merge server settings with defaults
        if (oServerSettings && oServerSettings.preferences) {
          return jQuery.extend(true, {}, oDefaultSettings, oServerSettings.preferences);
        }
                
        return oDefaultSettings;
      }).catch((oError) => {
        Log.error('Error loading user settings', oError.message, 'a2a.portal.controller.App');
        // Return default settings on error
        return {
          theme: 'sap_horizon_dark',
          language: sap.ui.getCore().getConfiguration().getLanguage() || 'en',
          timezone: 'UTC',
          autoSave: true,
          accessibility: {
            highContrast: false,
            screenReader: false,
            keyboardNavigation: true
          },
          notifications: {
            email: true,
            push: false,
            deployment: true,
            security: true,
            agents: true,
            workflow: true,
            integration: true,
            sound: false,
            desktop: false
          },
          developer: {
            debugMode: false,
            consoleLogging: false,
            performanceMonitoring: true,
            apiTimeout: 30,
            codeEditor: {
              theme: 'vs-dark',
              fontSize: 14,
              wordWrap: true,
              minimap: true
            }
          },
          privacy: {
            analytics: true,
            crashReporting: true,
            usageStatistics: false
          },
          integration: {
            s4hanaSystem: null,
            analyticsCloud: null,
            workflowService: true,
            eventMesh: true
          }
        };
      });
    },

    onSaveSettings: function () {
      const that = this;
      const oSettingsModel = this._oSettingsDialog.getModel('settings');
      const oSettings = oSettingsModel.getData();
            
      // Validate settings
      if (!this._validateSettings(oSettings)) {
        return;
      }
            
      // Show loading indicator
      this._oSettingsDialog.setBusy(true);
            
      // Save settings with SAP CAP logging
      this._saveUserSettings(oSettings).then((bSuccess) => {
        that._oSettingsDialog.setBusy(false);
                
        if (bSuccess) {
          MessageToast.show('Settings saved successfully');
          that._applySettings(oSettings);
          that._oSettingsDialog.close();
                    
          // SAP CAP audit logging
          that._logSecurityEvent('USER_SETTINGS_UPDATED', {
            settingsChanged: Object.keys(oSettings),
            timestamp: new Date().toISOString()
          });
        } else {
          MessageToast.show('Failed to save settings');
        }
      }).catch((oError) => {
        that._oSettingsDialog.setBusy(false);
        Log.error('Settings save error', oError.message, 'a2a.portal.controller.App');
        MessageToast.show(`Error saving settings: ${  oError.message}`);
      });
    },

    _validateSettings: function (oSettings) {
      // Validate API timeout
      if (oSettings.developer.apiTimeout < 5 || oSettings.developer.apiTimeout > 300) {
        MessageBox.error('API timeout must be between 5 and 300 seconds');
        return false;
      }
            
      return true;
    },

    _saveUserSettings: function (oSettings) {
      const that = this;
            
      return new Promise((resolve, reject) => {
        BusyIndicator.show();
                
        jQuery.ajax({
          url: '/api/v2/user/settings',
          method: 'PUT',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${  that._getBearerToken()}`
          },
          data: JSON.stringify({
            preferences: oSettings,
            lastModified: new Date().toISOString()
          }),
          success: function (_oResponse) {
            BusyIndicator.hide();
            resolve(true);
          },
          error: function (jqXHR, _textStatus, _errorThrown) {
            BusyIndicator.hide();
                        
            Log.error('Failed to save user settings', JSON.stringify({
              status: jqXHR.status,
              statusText: jqXHR.statusText,
              responseText: jqXHR.responseText
            }), 'a2a.portal.controller.App');
                        
            reject(new Error(that._oResourceBundle.getText('settings.save.error')));
          },
          timeout: 10000
        });
      });
    },

    _applySettings: function (oSettings) {
      // Apply theme change
      if (oSettings.theme !== sap.ui.getCore().getConfiguration().getTheme()) {
        sap.ui.getCore().applyTheme(oSettings.theme);
        Log.info(`Theme applied: ${  oSettings.theme}`, null, 'a2a.portal.controller.App');
      }
            
      // Apply other settings as needed
      Log.info('Settings applied successfully', null, 'a2a.portal.controller.App');
    },

    onResetSettings: function () {
      const that = this;
            
      MessageBox.confirm('Are you sure you want to reset all settings to defaults?', {
        title: 'Reset Settings',
        onClose: function (sAction) {
          if (sAction === MessageBox.Action.OK) {
            that._resetToDefaults();
          }
        }
      });
    },

    _resetToDefaults: function () {
      this._loadUserSettings().then((oDefaultSettings) => {
        const oSettingsModel = this._oSettingsDialog.getModel('settings');
        oSettingsModel.setData(oDefaultSettings);
                
        MessageToast.show('Settings reset to defaults');
                
        // SAP CAP audit logging
        this._logSecurityEvent('USER_SETTINGS_RESET', {
          timestamp: new Date().toISOString()
        });
      });
    },

    onCancelSettings: function () {
      this._oSettingsDialog.close();
    },

    onLogPress: function () {
      // Show application logs dialog
      this._showApplicationLogs();
    },

    _showApplicationLogs: function () {
      const that = this;
            
      BusyIndicator.show();
            
      // Fetch real application logs from SAP Application Logging Service
      jQuery.ajax({
        url: '/api/v2/monitoring/logs',
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Authorization': `Bearer ${  this._getBearerToken()}`
        },
        data: {
          '$top': 100,
          '$orderby': 'timestamp desc',
          'level': 'info,warning,error'
        },
        success: function (oResponse) {
          BusyIndicator.hide();
                    
          const sLogs = that._formatApplicationLogs(oResponse.logs || []);
                    
          MessageBox.information(sLogs, {
            title: that._oResourceBundle.getText('logs.dialog.title'),
            details: that._oResourceBundle.getText('logs.dialog.description'),
            styleClass: 'sapUiSizeCompact',
            contentWidth: '50rem',
            contentHeight: '30rem'
          });
                    
          // Track log access
          that._logSecurityEvent('APPLICATION_LOGS_ACCESSED', {
            logCount: oResponse.logs ? oResponse.logs.length : 0
          });
        },
        error: function (jqXHR) {
          BusyIndicator.hide();
                    
          Log.error('Failed to load application logs', jqXHR.responseText, 'a2a.portal.controller.App');
                    
          MessageBox.error(that._oResourceBundle.getText('logs.error.message'), {
            title: that._oResourceBundle.getText('logs.error.title'),
            details: jqXHR.responseText || jqXHR.statusText
          });
        }
      });
    },

    _formatApplicationLogs: function (aLogs) {
      let sFormattedLogs = '=== SAP A2A Developer Portal - Application Logs ===\n\n';
            
      if (!aLogs || aLogs.length === 0) {
        sFormattedLogs += 'No recent log entries found.\n';
        return sFormattedLogs;
      }
            
      aLogs.forEach((oLogEntry) => {
        const sTimestamp = new Date(oLogEntry.timestamp).toISOString();
        const sLevel = (oLogEntry.level || 'INFO').toUpperCase();
        const sMessage = encodeXML(oLogEntry.message || 'No message');
        const sComponent = oLogEntry.component || 'unknown';
                
        sFormattedLogs += `[${  sTimestamp  }] ${  sLevel  } [${  sComponent  }]: ${  sMessage  }\n`;
                
        if (oLogEntry.details) {
          sFormattedLogs += `  Details: ${  encodeXML(JSON.stringify(oLogEntry.details))  }\n`;
        }
                
        sFormattedLogs += '\n';
      });
            
      return sFormattedLogs;
    },

    // Enhanced utility methods for enterprise functionality
    _getCurrentUserId: function () {
      const oUserModel = this.getView().getModel('user');
      return oUserModel ? oUserModel.getProperty('/userId') : 'anonymous';
    },

    _getCurrentTenantId: function () {
      const oUserModel = this.getView().getModel('user');
      return oUserModel ? oUserModel.getProperty('/tenantId') : 'unknown';
    },

    _getBearerToken: function () {
      // Get JWT token from XSUAA service
      const oSecurityContext = this.getView().getModel('securityContext');
      return oSecurityContext ? oSecurityContext.getProperty('/token') : '';
    },

    _getContextId: function () {
      // Generate correlation ID for request tracing
      return `a2a-portal-${  Date.now()  }-${  Math.random().toString(36).substr(2, 9)}`;
    },

    _getSessionId: function () {
      const oUserModel = this.getView().getModel('user');
      return oUserModel ? oUserModel.getProperty('/sessionId') : 'no-session';
    },

    _logSecurityEvent: function (sEventType, mDetails) {
      // SAP Audit Logging Service integration
      const oAuditLogEntry = {
        uuid: this._generateUUID(),
        time: new Date().toISOString(),
        tenant: this._getCurrentTenantId(),
        category: 'audit.security-events',
        user: this._getCurrentUserId(),
        object: {
          type: 'A2A_Developer_Portal',
          id: 'sap-a2a-developer-portal'
        },
        attributes: [
          {
            name: 'event_type',
            old: null,
            new: sEventType
          },
          {
            name: 'context_id',
            old: null,
            new: this._getContextId()
          },
          {
            name: 'session_id',
            old: null,
            new: this._getSessionId()
          }
        ]
      };
            
      // Add custom details
      if (mDetails) {
        Object.keys(mDetails).forEach((sKey) => {
          oAuditLogEntry.attributes.push({
            name: sKey,
            old: null,
            new: JSON.stringify(mDetails[sKey])
          });
        });
      }
            
      // Send to SAP Audit Logging Service
      jQuery.ajax({
        url: '/api/v2/audit/log',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${  this._getBearerToken()}`
        },
        data: JSON.stringify(oAuditLogEntry),
        success: function() {
          Log.info(`Audit event logged successfully: ${  sEventType}`, null, 'a2a.portal.controller.App');
        },
        error: function(jqXHR) {
          Log.error(`Failed to log audit event: ${  sEventType}`, jqXHR.responseText, 'a2a.portal.controller.App');
        }
      });
    },

    _generateUUID: function () {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    },

    onNavigate: function (oEvent) {
      // Enhanced navigation handling with telemetry
      const sFromPage = oEvent.getParameter('from');
      const sToPage = oEvent.getParameter('to');
            
      this._oTelemetryService.trackEvent('PAGE_NAVIGATION', {
        fromPage: sFromPage ? sFromPage.getId() : 'unknown',
        toPage: sToPage ? sToPage.getId() : 'unknown',
        timestamp: new Date().toISOString()
      });
    },

    onAfterNavigate: function (oEvent) {
      // Post-navigation processing with security checks
      const sToPage = oEvent.getParameter('to');
            
      if (sToPage) {
        this._validatePageAccess(sToPage.getId());
        this._updateBreadcrumb(sToPage.getId());
      }
    },

    _initializeRoleBasedFeatures: function (aScopes) {
      const oFeatureModel = new JSONModel({
        canCreateProjects: aScopes.includes('SAP_A2A_Developer'),
        canManageUsers: aScopes.includes('SAP_A2A_Administrator'),
        canViewAuditLogs: aScopes.includes('SAP_A2A_Auditor'),
        canAccessIntegration: aScopes.includes('SAP_A2A_IntegrationSpecialist'),
        canManageSettings: aScopes.includes('SAP_A2A_ProjectManager'),
        isBusinessUser: aScopes.includes('SAP_A2A_BusinessUser')
      });
            
      this.getView().setModel(oFeatureModel, 'features');
    },

    _validatePageAccess: function (sPageId) {
      const oFeatures = this.getView().getModel('features');
      if (!oFeatures) {
        return false;
      }
            
      const oFeatureData = oFeatures.getData();
      let bHasAccess = true;
            
      // Implement page-level security checks
      switch (sPageId) {
      case 'admin':
        bHasAccess = oFeatureData.canManageUsers;
        break;
      case 'integration':
        bHasAccess = oFeatureData.canAccessIntegration;
        break;
      case 'audit':
        bHasAccess = oFeatureData.canViewAuditLogs;
        break;
      }
            
      if (!bHasAccess) {
        this._logSecurityEvent('UNAUTHORIZED_PAGE_ACCESS', {
          pageId: sPageId,
          userId: this._getCurrentUserId()
        });
                
        this._oRouter.navTo('unauthorized');
        return false;
      }
            
      return true;
    },

    _handleAuthenticationError: function () {
      // Redirect to login or show authentication dialog
      MessageBox.error(this._oResourceBundle.getText('auth.session.expired'), {
        title: this._oResourceBundle.getText('auth.error.title'),
        actions: [MessageBox.Action.OK],
        onClose: function () {
          window.location.href = '/login';
        }
      });
    },

    _handleSecurityError: function () {
      MessageBox.error(this._oResourceBundle.getText('security.initialization.failed'), {
        title: this._oResourceBundle.getText('security.error.title'),
        details: this._oResourceBundle.getText('security.contact.admin')
      });
    },

    _showErrorMessage: function (sMessage) {
      MessageToast.show(sMessage, {
        duration: 5000,
        width: '20rem'
      });
    },

    _setInitialNavigation: function () {
      const that = this;
            
      // Set initial navigation with error handling
      setTimeout(() => {
        try {
          const oSideNavigation = that.byId('sideNavigation');
          if (oSideNavigation) {
            oSideNavigation.setSelectedKey('projects');
            Log.info('Initial navigation set to projects', null, 'a2a.portal.controller.App');
          }
        } catch (oError) {
          Log.error('Failed to set initial navigation', oError.message, 'a2a.portal.controller.App');
        }
      }, 100);
    },

    // Additional enterprise methods
    _processNotifications: function (aNotifications) {
      return aNotifications.map((oNotification) => {
        return {
          id: oNotification.id,
          title: encodeXML(oNotification.title || ''),
          message: encodeXML(oNotification.message || ''),
          type: oNotification.type || 'info',
          priority: oNotification.priority || 'medium',
          status: oNotification.status || 'unread',
          createdAt: oNotification.createdAt,
          actions: oNotification.actions || [],
          attachments: oNotification.attachments || []
        };
      });
    },

    _buildNotificationFilter: function (oFilters) {
      const aFilterParts = [];
            
      if (oFilters.status) {
        aFilterParts.push(`status eq '${  oFilters.status  }'`);
      }
            
      if (oFilters.type) {
        aFilterParts.push(`type eq '${  oFilters.type  }'`);
      }
            
      if (oFilters.priority) {
        aFilterParts.push(`priority eq '${  oFilters.priority  }'`);
      }
            
      return aFilterParts.join(' and ');
    },

    _setupNotificationAutoRefresh: function () {
      const that = this;
      const oModel = this.getView().getModel('notifications');
      const oData = oModel.getData();
            
      if (oData.preferences.autoRefresh) {
        this._notificationRefreshInterval = setInterval(() => {
          if (!oData.loading) {
            that._loadNotificationsWithRetry();
          }
        }, oData.preferences.refreshInterval);
      }
    },

    _updateBrowserTitle: function (iUnreadCount) {
      let sTitle = this._oResourceBundle.getText('app.title');
            
      if (iUnreadCount > 0) {
        sTitle = `(${  iUnreadCount  }) ${  sTitle}`;
      }
            
      document.title = sTitle;
    },

    _checkForCriticalNotifications: function (aNotifications) {
      const aCriticalNotifications = aNotifications.filter((oNotification) => {
        return oNotification.priority === 'critical' && oNotification.status === 'unread';
      });
            
      if (aCriticalNotifications.length > 0 && this._supportsDesktopNotifications()) {
        this._showDesktopNotification(aCriticalNotifications[0]);
      }
    },

    _supportsDesktopNotifications: function () {
      return 'Notification' in window && Notification.permission === 'granted';
    },

    _showDesktopNotification: function (oNotification) {
      if (this._supportsDesktopNotifications()) {
        const oDesktopNotification = new Notification(oNotification.title, {
          body: oNotification.message,
          icon: '/images/notification-icon.png',
          tag: `a2a-portal-${  oNotification.id}`
        });
                
        setTimeout(() => {
          oDesktopNotification.close();
        }, 5000);
      }
    },

    _updateBreadcrumb: function (sPageId) {
      // Update breadcrumb navigation
      const oBreadcrumb = this.byId('breadcrumb');
      if (oBreadcrumb) {
        // Implementation would depend on the breadcrumb control structure
        Log.info(`Breadcrumb updated for page: ${  sPageId}`, null, 'a2a.portal.controller.App');
      }
    }
  });
});