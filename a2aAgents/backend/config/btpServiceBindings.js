'use strict';
/* No global declarations needed - globals not used in this file */

/**
 * BTP Service Bindings Configuration
 * Handles service bindings from Cloud Foundry VCAP_SERVICES
 */

const xsenv = require('@sap/xsenv');

class BTPServiceBindings {
  constructor() {
    this.services = {};
    this.loadServiceBindings();
  }

  loadServiceBindings() {
    try {
      // Load VCAP_SERVICES from environment
      const vcapServices = xsenv.getServices({
        // Database service
        hana: { label: 'hanatrial' },
        hanaCloud: { label: 'hana' },
                
        // Authentication
        xsuaa: { label: 'xsuaa' },
                
        // Destination service for external connections
        destination: { label: 'destination' },
                
        // Connectivity for on-premise systems
        connectivity: { label: 'connectivity' },
                
        // Redis for caching
        redis: { label: 'redis-cache' },
                
        // Service Manager for service-to-service communication
        serviceManager: { label: 'service-manager' },
                
        // Application logging
        appLogs: { label: 'application-logs' },
                
        // Alert notification
        alertNotification: { label: 'alert-notification' },
                
        // Auto-scaler
        autoscaler: { label: 'autoscaler' }
      });

      this.services = vcapServices;
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('✅ BTP service bindings loaded successfully');
            
    } catch (error) {
      console.error('❌ Failed to load BTP service bindings:', error.message);
            
      // Fallback for local development
      this.loadLocalFallbacks();
    }
  }

  loadLocalFallbacks() {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('⚠️ Using local development fallbacks for BTP services');
        
    this.services = {
      hana: {
        host: process.env.HANA_HOST || 'localhost',
        port: process.env.HANA_PORT || '30015',
        user: process.env.HANA_USER || 'SYSTEM',
        password: process.env.HANA_PASSWORD || '',
        database: process.env.HANA_DATABASE || 'A2A',
        schema: process.env.HANA_SCHEMA || 'A2A_AGENTS',
        encrypt: true,
        sslValidateCertificate: false
      },
      xsuaa: {
        clientid: process.env.XSUAA_CLIENT_ID || 'local-client',
        clientsecret: process.env.XSUAA_CLIENT_SECRET || 'local-secret',
        url: process.env.XSUAA_URL || 'http://localhost:8080/uaa',
        xsappname: process.env.XSUAA_XSAPPNAME || 'a2a-agents-local'
      },
      redis: {
        hostname: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || '6379',
        password: process.env.REDIS_PASSWORD || '',
        tls: process.env.REDIS_TLS === 'true'
      }
    };
  }

  /**
     * Get HANA database configuration for connection
     */
  getHANAConfig() {
    const hanaService = this.services.hana || this.services.hanaCloud;
        
    if (!hanaService) {
      throw new Error('HANA service binding not found. Ensure HANA service is bound to the application.');
    }

    // BTP HANA Cloud service binding format
    if (hanaService.credentials) {
      const creds = hanaService.credentials;
      return {
        host: creds.host || creds.hostname,
        port: creds.port || creds.sql_port || 443,
        user: creds.user || creds.hdi_user,
        password: creds.password || creds.hdi_password,
        database: creds.database || creds.db_name,
        schema: creds.schema || process.env.HANA_SCHEMA || 'A2A_AGENTS',
        encrypt: true,
        sslValidateCertificate: true,
        // Connection pool settings for BTP
        pool_size: parseInt(process.env.HANA_POOL_SIZE) || 10,
        max_overflow: parseInt(process.env.HANA_MAX_OVERFLOW) || 20,
        pool_timeout: parseInt(process.env.HANA_POOL_TIMEOUT) || 30,
        pool_recycle: parseInt(process.env.HANA_POOL_RECYCLE) || 3600,
        pool_pre_ping: true,
        // HANA-specific optimizations
        isolation_level: 'READ_COMMITTED',
        connection_timeout: 30,
        compress: true
      };
    }

    // Fallback to local config
    return this.services.hana;
  }

  /**
     * Get XSUAA configuration for authentication
     */
  getXSUAAConfig() {
    const xsuaaService = this.services.xsuaa;
        
    if (!xsuaaService) {
      throw new Error('XSUAA service binding not found. Ensure XSUAA service is bound to the application.');
    }

    if (xsuaaService.credentials) {
      const creds = xsuaaService.credentials;
      return {
        clientId: creds.clientid,
        clientSecret: creds.clientsecret,
        url: creds.url,
        uaaDomain: creds.uaadomain,
        xsAppName: creds.xsappname,
        identityZone: creds.identityzone,
        identityZoneId: creds.identityzoneid,
        tenantId: creds.tenantid,
        tenantMode: creds.tenantmode || 'dedicated',
        // JWT validation settings
        verificationKey: creds.verificationkey,
        trustedClientIdSuffix: creds.trustedclientidsuffix
      };
    }

    return this.services.xsuaa;
  }

  /**
     * Get Destination service configuration
     */
  getDestinationConfig() {
    const destService = this.services.destination;
        
    if (!destService) {
      console.warn('⚠️ Destination service not bound - external system access may be limited');
      return null;
    }

    if (destService.credentials) {
      return {
        uri: destService.credentials.uri,
        clientId: destService.credentials.clientid,
        clientSecret: destService.credentials.clientsecret,
        url: destService.credentials.url,
        tokenServiceUrl: destService.credentials.tokenServiceUrl
      };
    }

    return destService;
  }

  /**
     * Get Redis configuration for caching
     */
  getRedisConfig() {
    const redisService = this.services.redis;
        
    if (!redisService) {
      console.warn('⚠️ Redis service not bound - caching will be disabled');
      return null;
    }

    if (redisService.credentials) {
      const creds = redisService.credentials;
      return {
        host: creds.hostname || creds.host,
        port: creds.port || 6379,
        password: creds.password,
        tls: creds.tls_enabled || false,
        // Redis configuration for BTP
        db: 0,
        retryDelayOnFailover: 100,
        maxRetriesPerRequest: 3,
        lazyConnect: true,
        keepAlive: 30000,
        // Connection pool
        family: 4,
        connectTimeout: 10000,
        commandTimeout: 5000
      };
    }

    return this.services.redis;
  }

  /**
     * Get Service Manager configuration for service-to-service calls
     */
  getServiceManagerConfig() {
    const smService = this.services.serviceManager;
        
    if (!smService) {
      console.warn('⚠️ Service Manager not bound - service-to-service calls may fail');
      return null;
    }

    if (smService.credentials) {
      return {
        url: smService.credentials.url,
        clientId: smService.credentials.clientid,
        clientSecret: smService.credentials.clientsecret,
        tokenUrl: smService.credentials.token_url || `${smService.credentials.url  }/oauth/token`
      };
    }

    return smService;
  }

  /**
     * Get application logging configuration
     */
  getLoggingConfig() {
    const loggingService = this.services.appLogs;
        
    if (!loggingService) {
      console.warn('⚠️ Application logging service not bound - using console logging');
      return null;
    }

    if (loggingService.credentials) {
      return {
        endpoint: loggingService.credentials.endpoint,
        user: loggingService.credentials.user,
        password: loggingService.credentials.password,
        // Logging configuration
        level: process.env.LOG_LEVEL || 'info',
        format: process.env.LOG_FORMAT || 'json',
        retention: process.env.LOG_RETENTION_DAYS || '7'
      };
    }

    return loggingService;
  }

  /**
     * Get alert notification configuration
     */
  getAlertConfig() {
    const alertService = this.services.alertNotification;
        
    if (!alertService) {
      console.warn('⚠️ Alert notification service not bound - alerts will be logged only');
      return null;
    }

    if (alertService.credentials) {
      return {
        url: alertService.credentials.url,
        clientId: alertService.credentials.client_id,
        clientSecret: alertService.credentials.client_secret,
        oauth2Url: alertService.credentials.oauth2_url
      };
    }

    return alertService;
  }

  /**
     * Validate all critical service bindings
     */
  validateCriticalServices() {
    const critical = ['hana', 'xsuaa'];
    const missing = [];
        
    for (const service of critical) {
      if (!this.services[service] && !this.services[`${service  }Cloud`]) {
        missing.push(service);
      }
    }
        
    if (missing.length > 0) {
      throw new Error(`Critical BTP services not bound: ${missing.join(', ')}`);
    }
        
         
        
    // eslint-disable-next-line no-console
        
         
        
    // eslint-disable-next-line no-console
    console.log('✅ All critical BTP services validated');
    return true;
  }

  /**
     * Get all service binding status
     */
  getServiceStatus() {
    const status = {};
        
    const serviceChecks = {
      'HANA Database': () => this.getHANAConfig(),
      'XSUAA Authentication': () => this.getXSUAAConfig(),
      'Destination Service': () => this.getDestinationConfig(),
      'Redis Cache': () => this.getRedisConfig(),
      'Service Manager': () => this.getServiceManagerConfig(),
      'Application Logging': () => this.getLoggingConfig(),
      'Alert Notification': () => this.getAlertConfig()
    };
        
    for (const [name, checkFn] of Object.entries(serviceChecks)) {
      try {
        const config = checkFn();
        status[name] = config ? 'BOUND' : 'NOT_BOUND';
      } catch (error) {
        status[name] = `ERROR: ${  error.message}`;
      }
    }
        
    return status;
  }
}

// Export singleton instance
module.exports = new BTPServiceBindings();