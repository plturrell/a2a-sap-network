import fs from 'fs-extra';
import path from 'path';
import crypto from 'crypto';
import { ConfigTemplate, EnvironmentConfig } from './types';
import { defaultTemplates } from './templates';

export class ConfigManager {
  private projectPath: string;

  constructor(projectPath: string) {
    this.projectPath = projectPath;
  }

  async generateConfig(options: any): Promise<void> {
    // Generate environment-specific configurations
    const configs = {
      development: this.generateDevConfig(options),
      staging: this.generateStagingConfig(options),
      production: this.generateProdConfig(options)
    };

    // Write .env files
    await this.writeEnvFile('.env', configs.development);
    await this.writeEnvFile('.env.development', configs.development);
    await this.writeEnvFile('.env.staging', configs.staging);
    await this.writeEnvFile('.env.production', configs.production);

    // Write a2a.config.js
    await this.writeA2AConfig(options);

    // Write docker-compose for development
    if (options.projectType !== 'agent') {
      await this.writeDockerCompose(options);
    }
  }

  private generateDevConfig(options: any): EnvironmentConfig {
    const config: EnvironmentConfig = {
      // Core settings
      NODE_ENV: 'development',
      PORT: '3000',
      LOG_LEVEL: 'debug',
      
      // Security (auto-generated for dev)
      JWT_SECRET: this.generateSecret(),
      SESSION_SECRET: this.generateSecret(),
      
      // Registry
      A2A_REGISTRY_URL: 'http://localhost:3000',
      A2A_REGISTRY_API_KEY: 'dev-key-' + this.generateSecret(8),
      
      // Agent settings
      AGENT_NAME: options.projectName,
      AGENT_TYPE: options.agentType || 'custom',
      AGENT_VERSION: '0.1.0',
      AGENT_HEALTH_CHECK_INTERVAL: '30000',
      
      // Database
      DATABASE_TYPE: options.database,
      DATABASE_URL: this.getDatabaseUrl(options.database, 'development'),
      
      // Optional features
      ...(options.useBlockchain && this.getBlockchainDevConfig()),
      ...(options.useSAP && this.getSAPDevConfig()),
    };

    return config;
  }

  private generateStagingConfig(options: any): EnvironmentConfig {
    const devConfig = this.generateDevConfig(options);
    return {
      ...devConfig,
      NODE_ENV: 'staging',
      PORT: '3001',
      LOG_LEVEL: 'info',
      A2A_REGISTRY_URL: '${A2A_REGISTRY_URL}', // Placeholder for real URL
      DATABASE_URL: this.getDatabaseUrl(options.database, 'staging'),
      // Staging should have real secrets
      JWT_SECRET: '${JWT_SECRET}',
      SESSION_SECRET: '${SESSION_SECRET}',
    };
  }

  private generateProdConfig(options: any): EnvironmentConfig {
    const devConfig = this.generateDevConfig(options);
    return {
      ...devConfig,
      NODE_ENV: 'production',
      PORT: '${PORT}',
      LOG_LEVEL: 'warn',
      A2A_REGISTRY_URL: '${A2A_REGISTRY_URL}',
      DATABASE_URL: '${DATABASE_URL}',
      JWT_SECRET: '${JWT_SECRET}',
      SESSION_SECRET: '${SESSION_SECRET}',
      A2A_REGISTRY_API_KEY: '${A2A_REGISTRY_API_KEY}',
    };
  }

  private getBlockchainDevConfig(): Partial<EnvironmentConfig> {
    return {
      // Blockchain settings
      BLOCKCHAIN_ENABLED: 'true',
      BLOCKCHAIN_NETWORK: 'localhost',
      BLOCKCHAIN_RPC_URL: 'http://localhost:8545',
      BLOCKCHAIN_CHAIN_ID: '31337',
      
      // Auto-generated dev wallet
      BLOCKCHAIN_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
      
      // Contract addresses (will be populated after deployment)
      CONTRACT_AGENT_REGISTRY: '',
      CONTRACT_SERVICE_MARKETPLACE: '',
      CONTRACT_REPUTATION_SYSTEM: '',
      
      // Dev settings
      BLOCKCHAIN_AUTO_MINE: 'true',
      BLOCKCHAIN_GAS_PRICE: '0',
      BLOCKCHAIN_CONFIRMATIONS: '1',
    };
  }

  private getSAPDevConfig(): Partial<EnvironmentConfig> {
    return {
      // SAP settings
      SAP_ENABLED: 'true',
      SAP_SYSTEM_URL: 'http://localhost:4004',
      SAP_CLIENT: '100',
      SAP_AUTH_TYPE: 'basic',
      SAP_USERNAME: 'developer',
      SAP_PASSWORD: 'developer123',
      
      // XSUAA mock for local dev
      XSUAA_URL: 'http://localhost:8080/uaa',
      XSUAA_CLIENT_ID: 'sb-a2a-dev',
      XSUAA_CLIENT_SECRET: this.generateSecret(),
      
      // CAP settings
      CDS_REQUIRES_AUTH: 'false',
      CDS_REQUIRES_DB: 'true',
    };
  }

  private getDatabaseUrl(dbType: string, env: string): string {
    switch (dbType) {
      case 'sqlite':
        return env === 'development' 
          ? 'sqlite:///:memory:' 
          : `sqlite://./data/${env}.db`;
      case 'hana':
        return env === 'development'
          ? 'hana://localhost:30015'
          : '${DATABASE_URL}';
      default:
        return '';
    }
  }

  private generateSecret(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  private async writeEnvFile(filename: string, config: EnvironmentConfig): Promise<void> {
    const envPath = path.join(this.projectPath, filename);
    let content = '# A2A Framework Configuration\n';
    content += `# Generated at: ${new Date().toISOString()}\n\n`;

    // Group related configs
    const groups = {
      'Core Settings': ['NODE_ENV', 'PORT', 'LOG_LEVEL'],
      'Security': ['JWT_SECRET', 'SESSION_SECRET'],
      'Agent Configuration': ['AGENT_NAME', 'AGENT_TYPE', 'AGENT_VERSION', 'AGENT_HEALTH_CHECK_INTERVAL'],
      'Registry': ['A2A_REGISTRY_URL', 'A2A_REGISTRY_API_KEY'],
      'Database': ['DATABASE_TYPE', 'DATABASE_URL'],
      'Blockchain': Object.keys(config).filter(k => k.startsWith('BLOCKCHAIN_') || k.startsWith('CONTRACT_')),
      'SAP Integration': Object.keys(config).filter(k => k.startsWith('SAP_') || k.startsWith('XSUAA_') || k.startsWith('CDS_')),
    };

    for (const [groupName, keys] of Object.entries(groups)) {
      const groupKeys = keys.filter(k => k in config);
      if (groupKeys.length > 0) {
        content += `# ${groupName}\n`;
        for (const key of groupKeys) {
          content += `${key}=${config[key]}\n`;
        }
        content += '\n';
      }
    }

    await fs.writeFile(envPath, content);
  }

  private async writeA2AConfig(options: any): Promise<void> {
    const configPath = path.join(this.projectPath, 'a2a.config.js');
    const template = defaultTemplates[options.projectType] || defaultTemplates.agent;
    
    const config = `/**
 * A2A Framework Configuration
 * @type {import('@a2a/types').A2AConfig}
 */
module.exports = {
  // Project metadata
  name: '${options.projectName}',
  version: '0.1.0',
  type: '${options.projectType}',

  // Agent configuration
  agent: {
    type: '${options.agentType || 'custom'}',
    capabilities: ${JSON.stringify(template.capabilities || [], null, 4)},
    
    // Auto-discovery settings
    discovery: {
      enabled: true,
      interval: 30000,
      timeout: 5000,
    },
    
    // Health check
    health: {
      enabled: true,
      endpoint: '/health',
      interval: 30000,
    },
  },

  // Development settings
  development: {
    // Auto-reload on changes
    watch: true,
    
    // Mock external services
    mocks: {
      registry: ${options.projectType === 'agent'},
      blockchain: ${!options.useBlockchain},
      sap: ${!options.useSAP},
    },
    
    // Development tools
    tools: {
      inspector: true,
      profiler: false,
      metrics: true,
    },
  },

  // Network settings
  network: {
    // Retry configuration
    retry: {
      attempts: 3,
      delay: 1000,
      backoff: 2,
    },
    
    // Circuit breaker
    circuitBreaker: {
      enabled: true,
      threshold: 5,
      timeout: 60000,
    },
  },

  // Blockchain settings
  blockchain: {
    enabled: ${options.useBlockchain},
    networks: {
      development: {
        url: 'http://localhost:8545',
        accounts: ['0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80'],
      },
    },
  },

  // SAP settings
  sap: {
    enabled: ${options.useSAP},
    cap: {
      root: './srv',
      model: './db',
    },
  },

  // Logging
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: 'json',
    transports: ['console'],
  },
};
`;

    await fs.writeFile(configPath, config);
  }

  private async writeDockerCompose(options: any): Promise<void> {
    const dockerPath = path.join(this.projectPath, 'docker-compose.dev.yml');
    
    let services: any = {
      registry: {
        image: 'a2a/registry:latest',
        ports: ['3000:3000'],
        environment: {
          NODE_ENV: 'development',
        },
      },
    };

    if (options.useBlockchain) {
      services.blockchain = {
        image: 'ethereum/client-go:latest',
        command: '--dev --http --http.addr 0.0.0.0 --http.api eth,net,web3,debug,personal --http.corsdomain "*"',
        ports: ['8545:8545'],
      };
    }

    if (options.database === 'hana') {
      services.hana = {
        image: 'saplabs/hanaexpress:latest',
        ports: ['30015:30015'],
        environment: {
          MASTER_PASSWORD: 'HanaExpress1',
        },
      };
    }

    const compose = {
      version: '3.8',
      services,
      networks: {
        a2a: {
          driver: 'bridge',
        },
      },
    };

    await fs.writeFile(dockerPath, `# A2A Development Environment\n${JSON.stringify(compose, null, 2)}`);
  }
}