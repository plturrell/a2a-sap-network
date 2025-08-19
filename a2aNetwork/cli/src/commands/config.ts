import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import fs from 'fs-extra';
import path from 'path';
import dotenv from 'dotenv';

export const configCommand = new Command('config')
  .description('Configuration management')
  .command('set <key> <value>')
  .description('Set configuration value')
  .action(async (key: string, value: string) => {
    const envPath = path.join(process.cwd(), '.env');
    
    // Read existing .env
    let envConfig: any = {};
    if (await fs.pathExists(envPath)) {
      const envContent = await fs.readFile(envPath, 'utf8');
      envConfig = dotenv.parse(envContent);
    }
    
    // Update value
    envConfig[key] = value;
    
    // Write back to .env
    const newContent = Object.entries(envConfig)
      .map(([k, v]) => `${k}=${v}`)
      .join('\n');
    
    await fs.writeFile(envPath, newContent);
    
    console.log(chalk.green(`✓ Set ${key} = ${value}`));
  });

configCommand
  .command('get [key]')
  .description('Get configuration value(s)')
  .action(async (key?: string) => {
    const envPath = path.join(process.cwd(), '.env');
    
    if (!(await fs.pathExists(envPath))) {
      console.log(chalk.red('No .env file found'));
      return;
    }
    
    const envContent = await fs.readFile(envPath, 'utf8');
    const envConfig = dotenv.parse(envContent);
    
    if (key) {
      const value = envConfig[key];
      if (value !== undefined) {
        console.log(chalk.cyan(key) + '=' + chalk.white(value));
      } else {
        console.log(chalk.red(`Key '${key}' not found`));
      }
    } else {
      // Show all config
      console.log(chalk.blue('\nCurrent configuration:\n'));
      
      const groups = {
        'Core': ['NODE_ENV', 'PORT', 'LOG_LEVEL'],
        'Agent': ['AGENT_NAME', 'AGENT_TYPE', 'AGENT_VERSION'],
        'Registry': ['A2A_REGISTRY_URL', 'A2A_REGISTRY_API_KEY'],
        'Database': ['DATABASE_TYPE', 'DATABASE_URL'],
        'Blockchain': Object.keys(envConfig).filter(k => k.startsWith('BLOCKCHAIN_')),
        'SAP': Object.keys(envConfig).filter(k => k.startsWith('SAP_') || k.startsWith('XSUAA_')),
        'Other': Object.keys(envConfig).filter(k => {
          const knownKeys = [
            'NODE_ENV', 'PORT', 'LOG_LEVEL',
            'AGENT_NAME', 'AGENT_TYPE', 'AGENT_VERSION',
            'A2A_REGISTRY_URL', 'A2A_REGISTRY_API_KEY',
            'DATABASE_TYPE', 'DATABASE_URL'
          ];
          return !knownKeys.includes(k) && 
                 !k.startsWith('BLOCKCHAIN_') && 
                 !k.startsWith('SAP_') && 
                 !k.startsWith('XSUAA_');
        })
      };
      
      for (const [group, keys] of Object.entries(groups)) {
        const groupKeys = keys.filter(k => k in envConfig);
        if (groupKeys.length > 0) {
          console.log(chalk.yellow(`${group}:`));
          for (const k of groupKeys) {
            const value = envConfig[k];
            const displayValue = k.includes('SECRET') || k.includes('KEY') || k.includes('PASSWORD')
              ? '*'.repeat(Math.min(value.length, 8))
              : value;
            console.log(`  ${chalk.cyan(k)}=${chalk.white(displayValue)}`);
          }
          console.log();
        }
      }
    }
  });

configCommand
  .command('wizard')
  .description('Interactive configuration wizard')
  .action(async () => {
    console.log(chalk.blue('\n🧙 A2A Configuration Wizard\n'));
    
    const questions = [
      {
        type: 'input',
        name: 'AGENT_NAME',
        message: 'Agent name:',
        default: path.basename(process.cwd())
      },
      {
        type: 'list',
        name: 'AGENT_TYPE',
        message: 'Agent type:',
        choices: [
          'data-processor',
          'ai-ml',
          'storage',
          'orchestrator',
          'analytics',
          'custom'
        ]
      },
      {
        type: 'input',
        name: 'A2A_REGISTRY_URL',
        message: 'Registry URL:',
        default: 'http://localhost:3000'
      },
      {
        type: 'list',
        name: 'DATABASE_TYPE',
        message: 'Database type:',
        choices: ['sqlite', 'hana', 'none']
      },
      {
        type: 'confirm',
        name: 'enableBlockchain',
        message: 'Enable blockchain features?',
        default: false
      },
      {
        type: 'input',
        name: 'BLOCKCHAIN_RPC_URL',
        message: 'Blockchain RPC URL:',
        default: 'http://localhost:8545',
        when: (answers: any) => answers.enableBlockchain
      },
      {
        type: 'confirm',
        name: 'enableSAP',
        message: 'Enable SAP integration?',
        default: false
      },
      {
        type: 'input',
        name: 'SAP_SYSTEM_URL',
        message: 'SAP system URL:',
        default: 'http://localhost:4004',
        when: (answers: any) => answers.enableSAP
      }
    ];
    
    const answers = await inquirer.prompt(questions);
    
    // Generate .env content
    const envConfig: any = {
      NODE_ENV: 'development',
      PORT: '3001',
      LOG_LEVEL: 'debug',
      AGENT_NAME: answers.AGENT_NAME,
      AGENT_TYPE: answers.AGENT_TYPE,
      A2A_REGISTRY_URL: answers.A2A_REGISTRY_URL,
      DATABASE_TYPE: answers.DATABASE_TYPE
    };
    
    if (answers.enableBlockchain) {
      envConfig.BLOCKCHAIN_ENABLED = 'true';
      envConfig.BLOCKCHAIN_RPC_URL = answers.BLOCKCHAIN_RPC_URL;
      envConfig.BLOCKCHAIN_NETWORK = 'localhost';
    }
    
    if (answers.enableSAP) {
      envConfig.SAP_ENABLED = 'true';
      envConfig.SAP_SYSTEM_URL = answers.SAP_SYSTEM_URL;
    }
    
    // Write .env file
    const envContent = Object.entries(envConfig)
      .map(([key, value]) => `${key}=${value}`)
      .join('\n');
    
    const envPath = path.join(process.cwd(), '.env');
    await fs.writeFile(envPath, `# A2A Configuration\n# Generated by configuration wizard\n\n${envContent}\n`);
    
    console.log(chalk.green('\n✨ Configuration saved to .env\n'));
    
    // Show summary
    console.log(chalk.cyan('Configuration summary:'));
    for (const [key, value] of Object.entries(envConfig)) {
      console.log(chalk.gray(`  ${key}=${value}`));
    }
    console.log();
  });

configCommand
  .command('validate')
  .description('Validate current configuration')
  .action(async () => {
    console.log(chalk.blue('\n🔍 Validating A2A configuration...\n'));
    
    const issues: string[] = [];
    const warnings: string[] = [];
    
    // Check for .env file
    const envPath = path.join(process.cwd(), '.env');
    if (!(await fs.pathExists(envPath))) {
      issues.push('Missing .env file');
    } else {
      const envContent = await fs.readFile(envPath, 'utf8');
      const envConfig = dotenv.parse(envContent);
      
      // Required fields
      const required = ['AGENT_NAME', 'AGENT_TYPE'];
      for (const field of required) {
        if (!envConfig[field]) {
          issues.push(`Missing required field: ${field}`);
        }
      }
      
      // Port validation
      if (envConfig.PORT) {
        const port = parseInt(envConfig.PORT);
        if (isNaN(port) || port < 1 || port > 65535) {
          issues.push('Invalid PORT value');
        }
      }
      
      // URL validation
      const urlFields = ['A2A_REGISTRY_URL', 'BLOCKCHAIN_RPC_URL', 'SAP_SYSTEM_URL'];
      for (const field of urlFields) {
        if (envConfig[field]) {
          try {
            new URL(envConfig[field]);
          } catch {
            issues.push(`Invalid URL format: ${field}`);
          }
        }
      }
      
      // Database validation
      if (envConfig.DATABASE_TYPE === 'hana' && !envConfig.DATABASE_URL) {
        warnings.push('HANA database selected but no DATABASE_URL specified');
      }
      
      // Blockchain validation
      if (envConfig.BLOCKCHAIN_ENABLED === 'true') {
        if (!envConfig.BLOCKCHAIN_RPC_URL) {
          issues.push('Blockchain enabled but no RPC URL specified');
        }
      }
    }
    
    // Check for package.json
    const packagePath = path.join(process.cwd(), 'package.json');
    if (!(await fs.pathExists(packagePath))) {
      issues.push('Missing package.json file');
    }
    
    // Report results
    if (issues.length === 0 && warnings.length === 0) {
      console.log(chalk.green('✅ Configuration is valid!\n'));
    } else {
      if (issues.length > 0) {
        console.log(chalk.red('❌ Configuration issues:'));
        for (const issue of issues) {
          console.log(chalk.red(`  - ${issue}`));
        }
        console.log();
      }
      
      if (warnings.length > 0) {
        console.log(chalk.yellow('⚠️  Configuration warnings:'));
        for (const warning of warnings) {
          console.log(chalk.yellow(`  - ${warning}`));
        }
        console.log();
      }
    }
  });