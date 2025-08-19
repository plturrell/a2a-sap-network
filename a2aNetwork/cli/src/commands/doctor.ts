import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import which from 'which';
import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'child_process';

export const doctorCommand = new Command('doctor')
  .description('Diagnose and fix common issues')
  .action(async () => {
    console.log(chalk.blue('\n🏥 A2A Health Check\n'));
    
    const checks = [
      checkNodeVersion,
      checkNpmDependencies,
      checkEnvironmentFile,
      checkConfiguration,
      checkBlockchainTools,
      checkServices,
      checkConnectivity
    ];
    
    let totalIssues = 0;
    
    for (const check of checks) {
      const issues = await check();
      totalIssues += issues;
    }
    
    console.log('\n' + '='.repeat(50));
    
    if (totalIssues === 0) {
      console.log(chalk.green('\n🎉 Everything looks good! Your A2A environment is healthy.\n'));
    } else {
      console.log(chalk.yellow(`\n⚠️  Found ${totalIssues} issue(s). See suggestions above.\n`));
    }
  });

async function checkNodeVersion(): Promise<number> {
  const spinner = ora('Checking Node.js version...').start();
  let issues = 0;
  
  try {
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.substring(1).split('.')[0]);
    
    if (majorVersion >= 16) {
      spinner.succeed(`Node.js version: ${nodeVersion} ✓`);
    } else {
      spinner.fail(`Node.js version: ${nodeVersion} (requires >= 16.0.0)`);
      console.log(chalk.yellow('  💡 Update Node.js: https://nodejs.org/'));
      issues++;
    }
  } catch (error) {
    spinner.fail('Node.js not found');
    console.log(chalk.yellow('  💡 Install Node.js: https://nodejs.org/'));
    issues++;
  }
  
  return issues;
}

async function checkNpmDependencies(): Promise<number> {
  const spinner = ora('Checking dependencies...').start();
  let issues = 0;
  
  try {
    const packagePath = path.join(process.cwd(), 'package.json');
    
    if (!(await fs.pathExists(packagePath))) {
      spinner.warn('No package.json found (not in A2A project?)');
      return 0;
    }
    
    const nodeModulesPath = path.join(process.cwd(), 'node_modules');
    
    if (!(await fs.pathExists(nodeModulesPath))) {
      spinner.fail('Dependencies not installed');
      console.log(chalk.yellow('  💡 Run: npm install'));
      issues++;
    } else {
      // Check for A2A SDK
      const a2aSdkPath = path.join(nodeModulesPath, '@a2a', 'sdk');
      
      if (await fs.pathExists(a2aSdkPath)) {
        spinner.succeed('Dependencies installed ✓');
      } else {
        spinner.warn('A2A SDK not found in dependencies');
        console.log(chalk.yellow('  💡 Run: npm install @a2a/sdk'));
        issues++;
      }
    }
  } catch (error) {
    spinner.fail('Error checking dependencies');
    issues++;
  }
  
  return issues;
}

async function checkEnvironmentFile(): Promise<number> {
  const spinner = ora('Checking environment configuration...').start();
  let issues = 0;
  
  try {
    const envPath = path.join(process.cwd(), '.env');
    
    if (!(await fs.pathExists(envPath))) {
      spinner.fail('No .env file found');
      console.log(chalk.yellow('  💡 Run: a2a config wizard'));
      issues++;
    } else {
      const envContent = await fs.readFile(envPath, 'utf8');
      const requiredVars = ['AGENT_NAME', 'AGENT_TYPE'];
      const missingVars = requiredVars.filter(v => !envContent.includes(v + '='));
      
      if (missingVars.length > 0) {
        spinner.fail(`Missing environment variables: ${missingVars.join(', ')}`);
        console.log(chalk.yellow('  💡 Run: a2a config wizard'));
        issues++;
      } else {
        spinner.succeed('Environment configuration ✓');
      }
    }
  } catch (error) {
    spinner.fail('Error checking environment file');
    issues++;
  }
  
  return issues;
}

async function checkConfiguration(): Promise<number> {
  const spinner = ora('Checking A2A configuration...').start();
  let issues = 0;
  
  try {
    const configPath = path.join(process.cwd(), 'a2a.config.js');
    
    if (!(await fs.pathExists(configPath))) {
      spinner.warn('No a2a.config.js found (using defaults)');
    } else {
      try {
        const config = require(configPath);
        
        if (!config.name || !config.agent) {
          spinner.fail('Invalid A2A configuration');
          console.log(chalk.yellow('  💡 Regenerate config: a2a init --reconfigure'));
          issues++;
        } else {
          spinner.succeed('A2A configuration ✓');
        }
      } catch (error) {
        spinner.fail('Error parsing A2A configuration');
        console.log(chalk.yellow('  💡 Check syntax: a2a config validate'));
        issues++;
      }
    }
  } catch (error) {
    spinner.fail('Error checking A2A configuration');
    issues++;
  }
  
  return issues;
}

async function checkBlockchainTools(): Promise<number> {
  const spinner = ora('Checking blockchain tools...').start();
  let issues = 0;
  
  try {
    // Check if blockchain is enabled
    const envPath = path.join(process.cwd(), '.env');
    if (await fs.pathExists(envPath)) {
      const envContent = await fs.readFile(envPath, 'utf8');
      
      if (envContent.includes('BLOCKCHAIN_ENABLED=true')) {
        // Check for foundry tools
        try {
          await which('forge');
          await which('anvil');
          spinner.succeed('Blockchain tools (Foundry) ✓');
        } catch (error) {
          spinner.fail('Foundry not found');
          console.log(chalk.yellow('  💡 Install Foundry: curl -L https://foundry.paradigm.xyz | bash'));
          issues++;
        }
      } else {
        spinner.succeed('Blockchain tools (not required) ✓');
      }
    } else {
      spinner.succeed('Blockchain tools (not configured) ✓');
    }
  } catch (error) {
    spinner.fail('Error checking blockchain tools');
    issues++;
  }
  
  return issues;
}

async function checkServices(): Promise<number> {
  const spinner = ora('Checking service availability...').start();
  let issues = 0;
  
  try {
    // Check if main app can start
    const packagePath = path.join(process.cwd(), 'package.json');
    
    if (await fs.pathExists(packagePath)) {
      const pkg = JSON.parse(await fs.readFile(packagePath, 'utf8'));
      
      if (pkg.main && await fs.pathExists(path.join(process.cwd(), pkg.main))) {
        spinner.succeed('Main application file ✓');
      } else {
        spinner.fail('Main application file not found');
        console.log(chalk.yellow('  💡 Check package.json main field'));
        issues++;
      }
    } else {
      spinner.warn('No package.json found');
    }
  } catch (error) {
    spinner.fail('Error checking services');
    issues++;
  }
  
  return issues;
}

async function checkConnectivity(): Promise<number> {
  const spinner = ora('Checking network connectivity...').start();
  let issues = 0;
  
  try {
    // Simple connectivity check
    const { default: fetch } = await import('node-fetch');
    
    try {
      const response = await fetch('https://registry.npmjs.org/@a2a/sdk', { 
        timeout: 5000 
      });
      
      if (response.ok) {
        spinner.succeed('Network connectivity ✓');
      } else {
        spinner.warn('Limited network connectivity');
      }
    } catch (error) {
      spinner.warn('Network connectivity check failed');
      console.log(chalk.yellow('  💡 Check internet connection'));
    }
  } catch (error) {
    spinner.succeed('Network connectivity (check skipped) ✓');
  }
  
  return issues;
}