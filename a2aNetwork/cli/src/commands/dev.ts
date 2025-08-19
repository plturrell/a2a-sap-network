import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { spawn } from 'child_process';
import fs from 'fs-extra';
import path from 'path';

export const devCommand = new Command('dev')
  .description('Development commands')
  .command('start')
  .option('-m, --mock', 'Use mock services')
  .option('-b, --blockchain', 'Start local blockchain')
  .option('-r, --registry', 'Start local registry')
  .option('-a, --all', 'Start all services')
  .description('Start development environment')
  .action(async (options: any) => {
    console.log(chalk.blue('\n🚀 Starting A2A development environment...\n'));
    
    const services: any[] = [];
    
    // Check for a2a.config.js
    const configPath = path.join(process.cwd(), 'a2a.config.js');
    let config: any = {};
    
    if (await fs.pathExists(configPath)) {
      config = require(configPath);
    }
    
    // Start registry if requested or if --all
    if (options.registry || options.all) {
      services.push({
        name: 'Registry',
        command: 'docker',
        args: ['run', '-p', '3000:3000', '--rm', 'a2a/registry:latest'],
        fallback: {
          command: 'npx',
          args: ['@a2a/registry', '--port', '3000']
        }
      });
    }
    
    // Start blockchain if requested or if --all
    if (options.blockchain || options.all || config.blockchain?.enabled) {
      services.push({
        name: 'Blockchain',
        command: 'anvil',
        args: ['--port', '8545', '--accounts', '10'],
        fallback: {
          command: 'npx',
          args: ['hardhat', 'node']
        }
      });
    }
    
    // Start main application
    services.push({
      name: 'Application',
      command: 'npm',
      args: ['run', 'dev'],
      cwd: process.cwd()
    });
    
    // Start services
    for (const service of services) {
      await startService(service);
    }
    
    console.log(chalk.green('\n✨ Development environment is running!\n'));
    console.log(chalk.cyan('Services:'));
    services.forEach(s => {
      console.log(chalk.gray(`  - ${s.name}: ${chalk.green('running')}`));
    });
    console.log('\n' + chalk.yellow('Press Ctrl+C to stop all services\n'));
  });

devCommand
  .command('fund')
  .option('-a, --account <address>', 'Account address to fund')
  .option('-n, --amount <amount>', 'Amount to fund (in ETH)', '10')
  .description('Fund development account with test ETH')
  .action(async (options: any) => {
    const spinner = ora('Funding account...').start();
    
    try {
      const Web3 = require('web3');
      const web3 = new Web3('http://localhost:8545');
      
      const accounts = await web3.eth.getAccounts();
      const targetAccount = options.account || accounts[1];
      
      await web3.eth.sendTransaction({
        from: accounts[0],
        to: targetAccount,
        value: web3.utils.toWei(options.amount, 'ether')
      });
      
      spinner.succeed(`Funded ${targetAccount} with ${options.amount} ETH`);
    } catch (error) {
      spinner.fail('Failed to fund account');
      console.error(chalk.red('Make sure local blockchain is running: a2a dev start --blockchain'));
    }
  });

devCommand
  .command('mock')
  .description('Manage mock services')
  .command('create <agent>')
  .option('-s, --services <services>', 'Comma-separated list of services')
  .action(async (agent: string, options: any) => {
    const spinner = ora('Creating mock agent...').start();
    
    try {
      const mockPath = path.join(process.cwd(), '.a2a', 'mocks');
      await fs.ensureDir(mockPath);
      
      const services = options.services 
        ? options.services.split(',').map((s: string) => s.trim())
        : ['process'];
      
      const mockConfig = {
        name: agent,
        type: 'mock',
        services: services.reduce((acc: any, service: string) => {
          acc[service] = {
            response: { status: 'success', data: {} },
            latency: 100
          };
          return acc;
        }, {})
      };
      
      await fs.writeFile(
        path.join(mockPath, `${agent}.json`),
        JSON.stringify(mockConfig, null, 2)
      );
      
      spinner.succeed(`Mock agent '${agent}' created`);
    } catch (error) {
      spinner.fail('Failed to create mock');
      console.error(error);
    }
  });

async function startService(service: any): Promise<void> {
  const spinner = ora(`Starting ${service.name}...`).start();
  
  try {
    const child = spawn(service.command, service.args, {
      cwd: service.cwd || process.cwd(),
      detached: false,
      stdio: 'pipe'
    });
    
    child.on('error', async (err: any) => {
      if (err.code === 'ENOENT' && service.fallback) {
        // Try fallback command
        const fallbackChild = spawn(service.fallback.command, service.fallback.args, {
          cwd: service.cwd || process.cwd(),
          detached: false,
          stdio: 'pipe'
        });
        
        fallbackChild.on('error', () => {
          spinner.fail(`Failed to start ${service.name}`);
        });
        
        fallbackChild.on('spawn', () => {
          spinner.succeed(`${service.name} started (fallback)`);
        });
      } else {
        spinner.fail(`Failed to start ${service.name}`);
      }
    });
    
    child.on('spawn', () => {
      spinner.succeed(`${service.name} started`);
    });
    
    // Handle process cleanup
    process.on('exit', () => {
      child.kill();
    });
    
  } catch (error) {
    spinner.fail(`Failed to start ${service.name}`);
    throw error;
  }
}