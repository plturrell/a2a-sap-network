#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const { CloudDeployer } = require('../src/CloudDeployer');
const { KubernetesDeployer } = require('../src/KubernetesDeployer');
const { DockerBuilder } = require('../src/DockerBuilder');
const { ConfigValidator } = require('../src/ConfigValidator');
const { DeploymentMonitor } = require('../src/DeploymentMonitor');

const program = new Command();

program
  .name('a2a-deploy')
  .description('A2A Framework Production Deployment Tool')
  .version('1.0.0');

// Deploy command
program
  .command('deploy')
  .description('Deploy A2A agents to production')
  .option('-e, --environment <env>', 'Target environment', 'production')
  .option('-c, --config <file>', 'Deployment configuration file', 'deployment.yaml')
  .option('-p, --platform <platform>', 'Target platform (k8s|aws|azure|gcp)', 'k8s')
  .option('--dry-run', 'Perform a dry run without actual deployment')
  .option('--auto-approve', 'Skip confirmation prompts')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n🚀 A2A Production Deployment\n'));
      
      const deployer = getDeployer(options.platform);
      const result = await deployer.deploy(options);
      
      console.log(chalk.green('\n✅ Deployment completed successfully!'));
      console.log(chalk.cyan('Deployment details:'));
      console.log(JSON.stringify(result, null, 2));
      
    } catch (error) {
      console.error(chalk.red('\n❌ Deployment failed:'), error.message);
      process.exit(1);
    }
  });

// Build command
program
  .command('build')
  .description('Build A2A agents for deployment')
  .option('-t, --target <target>', 'Build target (docker|helm|terraform)', 'docker')
  .option('-c, --config <file>', 'Build configuration file', 'a2a.config.js')
  .option('--no-cache', 'Build without using cache')
  .option('--push', 'Push built images to registry')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n🔨 Building A2A agents...\n'));
      
      const builder = new DockerBuilder();
      const result = await builder.build(options);
      
      console.log(chalk.green('\n✅ Build completed successfully!'));
      console.log(chalk.cyan('Build artifacts:'));
      result.artifacts.forEach(artifact => {
        console.log(`  - ${artifact.type}: ${artifact.location}`);
      });
      
    } catch (error) {
      console.error(chalk.red('\n❌ Build failed:'), error.message);
      process.exit(1);
    }
  });

// Validate command
program
  .command('validate')
  .description('Validate deployment configuration')
  .option('-c, --config <file>', 'Configuration file to validate', 'deployment.yaml')
  .option('-e, --environment <env>', 'Target environment', 'production')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n🔍 Validating deployment configuration...\n'));
      
      const validator = new ConfigValidator();
      const result = await validator.validate(options);
      
      if (result.valid) {
        console.log(chalk.green('✅ Configuration is valid!'));
      } else {
        console.log(chalk.red('❌ Configuration validation failed:'));
        result.errors.forEach(error => {
          console.log(chalk.red(`  - ${error}`));
        });
        
        if (result.warnings.length > 0) {
          console.log(chalk.yellow('\n⚠️  Warnings:'));
          result.warnings.forEach(warning => {
            console.log(chalk.yellow(`  - ${warning}`));
          });
        }
        process.exit(1);
      }
      
    } catch (error) {
      console.error(chalk.red('\n❌ Validation failed:'), error.message);
      process.exit(1);
    }
  });

// Monitor command
program
  .command('monitor')
  .description('Monitor deployment status')
  .option('-d, --deployment <name>', 'Deployment name to monitor')
  .option('-e, --environment <env>', 'Environment to monitor', 'production')
  .option('-w, --watch', 'Watch for changes continuously')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n📊 Monitoring A2A deployment...\n'));
      
      const monitor = new DeploymentMonitor();
      
      if (options.watch) {
        await monitor.watch(options);
      } else {
        const status = await monitor.getStatus(options);
        console.log(chalk.cyan('Deployment Status:'));
        console.log(JSON.stringify(status, null, 2));
      }
      
    } catch (error) {
      console.error(chalk.red('\n❌ Monitoring failed:'), error.message);
      process.exit(1);
    }
  });

// Rollback command
program
  .command('rollback')
  .description('Rollback to previous deployment')
  .option('-d, --deployment <name>', 'Deployment name')
  .option('-v, --version <version>', 'Version to rollback to')
  .option('-e, --environment <env>', 'Target environment', 'production')
  .option('--auto-approve', 'Skip confirmation prompts')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n⏪ Rolling back A2A deployment...\n'));
      
      const deployer = getDeployer('k8s'); // Default to k8s for rollback
      const result = await deployer.rollback(options);
      
      console.log(chalk.green('✅ Rollback completed successfully!'));
      console.log(chalk.cyan('Rollback details:'));
      console.log(JSON.stringify(result, null, 2));
      
    } catch (error) {
      console.error(chalk.red('\n❌ Rollback failed:'), error.message);
      process.exit(1);
    }
  });

// Scale command
program
  .command('scale')
  .description('Scale A2A agents')
  .option('-d, --deployment <name>', 'Deployment name')
  .option('-r, --replicas <count>', 'Number of replicas', '3')
  .option('-e, --environment <env>', 'Target environment', 'production')
  .action(async (options) => {
    try {
      console.log(chalk.blue(`\n📈 Scaling A2A deployment to ${options.replicas} replicas...\n`));
      
      const deployer = getDeployer('k8s');
      const result = await deployer.scale(options);
      
      console.log(chalk.green('✅ Scaling completed successfully!'));
      console.log(chalk.cyan('Scale result:'));
      console.log(JSON.stringify(result, null, 2));
      
    } catch (error) {
      console.error(chalk.red('\n❌ Scaling failed:'), error.message);
      process.exit(1);
    }
  });

// Logs command
program
  .command('logs')
  .description('Fetch deployment logs')
  .option('-d, --deployment <name>', 'Deployment name')
  .option('-e, --environment <env>', 'Environment', 'production')
  .option('-f, --follow', 'Follow log output')
  .option('-t, --tail <lines>', 'Number of lines to show', '100')
  .action(async (options) => {
    try {
      console.log(chalk.blue('\n📋 Fetching A2A deployment logs...\n'));
      
      const monitor = new DeploymentMonitor();
      await monitor.getLogs(options);
      
    } catch (error) {
      console.error(chalk.red('\n❌ Failed to fetch logs:'), error.message);
      process.exit(1);
    }
  });

// Status command
program
  .command('status')
  .description('Get deployment status')
  .option('-e, --environment <env>', 'Environment', 'production')
  .option('-o, --output <format>', 'Output format (json|table)', 'table')
  .action(async (options) => {
    try {
      const monitor = new DeploymentMonitor();
      const status = await monitor.getOverallStatus(options);
      
      if (options.output === 'json') {
        console.log(JSON.stringify(status, null, 2));
      } else {
        console.log(chalk.blue('\n📊 A2A Deployment Status\n'));
        
        status.deployments.forEach(deployment => {
          const statusColor = deployment.status === 'healthy' ? chalk.green : 
                             deployment.status === 'degraded' ? chalk.yellow : chalk.red;
          
          console.log(`${statusColor('●')} ${deployment.name} (${deployment.environment})`);
          console.log(`  Status: ${statusColor(deployment.status)}`);
          console.log(`  Replicas: ${deployment.replicas.ready}/${deployment.replicas.desired}`);
          console.log(`  Version: ${deployment.version}`);
          console.log(`  Uptime: ${deployment.uptime}`);
          console.log();
        });
      }
      
    } catch (error) {
      console.error(chalk.red('\n❌ Failed to get status:'), error.message);
      process.exit(1);
    }
  });

function getDeployer(platform) {
  switch (platform) {
    case 'k8s':
    case 'kubernetes':
      return new KubernetesDeployer();
    case 'aws':
    case 'azure':
    case 'gcp':
      return new CloudDeployer(platform);
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}