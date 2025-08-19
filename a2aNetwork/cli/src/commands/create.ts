import { Command } from 'commander';
import chalk from 'chalk';
import fs from 'fs-extra';
import path from 'path';
import ora from 'ora';

export const createCommand = new Command('create')
  .description('Create new A2A components')
  .command('agent <name>')
  .option('-t, --type <type>', 'Agent type')
  .option('-c, --capabilities <items>', 'Comma-separated list of capabilities')
  .description('Create a new agent')
  .action(async (name: string, options: any) => {
    const spinner = ora('Creating agent...').start();
    
    try {
      const agentPath = path.join(process.cwd(), 'agents', name);
      await fs.ensureDir(agentPath);
      
      const capabilities = options.capabilities 
        ? options.capabilities.split(',').map((c: string) => c.trim())
        : [];
      
      const agentCode = `const { Agent } = require('@a2a/sdk');

const agent = new Agent({
  name: '${name}',
  type: '${options.type || 'custom'}',
  capabilities: ${JSON.stringify(capabilities, null, 2)}
});

// Add your services here
agent.addService('process', async (data) => {
  // Implementation
  return data;
});

// Start the agent
agent.start();

module.exports = agent;
`;

      await fs.writeFile(path.join(agentPath, 'index.js'), agentCode);
      
      // Create agent package.json
      const packageJson = {
        name: `@agents/${name}`,
        version: '0.1.0',
        main: 'index.js',
        scripts: {
          start: 'node index.js',
          dev: 'nodemon index.js'
        }
      };
      
      await fs.writeFile(
        path.join(agentPath, 'package.json'), 
        JSON.stringify(packageJson, null, 2)
      );
      
      spinner.succeed(`Agent '${name}' created successfully`);
      console.log(chalk.gray(`\nLocation: ${agentPath}`));
      console.log(chalk.cyan('\nNext steps:'));
      console.log(chalk.gray(`  cd agents/${name}`));
      console.log(chalk.gray('  npm install'));
      console.log(chalk.gray('  npm start\n'));
      
    } catch (error) {
      spinner.fail('Failed to create agent');
      console.error(error);
    }
  });

createCommand
  .command('workflow <name>')
  .description('Create a new workflow')
  .action(async (name: string) => {
    const spinner = ora('Creating workflow...').start();
    
    try {
      const workflowPath = path.join(process.cwd(), 'workflows');
      await fs.ensureDir(workflowPath);
      
      const workflow = `name: ${name}
description: ${name} workflow
version: 1.0.0

# Define workflow steps
steps:
  - id: step1
    name: First Step
    agent: processor
    service: process
    input:
      source: $input
    
  - id: step2
    name: Second Step
    agent: validator
    service: validate
    input:
      source: $steps.step1.output
    
  - id: step3
    name: Final Step
    agent: storage
    service: store
    input:
      source: $steps.step2.output

# Workflow configuration
config:
  timeout: 300000  # 5 minutes
  retry:
    attempts: 3
    delay: 1000
`;

      await fs.writeFile(path.join(workflowPath, `${name}.yaml`), workflow);
      
      spinner.succeed(`Workflow '${name}' created successfully`);
      console.log(chalk.gray(`\nLocation: workflows/${name}.yaml\n`));
      
    } catch (error) {
      spinner.fail('Failed to create workflow');
      console.error(error);
    }
  });

createCommand
  .command('service <name>')
  .description('Create a new service for current agent')
  .action(async (name: string) => {
    const spinner = ora('Creating service...').start();
    
    try {
      const servicesPath = path.join(process.cwd(), 'src', 'services');
      await fs.ensureDir(servicesPath);
      
      const serviceCode = `/**
 * ${name} Service
 */

class ${name.charAt(0).toUpperCase() + name.slice(1)}Service {
  constructor() {
    // Initialize service
  }

  async process(data) {
    // Service implementation
    return data;
  }

  async validate(data) {
    // Validation logic
    return true;
  }
}

module.exports = new ${name.charAt(0).toUpperCase() + name.slice(1)}Service();
`;

      await fs.writeFile(path.join(servicesPath, `${name}.js`), serviceCode);
      
      spinner.succeed(`Service '${name}' created successfully`);
      console.log(chalk.gray(`\nLocation: src/services/${name}.js\n`));
      
    } catch (error) {
      spinner.fail('Failed to create service');
      console.error(error);
    }
  });