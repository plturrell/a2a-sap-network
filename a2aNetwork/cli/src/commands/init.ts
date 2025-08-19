import { Command } from 'commander';
import inquirer from 'inquirer';
import chalk from 'chalk';
import ora from 'ora';
import fs from 'fs-extra';
import path from 'path';
import validateNpmPackageName from 'validate-npm-package-name';
import { ConfigManager } from '../config/ConfigManager';
import { ProjectScaffolder } from '../utils/ProjectScaffolder';
import { DependencyInstaller } from '../utils/DependencyInstaller';

interface InitAnswers {
  projectName: string;
  projectType: 'agent' | 'workflow' | 'full-stack';
  agentType?: string;
  useBlockchain: boolean;
  useSAP: boolean;
  database: 'sqlite' | 'hana' | 'none';
  installDependencies: boolean;
}

export const initCommand = new Command('init')
  .description('Initialize a new A2A project')
  .argument('[project-name]', 'Name of the project')
  .option('-t, --template <type>', 'Project template (agent|workflow|full-stack)')
  .option('--no-install', 'Skip dependency installation')
  .option('--use-defaults', 'Use default configuration values')
  .action(async (projectName: string | undefined, options: any) => {
    console.log(chalk.blue('\n🚀 Welcome to A2A Framework!\n'));

    try {
      // Get project configuration through prompts
      const answers = await getProjectConfig(projectName, options);
      
      // Create project directory
      const projectPath = path.join(process.cwd(), answers.projectName);
      const spinner = ora('Creating project structure...').start();

      // Check if directory exists
      if (await fs.pathExists(projectPath)) {
        spinner.fail(chalk.red(`Directory ${answers.projectName} already exists!`));
        process.exit(1);
      }

      // Create project structure
      await fs.ensureDir(projectPath);
      spinner.succeed('Project directory created');

      // Scaffold project based on type
      const scaffolder = new ProjectScaffolder(projectPath, answers);
      await scaffolder.scaffold();

      // Generate smart configuration
      const configManager = new ConfigManager(projectPath);
      await configManager.generateConfig(answers);

      // Install dependencies if requested
      if (answers.installDependencies) {
        const installer = new DependencyInstaller(projectPath);
        await installer.install(answers);
      }

      // Success message
      console.log(chalk.green('\n✨ Project created successfully!\n'));
      console.log(chalk.cyan('Next steps:'));
      console.log(chalk.gray(`  cd ${answers.projectName}`));
      
      if (!answers.installDependencies) {
        console.log(chalk.gray('  npm install'));
      }
      
      console.log(chalk.gray('  a2a dev start\n'));
      console.log(chalk.yellow('Happy coding! 🎉\n'));

    } catch (error) {
      console.error(chalk.red('\n❌ Error creating project:'), error);
      process.exit(1);
    }
  });

async function getProjectConfig(projectName: string | undefined, options: any): Promise<InitAnswers> {
  const questions: any[] = [];

  // Project name
  if (!projectName) {
    questions.push({
      type: 'input',
      name: 'projectName',
      message: 'Project name:',
      default: 'my-a2a-project',
      validate: (name: string) => {
        const validation = validateNpmPackageName(name);
        if (validation.validForNewPackages) {
          return true;
        }
        return 'Invalid project name';
      }
    });
  }

  // Project type
  if (!options.template) {
    questions.push({
      type: 'list',
      name: 'projectType',
      message: 'What type of project would you like to create?',
      choices: [
        { name: '🤖 Single Agent', value: 'agent' },
        { name: '🔄 Workflow (Multi-Agent)', value: 'workflow' },
        { name: '🏗️  Full Stack Application', value: 'full-stack' }
      ],
      default: 'agent'
    });
  }

  // Quick mode with defaults
  if (options.useDefaults) {
    const defaults: InitAnswers = {
      projectName: projectName || 'my-a2a-project',
      projectType: options.template || 'agent',
      agentType: 'data-processor',
      useBlockchain: false,
      useSAP: false,
      database: 'sqlite',
      installDependencies: options.install !== false
    };
    return defaults;
  }

  // Initial answers
  let answers = await inquirer.prompt(questions);
  
  if (projectName) {
    answers.projectName = projectName;
  }
  if (options.template) {
    answers.projectType = options.template;
  }

  // Agent-specific questions
  if (answers.projectType === 'agent') {
    const agentQuestions = await inquirer.prompt([
      {
        type: 'list',
        name: 'agentType',
        message: 'What type of agent?',
        choices: [
          { name: '📊 Data Processing Agent', value: 'data-processor' },
          { name: '🧠 AI/ML Agent', value: 'ai-ml' },
          { name: '💾 Storage Agent', value: 'storage' },
          { name: '🔄 Orchestration Agent', value: 'orchestrator' },
          { name: '📈 Analytics Agent', value: 'analytics' },
          { name: '🛠️  Custom Agent', value: 'custom' }
        ]
      }
    ]);
    answers = { ...answers, ...agentQuestions };
  }

  // Advanced configuration
  const advancedQuestions = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'useBlockchain',
      message: 'Enable blockchain features?',
      default: false,
      when: () => !options.useDefaults
    },
    {
      type: 'confirm',
      name: 'useSAP',
      message: 'Include SAP integration?',
      default: false,
      when: () => !options.useDefaults
    },
    {
      type: 'list',
      name: 'database',
      message: 'Choose a database:',
      choices: [
        { name: 'SQLite (Development)', value: 'sqlite' },
        { name: 'SAP HANA Cloud', value: 'hana' },
        { name: 'No database', value: 'none' }
      ],
      default: 'sqlite',
      when: () => !options.useDefaults
    },
    {
      type: 'confirm',
      name: 'installDependencies',
      message: 'Install dependencies now?',
      default: true,
      when: () => options.install !== false
    }
  ]);

  return { ...answers, ...advancedQuestions };
}