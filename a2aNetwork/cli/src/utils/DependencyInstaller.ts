import { spawn } from 'child_process';
import chalk from 'chalk';
import ora from 'ora';
import which from 'which';

export class DependencyInstaller {
  private projectPath: string;

  constructor(projectPath: string) {
    this.projectPath = projectPath;
  }

  async install(options: any): Promise<void> {
    const spinner = ora('Installing dependencies...').start();

    try {
      // Detect package manager
      const packageManager = await this.detectPackageManager();
      
      // Install Node.js dependencies
      await this.installNodeDependencies(packageManager);
      
      // Install Python dependencies if needed
      if (options.useBlockchain || options.projectType === 'full-stack') {
        await this.installPythonDependencies();
      }
      
      // Install blockchain tools if needed
      if (options.useBlockchain) {
        await this.installBlockchainTools();
      }

      spinner.succeed('Dependencies installed successfully');
    } catch (error) {
      spinner.fail('Failed to install dependencies');
      throw error;
    }
  }

  private async detectPackageManager(): Promise<string> {
    const managers = ['pnpm', 'yarn', 'npm'];
    
    for (const manager of managers) {
      try {
        await which(manager);
        return manager;
      } catch (e) {
        continue;
      }
    }
    
    return 'npm'; // Default fallback
  }

  private installNodeDependencies(packageManager: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const args = packageManager === 'npm' ? ['install'] : ['install'];
      
      const child = spawn(packageManager, args, {
        cwd: this.projectPath,
        stdio: 'pipe'
      });

      child.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`${packageManager} install failed with code ${code}`));
        }
      });

      child.on('error', reject);
    });
  }

  private async installPythonDependencies(): Promise<void> {
    // Check if Python is available
    try {
      await which('python3');
    } catch (e) {
      console.log(chalk.yellow('\n⚠️  Python not found. Skipping Python dependencies.'));
      return;
    }

    // Create requirements.txt
    const requirements = `# A2A Python Dependencies
web3==6.11.0
eth-account==0.9.0
requests==2.31.0
python-dotenv==1.0.0
`;

    const fs = await import('fs-extra');
    await fs.writeFile(`${this.projectPath}/requirements.txt`, requirements);

    // Install with pip
    return new Promise((resolve, reject) => {
      const child = spawn('pip3', ['install', '-r', 'requirements.txt'], {
        cwd: this.projectPath,
        stdio: 'pipe'
      });

      child.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          // Non-fatal: Python deps are optional
          console.log(chalk.yellow('⚠️  Python dependencies installation failed (optional)'));
          resolve();
        }
      });

      child.on('error', () => {
        console.log(chalk.yellow('⚠️  Python dependencies installation failed (optional)'));
        resolve();
      });
    });
  }

  private async installBlockchainTools(): Promise<void> {
    console.log(chalk.cyan('\n📦 Setting up blockchain tools...'));
    
    // Check if foundry is installed
    try {
      await which('forge');
      console.log(chalk.green('✓ Foundry already installed'));
    } catch (e) {
      console.log(chalk.yellow('\nFoundry not found. To install:'));
      console.log(chalk.gray('curl -L https://foundry.paradigm.xyz | bash'));
      console.log(chalk.gray('foundryup\n'));
    }
  }
}