import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { defaultTemplates } from '../config/templates';

export class ProjectScaffolder {
  private projectPath: string;
  private options: any;

  constructor(projectPath: string, options: any) {
    this.projectPath = projectPath;
    this.options = options;
  }

  async scaffold(): Promise<void> {
    const spinner = ora('Creating project structure...').start();

    try {
      // Create base directories
      await this.createBaseStructure();

      // Get template based on project type
      const templateKey = this.options.agentType || this.options.projectType;
      const template = defaultTemplates[templateKey] || defaultTemplates.agent;

      // Create template files
      if (template.files) {
        for (const [filePath, content] of Object.entries(template.files)) {
          const fullPath = path.join(this.projectPath, filePath);
          await fs.ensureDir(path.dirname(fullPath));
          await fs.writeFile(fullPath, content);
        }
      }

      // Create package.json
      await this.createPackageJson(template);

      // Create README
      await this.createReadme();

      // Create .gitignore
      await this.createGitignore();

      // Create test structure
      await this.createTestStructure();

      spinner.succeed('Project structure created');
    } catch (error) {
      spinner.fail('Failed to create project structure');
      throw error;
    }
  }

  private async createBaseStructure(): Promise<void> {
    const dirs = [
      'src',
      'src/services',
      'src/utils',
      'src/middleware',
      'config',
      'test',
      'test/unit',
      'test/integration',
      'docs'
    ];

    // Add specific directories based on project type
    if (this.options.projectType === 'workflow') {
      dirs.push('agents', 'workflows', 'orchestrator');
    }

    if (this.options.projectType === 'full-stack') {
      dirs.push('frontend', 'frontend/src', 'backend', 'backend/src', 'contracts', 'shared');
    }

    if (this.options.useBlockchain) {
      dirs.push('contracts', 'scripts', 'test/contracts');
    }

    for (const dir of dirs) {
      await fs.ensureDir(path.join(this.projectPath, dir));
    }
  }

  private async createPackageJson(template: any): Promise<void> {
    const packageJson: any = {
      name: this.options.projectName,
      version: '0.1.0',
      description: `A2A ${this.options.projectType} project`,
      main: 'src/index.js',
      scripts: {
        start: 'node src/index.js',
        dev: 'nodemon src/index.js',
        test: 'jest',
        'test:watch': 'jest --watch',
        'test:coverage': 'jest --coverage',
        lint: 'eslint src --ext .js,.jsx',
        'lint:fix': 'eslint src --ext .js,.jsx --fix'
      },
      keywords: ['a2a', 'agent', 'blockchain', 'microservices'],
      author: '',
      license: 'MIT',
      dependencies: {} as any,
      devDependencies: {} as any
    };

    // Add dependencies from template
    if (template.dependencies) {
      // Convert arrays to object with latest versions
      for (const dep of template.dependencies.required) {
        packageJson.dependencies[dep] = 'latest';
      }
      
      for (const dep of template.dependencies.dev) {
        packageJson.devDependencies[dep] = 'latest';
      }
    }

    // Add blockchain-specific scripts
    if (this.options.useBlockchain) {
      packageJson.scripts['compile'] = 'hardhat compile';
      packageJson.scripts['deploy'] = 'hardhat run scripts/deploy.js';
      packageJson.scripts['test:contracts'] = 'hardhat test';
      packageJson.devDependencies['hardhat'] = 'latest';
      packageJson.devDependencies['@nomiclabs/hardhat-waffle'] = 'latest';
    }

    // Add workflow-specific scripts
    if (this.options.projectType === 'workflow') {
      packageJson.scripts['start:all'] = 'concurrently "npm run start:processor" "npm run start:validator" "npm run start:orchestrator"';
      packageJson.scripts['start:processor'] = 'node agents/processor/index.js';
      packageJson.scripts['start:validator'] = 'node agents/validator/index.js';
      packageJson.scripts['start:orchestrator'] = 'node orchestrator/index.js';
    }

    // Add full-stack specific scripts
    if (this.options.projectType === 'full-stack') {
      packageJson.scripts['start:frontend'] = 'cd frontend && npm start';
      packageJson.scripts['start:backend'] = 'cd backend && npm start';
      packageJson.scripts['build'] = 'cd frontend && npm run build';
      packageJson.scripts['start:all'] = 'concurrently "npm run start:backend" "npm run start:frontend"';
    }

    const packagePath = path.join(this.projectPath, 'package.json');
    await fs.writeFile(packagePath, JSON.stringify(packageJson, null, 2));
  }

  private async createReadme(): Promise<void> {
    const readme = `# ${this.options.projectName}

${this.options.projectType === 'agent' ? 'A2A Agent' : this.options.projectType === 'workflow' ? 'A2A Multi-Agent Workflow' : 'A2A Full Stack Application'}

## Overview

This project was generated with the A2A CLI and includes:
${this.options.useBlockchain ? '- ✅ Blockchain integration' : ''}
${this.options.useSAP ? '- ✅ SAP integration' : ''}
${this.options.database !== 'none' ? `- ✅ ${this.options.database} database` : ''}

## Quick Start

\`\`\`bash
# Install dependencies
npm install

# Start in development mode
npm run dev

# Run tests
npm test
\`\`\`

## Project Structure

\`\`\`
${this.options.projectName}/
├── src/                # Source code
│   ├── index.js       # Main entry point
│   ├── services/      # Service implementations
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── test/              # Test files
├── .env              # Environment variables
└── a2a.config.js     # A2A configuration
\`\`\`

## Available Scripts

- \`npm start\` - Start the application
- \`npm run dev\` - Start in development mode with auto-reload
- \`npm test\` - Run tests
- \`npm run lint\` - Run linter
${this.options.useBlockchain ? '- `npm run deploy` - Deploy smart contracts' : ''}

## Configuration

Configuration is managed through:
1. Environment variables (\`.env\` file)
2. A2A configuration (\`a2a.config.js\`)

## Development

### Adding a New Service

\`\`\`javascript
agent.addService('myService', async (data) => {
  // Your service logic here
  return processedData;
});
\`\`\`

### Testing

Write tests in the \`test/\` directory:

\`\`\`javascript
describe('MyService', () => {
  it('should process data correctly', async () => {
    const result = await myService.process(testData);
    expect(result).toBeDefined();
  });
});
\`\`\`

## Deployment

1. Update production environment variables
2. Build the project (if needed)
3. Deploy using your preferred method

## Documentation

For more information, visit the [A2A Documentation](https://docs.a2a.network)

## License

MIT
`;

    const readmePath = path.join(this.projectPath, 'README.md');
    await fs.writeFile(readmePath, readme);
  }

  private async createGitignore(): Promise<void> {
    const gitignore = `# Dependencies
node_modules/
package-lock.json
yarn.lock

# Environment
.env
.env.*
!.env.example

# Logs
logs/
*.log
npm-debug.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Testing
coverage/
.nyc_output/

# Build
dist/
build/
lib/
*.tsbuildinfo

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Blockchain
cache/
artifacts/
typechain/
deployments/

# SAP
.cds_gen.log
gen/
mta_archives/

# Temporary files
tmp/
temp/
`;

    const gitignorePath = path.join(this.projectPath, '.gitignore');
    await fs.writeFile(gitignorePath, gitignore);
  }

  private async createTestStructure(): Promise<void> {
    // Create basic test file
    const testContent = `const { Agent } = require('@a2a/sdk');

describe('${this.options.projectName}', () => {
  let agent;

  beforeAll(() => {
    agent = new Agent({
      name: 'test-agent',
      type: '${this.options.agentType || 'custom'}'
    });
  });

  afterAll(async () => {
    await agent.stop();
  });

  describe('Agent initialization', () => {
    it('should create agent instance', () => {
      expect(agent).toBeDefined();
      expect(agent.name).toBe('test-agent');
    });

    it('should have required capabilities', () => {
      const capabilities = agent.getCapabilities();
      expect(Array.isArray(capabilities)).toBe(true);
    });
  });

  describe('Service functionality', () => {
    it('should respond to service calls', async () => {
      const result = await agent.call('hello', { test: true });
      expect(result).toBeDefined();
    });
  });
});
`;

    const testPath = path.join(this.projectPath, 'test', 'index.test.js');
    await fs.writeFile(testPath, testContent);

    // Create jest config
    const jestConfig = {
      testEnvironment: 'node',
      coverageDirectory: 'coverage',
      collectCoverageFrom: [
        'src/**/*.js',
        '!src/**/*.test.js'
      ],
      testMatch: [
        '**/test/**/*.test.js'
      ],
      verbose: true
    };

    const jestConfigPath = path.join(this.projectPath, 'jest.config.json');
    await fs.writeFile(jestConfigPath, JSON.stringify(jestConfig, null, 2));
  }
}