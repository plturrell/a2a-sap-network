# A2A CLI Tool

A command-line interface for the A2A (Agent-to-Agent) Framework that simplifies agent development and deployment.

## Features

- 🚀 **Quick Project Initialization** - Create new A2A projects in seconds
- ⚙️ **Smart Configuration** - Intelligent defaults and environment-specific configs
- 🛠️ **Developer Tools** - Built-in utilities for development and debugging
- 📦 **Project Templates** - Pre-built templates for common agent types
- 🔧 **Health Checks** - Diagnostic tools to identify and fix issues

## Installation

```bash
# Install globally
npm install -g @a2a/cli

# Or use npx for one-time usage
npx @a2a/cli init my-agent
```

## Quick Start

```bash
# Create a new agent project
a2a init my-data-processor

# Navigate to project
cd my-data-processor

# Start development environment
a2a dev start

# Run health check
a2a doctor
```

## Commands

### `a2a init [project-name]`

Initialize a new A2A project with interactive setup.

**Options:**
- `-t, --template <type>` - Project template (agent|workflow|full-stack)
- `--no-install` - Skip dependency installation
- `--use-defaults` - Use default configuration values

**Examples:**
```bash
a2a init my-agent                    # Interactive setup
a2a init my-agent -t agent           # Use agent template
a2a init my-agent --use-defaults     # Quick setup with defaults
```

### `a2a create`

Create new components within an existing project.

**Subcommands:**
- `agent <name>` - Create a new agent
- `workflow <name>` - Create a new workflow
- `service <name>` - Create a new service

**Examples:**
```bash
a2a create agent processor --type data-processor
a2a create workflow etl-pipeline
a2a create service validator
```

### `a2a dev`

Development environment management.

**Subcommands:**
- `start` - Start development environment
- `fund` - Fund development accounts with test tokens
- `mock create <agent>` - Create mock services

**Examples:**
```bash
a2a dev start --all              # Start all services
a2a dev start --blockchain       # Start with blockchain
a2a dev fund --account 0x123     # Fund specific account
```

### `a2a config`

Configuration management utilities.

**Subcommands:**
- `set <key> <value>` - Set configuration value
- `get [key]` - Get configuration value(s)
- `wizard` - Interactive configuration wizard
- `validate` - Validate current configuration

**Examples:**
```bash
a2a config set AGENT_NAME my-agent
a2a config get                    # Show all config
a2a config wizard                 # Interactive setup
a2a config validate               # Check configuration
```

### `a2a doctor`

Health check and diagnostic tool.

```bash
a2a doctor
```

Checks:
- ✅ Node.js version compatibility
- ✅ Dependencies installation
- ✅ Environment configuration
- ✅ Service availability
- ✅ Network connectivity

## Project Templates

### Agent Templates

- **data-processor** - Data cleaning and transformation
- **ai-ml** - Machine learning inference
- **storage** - Data storage and retrieval
- **orchestrator** - Workflow coordination
- **analytics** - Data analysis and reporting

### Workflow Templates

- **workflow** - Multi-agent workflow system
- **full-stack** - Complete application with UI

## Configuration

The CLI generates smart defaults for different environments:

### Development (.env.development)
```env
NODE_ENV=development
AGENT_NAME=my-agent
A2A_REGISTRY_URL=http://localhost:3000
DATABASE_TYPE=sqlite
BLOCKCHAIN_ENABLED=false
```

### Production (.env.production)
```env
NODE_ENV=production
AGENT_NAME=${AGENT_NAME}
A2A_REGISTRY_URL=${A2A_REGISTRY_URL}
DATABASE_URL=${DATABASE_URL}
BLOCKCHAIN_ENABLED=${BLOCKCHAIN_ENABLED}
```

## Smart Defaults

The CLI provides intelligent defaults based on your project type:

- **Blockchain**: Automatically configures local test network
- **Database**: Sets up SQLite for development, HANA for production
- **SAP Integration**: Configures CAP and XSUAA for enterprise
- **Security**: Generates secure secrets for development

## Development Workflow

1. **Initialize Project**
   ```bash
   a2a init my-agent --template data-processor
   cd my-agent
   ```

2. **Configure Environment**
   ```bash
   a2a config wizard
   ```

3. **Start Development**
   ```bash
   a2a dev start --all
   ```

4. **Create Components**
   ```bash
   a2a create service cleaner
   a2a create workflow data-pipeline
   ```

5. **Health Check**
   ```bash
   a2a doctor
   ```

6. **Deploy**
   ```bash
   npm run build
   npm run deploy
   ```

## Troubleshooting

### Common Issues

**"Command not found: a2a"**
```bash
npm install -g @a2a/cli
```

**"Dependencies not installed"**
```bash
a2a doctor  # Diagnose issues
npm install
```

**"Blockchain connection failed"**
```bash
a2a dev start --blockchain
a2a dev fund  # Fund test accounts
```

**"Configuration invalid"**
```bash
a2a config validate
a2a config wizard  # Reconfigure
```

## Contributing

1. Clone the repository
2. Install dependencies: `npm install`
3. Build: `npm run build`
4. Test: `npm test`
5. Link for local testing: `npm link`

## License

MIT

---

**Happy coding with A2A! 🎉**