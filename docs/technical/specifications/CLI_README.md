# A2A ChatAgent CLI

A comprehensive command-line interface for testing and interacting with the A2A ChatAgent and the entire 16-agent network. This CLI provides powerful tools for validation, monitoring, and real-world testing of the production ChatAgent implementation.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements-cli.txt

# Make CLI executable (optional)
chmod +x cli.py
```

### Basic Usage

```bash
# Interactive mode (recommended for getting started)
python cli.py --interactive

# Send a single message
python cli.py --message "Analyze cryptocurrency market trends"

# Test all agents with broadcast
python cli.py --broadcast "Process this data from your specialty"

# Run comprehensive test suite
python cli.py --test-all

# Test agent connectivity only
python cli.py --connectivity
```

## 🎯 Features

### Core Functionality
- **Single Message Testing** - Send individual messages through the ChatAgent
- **Multi-Agent Broadcasting** - Test message routing to multiple agents simultaneously  
- **Conversation Flow Testing** - Multi-turn conversation validation
- **Agent Connectivity Monitoring** - Real-time agent availability checking
- **Performance Metrics** - Comprehensive monitoring and analytics
- **Interactive Mode** - Real-time chat interface with the agent network

### Advanced Features
- **Configuration Management** - Support for development, testing, and production environments
- **Real-time Progress Monitoring** - Beautiful terminal UI with progress bars and status updates
- **Comprehensive Reporting** - Detailed test results and performance analytics
- **Error Handling & Retry Logic** - Robust error handling with intelligent retry mechanisms
- **Service Discovery Integration** - Automatic agent endpoint discovery in different environments

## 📋 Command Reference

### Interactive Commands

When in interactive mode (`--interactive`), you can use these commands:

```
/help     - Show help information
/agents   - Test agent connectivity
/metrics  - Show performance metrics  
/quit     - Exit interactive mode

# Or simply type messages to chat with the agent network
```

### CLI Arguments

```bash
# Configuration
--config FILE           Use custom configuration file
--verbose, -v          Enable verbose output

# Testing Commands
--message "text"        Send single message to ChatAgent
--target-agent AGENT    Target specific agent (use with --message)
--broadcast "text"      Broadcast message to multiple agents
--conversation          Test multi-turn conversation flow
--connectivity          Test agent connectivity only
--test-all             Run comprehensive test suite
--metrics              Show current performance metrics

# Modes
--interactive          Enter interactive chat mode (default if no other command)
```

## ⚙️ Configuration

The CLI supports multiple configuration formats and environments:

### Configuration Files

Create configuration files in YAML or JSON format:

```yaml
# config.yaml
base_url: "http://localhost:8000"
environment: "development"
log_level: "INFO"

database:
  type: "sqlite"
  connection_string: "sqlite+aiosqlite:///cli_test.db"

auth:
  jwt_secret: "your-secret-key"
  enable_jwt: true
  enable_api_key: true

test_agents:
  - "data-processor"
  - "nlp-agent"
  - "crypto-trader"
```

### Environment-Specific Configs

Pre-built configurations are available in `config-examples/`:

```bash
# Development (local testing)
python cli.py --config config-examples/development.yaml --interactive

# Production (full feature testing)
python cli.py --config config-examples/production.yaml --test-all

# Testing (CI/CD optimized)
python cli.py --config config-examples/testing.yaml --connectivity
```

### Environment Variables

Key settings can be overridden via environment variables:

```bash
# Authentication
export A2A_JWT_SECRET="your-production-secret"
export A2A_CHATAGENT_URL="https://api.a2a.network"

# Database
export POSTGRES_HOST="db.a2a.network"
export POSTGRES_USER="a2a_user"
export POSTGRES_PASSWORD="secure_password"
export POSTGRES_DB="a2a_production"

# Redis
export REDIS_URL="redis://cache.a2a.network:6379/0"

# Agent Discovery
export DATA_PROCESSOR_URL="https://data-processor.a2a.network"
export NLP_AGENT_URL="https://nlp.a2a.network"
# ... other agent URLs
```

## 🧪 Testing Scenarios

### Development Testing

Test individual features during development:

```bash
# Test basic message routing
python cli.py --message "Hello, can you help me with data analysis?"

# Test specific agent
python cli.py --message "Analyze this crypto data" --target-agent crypto-trader

# Check which agents are online
python cli.py --connectivity
```

### Integration Testing

Validate multi-agent interactions:

```bash
# Test all agents respond correctly
python cli.py --broadcast "Perform your specialty analysis on this data"

# Test conversation continuity
python cli.py --conversation

# Full system validation
python cli.py --test-all
```

### Production Validation

Verify production deployment:

```bash
# Use production config
python cli.py --config config-examples/production.yaml --test-all

# Monitor performance
python cli.py --config config-examples/production.yaml --metrics

# Interactive production testing
python cli.py --config config-examples/production.yaml --interactive
```

## 📊 Test Results & Monitoring

### Test Output Examples

The CLI provides rich, colorful output with detailed information:

```
🧪 Running comprehensive ChatAgent test suite...

┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent                   ┃ Status  ┃ Endpoint                        ┃ Details                         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ data-processor          │ 🟢 ONLINE │ http://localhost:8001          │ 45ms                            │
│ nlp-agent              │ 🟢 ONLINE │ http://localhost:8002          │ 32ms                            │
│ crypto-trader          │ 🔴 OFFLINE│ http://localhost:8003          │ Connection refused              │
└─────────────────────────┴─────────┴─────────────────────────────────┴─────────────────────────────────┘

📊 Summary: 14/16 agents responded successfully (87.5%)
```

### Performance Metrics

Monitor ChatAgent performance in real-time:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                   ┃ Value                                                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Average Response Time    │ 0.45s                                                          │
│ Success Rate            │ 98.7%                                                          │
│ Active Connections      │ 12                                                             │
│ Cache Hit Rate          │ 89.2%                                                          │
│ Error Rate              │ 0.3%                                                           │
└──────────────────────────┴────────────────────────────────────────────────────────────────┘
```

## 🛠️ Development & Customization

### Adding New Test Scenarios

Extend the CLI with custom test scenarios:

```python
async def custom_test_scenario(self):
    """Add your custom test logic here"""
    result = await self.test_single_message("Your custom test message")
    # Add custom validation logic
    return result
```

### Custom Configuration Options

Add new configuration options in your config file:

```yaml
custom_settings:
  api_timeout: 60
  custom_agents: ["my-agent"]
  test_data_path: "/path/to/test/data"
```

### Integration with CI/CD

Use the CLI in your automation pipelines:

```bash
#!/bin/bash
# CI/CD test script

# Run connectivity tests
python cli.py --config config-examples/testing.yaml --connectivity

# Run comprehensive tests
python cli.py --config config-examples/testing.yaml --test-all

# Check exit code for pass/fail
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
```

## 🔧 Troubleshooting

### Common Issues

**Agent Connection Failures**
```bash
# Check agent endpoints
python cli.py --connectivity

# Test with specific config
python cli.py --config config-examples/development.yaml --connectivity
```

**Authentication Issues**
```bash
# Verify JWT secret is set
echo $A2A_JWT_SECRET

# Test with development config
python cli.py --config config-examples/development.yaml --message "test"
```

**Database Connection Issues**
```bash
# Check database configuration
python cli.py --config your-config.yaml --metrics

# Use in-memory database for testing
python cli.py --config config-examples/testing.yaml --test-all
```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python cli.py --verbose --test-all
```

### Log Files

Check application logs for detailed error information:
- Development: Console output with colors
- Production: Structured JSON logs
- Testing: Minimal warning-level logs

## 🎯 Use Cases

### Quality Assurance
- Validate ChatAgent functionality before deployment
- Test agent network connectivity and health
- Performance benchmarking and regression testing

### Development Support  
- Interactive testing during feature development
- Quick validation of changes
- Debugging agent communication issues

### Production Monitoring
- Health check validation
- Performance monitoring
- Integration testing in live environments

### Demonstration & Sales
- Live demonstration of ChatAgent capabilities
- Showcase multi-agent coordination
- Interactive demos for stakeholders

## 🤝 Contributing

To contribute improvements to the CLI:

1. Follow the existing code structure and patterns
2. Add tests for new features
3. Update this documentation
4. Ensure compatibility with all configuration environments

## 📝 License

This CLI tool is part of the A2A ChatAgent project and follows the same licensing terms.