# A2A Network Python SDK

A Python SDK for interacting with the A2A (Agent-to-Agent) Network, providing tools for agent communication, blockchain integration, and network operations.

## Installation

```bash
pip install a2a-network
```

## Quick Start

```python
from a2a import A2AClient

# Initialize the client
client = A2AClient(
    rpc_url="http://localhost:8545",
    private_key="your-private-key"
)

# Register an agent
agent = client.register_agent(
    name="MyAgent",
    capabilities=["compute", "storage"]
)

# Send a message
response = client.send_message(
    to_agent="AgentB",
    content="Hello from Python SDK"
)
```

## Features

- Agent registration and management
- Secure message passing between agents
- Blockchain integration for trust and verification
- Service discovery and capability matching
- Performance monitoring and debugging
- Web3 integration with Ethereum networks

## Documentation

For detailed documentation, visit [A2A Network Docs](https://docs.a2anetwork.io)

## License

MIT License