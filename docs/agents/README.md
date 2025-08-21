# A2A Network Agents Documentation

## Overview
The A2A Network consists of 16 specialized agents that work together to process, validate, and manage data products through a sophisticated pipeline. Each agent has specific responsibilities and capabilities within the ecosystem.

## The 16 A2A Agents

### Core Processing Pipeline (Agents 0-5)
1. **[Agent 0 - Data Product Agent](agent0DataProduct.md)** (Port 8000)
   - Entry point for data products into the A2A Network
   - Creates and manages data products with Dublin Core metadata

2. **[Agent 1 - Data Standardization Agent](agent1Standardization.md)** (Port 8001)
   - Standardizes data formats and validates schemas
   - Transforms data into L4 hierarchical structure

3. **[Agent 2 - AI Preparation Agent](agent2AiPreparation.md)** (Port 8002)
   - Prepares data for AI/ML processing
   - Performs feature engineering and preprocessing

4. **[Agent 3 - Vector Processing Agent](agent3VectorProcessing.md)** (Port 8003)
   - Generates and processes vector embeddings
   - Enables semantic analysis and similarity search

5. **[Agent 4 - Calculation Validation Agent](agent4CalcValidation.md)** (Port 8004)
   - Validates calculations and numerical computations
   - Performs statistical analysis and accuracy checking

6. **[Agent 5 - QA Validation Agent](agent5QaValidation.md)** (Port 8005)
   - Performs quality assurance and validation checks
   - Ensures compliance and data integrity

### Management and Control Agents (Agents 6-8)
7. **[Agent 6 - Quality Control Manager](agent6QualityControl.md)** (Port 8006)
   - Manages quality control and routing decisions
   - Central quality gatekeeper for the network

8. **[Agent 7 - Agent Manager](agent7AgentManager.md)** (Port 8007)
   - Central agent that manages other agents in the network
   - Handles agent lifecycle and coordination

9. **[Agent 8 - Data Manager](agent8DataManager.md)** (Port 8008)
   - Centralized data storage and retrieval
   - Manages data persistence and caching

### Specialized Processing Agents (Agents 9-11)
10. **[Agent 9 - Reasoning Agent](agent9Reasoning.md)** (Port 8009)
    - Advanced reasoning and decision-making
    - Logical inference and problem solving

11. **[Agent 10 - Calculation Agent](agent10Calculation.md)** (Port 8010)
    - Performs complex calculations and mathematical operations
    - Self-healing calculation capabilities

12. **[Agent 11 - SQL Agent](agent11SQL.md)** (Port 8011)
    - Handles SQL operations and database interactions
    - Natural language to SQL translation

### Infrastructure Support Agents (Agents 12-15)
13. **[Agent 12 - Catalog Manager](agent12CatalogManager.md)** (Port 8012)
    - Manages service catalogs and resource discovery
    - ORD registry integration

14. **[Agent 13 - Agent Builder](agent13AgentBuilder.md)** (Port 8013)
    - Creates and deploys new agents dynamically
    - Template-based agent generation

15. **[Agent 14 - Embedding Fine-Tuner](agent14EmbeddingFineTuner.md)** (Port 8014)
    - Fine-tunes and optimizes embedding models
    - Works with Agents 2 and 3 for optimization

16. **[Agent 15 - Orchestrator Agent](agent15Orchestrator.md)** (Port 8015)
    - Orchestrates complex workflows across multiple agents
    - Pipeline management and coordination

## Data Flow Through the Network

```
Raw Data → Agent 0 (Data Product) → Agent 1 (Standardization) → Agent 2 (AI Prep)
                                                                           ↓
Agent 6 (QC Manager) ← Agent 5 (QA Valid) ← Agent 4 (Calc Valid) ← Agent 3 (Vector)
        ↓
   Approved Data → Publishing/Storage via Agent 8 (Data Manager)
```

## Agent Communication
- All agents communicate using the A2A Protocol v0.2.9
- Trust relationships are managed through blockchain smart contracts
- Each agent has a unique blockchain address for identity verification
- Agents register their capabilities with the Agent Manager (Agent 7)

## Common Agent Features
- **Blockchain Registration**: Each agent is registered on-chain with a unique address
- **Capability Advertisement**: Agents advertise their capabilities for discovery
- **Health Monitoring**: All agents report health status to Agent Manager
- **Async Operations**: Support for asynchronous processing
- **Error Handling**: Standardized error codes and recovery mechanisms

## Getting Started
1. Review individual agent documentation for specific capabilities
2. Understand the data flow through the pipeline
3. Configure agents according to your use case
4. Monitor agent health through Agent Manager
5. Use the Orchestrator Agent for complex workflows

## Configuration
Each agent can be configured through:
- Environment variables
- Configuration files (YAML/JSON)
- Runtime parameters
- Blockchain-based configuration updates

## Monitoring and Operations
- Agent Manager (Agent 7) provides centralized monitoring
- Each agent exposes health endpoints
- Metrics are collected for performance tracking
- Logs are aggregated for troubleshooting

## Security
- All inter-agent communication is authenticated
- Trust relationships are verified through blockchain
- Data encryption in transit and at rest
- Role-based access control for agent operations