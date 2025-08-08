# A2A Agent Organization Guide

## Overview
All A2A agents have been reorganized into a standardized folder structure with SDK versions as the primary implementation.

## Folder Structure
```
agents/
├── agent0_data_product/
│   ├── active/           # Current SDK implementation
│   │   ├── data_product_agent_sdk.py
│   │   └── agent0_router.py
│   └── legacy/           # Deprecated versions
│       ├── data_product_agent.py
│       └── enhanced_data_product_agent.py
├── agent1_standardization/
│   ├── active/
│   │   ├── data_standardization_agent_sdk.py
│   │   ├── data_standardization_agent_config.yaml
│   │   └── agent1_router.py
│   └── legacy/
│       ├── data_standardization_agent.py
│       ├── data_standardization_agent_v2.py
│       └── enhanced_standardization_agent.py
├── agent2_ai_preparation/
│   ├── active/
│   │   ├── ai_preparation_agent_sdk.py
│   │   └── agent2_router.py
│   └── legacy/
│       └── ai_preparation_agent.py
├── agent3_vector_processing/
│   ├── active/
│   │   ├── vector_processing_agent_sdk.py
│   │   └── agent3_router.py
│   └── legacy/
│       └── vector_processing_agent.py
├── agent_manager/
│   └── active/
│       ├── agent_manager_agent.py
│       └── agent_manager_router.py
├── data_manager/
│   ├── active/
│   │   └── data_manager_agent_sdk.py
│   └── legacy/
│       └── data_manager_agent.py
├── catalog_manager/
│   ├── active/
│   │   ├── catalog_manager_agent_sdk.py
│   │   └── catalog_manager_router.py
│   └── legacy/
│       └── catalog_manager_agent.py
└── agent_builder/
    └── active/
        └── agent_builder_agent_sdk.py
```

## Agent Ports & URLs
- **Agent 0** (Data Product): http://localhost:8001
- **Agent 1** (Standardization): http://localhost:8002
- **Agent 2** (AI Preparation): http://localhost:8003
- **Agent 3** (Vector Processing): http://localhost:8004
- **Data Manager**: http://localhost:8005
- **Catalog Manager**: http://localhost:8006
- **Agent Manager**: http://localhost:8007
- **Agent Builder**: http://localhost:8008

## SDK Launch Files
All agents now have SDK launch files:
- `launch_agent0_sdk.py`
- `launch_agent1_sdk.py`
- `launch_agent2_sdk.py` (NEW)
- `launch_agent3_sdk.py` (NEW)
- `launch_data_manager_sdk.py` (NEW)
- `launch_catalog_manager_sdk.py` (NEW)

## Import Changes
In your code, import agents from their active folders:
```python
# Old way (deprecated)
from app.a2a.agents.data_product_agent import DataProductRegistrationAgent

# New way (SDK version)
from app.a2a.agents.agent0_data_product.active.data_product_agent_sdk import DataProductRegistrationAgentSDK

# Or use the convenience imports from __init__.py
from app.a2a.agents import DataProductRegistrationAgentSDK
```

## Migration Status
✅ **Completed**:
- All agents organized into subfolders
- SDK versions in active/ folders
- Legacy versions in legacy/ folders
- SDK launch files created for all agents
- main.py updated to use new structure
- Router imports updated to SDK versions

⚠️ **Important Notes**:
1. Always use SDK versions from active/ folders
2. Legacy versions are deprecated and will be removed
3. Agent Manager doesn't have SDK version yet (it's already well-structured)
4. All agents follow A2A v0.2.9 protocol compliance

## Running Agents
To run any agent with SDK version:
```bash
# Individual agents
python launch_agent0_sdk.py
python launch_agent1_sdk.py
python launch_agent2_sdk.py
python launch_agent3_sdk.py
python launch_data_manager_sdk.py
python launch_catalog_manager_sdk.py

# Or run all together
python main.py  # Runs the main app with all agent routers
```