"""
A2A Agents Module - Organized Structure

Agent Organization:
- agent0_data_product/      - Data Product Registration Agent (Agent 0)
- agent1_standardization/   - Data Standardization Agent (Agent 1)
- agent2_ai_preparation/    - AI Preparation Agent (Agent 2)
- agent3_vector_processing/ - Vector Processing Agent (Agent 3)
- agent4_calc_validation/   - Calculation Validation Agent (Agent 4)
- agent_manager/           - Agent Manager (Orchestration)
- data_manager/            - Data Manager (Storage & CRUD)
- catalog_manager/         - Catalog Manager (ORD Registry)
- agent_builder/           - Agent Builder (Dynamic Agent Creation)

Each agent folder contains:
- active/  - Current SDK-based implementations
- legacy/  - Previous non-SDK implementations (deprecated)

All agents should use the SDK versions from the active/ folders.
"""

# Import active SDK agents for easy access
# from .agent0_data_product.active.data_product_agent_sdk import DataProductRegistrationAgentSDK
# from .agent1_standardization.active.data_standardization_agent_sdk import DataStandardizationAgentSDK
# from .agent2_ai_preparation.active.ai_preparation_agent_sdk import AIPreparationAgentSDK
# from .agent3_vector_processing.active.vector_processing_agent_sdk import VectorProcessingAgentSDK
from .agent4CalcValidation.active.calcValidationAgentSdk import CalcValidationAgentSDK
# from .agent5QaValidation.active.qaValidationAgentSdk import QAValidationAgentSDK  # Still has import issues
# from .agent_manager.active.agent_manager_agent import AgentManagerAgent
# from .data_manager.active.data_manager_agent_sdk import DataManagerAgentSDK
# from .catalog_manager.active.catalog_manager_agent_sdk import CatalogManagerAgentSDK
# from .agent_builder.active.agent_builder_agent_sdk import AgentBuilderAgentSDK

__all__ = [
    "DataProductRegistrationAgentSDK",
    "DataStandardizationAgentSDK",
    "AIPreparationAgentSDK",
    "VectorProcessingAgentSDK",
    "CalcValidationAgentSDK",
    "QAValidationAgentSDK",
    "AgentManagerAgent",
    "DataManagerAgentSDK",
    "CatalogManagerAgentSDK",
    "AgentBuilderAgentSDK"
]
