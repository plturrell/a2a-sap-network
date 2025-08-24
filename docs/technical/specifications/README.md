# A2A Technical Specifications

This directory contains all technical specifications for the A2A platform, organized with standardized naming conventions for easy navigation.

## Agent Specifications

### Core Processing Agents (00-05)
- `agent00_data_product_spec.md` - Agent 0: Data Product Agent specification
- `agent01_data_standardization_spec.md` - Agent 1: Data Standardization Agent specification
- `agent01_bpmn_workflow_spec.md` - Agent 1: BPMN workflow definition
- `agent02_ai_preparation_spec.md` - Agent 2: AI Preparation Agent specification
- `agent03_vector_processing_spec.md` - Agent 3: Vector Processing Agent specification
- `agent03_vector_processing_code_spec.md` - Agent 3: Code implementation specification
- `agent04_calc_validation_spec.md` - Agent 4: Calculation Validation Agent specification
- `agent05_qa_validation_spec.md` - Agent 5: QA Validation Agent specification

## Service Specifications

### Core Services
- `agent_api_specifications.md` - General agent API specifications
- `orchestrator_service_spec.md` - A2A Orchestrator service specification
- `registry_service_spec.md` - A2A Registry service specification
- `blockchain_contracts_spec.md` - Smart contracts specification
- `ord_registry_v2_spec.md` - ORD Registry v2 specification

## Naming Convention

All specification files follow this standardized naming pattern:

- **Agent Specs**: `agent{NN}_{name}_spec.md` (e.g., `agent00_data_product_spec.md`)
- **Service Specs**: `{service_name}_spec.md` (e.g., `orchestrator_service_spec.md`)
- **Workflow Specs**: `agent{NN}_{workflow_type}_spec.md` (e.g., `agent01_bpmn_workflow_spec.md`)

## Usage

These specifications define:

1. **Agent Capabilities** - What each agent can do
2. **API Interfaces** - How agents communicate
3. **Data Formats** - Expected input/output formats
4. **Workflow Definitions** - BPMN and process flows
5. **Integration Points** - How components connect

## Related Documentation

- **Architecture**: `/docs/technical/architecture/` - System architecture documentation
- **Implementation**: `/docs/consolidated-reports/` - Implementation reports and analysis
- **User Guides**: `/docs/` - User and admin documentation
