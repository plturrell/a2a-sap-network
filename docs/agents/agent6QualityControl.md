# Agent 6: Quality Control Manager Agent

## Overview
The Quality Control Manager Agent (Agent 6) serves as the central quality gatekeeper in the A2A Network. It assesses quality metrics from all validation agents, makes routing decisions, and determines whether data products meet the required standards for publication or need rework.

## Purpose
- Centralize quality control decisions across the pipeline
- Make intelligent routing decisions based on quality assessments
- Provide improvement recommendations for failed validations
- Control workflow progression and rework cycles
- Verify trust levels between agents

## Key Features
- **Quality Assessment**: Aggregates and analyzes quality metrics from all validation agents
- **Routing Decision**: Determines next steps based on quality scores
- **Improvement Recommendations**: Provides actionable feedback for quality improvement
- **Workflow Control**: Manages pipeline flow, rework, and approval processes
- **Trust Verification**: Validates trust relationships between agents

## Technical Details
- **Agent Type**: `qualityControlManager`
- **Agent Number**: 6
- **Default Port**: 8006
- **Blockchain Address**: `0x976EA74026E726554dB657fA54763abd0C3a0aa9`
- **Registration Block**: 9

## Capabilities
- `quality_assessment`
- `routing_decision`
- `improvement_recommendations`
- `workflow_control`
- `trust_verification`

## Input/Output
- **Input**: Quality reports from Agents 4 and 5, validation results
- **Output**: Routing decisions, improvement recommendations, approval status

## Integration Points
- Receives validation results from Agent 4 (Calculation Validation)
- Receives QA reports from Agent 5 (QA Validation)
- Communicates with Agent 7 (Agent Manager) for workflow orchestration
- Routes approved data to publishing or back for rework
- Interfaces with trust system for agent verification

## Decision Logic
```yaml
qualityControlManager:
  thresholds:
    auto_approve: 0.95
    manual_review: 0.80
    auto_reject: 0.60
  routing_rules:
    - condition: "score >= 0.95"
      action: "approve_and_publish"
    - condition: "0.80 <= score < 0.95"
      action: "manual_review"
    - condition: "0.60 <= score < 0.80"
      action: "rework_required"
    - condition: "score < 0.60"
      action: "reject"
```

## Usage Example
```python
from a2aNetwork.sdk import Agent

# Initialize Quality Control Manager
qc_manager = Agent(
    agent_type="qualityControlManager",
    endpoint="http://localhost:8006"
)

# Submit quality assessment
decision = qc_manager.assess_quality({
    "product_id": "prod_123",
    "validation_results": {
        "calculation_validation": {
            "score": 0.92,
            "passed": True,
            "issues": []
        },
        "qa_validation": {
            "score": 0.88,
            "passed": True,
            "issues": ["minor_formatting"]
        }
    },
    "metadata": {
        "priority": "high",
        "data_type": "financial"
    }
})

print(f"Decision: {decision['action']}")
print(f"Overall Score: {decision['overall_score']}")
if decision['recommendations']:
    print(f"Recommendations: {decision['recommendations']}")
```

## Workflow States
1. **Pending Review**: Awaiting quality assessment
2. **Approved**: Meets all quality criteria
3. **Manual Review**: Requires human intervention
4. **Rework Required**: Needs improvement before approval
5. **Rejected**: Failed quality standards

## Quality Metrics
- **Completeness**: Data completeness percentage
- **Accuracy**: Validation accuracy score
- **Consistency**: Cross-validation consistency
- **Compliance**: Regulatory compliance status
- **Performance**: Processing efficiency metrics

## Error Codes
- `QCM001`: Invalid validation results format
- `QCM002`: Missing required quality metrics
- `QCM003`: Trust verification failed
- `QCM004`: Routing decision conflict
- `QCM005`: Workflow state transition error

## Monitoring
- Approval/rejection rates
- Average quality scores by data type
- Rework cycle frequency
- Manual review queue length
- Trust verification success rate

## Dependencies
- Quality scoring algorithms
- Workflow management system
- Trust verification modules
- Decision rule engine
- Audit logging system