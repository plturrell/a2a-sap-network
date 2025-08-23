# A2A Agent Port Registry

## Official Port Assignments

| Agent ID | Agent Name | Port | Backend Directory | Status |
|----------|------------|------|-------------------|---------|
| Agent 0 | Data Product Agent | 8000 | agent0DataProduct | ✅ Ready |
| Agent 1 | Data Standardization Agent | 8001 | agent1Standardization | ✅ Ready |
| Agent 2 | AI Preparation Agent | 8002 | agent2AiPreparation | ✅ Fixed (was 8001) |
| Agent 3 | Vector Processing Agent | 8003 | agent3VectorProcessing | ✅ Ready |
| Agent 4 | Calculation Validation Agent | 8004 | agent4CalcValidation | ✅ Ready |
| Agent 5 | QA Validation Agent | 8005 | agent5QaValidation | ✅ Ready |
| Agent 6 | Quality Control Agent | 8006 | agent6QualityControl | ✅ Ready |
| Agent 7 | Agent Manager | 8007 | agentManager | ✅ Ready |
| Agent 8 | Data Manager | 8008 | dataManager | ✅ Ready |
| Agent 9 | Reasoning Agent | 8086 | reasoningAgent | ✅ Fixed (added router & server) |
| Agent 10 | Calculator Agent | 8010 | calculationAgent | ✅ Ready |
| Agent 11 | SQL Query Engine | 8011 | sqlAgent | ✅ Ready |
| Agent 12 | Registry Agent | 8012 | catalogManager | ✅ Ready |
| Agent 13 | Agent Builder | 8013 | agentBuilder | ✅ Ready |
| Agent 14 | Embedding Fine-Tuner | 8014 | embeddingFineTuner | ✅ Ready (real backend) |
| Agent 15 | Orchestrator Agent | 8015 | orchestratorAgent | ✅ Ready (real backend) |

## Port Conflict Resolution
- **Agent 2**: Changed from 8001 to 8002 to avoid conflict with Agent 1
- **Agent 9**: Uses port 8086 (non-sequential, as configured in adapter)

## Backend Server Files
The following agents have dedicated server files:
- Agent 9: `agent9_server.py` (newly created)
- Agent 14: `agent14_server.py` 
- Agent 15: `agent15_server.py`

## Startup Scripts
The following agents have startup scripts:
- Agent 9: `start_agent9.sh`
- Agent 14: `start_agent14.sh`
- Agent 15: `start_agent15.sh`

## Main Backend Integration
Agents 0-5 and some named agents are integrated into the main backend application via routers in `main.py`.

## Service Layer Status
All agents have complete service and adapter implementations in `/Users/apple/projects/a2a/a2aNetwork/srv/`

## Notes
1. All agents use HTTP (not HTTPS) for local development
2. Each agent has its own dedicated port to avoid conflicts
3. Agents can be started individually or through the main backend application
4. Port assignments should not be changed without updating all related configurations