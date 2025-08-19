# Internal BPMN Analysis - Reasoning Agent

## Agent Internal Workflow Architecture

### 1. Main Entry Points (Start Events)

#### A. External Entry Point
- **@a2a_skill**: `multi_agent_reasoning`
- **@a2a_handler**: `executeReasoningTask`
- **Input**: ReasoningRequest with question, context, architecture type

#### B. Architecture Decision Gateway
```
Decision Point: request.architecture
├── hierarchical → _orchestrate_hierarchical_reasoning()
├── peer_to_peer → swarm_skills.peer_to_peer_reasoning()
├── blackboard → orchestration_skills.blackboard_reasoning()
├── hub_and_spoke → (default to hierarchical)
├── graph_based → (default to hierarchical)
└── hybrid → (default to hierarchical)
```

### 2. Hierarchical Reasoning Process (Main Flow)

#### Phase 1: Question Analysis
```mermaid
graph TD
    A[Start: Question Analysis] --> B{QA Agent Available?}
    B -->|Yes| C[Call QA Validation Agent]
    B -->|No| D[Throw Exception: No QA Agent]
    C --> E[Extract Reasoning Chain]
    E --> F[Convert to Sub-Questions]
    F --> G[Store in decomposed_questions]
```

**BPMN Elements**:
- **Service Task**: Query QA Validation Agent (http://localhost:8007)
- **Business Rule**: Extract sub-questions from reasoning chain
- **Data Store**: Update `state.decomposed_questions`

#### Phase 2: Evidence Retrieval
```mermaid
graph TD
    A[Start: Evidence Retrieval] --> B[Health Check Data Manager]
    B --> C{Data Manager Available?}
    C -->|No| D[Throw Exception: No Data Manager]
    C -->|Yes| E[For Each Sub-Question]
    E --> F[Query Data Manager: retrieve_data]
    F --> G[Store Evidence Results]
    G --> H[Update state.evidence_pool]
```

**BPMN Elements**:
- **Service Task**: Health check Data Manager
- **Multi-Instance Sub-Process**: Query for each sub-question
- **Data Store**: Update `state.evidence_pool`

#### Phase 3: Reasoning
```mermaid
graph TD
    A[Start: Reasoning] --> B{External Reasoning Agents Available?}
    B -->|No| C[Throw Exception: No Reasoning Agents]
    B -->|Yes| D[Discover Reasoning Agents via Catalog Manager]
    D --> E[For Each Evidence/Question Pair]
    E --> F[Delegate to Reasoning Engine Agent]
    F --> G[Collect Reasoning Chains]
    G --> H[Update state.reasoning_chains]
```

**BPMN Elements**:
- **Service Task**: Query Catalog Manager for agent discovery
- **Multi-Instance Sub-Process**: Parallel reasoning tasks
- **Data Store**: Update `state.reasoning_chains`

#### Phase 4: Multi-Agent Debate (Optional)
```mermaid
graph TD
    A[Start: Debate] --> B{Debate Enabled?}
    B -->|No| H[Skip to Synthesis]
    B -->|Yes| C[Initialize Debate Round]
    C --> D[For Each Reasoning Chain]
    D --> E[Generate Argument Position]
    E --> F[Calculate Argument Confidence]
    F --> G{Consensus Reached?}
    G -->|No| I[Next Debate Round]
    G -->|Yes| J[Update Final Confidences]
    I --> C
    J --> H[Continue to Synthesis]
```

**BPMN Elements**:
- **Decision Gateway**: Check if debate enabled
- **Loop Sub-Process**: Debate rounds (max 3)
- **Business Rule**: Consensus detection algorithm

#### Phase 5: Answer Synthesis
```mermaid
graph TD
    A[Start: Synthesis] --> B{External Synthesis Agents Available?}
    B -->|No| C[Throw Exception: No Synthesis Agents]
    B -->|Yes| D[Query Agent Manager for Synthesizers]
    D --> E[Delegate to Answer Synthesizer]
    E --> F[Select Best Answer by Confidence]
    F --> G[Return Final Result]
```

**BPMN Elements**:
- **Service Task**: Query Agent Manager for synthesizer agents
- **Business Rule**: Select highest confidence answer
- **End Event**: Return result

### 3. Internal Skills Breakdown

#### A. MultiAgentReasoningSkills Class

**Skills Available**:
1. **@a2a_skill**: `hierarchical_question_decomposition`
   - Input: question, max_depth, strategy, context
   - Process: Recursive decomposition using strategy patterns
   - Output: Hierarchical sub-question tree

2. **@a2a_skill**: `multi_agent_consensus`
   - Input: proposals, consensus_method, threshold
   - Process: Voting, weighted average, debate, or emergence
   - Output: Consensus result with confidence

3. **@a2a_skill**: `blackboard_reasoning`
   - Input: problem_context, knowledge_sources
   - Process: Pattern recognition, logical reasoning, evidence evaluation
   - Output: Integrated reasoning result

**Internal Methods**:
- `_decompose_recursively()` - Recursive question breakdown
- `_generate_sub_questions()` - Question generation strategies
- `_voting_consensus()` - Democratic voting mechanism
- `_weighted_consensus()` - Weighted average consensus
- `_debate_consensus()` - Adversarial debate process
- `_emergence_consensus()` - Emergent consensus detection

#### B. Circuit Breaker Integration
```mermaid
graph TD
    A[A2A Agent Call] --> B[Get Circuit Breaker]
    B --> C{Breaker Open?}
    C -->|Yes| D[Throw Exception: Circuit Open]
    C -->|No| E[Execute Request]
    E --> F{Request Successful?}
    F -->|Yes| G[Record Success]
    F -->|No| H[Record Failure]
    H --> I{Failure Threshold Reached?}
    I -->|Yes| J[Open Circuit]
    I -->|No| K[Continue]
    G --> K[Return Result]
```

### 4. Sub-Agent Pool Management

#### Agent Role Mapping:
```
QUESTION_ANALYZER → QA Validation Agent (localhost:8007)
EVIDENCE_RETRIEVER → Data Manager (configured URL)
REASONING_ENGINE → Dynamic discovery via Catalog Manager
ANSWER_SYNTHESIZER → Dynamic discovery via Agent Manager
VALIDATOR → Dynamic discovery via Catalog Manager
```

#### Dynamic Agent Discovery Process:
```mermaid
graph TD
    A[Need Agent for Role] --> B[Check Sub-Agent Pool]
    B --> C{Agent Configured?}
    C -->|Yes| D[Use Configured Endpoint]
    C -->|No| E[Query Catalog Manager]
    E --> F{Agents Discovered?}
    F -->|Yes| G[Use First Available Agent]
    F -->|No| H[Throw Exception: No Agent Available]
```

### 5. Blockchain Message Flow

#### Message Signing Process:
```mermaid
graph TD
    A[Prepare A2A Message] --> B[Add Timestamp & Sender]
    B --> C{Trust Identity Available?}
    C -->|Yes| D[Sign with Private Key]
    C -->|No| E[Send Unsigned]
    D --> F[Add Signature & Signer]
    F --> G[Send to Target Agent]
    E --> G
```

#### Message Verification Process:
```mermaid
graph TD
    A[Receive A2A Response] --> B{Contains Signature?}
    B -->|No| C[Accept Unverified]
    B -->|Yes| D[Extract Signature & Signer]
    D --> E[Verify with Blockchain]
    E --> F{Signature Valid?}
    F -->|Yes| G[Accept Response]
    F -->|No| H[Throw Security Exception]
```

### 6. Error Handling & Failure Modes

#### No Fallback Policy:
```mermaid
graph TD
    A[External Agent Call Fails] --> B[Circuit Breaker Triggered?]
    B -->|Yes| C[Immediate Failure]
    B -->|No| D[Retry Logic]
    D --> E{Max Retries Reached?}
    E -->|Yes| F[Propagate Exception]
    E -->|No| G[Retry Request]
    F --> H[Session Cleanup]
    G --> A
```

### 7. Performance Monitoring Integration

#### Metrics Collection:
```mermaid
graph TD
    A[Reasoning Session Start] --> B[Initialize Session Metrics]
    B --> C[Execute Reasoning Flow]
    C --> D[Track Phase Timings]
    D --> E[Record Agent Interactions]
    E --> F[Calculate Success Metrics]
    F --> G[Update Global Metrics]
    G --> H[Export to Telemetry]
```

**Tracked Metrics**:
- `total_sessions`: Counter of reasoning sessions
- `successful_reasoning`: Counter of successful completions
- `average_confidence`: Average confidence scores
- `architecture_usage`: Usage per architecture type
- `average_reasoning_time`: Performance timing

### 8. Trust & Security Flow

#### Trust Initialization:
```mermaid
graph TD
    A[Agent Initialize] --> B[Load Private Key from ENV]
    B --> C[Call initialize_agent_trust()]
    C --> D[Register with Blockchain]
    D --> E[Store Trust Identity]
    E --> F[Enable Message Signing]
```

#### Trusted Agent Management:
```mermaid
graph TD
    A[Receive Agent Response] --> B[Extract Signer Address]
    B --> C{Signer in Trusted Set?}
    C -->|Yes| D[Higher Trust Score]
    C -->|No| E[Standard Verification]
    D --> F[Process Response]
    E --> F
```

## Summary

The Reasoning Agent implements a sophisticated internal BPMN workflow with:

✅ **Proper BPMN Structure**:
- Clear start/end events
- Decision gateways for architecture selection
- Service tasks for external agent calls
- Multi-instance sub-processes for parallel execution
- Data stores for state management
- Error boundary events for exception handling

✅ **No Internal Fallbacks**:
- All reasoning delegated to external agents
- Proper error propagation
- Circuit breaker protection

✅ **Blockchain Integration**:
- Message signing/verification
- Trust identity management
- Security exception handling

The agent follows enterprise BPMN 2.0 patterns for reliable, auditable, and scalable multi-agent coordination.