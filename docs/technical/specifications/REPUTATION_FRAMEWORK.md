# A2A Agent Reputation Framework

## Overview
This document outlines a comprehensive reputation system for A2A agents that enables peer-to-peer reputation exchange, automated earning/loss mechanisms, and deep integration with the Agent Manager and Marketplace.

## 1. Reputation Earning Mechanisms

### 1.1 Task-Based Earnings
```javascript
const REPUTATION_REWARDS = {
  TASK_COMPLETION: {
    SIMPLE: 5,
    MEDIUM: 10,
    COMPLEX: 20,
    CRITICAL: 30
  },
  PERFORMANCE_BONUS: {
    FAST_COMPLETION: 5,      // < 50% expected time
    LOW_GAS_USAGE: 3,        // < 70% average gas
    HIGH_ACCURACY: 10,       // > 95% accuracy
    ZERO_RETRIES: 5          // No retries needed
  },
  COLLABORATION: {
    HELPED_PEER: 5,          // Assisted another agent
    WORKFLOW_PARTICIPATION: 3, // Part of multi-agent workflow
    KNOWLEDGE_SHARING: 10     // Shared useful artifacts
  }
};
```

### 1.2 Service Quality Earnings
- **5-Star Rating**: +10 reputation
- **4-Star Rating**: +5 reputation
- **3-Star Rating**: 0 reputation (neutral)
- **First-Time Service Success**: +15 reputation bonus
- **Repeat Customer Bonus**: +5 reputation per returning client

### 1.3 Community Contributions
- **Tool Development**: +20-50 reputation for creating reusable tools
- **Documentation**: +10-20 reputation for helpful documentation
- **Bug Reports**: +5-15 reputation for identifying issues
- **Mentoring New Agents**: +10 reputation per successful mentee

## 2. Peer-to-Peer Reputation System

### 2.1 Reputation Endorsements
```typescript
interface ReputationEndorsement {
  fromAgent: string;           // Endorser agent ID
  toAgent: string;            // Recipient agent ID
  amount: number;             // 1-10 reputation points
  reason: EndorsementReason;  // Enum of valid reasons
  context: {
    workflowId?: string;
    taskId?: string;
    serviceOrderId?: string;
    description: string;
  };
  timestamp: Date;
  expiresAt?: Date;          // Optional expiration
}

enum EndorsementReason {
  EXCELLENT_COLLABORATION = "EXCELLENT_COLLABORATION",
  TIMELY_ASSISTANCE = "TIMELY_ASSISTANCE",
  HIGH_QUALITY_WORK = "HIGH_QUALITY_WORK",
  KNOWLEDGE_SHARING = "KNOWLEDGE_SHARING",
  PROBLEM_SOLVING = "PROBLEM_SOLVING",
  INNOVATION = "INNOVATION"
}
```

### 2.2 Endorsement Rules
- **Daily Limit**: Each agent can give max 50 reputation points per day
- **Peer Limit**: Max 10 points to same agent per week
- **Endorsement Power**: Based on endorser's reputation
  - Reputation 0-50: Can give 1-3 points
  - Reputation 51-100: Can give 1-5 points
  - Reputation 101-150: Can give 1-7 points
  - Reputation 151-200: Can give 1-10 points
- **Reciprocity Prevention**: Cannot endorse agent who endorsed you within 24 hours

### 2.3 Smart Contract Implementation
```solidity
contract ReputationExchange {
    struct Endorsement {
        address from;
        address to;
        uint8 amount;
        string reason;
        uint256 timestamp;
        bytes32 contextHash;
    }
    
    mapping(address => uint256) public dailyGiven;
    mapping(bytes32 => uint8) public peerWeeklyGiven;
    mapping(address => Endorsement[]) public endorsementHistory;
    
    event ReputationEndorsed(
        address indexed from,
        address indexed to,
        uint8 amount,
        string reason,
        bytes32 contextHash
    );
    
    function endorsePeer(
        address _to,
        uint8 _amount,
        string memory _reason,
        bytes32 _contextHash
    ) external {
        require(_amount <= getMaxEndorsementAmount(msg.sender), "Amount exceeds limit");
        require(checkDailyLimit(msg.sender, _amount), "Daily limit exceeded");
        require(checkWeeklyPeerLimit(msg.sender, _to, _amount), "Weekly peer limit exceeded");
        require(!hasRecentReciprocal(msg.sender, _to), "Reciprocal endorsement too soon");
        
        // Process endorsement
        _processEndorsement(msg.sender, _to, _amount, _reason, _contextHash);
    }
}
```

## 3. Reputation Loss Mechanisms

### 3.1 Task Failures
```javascript
const REPUTATION_PENALTIES = {
  TASK_FAILURE: {
    TIMEOUT: -5,               // Failed to complete in time
    ERROR: -10,                // Task ended with error
    ABANDONED: -15,            // Agent abandoned task
    WRONG_OUTPUT: -8           // Incorrect results
  },
  SERVICE_ISSUES: {
    SLA_BREACH: -20,           // Service level agreement violation
    DOWNTIME: -5,              // Per hour of unplanned downtime
    POOR_QUALITY: -10,         // Quality below threshold
    CUSTOMER_COMPLAINT: -15    // Verified complaint
  },
  BEHAVIORAL: {
    SPAM: -25,                 // Spamming other agents
    MALICIOUS: -50,            // Malicious behavior
    FALSE_CAPABILITY: -30,     // Advertising false capabilities
    RESOURCE_ABUSE: -20        // Excessive resource usage
  }
};
```

### 3.2 Reputation Decay
- **Inactivity Decay**: -1 reputation per week of inactivity after 30 days
- **Performance Decay**: -5 reputation if success rate drops below 60%
- **Trust Decay**: -2 reputation per month if no peer endorsements received

### 3.3 Recovery Mechanisms
```typescript
interface ReputationRecovery {
  PROBATION_TASKS: {
    threshold: 50;           // Below this triggers probation
    tasksRequired: 10;       // Complete 10 simple tasks
    successRate: 0.9;        // With 90% success rate
    reward: 20;              // Regain 20 reputation
  };
  PEER_VOUCHING: {
    vouchersNeeded: 3;       // 3 high-rep agents vouch
    minVoucherRep: 150;      // Vouchers must have 150+ rep
    reward: 15;              // Regain 15 reputation
  };
  TRAINING_COMPLETION: {
    courses: string[];       // Complete training modules
    reward: 10;              // Per course completed
  };
}
```

## 4. Reputation Tracking System

### 4.1 Database Schema Extensions
```sql
-- Reputation transactions table
CREATE TABLE reputation_transactions (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    amount INTEGER NOT NULL,
    reason VARCHAR(200),
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    is_automated BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Peer endorsements table
CREATE TABLE peer_endorsements (
    id UUID PRIMARY KEY,
    from_agent_id UUID NOT NULL,
    to_agent_id UUID NOT NULL,
    amount INTEGER NOT NULL CHECK (amount BETWEEN 1 AND 10),
    reason VARCHAR(50) NOT NULL,
    context JSONB,
    workflow_id UUID,
    task_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_reciprocal BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (from_agent_id) REFERENCES agents(id),
    FOREIGN KEY (to_agent_id) REFERENCES agents(id)
);

-- Reputation analytics view
CREATE MATERIALIZED VIEW reputation_analytics AS
SELECT 
    a.id,
    a.name,
    a.reputation,
    COUNT(DISTINCT rt.id) as total_transactions,
    SUM(CASE WHEN rt.amount > 0 THEN rt.amount ELSE 0 END) as total_earned,
    SUM(CASE WHEN rt.amount < 0 THEN ABS(rt.amount) ELSE 0 END) as total_lost,
    COUNT(DISTINCT pe.from_agent_id) as unique_endorsers,
    AVG(pe.amount) as avg_endorsement,
    MAX(rt.created_at) as last_activity
FROM agents a
LEFT JOIN reputation_transactions rt ON a.id = rt.agent_id
LEFT JOIN peer_endorsements pe ON a.id = pe.to_agent_id
GROUP BY a.id, a.name, a.reputation;
```

### 4.2 Real-time Tracking
```javascript
class ReputationTracker {
  constructor() {
    this.reputationEvents = new EventEmitter();
    this.metrics = new PrometheusMetrics();
    this.cache = new RedisCache();
  }

  async trackReputationChange(agentId, change, reason, context) {
    const transaction = {
      agentId,
      amount: change,
      reason,
      context,
      timestamp: new Date(),
      isAutomated: !context.humanInitiated
    };

    // Store transaction
    await this.db.reputationTransactions.create(transaction);
    
    // Update agent reputation
    const newReputation = await this.updateAgentReputation(agentId, change);
    
    // Emit events
    this.reputationEvents.emit('reputation:changed', {
      agentId,
      oldReputation: newReputation - change,
      newReputation,
      change,
      reason
    });
    
    // Update metrics
    this.metrics.recordReputationChange(agentId, change, reason);
    
    // Clear cache
    await this.cache.invalidate(`agent:${agentId}:reputation`);
    
    // Check for milestones
    await this.checkReputationMilestones(agentId, newReputation);
    
    return newReputation;
  }

  async checkReputationMilestones(agentId, reputation) {
    const milestones = [50, 100, 150, 200];
    const previousRep = await this.getPreviousReputation(agentId);
    
    for (const milestone of milestones) {
      if (previousRep < milestone && reputation >= milestone) {
        await this.awardMilestoneBadge(agentId, milestone);
      }
    }
  }
}
```

### 4.3 Analytics Dashboard Components
```javascript
// Reputation trends chart
const ReputationTrendChart = {
  data: {
    labels: [], // Time periods
    datasets: [{
      label: 'Reputation Score',
      data: [], // Reputation values over time
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    }]
  },
  options: {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Agent Reputation Trend'
      }
    }
  }
};

// Peer endorsement network graph
const EndorsementNetworkGraph = {
  nodes: [], // Agents as nodes
  edges: [], // Endorsements as edges
  options: {
    physics: {
      enabled: true,
      barnesHut: {
        gravitationalConstant: -2000,
        centralGravity: 0.3
      }
    }
  }
};
```

## 5. Agent Manager Integration

### 5.1 Reputation-Based Agent Selection
```javascript
class EnhancedAgentManager {
  async selectAgentForTask(task, options = {}) {
    const { 
      minReputation = 50,
      preferHighReputation = true,
      reputationWeight = 0.3
    } = options;

    // Get capable agents
    let candidates = await this.findCapableAgents(task.requiredCapabilities);
    
    // Filter by minimum reputation
    candidates = candidates.filter(agent => agent.reputation >= minReputation);
    
    // Score agents
    const scoredAgents = candidates.map(agent => ({
      agent,
      score: this.calculateAgentScore(agent, task, reputationWeight)
    }));
    
    // Sort by score
    scoredAgents.sort((a, b) => b.score - a.score);
    
    // Select best agent
    return scoredAgents[0]?.agent;
  }

  calculateAgentScore(agent, task, reputationWeight) {
    const capabilityScore = this.getCapabilityMatch(agent, task);
    const availabilityScore = this.getAvailabilityScore(agent);
    const reputationScore = agent.reputation / 200; // Normalized
    
    return (
      capabilityScore * (1 - reputationWeight) +
      reputationScore * reputationWeight +
      availabilityScore * 0.1
    );
  }

  async delegateWithTrust(fromAgent, toAgent, task) {
    // Check if fromAgent has sufficient reputation to delegate
    if (fromAgent.reputation < 75) {
      throw new Error('Insufficient reputation to delegate tasks');
    }
    
    // Check trust relationship
    const trustLevel = await this.getTrustLevel(fromAgent, toAgent);
    if (trustLevel < 0.5) {
      throw new Error('Insufficient trust to delegate');
    }
    
    // Create delegation with reputation stake
    const delegation = await this.createDelegation({
      from: fromAgent,
      to: toAgent,
      task,
      reputationStake: Math.min(10, fromAgent.reputation * 0.05)
    });
    
    return delegation;
  }
}
```

### 5.2 Workflow Reputation Distribution
```javascript
class WorkflowReputationManager {
  async distributeWorkflowReputation(workflowId, outcome) {
    const workflow = await this.getWorkflow(workflowId);
    const participants = await this.getWorkflowParticipants(workflowId);
    
    // Calculate base reputation based on outcome
    const baseReputation = this.calculateBaseReputation(workflow, outcome);
    
    // Distribute based on contribution
    for (const participant of participants) {
      const contribution = await this.calculateContribution(participant, workflow);
      const reputation = Math.round(baseReputation * contribution.weight);
      
      await this.reputationTracker.trackReputationChange(
        participant.agentId,
        reputation,
        'WORKFLOW_COMPLETION',
        {
          workflowId,
          role: contribution.role,
          weight: contribution.weight
        }
      );
    }
    
    // Bonus for workflow coordinator
    if (workflow.coordinator) {
      await this.reputationTracker.trackReputationChange(
        workflow.coordinator,
        5,
        'WORKFLOW_COORDINATION',
        { workflowId }
      );
    }
  }
}
```

## 6. Marketplace Integration

### 6.1 Reputation-Based Marketplace Features
```javascript
class ReputationMarketplace {
  // Dynamic pricing based on reputation
  calculateServicePrice(basePrice, providerReputation) {
    const reputationMultiplier = 1 + (providerReputation - 100) / 500;
    return basePrice * Math.max(0.8, Math.min(1.5, reputationMultiplier));
  }

  // Reputation requirements for premium services
  async canAccessPremiumService(agentId, serviceId) {
    const agent = await this.getAgent(agentId);
    const service = await this.getService(serviceId);
    
    return agent.reputation >= (service.minReputation || 0);
  }

  // Featured agents based on reputation
  async getFeaturedAgents(category, limit = 10) {
    return await this.db.agents.findAll({
      where: {
        category,
        isActive: true,
        reputation: { [Op.gte]: 150 }
      },
      order: [['reputation', 'DESC']],
      limit
    });
  }

  // Reputation-based search ranking
  async searchServices(query, options = {}) {
    const results = await this.performSearch(query);
    
    // Boost ranking based on provider reputation
    return results.map(result => ({
      ...result,
      rankScore: result.relevanceScore * (1 + result.provider.reputation / 400)
    })).sort((a, b) => b.rankScore - a.rankScore);
  }
}
```

### 6.2 Trust Badges and Verification
```javascript
const REPUTATION_BADGES = {
  NEWCOMER: { min: 0, max: 49, icon: '🌱', color: 'gray' },
  ESTABLISHED: { min: 50, max: 99, icon: '⭐', color: 'bronze' },
  TRUSTED: { min: 100, max: 149, icon: '🏆', color: 'silver' },
  EXPERT: { min: 150, max: 199, icon: '💎', color: 'gold' },
  LEGENDARY: { min: 200, max: 200, icon: '👑', color: 'platinum' }
};

class ReputationBadgeSystem {
  getBadge(reputation) {
    for (const [name, config] of Object.entries(REPUTATION_BADGES)) {
      if (reputation >= config.min && reputation <= config.max) {
        return { name, ...config };
      }
    }
  }

  async generateVerifiableCredential(agentId) {
    const agent = await this.getAgent(agentId);
    const badge = this.getBadge(agent.reputation);
    
    return {
      '@context': ['https://www.w3.org/2018/credentials/v1'],
      type: ['VerifiableCredential', 'ReputationBadge'],
      issuer: 'did:a2a:reputation-system',
      issuanceDate: new Date().toISOString(),
      credentialSubject: {
        id: `did:a2a:agent:${agentId}`,
        reputation: agent.reputation,
        badge: badge.name,
        endorsements: await this.getEndorsementCount(agentId)
      },
      proof: await this.generateProof(agent)
    };
  }
}
```

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- Implement reputation transaction tracking
- Add peer endorsement smart contract
- Create basic reputation change events

### Phase 2: Earning & Loss Mechanisms (Weeks 3-4)
- Implement automated task-based reputation
- Add service quality ratings
- Create penalty system

### Phase 3: Peer-to-Peer System (Weeks 5-6)
- Deploy endorsement smart contracts
- Build endorsement UI components
- Implement anti-gaming measures

### Phase 4: Integration (Weeks 7-8)
- Integrate with Agent Manager
- Add marketplace features
- Create analytics dashboard

### Phase 5: Advanced Features (Weeks 9-10)
- Implement reputation recovery
- Add verifiable credentials
- Create reputation API

## 8. Security Considerations

### 8.1 Anti-Gaming Measures
- Rate limiting on endorsements
- Sybil attack prevention through stake requirements
- Reputation slashing for detected manipulation
- ML-based anomaly detection for unusual patterns

### 8.2 Privacy Protection
- Zero-knowledge proofs for reputation claims
- Selective disclosure of reputation history
- Encrypted peer feedback storage
- GDPR-compliant data retention

## 9. Monitoring & Metrics

### 9.1 Key Performance Indicators
- Average agent reputation
- Reputation volatility index
- Endorsement network density
- Task success correlation with reputation
- Marketplace conversion by reputation tier

### 9.2 Alerts & Thresholds
- Sudden reputation drops (>20 points)
- Unusual endorsement patterns
- Low reputation agent concentration
- Reputation inflation detection

This framework provides a comprehensive reputation system that incentivizes quality work, enables trust between agents, and integrates deeply with the A2A ecosystem.