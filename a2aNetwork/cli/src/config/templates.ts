import { ConfigTemplate } from './types';

export const defaultTemplates: { [key: string]: ConfigTemplate } = {
  'data-processor': {
    name: 'Data Processing Agent',
    description: 'Agent for data cleaning, transformation, and validation',
    capabilities: ['data_cleaning', 'data_validation', 'data_transformation', 'format_conversion'],
    dependencies: {
      required: ['express', 'axios', '@a2a/sdk'],
      optional: ['joi', 'ajv', 'csv-parser', 'xlsx'],
      dev: ['jest', 'supertest', 'nodemon']
    },
    files: {
      'src/index.js': `const { Agent } = require('@a2a/sdk');
const express = require('express');

const agent = new Agent({
  name: process.env.AGENT_NAME,
  type: 'data-processor',
  capabilities: ['data_cleaning', 'data_validation']
});

// Data processing endpoint
agent.addService('process', async (data) => {
  // Your data processing logic here
  const cleaned = cleanData(data);
  const validated = validateData(cleaned);
  return validated;
});

function cleanData(data) {
  // Implement data cleaning
  return data;
}

function validateData(data) {
  // Implement validation
  return data;
}

// Start the agent
agent.start();
`,
      'src/services/cleaner.js': `// Data cleaning service
module.exports = {
  cleanText: (text) => {
    return text.trim().toLowerCase();
  },
  
  cleanNumber: (num) => {
    return parseFloat(num) || 0;
  },
  
  cleanDate: (date) => {
    return new Date(date).toISOString();
  }
};
`
    }
  },

  'ai-ml': {
    name: 'AI/ML Agent',
    description: 'Agent for machine learning inference and AI operations',
    capabilities: ['inference', 'model_serving', 'feature_extraction', 'prediction'],
    dependencies: {
      required: ['express', 'axios', '@a2a/sdk', '@tensorflow/tfjs-node'],
      optional: ['onnxruntime-node', 'sharp', 'natural'],
      dev: ['jest', 'supertest', 'nodemon']
    },
    files: {
      'src/index.js': `const { Agent } = require('@a2a/sdk');
const tf = require('@tensorflow/tfjs-node');

const agent = new Agent({
  name: process.env.AGENT_NAME,
  type: 'ai-ml',
  capabilities: ['inference', 'prediction']
});

let model;

// Initialize model
agent.on('start', async () => {
  // Load your model here
  // model = await tf.loadLayersModel('path/to/model.json');
  console.log('Model loaded successfully');
});

// Inference endpoint
agent.addService('predict', async (input) => {
  if (!model) {
    throw new Error('Model not loaded');
  }
  
  // Prepare input
  const tensor = tf.tensor(input);
  
  // Make prediction
  const prediction = model.predict(tensor);
  const result = await prediction.array();
  
  // Cleanup
  tensor.dispose();
  prediction.dispose();
  
  return result;
});

// Start the agent
agent.start();
`
    }
  },

  'orchestrator': {
    name: 'Orchestration Agent',
    description: 'Agent for coordinating workflows between multiple agents',
    capabilities: ['workflow_execution', 'task_scheduling', 'agent_coordination', 'pipeline_management'],
    dependencies: {
      required: ['express', 'axios', '@a2a/sdk', 'bull', 'p-queue'],
      optional: ['node-cron', 'agenda', 'graphlib'],
      dev: ['jest', 'supertest', 'nodemon']
    },
    files: {
      'src/index.js': `const { Agent, Registry } = require('@a2a/sdk');
const Queue = require('bull');

const agent = new Agent({
  name: process.env.AGENT_NAME,
  type: 'orchestrator',
  capabilities: ['workflow_execution', 'task_scheduling']
});

const taskQueue = new Queue('tasks');
const registry = new Registry();

// Workflow execution
agent.addService('executeWorkflow', async (workflow) => {
  const { steps, data } = workflow;
  let result = data;
  
  for (const step of steps) {
    // Discover agent for this step
    const targetAgent = await registry.discover(step.capability);
    
    if (!targetAgent) {
      throw new Error(\`No agent found for capability: \${step.capability}\`);
    }
    
    // Execute step
    result = await targetAgent.call(step.service, result);
  }
  
  return result;
});

// Task scheduling
agent.addService('scheduleTask', async (task) => {
  const job = await taskQueue.add(task, {
    delay: task.delay || 0,
    attempts: task.attempts || 3,
    backoff: {
      type: 'exponential',
      delay: 2000
    }
  });
  
  return { jobId: job.id };
});

// Process tasks
taskQueue.process(async (job) => {
  const { agentName, service, data } = job.data;
  const targetAgent = await registry.getAgent(agentName);
  return await targetAgent.call(service, data);
});

// Start the agent
agent.start();
`,
      'workflows/example.yaml': `name: DataProcessingPipeline
description: Example data processing workflow
steps:
  - name: Clean Data
    capability: data_cleaning
    service: clean
    
  - name: Validate Data
    capability: data_validation
    service: validate
    
  - name: Transform Data
    capability: data_transformation
    service: transform
    
  - name: Store Data
    capability: data_storage
    service: store
`
    }
  },

  'agent': {
    name: 'Generic Agent',
    description: 'Basic agent template',
    capabilities: [],
    dependencies: {
      required: ['express', 'axios', '@a2a/sdk'],
      optional: [],
      dev: ['jest', 'supertest', 'nodemon']
    },
    files: {
      'src/index.js': `const { Agent } = require('@a2a/sdk');

const agent = new Agent({
  name: process.env.AGENT_NAME,
  type: process.env.AGENT_TYPE || 'custom',
  capabilities: []  // Add your capabilities
});

// Add your services here
agent.addService('hello', async (data) => {
  return { message: 'Hello from ' + agent.name, data };
});

// Start the agent
agent.start();
`
    }
  },

  'workflow': {
    name: 'Multi-Agent Workflow',
    description: 'Template for multi-agent applications',
    capabilities: [],
    dependencies: {
      required: ['express', 'axios', '@a2a/sdk', '@a2a/workflow'],
      optional: ['bull', 'p-queue'],
      dev: ['jest', 'supertest', 'nodemon', 'concurrently']
    },
    files: {
      'agents/processor/index.js': `const { Agent } = require('@a2a/sdk');

const agent = new Agent({
  name: 'processor',
  capabilities: ['data_processing']
});

agent.addService('process', async (data) => {
  // Processing logic
  return { ...data, processed: true };
});

agent.start();
`,
      'agents/validator/index.js': `const { Agent } = require('@a2a/sdk');

const agent = new Agent({
  name: 'validator',
  capabilities: ['data_validation']
});

agent.addService('validate', async (data) => {
  // Validation logic
  const isValid = data.processed === true;
  return { ...data, valid: isValid };
});

agent.start();
`,
      'orchestrator/index.js': `const { Workflow, Registry } = require('@a2a/sdk');

const workflow = new Workflow({
  name: 'data-pipeline',
  registry: new Registry()
});

// Define workflow
workflow.step('process', { capability: 'data_processing' });
workflow.step('validate', { capability: 'data_validation' });

// Execute workflow
async function run() {
  const result = await workflow.execute({ 
    input: 'test data' 
  });
  console.log('Workflow result:', result);
}

run();
`
    }
  },

  'full-stack': {
    name: 'Full Stack A2A Application',
    description: 'Complete application with UI, agents, and blockchain',
    capabilities: [],
    dependencies: {
      required: [
        'express', 
        'axios', 
        '@a2a/sdk', 
        '@a2a/ui',
        'react',
        'react-dom',
        'web3'
      ],
      optional: [
        '@sap/cds',
        'ethers',
        'hardhat'
      ],
      dev: [
        'jest',
        'supertest',
        'nodemon',
        'concurrently',
        '@types/node',
        'typescript'
      ]
    },
    files: {
      'frontend/src/App.js': `import React, { useState, useEffect } from 'react';
import { A2AClient } from '@a2a/ui';

function App() {
  const [agents, setAgents] = useState([]);
  const client = new A2AClient();
  
  useEffect(() => {
    client.getAgents().then(setAgents);
  }, []);
  
  return (
    <div className="app">
      <h1>A2A Network Dashboard</h1>
      <div className="agents">
        {agents.map(agent => (
          <div key={agent.id} className="agent-card">
            <h3>{agent.name}</h3>
            <p>Type: {agent.type}</p>
            <p>Status: {agent.status}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
`,
      'backend/src/server.js': `const express = require('express');
const { Registry, Blockchain } = require('@a2a/sdk');

const app = express();
const registry = new Registry();
const blockchain = new Blockchain();

app.use(express.json());

// API routes
app.get('/api/agents', async (req, res) => {
  const agents = await registry.getAllAgents();
  res.json(agents);
});

app.post('/api/agents/register', async (req, res) => {
  const { agent } = req.body;
  const tx = await blockchain.registerAgent(agent);
  const registered = await registry.register(agent);
  res.json({ tx, registered });
});

app.listen(process.env.PORT || 3001);
`,
      'contracts/AgentRegistry.sol': `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AgentRegistry {
    struct Agent {
        address owner;
        string name;
        string agentType;
        bool active;
    }
    
    mapping(address => Agent) public agents;
    
    event AgentRegistered(address indexed agent, string name);
    
    function registerAgent(string memory _name, string memory _agentType) public {
        agents[msg.sender] = Agent({
            owner: msg.sender,
            name: _name,
            agentType: _agentType,
            active: true
        });
        
        emit AgentRegistered(msg.sender, _name);
    }
}
`
    }
  }
};