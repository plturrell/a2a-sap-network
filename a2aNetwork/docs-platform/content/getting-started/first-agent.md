---
title: Your First A2A Agent
description: Create and run your first A2A agent in minutes
difficulty: beginner
duration: 15 minutes
interactive: true
---

# Your First A2A Agent

Welcome to A2A! In this tutorial, you'll create your first agent and see it in action. By the end, you'll have a working data processing agent that can clean and validate data.

## What You'll Learn

- How to create an A2A agent
- How to add services to your agent
- How to test your agent locally
- Basic agent communication patterns

## Prerequisites

- Node.js 16+ installed
- Basic JavaScript knowledge
- A2A CLI installed (`npm install -g @a2a/cli`)

## Step 1: Create Your Agent

Let's start by creating a simple data processing agent. This agent will be able to clean text data and validate it.

```interactive:javascript
const { A2A } = global;

// Create a new agent
const agent = new A2A.Agent({
  name: 'data-cleaner',
  type: 'data-processor',
  capabilities: ['text_cleaning', 'data_validation']
});

console.log('Agent created:', agent.config.name);
```

## Step 2: Add Services

Now let's add some useful services to our agent. Services are the functions that other agents (or users) can call.

```interactive:javascript
// Add a text cleaning service
agent.addService('clean', async (data) => {
  if (!data || typeof data !== 'string') {
    throw new Error('Input must be a string');
  }
  
  // Clean the text
  const cleaned = data
    .trim()                    // Remove whitespace
    .toLowerCase()             // Convert to lowercase
    .replace(/[^\w\s]/g, '')   // Remove special characters
    .replace(/\s+/g, ' ');     // Normalize spaces
  
  console.log(`Cleaned: "${data}" → "${cleaned}"`);
  return { original: data, cleaned: cleaned };
});

// Add a validation service
agent.addService('validate', async (data) => {
  const rules = {
    minLength: 3,
    maxLength: 100,
    noNumbers: true
  };
  
  const errors = [];
  
  if (data.length < rules.minLength) {
    errors.push(`Text too short (minimum ${rules.minLength} characters)`);
  }
  
  if (data.length > rules.maxLength) {
    errors.push(`Text too long (maximum ${rules.maxLength} characters)`);
  }
  
  if (rules.noNumbers && /\d/.test(data)) {
    errors.push('Text contains numbers');
  }
  
  const isValid = errors.length === 0;
  console.log(`Validation result: ${isValid ? '✅ Valid' : '❌ Invalid'}`);
  
  return {
    valid: isValid,
    errors: errors,
    data: data
  };
});

console.log('Services added successfully!');
```

## Step 3: Start Your Agent

```interactive:javascript
// Start the agent
await agent.start();

// Check agent status
const status = agent.getStatus();
console.log('Agent status:', status);
```

## Step 4: Test Your Agent

Now let's test our agent by calling its services:

```interactive:javascript
// Test the cleaning service
const testText = "  Hello, World! 123  ";
const cleanResult = await agent.call('clean', testText);
console.log('Clean result:', cleanResult);

// Test the validation service
const validationResult = await agent.call('validate', cleanResult.cleaned);
console.log('Validation result:', validationResult);
```

## Step 5: Create a Complete Workflow

Let's combine both services into a complete data processing workflow:

```interactive:javascript
// Combined workflow function
async function processData(rawData) {
  console.log(`\n🔄 Processing: "${rawData}"`);
  
  try {
    // Step 1: Clean the data
    const cleanResult = await agent.call('clean', rawData);
    console.log(`✨ Cleaned: "${cleanResult.cleaned}"`);
    
    // Step 2: Validate the cleaned data
    const validationResult = await agent.call('validate', cleanResult.cleaned);
    
    if (validationResult.valid) {
      console.log('✅ Data is valid and ready to use!');
      return {
        success: true,
        processedData: cleanResult.cleaned,
        original: rawData
      };
    } else {
      console.log('❌ Data validation failed:', validationResult.errors);
      return {
        success: false,
        errors: validationResult.errors,
        original: rawData
      };
    }
  } catch (error) {
    console.log('💥 Processing failed:', error.message);
    return {
      success: false,
      error: error.message,
      original: rawData
    };
  }
}

// Test with different inputs
await processData("  Hello, A2A World!  ");
await processData("x");  // Too short
await processData("Hello123");  // Contains numbers
```

## Understanding What Happened

Congratulations! You've just created your first A2A agent. Here's what happened:

1. **Agent Creation**: You created an agent with a name, type, and capabilities
2. **Service Definition**: You added two services (`clean` and `validate`) that other agents can call
3. **Agent Startup**: You started the agent, making it available for communication
4. **Service Calls**: You tested the agent by calling its services directly
5. **Workflow**: You created a complete data processing workflow

## Key Concepts

### Agent Configuration
```javascript
const agent = new A2A.Agent({
  name: 'data-cleaner',           // Unique identifier
  type: 'data-processor',         // Agent category
  capabilities: ['text_cleaning'] // What the agent can do
});
```

### Services
Services are the core functionality of agents. They:
- Accept input data
- Process it according to their logic
- Return results or throw errors
- Can be called by other agents or users

### Error Handling
Always handle errors gracefully in your services:
```javascript
agent.addService('myService', async (data) => {
  try {
    // Your logic here
    return result;
  } catch (error) {
    throw new Error(`Service failed: ${error.message}`);
  }
});
```

## Next Steps

Now that you have a working agent, you can:

1. **Add More Services**: Extend your agent with additional functionality
2. **Connect to Registry**: Register your agent so others can discover it
3. **Create Multi-Agent Workflows**: Build systems where multiple agents work together
4. **Add Error Recovery**: Make your agent more robust with retry logic
5. **Deploy to Production**: Scale your agent for real-world use

## Try It Yourself

Experiment with the code above:
- Add a new service that counts words
- Modify the validation rules
- Create an agent that processes numbers instead of text
- Add logging to track what your agent is doing

## Troubleshooting

**Agent won't start?**
- Check that the agent name is unique
- Ensure all services are properly defined

**Service calls failing?**
- Verify the service name is correct
- Check that input data matches expected format
- Look for error messages in the console

**Need help?**
- Check the [API Reference](/docs/api-reference/agent-api)
- Join our [Community Discord](https://discord.gg/a2a)
- Open an issue on [GitHub](https://github.com/a2a-framework/core)

Ready for the next challenge? Learn about [Multi-Agent Communication](/docs/agent-development/communication) or explore [Workflow Orchestration](/docs/multi-agent-systems/workflows).