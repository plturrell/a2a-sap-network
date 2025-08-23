# A2A Framework for Financial Data Standardization Agents

## Understanding the A2A protocol for semantic standardization

The Agent-to-Agent (A2A) protocol is an open standard developed by Google and donated to the Linux Foundation, designed to enable AI agents built on different frameworks to communicate and collaborate as peers. As of August 2025, it's backed by over 50 technology partners including Atlassian, MongoDB, Salesforce, and PayPal. The protocol provides a standardized HTTP-based communication framework that's particularly well-suited for building data standardization agents that serve as the first step in larger processing pipelines.

A2A operates on the principle of "opaque execution" - agents collaborate based on declared capabilities without exposing their internal memory, tools, or proprietary logic. This makes it ideal for integrating existing JavaScript standardization frameworks while maintaining encapsulation and security.

## Core A2A specifications and agent definitions

### Agent Card structure

The Agent Card is the fundamental building block of A2A agent definitions. It's a JSON document that serves as a machine-readable "business card" declaring an agent's capabilities:

```json
{
  "name": "Financial Data Standardizer",
  "description": "Standardizes financial entities including locations, accounts, products, books, and measures",
  "url": "https://api.example.com/a2a/v1",
  "version": "1.0.0",
  "protocolVersion": "0.2.9",
  "provider": {
    "organization": "Your Organization",
    "url": "https://yourorg.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "defaultInputModes": ["text/plain", "text/csv", "application/json"],
  "defaultOutputModes": ["application/json", "text/csv"],
  "skills": [
    {
      "id": "location-standardization",
      "name": "Location Standardization",
      "description": "Normalize and standardize location data using semantic matching",
      "tags": ["standardization", "location", "entity-recognition"],
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["application/json"],
      "examples": [
        "Standardize 'NYC' to 'New York City, NY, USA'",
        "Convert location references to standard format"
      ]
    },
    {
      "id": "account-standardization",
      "name": "Account Standardization",
      "description": "Standardize financial account references and types",
      "tags": ["standardization", "financial", "accounts"],
      "inputModes": ["text/csv", "application/json"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "type": "http",
      "scheme": "bearer",
      "bearerFormat": "JWT"
    }
  },
  "security": [{"bearer": []}]
}
```

### Communication protocols

A2A supports three transport protocols (agents must implement at least one):

1. **JSON-RPC 2.0 over HTTP(S)** - Primary protocol using standard JSON-RPC request/response
2. **REST-style HTTP+JSON** - RESTful endpoints with appropriate HTTP verbs
3. **gRPC** - Protocol Buffers v3 with TLS encryption required

All protocols support Server-Sent Events (SSE) for streaming real-time updates during long-running standardization tasks.

## Integrating JavaScript standardization frameworks

### Creating an A2A standardization agent

Here's how to wrap your existing JavaScript standardizers within an A2A-compliant agent using the official JavaScript SDK:

```typescript
import { 
  AgentExecutor, 
  RequestContext, 
  ExecutionEventBus,
  TaskStatusUpdateEvent,
  TaskArtifactUpdateEvent,
  A2AExpressApp,
  DefaultRequestHandler,
  InMemoryTaskStore
} from "@a2a-js/sdk";
import { v4 as uuidv4 } from "uuid";

// Import your existing standardizers
import { LocationStandardizer } from "./standardizers/LocationStandardizer";
import { AccountStandardizer } from "./standardizers/AccountStandardizer";
import { ProductStandardizer } from "./standardizers/ProductStandardizer";
import { BookStandardizer } from "./standardizers/BookStandardizer";
import { MeasureStandardizer } from "./standardizers/MeasureStandardizer";
import { NewsSearchStandardizer } from "./standardizers/NewsSearchStandardizer";

class FinancialStandardizationExecutor implements AgentExecutor {
  private standardizers: Map<string, any>;
  private cancelledTasks = new Set<string>();

  constructor() {
    // Initialize your existing standardizers
    this.standardizers = new Map([
      ["location", new LocationStandardizer()],
      ["account", new AccountStandardizer()],
      ["product", new ProductStandardizer()],
      ["book", new BookStandardizer()],
      ["measure", new MeasureStandardizer()],
      ["news", new NewsSearchStandardizer()]
    ]);
  }

  async cancelTask(taskId: string, eventBus: ExecutionEventBus): Promise<void> {
    this.cancelledTasks.add(taskId);
  }

  async execute(
    requestContext: RequestContext,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    const { userMessage, taskId, contextId } = requestContext;
    
    try {
      // 1. Update task status to working
      this.publishStatusUpdate(eventBus, taskId, contextId, "working", 
        "Analyzing standardization request...");

      // 2. Extract data and determine standardization type
      const { data, type } = this.extractDataFromMessage(userMessage);
      
      // 3. Check for cancellation
      if (this.cancelledTasks.has(taskId)) {
        this.publishStatusUpdate(eventBus, taskId, contextId, "canceled");
        eventBus.finished();
        return;
      }

      // 4. Process with appropriate standardizer
      const standardizer = this.standardizers.get(type);
      if (!standardizer) {
        throw new Error(`Unsupported standardization type: ${type}`);
      }

      // Stream progress updates for large datasets
      const progressCallback = (progress: number) => {
        this.publishStatusUpdate(eventBus, taskId, contextId, "working",
          `Processing: ${Math.round(progress * 100)}% complete`);
      };

      const standardizedData = await standardizer.standardize(data, {
        onProgress: progressCallback
      });

      // 5. Publish results as artifact
      const artifactUpdate: TaskArtifactUpdateEvent = {
        kind: "artifact-update",
        taskId,
        contextId,
        artifact: {
          artifactId: `standardized-${type}-${uuidv4()}`,
          name: `Standardized ${type} Data`,
          description: `${type} data standardized with entity recognition`,
          parts: [{
            kind: "data",
            data: {
              type: type,
              original_count: Array.isArray(data) ? data.length : 1,
              standardized_count: standardizedData.length,
              results: standardizedData,
              metadata: {
                standardizer_version: standardizer.version,
                processed_at: new Date().toISOString(),
                recognition_confidence: standardizedData.map(item => item.confidence)
              }
            }
          }]
        },
        append: false,
        lastChunk: true
      };
      eventBus.publish(artifactUpdate);

      // 6. Complete task
      this.publishStatusUpdate(eventBus, taskId, contextId, "completed",
        `Successfully standardized ${standardizedData.length} ${type} entities`);
      
    } catch (error) {
      this.publishErrorStatus(eventBus, taskId, contextId, error);
    } finally {
      eventBus.finished();
    }
  }

  private extractDataFromMessage(message: any): { data: any, type: string } {
    // Handle different input formats
    const parts = message.parts || [];
    
    for (const part of parts) {
      if (part.kind === "text") {
        // Parse text for standardization commands
        const match = part.text.match(/standardize\s+(\w+):\s*(.+)/i);
        if (match) {
          return {
            type: match[1].toLowerCase(),
            data: this.parseTextData(match[2])
          };
        }
      } else if (part.kind === "file" && part.file.mimeType === "text/csv") {
        // Handle CSV file uploads
        const csvData = Buffer.from(part.file.bytes, 'base64').toString();
        return {
          type: this.detectTypeFromCSV(csvData),
          data: this.parseCSV(csvData)
        };
      } else if (part.kind === "data") {
        // Handle structured data
        return {
          type: part.data.type || this.detectTypeFromData(part.data),
          data: part.data.items || part.data
        };
      }
    }
    
    throw new Error("No valid standardization data found in message");
  }

  private publishStatusUpdate(
    eventBus: ExecutionEventBus,
    taskId: string,
    contextId: string,
    state: string,
    message?: string
  ): void {
    const update: TaskStatusUpdateEvent = {
      kind: "status-update",
      taskId,
      contextId,
      status: {
        state,
        timestamp: new Date().toISOString(),
        ...(message && {
          message: {
            kind: "message",
            role: "agent",
            messageId: uuidv4(),
            parts: [{ kind: "text", text: message }],
            taskId,
            contextId
          }
        })
      },
      final: ["completed", "failed", "canceled"].includes(state)
    };
    eventBus.publish(update);
  }
}
```

### Server setup and deployment

```typescript
import express from 'express';
import { agentCard } from './agentCard';

// Create the A2A server
const taskStore = new InMemoryTaskStore();
const agentExecutor = new FinancialStandardizationExecutor();
const requestHandler = new DefaultRequestHandler(
  agentCard,
  taskStore,
  agentExecutor
);

const appBuilder = new A2AExpressApp(requestHandler);
const app = appBuilder.setupRoutes(express(), "");

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Financial Standardization A2A Agent running on port ${PORT}`);
  console.log(`Agent Card: http://localhost:${PORT}/.well-known/agent.json`);
});
```

## Handling CSV data and structured transformations

### Input/output specifications for CSV processing

A2A agents can handle CSV data through multiple mechanisms:

```typescript
// Agent skill for CSV processing
{
  "id": "csv-batch-standardization",
  "name": "CSV Batch Standardization",
  "description": "Process CSV files containing financial entities for bulk standardization",
  "inputModes": ["text/csv", "application/vnd.ms-excel"],
  "outputModes": ["application/json", "text/csv"],
  "examples": [
    "Upload customer locations CSV for standardization",
    "Process account list with entity recognition"
  ]
}
```

### Processing CSV data with streaming

```typescript
private async processCSVStream(
  csvStream: ReadableStream,
  standardizer: any,
  eventBus: ExecutionEventBus,
  taskId: string
): Promise<any[]> {
  const results = [];
  let rowCount = 0;
  let processedCount = 0;

  const parser = csv.parse({
    columns: true,
    skip_empty_lines: true
  });

  parser.on('readable', async () => {
    let record;
    while (record = parser.read()) {
      rowCount++;
      
      // Process in batches for efficiency
      const standardized = await standardizer.standardizeRecord(record);
      results.push(standardized);
      processedCount++;

      // Stream progress updates
      if (processedCount % 100 === 0) {
        const progress = processedCount / rowCount;
        this.publishProgressUpdate(eventBus, taskId, progress);
      }
    }
  });

  return new Promise((resolve, reject) => {
    parser.on('end', () => resolve(results));
    parser.on('error', reject);
    csvStream.pipe(parser);
  });
}
```

## Multi-step workflow implementation

### Standardization as first step in processing pipeline

A2A excels at orchestrating multi-step workflows where standardization is the initial processing stage:

```typescript
// Workflow orchestrator agent
class DataProcessingOrchestrator implements AgentExecutor {
  private a2aClient: A2AClient;

  async execute(requestContext: RequestContext, eventBus: ExecutionEventBus) {
    const { taskId, contextId, userMessage } = requestContext;
    
    // Step 1: Standardization
    const standardizationResponse = await this.delegateToAgent(
      "https://standardizer.example.com",
      userMessage,
      contextId
    );

    // Step 2: Validation (using standardized data)
    const validationResponse = await this.delegateToAgent(
      "https://validator.example.com",
      {
        kind: "message",
        role: "user",
        parts: [{
          kind: "data",
          data: standardizationResponse.artifacts[0].parts[0].data
        }]
      },
      contextId
    );

    // Step 3: Enrichment
    const enrichmentResponse = await this.delegateToAgent(
      "https://enrichment.example.com",
      {
        kind: "message", 
        role: "user",
        parts: [{
          kind: "data",
          data: validationResponse.artifacts[0].parts[0].data
        }]
      },
      contextId
    );

    // Publish final aggregated results
    this.publishAggregatedResults(eventBus, taskId, contextId, {
      standardization: standardizationResponse,
      validation: validationResponse,
      enrichment: enrichmentResponse
    });
  }

  private async delegateToAgent(
    agentUrl: string,
    message: any,
    contextId: string
  ): Promise<any> {
    const client = new A2AClient(agentUrl);
    const response = await client.sendMessage({
      message,
      contextId,
      configuration: {
        blocking: true,
        acceptedOutputModes: ["application/json"]
      }
    });
    
    if (response.error) {
      throw new Error(`Agent error: ${response.error.message}`);
    }
    
    return response.result;
  }
}
```

### Context management across agents

```typescript
// Maintain context across standardization steps
const workflowContext = {
  contextId: "financial-processing-" + uuidv4(),
  metadata: {
    workflow: "customer-data-pipeline",
    initiated_by: "batch-processor",
    original_format: "csv",
    target_schema: "unified-customer-v2"
  }
};

// Pass context through entire pipeline
const standardizedData = await standardizationAgent.process(data, workflowContext);
const validatedData = await validationAgent.process(standardizedData, workflowContext);
const storedData = await storageAgent.process(validatedData, workflowContext);
```

## Semantic standardization patterns in A2A

### Entity recognition and semantic processing

```typescript
class SemanticStandardizationAgent {
  private entityRecognizer: EntityRecognizer;
  private semanticMatcher: SemanticMatcher;

  async standardizeWithSemantics(input: string): Promise<StandardizedEntity> {
    // 1. Extract entities
    const entities = await this.entityRecognizer.extract(input);
    
    // 2. Semantic matching against knowledge base
    const matches = await this.semanticMatcher.findBestMatches(entities, {
      threshold: 0.85,
      context: "financial-entities"
    });
    
    // 3. Apply business rules
    const standardized = this.applyStandardizationRules(matches);
    
    // 4. Return with confidence scores
    return {
      original: input,
      standardized: standardized,
      entities: entities,
      confidence: matches.map(m => m.confidence),
      metadata: {
        recognizer_version: this.entityRecognizer.version,
        rules_applied: standardized.rulesApplied
      }
    };
  }
}
```

### Best practices for semantic agents

1. **Declare semantic capabilities explicitly**:
```json
{
  "skills": [{
    "id": "semantic-entity-recognition",
    "name": "Semantic Entity Recognition",
    "description": "Extract and normalize entities using NLP and knowledge graphs",
    "tags": ["nlp", "semantic", "entity-recognition", "knowledge-graph"],
    "metadata": {
      "supported_languages": ["en", "es", "fr"],
      "entity_types": ["organization", "location", "financial_instrument", "person"],
      "knowledge_sources": ["wikidata", "geonames", "gleif"]
    }
  }]
}
```

2. **Implement confidence scoring**:
```typescript
interface SemanticResult {
  entity: string;
  standardized_form: string;
  confidence: number;
  alternative_matches: Array<{
    form: string;
    confidence: number;
  }>;
  semantic_context: {
    category: string;
    relationships: string[];
  };
}
```

3. **Support incremental refinement**:
```typescript
// Allow agents to request clarification for ambiguous entities
if (result.confidence < 0.7) {
  const clarificationUpdate: TaskStatusUpdateEvent = {
    kind: "status-update",
    taskId,
    status: {
      state: "input-required",
      message: {
        role: "agent",
        parts: [{
          kind: "text",
          text: `Multiple possible matches for "${entity}". Please specify:`,
        }, {
          kind: "data",
          data: {
            options: result.alternative_matches
          }
        }]
      }
    }
  };
  eventBus.publish(clarificationUpdate);
}
```

## Production deployment considerations

### Security implementation

```typescript
// Implement authentication middleware
app.use('/a2a/*', async (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      jsonrpc: "2.0",
      error: {
        code: -32001,
        message: "Authentication required"
      }
    });
  }
  
  try {
    const token = authHeader.substring(7);
    const decoded = await verifyJWT(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(403).json({
      jsonrpc: "2.0",
      error: {
        code: -32002,
        message: "Invalid authentication token"
      }
    });
  }
});
```

### Error handling and validation

```typescript
class RobustStandardizationExecutor extends FinancialStandardizationExecutor {
  async execute(requestContext: RequestContext, eventBus: ExecutionEventBus) {
    const { taskId, contextId } = requestContext;
    
    try {
      // Input validation
      this.validateRequest(requestContext);
      
      // Process with timeout
      const timeout = new Promise((_, reject) => 
        setTimeout(() => reject(new Error("Processing timeout")), 300000)
      );
      
      await Promise.race([
        super.execute(requestContext, eventBus),
        timeout
      ]);
      
    } catch (error) {
      // Structured error reporting
      const errorCode = this.mapErrorToCode(error);
      const errorUpdate: TaskStatusUpdateEvent = {
        kind: "status-update",
        taskId,
        contextId,
        status: {
          state: "failed",
          error: {
            code: errorCode,
            message: error.message,
            details: {
              type: error.constructor.name,
              stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
            }
          },
          timestamp: new Date().toISOString()
        },
        final: true
      };
      eventBus.publish(errorUpdate);
      eventBus.finished();
    }
  }

  private mapErrorToCode(error: Error): string {
    if (error.message.includes("validation")) return "VALIDATION_ERROR";
    if (error.message.includes("timeout")) return "TIMEOUT_ERROR";
    if (error.message.includes("standardization")) return "STANDARDIZATION_ERROR";
    return "INTERNAL_ERROR";
  }
}
```

### Performance optimization

```typescript
// Implement caching for frequently standardized entities
class CachedStandardizer {
  private cache: LRUCache<string, StandardizedEntity>;
  private standardizer: BaseStandardizer;

  constructor(standardizer: BaseStandardizer, cacheSize: number = 10000) {
    this.standardizer = standardizer;
    this.cache = new LRUCache({ max: cacheSize });
  }

  async standardize(input: string): Promise<StandardizedEntity> {
    const cacheKey = this.generateCacheKey(input);
    
    // Check cache first
    const cached = this.cache.get(cacheKey);
    if (cached) {
      return { ...cached, metadata: { ...cached.metadata, cache_hit: true } };
    }

    // Standardize and cache
    const result = await this.standardizer.standardize(input);
    this.cache.set(cacheKey, result);
    
    return result;
  }
}
```

## Testing and development tools

### Using the A2A Inspector

The A2A Inspector is essential for testing standardization agents:

```bash
# Run with Docker
docker run -d -p 8080:8080 a2a-inspector

# Or clone and run locally
git clone https://github.com/a2aproject/a2a-inspector.git
cd a2a-inspector
uv sync
cd frontend && npm install && npm run build -- --watch &
cd ../backend && uv run app.py
```

### Integration testing

```typescript
import { A2AClient } from "@a2a-js/sdk";

describe("Financial Standardization Agent", () => {
  let client: A2AClient;
  
  beforeAll(() => {
    client = new A2AClient("http://localhost:3000");
  });

  test("should standardize location entities", async () => {
    const response = await client.sendMessage({
      message: {
        messageId: "test-001",
        role: "user",
        parts: [{
          kind: "text",
          text: "standardize location: NYC, New York, United States"
        }]
      }
    });

    expect(response.result.status.state).toBe("completed");
    expect(response.result.artifacts[0].parts[0].data.results[0])
      .toMatchObject({
        standardized: "New York City, NY, USA",
        confidence: expect.any(Number)
      });
  });

  test("should handle CSV batch processing", async () => {
    const csvData = `name,location
    Apple Inc,Cupertino CA
    Microsoft,Redmond Washington
    Google,Mountain View`;

    const response = await client.sendMessage({
      message: {
        messageId: "test-002",
        role: "user",
        parts: [{
          kind: "file",
          file: {
            name: "companies.csv",
            mimeType: "text/csv",
            bytes: Buffer.from(csvData).toString('base64')
          }
        }]
      }
    });

    expect(response.result.artifacts[0].parts[0].data.standardized_count).toBe(3);
  });
});
```

## Key implementation takeaways

Building A2A-compliant standardization agents requires careful attention to several critical aspects. **Agent Card design** must accurately reflect your standardizer's capabilities with clear skill definitions and supported data formats. **Error handling** should be comprehensive, with meaningful error codes and graceful degradation for partial failures. **Performance considerations** include implementing caching for frequently standardized entities and streaming for large datasets. **Security** must be built-in from the start, with proper authentication and input validation.

The A2A framework's strength lies in its ability to create composable, interoperable agents that can work together in complex workflows. By wrapping your existing JavaScript standardization classes within the A2A protocol, you create powerful building blocks that can be discovered, invoked, and orchestrated by other agents in the ecosystem. This approach enables you to leverage your existing standardization logic while gaining the benefits of standardized communication, discovery, and workflow orchestration that A2A provides.