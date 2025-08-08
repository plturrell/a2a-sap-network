const cds = require('@sap/cds');
const { GET, POST, PATCH, DELETE, expect } = cds.test(__dirname + '/../..');

describe('A2A Service Integration Tests', () => {
  let agentId, serviceId, workflowId;
  
  beforeAll(async () => {
    // Deploy to test database
    await cds.deploy(__dirname + '/../../srv').to('sqlite::memory:');
  });
  
  describe('Agent Management', () => {
    test('POST /odata/v4/a2a/Agents - Create new agent', async () => {
      const agent = global.testUtils.generateTestAgent();
      
      const { status, data } = await POST('/odata/v4/a2a/Agents', agent);
      
      expect(status).toBe(201);
      expect(data).toMatchObject({
        ID: expect.any(String),
        address: agent.address,
        name: agent.name,
        reputation: 100,
        isActive: true
      });
      
      agentId = data.ID;
    });
    
    test('GET /odata/v4/a2a/Agents - List all agents', async () => {
      const { status, data } = await GET('/odata/v4/a2a/Agents');
      
      expect(status).toBe(200);
      expect(data.value).toBeInstanceOf(Array);
      expect(data.value.length).toBeGreaterThan(0);
      expect(data.value[0]).toMatchObject({
        ID: expect.any(String),
        address: expect.any(String),
        name: expect.any(String)
      });
    });
    
    test('GET /odata/v4/a2a/Agents(ID) - Get specific agent', async () => {
      const { status, data } = await GET(`/odata/v4/a2a/Agents(${agentId})`);
      
      expect(status).toBe(200);
      expect(data.ID).toBe(agentId);
    });
    
    test('PATCH /odata/v4/a2a/Agents(ID) - Update agent', async () => {
      const updates = { reputation: 95 };
      
      const { status, data } = await PATCH(`/odata/v4/a2a/Agents(${agentId})`, updates);
      
      expect(status).toBe(200);
      expect(data.reputation).toBe(95);
    });
    
    test('GET /odata/v4/a2a/TopAgents - View top agents', async () => {
      const { status, data } = await GET('/odata/v4/a2a/TopAgents');
      
      expect(status).toBe(200);
      expect(data.value).toBeInstanceOf(Array);
      // Should be sorted by reputation desc
      if (data.value.length > 1) {
        expect(data.value[0].reputation).toBeGreaterThanOrEqual(data.value[1].reputation);
      }
    });
  });
  
  describe('Service Marketplace', () => {
    test('POST /odata/v4/a2a/Services - Create new service', async () => {
      const service = global.testUtils.generateTestService(agentId);
      
      const { status, data } = await POST('/odata/v4/a2a/Services', service);
      
      expect(status).toBe(201);
      expect(data).toMatchObject({
        ID: expect.any(String),
        provider_ID: agentId,
        name: service.name,
        isActive: true
      });
      
      serviceId = data.ID;
    });
    
    test('GET /odata/v4/a2a/Services?$expand=provider - Get services with provider', async () => {
      const { status, data } = await GET('/odata/v4/a2a/Services?$expand=provider');
      
      expect(status).toBe(200);
      expect(data.value[0]).toMatchObject({
        ID: expect.any(String),
        provider: expect.objectContaining({
          ID: expect.any(String),
          name: expect.any(String)
        })
      });
    });
    
    test('POST /odata/v4/a2a/ServiceOrders - Create service order', async () => {
      const order = {
        service_ID: serviceId,
        consumer_ID: agentId,
        status: 'pending',
        callCount: 0,
        totalAmount: 0
      };
      
      const { status, data } = await POST('/odata/v4/a2a/ServiceOrders', order);
      
      expect(status).toBe(201);
      expect(data).toMatchObject({
        ID: expect.any(String),
        service_ID: serviceId,
        consumer_ID: agentId,
        status: 'pending'
      });
    });
  });
  
  describe('Workflow Management', () => {
    test('POST /odata/v4/a2a/Workflows - Create workflow', async () => {
      const workflow = global.testUtils.generateTestWorkflow(agentId);
      
      const { status, data } = await POST('/odata/v4/a2a/Workflows', workflow);
      
      expect(status).toBe(201);
      expect(data).toMatchObject({
        ID: expect.any(String),
        name: workflow.name,
        owner_ID: agentId,
        isActive: true
      });
      
      workflowId = data.ID;
    });
    
    test('POST /odata/v4/a2a/WorkflowExecutions - Execute workflow', async () => {
      const execution = {
        workflow_ID: workflowId,
        executionId: `exec-${Date.now()}`,
        status: 'running',
        startedAt: new Date()
      };
      
      const { status, data } = await POST('/odata/v4/a2a/WorkflowExecutions', execution);
      
      expect(status).toBe(201);
      expect(data).toMatchObject({
        ID: expect.any(String),
        workflow_ID: workflowId,
        status: 'running'
      });
    });
  });
  
  describe('Network Statistics', () => {
    test('GET /odata/v4/a2a/NetworkStats - Get network statistics', async () => {
      // First create some stats
      await POST('/odata/v4/a2a/NetworkStats', {
        totalAgents: 10,
        activeAgents: 8,
        totalServices: 5,
        totalCapabilities: 20,
        totalMessages: 100,
        totalTransactions: 50,
        averageReputation: 95.5,
        networkLoad: 0.65,
        gasPrice: 20,
        validFrom: new Date()
      });
      
      const { status, data } = await GET('/odata/v4/a2a/NetworkStats');
      
      expect(status).toBe(200);
      expect(data.value).toBeInstanceOf(Array);
      expect(data.value[0]).toMatchObject({
        totalAgents: expect.any(Number),
        activeAgents: expect.any(Number),
        networkLoad: expect.any(Number)
      });
    });
  });
  
  describe('Multi-tenancy', () => {
    test('GET /odata/v4/a2a/TenantSettings - Get tenant settings', async () => {
      // Create tenant settings
      await POST('/odata/v4/a2a/TenantSettings', {
        tenant: 'test-tenant',
        settings: {
          maxAgents: 1000,
          maxServices: 100,
          maxWorkflows: 50,
          features: {
            blockchain: true,
            ai: true,
            analytics: true
          }
        }
      });
      
      const { status, data } = await GET('/odata/v4/a2a/TenantSettings');
      
      expect(status).toBe(200);
      expect(data.value[0]).toMatchObject({
        tenant: 'test-tenant',
        settings: expect.objectContaining({
          maxAgents: 1000,
          features: expect.objectContaining({
            blockchain: true
          })
        })
      });
    });
  });
  
  describe('Error Handling', () => {
    test('GET /odata/v4/a2a/Agents(invalid-id) - Should return 404', async () => {
      const { status } = await GET('/odata/v4/a2a/Agents(00000000-0000-0000-0000-000000000000)');
      expect(status).toBe(404);
    });
    
    test('POST /odata/v4/a2a/Agents - Missing required field', async () => {
      const { status, data } = await POST('/odata/v4/a2a/Agents', {
        name: 'Missing Address Agent'
        // address is required but missing
      });
      
      expect(status).toBe(400);
      expect(data.error.message).toContain('address');
    });
  });
  
  describe('Cleanup', () => {
    test('DELETE /odata/v4/a2a/Agents(ID) - Delete agent', async () => {
      const { status } = await DELETE(`/odata/v4/a2a/Agents(${agentId})`);
      expect(status).toBe(204);
    });
  });
});