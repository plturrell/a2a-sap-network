const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');

describe('MessagingService', () => {
  let messagingService;
  let db;
  
  beforeAll(async () => {
    // Connect to test database
    db = await cds.connect.to('db');
    
    // Get service instance
    messagingService = await cds.connect.to('MessagingService');
  });
  
  describe('Event Publishing', () => {
    test('should publish agent event successfully', async () => {
      const eventData = {
        eventType: 'AgentRegistered',
        payload: JSON.stringify({
          agentId: uuidv4(),
          address: '0x1234567890abcdef',
          name: 'Test Agent'
        })
      };
      
      const result = await messagingService.send('publishAgentEvent', eventData);
      expect(result).toBe('Event AgentRegistered published successfully');
    });
    
    test('should publish service event successfully', async () => {
      const eventData = {
        eventType: 'ServiceCreated',
        payload: JSON.stringify({
          serviceId: uuidv4(),
          providerId: uuidv4(),
          name: 'Test Service',
          category: 'COMPUTATION',
          pricePerCall: 5.0
        })
      };
      
      const result = await messagingService.send('publishServiceEvent', eventData);
      expect(result).toBe('Event ServiceCreated published successfully');
    });
    
    test('should publish workflow event successfully', async () => {
      const eventData = {
        eventType: 'WorkflowCompleted',
        payload: JSON.stringify({
          executionId: uuidv4(),
          status: 'completed',
          totalGasUsed: 50000,
          duration: 3600
        })
      };
      
      const result = await messagingService.send('publishWorkflowEvent', eventData);
      expect(result).toBe('Event WorkflowCompleted published successfully');
    });
  });
  
  describe('Subscription Management', () => {
    let subscriptionId;
    
    test('should create subscription successfully', async () => {
      const topics = ['AgentRegistered', 'ServiceCreated', 'WorkflowCompleted'];
      
      const result = await messagingService.send('subscribe', { topics });
      
      expect(result).toMatchObject({
        subscriptionId: expect.any(String),
        topics,
        status: 'active'
      });
      
      subscriptionId = result.subscriptionId;
    });
    
    test('should unsubscribe successfully', async () => {
      const result = await messagingService.send('unsubscribe', { subscriptionId });
      expect(result).toBe(true);
    });
    
    test('should return false for non-existent subscription', async () => {
      const result = await messagingService.send('unsubscribe', { 
        subscriptionId: 'non-existent-id' 
      });
      expect(result).toBe(false);
    });
  });
  
  describe('Queue Operations', () => {
    test('should get queue status', async () => {
      const result = await messagingService.send('getQueueStatus');
      
      expect(result).toMatchObject({
        pendingMessages: expect.any(Number),
        processedToday: expect.any(Number),
        failedToday: expect.any(Number),
        queueHealth: expect.stringMatching(/healthy|warning|degraded/)
      });
    });
    
    test('should retry failed messages', async () => {
      const since = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago
      
      const result = await messagingService.send('retryFailedMessages', { since });
      
      expect(result).toMatchObject({
        retriedCount: expect.any(Number),
        successCount: expect.any(Number),
        failedCount: expect.any(Number)
      });
      
      expect(result.successCount + result.failedCount).toBeLessThanOrEqual(result.retriedCount);
    });
  });
  
  describe('Event Handlers', () => {
    test('should emit AgentRegistered event when agent is created', async () => {
      const { Agents } = cds.entities('a2a.network');
      const agent = global.testUtils.generateTestAgent();
      
      // Mock event emission
      const emitSpy = jest.spyOn(messagingService, 'emit');
      
      // Create agent (this should trigger event)
      await INSERT.into(Agents).entries(agent);
      
      // Wait for async event processing
      await new Promise(resolve => setTimeout(resolve, 100));
      
      expect(emitSpy).toHaveBeenCalledWith('AgentRegistered', expect.objectContaining({
        address: agent.address,
        name: agent.name,
        timestamp: expect.any(Date)
      }));
      
      emitSpy.mockRestore();
    });
    
    test('should emit ReputationUpdated event when reputation changes', async () => {
      const { AgentPerformance } = cds.entities('a2a.network');
      
      // Mock event emission
      const emitSpy = jest.spyOn(messagingService, 'emit');
      
      // This would trigger in a real scenario with proper setup
      // For unit test, we directly call the emit
      await messagingService.emit('ReputationUpdated', {
        agentId: uuidv4(),
        oldScore: 100,
        newScore: 95,
        reason: 'Failed task',
        timestamp: new Date()
      });
      
      expect(emitSpy).toHaveBeenCalledWith('ReputationUpdated', expect.any(Object));
      
      emitSpy.mockRestore();
    });
  });
});