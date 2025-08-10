using a2a.network as network from '../db/schema';

/**
 * Blockchain Service for A2A Network
 * Handles all blockchain interactions
 */
@impl: './blockchain-service.js'
service BlockchainService {
    
    // Expose entities for UI tiles
    entity BlockchainStats {
        key ID: UUID;
        blockHeight: Integer;
        gasPrice: Decimal(10,2);
        networkStatus: String;
        totalTransactions: Integer;
        averageBlockTime: Decimal(8,2);
        timestamp: DateTime;
    }
    
    // Agent operations
    action registerAgent(agentId: String, address: String, name: String, endpoint: String) returns String;
    action updateAgentReputation(agentAddress: String, newScore: Integer) returns String;
    action deactivateAgent(agentAddress: String) returns String;
    
    // Service marketplace operations  
    action listService(serviceId: String, name: String, description: String, pricePerCall: Decimal, minReputation: Integer) returns String;
    action createServiceOrder(serviceId: String, consumer: String, amount: Decimal) returns String;
    action completeServiceOrder(orderId: String, rating: Integer) returns String;
    
    // Capability operations
    action registerCapability(name: String, description: String, category: Integer) returns String;
    action addAgentCapability(agentAddress: String, capabilityId: String) returns String;
    
    // Message routing
    action sendMessage(from: String, to: String, messageHash: String, protocol: String) returns String;
    action confirmMessageDelivery(messageHash: String) returns String;
    
    // Workflow operations
    action deployWorkflow(workflowDefinition: String) returns String;
    action executeWorkflow(workflowId: String, parameters: String) returns String;
    
    // Query functions
    function getAgentInfo(agentAddress: String) returns String;
    function getAgentReputation(agentAddress: String) returns Integer;
    function getServiceInfo(serviceId: String) returns String;
    function getNetworkStats() returns String;
    
    // Synchronization
    action syncBlockchain() returns {
        synced: Integer;
        pending: Integer;
        failed: Integer;
    };
    
    // Events emitted by blockchain
    event AgentRegistered {
        agentId: String;
        address: String;
        name: String;
        timestamp: DateTime;
    }
    
    event ServiceCreated {
        serviceId: String;
        providerId: String;
        name: String;
        price: Decimal(10,4);
    }
    
    event ReputationUpdated {
        agentId: String;
        oldScore: Integer;
        newScore: Integer;
        reason: String;
    }
    
    event TransactionCompleted {
        txHash: String;
        ![from]: String;
        ![to]: String;
        gasUsed: Integer;
    }
}