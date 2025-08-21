// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./CrossChainBridge.sol";

/**
 * @title L2AgentRegistry
 * @notice Layer 2 optimized agent registry with cross-chain synchronization
 * @dev Optimized for high throughput on L2s with mainnet synchronization
 */
contract L2AgentRegistry is AccessControl, ReentrancyGuard, Pausable {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant SYNC_OPERATOR_ROLE = keccak256("SYNC_OPERATOR_ROLE");

    CrossChainBridge public immutable crossChainBridge;
    uint256 public immutable mainnetChainId;

    // Optimized agent structure for L2
    struct L2Agent {
        address owner;
        bytes32 nameHash;           // Hash of agent name for privacy
        bytes32 endpointHash;       // Hash of endpoint URL
        bytes32[] capabilityHashes; // Hashed capabilities
        uint128 reputation;         // Reduced from uint256 for packing
        uint128 registeredAt;       // Block number instead of timestamp
        bool active;
        bool syncedToMainnet;       // Whether synced to mainnet
        uint64 mainnetSyncBlock;    // Last mainnet sync block
    }

    // Batch sync structure for efficiency
    struct SyncBatch {
        bytes32 batchId;
        L2Agent[] agents;
        uint256 timestamp;
        bool processed;
    }

    // State variables
    mapping(address => L2Agent) public l2Agents;
    mapping(bytes32 => address[]) public capabilityAgents; // capability -> agents
    mapping(bytes32 => SyncBatch) public syncBatches;
    
    address[] public allAgents;
    uint256 public activeAgentsCount;
    uint256 public totalSyncBatches;
    uint256 public constant BATCH_SIZE = 100; // Agents per sync batch
    uint256 public constant SYNC_INTERVAL = 1 hours; // Minimum sync interval
    uint256 public lastMainnetSync;

    // Events
    event L2AgentRegistered(
        address indexed agent,
        bytes32 nameHash,
        bytes32 indexed capabilityHash,
        uint128 reputation
    );
    event AgentSyncedToMainnet(address indexed agent, bytes32 indexed batchId);
    event BatchCreatedForSync(bytes32 indexed batchId, uint256 agentCount);
    event MainnetSyncCompleted(bytes32 indexed batchId, uint256 blockNumber);
    event ReputationUpdated(address indexed agent, uint128 newReputation);

    // Custom errors
    error AgentNotRegistered(address agent);
    error AgentAlreadyRegistered(address agent);
    error InvalidSyncBatch(bytes32 batchId);
    error SyncTooFrequent(uint256 lastSync, uint256 minInterval);
    error CrossChainBridgeError(string reason);

    constructor(
        address admin,
        address pauser,
        address payable crossChainBridge_,
        uint256 mainnetChainId_
    ) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PAUSER_ROLE, pauser);
        _grantRole(SYNC_OPERATOR_ROLE, admin);
        crossChainBridge = CrossChainBridge(crossChainBridge_);
        mainnetChainId = mainnetChainId_;
    }

    /**
     * @notice Register agent on L2 (low cost, high throughput)
     * @param nameHash Hash of agent name
     * @param endpointHash Hash of endpoint URL
     * @param capabilityHashes Array of capability hashes
     */
    function registerL2Agent(
        bytes32 nameHash,
        bytes32 endpointHash,
        bytes32[] calldata capabilityHashes
    ) external whenNotPaused nonReentrant {
        if (l2Agents[msg.sender].owner != address(0)) {
            revert AgentAlreadyRegistered(msg.sender);
        }
        require(nameHash != bytes32(0), "Invalid name hash");
        require(endpointHash != bytes32(0), "Invalid endpoint hash");
        require(capabilityHashes.length > 0, "No capabilities provided");

        l2Agents[msg.sender] = L2Agent({
            owner: msg.sender,
            nameHash: nameHash,
            endpointHash: endpointHash,
            capabilityHashes: capabilityHashes,
            reputation: 100, // Starting reputation
            registeredAt: uint128(block.number),
            active: true,
            syncedToMainnet: false,
            mainnetSyncBlock: 0
        });

        // Add to capability indexes
        for (uint256 i = 0; i < capabilityHashes.length; i++) {
            capabilityAgents[capabilityHashes[i]].push(msg.sender);
        }

        allAgents.push(msg.sender);
        activeAgentsCount++;

        emit L2AgentRegistered(
            msg.sender,
            nameHash,
            capabilityHashes[0], // Emit primary capability
            100
        );
    }

    /**
     * @notice Deactivate agent on L2
     */
    function deactivateL2Agent() external whenNotPaused {
        L2Agent storage agent = l2Agents[msg.sender];
        if (agent.owner == address(0)) revert AgentNotRegistered(msg.sender);
        require(agent.active, "Agent already inactive");

        agent.active = false;
        if (activeAgentsCount > 0) activeAgentsCount--;
    }

    /**
     * @notice Reactivate agent on L2
     */
    function reactivateL2Agent() external whenNotPaused {
        L2Agent storage agent = l2Agents[msg.sender];
        if (agent.owner == address(0)) revert AgentNotRegistered(msg.sender);
        require(!agent.active, "Agent already active");

        agent.active = true;
        activeAgentsCount++;
    }

    /**
     * @notice Find agents by capability hash (optimized for L2)
     * @param capabilityHash Hash of the capability to search for
     * @param offset Pagination offset
     * @param limit Maximum results
     * @return agents Array of agent addresses
     * @return total Total agents with this capability
     */
    function findL2AgentsByCapability(
        bytes32 capabilityHash,
        uint256 offset,
        uint256 limit
    ) external view returns (
        address[] memory agents,
        uint256 total
    ) {
        address[] storage capabilityList = capabilityAgents[capabilityHash];
        
        // Count active agents
        uint256 activeCount = 0;
        for (uint256 i = 0; i < capabilityList.length; i++) {
            if (l2Agents[capabilityList[i]].active) {
                activeCount++;
            }
        }
        
        total = activeCount;
        if (offset >= total) return (new address[](0), total);
        
        uint256 end = offset + limit;
        if (end > total) end = total;
        uint256 resultCount = end - offset;
        
        agents = new address[](resultCount);
        uint256 currentIndex = 0;
        uint256 resultIndex = 0;
        
        for (uint256 i = 0; i < capabilityList.length && resultIndex < resultCount; i++) {
            address agent = capabilityList[i];
            if (l2Agents[agent].active) {
                if (currentIndex >= offset) {
                    agents[resultIndex] = agent;
                    resultIndex++;
                }
                currentIndex++;
            }
        }
    }

    /**
     * @notice Create batch for mainnet synchronization
     * @param agentAddresses Array of agent addresses to sync
     * @return batchId Unique identifier for the sync batch
     */
    function createSyncBatch(
        address[] calldata agentAddresses
    ) external onlyRole(SYNC_OPERATOR_ROLE) returns (bytes32 batchId) {
        require(agentAddresses.length <= BATCH_SIZE, "Batch too large");
        require(
            block.timestamp >= lastMainnetSync + SYNC_INTERVAL,
            "Sync too frequent"
        );

        batchId = keccak256(abi.encodePacked(
            block.timestamp,
            totalSyncBatches,
            agentAddresses
        ));

        L2Agent[] memory agents = new L2Agent[](agentAddresses.length);
        for (uint256 i = 0; i < agentAddresses.length; i++) {
            address agentAddr = agentAddresses[i];
            L2Agent storage agent = l2Agents[agentAddr];
            if (agent.owner == address(0)) revert AgentNotRegistered(agentAddr);
            
            agents[i] = agent;
        }

        syncBatches[batchId] = SyncBatch({
            batchId: batchId,
            agents: agents,
            timestamp: block.timestamp,
            processed: false
        });

        totalSyncBatches++;
        emit BatchCreatedForSync(batchId, agentAddresses.length);
    }

    /**
     * @notice Sync batch to mainnet via cross-chain bridge
     * @param batchId ID of the batch to sync
     */
    function syncBatchToMainnet(
        bytes32 batchId
    ) external payable onlyRole(SYNC_OPERATOR_ROLE) whenNotPaused {
        SyncBatch storage batch = syncBatches[batchId];
        if (batch.timestamp == 0) revert InvalidSyncBatch(batchId);
        require(!batch.processed, "Batch already processed");

        // Encode batch data for cross-chain message
        bytes memory batchData = abi.encode(batchId, batch.agents, batch.timestamp);
        
        try crossChainBridge.sendCrossChainMessage{value: msg.value}(
            mainnetChainId,
            address(this), // Assuming mainnet has same contract address
            batchData
        ) returns (bytes32 messageHash) {
            batch.processed = true;
            lastMainnetSync = block.timestamp;
            
            // Mark agents as synced
            for (uint256 i = 0; i < batch.agents.length; i++) {
                L2Agent storage agent = l2Agents[batch.agents[i].owner];
                agent.syncedToMainnet = true;
                agent.mainnetSyncBlock = uint64(block.number);
                
                emit AgentSyncedToMainnet(batch.agents[i].owner, batchId);
            }
            
            emit MainnetSyncCompleted(batchId, block.number);
        } catch Error(string memory reason) {
            revert CrossChainBridgeError(reason);
        }
    }

    /**
     * @notice Process incoming sync from mainnet (called by cross-chain bridge)
     * @param batchData Encoded batch data from mainnet
     */
    function processMainnetSync(
        bytes calldata batchData
    ) external {
        require(msg.sender == address(crossChainBridge), "Only bridge can call");
        
        (bytes32 batchId, L2Agent[] memory agents, uint256 timestamp) = 
            abi.decode(batchData, (bytes32, L2Agent[], uint256));
        
        // Update local agents with mainnet data
        for (uint256 i = 0; i < agents.length; i++) {
            L2Agent storage localAgent = l2Agents[agents[i].owner];
            if (localAgent.owner != address(0)) {
                // Update reputation from mainnet
                localAgent.reputation = agents[i].reputation;
                emit ReputationUpdated(agents[i].owner, agents[i].reputation);
            }
        }
    }

    /**
     * @notice Update agent reputation (local L2 only)
     * @param agent Agent address
     * @param newReputation New reputation value
     */
    function updateReputationL2(
        address agent,
        uint128 newReputation
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        L2Agent storage agentData = l2Agents[agent];
        if (agentData.owner == address(0)) revert AgentNotRegistered(agent);
        
        agentData.reputation = newReputation;
        // Mark as not synced since reputation changed
        agentData.syncedToMainnet = false;
        
        emit ReputationUpdated(agent, newReputation);
    }

    /**
     * @notice Get L2 agent information
     * @param agent Agent address
     * @return L2 agent data
     */
    function getL2Agent(address agent) external view returns (L2Agent memory) {
        L2Agent storage agentData = l2Agents[agent];
        if (agentData.owner == address(0)) revert AgentNotRegistered(agent);
        return agentData;
    }

    /**
     * @notice Get agents pending mainnet sync
     * @param limit Maximum results
     * @return pendingAgents Array of agent addresses
     * @return count Number of pending agents
     */
    function getPendingSyncAgents(
        uint256 limit
    ) external view returns (
        address[] memory pendingAgents,
        uint256 count
    ) {
        // Count agents not synced to mainnet
        uint256 pendingCount = 0;
        for (uint256 i = 0; i < allAgents.length; i++) {
            if (!l2Agents[allAgents[i]].syncedToMainnet && l2Agents[allAgents[i]].active) {
                pendingCount++;
            }
        }
        
        if (pendingCount == 0) return (new address[](0), 0);
        
        uint256 resultCount = pendingCount > limit ? limit : pendingCount;
        pendingAgents = new address[](resultCount);
        
        uint256 resultIndex = 0;
        for (uint256 i = 0; i < allAgents.length && resultIndex < resultCount; i++) {
            if (!l2Agents[allAgents[i]].syncedToMainnet && l2Agents[allAgents[i]].active) {
                pendingAgents[resultIndex] = allAgents[i];
                resultIndex++;
            }
        }
        
        count = pendingCount;
    }

    /**
     * @notice Get total statistics
     * @return totalAgents Total registered agents
     * @return activeAgents Currently active agents
     * @return syncedAgents Agents synced to mainnet
     * @return pendingSync Agents pending sync
     */
    function getL2Statistics() external view returns (
        uint256 totalAgents,
        uint256 activeAgents,
        uint256 syncedAgents,
        uint256 pendingSync
    ) {
        totalAgents = allAgents.length;
        activeAgents = activeAgentsCount;
        
        uint256 synced = 0;
        uint256 pending = 0;
        
        for (uint256 i = 0; i < allAgents.length; i++) {
            L2Agent storage agent = l2Agents[allAgents[i]];
            if (agent.active) {
                if (agent.syncedToMainnet) {
                    synced++;
                } else {
                    pending++;
                }
            }
        }
        
        syncedAgents = synced;
        pendingSync = pending;
    }

    /**
     * @notice Pause contract
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause contract
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
}