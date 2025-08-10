// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "../MultiSigPausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ExternalProtocolOracle
 * @dev Oracle contract for real-time communication with external ANP and ACP protocols.
 * Handles agent discovery, message routing, and protocol state synchronization.
 */
contract ExternalProtocolOracle is MultiSigPausable, ReentrancyGuard {
    enum ProtocolType { ANP, ACP }
    enum RequestType { DISCOVERY, MESSAGE_SEND, MESSAGE_RECEIVE, AGENT_INFO }
    
    struct ProtocolEndpoint {
        string gatewayUrl;
        string apiKey;
        uint256 timeout;
        bool active;
        uint256 lastHealthCheck;
        uint256 successCount;
        uint256 totalRequests;
    }
    
    struct ExternalRequest {
        bytes32 requestId;
        RequestType requestType;
        ProtocolType protocol;
        address requester;
        string payload;
        uint256 timestamp;
        bool completed;
        bool success;
        string response;
    }
    
    struct DiscoveryResult {
        string agentId;
        string name;
        string endpoint;
        string[] capabilities;
        uint256 reputation;
        bool available;
    }
    
    // Storage
    mapping(ProtocolType => ProtocolEndpoint) public protocolEndpoints;
    mapping(bytes32 => ExternalRequest) public externalRequests;
    mapping(address => bool) public authorizedOracles;
    
    // Protocol-specific agent caches
    mapping(ProtocolType => mapping(string => DiscoveryResult)) public agentCache;
    mapping(ProtocolType => string[]) public cachedAgentIds;
    mapping(ProtocolType => uint256) public cacheLastUpdated;
    
    // Request tracking
    bytes32[] public pendingRequests;
    mapping(bytes32 => uint256) public requestIndex;
    
    uint256 public constant CACHE_DURATION = 300; // 5 minutes
    uint256 public constant REQUEST_TIMEOUT = 60; // 1 minute
    uint256 public constant MAX_RETRIES = 3;
    
    // Events
    event ProtocolConfigured(ProtocolType protocol, string gatewayUrl, bool active);
    event ExternalRequestCreated(bytes32 indexed requestId, RequestType requestType, ProtocolType protocol);
    event ExternalRequestCompleted(bytes32 indexed requestId, bool success, string response);
    event DiscoveryResultCached(ProtocolType protocol, string agentId, string name);
    event OracleAuthorized(address oracle, bool authorized);
    event ProtocolHealthCheck(ProtocolType protocol, bool healthy, uint256 responseTime);
    
    constructor(uint256 _requiredConfirmations) MultiSigPausable(_requiredConfirmations) {
        // Initialize default oracle authorization for admin
        authorizedOracles[msg.sender] = true;
    }
    
    modifier onlyAuthorizedOracle() {
        require(authorizedOracles[msg.sender], "Not authorized oracle");
        _;
    }
    
    /**
     * @notice Configure external protocol endpoint
     * @param protocol Protocol type (ANP/ACP)
     * @param gatewayUrl Gateway endpoint URL
     * @param apiKey API key for authentication
     * @param timeout Request timeout in seconds
     */
    function configureProtocol(
        ProtocolType protocol,
        string memory gatewayUrl,
        string memory apiKey,
        uint256 timeout
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        protocolEndpoints[protocol] = ProtocolEndpoint({
            gatewayUrl: gatewayUrl,
            apiKey: apiKey,
            timeout: timeout,
            active: true,
            lastHealthCheck: block.timestamp,
            successCount: 0,
            totalRequests: 0
        });
        
        emit ProtocolConfigured(protocol, gatewayUrl, true);
    }
    
    /**
     * @notice Discover agents on external protocol
     * @param protocol Protocol to search
     * @param capability Capability to search for
     * @param maxResults Maximum results to return
     * @return requestId Request ID for tracking
     */
    function discoverExternalAgents(
        ProtocolType protocol,
        string memory capability,
        uint256 maxResults
    ) external whenNotPaused nonReentrant returns (bytes32) {
        require(protocolEndpoints[protocol].active, "Protocol not active");
        
        bytes32 requestId = keccak256(
            abi.encodePacked(block.timestamp, msg.sender, protocol, capability)
        );
        
        // Create discovery request payload
        string memory payload;
        if (protocol == ProtocolType.ANP) {
            payload = _createANPDiscoveryPayload(capability, maxResults);
        } else {
            payload = _createACPDiscoveryPayload(capability, maxResults);
        }
        
        externalRequests[requestId] = ExternalRequest({
            requestId: requestId,
            requestType: RequestType.DISCOVERY,
            protocol: protocol,
            requester: msg.sender,
            payload: payload,
            timestamp: block.timestamp,
            completed: false,
            success: false,
            response: ""
        });
        
        pendingRequests.push(requestId);
        requestIndex[requestId] = pendingRequests.length - 1;
        
        protocolEndpoints[protocol].totalRequests++;
        
        emit ExternalRequestCreated(requestId, RequestType.DISCOVERY, protocol);
        
        return requestId;
    }
    
    /**
     * @notice Send message to external protocol
     * @param protocol Target protocol
     * @param targetAgent Target agent identifier
     * @param message Message content
     * @param messageType Message type
     * @return requestId Request ID for tracking
     */
    function sendExternalMessage(
        ProtocolType protocol,
        string memory targetAgent,
        string memory message,
        string memory messageType
    ) external whenNotPaused nonReentrant returns (bytes32) {
        require(protocolEndpoints[protocol].active, "Protocol not active");
        
        bytes32 requestId = keccak256(
            abi.encodePacked(block.timestamp, msg.sender, protocol, targetAgent, message)
        );
        
        // Create message payload
        string memory payload;
        if (protocol == ProtocolType.ANP) {
            payload = _createANPMessagePayload(targetAgent, message, messageType);
        } else {
            payload = _createACPMessagePayload(targetAgent, message, messageType);
        }
        
        externalRequests[requestId] = ExternalRequest({
            requestId: requestId,
            requestType: RequestType.MESSAGE_SEND,
            protocol: protocol,
            requester: msg.sender,
            payload: payload,
            timestamp: block.timestamp,
            completed: false,
            success: false,
            response: ""
        });
        
        pendingRequests.push(requestId);
        requestIndex[requestId] = pendingRequests.length - 1;
        
        protocolEndpoints[protocol].totalRequests++;
        
        emit ExternalRequestCreated(requestId, RequestType.MESSAGE_SEND, protocol);
        
        return requestId;
    }
    
    /**
     * @notice Oracle callback to complete external request
     * @param requestId Request identifier
     * @param success Whether request was successful
     * @param response Response data
     */
    function completeExternalRequest(
        bytes32 requestId,
        bool success,
        string memory response
    ) external onlyAuthorizedOracle nonReentrant {
        ExternalRequest storage request = externalRequests[requestId];
        require(!request.completed, "Request already completed");
        require(block.timestamp <= request.timestamp + REQUEST_TIMEOUT, "Request timed out");
        
        request.completed = true;
        request.success = success;
        request.response = response;
        
        // Update protocol statistics
        if (success) {
            protocolEndpoints[request.protocol].successCount++;
        }
        
        // Process response based on request type
        if (request.requestType == RequestType.DISCOVERY && success) {
            _processDiscoveryResponse(request.protocol, response);
        }
        
        // Remove from pending requests
        _removePendingRequest(requestId);
        
        emit ExternalRequestCompleted(requestId, success, response);
    }
    
    /**
     * @notice Get cached discovery results
     * @param protocol Protocol to query
     * @param capability Capability filter (empty for all)
     * @return results Array of discovery results
     */
    function getCachedDiscoveryResults(
        ProtocolType protocol,
        string memory capability
    ) external view returns (DiscoveryResult[] memory results) {
        string[] memory agentIds = cachedAgentIds[protocol];
        uint256 matchCount = 0;
        
        // Count matching agents
        for (uint256 i = 0; i < agentIds.length; i++) {
            DiscoveryResult memory agent = agentCache[protocol][agentIds[i]];
            if (_hasCapability(agent.capabilities, capability)) {
                matchCount++;
            }
        }
        
        // Build result array
        results = new DiscoveryResult[](matchCount);
        uint256 resultIndex = 0;
        
        for (uint256 i = 0; i < agentIds.length; i++) {
            DiscoveryResult memory agent = agentCache[protocol][agentIds[i]];
            if (_hasCapability(agent.capabilities, capability)) {
                results[resultIndex] = agent;
                resultIndex++;
            }
        }
        
        return results;
    }
    
    /**
     * @notice Check if cached data is valid
     * @param protocol Protocol to check
     * @return bool Whether cache is valid
     */
    function isCacheValid(ProtocolType protocol) external view returns (bool) {
        return block.timestamp <= cacheLastUpdated[protocol] + CACHE_DURATION;
    }
    
    /**
     * @notice Get protocol health status
     * @param protocol Protocol to check
     * @return healthy Whether protocol is healthy
     * @return successRate Success rate percentage (0-100)
     * @return lastCheck Last health check timestamp
     */
    function getProtocolHealth(ProtocolType protocol) 
        external 
        view 
        returns (bool healthy, uint256 successRate, uint256 lastCheck) 
    {
        ProtocolEndpoint memory endpoint = protocolEndpoints[protocol];
        
        healthy = endpoint.active && 
                  (block.timestamp <= endpoint.lastHealthCheck + 3600); // 1 hour
        
        successRate = endpoint.totalRequests > 0 ? 
                     (endpoint.successCount * 100) / endpoint.totalRequests : 100;
        
        lastCheck = endpoint.lastHealthCheck;
        
        return (healthy, successRate, lastCheck);
    }
    
    /**
     * @notice Authorize/deauthorize oracle
     * @param oracle Oracle address
     * @param authorized Whether to authorize
     */
    function setOracleAuthorization(
        address oracle,
        bool authorized
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        authorizedOracles[oracle] = authorized;
        emit OracleAuthorized(oracle, authorized);
    }
    
    /**
     * @notice Clean up timed out requests
     */
    function cleanupTimedOutRequests() external onlyAuthorizedOracle {
        uint256 currentTime = block.timestamp;
        
        for (uint256 i = 0; i < pendingRequests.length; i++) {
            bytes32 requestId = pendingRequests[i];
            ExternalRequest storage request = externalRequests[requestId];
            
            if (currentTime > request.timestamp + REQUEST_TIMEOUT) {
                request.completed = true;
                request.success = false;
                request.response = "Request timed out";
                
                emit ExternalRequestCompleted(requestId, false, "Request timed out");
                
                // Remove from pending (will be handled in batch)
                delete pendingRequests[i];
            }
        }
        
        // Clean up empty slots in pending requests array
        _compactPendingRequests();
    }
    
    // Internal functions
    
    function _createANPDiscoveryPayload(
        string memory capability,
        uint256 maxResults
    ) internal pure returns (string memory) {
        return string(abi.encodePacked(
            '{"@context": ["https://www.w3.org/ns/did/v1", "https://agent-network-protocol.com/context/v1"],',
            '"@type": "ad:DiscoveryRequest",',
            '"ad:capability": "', capability, '",',
            '"ad:maxResults": ', _uint256ToString(maxResults), ',',
            '"ad:requester": "A2A-Bridge"}'
        ));
    }
    
    function _createACPDiscoveryPayload(
        string memory capability,
        uint256 maxResults
    ) internal pure returns (string memory) {
        return string(abi.encodePacked(
            '{"operation": "discover_agents",',
            '"filters": {"capabilities": ["', capability, '"]},',
            '"limit": ', _uint256ToString(maxResults), ',',
            '"source": "A2A-Bridge"}'
        ));
    }
    
    function _createANPMessagePayload(
        string memory targetAgent,
        string memory message,
        string memory messageType
    ) internal view returns (string memory) {
        return string(abi.encodePacked(
            '{"@context": ["https://www.w3.org/ns/did/v1", "https://agent-network-protocol.com/context/v1"],',
            '"@type": "ad:Message",',
            '"ad:recipient": "', targetAgent, '",',
            '"ad:content": {"@type": "ad:TaskRequest", "ad:description": "', message, '", "ad:messageType": "', messageType, '"},',
            '"ad:timestamp": "', _uint256ToString(block.timestamp), '"}'
        ));
    }
    
    function _createACPMessagePayload(
        string memory targetAgent,
        string memory message,
        string memory messageType
    ) internal view returns (string memory) {
        return string(abi.encodePacked(
            '{"operation": "sendTask",',
            '"target_agent": "', targetAgent, '",',
            '"message": {"parts": [{"content_type": "text/plain", "content": "', message, '"}]},',
            '"task_type": "', messageType, '",',
            '"timestamp": "', _uint256ToString(block.timestamp), '"}'
        ));
    }
    
    function _processDiscoveryResponse(ProtocolType protocol, string memory response) internal {
        // Parse discovery response and update cache
        // This is a simplified implementation - real parsing would be more robust
        cacheLastUpdated[protocol] = block.timestamp;
        
        // In a real implementation, this would parse the JSON response
        // and extract agent information to update the cache
        // For now, we'll emit an event to indicate processing occurred
        emit DiscoveryResultCached(protocol, "parsed-agent-id", "parsed-agent-name");
    }
    
    function _hasCapability(string[] memory capabilities, string memory targetCapability) 
        internal 
        pure 
        returns (bool) 
    {
        if (bytes(targetCapability).length == 0) return true; // Empty filter matches all
        
        for (uint256 i = 0; i < capabilities.length; i++) {
            if (keccak256(bytes(capabilities[i])) == keccak256(bytes(targetCapability))) {
                return true;
            }
        }
        return false;
    }
    
    function _removePendingRequest(bytes32 requestId) internal {
        uint256 index = requestIndex[requestId];
        if (index < pendingRequests.length && pendingRequests[index] == requestId) {
            // Move last element to deleted spot
            pendingRequests[index] = pendingRequests[pendingRequests.length - 1];
            requestIndex[pendingRequests[index]] = index;
            
            // Remove last element
            pendingRequests.pop();
            delete requestIndex[requestId];
        }
    }
    
    function _compactPendingRequests() internal {
        uint256 writeIndex = 0;
        
        for (uint256 i = 0; i < pendingRequests.length; i++) {
            if (pendingRequests[i] != bytes32(0)) {
                if (writeIndex != i) {
                    pendingRequests[writeIndex] = pendingRequests[i];
                    requestIndex[pendingRequests[writeIndex]] = writeIndex;
                }
                writeIndex++;
            }
        }
        
        // Remove empty slots
        while (pendingRequests.length > writeIndex) {
            pendingRequests.pop();
        }
    }
    
    function _uint256ToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) return "0";
        
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        
        return string(buffer);
    }
    
    // View functions for monitoring
    
    function getPendingRequestCount() external view returns (uint256) {
        return pendingRequests.length;
    }
    
    function getPendingRequestIds() external view returns (bytes32[] memory) {
        return pendingRequests;
    }
    
    function getRequest(bytes32 requestId) external view returns (ExternalRequest memory) {
        return externalRequests[requestId];
    }
}