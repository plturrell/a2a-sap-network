// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../AgentRegistry.sol";
import "../MessageRouter.sol";
import "../MultiSigPausable.sol";
import "./MessageTranslator.sol";
import "./ExternalProtocolOracle.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title ProtocolBridge
 * @dev Bridge contract for integrating A2A protocol with ANP (Agent Network Protocol) 
 * and ACP (Agent Communication Protocol). Provides identity translation, message 
 * format conversion, and cross-protocol discovery mechanisms.
 */
contract ProtocolBridge is MultiSigPausable, ReentrancyGuard {
    using MessageTranslator for string;
    
    AgentRegistry public immutable registry;
    MessageRouter public immutable router;
    ExternalProtocolOracle public immutable oracle;

    enum ProtocolType { A2A, ANP, ACP }

    struct ExternalAgent {
        string did;              // DID for ANP integration
        string endpoint;         // External protocol endpoint
        ProtocolType protocol;   // Source protocol type
        mapping(bytes32 => string) capabilities; // Protocol-specific capabilities
        uint256 lastSyncTime;
        bool active;
    }

    struct CrossProtocolMessage {
        bytes32 messageId;
        address a2aAgent;        // Internal A2A agent
        string externalAgent;    // External agent identifier
        ProtocolType targetProtocol;
        string originalFormat;   // Original message format
        string translatedContent; // Translated to target protocol
        uint256 timestamp;
        bool processed;
    }

    // Mappings for cross-protocol functionality
    mapping(address => ExternalAgent) public externalAgents;
    mapping(string => address) public didToAddress;
    mapping(bytes32 => CrossProtocolMessage) public bridgeMessages;
    
    // Protocol-specific endpoints and configurations
    mapping(ProtocolType => string) public protocolEndpoints;
    mapping(ProtocolType => bool) public protocolEnabled;

    // Events
    event ExternalAgentRegistered(address indexed a2aAgent, string did, ProtocolType protocol);
    event MessageBridged(bytes32 indexed messageId, ProtocolType fromProtocol, ProtocolType toProtocol);
    event ProtocolConfigUpdated(ProtocolType protocol, string endpoint, bool enabled);
    event DiscoveryCrossProtocol(string query, ProtocolType[] protocols, uint256 resultCount);

    constructor(
        address _registry, 
        address _router,
        address _oracle, 
        uint256 _requiredConfirmations
    ) MultiSigPausable(_requiredConfirmations) {
        registry = AgentRegistry(_registry);
        router = MessageRouter(_router);
        oracle = ExternalProtocolOracle(_oracle);
        
        // Initialize protocol configurations
        protocolEnabled[ProtocolType.A2A] = true;
        protocolEnabled[ProtocolType.ANP] = false;
        protocolEnabled[ProtocolType.ACP] = false;
    }

    modifier onlyRegisteredAgent() {
        AgentRegistry.Agent memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not registered or inactive");
        _;
    }

    /**
     * @notice Register an external agent identity for cross-protocol bridging
     * @param did Decentralized identifier for ANP integration
     * @param endpoint External protocol endpoint
     * @param protocol Target protocol type
     * @param capabilities Array of capability identifiers in external protocol format
     */
    function registerExternalAgent(
        string memory did,
        string memory endpoint,
        ProtocolType protocol,
        bytes32[] memory capabilities
    ) external onlyRegisteredAgent whenNotPaused nonReentrant {
        require(bytes(did).length > 0, "DID required");
        require(bytes(endpoint).length > 0, "Endpoint required");
        require(protocol != ProtocolType.A2A, "Cannot register A2A as external");
        require(protocolEnabled[protocol], "Protocol not enabled");

        ExternalAgent storage extAgent = externalAgents[msg.sender];
        extAgent.did = did;
        extAgent.endpoint = endpoint;
        extAgent.protocol = protocol;
        extAgent.lastSyncTime = block.timestamp;
        extAgent.active = true;

        // Store capabilities
        for (uint256 i = 0; i < capabilities.length; i++) {
            extAgent.capabilities[capabilities[i]] = string(abi.encodePacked(capabilities[i]));
        }

        didToAddress[did] = msg.sender;
        emit ExternalAgentRegistered(msg.sender, did, protocol);
    }

    /**
     * @notice Bridge a message from A2A to external protocol (ANP/ACP)
     * @param messageId Original A2A message ID
     * @param targetProtocol Destination protocol
     * @param externalAgentId External agent identifier
     * @return bridgeMessageId Unique identifier for the bridged message
     */
    function bridgeMessageToExternal(
        bytes32 messageId,
        ProtocolType targetProtocol,
        string memory externalAgentId
    ) external onlyRegisteredAgent whenNotPaused nonReentrant returns (bytes32) {
        require(protocolEnabled[targetProtocol], "Target protocol not enabled");
        
        MessageRouter.Message memory originalMessage = router.getMessage(messageId);
        require(originalMessage.to == msg.sender, "Not message recipient");

        bytes32 bridgeMessageId = keccak256(
            abi.encodePacked(messageId, targetProtocol, externalAgentId, block.timestamp)
        );

        CrossProtocolMessage storage bridgeMsg = bridgeMessages[bridgeMessageId];
        bridgeMsg.messageId = bridgeMessageId;
        bridgeMsg.a2aAgent = msg.sender;
        bridgeMsg.externalAgent = externalAgentId;
        bridgeMsg.targetProtocol = targetProtocol;
        bridgeMsg.originalFormat = originalMessage.content;
        // Get external agent info for DID resolution
        ExternalAgent storage extAgent = externalAgents[msg.sender];
        
        if (targetProtocol == ProtocolType.ANP) {
            bridgeMsg.translatedContent = MessageTranslator.translateToANP(
                originalMessage.content,
                originalMessage.messageType,
                extAgent.did,
                externalAgentId
            );
        } else if (targetProtocol == ProtocolType.ACP) {
            bridgeMsg.translatedContent = MessageTranslator.translateToACP(
                originalMessage.content,
                originalMessage.messageType,
                string(abi.encodePacked("session_", block.timestamp))
            );
        } else {
            bridgeMsg.translatedContent = originalMessage.content;
        }
        bridgeMsg.timestamp = block.timestamp;
        bridgeMsg.processed = false;

        emit MessageBridged(bridgeMessageId, ProtocolType.A2A, targetProtocol);
        return bridgeMessageId;
    }

    /**
     * @notice Bridge a message from external protocol to A2A
     * @param externalMessageId External protocol message identifier
     * @param fromProtocol Source protocol type
     * @param targetA2AAgent A2A agent address
     * @param content Message content in external format
     * @param messageType Message type identifier
     * @return a2aMessageId A2A message ID
     */
    function bridgeMessageFromExternal(
        string memory externalMessageId,
        ProtocolType fromProtocol,
        address targetA2AAgent,
        string memory content,
        bytes32 messageType
    ) external onlyRole(DEFAULT_ADMIN_ROLE) whenNotPaused nonReentrant returns (bytes32) {
        require(protocolEnabled[fromProtocol], "Source protocol not enabled");
        require(fromProtocol != ProtocolType.A2A, "Cannot bridge from A2A to A2A");

        AgentRegistry.Agent memory targetAgent = registry.getAgent(targetA2AAgent);
        require(targetAgent.active, "Target agent not active");

        // Translate external protocol message to A2A format
        string memory translatedContent;
        if (fromProtocol == ProtocolType.ANP) {
            (translatedContent, , ) = MessageTranslator.parseFromANP(content);
        } else if (fromProtocol == ProtocolType.ACP) {
            (translatedContent, , ) = MessageTranslator.parseFromACP(content);
        } else {
            translatedContent = content;
        }

        // Send message through A2A MessageRouter via admin role (bridge acts as intermediary)
        bytes32 a2aMessageId = router.sendMessage(targetA2AAgent, translatedContent, messageType);

        bytes32 bridgeMessageId = keccak256(
            abi.encodePacked(externalMessageId, fromProtocol, targetA2AAgent, block.timestamp)
        );

        CrossProtocolMessage storage bridgeMsg = bridgeMessages[bridgeMessageId];
        bridgeMsg.messageId = bridgeMessageId;
        bridgeMsg.a2aAgent = targetA2AAgent;
        bridgeMsg.externalAgent = externalMessageId;
        bridgeMsg.targetProtocol = ProtocolType.A2A;
        bridgeMsg.originalFormat = content;
        bridgeMsg.translatedContent = translatedContent;
        bridgeMsg.timestamp = block.timestamp;
        bridgeMsg.processed = true;

        emit MessageBridged(bridgeMessageId, fromProtocol, ProtocolType.A2A);
        return a2aMessageId;
    }

    /**
     * @notice Discover agents across multiple protocols
     * @param capabilityQuery Capability search query
     * @param protocols Array of protocols to search
     * @param limit Maximum results per protocol
     * @return a2aAgents Array of A2A agent addresses
     * @return externalAgentIds Array of external agent identifiers
     */
    function crossProtocolDiscovery(
        bytes32 capabilityQuery,
        ProtocolType[] memory protocols,
        uint256 limit
    ) external view returns (
        address[] memory a2aAgents,
        string[] memory externalAgentIds
    ) {
        // Find A2A agents with the capability
        address[] memory a2aResults = registry.findAgentsByCapability(capabilityQuery);
        
        // Limit results
        uint256 a2aCount = a2aResults.length > limit ? limit : a2aResults.length;
        a2aAgents = new address[](a2aCount);
        for (uint256 i = 0; i < a2aCount; i++) {
            a2aAgents[i] = a2aResults[i];
        }

        // Discover external agents through oracle
        uint256 totalExternalCount = 0;
        
        for (uint256 i = 0; i < protocols.length; i++) {
            if (protocols[i] != ProtocolType.A2A && protocolEnabled[protocols[i]]) {
                ExternalProtocolOracle.ProtocolType oracleProtocol = 
                    protocols[i] == ProtocolType.ANP ? 
                    ExternalProtocolOracle.ProtocolType.ANP : 
                    ExternalProtocolOracle.ProtocolType.ACP;
                
                // Get cached results if available and valid
                if (oracle.isCacheValid(oracleProtocol)) {
                    ExternalProtocolOracle.DiscoveryResult[] memory cachedResults = 
                        oracle.getCachedDiscoveryResults(oracleProtocol, _bytes32ToString(capabilityQuery));
                    
                    for (uint256 j = 0; j < cachedResults.length && totalExternalCount < limit; j++) {
                        if (cachedResults[j].available) {
                            totalExternalCount++;
                        }
                    }
                } else {
                    // Would initiate new discovery request (async) - for view function, skip
                    // oracle.discoverExternalAgents(oracleProtocol, _bytes32ToString(capabilityQuery), limit);
                }
            }
        }

        // Populate external agent results from cache
        externalAgentIds = new string[](totalExternalCount);
        uint256 externalIndex = 0;
        
        for (uint256 i = 0; i < protocols.length && externalIndex < totalExternalCount; i++) {
            if (protocols[i] != ProtocolType.A2A && protocolEnabled[protocols[i]]) {
                ExternalProtocolOracle.ProtocolType oracleProtocol = 
                    protocols[i] == ProtocolType.ANP ? 
                    ExternalProtocolOracle.ProtocolType.ANP : 
                    ExternalProtocolOracle.ProtocolType.ACP;
                
                if (oracle.isCacheValid(oracleProtocol)) {
                    ExternalProtocolOracle.DiscoveryResult[] memory cachedResults = 
                        oracle.getCachedDiscoveryResults(oracleProtocol, _bytes32ToString(capabilityQuery));
                    
                    for (uint256 j = 0; j < cachedResults.length && externalIndex < totalExternalCount; j++) {
                        if (cachedResults[j].available) {
                            externalAgentIds[externalIndex] = cachedResults[j].agentId;
                            externalIndex++;
                        }
                    }
                }
            }
        }
        
        return (a2aAgents, externalAgentIds);
    }

    /**
     * @notice Configure protocol endpoints and enable/disable protocols
     * @param protocol Protocol type to configure
     * @param endpoint Protocol endpoint URL
     * @param enabled Whether the protocol is enabled
     */
    function configureProtocol(
        ProtocolType protocol,
        string memory endpoint,
        bool enabled
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        protocolEndpoints[protocol] = endpoint;
        protocolEnabled[protocol] = enabled;
        emit ProtocolConfigUpdated(protocol, endpoint, enabled);
    }

    /**
     * @notice Get external agent information
     * @param a2aAgent A2A agent address
     * @return did Decentralized identifier
     * @return endpoint External endpoint
     * @return protocol Protocol type
     * @return active Whether the external agent is active
     */
    function getExternalAgent(address a2aAgent) external view returns (
        string memory did,
        string memory endpoint,
        ProtocolType protocol,
        bool active
    ) {
        ExternalAgent storage extAgent = externalAgents[a2aAgent];
        return (extAgent.did, extAgent.endpoint, extAgent.protocol, extAgent.active);
    }


    /**
     * @dev Convert bytes32 to string
     */
    function _bytes32ToString(bytes32 _bytes32) internal pure returns (string memory) {
        uint8 i = 0;
        while (i < 32 && _bytes32[i] != 0) {
            i++;
        }
        bytes memory bytesArray = new bytes(i);
        for (i = 0; i < 32 && _bytes32[i] != 0; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }

    /**
     * @dev Convert uint256 to string
     */
    function _uint256ToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
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
}