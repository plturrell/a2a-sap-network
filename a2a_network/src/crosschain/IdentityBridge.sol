// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../AgentRegistry.sol";
import "../MultiSigPausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title IdentityBridge
 * @dev Handles identity translation and verification between A2A protocol
 * and external protocols (ANP/ACP). Manages DID resolution, agent card 
 * conversion, and cross-protocol identity verification.
 */
contract IdentityBridge is MultiSigPausable, ReentrancyGuard {
    using ECDSA for bytes32;

    AgentRegistry public immutable registry;

    struct DIDDocument {
        string did;
        address a2aAddress;
        string[] serviceEndpoints;
        bytes32[] verificationMethods;
        mapping(string => string) attributes;
        uint256 createdAt;
        uint256 updatedAt;
        bool active;
    }

    struct AgentCardANP {
        string context;
        string id;              // DID
        string agentType;
        string name;
        string description;
        string[] capabilities;
        string[] interfaces;
        AuthenticationMethod auth;
    }

    struct AgentDetailACP {
        string name;
        string description;
        string version;
        Operation[] operations;
        string[] supportedContentTypes;
        AuthenticationMethod authentication;
        string[] endpoints;
    }

    struct AuthenticationMethod {
        string scheme;      // "bearer", "basic", "mutual-tls"
        string[] parameters;
    }

    struct Operation {
        string name;
        string description;
        string inputSchema;
        string outputSchema;
        string method;      // HTTP method
    }

    // Storage
    mapping(address => DIDDocument) public didDocuments;
    mapping(string => address) public didToAddress;
    mapping(bytes32 => bool) public verifiedDIDs;
    
    // Protocol-specific identity mappings
    mapping(address => AgentCardANP) public anpAgentCards;
    mapping(address => AgentDetailACP) public acpAgentDetails;

    // Events
    event DIDCreated(address indexed agent, string did);
    event DIDUpdated(address indexed agent, string did);
    event IdentityVerified(address indexed agent, string did, bytes32 proof);
    event AgentCardConverted(address indexed agent, string protocol, string format);

    constructor(
        address _registry,
        uint256 _requiredConfirmations
    ) MultiSigPausable(_requiredConfirmations) {
        registry = AgentRegistry(_registry);
    }

    modifier onlyRegisteredAgent() {
        AgentRegistry.Agent memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not registered or inactive");
        _;
    }

    /**
     * @notice Create a DID document for an A2A agent
     * @param didUri The DID URI (e.g., "did:wba:agent123")
     * @param serviceEndpoints Array of service endpoints
     * @param verificationMethods Array of verification method identifiers
     */
    function createDIDDocument(
        string memory didUri,
        string[] memory serviceEndpoints,
        bytes32[] memory verificationMethods
    ) external onlyRegisteredAgent whenNotPaused nonReentrant {
        require(bytes(didUri).length > 0, "DID URI required");
        require(didToAddress[didUri] == address(0), "DID already exists");
        require(bytes(didDocuments[msg.sender].did).length == 0, "Agent already has DID");

        DIDDocument storage didDoc = didDocuments[msg.sender];
        didDoc.did = didUri;
        didDoc.a2aAddress = msg.sender;
        didDoc.serviceEndpoints = serviceEndpoints;
        didDoc.verificationMethods = verificationMethods;
        didDoc.createdAt = block.timestamp;
        didDoc.updatedAt = block.timestamp;
        didDoc.active = true;

        didToAddress[didUri] = msg.sender;
        emit DIDCreated(msg.sender, didUri);
    }

    /**
     * @notice Convert A2A Agent to ANP Agent Description Protocol format
     * @param agent A2A agent address
     * @return AgentCardANP structure in ANP format
     */
    function convertToANPAgentCard(address agent) 
        external 
        view 
        returns (AgentCardANP memory) 
    {
        AgentRegistry.Agent memory a2aAgent = registry.getAgent(agent);
        require(a2aAgent.active, "Agent not active");
        
        DIDDocument storage didDoc = didDocuments[agent];
        require(bytes(didDoc.did).length > 0, "No DID document found");

        AgentCardANP memory anpCard;
        anpCard.context = "https://www.w3.org/ns/did/v1,https://agent-network-protocol.com/context/v1";
        anpCard.id = didDoc.did;
        anpCard.agentType = "Agent";
        anpCard.name = a2aAgent.name;
        anpCard.description = string(abi.encodePacked("A2A Agent: ", a2aAgent.name));
        
        // Convert capabilities
        anpCard.capabilities = _convertCapabilitiesToANP(a2aAgent.capabilities);
        
        // Convert endpoints
        anpCard.interfaces = didDoc.serviceEndpoints;
        
        // Set authentication
        anpCard.auth.scheme = "bearer";
        anpCard.auth.parameters = new string[](1);
        anpCard.auth.parameters[0] = "ethereum-signature";

        return anpCard;
    }

    /**
     * @notice Convert A2A Agent to ACP Agent Detail format
     * @param agent A2A agent address
     * @return AgentDetailACP structure in ACP format
     */
    function convertToACPAgentDetail(address agent) 
        external 
        view 
        returns (AgentDetailACP memory) 
    {
        AgentRegistry.Agent memory a2aAgent = registry.getAgent(agent);
        require(a2aAgent.active, "Agent not active");

        AgentDetailACP memory acpDetail;
        acpDetail.name = a2aAgent.name;
        acpDetail.description = string(abi.encodePacked("A2A Agent: ", a2aAgent.name));
        acpDetail.version = "1.0.0";
        
        // Convert capabilities to operations
        acpDetail.operations = _convertCapabilitiesToOperations(a2aAgent.capabilities);
        
        // Set supported content types
        acpDetail.supportedContentTypes = new string[](3);
        acpDetail.supportedContentTypes[0] = "text/plain";
        acpDetail.supportedContentTypes[1] = "application/json";
        acpDetail.supportedContentTypes[2] = "multipart/mixed";
        
        // Set authentication
        acpDetail.authentication.scheme = "bearer";
        acpDetail.authentication.parameters = new string[](1);
        acpDetail.authentication.parameters[0] = "ethereum-address";
        
        // Set endpoints
        acpDetail.endpoints = new string[](1);
        acpDetail.endpoints[0] = a2aAgent.endpoint;

        return acpDetail;
    }

    /**
     * @notice Verify external agent identity using cryptographic proof
     * @param did The DID to verify
     * @param proof Cryptographic proof (signature)
     * @param message Original message that was signed
     */
    function verifyExternalIdentity(
        string memory did,
        bytes memory proof,
        bytes32 message
    ) external onlyRole(DEFAULT_ADMIN_ROLE) nonReentrant {
        require(bytes(did).length > 0, "DID required");
        address agent = didToAddress[did];
        require(agent != address(0), "DID not found");

        // Verify signature  
        bytes32 messageHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", message));
        address signer = ECDSA.recover(messageHash, proof);
        require(signer == agent, "Invalid signature");

        bytes32 verificationKey = keccak256(abi.encodePacked(did, proof));
        verifiedDIDs[verificationKey] = true;

        emit IdentityVerified(agent, did, verificationKey);
    }

    /**
     * @notice Update DID document attributes
     * @param key Attribute key
     * @param value Attribute value
     */
    function updateDIDAttribute(
        string memory key,
        string memory value
    ) external onlyRegisteredAgent whenNotPaused {
        DIDDocument storage didDoc = didDocuments[msg.sender];
        require(bytes(didDoc.did).length > 0, "No DID document found");

        didDoc.attributes[key] = value;
        didDoc.updatedAt = block.timestamp;

        emit DIDUpdated(msg.sender, didDoc.did);
    }

    /**
     * @notice Resolve DID to A2A agent address
     * @param did The DID to resolve
     * @return agent A2A agent address
     */
    function resolveDID(string memory did) external view returns (address agent) {
        return didToAddress[did];
    }

    /**
     * @notice Get DID document for an agent
     * @param agent A2A agent address
     * @return did DID URI
     * @return serviceEndpoints Array of service endpoints
     * @return createdAt Creation timestamp
     * @return active Whether the DID is active
     */
    function getDIDDocument(address agent) external view returns (
        string memory did,
        string[] memory serviceEndpoints,
        uint256 createdAt,
        bool active
    ) {
        DIDDocument storage didDoc = didDocuments[agent];
        return (didDoc.did, didDoc.serviceEndpoints, didDoc.createdAt, didDoc.active);
    }

    /**
     * @notice Check if a DID verification is valid
     * @param did The DID to check
     * @param proof The proof to verify
     * @return bool Whether the verification is valid
     */
    function isVerifiedDID(string memory did, bytes memory proof) external view returns (bool) {
        bytes32 verificationKey = keccak256(abi.encodePacked(did, proof));
        return verifiedDIDs[verificationKey];
    }

    /**
     * @dev Convert A2A capabilities to ANP format
     */
    function _convertCapabilitiesToANP(bytes32[] memory capabilities) 
        internal 
        pure 
        returns (string[] memory) 
    {
        string[] memory anpCapabilities = new string[](capabilities.length);
        for (uint256 i = 0; i < capabilities.length; i++) {
            anpCapabilities[i] = string(abi.encodePacked(
                '{"@type": "ad:Capability", "name": "', 
                _bytes32ToString(capabilities[i]), 
                '"}'
            ));
        }
        return anpCapabilities;
    }

    /**
     * @dev Convert A2A capabilities to ACP operations
     */
    function _convertCapabilitiesToOperations(bytes32[] memory capabilities) 
        internal 
        pure 
        returns (Operation[] memory) 
    {
        Operation[] memory operations = new Operation[](capabilities.length);
        for (uint256 i = 0; i < capabilities.length; i++) {
            operations[i] = Operation({
                name: _bytes32ToString(capabilities[i]),
                description: string(abi.encodePacked("A2A capability: ", _bytes32ToString(capabilities[i]))),
                inputSchema: "application/json",
                outputSchema: "application/json",
                method: "POST"
            });
        }
        return operations;
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
}