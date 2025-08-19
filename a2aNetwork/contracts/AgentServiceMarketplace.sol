// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "./interfaces/IAgentRegistry.sol";

/**
 * @title AgentServiceMarketplace
 * @notice Enables agents to offer and purchase services from each other
 * @dev Implements service listing, bidding, execution, and payment
 */
contract AgentServiceMarketplace is 
    AccessControlUpgradeable, 
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable 
{
    bytes32 public constant DISPUTE_RESOLVER_ROLE = keccak256("DISPUTE_RESOLVER_ROLE");
    
    IAgentRegistry public agentRegistry;
    
    enum ServiceStatus { Listed, InProgress, Completed, Disputed, Cancelled }
    enum ServiceType { OneTime, Subscription, OnDemand }
    
    // GAS OPTIMIZATION: Pack struct to minimize storage slots
    struct Service {
        address provider;           // slot 1: 20 bytes
        uint128 basePrice;         // slot 1: 16 bytes (total 36 bytes, 28 bytes padding)
        
        uint64 minReputation;      // slot 2: 8 bytes
        uint32 maxConcurrent;      // slot 2: 4 bytes  
        uint32 currentActive;      // slot 2: 4 bytes
        ServiceType serviceType;   // slot 2: 1 byte (assuming enum fits in 1 byte)
        bool active;              // slot 2: 1 byte (total 18 bytes, 14 bytes padding)
        
        string name;              // slot 3+: dynamic
        string description;       // dynamic
        string[] capabilities;    // dynamic
    }
    
    // GAS OPTIMIZATION: Pack ServiceRequest struct
    struct ServiceRequest {
        uint256 serviceId;         // slot 1: 32 bytes
        address requester;         // slot 2: 20 bytes
        address provider;          // slot 3: 20 bytes
        uint128 agreedPrice;       // slot 4: 16 bytes
        uint128 escrowAmount;      // slot 4: 16 bytes (total 32 bytes)
        uint64 deadline;          // slot 5: 8 bytes
        ServiceStatus status;      // slot 5: 1 byte (total 9 bytes, 23 bytes padding)
        bytes parameters;         // slot 6+: dynamic
        bytes result;            // dynamic
    }
    
    struct Bid {
        address agent;
        uint256 price;
        uint256 estimatedTime;
        string proposal;
    }
    
    mapping(uint256 => Service) public services;
    mapping(uint256 => ServiceRequest) public requests;
    mapping(uint256 => Bid[]) public serviceBids;
    mapping(address => uint256[]) public agentServices;
    mapping(address => uint256[]) public agentRequests;
    mapping(address => uint256) public agentEarnings;
    mapping(address => uint256) public agentSpending;
    
    uint256 public serviceCounter;
    uint256 public requestCounter;
    uint256 public platformFeePercent = 250; // 2.5%
    uint256 public disputeWindow = 3 days;
    
    event ServiceListed(uint256 indexed serviceId, address indexed provider, string name);
    event ServiceRequested(uint256 indexed requestId, uint256 indexed serviceId, address indexed requester);
    event BidSubmitted(uint256 indexed serviceId, address indexed bidder, uint256 price);
    event ServiceStarted(uint256 indexed requestId, address indexed provider);
    event ServiceCompleted(uint256 indexed requestId, bytes result);
    event PaymentReleased(uint256 indexed requestId, uint256 amount);
    event DisputeRaised(uint256 indexed requestId, address indexed disputer);
    
    function initialize(address _agentRegistry) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(DISPUTE_RESOLVER_ROLE, msg.sender);
        
        agentRegistry = IAgentRegistry(_agentRegistry);
    }
    
    /**
     * @notice List a new service offering
     */
    function listService(
        string memory name,
        string memory description,
        string[] memory capabilities,
        uint256 basePrice,
        ServiceType serviceType,
        uint256 minReputation,
        uint256 maxConcurrent
    ) external returns (uint256) {
        require(agentRegistry.isRegistered(msg.sender), "Not registered agent");
        
        uint256 serviceId = ++serviceCounter;
        
        services[serviceId] = Service({
            provider: msg.sender,
            name: name,
            description: description,
            capabilities: capabilities,
            basePrice: basePrice,
            serviceType: serviceType,
            minReputation: minReputation,
            maxConcurrent: maxConcurrent,
            currentActive: 0,
            active: true
        });
        
        agentServices[msg.sender].push(serviceId);
        emit ServiceListed(serviceId, msg.sender, name);
        
        return serviceId;
    }
    
    /**
     * @notice Request a service with direct selection
     */
    function requestService(
        uint256 serviceId,
        bytes memory parameters,
        uint256 deadline
    ) external payable returns (uint256) {
        Service storage service = services[serviceId];
        require(service.active, "Service not active");
        require(msg.value >= service.basePrice, "Insufficient payment");
        require(agentRegistry.getReputation(msg.sender) >= service.minReputation, "Insufficient reputation");
        require(service.currentActive < service.maxConcurrent, "Service at capacity");
        
        uint256 requestId = ++requestCounter;
        
        requests[requestId] = ServiceRequest({
            serviceId: serviceId,
            requester: msg.sender,
            provider: service.provider,
            agreedPrice: msg.value,
            deadline: deadline,
            status: ServiceStatus.Listed,
            parameters: parameters,
            result: "",
            escrowAmount: msg.value
        });
        
        service.currentActive++;
        agentRequests[msg.sender].push(requestId);
        
        emit ServiceRequested(requestId, serviceId, msg.sender);
        return requestId;
    }
    
    /**
     * @notice Submit bid for an open service request
     */
    function submitBid(
        uint256 serviceId,
        uint256 price,
        uint256 estimatedTime,
        string memory proposal
    ) external {
        require(agentRegistry.isRegistered(msg.sender), "Not registered agent");
        Service memory service = services[serviceId];
        require(service.active, "Service not active");
        
        serviceBids[serviceId].push(Bid({
            agent: msg.sender,
            price: price,
            estimatedTime: estimatedTime,
            proposal: proposal
        }));
        
        emit BidSubmitted(serviceId, msg.sender, price);
    }
    
    /**
     * @notice Start working on a service request
     */
    function startService(uint256 requestId) external {
        ServiceRequest storage request = requests[requestId];
        require(request.provider == msg.sender, "Not the provider");
        require(request.status == ServiceStatus.Listed, "Invalid status");
        
        request.status = ServiceStatus.InProgress;
        emit ServiceStarted(requestId, msg.sender);
    }
    
    /**
     * @notice Submit service completion
     */
    function completeService(uint256 requestId, bytes memory result) external {
        ServiceRequest storage request = requests[requestId];
        require(request.provider == msg.sender, "Not the provider");
        require(request.status == ServiceStatus.InProgress, "Not in progress");
        require(block.timestamp <= request.deadline, "Past deadline");
        
        request.status = ServiceStatus.Completed;
        request.result = result;
        
        emit ServiceCompleted(requestId, result);
    }
    
    /**
     * @notice Release payment after service completion
     */
    function releasePayment(uint256 requestId) external nonReentrant {
        ServiceRequest storage request = requests[requestId];
        require(request.requester == msg.sender, "Not the requester");
        require(request.status == ServiceStatus.Completed, "Not completed");
        require(request.escrowAmount > 0, "No escrow");
        
        uint256 platformFee = (request.escrowAmount * platformFeePercent) / 10000;
        uint256 providerPayment = request.escrowAmount - platformFee;
        
        // SECURITY FIX: Update state BEFORE external calls to prevent reentrancy
        uint256 escrowToRelease = request.escrowAmount;
        request.escrowAmount = 0;
        services[request.serviceId].currentActive--;
        
        agentEarnings[request.provider] += providerPayment;
        agentSpending[request.requester] += request.agreedPrice;
        
        // SECURITY FIX: Use call instead of transfer for better error handling
        (bool success, ) = payable(request.provider).call{value: providerPayment}("");
        require(success, "Payment transfer failed");
        
        emit PaymentReleased(requestId, providerPayment);
    }
    
    /**
     * @notice Raise a dispute
     */
    function raiseDispute(uint256 requestId) external {
        ServiceRequest storage request = requests[requestId];
        require(
            request.requester == msg.sender || request.provider == msg.sender,
            "Not a party"
        );
        require(request.status == ServiceStatus.Completed, "Not completed");
        require(block.timestamp <= request.deadline + disputeWindow, "Dispute window closed");
        
        request.status = ServiceStatus.Disputed;
        emit DisputeRaised(requestId, msg.sender);
    }
    
    /**
     * @notice Resolve a dispute (admin only)
     */
    function resolveDispute(
        uint256 requestId,
        uint256 providerPercent
    ) external onlyRole(DISPUTE_RESOLVER_ROLE) {
        ServiceRequest storage request = requests[requestId];
        require(request.status == ServiceStatus.Disputed, "Not disputed");
        require(providerPercent <= 100, "Invalid percentage");
        
        uint256 providerAmount = (request.escrowAmount * providerPercent) / 100;
        uint256 requesterRefund = request.escrowAmount - providerAmount;
        
        // SECURITY FIX: Update state BEFORE external calls
        uint256 totalEscrow = request.escrowAmount;
        request.escrowAmount = 0;
        request.status = ServiceStatus.Completed;
        services[request.serviceId].currentActive--;
        
        if (providerAmount > 0) {
            agentEarnings[request.provider] += providerAmount;
            (bool successProvider, ) = payable(request.provider).call{value: providerAmount}("");
            require(successProvider, "Provider payment failed");
        }
        
        if (requesterRefund > 0) {
            (bool successRequester, ) = payable(request.requester).call{value: requesterRefund}("");
            require(successRequester, "Requester refund failed");
        }
    }
    
    /**
     * @notice Get agent's service statistics
     */
    function getAgentStats(address agent) external view returns (
        uint256 servicesProvided,
        uint256 servicesRequested,
        uint256 totalEarnings,
        uint256 totalSpending,
        uint256 completionRate
    ) {
        servicesProvided = agentServices[agent].length;
        servicesRequested = agentRequests[agent].length;
        totalEarnings = agentEarnings[agent];
        totalSpending = agentSpending[agent];
        
        uint256 completed;
        for (uint256 i = 0; i < agentRequests[agent].length; i++) {
            if (requests[agentRequests[agent][i]].status == ServiceStatus.Completed) {
                completed++;
            }
        }
        
        completionRate = servicesRequested > 0 ? (completed * 100) / servicesRequested : 0;
    }
    
    // SECURITY ENHANCEMENT: Advanced upgrade authorization with timelock and validation
    uint256 public constant UPGRADE_DELAY = 48 hours;
    uint256 public upgradeProposalTime;
    address public proposedImplementation;
    bytes32 public proposedImplementationCodeHash;
    
    event UpgradeProposed(address indexed newImplementation, uint256 proposalTime);
    event UpgradeExecuted(address indexed newImplementation);
    event UpgradeCancelled(address indexed proposedImplementation);
    
    /**
     * @notice Propose a contract upgrade with timelock
     */
    function proposeUpgrade(address newImplementation) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(newImplementation != address(0), "Invalid implementation address");
        require(newImplementation.code.length > 0, "Implementation must have code");
        
        // Get code hash for verification
        bytes32 codeHash;
        assembly {
            codeHash := extcodehash(newImplementation)
        }
        
        proposedImplementation = newImplementation;
        proposedImplementationCodeHash = codeHash;
        upgradeProposalTime = block.timestamp;
        
        emit UpgradeProposed(newImplementation, block.timestamp);
    }
    
    /**
     * @notice Execute a proposed upgrade after timelock
     */
    function executeUpgrade() external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(proposedImplementation != address(0), "No upgrade proposed");
        require(block.timestamp >= upgradeProposalTime + UPGRADE_DELAY, "Timelock not expired");
        
        // Verify implementation hasn't changed
        bytes32 currentCodeHash;
        assembly {
            currentCodeHash := extcodehash(proposedImplementation)
        }
        require(currentCodeHash == proposedImplementationCodeHash, "Implementation code changed");
        
        address implementation = proposedImplementation;
        
        // Clear proposal
        proposedImplementation = address(0);
        proposedImplementationCodeHash = bytes32(0);
        upgradeProposalTime = 0;
        
        // Execute upgrade
        _upgradeTo(implementation);
        
        emit UpgradeExecuted(implementation);
    }
    
    /**
     * @notice Cancel a proposed upgrade
     */
    function cancelUpgrade() external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(proposedImplementation != address(0), "No upgrade proposed");
        
        address cancelled = proposedImplementation;
        
        proposedImplementation = address(0);
        proposedImplementationCodeHash = bytes32(0);
        upgradeProposalTime = 0;
        
        emit UpgradeCancelled(cancelled);
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {
        // This function is called by UUPSUpgradeable, but we override the flow
        // to use our timelock mechanism instead
        revert("Use proposeUpgrade and executeUpgrade instead");
    }
}