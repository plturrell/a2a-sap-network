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
    
    struct Service {
        address provider;
        string name;
        string description;
        string[] capabilities;
        uint256 basePrice;
        ServiceType serviceType;
        uint256 minReputation;
        uint256 maxConcurrent;
        uint256 currentActive;
        bool active;
    }
    
    struct ServiceRequest {
        uint256 serviceId;
        address requester;
        address provider;
        uint256 agreedPrice;
        uint256 deadline;
        ServiceStatus status;
        bytes parameters;
        bytes result;
        uint256 escrowAmount;
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
        
        request.escrowAmount = 0;
        services[request.serviceId].currentActive--;
        
        agentEarnings[request.provider] += providerPayment;
        agentSpending[request.requester] += request.agreedPrice;
        
        payable(request.provider).transfer(providerPayment);
        
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
        
        request.escrowAmount = 0;
        request.status = ServiceStatus.Completed;
        services[request.serviceId].currentActive--;
        
        if (providerAmount > 0) {
            agentEarnings[request.provider] += providerAmount;
            payable(request.provider).transfer(providerAmount);
        }
        
        if (requesterRefund > 0) {
            payable(request.requester).transfer(requesterRefund);
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
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}