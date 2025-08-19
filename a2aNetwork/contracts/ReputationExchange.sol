// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title ReputationExchange
 * @dev Smart contract for peer-to-peer reputation endorsements and tracking
 */
contract ReputationExchange is Ownable, ReentrancyGuard, Pausable {
    using Counters for Counters.Counter;
    
    // Constants
    uint8 public constant MAX_REPUTATION = 200;
    uint8 public constant MIN_REPUTATION = 0;
    uint8 public constant DEFAULT_REPUTATION = 100;
    uint8 public constant MAX_ENDORSEMENT_AMOUNT = 10;
    uint8 public constant DAILY_ENDORSEMENT_LIMIT = 50;
    uint8 public constant WEEKLY_PEER_LIMIT = 10;
    uint256 public constant RECIPROCAL_COOLDOWN = 24 hours;
    
    // Counters
    Counters.Counter private _endorsementIdCounter;
    
    // Structs
    struct Agent {
        address agentAddress;
        uint8 reputation;
        uint256 lastUpdated;
        bool isActive;
        string name;
        string endpoint;
    }
    
    struct Endorsement {
        uint256 endorsementId;
        address fromAgent;
        address toAgent;
        uint8 amount;
        string reason;
        bytes32 contextHash;
        uint256 timestamp;
        bool isVerified;
    }
    
    struct DailyLimits {
        uint256 date; // Unix timestamp of the day
        uint8 given;
    }
    
    struct WeeklyPeerLimits {
        uint256 week; // Week identifier
        uint8 given;
    }
    
    // State variables
    mapping(address => Agent) public agents;
    mapping(uint256 => Endorsement) public endorsements;
    mapping(address => DailyLimits) public dailyLimits;
    mapping(bytes32 => WeeklyPeerLimits) public weeklyPeerLimits;
    mapping(bytes32 => uint256) public reciprocalCooldowns; // Hash of agent pair => timestamp
    
    // Events
    event AgentRegistered(
        address indexed agentAddress,
        string name,
        string endpoint
    );
    
    event ReputationChanged(
        address indexed agentAddress,
        uint8 oldReputation,
        uint8 newReputation,
        string reason
    );
    
    event EndorsementCreated(
        uint256 indexed endorsementId,
        address indexed fromAgent,
        address indexed toAgent,
        uint8 amount,
        string reason,
        bytes32 contextHash
    );
    
    event EndorsementVerified(
        uint256 indexed endorsementId,
        address verifiedBy
    );
    
    event ReputationMilestone(
        address indexed agentAddress,
        uint8 milestone,
        string badge
    );
    
    // Modifiers
    modifier onlyActiveAgent(address _agent) {
        require(agents[_agent].isActive, "Agent is not active");
        _;
    }
    
    modifier validReputation(uint8 _reputation) {
        require(_reputation >= MIN_REPUTATION && _reputation <= MAX_REPUTATION, "Invalid reputation value");
        _;
    }
    
    modifier canEndorse(address _from, address _to, uint8 _amount) {
        require(_from != _to, "Cannot endorse yourself");
        require(agents[_from].isActive, "Endorser is not active");
        require(agents[_to].isActive, "Recipient is not active");
        require(_amount > 0 && _amount <= getMaxEndorsementAmount(_from), "Invalid endorsement amount");
        require(checkDailyLimit(_from, _amount), "Daily endorsement limit exceeded");
        require(checkWeeklyPeerLimit(_from, _to, _amount), "Weekly peer limit exceeded");
        require(!hasRecentReciprocal(_from, _to), "Reciprocal endorsement cooldown active");
        _;
    }
    
    // Constructor
    constructor() {
        _pause(); // Start paused for security
    }
    
    /**
     * @dev Register a new agent
     */
    function registerAgent(
        address _agentAddress,
        string memory _name,
        string memory _endpoint
    ) external onlyOwner {
        require(_agentAddress != address(0), "Invalid agent address");
        require(bytes(_name).length > 0, "Name cannot be empty");
        require(!agents[_agentAddress].isActive, "Agent already registered");
        
        agents[_agentAddress] = Agent({
            agentAddress: _agentAddress,
            reputation: DEFAULT_REPUTATION,
            lastUpdated: block.timestamp,
            isActive: true,
            name: _name,
            endpoint: _endpoint
        });
        
        emit AgentRegistered(_agentAddress, _name, _endpoint);
        emit ReputationChanged(_agentAddress, 0, DEFAULT_REPUTATION, "INITIAL_REGISTRATION");
    }
    
    /**
     * @dev Endorse another agent
     */
    function endorsePeer(
        address _to,
        uint8 _amount,
        string memory _reason,
        bytes32 _contextHash
    ) external whenNotPaused nonReentrant onlyActiveAgent(msg.sender) canEndorse(msg.sender, _to, _amount) {
        
        // Create endorsement
        _endorsementIdCounter.increment();
        uint256 endorsementId = _endorsementIdCounter.current();
        
        endorsements[endorsementId] = Endorsement({
            endorsementId: endorsementId,
            fromAgent: msg.sender,
            toAgent: _to,
            amount: _amount,
            reason: _reason,
            contextHash: _contextHash,
            timestamp: block.timestamp,
            isVerified: false
        });
        
        // Update limits
        _updateDailyLimit(msg.sender, _amount);
        _updateWeeklyPeerLimit(msg.sender, _to, _amount);
        _updateReciprocalCooldown(msg.sender, _to);
        
        // Apply reputation change
        _increaseReputation(_to, _amount, "PEER_ENDORSEMENT");
        
        emit EndorsementCreated(endorsementId, msg.sender, _to, _amount, _reason, _contextHash);
    }
    
    /**
     * @dev Apply reputation change (admin only)
     */
    function applyReputationChange(
        address _agent,
        int8 _change,
        string memory _reason
    ) external onlyOwner onlyActiveAgent(_agent) {
        uint8 oldReputation = agents[_agent].reputation;
        uint8 newReputation = _calculateNewReputation(oldReputation, _change);
        
        agents[_agent].reputation = newReputation;
        agents[_agent].lastUpdated = block.timestamp;
        
        emit ReputationChanged(_agent, oldReputation, newReputation, _reason);
        
        // Check for milestones
        _checkMilestones(_agent, oldReputation, newReputation);
    }
    
    /**
     * @dev Verify an endorsement (can be called by involved parties or owner)
     */
    function verifyEndorsement(uint256 _endorsementId) external {
        Endorsement storage endorsement = endorsements[_endorsementId];
        require(endorsement.endorsementId != 0, "Endorsement does not exist");
        require(!endorsement.isVerified, "Endorsement already verified");
        require(
            msg.sender == endorsement.fromAgent || 
            msg.sender == endorsement.toAgent || 
            msg.sender == owner(),
            "Not authorized to verify"
        );
        
        endorsement.isVerified = true;
        emit EndorsementVerified(_endorsementId, msg.sender);
    }
    
    /**
     * @dev Get agent information
     */
    function getAgent(address _agent) external view returns (Agent memory) {
        return agents[_agent];
    }
    
    /**
     * @dev Get endorsement information
     */
    function getEndorsement(uint256 _endorsementId) external view returns (Endorsement memory) {
        return endorsements[_endorsementId];
    }
    
    /**
     * @dev Get maximum endorsement amount based on reputation
     */
    function getMaxEndorsementAmount(address _agent) public view returns (uint8) {
        uint8 reputation = agents[_agent].reputation;
        if (reputation <= 50) return 3;
        if (reputation <= 100) return 5;
        if (reputation <= 150) return 7;
        return 10;
    }
    
    /**
     * @dev Check if daily limit allows endorsement
     */
    function checkDailyLimit(address _agent, uint8 _amount) public view returns (bool) {
        uint256 today = block.timestamp / 1 days;
        DailyLimits memory limits = dailyLimits[_agent];
        
        if (limits.date != today) {
            return _amount <= DAILY_ENDORSEMENT_LIMIT;
        }
        
        return (limits.given + _amount) <= DAILY_ENDORSEMENT_LIMIT;
    }
    
    /**
     * @dev Check if weekly peer limit allows endorsement
     */
    function checkWeeklyPeerLimit(address _from, address _to, uint8 _amount) public view returns (bool) {
        bytes32 key = keccak256(abi.encodePacked(_from, _to, getCurrentWeek()));
        WeeklyPeerLimits memory limits = weeklyPeerLimits[key];
        
        return (limits.given + _amount) <= WEEKLY_PEER_LIMIT;
    }
    
    /**
     * @dev Check if there's a recent reciprocal endorsement
     */
    function hasRecentReciprocal(address _from, address _to) public view returns (bool) {
        bytes32 key = keccak256(abi.encodePacked(_to, _from)); // Note: reversed order
        uint256 lastReciprocal = reciprocalCooldowns[key];
        
        return (block.timestamp - lastReciprocal) < RECIPROCAL_COOLDOWN;
    }
    
    /**
     * @dev Get current week identifier
     */
    function getCurrentWeek() public view returns (uint256) {
        return block.timestamp / 1 weeks;
    }
    
    /**
     * @dev Get reputation badge for given reputation score
     */
    function getReputationBadge(uint8 _reputation) external pure returns (string memory) {
        if (_reputation < 50) return "NEWCOMER";
        if (_reputation < 100) return "ESTABLISHED";
        if (_reputation < 150) return "TRUSTED";
        if (_reputation < 200) return "EXPERT";
        return "LEGENDARY";
    }
    
    // Internal functions
    
    function _increaseReputation(address _agent, uint8 _amount, string memory _reason) internal {
        uint8 oldReputation = agents[_agent].reputation;
        uint8 newReputation = oldReputation + _amount;
        if (newReputation > MAX_REPUTATION) {
            newReputation = MAX_REPUTATION;
        }
        
        agents[_agent].reputation = newReputation;
        agents[_agent].lastUpdated = block.timestamp;
        
        emit ReputationChanged(_agent, oldReputation, newReputation, _reason);
        _checkMilestones(_agent, oldReputation, newReputation);
    }
    
    function _calculateNewReputation(uint8 _current, int8 _change) internal pure returns (uint8) {
        if (_change >= 0) {
            uint8 increase = uint8(_change);
            return _current + increase > MAX_REPUTATION ? MAX_REPUTATION : _current + increase;
        } else {
            uint8 decrease = uint8(-_change);
            return _current < decrease ? MIN_REPUTATION : _current - decrease;
        }
    }
    
    function _updateDailyLimit(address _agent, uint8 _amount) internal {
        uint256 today = block.timestamp / 1 days;
        DailyLimits storage limits = dailyLimits[_agent];
        
        if (limits.date != today) {
            limits.date = today;
            limits.given = _amount;
        } else {
            limits.given += _amount;
        }
    }
    
    function _updateWeeklyPeerLimit(address _from, address _to, uint8 _amount) internal {
        bytes32 key = keccak256(abi.encodePacked(_from, _to, getCurrentWeek()));
        weeklyPeerLimits[key].given += _amount;
    }
    
    function _updateReciprocalCooldown(address _from, address _to) internal {
        bytes32 key = keccak256(abi.encodePacked(_from, _to));
        reciprocalCooldowns[key] = block.timestamp;
    }
    
    function _checkMilestones(address _agent, uint8 _oldRep, uint8 _newRep) internal {
        uint8[4] memory milestones = [50, 100, 150, 200];
        string[4] memory badges = ["ESTABLISHED", "TRUSTED", "EXPERT", "LEGENDARY"];
        
        for (uint i = 0; i < milestones.length; i++) {
            if (_oldRep < milestones[i] && _newRep >= milestones[i]) {
                emit ReputationMilestone(_agent, milestones[i], badges[i]);
            }
        }
    }
    
    // Admin functions
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function deactivateAgent(address _agent) external onlyOwner {
        require(agents[_agent].isActive, "Agent is not active");
        agents[_agent].isActive = false;
    }
    
    function reactivateAgent(address _agent) external onlyOwner {
        require(!agents[_agent].isActive, "Agent is already active");
        require(agents[_agent].agentAddress != address(0), "Agent not registered");
        agents[_agent].isActive = true;
    }
    
    // Emergency functions
    
    function emergencyResetReputation(address _agent, uint8 _newReputation) 
        external 
        onlyOwner 
        validReputation(_newReputation) 
    {
        uint8 oldReputation = agents[_agent].reputation;
        agents[_agent].reputation = _newReputation;
        agents[_agent].lastUpdated = block.timestamp;
        
        emit ReputationChanged(_agent, oldReputation, _newReputation, "EMERGENCY_RESET");
    }
    
    function emergencyWithdraw() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}