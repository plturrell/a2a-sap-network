// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/governance/TimelockControllerUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title A2ATimelock
 * @notice Timelock controller for A2A Network governance with enhanced security features
 * @dev Extends OpenZeppelin TimelockController with custom security mechanisms
 */
contract A2ATimelock is 
    TimelockControllerUpgradeable,
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable
{
    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");
    bytes32 public constant GUARDIAN_ROLE = keccak256("GUARDIAN_ROLE");
    bytes32 public constant SCHEDULER_ROLE = keccak256("SCHEDULER_ROLE");
    
    struct OperationMetadata {
        uint256 category;
        uint256 riskLevel;
        string description;
        address[] reviewers;
        mapping(address => bool) hasReviewed;
        uint256 reviewCount;
        bool emergencyBypass;
        uint256 scheduledAt;
    }
    
    enum RiskLevel {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }
    
    mapping(bytes32 => OperationMetadata) public operationMetadata;
    mapping(uint256 => uint256) public categoryDelays;
    mapping(RiskLevel => uint256) public riskDelays;
    mapping(bytes32 => bool) public vetoedOperations;
    
    uint256 public constant MIN_DELAY = 1 days;
    uint256 public constant MAX_DELAY = 30 days;
    uint256 public constant EMERGENCY_DELAY = 6 hours;
    uint256 public constant REVIEW_PERIOD = 12 hours;
    
    bool public emergencyMode;
    uint256 public emergencyActivatedAt;
    uint256 public guardianVetoWindow = 24 hours;
    
    event OperationScheduledWithMetadata(
        bytes32 indexed id,
        uint256 category,
        uint256 riskLevel,
        string description
    );
    event OperationReviewed(bytes32 indexed id, address reviewer);
    event OperationVetoed(bytes32 indexed id, address guardian, string reason);
    event EmergencyBypassActivated(bytes32 indexed id, address activator);
    event CategoryDelayUpdated(uint256 category, uint256 delay);
    event RiskDelayUpdated(RiskLevel riskLevel, uint256 delay);
    
    modifier onlyGuardian() {
        require(hasRole(GUARDIAN_ROLE, msg.sender), "Not a guardian");
        _;
    }
    
    modifier validRiskLevel(uint256 riskLevel) {
        require(riskLevel <= uint256(RiskLevel.CRITICAL), "Invalid risk level");
        _;
    }
    
    modifier notVetoed(bytes32 id) {
        require(!vetoedOperations[id], "Operation has been vetoed");
        _;
    }
    
    function initialize(
        uint256 minDelay,
        address[] memory proposers,
        address[] memory executors,
        address admin
    ) public initializer {
        __TimelockController_init(minDelay, proposers, executors, admin);
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(GUARDIAN_ROLE, admin);
        _grantRole(EMERGENCY_ROLE, admin);
        
        // Initialize risk-based delays
        riskDelays[RiskLevel.LOW] = 1 days;
        riskDelays[RiskLevel.MEDIUM] = 3 days;
        riskDelays[RiskLevel.HIGH] = 7 days;
        riskDelays[RiskLevel.CRITICAL] = 14 days;
        
        // Initialize category delays
        categoryDelays[0] = 2 days; // Protocol upgrades
        categoryDelays[1] = 1 days; // Parameter changes
        categoryDelays[2] = 5 days; // Treasury operations
        categoryDelays[3] = EMERGENCY_DELAY; // Emergency actions
    }
    
    /**
     * @notice Schedule operation with enhanced metadata
     */
    function scheduleWithMetadata(
        address target,
        uint256 value,
        bytes calldata data,
        bytes32 predecessor,
        bytes32 salt,
        uint256 delay,
        uint256 category,
        uint256 riskLevel,
        string memory description,
        address[] memory reviewers
    ) external onlyRole(PROPOSER_ROLE) validRiskLevel(riskLevel) returns (bytes32) {
        
        // Calculate minimum delay based on category and risk
        uint256 minRequiredDelay = _calculateMinDelay(category, RiskLevel(riskLevel));
        require(delay >= minRequiredDelay, "Delay too short for category/risk");
        
        bytes32 id = hashOperation(target, value, data, predecessor, salt);
        
        // Store metadata
        OperationMetadata storage metadata = operationMetadata[id];
        metadata.category = category;
        metadata.riskLevel = riskLevel;
        metadata.description = description;
        metadata.reviewers = reviewers;
        metadata.reviewCount = 0;
        metadata.emergencyBypass = false;
        metadata.scheduledAt = block.timestamp;
        
        // Schedule the operation
        schedule(target, value, data, predecessor, salt, delay);
        
        emit OperationScheduledWithMetadata(id, category, riskLevel, description);
        
        return id;
    }
    
    /**
     * @notice Batch schedule multiple operations
     */
    function batchScheduleWithMetadata(
        address[] calldata targets,
        uint256[] calldata values,
        bytes[] calldata payloads,
        bytes32 predecessor,
        bytes32 salt,
        uint256 delay,
        uint256 category,
        uint256 riskLevel,
        string memory description
    ) external onlyRole(PROPOSER_ROLE) validRiskLevel(riskLevel) returns (bytes32) {
        
        require(targets.length == values.length, "Length mismatch");
        require(targets.length == payloads.length, "Length mismatch");
        require(targets.length <= 10, "Too many operations");
        
        uint256 minRequiredDelay = _calculateMinDelay(category, RiskLevel(riskLevel));
        require(delay >= minRequiredDelay, "Delay too short for category/risk");
        
        bytes32 id = hashOperationBatch(targets, values, payloads, predecessor, salt);
        
        // Store metadata
        OperationMetadata storage metadata = operationMetadata[id];
        metadata.category = category;
        metadata.riskLevel = riskLevel;
        metadata.description = description;
        metadata.reviewCount = 0;
        metadata.emergencyBypass = false;
        metadata.scheduledAt = block.timestamp;
        
        // Schedule the batch operation
        scheduleBatch(targets, values, payloads, predecessor, salt, delay);
        
        emit OperationScheduledWithMetadata(id, category, riskLevel, description);
        
        return id;
    }
    
    /**
     * @notice Review an operation (for required reviewers)
     */
    function reviewOperation(bytes32 id, bool approve, string memory comments) 
        external 
        nonReentrant 
    {
        OperationMetadata storage metadata = operationMetadata[id];
        require(metadata.scheduledAt > 0, "Operation not found");
        require(!metadata.hasReviewed[msg.sender], "Already reviewed");
        
        // Check if sender is a required reviewer
        bool isRequiredReviewer = false;
        for (uint256 i = 0; i < metadata.reviewers.length; i++) {
            if (metadata.reviewers[i] == msg.sender) {
                isRequiredReviewer = true;
                break;
            }
        }
        require(isRequiredReviewer, "Not a required reviewer");
        
        if (approve) {
            metadata.hasReviewed[msg.sender] = true;
            metadata.reviewCount++;
        } else {
            // Reviewer disapproval vetoes the operation
            vetoedOperations[id] = true;
        }
        
        emit OperationReviewed(id, msg.sender);
    }
    
    /**
     * @notice Guardian veto function
     */
    function vetoOperation(bytes32 id, string memory reason) 
        external 
        onlyGuardian 
        nonReentrant 
    {
        require(isOperationPending(id), "Operation not pending");
        require(
            block.timestamp <= operationMetadata[id].scheduledAt + guardianVetoWindow,
            "Veto window expired"
        );
        
        vetoedOperations[id] = true;
        
        emit OperationVetoed(id, msg.sender, reason);
    }
    
    /**
     * @notice Emergency bypass for critical operations
     */
    function emergencyBypass(bytes32 id, string memory justification) 
        external 
        onlyRole(EMERGENCY_ROLE) 
        nonReentrant 
    {
        require(isOperationPending(id), "Operation not pending");
        require(
            operationMetadata[id].riskLevel <= uint256(RiskLevel.HIGH),
            "Cannot bypass critical operations"
        );
        require(emergencyMode, "Emergency mode not active");
        
        operationMetadata[id].emergencyBypass = true;
        
        // Set operation to be ready immediately
        _setOperationReady(id);
        
        emit EmergencyBypassActivated(id, msg.sender);
    }
    
    /**
     * @notice Execute operation with additional checks
     */
    function execute(
        address target,
        uint256 value,
        bytes calldata payload,
        bytes32 predecessor,
        bytes32 salt
    ) public payable override nonReentrant notVetoed(hashOperation(target, value, payload, predecessor, salt)) {
        bytes32 id = hashOperation(target, value, payload, predecessor, salt);
        
        // Check if required reviews are completed (unless emergency bypass)
        if (!operationMetadata[id].emergencyBypass) {
            _checkRequiredReviews(id);
        }
        
        super.execute(target, value, payload, predecessor, salt);
    }
    
    /**
     * @notice Execute batch operation with additional checks
     */
    function executeBatch(
        address[] calldata targets,
        uint256[] calldata values,
        bytes[] calldata payloads,
        bytes32 predecessor,
        bytes32 salt
    ) public payable override nonReentrant notVetoed(hashOperationBatch(targets, values, payloads, predecessor, salt)) {
        bytes32 id = hashOperationBatch(targets, values, payloads, predecessor, salt);
        
        // Check if required reviews are completed (unless emergency bypass)
        if (!operationMetadata[id].emergencyBypass) {
            _checkRequiredReviews(id);
        }
        
        super.executeBatch(targets, values, payloads, predecessor, salt);
    }
    
    /**
     * @notice Cancel operation
     */
    function cancel(bytes32 id) public override onlyRole(CANCELLER_ROLE) {
        super.cancel(id);
        
        // Clean up metadata
        delete operationMetadata[id];
        delete vetoedOperations[id];
    }
    
    /**
     * @notice Activate emergency mode
     */
    function activateEmergencyMode() external onlyRole(EMERGENCY_ROLE) {
        emergencyMode = true;
        emergencyActivatedAt = block.timestamp;
    }
    
    /**
     * @notice Deactivate emergency mode
     */
    function deactivateEmergencyMode() external onlyRole(DEFAULT_ADMIN_ROLE) {
        emergencyMode = false;
        emergencyActivatedAt = 0;
    }
    
    /**
     * @notice Update category delay
     */
    function updateCategoryDelay(uint256 category, uint256 delay) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(delay >= MIN_DELAY && delay <= MAX_DELAY, "Invalid delay");
        categoryDelays[category] = delay;
        emit CategoryDelayUpdated(category, delay);
    }
    
    /**
     * @notice Update risk-based delay
     */
    function updateRiskDelay(RiskLevel riskLevel, uint256 delay) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(delay >= MIN_DELAY && delay <= MAX_DELAY, "Invalid delay");
        riskDelays[riskLevel] = delay;
        emit RiskDelayUpdated(riskLevel, delay);
    }
    
    /**
     * @notice Update guardian veto window
     */
    function updateGuardianVetoWindow(uint256 window) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(window >= 1 hours && window <= 7 days, "Invalid window");
        guardianVetoWindow = window;
    }
    
    /**
     * @notice Add guardian
     */
    function addGuardian(address guardian) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _grantRole(GUARDIAN_ROLE, guardian);
    }
    
    /**
     * @notice Remove guardian
     */
    function removeGuardian(address guardian) external onlyRole(DEFAULT_ADMIN_ROLE) {
        _revokeRole(GUARDIAN_ROLE, guardian);
    }
    
    /**
     * @notice Get operation metadata
     */
    function getOperationMetadata(bytes32 id) 
        external 
        view 
        returns (
            uint256 category,
            uint256 riskLevel,
            string memory description,
            address[] memory reviewers,
            uint256 reviewCount,
            bool emergencyBypass,
            uint256 scheduledAt
        ) 
    {
        OperationMetadata storage metadata = operationMetadata[id];
        return (
            metadata.category,
            metadata.riskLevel,
            metadata.description,
            metadata.reviewers,
            metadata.reviewCount,
            metadata.emergencyBypass,
            metadata.scheduledAt
        );
    }
    
    /**
     * @notice Check if operation requires reviews and if they're completed
     */
    function _checkRequiredReviews(bytes32 id) internal view {
        OperationMetadata storage metadata = operationMetadata[id];
        
        // High and critical risk operations require all reviewers to approve
        if (metadata.riskLevel >= uint256(RiskLevel.HIGH)) {
            require(
                metadata.reviewCount >= metadata.reviewers.length,
                "Required reviews not completed"
            );
        }
        // Medium risk operations require majority approval
        else if (metadata.riskLevel == uint256(RiskLevel.MEDIUM)) {
            require(
                metadata.reviewCount >= (metadata.reviewers.length + 1) / 2,
                "Majority review not achieved"
            );
        }
        // Low risk operations don't require reviews
    }
    
    /**
     * @notice Calculate minimum delay based on category and risk
     */
    function _calculateMinDelay(uint256 category, RiskLevel riskLevel) 
        internal 
        view 
        returns (uint256) 
    {
        uint256 categoryDelay = categoryDelays[category];
        uint256 riskDelay = riskDelays[riskLevel];
        
        // Return the maximum of category and risk delays
        return categoryDelay > riskDelay ? categoryDelay : riskDelay;
    }
    
    /**
     * @notice Internal function to set operation as ready (for emergency bypass)
     */
    function _setOperationReady(bytes32 id) internal {
        // This would require access to internal state of TimelockController
        // In practice, this might need to be implemented differently
        // or use a modified version of TimelockController
    }
    
    /**
     * @notice Check if address has guardian role
     */
    function isGuardian(address account) external view returns (bool) {
        return hasRole(GUARDIAN_ROLE, account);
    }
    
    /**
     * @notice Get minimum delay for a category and risk level
     */
    function getMinDelayFor(uint256 category, uint256 riskLevel) 
        external 
        view 
        validRiskLevel(riskLevel)
        returns (uint256) 
    {
        return _calculateMinDelay(category, RiskLevel(riskLevel));
    }
    
    function _authorizeUpgrade(address newImplementation) 
        internal 
        override 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {}
    
    /**
     * @notice Override supportsInterface for multiple inheritance
     */
    function supportsInterface(bytes4 interfaceId)
        public
        view
        virtual
        override(TimelockControllerUpgradeable, AccessControlUpgradeable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}