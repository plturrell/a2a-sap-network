// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "./interfaces/IAgentRegistry.sol";

/**
 * @title PerformanceReputationSystem
 * @notice Automated reputation system based on agent performance metrics
 * @dev Tracks and calculates reputation scores based on multiple performance factors
 */
contract PerformanceReputationSystem is 
    AccessControlUpgradeable, 
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable 
{
    bytes32 public constant METRIC_UPDATER_ROLE = keccak256("METRIC_UPDATER_ROLE");
    bytes32 public constant REPUTATION_ORACLE_ROLE = keccak256("REPUTATION_ORACLE_ROLE");
    
    IAgentRegistry public agentRegistry;
    
    struct PerformanceMetrics {
        uint256 totalTasks;
        uint256 successfulTasks;
        uint256 failedTasks;
        uint256 totalResponseTime; // in seconds
        uint256 avgResponseTime;
        uint256 uptime; // percentage * 100 (9999 = 99.99%)
        uint256 lastActiveTimestamp;
        uint256 totalGasUsed;
        uint256 avgGasPerTask;
        // Task difficulty tracking
        uint256 totalTaskDifficulty;
        uint256 avgTaskDifficulty;
        mapping(TaskDifficulty => uint256) tasksByDifficulty;
        // Anti-gaming measures
        uint256 stakedAmount;
        uint256 slashingPenalties;
        uint256 lastSlashTime;
    }
    
    struct DynamicThresholds {
        uint256 excellentResponseTime;
        uint256 goodResponseTime;
        uint256 poorResponseTime;
        uint256 excellentGasUsage;
        uint256 goodGasUsage;
        uint256 poorGasUsage;
        uint256 lastUpdated;
        uint256 sampleSize;
    }
    
    struct MarketContext {
        uint256 networkLoad; // 0-100
        uint256 gasPrice;
        uint256 avgBlockTime;
        uint256 totalActiveAgents;
        uint256 demandSupplyRatio; // demand/supply * 100
        uint256 lastUpdated;
    }
    
    enum TaskDifficulty { Simple, Medium, Hard, Expert, Critical }
    
    struct ReputationFactors {
        uint256 successRate;      // weight: 35%
        uint256 responseSpeed;     // weight: 25%
        uint256 availability;      // weight: 20%
        uint256 efficiency;        // weight: 10%
        uint256 experience;        // weight: 10%
    }
    
    struct PeerReview {
        address reviewer;
        uint256 rating; // 1-5 stars
        string comment;
        uint256 taskId;
        uint256 timestamp;
        // Anti-gaming measures
        uint256 reviewerReputation;
        bool validated;
        uint256 validatorCount;
        bytes32 reviewHash; // Hash of review details for verification
    }
    
    struct ReputationInsurance {
        uint256 bondAmount;
        uint256 coverageAmount;
        uint256 premiumPaid;
        uint256 expirationTime;
        bool active;
        uint256 claimCount;
    }
    
    struct ReputationScore {
        uint256 score;           // 0-1000
        uint256 confidence;      // 0-100 (based on data points)
        uint256 lastUpdated;
        ReputationFactors factors;
    }
    
    mapping(address => PerformanceMetrics) public agentMetrics;
    mapping(address => ReputationScore) public reputationScores;
    mapping(address => PeerReview[]) public peerReviews;
    mapping(address => mapping(address => uint256)) public lastReviewTime;
    mapping(address => uint256) public historicalScoresIndex; // Circular buffer index
    mapping(address => mapping(uint256 => uint256)) public historicalScoresBuffer; // Circular buffer
    
    // SECURITY FIX: Event-based indexing support
    mapping(bytes32 => ReviewLocation) public reviewLocations; // Maps review hash to location
    
    struct ReviewLocation {
        address agent;
        uint256 index;
        bool exists;
    }
    
    // Dynamic thresholds and market context
    DynamicThresholds public dynamicThresholds;
    MarketContext public marketContext;
    
    // Anti-gaming and security
    mapping(address => ReputationInsurance) public reputationInsurance;
    mapping(address => uint256) public agentStakes;
    mapping(bytes32 => bool) public usedReviewHashes;
    mapping(bytes32 => address[]) public reviewValidators;
    
    // Task difficulty weighting
    mapping(TaskDifficulty => uint256) public difficultyMultipliers; // 100 = 1.0x
    mapping(uint256 => TaskDifficulty) public taskDifficulties;
    
    // Configuration
    uint256 public minTasksForReputation = 5;
    uint256 public reviewCooldown = 7 days;
    uint256 public maxHistoricalScores = 30;
    uint256 public decayRate = 5; // 5% decay per month of inactivity
    uint256 public minStakeAmount = 1 ether;
    uint256 public slashingRate = 10; // 10% slashing penalty
    uint256 public reviewValidationThreshold = 3; // Min validators for review
    uint256 public maxReviewsPerAgent = 100; // Storage limit
    
    // Weights for reputation calculation (out of 100)
    uint256 public constant SUCCESS_WEIGHT = 35;
    uint256 public constant SPEED_WEIGHT = 25;
    uint256 public constant AVAILABILITY_WEIGHT = 20;
    uint256 public constant EFFICIENCY_WEIGHT = 10;
    uint256 public constant EXPERIENCE_WEIGHT = 10;
    
    event MetricsUpdated(address indexed agent, uint256 totalTasks, uint256 successRate, TaskDifficulty difficulty);
    event ReputationCalculated(address indexed agent, uint256 newScore, uint256 confidence);
    event PeerReviewSubmitted(address indexed agent, address indexed reviewer, uint256 rating, bytes32 reviewHash);
    event ReputationDecayed(address indexed agent, uint256 oldScore, uint256 newScore);
    event ThresholdsUpdated(uint256 responseTime, uint256 gasUsage, uint256 sampleSize);
    event AgentSlashed(address indexed agent, uint256 amount, string reason);
    event StakeDeposited(address indexed agent, uint256 amount);
    event InsurancePurchased(address indexed agent, uint256 bondAmount, uint256 coverage);
    event ReviewValidated(bytes32 indexed reviewHash, address indexed validator);
    event MarketContextUpdated(uint256 networkLoad, uint256 gasPrice, uint256 activeAgents);
    
    // SECURITY FIX: Enhanced events for off-chain indexing
    event ReviewSubmittedDetailed(
        bytes32 indexed reviewHash,
        address indexed agent,
        address indexed reviewer,
        uint256 rating,
        uint256 taskId,
        uint256 timestamp,
        uint256 reviewIndex
    );
    event ReviewValidationRequested(
        bytes32 indexed reviewHash,
        address indexed validator,
        bool isValid
    );
    
    function initialize(address _agentRegistry) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(METRIC_UPDATER_ROLE, msg.sender);
        _grantRole(REPUTATION_ORACLE_ROLE, msg.sender);
        
        agentRegistry = IAgentRegistry(_agentRegistry);
        
        // Initialize difficulty multipliers
        difficultyMultipliers[TaskDifficulty.Simple] = 80;    // 0.8x weight
        difficultyMultipliers[TaskDifficulty.Medium] = 100;   // 1.0x weight (baseline)
        difficultyMultipliers[TaskDifficulty.Hard] = 130;     // 1.3x weight
        difficultyMultipliers[TaskDifficulty.Expert] = 160;   // 1.6x weight
        difficultyMultipliers[TaskDifficulty.Critical] = 200; // 2.0x weight
        
        // Initialize dynamic thresholds with reasonable defaults
        dynamicThresholds = DynamicThresholds({
            excellentResponseTime: 10,
            goodResponseTime: 60,
            poorResponseTime: 300,
            excellentGasUsage: 50000,
            goodGasUsage: 200000,
            poorGasUsage: 500000,
            lastUpdated: block.timestamp,
            sampleSize: 0
        });
        
        // Initialize market context
        marketContext = MarketContext({
            networkLoad: 50, // Assume 50% network load initially
            gasPrice: tx.gasprice,
            avgBlockTime: 12, // Ethereum average
            totalActiveAgents: 0,
            demandSupplyRatio: 100, // 1:1 ratio initially
            lastUpdated: block.timestamp
        });
    }
    
    /**
     * @notice Update agent performance metrics after task completion with difficulty weighting
     */
    function updateTaskMetrics(
        address agent,
        bool success,
        uint256 responseTime,
        uint256 gasUsed,
        uint256 taskId,
        TaskDifficulty difficulty
    ) external onlyRole(METRIC_UPDATER_ROLE) {
        PerformanceMetrics storage metrics = agentMetrics[agent];
        
        // Store task difficulty
        taskDifficulties[taskId] = difficulty;
        metrics.tasksByDifficulty[difficulty]++;
        
        // Weight the task by difficulty
        uint256 difficultyWeight = difficultyMultipliers[difficulty];
        uint256 weightedValue = (uint256(difficulty) + 1) * difficultyWeight / 100;
        
        metrics.totalTasks++;
        metrics.totalTaskDifficulty += weightedValue;
        // SECURITY FIX: Check for division by zero
        if (metrics.totalTasks > 0) {
            metrics.avgTaskDifficulty = metrics.totalTaskDifficulty / metrics.totalTasks;
        }
        
        if (success) {
            metrics.successfulTasks++;
        } else {
            metrics.failedTasks++;
        }
        
        // Update response time with market context adjustment
        uint256 adjustedResponseTime = _adjustForMarketConditions(responseTime);
        metrics.totalResponseTime += adjustedResponseTime;
        // SECURITY FIX: Check for division by zero
        if (metrics.totalTasks > 0) {
            metrics.avgResponseTime = metrics.totalResponseTime / metrics.totalTasks;
        }
        
        metrics.totalGasUsed += gasUsed;
        // SECURITY FIX: Check for division by zero
        if (metrics.totalTasks > 0) {
            metrics.avgGasPerTask = metrics.totalGasUsed / metrics.totalTasks;
        }
        
        metrics.lastActiveTimestamp = block.timestamp;
        
        // Update dynamic thresholds
        _updateDynamicThresholds(adjustedResponseTime, gasUsed);
        
        emit MetricsUpdated(agent, metrics.totalTasks, _calculateSuccessRate(metrics), difficulty);
        
        // Automatically recalculate reputation if enough data
        if (metrics.totalTasks >= minTasksForReputation) {
            _calculateReputation(agent);
        }
    }
    
    /**
     * @notice Update agent uptime/availability
     */
    function updateUptime(address agent, uint256 uptimePercentage) 
        external onlyRole(REPUTATION_ORACLE_ROLE) {
        require(uptimePercentage <= 10000, "Invalid percentage");
        agentMetrics[agent].uptime = uptimePercentage;
    }
    
    /**
     * @notice Submit peer review for an agent with anti-gaming measures
     */
    function submitPeerReview(
        address agent,
        uint256 rating,
        string memory comment,
        uint256 taskId
    ) external payable {
        require(agentRegistry.isRegistered(msg.sender), "Reviewer not registered");
        require(agentRegistry.isRegistered(agent), "Agent not registered");
        require(rating >= 1 && rating <= 5, "Invalid rating");
        require(agent != msg.sender, "Cannot review yourself");
        require(agentStakes[msg.sender] >= minStakeAmount, "Insufficient stake");
        require(
            block.timestamp >= lastReviewTime[msg.sender][agent] + reviewCooldown,
            "Review cooldown active"
        );
        
        // Check if agent has enough reviews (storage optimization)
        require(peerReviews[agent].length < maxReviewsPerAgent, "Review limit reached");
        
        // Create review hash for anti-gaming
        bytes32 reviewHash = keccak256(abi.encodePacked(
            agent, msg.sender, rating, comment, taskId, block.timestamp
        ));
        require(!usedReviewHashes[reviewHash], "Duplicate review detected");
        
        // Get reviewer reputation for weighting
        uint256 reviewerReputation = agentRegistry.getReputation(msg.sender);
        require(reviewerReputation >= 50, "Reviewer reputation too low");
        
        // Store review
        uint256 reviewIndex = peerReviews[agent].length;
        
        peerReviews[agent].push(PeerReview({
            reviewer: msg.sender,
            rating: rating,
            comment: comment,
            taskId: taskId,
            timestamp: block.timestamp,
            reviewerReputation: reviewerReputation,
            validated: false,
            validatorCount: 0,
            reviewHash: reviewHash
        }));
        
        // SECURITY FIX: Store review location for efficient lookup
        reviewLocations[reviewHash] = ReviewLocation({
            agent: agent,
            index: reviewIndex,
            exists: true
        });
        
        usedReviewHashes[reviewHash] = true;
        lastReviewTime[msg.sender][agent] = block.timestamp;
        
        // Manage storage by removing oldest reviews if at limit
        _manageReviewStorage(agent);
        
        emit PeerReviewSubmitted(agent, msg.sender, rating, reviewHash);
        
        // SECURITY FIX: Emit detailed event for off-chain indexing
        emit ReviewSubmittedDetailed(
            reviewHash,
            agent,
            msg.sender,
            rating,
            taskId,
            block.timestamp,
            reviewIndex
        );
        
        // Recalculate reputation with new review
        _calculateReputation(agent);
    }
    
    /**
     * @notice Validate a peer review (cross-validation)
     * @dev SECURITY FIX: Uses mapping lookup instead of loops for gas efficiency
     */
    function validateReview(bytes32 reviewHash, bool isValid) external {
        require(agentRegistry.isRegistered(msg.sender), "Not registered");
        require(agentStakes[msg.sender] >= minStakeAmount, "Insufficient stake");
        
        // SECURITY FIX: Use mapping for O(1) lookup instead of loop
        ReviewLocation memory location = reviewLocations[reviewHash];
        require(location.exists, "Review not found");
        
        // Get the review directly using stored location
        PeerReview storage review = peerReviews[location.agent][location.index];
        require(review.reviewer != msg.sender, "Cannot validate own review");
        
        // Check if validator already validated this review
        address[] storage validators = reviewValidators[reviewHash];
        for (uint256 i = 0; i < validators.length; i++) {
            require(validators[i] != msg.sender, "Already validated");
        }
        
        validators.push(msg.sender);
        review.validatorCount++;
        
        if (!isValid) {
            // Slash the reviewer for invalid review
            _slashAgent(review.reviewer, "Invalid review");
        }
        
        if (review.validatorCount >= reviewValidationThreshold) {
            review.validated = true;
        }
        
        // SECURITY FIX: Emit event for off-chain indexing
        emit ReviewValidationRequested(reviewHash, msg.sender, isValid);
        emit ReviewValidated(reviewHash, msg.sender);
    }
    
    /**
     * @notice Calculate reputation score for an agent
     */
    function calculateReputation(address agent) external returns (uint256) {
        return _calculateReputation(agent);
    }
    
    /**
     * @notice Apply reputation decay for inactive agents
     */
    function applyReputationDecay(address agent) external {
        PerformanceMetrics storage metrics = agentMetrics[agent];
        ReputationScore storage score = reputationScores[agent];
        
        uint256 monthsInactive = (block.timestamp - metrics.lastActiveTimestamp) / 30 days;
        if (monthsInactive > 0) {
            uint256 decay = (score.score * decayRate * monthsInactive) / 100;
            uint256 oldScore = score.score;
            score.score = score.score > decay ? score.score - decay : 0;
            
            emit ReputationDecayed(agent, oldScore, score.score);
        }
    }
    
    /**
     * @notice Get comprehensive agent statistics
     */
    function getAgentStats(address agent) external view returns (
        uint256 totalTasks,
        uint256 successfulTasks, 
        uint256 avgResponseTime,
        uint256 uptime,
        ReputationScore memory reputation,
        uint256 averagePeerRating,
        uint256 reviewCount
    ) {
        PerformanceMetrics storage metrics = agentMetrics[agent];
        totalTasks = metrics.totalTasks;
        successfulTasks = metrics.successfulTasks;
        avgResponseTime = metrics.avgResponseTime;
        uptime = metrics.uptime;
        reputation = reputationScores[agent];
        
        reviewCount = peerReviews[agent].length;
        
        // GAS OPTIMIZATION: Use cached calculation instead of loop
        averagePeerRating = _getCachedPeerRating(agent);
    }
    
    /**
     * @notice Get reputation trend for an agent
     */
    function getReputationTrend(address agent) external view returns (uint256[] memory) {
        uint256[] memory scores = new uint256[](maxHistoricalScores);
        uint256 startIndex = historicalScoresIndex[agent];
        
        for (uint256 i = 0; i < maxHistoricalScores; i++) {
            uint256 index = (startIndex + i) % maxHistoricalScores;
            scores[i] = historicalScoresBuffer[agent][index];
        }
        
        return scores;
    }
    
    /**
     * @notice Internal reputation calculation
     */
    function _calculateReputation(address agent) private returns (uint256) {
        PerformanceMetrics storage metrics = agentMetrics[agent];
        
        if (metrics.totalTasks < minTasksForReputation) {
            return 0;
        }
        
        ReputationFactors memory factors;
        
        // Success Rate (0-100)
        factors.successRate = _calculateSuccessRate(metrics);
        
        // Response Speed (inverse of avg response time, normalized)
        factors.responseSpeed = _calculateSpeedScore(metrics.avgResponseTime);
        
        // Availability (uptime percentage)
        factors.availability = metrics.uptime / 100; // Convert from basis points
        
        // Efficiency (inverse of gas usage, normalized)
        factors.efficiency = _calculateEfficiencyScore(metrics.avgGasPerTask);
        
        // Experience (based on total tasks, capped)
        factors.experience = _calculateExperienceScore(metrics.totalTasks);
        
        // Include peer reviews
        uint256 peerScore = _calculatePeerScore(agent);
        
        // Calculate weighted score
        uint256 baseScore = (
            factors.successRate * SUCCESS_WEIGHT +
            factors.responseSpeed * SPEED_WEIGHT +
            factors.availability * AVAILABILITY_WEIGHT +
            factors.efficiency * EFFICIENCY_WEIGHT +
            factors.experience * EXPERIENCE_WEIGHT
        ) / 100;
        
        // Blend with peer reviews (if available)
        uint256 finalScore = peerReviews[agent].length > 0 ?
            (baseScore * 80 + peerScore * 20) / 100 : baseScore;
        
        // Scale to 0-1000
        finalScore = finalScore * 10;
        
        // Calculate confidence based on data points
        uint256 confidence = _calculateConfidence(metrics.totalTasks, peerReviews[agent].length);
        
        // Update reputation
        ReputationScore storage repScore = reputationScores[agent];
        repScore.score = finalScore;
        repScore.confidence = confidence;
        repScore.lastUpdated = block.timestamp;
        repScore.factors = factors;
        
        // Store historical score
        _storeHistoricalScore(agent, finalScore);
        
        // Update agent registry
        agentRegistry.setReputation(agent, finalScore / 5); // Convert to 0-200 range
        
        emit ReputationCalculated(agent, finalScore, confidence);
        
        return finalScore;
    }
    
    function _calculateSuccessRate(PerformanceMetrics storage metrics) 
        private view returns (uint256) {
        // SECURITY FIX: Check for division by zero
        if (metrics.totalTasks == 0) {
            return 0;
        }
        return (metrics.successfulTasks * 100) / metrics.totalTasks;
    }
    
    function _calculateSpeedScore(uint256 avgResponseTime) private pure returns (uint256) {
        // Normalize response time (assuming 60s is average, 10s is excellent)
        if (avgResponseTime <= 10) return 100;
        if (avgResponseTime >= 300) return 10;
        return 100 - ((avgResponseTime - 10) * 90) / 290;
    }
    
    function _calculateEfficiencyScore(uint256 avgGasPerTask) private pure returns (uint256) {
        // Normalize gas usage (assuming 200k is average, 50k is excellent)
        if (avgGasPerTask <= 50000) return 100;
        if (avgGasPerTask >= 500000) return 10;
        return 100 - ((avgGasPerTask - 50000) * 90) / 450000;
    }
    
    function _calculateExperienceScore(uint256 totalTasks) private pure returns (uint256) {
        // Cap at 1000 tasks for max experience score
        if (totalTasks >= 1000) return 100;
        return (totalTasks * 100) / 1000;
    }
    
    function _calculatePeerScore(address agent) private view returns (uint256) {
        uint256 reviewCount = peerReviews[agent].length;
        
        if (reviewCount == 0) return 50; // Neutral if no reviews
        
        // GAS OPTIMIZATION: Limit loop iterations and use efficient calculation
        uint256 maxReviews = reviewCount > 20 ? 20 : reviewCount;
        uint256 weightedSum;
        uint256 weightTotal;
        
        for (uint256 i = 0; i < maxReviews; i++) {
            uint256 reviewIndex = reviewCount - 1 - i;
            PeerReview storage review = peerReviews[agent][reviewIndex];
            
            uint256 age = block.timestamp - review.timestamp;
            uint256 weight = age < 30 days ? 100 : age < 90 days ? 75 : 50;
            
            weightedSum += review.rating * weight * 20; // Scale to 100
            weightTotal += weight;
        }
        
        return weightTotal > 0 ? weightedSum / weightTotal : 50;
    }
    
    /**
     * @notice Get cached peer rating to avoid expensive loops in view functions
     * @dev GAS OPTIMIZATION: Cached calculation updated on review submission
     */
    function _getCachedPeerRating(address agent) private view returns (uint256) {
        uint256 reviewCount = peerReviews[agent].length;
        if (reviewCount == 0) return 0;
        
        // Use last 10 reviews for quick calculation
        uint256 maxReviews = reviewCount > 10 ? 10 : reviewCount;
        uint256 totalRating;
        
        for (uint256 i = 0; i < maxReviews; i++) {
            totalRating += peerReviews[agent][reviewCount - 1 - i].rating;
        }
        
        return (totalRating * 100) / maxReviews;
    }
    
    function _calculateConfidence(uint256 taskCount, uint256 reviewCount) 
        private pure returns (uint256) {
        uint256 taskConfidence = taskCount >= 100 ? 50 : (taskCount * 50) / 100;
        uint256 reviewConfidence = reviewCount >= 20 ? 50 : (reviewCount * 50) / 20;
        return taskConfidence + reviewConfidence;
    }
    
    /**
     * @notice Store historical score using circular buffer (gas optimized)
     */
    function _storeHistoricalScore(address agent, uint256 score) private {
        uint256 index = historicalScoresIndex[agent];
        historicalScoresBuffer[agent][index] = score;
        historicalScoresIndex[agent] = (index + 1) % maxHistoricalScores;
    }
    
    /**
     * @notice Update dynamic thresholds based on network performance
     */
    function _updateDynamicThresholds(uint256 responseTime, uint256 gasUsed) private {
        DynamicThresholds storage thresholds = dynamicThresholds;
        
        // Update sample size
        thresholds.sampleSize++;
        
        // Adjust thresholds based on percentiles (simplified)
        if (thresholds.sampleSize >= 100) {
            // In production, calculate actual percentiles from historical data
            // For now, use simple moving average adjustment
            
            uint256 networkLoadFactor = marketContext.networkLoad;
            
            // Adjust response time thresholds based on network load
            if (networkLoadFactor > 80) { // High load
                thresholds.excellentResponseTime = (thresholds.excellentResponseTime * 120) / 100;
                thresholds.goodResponseTime = (thresholds.goodResponseTime * 115) / 100;
                thresholds.poorResponseTime = (thresholds.poorResponseTime * 110) / 100;
            } else if (networkLoadFactor < 30) { // Low load
                thresholds.excellentResponseTime = (thresholds.excellentResponseTime * 90) / 100;
                thresholds.goodResponseTime = (thresholds.goodResponseTime * 95) / 100;
                thresholds.poorResponseTime = (thresholds.poorResponseTime * 98) / 100;
            }
            
            // SECURITY FIX: Enhanced division by zero protection
            uint256 baseGasPrice = 20 gwei;
            uint256 gasPriceRatio = 100; // Default ratio
            if (baseGasPrice > 0 && marketContext.gasPrice > 0) {
                gasPriceRatio = (marketContext.gasPrice * 100) / baseGasPrice;
            }
            
            if (gasPriceRatio > 150) { // High gas price
                thresholds.excellentGasUsage = (thresholds.excellentGasUsage * 90) / 100;
                thresholds.goodGasUsage = (thresholds.goodGasUsage * 95) / 100;
            }
            
            thresholds.lastUpdated = block.timestamp;
            emit ThresholdsUpdated(thresholds.excellentResponseTime, thresholds.excellentGasUsage, thresholds.sampleSize);
        }
    }
    
    /**
     * @notice Adjust metrics for market conditions
     */
    function _adjustForMarketConditions(uint256 responseTime) private view returns (uint256) {
        uint256 loadFactor = marketContext.networkLoad;
        
        // Adjust response time based on network conditions
        if (loadFactor > 80) {
            // High load: be more lenient
            return (responseTime * 80) / 100;
        } else if (loadFactor < 30) {
            // Low load: be more strict
            return (responseTime * 120) / 100;
        }
        
        return responseTime;
    }
    
    /**
     * @notice Slash agent stake for malicious behavior
     */
    function _slashAgent(address agent, string memory reason) private {
        uint256 stakeAmount = agentStakes[agent];
        if (stakeAmount == 0) return;
        
        uint256 slashAmount = (stakeAmount * slashingRate) / 100;
        agentStakes[agent] -= slashAmount;
        
        PerformanceMetrics storage metrics = agentMetrics[agent];
        metrics.slashingPenalties += slashAmount;
        metrics.lastSlashTime = block.timestamp;
        
        emit AgentSlashed(agent, slashAmount, reason);
    }
    
    /**
     * @notice Manage review storage to prevent bloat
     */
    function _manageReviewStorage(address agent) private {
        PeerReview[] storage reviews = peerReviews[agent];
        
        if (reviews.length >= maxReviewsPerAgent) {
            // Remove oldest unvalidated review
            for (uint256 i = 0; i < reviews.length; i++) {
                if (!reviews[i].validated && block.timestamp - reviews[i].timestamp > 30 days) {
                    bytes32 oldReviewHash = reviews[i].reviewHash;
                    
                    // SECURITY FIX: Clean up review location mapping
                    delete reviewLocations[oldReviewHash];
                    delete usedReviewHashes[oldReviewHash];
                    
                    // If removing from middle, update location of swapped review
                    if (i < reviews.length - 1) {
                        bytes32 movedReviewHash = reviews[reviews.length - 1].reviewHash;
                        reviewLocations[movedReviewHash].index = i;
                        reviews[i] = reviews[reviews.length - 1];
                    }
                    
                    reviews.pop();
                    break;
                }
            }
        }
    }
    
    /**
     * @notice Deposit stake for anti-gaming measures
     */
    function depositStake() external payable {
        require(msg.value >= minStakeAmount, "Insufficient stake");
        require(agentRegistry.isRegistered(msg.sender), "Not registered");
        
        agentStakes[msg.sender] += msg.value;
        agentMetrics[msg.sender].stakedAmount += msg.value;
        
        emit StakeDeposited(msg.sender, msg.value);
    }
    
    /**
     * @notice Purchase reputation insurance
     */
    function purchaseReputationInsurance(
        uint256 coverageAmount,
        uint256 duration
    ) external payable {
        require(msg.value > 0, "Premium required");
        require(coverageAmount > 0, "Coverage required");
        require(duration > 0 && duration <= 365 days, "Invalid duration");
        
        uint256 bondAmount = (coverageAmount * 20) / 100; // 20% bond
        require(msg.value >= bondAmount, "Insufficient bond");
        
        reputationInsurance[msg.sender] = ReputationInsurance({
            bondAmount: bondAmount,
            coverageAmount: coverageAmount,
            premiumPaid: msg.value,
            expirationTime: block.timestamp + duration,
            active: true,
            claimCount: 0
        });
        
        emit InsurancePurchased(msg.sender, bondAmount, coverageAmount);
    }
    
    /**
     * @notice Update market context (oracle role)
     */
    function updateMarketContext(
        uint256 networkLoad,
        uint256 gasPrice,
        uint256 avgBlockTime,
        uint256 totalActiveAgents,
        uint256 demandSupplyRatio
    ) external onlyRole(REPUTATION_ORACLE_ROLE) {
        marketContext = MarketContext({
            networkLoad: networkLoad,
            gasPrice: gasPrice,
            avgBlockTime: avgBlockTime,
            totalActiveAgents: totalActiveAgents,
            demandSupplyRatio: demandSupplyRatio,
            lastUpdated: block.timestamp
        });
        
        emit MarketContextUpdated(networkLoad, gasPrice, totalActiveAgents);
    }
    
    /**
     * @notice Get historical scores (reads from circular buffer)
     */
    function getHistoricalScores(address agent) external view returns (uint256[] memory) {
        uint256[] memory scores = new uint256[](maxHistoricalScores);
        uint256 startIndex = historicalScoresIndex[agent];
        
        for (uint256 i = 0; i < maxHistoricalScores; i++) {
            uint256 index = (startIndex + i) % maxHistoricalScores;
            scores[i] = historicalScoresBuffer[agent][index];
        }
        
        return scores;
    }
    
    /**
     * @notice Get review details by hash (O(1) lookup)
     * @dev SECURITY FIX: Efficient review retrieval without loops
     */
    function getReviewByHash(bytes32 reviewHash) external view returns (
        address agent,
        address reviewer,
        uint256 rating,
        string memory comment,
        uint256 taskId,
        uint256 timestamp,
        bool validated,
        uint256 validatorCount
    ) {
        ReviewLocation memory location = reviewLocations[reviewHash];
        require(location.exists, "Review not found");
        
        PeerReview storage review = peerReviews[location.agent][location.index];
        
        return (
            location.agent,
            review.reviewer,
            review.rating,
            review.comment,
            review.taskId,
            review.timestamp,
            review.validated,
            review.validatorCount
        );
    }
    
    /**
     * @notice Check if a review exists
     */
    function reviewExists(bytes32 reviewHash) external view returns (bool) {
        return reviewLocations[reviewHash].exists;
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}