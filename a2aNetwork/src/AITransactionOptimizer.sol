// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "./AgentRegistry.sol";
import "./MultiSigPausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title AITransactionOptimizer
 * @dev AI-powered transaction optimization for smart contract operations
 * Provides intelligent gas pricing, batch processing, and transaction scheduling
 * with machine learning-driven optimization strategies.
 */
contract AITransactionOptimizer is MultiSigPausable, ReentrancyGuard {
    // ML-based transaction optimization parameters
    struct OptimizationModel {
        uint256 baseGasPrice;
        uint256 priority;
        uint256 congestionMultiplier;
        uint256 networkLoadFactor;
        uint256 lastUpdated;
        bool isActive;
    }
    
    struct TransactionBatch {
        bytes32 batchId;
        address[] targets;
        bytes[] calldata;
        uint256[] values;
        uint256 estimatedGas;
        uint256 optimalGasPrice;
        uint256 scheduledTime;
        bool executed;
        uint256 actualGasUsed;
    }
    
    struct GasOptimization {
        uint256 predictedPrice;
        uint256 confidence;
        uint256 savingsPotential;
        uint256 executionWindow;
        string strategy;
    }
    
    // AI optimization models for different transaction types
    mapping(bytes32 => OptimizationModel) public optimizationModels;
    mapping(bytes32 => TransactionBatch) public transactionBatches;
    mapping(address => uint256) public agentOptimizationScores;
    mapping(uint256 => uint256) public networkCongestionHistory;
    
    // Dynamic pricing parameters updated by AI
    uint256 public baseFeeMultiplier = 110; // 110% of base fee by default
    uint256 public priorityFeeMultiplier = 120; // 120% priority fee by default
    uint256 public batchSizeOptimal = 10;
    uint256 public congestionThreshold = 80; // 80% network usage
    
    // Historical data for ML training
    uint256[] public gasUsageHistory;
    uint256[] public gasPriceHistory;
    uint256[] public networkLoadHistory;
    uint256 public currentBlockLoad;
    
    // AI-driven optimization strategies
    bytes32 public constant FAST_EXECUTION = keccak256("FAST_EXECUTION");
    bytes32 public constant COST_OPTIMIZATION = keccak256("COST_OPTIMIZATION");
    bytes32 public constant BALANCED_APPROACH = keccak256("BALANCED_APPROACH");
    bytes32 public constant PREDICTIVE_SCHEDULING = keccak256("PREDICTIVE_SCHEDULING");
    
    // Events for AI model updates
    event OptimizationModelUpdated(bytes32 indexed modelType, uint256 baseGasPrice, uint256 priority);
    event TransactionBatchCreated(bytes32 indexed batchId, uint256 estimatedGas, uint256 scheduledTime);
    event TransactionBatchExecuted(bytes32 indexed batchId, uint256 actualGasUsed, uint256 savings);
    event GasOptimizationApplied(address indexed agent, uint256 oldPrice, uint256 newPrice, string strategy);
    event NetworkCongestionUpdated(uint256 congestionLevel, uint256 recommendedDelay);
    event AIModelRetrained(bytes32 indexed modelType, uint256 accuracy, uint256 trainingDataPoints);
    
    AgentRegistry public immutable registry;
    
    constructor(address _registry, uint256 _requiredConfirmations) 
        MultiSigPausable(_requiredConfirmations) {
        registry = AgentRegistry(_registry);
        _initializeOptimizationModels();
    }
    
    modifier onlyRegisteredAgent() {
        AgentRegistry.Agent memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not registered or inactive");
        _;
    }
    
    /**
     * @notice Get AI-optimized gas price for a transaction
     * @param transactionType Type of transaction for optimization
     * @param priority Desired priority level (1-4, higher is more urgent)
     * @param dataSize Size of transaction data in bytes
     * @return optimization Gas optimization recommendations
     */
    function getOptimalGasPrice(
        bytes32 transactionType,
        uint256 priority,
        uint256 dataSize
    ) external view returns (GasOptimization memory optimization) {
        OptimizationModel memory model = optimizationModels[transactionType];
        
        if (!model.isActive) {
            model = optimizationModels[BALANCED_APPROACH];
        }
        
        // AI-driven gas price calculation
        uint256 basePrice = _calculateBasePrice(model, priority);
        uint256 congestionAdjustment = _calculateCongestionAdjustment();
        uint256 dataAdjustment = _calculateDataSizeAdjustment(dataSize);
        
        uint256 predictedPrice = basePrice * congestionAdjustment * dataAdjustment / 10000;
        
        // Calculate confidence based on recent prediction accuracy
        uint256 confidence = _calculatePredictionConfidence(transactionType);
        
        // Calculate potential savings compared to standard pricing
        uint256 standardPrice = block.basefee * 150 / 100; // 150% of base fee
        uint256 savingsPotential = standardPrice > predictedPrice ? 
            standardPrice - predictedPrice : 0;
        
        // Determine optimal execution window
        uint256 executionWindow = _calculateOptimalExecutionWindow(priority);
        
        optimization = GasOptimization({
            predictedPrice: predictedPrice,
            confidence: confidence,
            savingsPotential: savingsPotential,
            executionWindow: executionWindow,
            strategy: _getOptimizationStrategy(transactionType)
        });
    }
    
    /**
     * @notice Create an AI-optimized transaction batch
     * @param targets Array of target contract addresses
     * @param calldataArray Array of function call data
     * @param values Array of ETH values for each call
     * @param priority Batch execution priority
     * @return batchId Unique identifier for the created batch
     */
    function createOptimizedBatch(
        address[] memory targets,
        bytes[] memory calldataArray,
        uint256[] memory values,
        uint256 priority
    ) external onlyRegisteredAgent nonReentrant returns (bytes32 batchId) {
        require(targets.length == calldataArray.length, "Array length mismatch");
        require(targets.length == values.length, "Value array length mismatch");
        require(targets.length > 0, "Empty batch");
        require(targets.length <= batchSizeOptimal * 2, "Batch too large");
        
        batchId = keccak256(abi.encodePacked(
            msg.sender, 
            targets, 
            block.timestamp, 
            block.difficulty
        ));
        
        // Estimate gas for the batch
        uint256 estimatedGas = _estimateBatchGas(targets, calldataArray, values);
        
        // Calculate optimal gas price using AI model
        GasOptimization memory gasOpt = this.getOptimalGasPrice(
            BALANCED_APPROACH,
            priority,
            _calculateTotalDataSize(calldataArray)
        );
        
        // Schedule execution time based on network conditions
        uint256 scheduledTime = _calculateOptimalScheduleTime(priority, gasOpt.executionWindow);
        
        transactionBatches[batchId] = TransactionBatch({
            batchId: batchId,
            targets: targets,
            calldata: calldataArray,
            values: values,
            estimatedGas: estimatedGas,
            optimalGasPrice: gasOpt.predictedPrice,
            scheduledTime: scheduledTime,
            executed: false,
            actualGasUsed: 0
        });
        
        emit TransactionBatchCreated(batchId, estimatedGas, scheduledTime);
    }
    
    /**
     * @notice Execute an optimized transaction batch
     * @param batchId Identifier of the batch to execute
     */
    function executeBatch(bytes32 batchId) external nonReentrant {
        TransactionBatch storage batch = transactionBatches[batchId];
        require(!batch.executed, "Batch already executed");
        require(block.timestamp >= batch.scheduledTime, "Batch not ready for execution");
        
        uint256 gasStart = gasleft();
        bool[] memory results = new bool[](batch.targets.length);
        
        // Execute all transactions in the batch
        for (uint256 i = 0; i < batch.targets.length; i++) {
            (bool success,) = batch.targets[i].call{value: batch.values[i]}(batch.calldata[i]);
            results[i] = success;
        }
        
        uint256 gasUsed = gasStart - gasleft();
        batch.actualGasUsed = gasUsed;
        batch.executed = true;
        
        // Calculate savings achieved
        uint256 standardGasCost = gasUsed * block.basefee * 150 / 100;
        uint256 optimizedGasCost = gasUsed * batch.optimalGasPrice;
        uint256 savings = standardGasCost > optimizedGasCost ? 
            standardGasCost - optimizedGasCost : 0;
        
        // Update optimization score for the agent
        _updateAgentOptimizationScore(msg.sender, gasUsed, batch.estimatedGas, savings);
        
        // Record data for ML model improvement
        _recordOptimizationData(batchId, gasUsed, savings);
        
        emit TransactionBatchExecuted(batchId, gasUsed, savings);
    }
    
    /**
     * @notice Update optimization models with new AI parameters (admin only)
     * @param modelType Type of optimization model to update
     * @param baseGasPrice New base gas price parameter
     * @param priority Priority multiplier
     * @param congestionMultiplier Network congestion adjustment factor
     * @param networkLoadFactor Network load consideration factor
     */
    function updateOptimizationModel(
        bytes32 modelType,
        uint256 baseGasPrice,
        uint256 priority,
        uint256 congestionMultiplier,
        uint256 networkLoadFactor
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(baseGasPrice > 0, "Invalid base gas price");
        require(priority > 0 && priority <= 1000, "Invalid priority");
        require(congestionMultiplier >= 100 && congestionMultiplier <= 500, "Invalid congestion multiplier");
        require(networkLoadFactor >= 100 && networkLoadFactor <= 300, "Invalid network load factor");
        
        optimizationModels[modelType] = OptimizationModel({
            baseGasPrice: baseGasPrice,
            priority: priority,
            congestionMultiplier: congestionMultiplier,
            networkLoadFactor: networkLoadFactor,
            lastUpdated: block.timestamp,
            isActive: true
        });
        
        emit OptimizationModelUpdated(modelType, baseGasPrice, priority);
    }
    
    /**
     * @notice Update network congestion metrics (called by oracle or automated system)
     * @param congestionLevel Current network congestion percentage (0-100)
     * @param blockLoad Current block utilization percentage
     */
    function updateNetworkMetrics(
        uint256 congestionLevel,
        uint256 blockLoad
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(congestionLevel <= 100, "Invalid congestion level");
        require(blockLoad <= 100, "Invalid block load");
        
        currentBlockLoad = blockLoad;
        networkCongestionHistory[block.timestamp] = congestionLevel;
        
        // Update historical data for ML training
        if (gasUsageHistory.length >= 1000) {
            // Remove oldest entry to maintain fixed size
            for (uint256 i = 0; i < gasUsageHistory.length - 1; i++) {
                gasUsageHistory[i] = gasUsageHistory[i + 1];
                gasPriceHistory[i] = gasPriceHistory[i + 1];
                networkLoadHistory[i] = networkLoadHistory[i + 1];
            }
            gasUsageHistory[gasUsageHistory.length - 1] = block.gaslimit - gasleft();
            gasPriceHistory[gasPriceHistory.length - 1] = tx.gasprice;
            networkLoadHistory[networkLoadHistory.length - 1] = blockLoad;
        } else {
            gasUsageHistory.push(block.gaslimit - gasleft());
            gasPriceHistory.push(tx.gasprice);
            networkLoadHistory.push(blockLoad);
        }
        
        uint256 recommendedDelay = congestionLevel > congestionThreshold ? 
            (congestionLevel - congestionThreshold) * 30 : 0; // 30 seconds per % over threshold
        
        emit NetworkCongestionUpdated(congestionLevel, recommendedDelay);
    }
    
    /**
     * @notice Train AI models with accumulated data (admin only)
     * @param modelType Type of model to retrain
     * @return accuracy Estimated accuracy of the retrained model
     */
    function retrainModel(bytes32 modelType) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
        returns (uint256 accuracy) 
    {
        require(gasUsageHistory.length >= 100, "Insufficient training data");
        
        // Simulate ML model retraining using historical data
        uint256 trainingDataPoints = gasUsageHistory.length;
        
        // Calculate model accuracy based on historical prediction vs actual performance
        uint256 totalError = 0;
        uint256 validPredictions = 0;
        
        for (uint256 i = 10; i < trainingDataPoints; i++) {
            uint256 predicted = _simulatePrediction(i);
            uint256 actual = gasUsageHistory[i];
            
            if (actual > 0) {
                uint256 error = predicted > actual ? 
                    ((predicted - actual) * 100) / actual :
                    ((actual - predicted) * 100) / actual;
                totalError += error;
                validPredictions++;
            }
        }
        
        accuracy = validPredictions > 0 ? 
            (100 * validPredictions - totalError) / validPredictions : 0;
        
        // Update model parameters based on retraining results
        if (accuracy > 70) {
            OptimizationModel storage model = optimizationModels[modelType];
            model.lastUpdated = block.timestamp;
            // In a real implementation, this would update model parameters
            // based on the ML training results
        }
        
        emit AIModelRetrained(modelType, accuracy, trainingDataPoints);
    }
    
    /**
     * @notice Get agent's optimization performance score
     * @param agent Address of the agent
     * @return score Optimization score (0-1000, higher is better)
     */
    function getAgentOptimizationScore(address agent) external view returns (uint256 score) {
        return agentOptimizationScores[agent];
    }
    
    /**
     * @notice Get historical network metrics for analysis
     * @param timeRange Number of recent data points to return
     * @return gasUsage Recent gas usage data
     * @return gasPrices Recent gas price data
     * @return networkLoad Recent network load data
     */
    function getHistoricalMetrics(uint256 timeRange) 
        external 
        view 
        returns (
            uint256[] memory gasUsage,
            uint256[] memory gasPrices,
            uint256[] memory networkLoad
        ) 
    {
        uint256 dataPoints = gasUsageHistory.length;
        uint256 returnCount = timeRange > dataPoints ? dataPoints : timeRange;
        
        gasUsage = new uint256[](returnCount);
        gasPrices = new uint256[](returnCount);
        networkLoad = new uint256[](returnCount);
        
        for (uint256 i = 0; i < returnCount; i++) {
            uint256 index = dataPoints - returnCount + i;
            gasUsage[i] = gasUsageHistory[index];
            gasPrices[i] = gasPriceHistory[index];
            networkLoad[i] = networkLoadHistory[index];
        }
    }
    
    // Private helper functions
    function _initializeOptimizationModels() private {
        // Initialize default optimization models
        optimizationModels[FAST_EXECUTION] = OptimizationModel({
            baseGasPrice: 20 gwei,
            priority: 200,
            congestionMultiplier: 150,
            networkLoadFactor: 120,
            lastUpdated: block.timestamp,
            isActive: true
        });
        
        optimizationModels[COST_OPTIMIZATION] = OptimizationModel({
            baseGasPrice: 5 gwei,
            priority: 50,
            congestionMultiplier: 110,
            networkLoadFactor: 100,
            lastUpdated: block.timestamp,
            isActive: true
        });
        
        optimizationModels[BALANCED_APPROACH] = OptimizationModel({
            baseGasPrice: 10 gwei,
            priority: 100,
            congestionMultiplier: 125,
            networkLoadFactor: 110,
            lastUpdated: block.timestamp,
            isActive: true
        });
        
        optimizationModels[PREDICTIVE_SCHEDULING] = OptimizationModel({
            baseGasPrice: 8 gwei,
            priority: 75,
            congestionMultiplier: 120,
            networkLoadFactor: 105,
            lastUpdated: block.timestamp,
            isActive: true
        });
    }
    
    function _calculateBasePrice(OptimizationModel memory model, uint256 priority) 
        private 
        view 
        returns (uint256) 
    {
        uint256 basePrice = model.baseGasPrice;
        
        // Adjust for current base fee
        uint256 currentBaseFee = block.basefee;
        if (currentBaseFee > 0) {
            basePrice = (basePrice + currentBaseFee) / 2; // Average with current base fee
        }
        
        // Apply priority multiplier
        basePrice = basePrice * (100 + priority * model.priority / 100) / 100;
        
        return basePrice;
    }
    
    function _calculateCongestionAdjustment() private view returns (uint256) {
        // Use recent congestion data to adjust pricing
        uint256 recentCongestion = _getRecentAverageCongestion();
        
        if (recentCongestion > congestionThreshold) {
            return 100 + (recentCongestion - congestionThreshold);
        }
        
        return 100; // No adjustment needed
    }
    
    function _calculateDataSizeAdjustment(uint256 dataSize) private pure returns (uint256) {
        // Larger transactions need slightly higher gas prices for inclusion
        if (dataSize > 10000) {
            return 110; // 10% increase
        } else if (dataSize > 5000) {
            return 105; // 5% increase
        }
        return 100; // No adjustment
    }
    
    function _calculatePredictionConfidence(bytes32 modelType) private view returns (uint256) {
        OptimizationModel memory model = optimizationModels[modelType];
        
        // Confidence decreases over time since last model update
        uint256 timeSinceUpdate = block.timestamp - model.lastUpdated;
        uint256 maxAge = 1 days;
        
        if (timeSinceUpdate >= maxAge) {
            return 50; // Minimum confidence
        }
        
        return 100 - (timeSinceUpdate * 50 / maxAge);
    }
    
    function _calculateOptimalExecutionWindow(uint256 priority) private view returns (uint256) {
        if (priority >= 4) return 0; // Execute immediately
        if (priority == 3) return 300; // 5 minutes
        if (priority == 2) return 1800; // 30 minutes
        return 3600; // 1 hour for low priority
    }
    
    function _getOptimizationStrategy(bytes32 transactionType) private pure returns (string memory) {
        if (transactionType == FAST_EXECUTION) return "fast";
        if (transactionType == COST_OPTIMIZATION) return "economical";
        if (transactionType == PREDICTIVE_SCHEDULING) return "predictive";
        return "balanced";
    }
    
    function _estimateBatchGas(
        address[] memory targets,
        bytes[] memory calldataArray,
        uint256[] memory values
    ) private view returns (uint256) {
        uint256 totalGas = 21000; // Base transaction cost
        
        for (uint256 i = 0; i < targets.length; i++) {
            // Estimate gas for each call (simplified estimation)
            totalGas += 25000 + calldataArray[i].length * 16;
            if (values[i] > 0) totalGas += 9000; // ETH transfer cost
        }
        
        return totalGas;
    }
    
    function _calculateTotalDataSize(bytes[] memory calldataArray) private pure returns (uint256) {
        uint256 totalSize = 0;
        for (uint256 i = 0; i < calldataArray.length; i++) {
            totalSize += calldataArray[i].length;
        }
        return totalSize;
    }
    
    function _calculateOptimalScheduleTime(uint256 priority, uint256 executionWindow) 
        private 
        view 
        returns (uint256) 
    {
        if (priority >= 4 || executionWindow == 0) {
            return block.timestamp; // Execute immediately
        }
        
        // Schedule for optimal network conditions within the window
        uint256 avgCongestion = _getRecentAverageCongestion();
        if (avgCongestion < congestionThreshold) {
            return block.timestamp; // Network is good, execute now
        }
        
        // Delay execution to potentially better conditions
        return block.timestamp + (executionWindow / 2);
    }
    
    function _getRecentAverageCongestion() private view returns (uint256) {
        uint256 sum = 0;
        uint256 count = 0;
        uint256 cutoff = block.timestamp - 1 hours;
        
        // This is a simplified implementation
        // In practice, you'd iterate through recent congestion data
        return currentBlockLoad; // Fallback to current block load
    }
    
    function _updateAgentOptimizationScore(
        address agent,
        uint256 actualGas,
        uint256 estimatedGas,
        uint256 savings
    ) private {
        uint256 currentScore = agentOptimizationScores[agent];
        
        // Calculate accuracy bonus
        uint256 accuracyBonus = 0;
        if (estimatedGas > 0) {
            uint256 accuracy = actualGas > estimatedGas ?
                (estimatedGas * 100) / actualGas :
                (actualGas * 100) / estimatedGas;
            
            if (accuracy > 90) accuracyBonus = 10;
            else if (accuracy > 80) accuracyBonus = 5;
        }
        
        // Calculate savings bonus
        uint256 savingsBonus = savings / 1 gwei; // 1 point per gwei saved
        if (savingsBonus > 50) savingsBonus = 50; // Cap at 50 points
        
        // Update score (weighted average with decay)
        uint256 newScore = (currentScore * 9 + accuracyBonus + savingsBonus) / 10;
        if (newScore > 1000) newScore = 1000;
        
        agentOptimizationScores[agent] = newScore;
    }
    
    function _recordOptimizationData(bytes32 batchId, uint256 gasUsed, uint256 savings) private {
        // Record data for future ML model training
        // This would typically feed into an off-chain ML pipeline
    }
    
    function _simulatePrediction(uint256 index) private view returns (uint256) {
        // Simplified prediction simulation using moving average
        if (index < 5) return gasUsageHistory[index];
        
        uint256 sum = 0;
        for (uint256 i = index - 5; i < index; i++) {
            sum += gasUsageHistory[i];
        }
        
        return sum / 5;
    }
}