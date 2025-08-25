// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title IntelligentGasManager
 * @notice AI-powered gas optimization system for smart contract transactions
 * @dev Implements predictive gas pricing, transaction bundling, and ML-based optimization
 */
contract IntelligentGasManager is AccessControlUpgradeable, UUPSUpgradeable {
    bytes32 public constant GAS_OPTIMIZER_ROLE = keccak256("GAS_OPTIMIZER_ROLE");
    bytes32 public constant AI_ORACLE_ROLE = keccak256("AI_ORACLE_ROLE");
    
    struct GasPrediction {
        uint256 predictedGasPrice;
        uint256 confidence; // 0-100
        uint256 timestamp;
        uint256 networkLoad;
        uint256 validUntil;
    }
    
    struct TransactionProfile {
        bytes4 functionSelector;
        string contractName;
        uint256 averageGasUsed;
        uint256 maxGasUsed;
        uint256 minGasUsed;
        uint256 executionCount;
        uint256 successRate; // out of 100
        uint256 lastUpdated;
    }
    
    struct GasOptimizationStrategy {
        uint256 strategyId;
        string name;
        uint256 expectedSavings; // percentage
        bool requiresBundling;
        uint256 minTransactionsForBundle;
        uint256 maxBundleSize;
        bytes strategyParams;
    }
    
    struct NetworkConditions {
        uint256 currentGasPrice;
        uint256 blockUtilization; // percentage
        uint256 pendingTransactions;
        uint256 avgBlockTime;
        uint256 congestionLevel; // 1-10
        uint256 timestamp;
    }
    
    // Storage
    mapping(bytes32 => GasPrediction) public gasPredictions;
    mapping(bytes4 => TransactionProfile) public transactionProfiles;
    mapping(uint256 => GasOptimizationStrategy) public optimizationStrategies;
    mapping(address => uint256) public contractGasBudgets;
    mapping(address => uint256) public gasSpent;
    
    NetworkConditions public currentNetworkConditions;
    GasPrediction public latestPrediction;
    
    uint256 public predictionUpdateInterval = 30 seconds;
    uint256 public maxPredictionAge = 5 minutes;
    uint256 public gasBufferPercentage = 10; // 10% safety buffer
    
    // Events
    event GasPredictionUpdated(uint256 predictedPrice, uint256 confidence, uint256 networkLoad);
    event TransactionOptimized(address indexed contract_, bytes4 funcSelector, uint256 originalGas, uint256 optimizedGas);
    event BundleCreated(bytes32 indexed bundleId, uint256 transactionCount, uint256 gasSaved);
    event NetworkConditionsUpdated(uint256 gasPrice, uint256 congestion, uint256 utilization);
    event GasBudgetExceeded(address indexed contract_, uint256 budgetLimit, uint256 actualSpent);
    
    function initialize(address admin) external initializer {
        __AccessControl_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(GAS_OPTIMIZER_ROLE, admin);
        
        // Initialize default optimization strategies
        _initializeDefaultStrategies();
    }
    
    /**
     * @notice AI-powered gas price prediction
     * @dev Updates gas price prediction based on ML analysis of network conditions
     */
    function updateGasPrediction(
        uint256 predictedGasPrice,
        uint256 confidence,
        uint256 networkLoad,
        uint256 validDuration
    ) external onlyRole(AI_ORACLE_ROLE) {
        require(confidence <= 100, "Confidence must be 0-100");
        require(validDuration >= 30 seconds && validDuration <= 1 hours, "Invalid duration");
        
        bytes32 predictionId = keccak256(abi.encodePacked(block.timestamp, predictedGasPrice));
        
        GasPrediction storage prediction = gasPredictions[predictionId];
        prediction.predictedGasPrice = predictedGasPrice;
        prediction.confidence = confidence;
        prediction.timestamp = block.timestamp;
        prediction.networkLoad = networkLoad;
        prediction.validUntil = block.timestamp + validDuration;
        
        // Update latest prediction if confidence is higher
        if (confidence >= latestPrediction.confidence || 
            block.timestamp > latestPrediction.validUntil) {
            latestPrediction = prediction;
        }
        
        emit GasPredictionUpdated(predictedGasPrice, confidence, networkLoad);
    }
    
    /**
     * @notice Get optimized gas parameters for a transaction
     * @dev Returns AI-optimized gas price and limit based on function and network conditions
     */
    function getOptimizedGasParameters(
        address contractAddress,
        bytes4 functionSelector,
        bytes calldata callData
    ) external view returns (
        uint256 gasPrice,
        uint256 gasLimit,
        uint256 maxFeePerGas,
        uint256 maxPriorityFeePerGas
    ) {
        // Get transaction profile
        TransactionProfile storage profile = transactionProfiles[functionSelector];
        
        // Base gas prediction
        if (block.timestamp <= latestPrediction.validUntil && latestPrediction.confidence >= 70) {
            gasPrice = latestPrediction.predictedGasPrice;
        } else {
            gasPrice = _fallbackGasPrice();
        }
        
        // Calculate gas limit with AI optimization
        if (profile.executionCount > 0) {
            gasLimit = _calculateOptimizedGasLimit(profile, callData);
        } else {
            gasLimit = 500000; // Conservative default for new functions
        }
        
        // EIP-1559 parameters
        (maxFeePerGas, maxPriorityFeePerGas) = _calculateEIP1559Parameters(gasPrice);
        
        return (gasPrice, gasLimit, maxFeePerGas, maxPriorityFeePerGas);
    }
    
    /**
     * @notice Update transaction execution profile
     * @dev Records actual gas usage for ML model improvement
     */
    function updateTransactionProfile(
        bytes4 functionSelector,
        string calldata contractName,
        uint256 actualGasUsed,
        bool success
    ) external onlyRole(GAS_OPTIMIZER_ROLE) {
        TransactionProfile storage profile = transactionProfiles[functionSelector];
        
        if (profile.executionCount == 0) {
            // First execution
            profile.functionSelector = functionSelector;
            profile.contractName = contractName;
            profile.averageGasUsed = actualGasUsed;
            profile.maxGasUsed = actualGasUsed;
            profile.minGasUsed = actualGasUsed;
            profile.successRate = success ? 100 : 0;
        } else {
            // Update running statistics
            profile.averageGasUsed = (profile.averageGasUsed * profile.executionCount + actualGasUsed) / 
                                   (profile.executionCount + 1);
            
            if (actualGasUsed > profile.maxGasUsed) {
                profile.maxGasUsed = actualGasUsed;
            }
            if (actualGasUsed < profile.minGasUsed) {
                profile.minGasUsed = actualGasUsed;
            }
            
            // Update success rate
            uint256 totalSuccesses = (profile.successRate * profile.executionCount) / 100;
            if (success) totalSuccesses++;
            profile.successRate = (totalSuccesses * 100) / (profile.executionCount + 1);
        }
        
        profile.executionCount++;
        profile.lastUpdated = block.timestamp;
    }
    
    /**
     * @notice Create optimized transaction bundle
     * @dev Bundles multiple transactions for gas efficiency
     */
    function createTransactionBundle(
        address[] calldata targets,
        bytes[] calldata calldatas,
        uint256[] calldata values
    ) external returns (bytes32 bundleId, uint256 estimatedGasSaved) {
        require(targets.length == calldatas.length && targets.length == values.length, "Array length mismatch");
        require(targets.length >= 2, "Bundle requires at least 2 transactions");
        
        bundleId = keccak256(abi.encodePacked(targets, calldatas, block.timestamp));
        
        // Calculate gas savings from bundling
        uint256 individualGasCost = 0;
        for (uint256 i = 0; i < targets.length; i++) {
            bytes4 selector = bytes4(calldatas[i][:4]);
            TransactionProfile storage profile = transactionProfiles[selector];
            individualGasCost += profile.averageGasUsed > 0 ? profile.averageGasUsed : 200000;
            individualGasCost += 21000; // Base transaction cost
        }
        
        uint256 bundledGasCost = individualGasCost - (21000 * (targets.length - 1)); // Save base cost
        estimatedGasSaved = individualGasCost > bundledGasCost ? 
                           individualGasCost - bundledGasCost : 0;
        
        emit BundleCreated(bundleId, targets.length, estimatedGasSaved);
        
        return (bundleId, estimatedGasSaved);
    }
    
    /**
     * @notice Update network conditions for gas optimization
     */
    function updateNetworkConditions(
        uint256 currentGasPrice_,
        uint256 blockUtilization,
        uint256 pendingTransactions,
        uint256 avgBlockTime,
        uint256 congestionLevel
    ) external onlyRole(AI_ORACLE_ROLE) {
        currentNetworkConditions = NetworkConditions({
            currentGasPrice: currentGasPrice_,
            blockUtilization: blockUtilization,
            pendingTransactions: pendingTransactions,
            avgBlockTime: avgBlockTime,
            congestionLevel: congestionLevel,
            timestamp: block.timestamp
        });
        
        emit NetworkConditionsUpdated(currentGasPrice_, congestionLevel, blockUtilization);
    }
    
    /**
     * @notice Set gas budget for a contract
     */
    function setGasBudget(address contractAddress, uint256 budgetLimit) 
        external onlyRole(DEFAULT_ADMIN_ROLE) {
        contractGasBudgets[contractAddress] = budgetLimit;
    }
    
    /**
     * @notice Check if transaction is within gas budget
     */
    function checkGasBudget(address contractAddress, uint256 gasToSpend) 
        external view returns (bool withinBudget) {
        uint256 budget = contractGasBudgets[contractAddress];
        if (budget == 0) return true; // No budget limit set
        
        return gasSpent[contractAddress] + gasToSpend <= budget;
    }
    
    /**
     * @notice Record gas expenditure
     */
    function recordGasSpent(address contractAddress, uint256 gasAmount) 
        external onlyRole(GAS_OPTIMIZER_ROLE) {
        gasSpent[contractAddress] += gasAmount;
        
        uint256 budget = contractGasBudgets[contractAddress];
        if (budget > 0 && gasSpent[contractAddress] > budget) {
            emit GasBudgetExceeded(contractAddress, budget, gasSpent[contractAddress]);
        }
    }
    
    /**
     * @notice Get transaction profile for ML analysis
     */
    function getTransactionProfilesForML(bytes4[] calldata selectors) 
        external view returns (
            uint256[][] memory profileData,
            uint256[] memory gasUsages,
            uint256[] memory successRates
        ) {
        profileData = new uint256[][](selectors.length);
        gasUsages = new uint256[](selectors.length);
        successRates = new uint256[](selectors.length);
        
        for (uint256 i = 0; i < selectors.length; i++) {
            TransactionProfile storage profile = transactionProfiles[selectors[i]];
            
            profileData[i] = new uint256[](5);
            profileData[i][0] = profile.executionCount;
            profileData[i][1] = profile.averageGasUsed;
            profileData[i][2] = profile.maxGasUsed;
            profileData[i][3] = profile.minGasUsed;
            profileData[i][4] = block.timestamp - profile.lastUpdated;
            
            gasUsages[i] = profile.averageGasUsed;
            successRates[i] = profile.successRate;
        }
        
        return (profileData, gasUsages, successRates);
    }
    
    // Internal functions
    function _calculateOptimizedGasLimit(
        TransactionProfile storage profile,
        bytes calldata callData
    ) private view returns (uint256) {
        // Base calculation on historical data
        uint256 baseLimit = profile.averageGasUsed;
        
        // Adjust for call data size (simplified model)
        uint256 callDataAdjustment = callData.length * 16; // 16 gas per byte
        baseLimit += callDataAdjustment;
        
        // Apply safety buffer
        baseLimit = (baseLimit * (100 + gasBufferPercentage)) / 100;
        
        // Cap at max observed + buffer
        uint256 maxLimit = (profile.maxGasUsed * 120) / 100;
        return baseLimit > maxLimit ? maxLimit : baseLimit;
    }
    
    function _calculateEIP1559Parameters(uint256 baseGasPrice) 
        private view returns (uint256 maxFeePerGas, uint256 maxPriorityFeePerGas) {
        // Simplified EIP-1559 calculation
        maxPriorityFeePerGas = baseGasPrice / 10; // 10% tip
        maxFeePerGas = baseGasPrice + maxPriorityFeePerGas;
        
        // Adjust based on network congestion
        if (currentNetworkConditions.congestionLevel >= 7) {
            maxPriorityFeePerGas = (maxPriorityFeePerGas * 150) / 100; // 50% increase
            maxFeePerGas = baseGasPrice + maxPriorityFeePerGas;
        }
        
        return (maxFeePerGas, maxPriorityFeePerGas);
    }
    
    function _fallbackGasPrice() private view returns (uint256) {
        // Use network conditions if available
        if (block.timestamp <= currentNetworkConditions.timestamp + 300) { // 5 minute validity
            return currentNetworkConditions.currentGasPrice;
        }
        
        // Conservative fallback
        return 20 gwei;
    }
    
    function _initializeDefaultStrategies() private {
        // Strategy 1: Bundle similar transactions
        optimizationStrategies[1] = GasOptimizationStrategy({
            strategyId: 1,
            name: "Transaction Bundling",
            expectedSavings: 15, // 15% savings
            requiresBundling: true,
            minTransactionsForBundle: 2,
            maxBundleSize: 10,
            strategyParams: ""
        });
        
        // Strategy 2: Off-peak execution
        optimizationStrategies[2] = GasOptimizationStrategy({
            strategyId: 2,
            name: "Off-Peak Scheduling",
            expectedSavings: 25, // 25% savings during low congestion
            requiresBundling: false,
            minTransactionsForBundle: 0,
            maxBundleSize: 0,
            strategyParams: abi.encode(3600) // 1 hour delay tolerance
        });
    }
    
    function _authorizeUpgrade(address newImplementation) 
        internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}