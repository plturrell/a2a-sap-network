// Web3Manager temporarily disabled for CAP startup - blockchain features not needed for launchpad
// const Web3Manager = require('../utils/Web3Manager');
const logger = require('../utils/logger');
const { validateAddress, validateRating } = require('../utils/validators');
const cacheService = require('./cacheService');

class AgentRatingService {
    constructor() {
        // this.web3Manager = new Web3Manager();
        this.reputationContract = null;
        this.agentRegistryContract = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            // Initialize contracts
            this.reputationContract = await this.web3Manager.getContract('PerformanceReputationSystem');
            this.agentRegistryContract = await this.web3Manager.getContract('AgentRegistry');
            this.initialized = true;
            logger.info('AgentRatingService initialized successfully');
        } catch (error) {
            logger.error('Failed to initialize AgentRatingService:', error);
            throw error;
        }
    }

    async getAgentDetails(agentId) {
        if (!this.initialized) await this.initialize();

        try {
            // Check cache first
            const cacheKey = `agent_details_${agentId}`;
            const cached = await cacheService.get(cacheKey);
            if (cached) return cached;

            // Get agent info from registry
            const agentInfo = await this.agentRegistryContract.methods.getAgent(agentId).call();

            // Get reputation metrics
            const metrics = await this.reputationContract.methods.getAgentMetrics(agentId).call();
            const reputation = await this.reputationContract.methods.getAgentReputation(agentId).call();

            // Calculate overall rating from reputation score (0-1000) to stars (0-5)
            const overallRating = (reputation.currentScore / 1000) * 5;

            // Get rating distribution
            const ratingDistribution = await this._getRatingDistribution(agentId);

            const details = {
                agentId: agentId,
                name: agentInfo.name || `Agent ${agentId}`,
                address: agentInfo.owner,
                overallRating: parseFloat(overallRating.toFixed(2)),
                totalReviews: parseInt(metrics.totalTasks),
                successRate: parseFloat((metrics.successRate / 100).toFixed(2)),
                avgResponseTime: parseInt(metrics.avgResponseTime),
                status: agentInfo.isActive ? 'Active' : 'Inactive',
                ratingDistribution: ratingDistribution,
                experience: parseInt(reputation.experience),
                badges: reputation.badges || []
            };

            // Cache for 5 minutes
            await cacheService.set(cacheKey, details, 300);

            return details;
        } catch (error) {
            logger.error(`Error getting agent details for ${agentId}:`, error);
            throw error;
        }
    }

    async getAgentReviews(agentId, options = {}) {
        if (!this.initialized) await this.initialize();

        try {
            // Get reviews from blockchain events
            const events = await this.reputationContract.getPastEvents('ReviewSubmitted', {
                filter: { agent: agentId },
                fromBlock: options.fromBlock || 0,
                toBlock: 'latest'
            });

            const reviews = await Promise.all(events.map(async (event) => {
                const review = await this.reputationContract.methods
                    .getReview(agentId, event.returnValues.reviewId)
                    .call();

                return {
                    id: event.returnValues.reviewId,
                    reviewerAddress: review.reviewer,
                    reviewerName: await this._getReviewerName(review.reviewer),
                    taskId: review.taskId,
                    overallRating: parseInt(review.rating),
                    performanceRating: parseInt(review.performanceScore || review.rating),
                    accuracyRating: parseInt(review.accuracyScore || review.rating),
                    communicationRating: parseInt(review.communicationScore || review.rating),
                    comments: review.comments || '',
                    timestamp: new Date(parseInt(review.timestamp) * 1000),
                    validationStatus: this._getValidationStatus(review),
                    validatorCount: parseInt(review.validatorCount || 0),
                    blockNumber: event.blockNumber,
                    transactionHash: event.transactionHash
                };
            }));

            // Sort by timestamp descending
            reviews.sort((a, b) => b.timestamp - a.timestamp);

            // Apply pagination if specified
            if (options.limit) {
                const start = options.offset || 0;
                return reviews.slice(start, start + options.limit);
            }

            return reviews;
        } catch (error) {
            logger.error(`Error getting reviews for agent ${agentId}:`, error);
            throw error;
        }
    }

    async submitReview(agentId, reviewData, reviewerAddress) {
        if (!this.initialized) await this.initialize();

        try {
            // Validate inputs
            validateAddress(reviewerAddress);
            validateRating(reviewData.overallRating);

            if (!reviewData.taskId) {
                throw new Error('Task ID is required');
            }

            // Prepare transaction data
            const txData = this.reputationContract.methods.submitReview(
                agentId,
                reviewData.taskId,
                reviewData.overallRating,
                reviewData.comments || '',
                {
                    performanceScore: reviewData.performanceRating || reviewData.overallRating,
                    accuracyScore: reviewData.accuracyRating || reviewData.overallRating,
                    communicationScore: reviewData.communicationRating || reviewData.overallRating
                }
            ).encodeABI();

            // Estimate gas
            const gasEstimate = await this.reputationContract.methods.submitReview(
                agentId,
                reviewData.taskId,
                reviewData.overallRating,
                reviewData.comments || '',
                {
                    performanceScore: reviewData.performanceRating || reviewData.overallRating,
                    accuracyScore: reviewData.accuracyRating || reviewData.overallRating,
                    communicationScore: reviewData.communicationRating || reviewData.overallRating
                }
            ).estimateGas({ from: reviewerAddress, value: Web3.utils.toWei(reviewData.stakeAmount || '0.01', 'ether') });

            // Return transaction data for frontend to sign and send
            return {
                to: this.reputationContract.options.address,
                data: txData,
                value: Web3.utils.toWei(reviewData.stakeAmount || '0.01', 'ether'),
                gas: Math.ceil(gasEstimate * 1.2), // Add 20% buffer
                gasPrice: await this.web3Manager.web3.eth.getGasPrice()
            };
        } catch (error) {
            logger.error(`Error preparing review submission for agent ${agentId}:`, error);
            throw error;
        }
    }

    async validateReview(agentId, reviewId, validatorAddress) {
        if (!this.initialized) await this.initialize();

        try {
            validateAddress(validatorAddress);

            // Prepare transaction data
            const txData = this.reputationContract.methods.validateReview(
                agentId,
                reviewId,
                true // Approve the review
            ).encodeABI();

            // Estimate gas
            const gasEstimate = await this.reputationContract.methods.validateReview(
                agentId,
                reviewId,
                true
            ).estimateGas({ from: validatorAddress });

            // Return transaction data for frontend to sign and send
            return {
                to: this.reputationContract.options.address,
                data: txData,
                gas: Math.ceil(gasEstimate * 1.2),
                gasPrice: await this.web3Manager.web3.eth.getGasPrice()
            };
        } catch (error) {
            logger.error(`Error validating review ${reviewId} for agent ${agentId}:`, error);
            throw error;
        }
    }

    async _getRatingDistribution(agentId) {
        try {
            const reviews = await this.getAgentReviews(agentId);
            const distribution = {
                '5': 0,
                '4': 0,
                '3': 0,
                '2': 0,
                '1': 0
            };

            reviews.forEach(review => {
                const rating = Math.round(review.overallRating);
                if (rating >= 1 && rating <= 5) {
                    distribution[rating.toString()]++;
                }
            });

            return distribution;
        } catch (error) {
            logger.error(`Error calculating rating distribution for agent ${agentId}:`, error);
            return { '5': 0, '4': 0, '3': 0, '2': 0, '1': 0 };
        }
    }

    async _getReviewerName(address) {
        try {
            // Try to get name from agent registry if reviewer is also an agent
            const agentInfo = await this.agentRegistryContract.methods.getAgentByOwner(address).call();
            if (agentInfo && agentInfo.name) {
                return agentInfo.name;
            }
        } catch (error) {
            // Reviewer might not be an agent
        }

        // Return shortened address as fallback
        return `${address.substr(0, 6)}...${address.substr(-4)}`;
    }

    _getValidationStatus(review) {
        if (review.isValidated) {
            return 'Validated';
        } else if (review.isRejected) {
            return 'Rejected';
        } else {
            return 'Pending';
        }
    }
}

module.exports = new AgentRatingService();