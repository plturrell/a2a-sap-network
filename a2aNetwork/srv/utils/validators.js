/**
 * Validation utilities for agent rating service
 */

const Web3 = require('web3');

/**
 * Validate Ethereum address
 * @param {string} address - The address to validate
 * @throws {Error} If address is invalid
 */
function validateAddress(address) {
    if (!address || typeof address !== 'string') {
        throw new Error('Address is required');
    }
    
    if (!Web3.utils.isAddress(address)) {
        throw new Error('Invalid Ethereum address');
    }
    
    return true;
}

/**
 * Validate rating value
 * @param {number} rating - The rating to validate
 * @throws {Error} If rating is invalid
 */
function validateRating(rating) {
    if (rating === undefined || rating === null) {
        throw new Error('Rating is required');
    }
    
    const numRating = Number(rating);
    if (isNaN(numRating) || numRating < 1 || numRating > 5) {
        throw new Error('Rating must be between 1 and 5');
    }
    
    return true;
}

/**
 * Validate task ID
 * @param {string} taskId - The task ID to validate
 * @throws {Error} If task ID is invalid
 */
function validateTaskId(taskId) {
    if (!taskId || typeof taskId !== 'string') {
        throw new Error('Task ID is required');
    }
    
    if (taskId.length < 1 || taskId.length > 100) {
        throw new Error('Task ID must be between 1 and 100 characters');
    }
    
    return true;
}

/**
 * Validate review comments
 * @param {string} comments - The comments to validate
 * @throws {Error} If comments are invalid
 */
function validateComments(comments) {
    if (comments && typeof comments !== 'string') {
        throw new Error('Comments must be a string');
    }
    
    if (comments && comments.length > 1000) {
        throw new Error('Comments must not exceed 1000 characters');
    }
    
    return true;
}

/**
 * Validate stake amount
 * @param {string|number} amount - The stake amount to validate
 * @throws {Error} If amount is invalid
 */
function validateStakeAmount(amount) {
    if (amount === undefined || amount === null) {
        return true; // Optional, use default
    }
    
    const numAmount = Number(amount);
    if (isNaN(numAmount) || numAmount < 0) {
        throw new Error('Stake amount must be a positive number');
    }
    
    if (numAmount > 10) {
        throw new Error('Stake amount cannot exceed 10 ETH');
    }
    
    return true;
}

/**
 * Validate pagination parameters
 * @param {object} params - The pagination parameters
 * @throws {Error} If parameters are invalid
 */
function validatePagination(params) {
    if (params.limit !== undefined) {
        const limit = Number(params.limit);
        if (isNaN(limit) || limit < 1 || limit > 100) {
            throw new Error('Limit must be between 1 and 100');
        }
    }
    
    if (params.offset !== undefined) {
        const offset = Number(params.offset);
        if (isNaN(offset) || offset < 0) {
            throw new Error('Offset must be a non-negative number');
        }
    }
    
    return true;
}

module.exports = {
    validateAddress,
    validateRating,
    validateTaskId,
    validateComments,
    validateStakeAmount,
    validatePagination
};