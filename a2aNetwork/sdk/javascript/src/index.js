/**
 * A2A Network JavaScript SDK
 * Official SDK for integrating with A2A Network infrastructure
 */

const A2AClient = require('./client/A2AClient');
const AgentManager = require('./services/AgentManager');
const MessageManager = require('./services/MessageManager');
const TokenManager = require('./services/TokenManager');
const GovernanceManager = require('./services/GovernanceManager');
const ScalabilityManager = require('./services/ScalabilityManager');
const ReputationManager = require('./services/ReputationManager');

// Utilities
const crypto = require('./utils/crypto');
const validation = require('./utils/validation');
const formatting = require('./utils/formatting');
const errors = require('./utils/errors');

// Constants
const networks = require('./constants/networks');
const contracts = require('./constants/contracts');
const config = require('./constants/config');

module.exports = {
    // Main client
    A2AClient,

    // Service managers
    AgentManager,
    MessageManager,
    TokenManager,
    GovernanceManager,
    ScalabilityManager,
    ReputationManager,

    // Utilities
    crypto,
    validation,
    formatting,
    errors,

    // Constants
    networks,
    contracts,
    config,

    // Version
    VERSION: '1.0.0'
};