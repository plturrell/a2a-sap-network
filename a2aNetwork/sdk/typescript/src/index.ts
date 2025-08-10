/**
 * A2A Network TypeScript SDK
 * Official SDK for integrating with A2A Network infrastructure
 */

export { A2AClient } from './client/a2aClient';
export { AgentManager } from './services/agentManager';
export { MessageManager } from './services/messageManager';
export { TokenManager } from './services/tokenManager';
export { GovernanceManager } from './services/governanceManager';
export { ScalabilityManager } from './services/scalabilityManager';
export { ReputationManager } from './services/reputationManager';

// Types and interfaces
export * from './types/agent';
export * from './types/message';
export * from './types/token';
export * from './types/governance';
export * from './types/scalability';
export * from './types/reputation';
export * from './types/common';

// Utilities
export * from './utils/crypto';
export * from './utils/validation';
export * from './utils/formatting';
export * from './utils/errors';

// Constants
export * from './constants/networks';
export * from './constants/contracts';
export { DEFAULT_CONFIG as SDK_DEFAULT_CONFIG } from './constants/config';

// Version
export const VERSION = '1.0.0';