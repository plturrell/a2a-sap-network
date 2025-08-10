/**
 * A2A Network TypeScript SDK
 * Official SDK for integrating with A2A Network infrastructure
 */

export { A2AClient } from './client/A2AClient';
export { AgentManager } from './services/AgentManager';
export { MessageManager } from './services/MessageManager';
export { TokenManager } from './services/TokenManager';
export { GovernanceManager } from './services/GovernanceManager';
export { ScalabilityManager } from './services/ScalabilityManager';
export { ReputationManager } from './services/ReputationManager';

// Types and interfaces
export * from './types/Agent';
export * from './types/Message';
export * from './types/Token';
export * from './types/Governance';
export * from './types/Scalability';
export * from './types/Reputation';
export * from './types/Common';

// Utilities
export * from './utils/crypto';
export * from './utils/validation';
export * from './utils/formatting';
export * from './utils/errors';

// Constants
export * from './constants/networks';
export * from './constants/contracts';
export * from './constants/config';

// Version
export const VERSION = '1.0.0';