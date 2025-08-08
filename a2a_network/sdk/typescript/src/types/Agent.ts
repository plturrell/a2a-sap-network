import { Address, Hash, Timestamp } from './Common';

/**
 * Agent-related types and interfaces
 */

export interface Agent {
    id: string;
    owner: Address;
    name: string;
    description: string;
    endpoint: string;
    isActive: boolean;
    registrationDate: Date;
    lastActive: Date;
    messageCount: number;
    capabilities: AgentCapabilities;
    metadata: string;
    status?: AgentStatus;
    version?: string;
    tags?: string[];
}

export interface AgentProfile extends Agent {
    reputation: {
        score: number;
        rank: number;
        totalTasks: number;
        successRate: number;
        avgResponseTime: number;
        totalEarnings: string;
    };
    performance: {
        quality: number;
        speed: number;
        reliability: number;
        innovation: number;
    };
    skills: string[];
    statistics?: AgentStatistics;
    availability?: AgentAvailability;
}

export interface AgentCapabilities {
    [capability: string]: boolean;
}

export interface AgentRegistrationParams {
    name: string;
    description: string;
    endpoint: string;
    capabilities: AgentCapabilities;
    metadata?: string;
    initialStake?: string;
    referrerCode?: string;
}

export interface AgentUpdateParams {
    name?: string;
    description?: string;
    endpoint?: string;
    capabilities?: Partial<AgentCapabilities>;
    metadata?: string;
    tags?: string[];
}

export enum AgentStatus {
    ACTIVE = 'active',
    INACTIVE = 'inactive',
    SUSPENDED = 'suspended',
    MAINTENANCE = 'maintenance',
    PENDING_APPROVAL = 'pending_approval',
    DEREGISTERED = 'deregistered'
}

export interface AgentStatistics {
    totalMessages: number;
    successfulTasks: number;
    failedTasks: number;
    avgResponseTime: number;
    uptime: number;
    earnings: string;
    ratingsReceived: number;
    averageRating: number;
    lastSeenAt: Date;
    joinedAt: Date;
    totalConnections: number;
    activeConnections: number;
    bandwidthUsed: number; // in bytes
    storageUsed: number; // in bytes
}

export interface AgentAvailability {
    isOnline: boolean;
    lastOnline: Date;
    currentLoad: number; // 0-100 percentage
    maxConcurrentTasks: number;
    currentTasks: number;
    averageTaskDuration: number; // in seconds
    workingHours?: {
        timezone: string;
        schedule: {
            [day: string]: {
                start: string; // HH:mm format
                end: string;   // HH:mm format
            };
        };
    };
    maintenanceWindows?: {
        start: Date;
        end: Date;
        reason: string;
        recurring?: boolean;
    }[];
}

export interface AgentSearchCriteria {
    skills?: string[];
    minReputation?: number;
    maxResponseTime?: number;
    region?: string;
    availability?: 'online' | 'offline' | 'any';
    priceRange?: {
        min: string;
        max: string;
    };
    capabilities?: string[];
    languages?: string[];
    experienceLevel?: 'beginner' | 'intermediate' | 'expert' | 'any';
    verificationLevel?: 'unverified' | 'email' | 'kyc' | 'premium';
}

export interface AgentSearchResult {
    agent: Agent;
    matchScore: number;
    relevanceFactors: {
        skillMatch: number;
        reputationScore: number;
        responseTime: number;
        priceMatch: number;
        availability: number;
    };
    estimatedCost?: string;
    estimatedCompletionTime?: number; // in seconds
}

export interface AgentTask {
    id: string;
    agentId: string;
    clientAddress: Address;
    title: string;
    description: string;
    requirements: string[];
    budget: string;
    deadline?: Date;
    status: TaskStatus;
    createdAt: Date;
    startedAt?: Date;
    completedAt?: Date;
    result?: string;
    feedback?: TaskFeedback;
    metadata: string;
}

export enum TaskStatus {
    PENDING = 'pending',
    ACCEPTED = 'accepted',
    IN_PROGRESS = 'in_progress',
    COMPLETED = 'completed',
    CANCELLED = 'cancelled',
    DISPUTED = 'disputed',
    FAILED = 'failed'
}

export interface TaskFeedback {
    rating: number; // 1-5 stars
    comment: string;
    categories: {
        quality: number;
        communication: number;
        timeliness: number;
        professionalism: number;
    };
    wouldRecommend: boolean;
    submittedAt: Date;
    verified: boolean;
}

export interface AgentConnection {
    agentId: string;
    clientAddress: Address;
    connectionId: string;
    establishedAt: Date;
    lastInteractionAt: Date;
    status: ConnectionStatus;
    messageCount: number;
    totalValue: string; // Total value of transactions
    avgResponseTime: number;
    trustScore: number; // 0-100
}

export enum ConnectionStatus {
    ACTIVE = 'active',
    IDLE = 'idle',
    SUSPENDED = 'suspended',
    TERMINATED = 'terminated'
}

export interface AgentMetrics {
    timestamp: Date;
    totalAgents: number;
    activeAgents: number;
    totalTasks: number;
    activeTasks: number;
    avgTaskCompletionTime: number;
    successRate: number;
    networkUtilization: number;
    topPerformingAgents: {
        agentId: string;
        name: string;
        score: number;
    }[];
    recentActivity: {
        registrations: number;
        deregistrations: number;
        tasksCreated: number;
        tasksCompleted: number;
    };
}

export interface AgentCapability {
    name: string;
    category: string;
    description: string;
    version: string;
    parameters?: {
        [key: string]: {
            type: string;
            required: boolean;
            description: string;
            defaultValue?: any;
        };
    };
    pricing?: {
        model: 'fixed' | 'usage' | 'subscription';
        amount: string;
        currency: string;
    };
}

export interface AgentEndpoint {
    url: string;
    protocol: 'http' | 'https' | 'ws' | 'wss';
    authMethod?: 'none' | 'bearer' | 'api_key' | 'signature';
    healthCheckPath?: string;
    rateLimit?: {
        requestsPerSecond: number;
        burstSize?: number;
    };
    timeout?: number; // in milliseconds
    retryPolicy?: {
        maxRetries: number;
        backoffStrategy: 'linear' | 'exponential';
        initialDelay: number;
    };
}

export interface AgentConfiguration {
    maxConcurrentConnections: number;
    maxMessageSize: number; // in bytes
    supportedProtocols: string[];
    requiredCapabilities: string[];
    optionalCapabilities: string[];
    securityLevel: 'basic' | 'standard' | 'high' | 'enterprise';
    dataRetentionDays: number;
    loggingLevel: 'none' | 'basic' | 'detailed' | 'debug';
    monitoringEnabled: boolean;
    autoUpdateEnabled: boolean;
}

export interface AgentAnalytics {
    agentId: string;
    period: {
        start: Date;
        end: Date;
    };
    metrics: {
        requestCount: number;
        uniqueClients: number;
        averageResponseTime: number;
        errorRate: number;
        revenue: string;
        topRequestTypes: {
            type: string;
            count: number;
            avgProcessingTime: number;
        }[];
        clientSatisfaction: {
            averageRating: number;
            totalRatings: number;
            ratingDistribution: { [rating: number]: number };
        };
        performanceTrends: {
            date: Date;
            responseTime: number;
            requestCount: number;
            errorCount: number;
        }[];
    };
    insights: {
        recommendations: string[];
        optimizationOpportunities: string[];
        riskFactors: string[];
    };
}

// Event types for agents
export interface AgentEvent {
    type: AgentEventType;
    agentId: string;
    timestamp: Date;
    data: any;
    blockNumber?: number;
    transactionHash?: Hash;
}

export enum AgentEventType {
    REGISTERED = 'registered',
    UPDATED = 'updated',
    STATUS_CHANGED = 'status_changed',
    TASK_RECEIVED = 'task_received',
    TASK_COMPLETED = 'task_completed',
    CONNECTION_ESTABLISHED = 'connection_established',
    CONNECTION_TERMINATED = 'connection_terminated',
    REPUTATION_UPDATED = 'reputation_updated',
    CAPABILITY_ADDED = 'capability_added',
    CAPABILITY_REMOVED = 'capability_removed',
    MAINTENANCE_STARTED = 'maintenance_started',
    MAINTENANCE_ENDED = 'maintenance_ended'
}

// Validation schemas
export interface AgentValidationRules {
    name: {
        minLength: number;
        maxLength: number;
        pattern?: RegExp;
    };
    description: {
        minLength: number;
        maxLength: number;
    };
    endpoint: {
        protocols: string[];
        requireSSL: boolean;
    };
    capabilities: {
        required: string[];
        maximum: number;
    };
}

export const DEFAULT_AGENT_VALIDATION_RULES: AgentValidationRules = {
    name: {
        minLength: 3,
        maxLength: 50,
        pattern: /^[a-zA-Z0-9\s\-_]+$/
    },
    description: {
        minLength: 10,
        maxLength: 500
    },
    endpoint: {
        protocols: ['https', 'wss'],
        requireSSL: true
    },
    capabilities: {
        required: ['basic_messaging'],
        maximum: 20
    }
};

// Standard capability categories
export const CAPABILITY_CATEGORIES = {
    COMMUNICATION: 'communication',
    DATA_PROCESSING: 'data_processing',
    AI_ML: 'ai_ml',
    INTEGRATION: 'integration',
    SECURITY: 'security',
    ANALYTICS: 'analytics',
    AUTOMATION: 'automation',
    CUSTOM: 'custom'
} as const;

export type CapabilityCategory = typeof CAPABILITY_CATEGORIES[keyof typeof CAPABILITY_CATEGORIES];