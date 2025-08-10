import { Address, Hash } from './common';

/**
 * Message-related types and interfaces
 */

export interface Message {
    id: string;
    sender: Address;
    recipient: Address;
    content: string;
    messageType: MessageType;
    status: MessageStatus;
    timestamp: Date;
    priority: 'low' | 'normal' | 'high';
    isEncrypted: boolean;
    replyToId?: string;
    metadata: string;
    gasUsed?: number;
    fee?: string;
    decryptedContent?: string;
    deliveredAt?: Date;
    readAt?: Date;
    expiresAt?: Date;
    attachments?: MessageAttachment[];
    reactions?: MessageReaction[];
    threadId?: string;
    forwardedFrom?: string;
}

export enum MessageType {
    DIRECT = 'direct',
    BROADCAST = 'broadcast',
    REPLY = 'reply',
    SYSTEM = 'system',
    TASK_REQUEST = 'task_request',
    TASK_RESPONSE = 'task_response',
    NOTIFICATION = 'notification',
    FILE_TRANSFER = 'file_transfer'
}

export enum MessageStatus {
    PENDING = 'pending',
    SENT = 'sent',
    DELIVERED = 'delivered',
    READ = 'read',
    FAILED = 'failed',
    DELETED = 'deleted',
    EXPIRED = 'expired'
}

export interface SendMessageParams {
    recipientAddress: Address;
    content: string | object;
    messageType?: MessageType;
    priority?: 'low' | 'normal' | 'high';
    encrypted?: boolean;
    recipientPublicKey?: string;
    replyToId?: string;
    metadata?: string;
    expiresIn?: number; // seconds
    attachments?: MessageAttachment[];
    requireDeliveryReceipt?: boolean;
    requireReadReceipt?: boolean;
}

export interface MessageFilter {
    type?: 'sent' | 'received' | 'all';
    status?: MessageStatus;
    messageType?: MessageType;
    sender?: Address;
    recipient?: Address;
    fromDate?: Date;
    toDate?: Date;
    searchTerm?: string;
    hasAttachments?: boolean;
    isEncrypted?: boolean;
    priority?: 'low' | 'normal' | 'high';
    threadId?: string;
    limit?: number;
    offset?: number;
    sortBy?: 'timestamp' | 'priority' | 'sender' | 'status';
    sortOrder?: 'asc' | 'desc';
}

export interface MessageThread {
    id: string;
    messages: Message[];
    participants: Address[];
    messageCount: number;
    createdAt: Date;
    lastMessageAt: Date;
    isActive: boolean;
    subject?: string;
    tags?: string[];
    archivalDate?: Date;
    permissions?: {
        [address: string]: ThreadPermission[];
    };
}

export enum ThreadPermission {
    READ = 'read',
    WRITE = 'write',
    ADMIN = 'admin',
    DELETE = 'delete',
    INVITE = 'invite'
}

export interface MessageAttachment {
    id: string;
    name: string;
    type: string; // MIME type
    size: number; // in bytes
    hash: Hash; // Content hash for integrity
    url?: string; // IPFS or storage URL
    isEncrypted: boolean;
    metadata?: {
        description?: string;
        tags?: string[];
        createdBy?: Address;
        createdAt?: Date;
    };
}

export interface MessageReaction {
    emoji: string;
    user: Address;
    timestamp: Date;
    messageId: string;
}

export interface EncryptedMessage {
    encryptedContent: string;
    encryptionMethod: string;
    keyId?: string;
    nonce: string;
    authTag?: string;
    metadata?: {
        algorithm: string;
        keySize: number;
        timestamp: Date;
    };
}

export interface MessageDeliveryReceipt {
    messageId: string;
    recipient: Address;
    deliveredAt: Date;
    deliveryMethod: string;
    confirmationHash: Hash;
    retryCount?: number;
    errorMessage?: string;
}

export interface MessageReadReceipt {
    messageId: string;
    reader: Address;
    readAt: Date;
    confirmationHash: Hash;
    deviceInfo?: {
        userAgent: string;
        ipHash: string;
        platform: string;
    };
}

export interface MessageQueue {
    id: string;
    name: string;
    priority: number;
    messages: QueuedMessage[];
    maxSize: number;
    currentSize: number;
    processingRate: number; // messages per second
    retryPolicy: {
        maxRetries: number;
        backoffStrategy: 'linear' | 'exponential';
        initialDelay: number;
        maxDelay: number;
    };
    deadLetterQueue?: string;
    isActive: boolean;
}

export interface QueuedMessage {
    id: string;
    originalMessage: Message;
    queuedAt: Date;
    attempts: number;
    nextRetryAt?: Date;
    lastError?: string;
    processingStartedAt?: Date;
    estimatedProcessingTime?: number;
}

export interface MessageTemplate {
    id: string;
    name: string;
    description: string;
    content: string;
    messageType: MessageType;
    variables: {
        [key: string]: {
            type: 'string' | 'number' | 'date' | 'boolean' | 'address';
            required: boolean;
            defaultValue?: any;
            validation?: RegExp;
        };
    };
    category: string;
    tags: string[];
    createdBy: Address;
    createdAt: Date;
    lastUsed?: Date;
    usageCount: number;
    isPublic: boolean;
}

export interface MessageStatistics {
    totalSent: number;
    totalReceived: number;
    unreadCount: number;
    avgResponseTime: number;
    messageTypes: {
        [type in MessageType]: number;
    };
    priorityDistribution: {
        low: number;
        normal: number;
        high: number;
    };
    encryptedRatio: number;
    failureRate: number;
    topContacts: {
        address: Address;
        messageCount: number;
        lastMessageAt: Date;
    }[];
    dailyActivity: {
        date: Date;
        sent: number;
        received: number;
        avgResponseTime: number;
    }[];
    peakHours: {
        hour: number;
        messageCount: number;
        avgResponseTime: number;
    }[];
}

export interface MessageConfiguration {
    maxMessageSize: number; // in bytes
    maxAttachmentSize: number; // in bytes
    maxAttachmentsPerMessage: number;
    allowedAttachmentTypes: string[];
    defaultEncryption: boolean;
    autoDeleteAfterDays?: number;
    rateLimits: {
        messagesPerMinute: number;
        messagesPerHour: number;
        messagesPerDay: number;
    };
    spamFilter: {
        enabled: boolean;
        maxSimilarContent: number;
        maxFrequencyPerRecipient: number;
        keywordBlacklist: string[];
    };
    deliveryOptions: {
        maxRetries: number;
        retryIntervals: number[]; // in seconds
        timeoutSeconds: number;
        requireConfirmation: boolean;
    };
}

export interface MessageNotification {
    id: string;
    messageId: string;
    recipient: Address;
    type: NotificationType;
    content: string;
    isRead: boolean;
    createdAt: Date;
    readAt?: Date;
    channels: NotificationChannel[];
    priority: 'low' | 'normal' | 'high';
    expiresAt?: Date;
    metadata?: {
        [key: string]: any;
    };
}

export enum NotificationType {
    NEW_MESSAGE = 'new_message',
    MESSAGE_DELIVERED = 'message_delivered',
    MESSAGE_READ = 'message_read',
    MESSAGE_FAILED = 'message_failed',
    THREAD_UPDATED = 'thread_updated',
    MENTION = 'mention',
    REACTION = 'reaction',
    FILE_RECEIVED = 'file_received'
}

export enum NotificationChannel {
    PUSH = 'push',
    EMAIL = 'email',
    SMS = 'sms',
    WEBHOOK = 'webhook',
    IN_APP = 'in_app'
}

export interface MessageSearchQuery {
    query: string;
    filters?: {
        dateRange?: {
            start: Date;
            end: Date;
        };
        sender?: Address;
        recipient?: Address;
        messageType?: MessageType;
        hasAttachments?: boolean;
        isEncrypted?: boolean;
        priority?: 'low' | 'normal' | 'high';
        threadId?: string;
    };
    pagination?: {
        offset: number;
        limit: number;
    };
    sorting?: {
        field: 'timestamp' | 'relevance' | 'priority';
        order: 'asc' | 'desc';
    };
    highlighting?: boolean;
}

export interface MessageSearchResult {
    messages: Array<Message & {
        relevanceScore: number;
        highlights?: {
            field: string;
            matches: string[];
        }[];
    }>;
    totalCount: number;
    searchTime: number;
    suggestions?: string[];
    facets?: {
        senders: { address: Address; count: number }[];
        messageTypes: { type: MessageType; count: number }[];
        dates: { date: string; count: number }[];
    };
}

export interface MessageAnalytics {
    period: {
        start: Date;
        end: Date;
    };
    metrics: {
        totalMessages: number;
        uniqueContacts: number;
        averageResponseTime: number;
        messageVelocity: number; // messages per hour
        engagementRate: number; // read messages / total messages
        threadDepth: number; // average messages per thread
        encryptionRate: number; // encrypted messages / total messages
    };
    trends: {
        hourlyDistribution: { hour: number; count: number }[];
        dailyGrowth: { date: Date; count: number; growth: number }[];
        messageTypeDistribution: { type: MessageType; count: number; percentage: number }[];
        contactActivity: { address: Address; messageCount: number; lastActive: Date }[];
    };
    insights: {
        mostActiveHours: number[];
        topContacts: Address[];
        responseTimePatterns: string[];
        communicationEfficiency: number;
    };
}

// Event types for messages
export interface MessageEvent {
    type: MessageEventType;
    messageId: string;
    timestamp: Date;
    data: any;
    blockNumber?: number;
    transactionHash?: Hash;
}

export enum MessageEventType {
    SENT = 'sent',
    DELIVERED = 'delivered',
    READ = 'read',
    FAILED = 'failed',
    DELETED = 'deleted',
    ENCRYPTED = 'encrypted',
    DECRYPTED = 'decrypted',
    FORWARDED = 'forwarded',
    REACTION_ADDED = 'reaction_added',
    REACTION_REMOVED = 'reaction_removed',
    ATTACHMENT_UPLOADED = 'attachment_uploaded',
    ATTACHMENT_DOWNLOADED = 'attachment_downloaded'
}

export interface MessageBatch {
    id: string;
    messages: Message[];
    status: BatchStatus;
    createdAt: Date;
    processedAt?: Date;
    completedAt?: Date;
    failedCount: number;
    successCount: number;
    totalCount: number;
    errors: {
        messageId: string;
        error: string;
        timestamp: Date;
    }[];
}

export enum BatchStatus {
    PENDING = 'pending',
    PROCESSING = 'processing',
    COMPLETED = 'completed',
    FAILED = 'failed',
    CANCELLED = 'cancelled'
}

// Default configurations
export const DEFAULT_MESSAGE_CONFIG: MessageConfiguration = {
    maxMessageSize: 1024 * 1024, // 1MB
    maxAttachmentSize: 10 * 1024 * 1024, // 10MB
    maxAttachmentsPerMessage: 10,
    allowedAttachmentTypes: ['image/*', 'text/*', 'application/pdf', 'application/json'],
    defaultEncryption: false,
    autoDeleteAfterDays: undefined,
    rateLimits: {
        messagesPerMinute: 30,
        messagesPerHour: 500,
        messagesPerDay: 5000
    },
    spamFilter: {
        enabled: true,
        maxSimilarContent: 5,
        maxFrequencyPerRecipient: 10,
        keywordBlacklist: []
    },
    deliveryOptions: {
        maxRetries: 3,
        retryIntervals: [30, 60, 300], // 30s, 1m, 5m
        timeoutSeconds: 60,
        requireConfirmation: false
    }
};