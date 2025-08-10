import { ethers } from 'ethers';
import type { A2AClient } from '../client/a2aClient';
import type { ContractEvent } from '../types/common';
import { 
    Message, 
    MessageStatus, 
    MessageType,
    SendMessageParams,
    MessageFilter,
    MessageThread,
    EncryptedMessage
} from '../types/message';
import { A2AError, ErrorCode } from '../utils/errors';
import { isValidAddress } from '../utils/validation';
import { encryptMessage, decryptMessage, generateKeyPair } from '../utils/crypto';

/**
 * Message management service for A2A Network
 */
export class MessageManager {
    constructor(private client: A2AClient) {}

    /**
     * Send a message to another agent
     */
    async sendMessage(params: SendMessageParams): Promise<{ 
        transactionHash: string; 
        messageId: string;
        estimatedDelivery?: Date;
    }> {
        try {
            const contract = this.client.getContract('MessageRouter');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required to send message');
            }

            if (!params.recipientAddress || !isValidAddress(params.recipientAddress)) {
                throw new A2AError(ErrorCode.INVALID_ADDRESS, 'Invalid recipient address');
            }

            // Prepare message data
            let messageData: string;
            let isEncrypted = false;

            if (params.encrypted && params.recipientPublicKey) {
                // Encrypt message
                const encryptedData = await encryptMessage(typeof params.content === 'string' ? params.content : JSON.stringify(params.content), params.recipientPublicKey);
                messageData = JSON.stringify(encryptedData);
                isEncrypted = true;
            } else {
                messageData = typeof params.content === 'string' 
                    ? params.content 
                    : JSON.stringify(params.content);
            }

            // Calculate message fee
            const messageFee = await this.calculateMessageFee(messageData, params.priority || 'normal');

            // Send message transaction
            const tx = await contract.sendMessage(
                params.recipientAddress,
                messageData,
                this.encodeMessageType(params.messageType || MessageType.DIRECT),
                params.priority || 'normal',
                isEncrypted,
                params.replyToId || ethers.ZeroHash,
                params.metadata || '{}',
                { value: messageFee }
            );

            const receipt = await tx.wait();
            
            // Extract message ID from events
            const event = receipt.events?.find((e: ContractEvent) => e.event === 'MessageSent');
            const messageId = event?.args?.messageId as string;

            if (!messageId) {
                throw new A2AError(ErrorCode.SEND_FAILED, 'Failed to get message ID');
            }

            // Estimate delivery time based on priority
            let estimatedDelivery: Date | undefined;
            if (params.priority === 'high') {
                estimatedDelivery = new Date(Date.now() + 1000); // 1 second
            } else if (params.priority === 'normal') {
                estimatedDelivery = new Date(Date.now() + 5000); // 5 seconds
            } else {
                estimatedDelivery = new Date(Date.now() + 30000); // 30 seconds
            }

            return {
                transactionHash: tx.hash,
                messageId: messageId.toString(),
                estimatedDelivery
            };

        } catch (error: unknown) {
            if (error instanceof A2AError) throw error;
            const message = error instanceof Error ? error.message : 'Send failed';
            throw new A2AError(ErrorCode.SEND_FAILED, message);
        }
    }

    /**
     * Get message by ID
     */
    async getMessage(messageId: string): Promise<Message> {
        try {
            const contract = this.client.getContract('MessageRouter');
            const result = await contract.getMessage(messageId);

            const message: Message = {
                id: messageId,
                sender: result.sender,
                recipient: result.recipient,
                content: result.content,
                messageType: this.decodeMessageType(result.messageType),
                status: this.decodeMessageStatus(result.status),
                timestamp: new Date(result.timestamp.toNumber() * 1000),
                priority: result.priority,
                isEncrypted: result.isEncrypted,
                replyToId: result.replyToId === ethers.ZeroHash ? undefined : result.replyToId,
                metadata: result.metadata,
                gasUsed: result.gasUsed?.toNumber(),
                fee: result.fee ? ethers.formatEther(result.fee) : undefined
            };

            // Decrypt if necessary and possible
            if (message.isEncrypted) {
                try {
                    const signerAddress = await this.client.getSigner()?.getAddress();
                    if (signerAddress && 
                        (signerAddress.toLowerCase() === message.sender.toLowerCase() || 
                         signerAddress.toLowerCase() === message.recipient.toLowerCase())) {
                        // Try to decrypt - would need private key management
                        message.decryptedContent = await this.tryDecryptMessage(message.content);
                    }
                } catch (error: unknown) {
                    const errorMessage = error instanceof Error ? error.message : 'Decryption failed';
                    console.warn('Failed to decrypt message:', {
                        messageId: message.id,
                        error: errorMessage
                    });
                    // Keep message encrypted when decryption fails
                }
            }

            return message;

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Fetch failed';
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch message: ${message}`);
        }
    }

    /**
     * Get messages for current user (inbox/outbox)
     */
    async getMessages(filter: MessageFilter = {}): Promise<{
        messages: Message[];
        total: number;
        hasMore: boolean;
    }> {
        try {
            const signer = this.client.getSigner();
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required to fetch messages');
            }

            const userAddress = await signer.getAddress();
            const contract = this.client.getContract('MessageRouter');

            // Get message count for pagination
            const totalSent = await contract.getSentMessageCount(userAddress);
            const totalReceived = await contract.getReceivedMessageCount(userAddress);
            const total = totalSent.add(totalReceived).toNumber();

            // Apply filters and pagination
            const limit = filter.limit || 50;
            const offset = filter.offset || 0;

            let messageIds: string[] = [];

            if (filter.type === 'sent') {
                const sentIds = await contract.getSentMessages(userAddress, offset, limit);
                messageIds = sentIds.map((id: bigint) => id.toString());
            } else if (filter.type === 'received') {
                const receivedIds = await contract.getReceivedMessages(userAddress, offset, limit);
                messageIds = receivedIds.map((id: bigint) => id.toString());
            } else {
                // Get both sent and received
                const [sentIds, receivedIds] = await Promise.all([
                    contract.getSentMessages(userAddress, 0, Math.ceil(limit / 2)),
                    contract.getReceivedMessages(userAddress, 0, Math.ceil(limit / 2))
                ]);
                
                messageIds = [
                    ...sentIds.map((id: bigint) => id.toString()),
                    ...receivedIds.map((id: bigint) => id.toString())
                ].slice(offset, offset + limit);
            }

            // Fetch message details
            const messages = await Promise.all(
                messageIds.map(async (id) => {
                    try {
                        return await this.getMessage(id);
                    } catch (error: unknown) {
                        const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
                        console.warn('Failed to fetch message details:', {
                            messageId: id,
                            error: errorMessage
                        });
                        return null; // Return null for failed fetches to filter out later
                    }
                })
            );

            const validMessages = messages
                .filter((msg): msg is Message => msg !== null)
                .filter(msg => this.matchesFilter(msg, filter))
                .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

            return {
                messages: validMessages,
                total,
                hasMore: offset + limit < total
            };

        } catch (error: unknown) {
            if (error instanceof A2AError) throw error;
            const message = error instanceof Error ? error.message : 'Fetch failed';
            throw new A2AError(ErrorCode.FETCH_FAILED, message);
        }
    }

    /**
     * Get message thread (conversation)
     */
    async getMessageThread(messageId: string): Promise<MessageThread> {
        try {
            const rootMessage = await this.getMessage(messageId);
            
            // Find the root message of the thread
            let currentMessage = rootMessage;
            while (currentMessage.replyToId) {
                currentMessage = await this.getMessage(currentMessage.replyToId);
            }
            
            const rootId = currentMessage.id;
            
            // Get all messages in thread
            const contract = this.client.getContract('MessageRouter');
            const threadMessages = await contract.getThreadMessages(rootId);
            
            const messages = await Promise.all(
                threadMessages.map(async (id: bigint) => {
                    try {
                        return await this.getMessage(id.toString());
                    } catch (error: unknown) {
                        const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
                        console.warn('Failed to fetch thread message:', {
                            messageId: id.toString(),
                            error: errorMessage
                        });
                        return null; // Return null for failed fetches to filter out later
                    }
                })
            );

            const validMessages = messages
                .filter((msg): msg is Message => msg !== null)
                .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

            // Build thread structure
            const participants = [...new Set([
                ...validMessages.map(m => m.sender),
                ...validMessages.map(m => m.recipient)
            ])];

            return {
                id: rootId,
                messages: validMessages,
                participants,
                messageCount: validMessages.length,
                createdAt: validMessages[0]?.timestamp || new Date(),
                lastMessageAt: validMessages[validMessages.length - 1]?.timestamp || new Date(),
                isActive: validMessages.some(m => 
                    Date.now() - m.timestamp.getTime() < 24 * 60 * 60 * 1000 // Active if message in last 24h
                )
            };

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Fetch failed';
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch thread: ${message}`);
        }
    }

    /**
     * Reply to a message
     */
    async replyToMessage(
        originalMessageId: string, 
        content: string | object,
        options?: {
            encrypted?: boolean;
            priority?: 'low' | 'normal' | 'high';
            metadata?: string;
        }
    ): Promise<{ transactionHash: string; messageId: string }> {
        const originalMessage = await this.getMessage(originalMessageId);
        
        return this.sendMessage({
            recipientAddress: originalMessage.sender,
            content,
            messageType: MessageType.REPLY,
            replyToId: originalMessageId,
            encrypted: options?.encrypted,
            priority: options?.priority,
            metadata: options?.metadata
        });
    }

    /**
     * Mark message as read
     */
    async markAsRead(messageId: string): Promise<{ transactionHash: string }> {
        try {
            const contract = this.client.getContract('MessageRouter');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required');
            }

            const tx = await contract.markMessageAsRead(messageId);
            await tx.wait();

            return { transactionHash: tx.hash };

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Update failed';
            throw new A2AError(ErrorCode.UPDATE_FAILED, message);
        }
    }

    /**
     * Delete message (soft delete)
     */
    async deleteMessage(messageId: string): Promise<{ transactionHash: string }> {
        try {
            const contract = this.client.getContract('MessageRouter');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required');
            }

            // Verify user can delete this message
            const message = await this.getMessage(messageId);
            const userAddress = await signer.getAddress();
            
            if (message.sender.toLowerCase() !== userAddress.toLowerCase() &&
                message.recipient.toLowerCase() !== userAddress.toLowerCase()) {
                throw new A2AError(ErrorCode.UNAUTHORIZED, 'Not authorized to delete this message');
            }

            const tx = await contract.deleteMessage(messageId);
            await tx.wait();

            return { transactionHash: tx.hash };

        } catch (error: unknown) {
            if (error instanceof A2AError) throw error;
            const message = error instanceof Error ? error.message : 'Delete failed';
            throw new A2AError(ErrorCode.DELETE_FAILED, message);
        }
    }

    /**
     * Get message statistics for user
     */
    async getMessageStats(): Promise<{
        totalSent: number;
        totalReceived: number;
        unreadCount: number;
        avgResponseTime: number;
        topContacts: { address: string; messageCount: number }[];
    }> {
        try {
            const signer = this.client.getSigner();
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required');
            }

            const userAddress = await signer.getAddress();
            const contract = this.client.getContract('MessageRouter');

            const [totalSent, totalReceived, unreadCount] = await Promise.all([
                contract.getSentMessageCount(userAddress),
                contract.getReceivedMessageCount(userAddress),
                contract.getUnreadMessageCount(userAddress)
            ]);

            // Get recent messages to calculate stats
            const recentMessages = await this.getMessages({ limit: 100 });
            
            // Calculate average response time
            let totalResponseTime = 0;
            let responseCount = 0;
            
            for (const message of recentMessages.messages) {
                if (message.replyToId) {
                    try {
                        const originalMessage = await this.getMessage(message.replyToId);
                        const responseTime = message.timestamp.getTime() - originalMessage.timestamp.getTime();
                        totalResponseTime += responseTime;
                        responseCount++;
                    } catch (error: unknown) {
                        const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
                        console.warn('Failed to fetch original message for response time calculation:', {
                            replyToId: message.replyToId,
                            messageId: message.id,
                            error: errorMessage
                        });
                        // Skip this response time calculation
                    }
                }
            }

            const avgResponseTime = responseCount > 0 ? totalResponseTime / responseCount : 0;

            // Calculate top contacts
            const contactCounts = new Map<string, number>();
            for (const message of recentMessages.messages) {
                const contact = message.sender.toLowerCase() === userAddress.toLowerCase() 
                    ? message.recipient 
                    : message.sender;
                contactCounts.set(contact, (contactCounts.get(contact) || 0) + 1);
            }

            const topContacts = Array.from(contactCounts.entries())
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5)
                .map(([address, messageCount]) => ({ address, messageCount }));

            return {
                totalSent: totalSent.toNumber(),
                totalReceived: totalReceived.toNumber(),
                unreadCount: unreadCount.toNumber(),
                avgResponseTime: Math.round(avgResponseTime / 1000), // Convert to seconds
                topContacts
            };

        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Fetch failed';
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch stats: ${message}`);
        }
    }

    /**
     * Subscribe to new messages
     */
    async subscribeToMessages(callback: (message: Message) => void): Promise<string> {
        const signer = this.client.getSigner();
        if (!signer) {
            throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required for subscription');
        }

        const userAddress = await signer.getAddress();

        return this.client.subscribe('MessageRouter', 'MessageSent', async (
            messageId: string,
            sender: string,
            recipient: string
        ) => {
            // Only notify if user is sender or recipient
            if (sender.toLowerCase() === userAddress.toLowerCase() ||
                recipient.toLowerCase() === userAddress.toLowerCase()) {
                try {
                    const message = await this.getMessage(messageId);
                    callback(message);
                } catch (error: unknown) {
                    const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
                    console.error('Failed to fetch message for subscription callback:', {
                        messageId,
                        sender,
                        recipient,
                        error: errorMessage
                    });
                    // Don't throw error to prevent breaking subscription
                }
            }
        });
    }

    /**
     * Calculate message fee based on size and priority
     */
    async calculateMessageFee(content: string, priority: string = 'normal'): Promise<bigint> {
        try {
            const contract = this.client.getContract('MessageRouter');
            return await contract.calculateMessageFee(
                Buffer.byteLength(content, 'utf8'),
                priority === 'high' ? 2 : priority === 'low' ? 0 : 1
            );
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : 'Calculation failed';
            throw new A2AError(ErrorCode.CALCULATION_FAILED, `Failed to calculate fee: ${message}`);
        }
    }

    /**
     * Generate key pair for message encryption
     */
    generateKeyPair(): { publicKey: string; privateKey: string } {
        return generateKeyPair();
    }

    /**
     * Encrypt message content
     */
    async encryptMessageContent(content: string, recipientPublicKey: string): Promise<EncryptedMessage> {
        const encrypted = await encryptMessage(content, recipientPublicKey);
        return {
            encryptedContent: encrypted,
            encryptionMethod: 'eth-crypto',
            nonce: 'placeholder',
            metadata: {
                algorithm: 'eth-crypto',
                keySize: 256,
                timestamp: new Date()
            }
        };
    }

    /**
     * Decrypt message content
     */
    async decryptMessageContent(encryptedMessage: EncryptedMessage, privateKey: string): Promise<string> {
        return await decryptMessage(encryptedMessage.encryptedContent, privateKey);
    }

    // Private helper methods

    private encodeMessageType(type: MessageType): number {
        const typeMap: Record<MessageType, number> = {
            [MessageType.DIRECT]: 0,
            [MessageType.BROADCAST]: 1,
            [MessageType.REPLY]: 2,
            [MessageType.SYSTEM]: 3,
            [MessageType.TASK_REQUEST]: 4,
            [MessageType.TASK_RESPONSE]: 5,
            [MessageType.NOTIFICATION]: 6,
            [MessageType.FILE_TRANSFER]: 7
        };
        return typeMap[type] || 0;
    }

    private decodeMessageType(typeCode: number): MessageType {
        const typeMap: Record<number, MessageType> = {
            0: MessageType.DIRECT,
            1: MessageType.BROADCAST,
            2: MessageType.REPLY,
            3: MessageType.SYSTEM,
            4: MessageType.TASK_REQUEST,
            5: MessageType.TASK_RESPONSE,
            6: MessageType.NOTIFICATION,
            7: MessageType.FILE_TRANSFER
        };
        return typeMap[typeCode] || MessageType.DIRECT;
    }

    private decodeMessageStatus(statusCode: number): MessageStatus {
        const statusMap: Record<number, MessageStatus> = {
            0: MessageStatus.PENDING,
            1: MessageStatus.SENT,
            2: MessageStatus.DELIVERED,
            3: MessageStatus.READ,
            4: MessageStatus.FAILED,
            5: MessageStatus.DELETED
        };
        return statusMap[statusCode] || MessageStatus.PENDING;
    }

    private matchesFilter(message: Message, filter: MessageFilter): boolean {
        if (filter.status && message.status !== filter.status) return false;
        if (filter.messageType && message.messageType !== filter.messageType) return false;
        if (filter.sender && message.sender.toLowerCase() !== filter.sender.toLowerCase()) return false;
        if (filter.recipient && message.recipient.toLowerCase() !== filter.recipient.toLowerCase()) return false;
        if (filter.fromDate && message.timestamp < filter.fromDate) return false;
        if (filter.toDate && message.timestamp > filter.toDate) return false;
        if (filter.searchTerm && !message.content.toLowerCase().includes(filter.searchTerm.toLowerCase())) return false;
        
        return true;
    }

    private async tryDecryptMessage(encryptedContent: string): Promise<string | undefined> {
        try {
            const signer = await this.client.getSigner();
            if (!signer) {
                return undefined;
            }
            
            // Parse encrypted message format
            let encryptedData;
            try {
                encryptedData = JSON.parse(encryptedContent);
            } catch {
                // If not JSON, treat as raw encrypted string - can't decrypt
                return undefined;
            }
            
            if (!encryptedData.encrypted || !encryptedData.data) {
                return undefined;
            }
            
            // Use the decryptMessage function from crypto utils
            const { decryptMessage } = await import('../utils/crypto');
            const privateKey = await this.getSignerPrivateKey(signer);
            
            if (!privateKey) {
                console.warn('Private key not available for decryption');
                return undefined;
            }
            
            return await decryptMessage(encryptedContent, privateKey);
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Decryption failed';
            console.warn('Message decryption failed:', errorMessage);
            return undefined;
        }
    }
    
    private async getSignerPrivateKey(signer: ethers.Signer): Promise<string | undefined> {
        try {
            // For Wallet signers, we can access the private key directly
            if (signer instanceof ethers.Wallet) {
                return signer.privateKey;
            }
            
            // For other signer types (MetaMask, etc.), we can't access the private key
            // In those cases, encryption/decryption would need to be handled client-side
            // or use a different encryption scheme like ECIES with public keys
            return undefined;
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get private key';
            console.warn('Failed to get private key from signer:', errorMessage);
            return undefined;
        }
    }
}