/**
 * @fileoverview Webhook Service for Deployment Notifications
 * @since 1.0.0
 * @module webhookService
 * 
 * Handles webhook delivery and notification system for deployment events
 * Integrates with Slack, Teams, email, and custom endpoints
 */

const EventEmitter = require('events');
const fetch = require('node-fetch');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class WebhookService extends EventEmitter {
    constructor() {
        super();
        this.webhooks = new Map();
        this.notificationQueue = [];
        this.processing = false;
        this.retryAttempts = 3;
        this.retryDelay = 5000; // 5 seconds
        
        this.loadWebhooks();
        this.startQueueProcessor();
    }
    
    async loadWebhooks() {
        try {
            const webhooksPath = path.join(__dirname, '../config/webhooks.json');
            const webhooksConfig = JSON.parse(await fs.readFile(webhooksPath, 'utf8'));
            
            webhooksConfig.endpoints.forEach(webhook => {
                this.registerWebhook(webhook);
            });
            
            console.log(`Loaded ${this.webhooks.size} webhook endpoints`);
        } catch (error) {
            console.warn('No webhook configuration found, using defaults');
            this.setupDefaultWebhooks();
        }
    }
    
    setupDefaultWebhooks() {
        // Default Slack webhook (if configured)
        if (process.env.SLACK_WEBHOOK_URL) {
            this.registerWebhook({
                id: 'slack-deployments',
                name: 'Slack Deployments',
                url: process.env.SLACK_WEBHOOK_URL,
                type: 'slack',
                events: ['deploymentStarted', 'deploymentCompleted', 'deploymentFailed'],
                active: true
            });
        }
        
        // Default Teams webhook (if configured)
        if (process.env.TEAMS_WEBHOOK_URL) {
            this.registerWebhook({
                id: 'teams-deployments',
                name: 'Teams Deployments',
                url: process.env.TEAMS_WEBHOOK_URL,
                type: 'teams',
                events: ['deploymentStarted', 'deploymentCompleted', 'deploymentFailed'],
                active: true
            });
        }
        
        // Default email notifications
        if (process.env.SMTP_HOST) {
            this.registerWebhook({
                id: 'email-notifications',
                name: 'Email Notifications',
                url: 'internal://email',
                type: 'email',
                events: ['deploymentCompleted', 'deploymentFailed', 'deploymentRollback'],
                active: true,
                config: {
                    recipients: process.env.NOTIFICATION_EMAILS?.split(',') || []
                }
            });
        }
    }
    
    registerWebhook(webhook) {
        const webhookConfig = {
            id: webhook.id,
            name: webhook.name,
            url: webhook.url,
            type: webhook.type || 'generic',
            events: webhook.events || [],
            active: webhook.active !== false,
            config: webhook.config || {},
            secret: webhook.secret || null,
            headers: webhook.headers || {},
            retryAttempts: webhook.retryAttempts || this.retryAttempts,
            created: new Date().toISOString(),
            lastTriggered: null,
            totalCalls: 0,
            successfulCalls: 0,
            failedCalls: 0
        };
        
        this.webhooks.set(webhook.id, webhookConfig);
        console.log(`Registered webhook: ${webhook.name} (${webhook.type})`);
    }
    
    async notify(eventType, data) {
        const notification = {
            id: crypto.randomUUID(),
            eventType,
            data,
            timestamp: new Date().toISOString(),
            attempts: 0
        };
        
        this.notificationQueue.push(notification);
        
        if (!this.processing) {
            this.processQueue();
        }
        
        return notification.id;
    }
    
    async processQueue() {
        if (this.processing || this.notificationQueue.length === 0) {
            return;
        }
        
        this.processing = true;
        
        while (this.notificationQueue.length > 0) {
            const notification = this.notificationQueue.shift();
            
            // Find webhooks that should receive this event
            const relevantWebhooks = Array.from(this.webhooks.values())
                .filter(webhook => 
                    webhook.active && 
                    webhook.events.includes(notification.eventType)
                );
            
            // Send to all relevant webhooks
            await Promise.all(
                relevantWebhooks.map(webhook => 
                    this.sendWebhook(webhook, notification)
                )
            );
        }
        
        this.processing = false;
    }
    
    async sendWebhook(webhook, notification) {
        const payload = this.formatPayload(webhook, notification);
        
        try {
            let response;
            
            if (webhook.type === 'email') {
                response = await this.sendEmail(webhook, notification);
            } else {
                response = await this.sendHttpWebhook(webhook, payload);
            }
            
            // Update webhook statistics
            webhook.totalCalls++;
            webhook.successfulCalls++;
            webhook.lastTriggered = new Date().toISOString();
            
            console.log(`✅ Webhook sent successfully: ${webhook.name}`);
            
            this.emit('webhookSuccess', {
                webhookId: webhook.id,
                eventType: notification.eventType,
                response: response
            });
            
        } catch (error) {
            webhook.totalCalls++;
            webhook.failedCalls++;
            
            console.error(`❌ Webhook failed: ${webhook.name}`, error.message);
            
            // Retry logic
            if (notification.attempts < webhook.retryAttempts) {
                notification.attempts++;
                console.log(`🔄 Retrying webhook ${webhook.name} (attempt ${notification.attempts})`);
                
                setTimeout(() => {
                    this.notificationQueue.unshift(notification);
                    if (!this.processing) {
                        this.processQueue();
                    }
                }, this.retryDelay * notification.attempts);
            } else {
                this.emit('webhookFailure', {
                    webhookId: webhook.id,
                    eventType: notification.eventType,
                    error: error.message,
                    finalAttempt: true
                });
            }
        }
    }
    
    async sendHttpWebhook(webhook, payload) {
        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'A2A-Platform-Webhook/1.0',
            ...webhook.headers
        };
        
        // Add webhook signature if secret is configured
        if (webhook.secret) {
            const signature = crypto
                .createHmac('sha256', webhook.secret)
                .update(JSON.stringify(payload))
                .digest('hex');
            headers['X-A2A-Signature'] = `sha256=${signature}`;
        }
        
        const response = await fetch(webhook.url, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload),
            timeout: 10000
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return {
            status: response.status,
            statusText: response.statusText,
            headers: Object.fromEntries(response.headers.entries())
        };
    }
    
    async sendEmail(webhook, notification) {
        const nodemailer = require('nodemailer');
        
        const transporter = nodemailer.createTransporter({
            host: process.env.SMTP_HOST,
            port: process.env.SMTP_PORT || 587,
            secure: process.env.SMTP_SECURE === 'true',
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS
            }
        });
        
        const emailContent = this.formatEmailContent(notification);
        
        const mailOptions = {
            from: process.env.SMTP_FROM || 'A2A Platform <noreply@a2a-platform.com>',
            to: webhook.config.recipients.join(', '),
            subject: emailContent.subject,
            html: emailContent.html,
            text: emailContent.text
        };
        
        const result = await transporter.sendMail(mailOptions);
        return result;
    }
    
    formatPayload(webhook, notification) {
        const basePayload = {
            event: notification.eventType,
            timestamp: notification.timestamp,
            source: 'A2A Platform',
            data: notification.data
        };
        
        switch (webhook.type) {
            case 'slack':
                return this.formatSlackPayload(basePayload);
            case 'teams':
                return this.formatTeamsPayload(basePayload);
            case 'discord':
                return this.formatDiscordPayload(basePayload);
            default:
                return basePayload;
        }
    }
    
    formatSlackPayload(payload) {
        const { event, data, timestamp } = payload;
        
        let color = '#0070f2';
        let emoji = '🚀';
        
        switch (event) {
            case 'deploymentStarted':
                color = '#ff9500';
                emoji = '🚀';
                break;
            case 'deploymentCompleted':
                color = 'good';
                emoji = '✅';
                break;
            case 'deploymentFailed':
                color = 'danger';
                emoji = '❌';
                break;
            case 'deploymentRollback':
                color = 'warning';
                emoji = '⚠️';
                break;
        }
        
        return {
            text: `${emoji} A2A Platform Deployment Update`,
            attachments: [{
                color,
                fields: [
                    {
                        title: 'Application',
                        value: data.appName || 'Unknown',
                        short: true
                    },
                    {
                        title: 'Environment',
                        value: data.environment || 'Unknown',
                        short: true
                    },
                    {
                        title: 'Version',
                        value: data.version || 'Unknown',
                        short: true
                    },
                    {
                        title: 'Status',
                        value: this.formatEventName(event),
                        short: true
                    }
                ],
                footer: 'A2A Platform',
                ts: Math.floor(new Date(timestamp).getTime() / 1000)
            }]
        };
    }
    
    formatTeamsPayload(payload) {
        const { event, data, timestamp } = payload;
        
        let themeColor = '0070f2';
        
        switch (event) {
            case 'deploymentCompleted':
                themeColor = '00ff00';
                break;
            case 'deploymentFailed':
                themeColor = 'ff0000';
                break;
            case 'deploymentRollback':
                themeColor = 'ff9500';
                break;
        }
        
        return {
            '@type': 'MessageCard',
            '@context': 'http://schema.org/extensions',
            summary: `A2A Platform: ${this.formatEventName(event)}`,
            themeColor,
            sections: [{
                activityTitle: `A2A Platform Deployment: ${this.formatEventName(event)}`,
                activitySubtitle: `${data.appName} v${data.version} to ${data.environment}`,
                facts: [
                    { name: 'Application', value: data.appName || 'Unknown' },
                    { name: 'Environment', value: data.environment || 'Unknown' },
                    { name: 'Version', value: data.version || 'Unknown' },
                    { name: 'Timestamp', value: new Date(timestamp).toLocaleString() }
                ]
            }]
        };
    }
    
    formatDiscordPayload(payload) {
        const { event, data, timestamp } = payload;
        
        let color = 0x0070f2;
        
        switch (event) {
            case 'deploymentCompleted':
                color = 0x00ff00;
                break;
            case 'deploymentFailed':
                color = 0xff0000;
                break;
            case 'deploymentRollback':
                color = 0xff9500;
                break;
        }
        
        return {
            embeds: [{
                title: 'A2A Platform Deployment Update',
                description: `${this.formatEventName(event)}: ${data.appName} v${data.version}`,
                color,
                fields: [
                    { name: 'Application', value: data.appName || 'Unknown', inline: true },
                    { name: 'Environment', value: data.environment || 'Unknown', inline: true },
                    { name: 'Version', value: data.version || 'Unknown', inline: true }
                ],
                timestamp: new Date(timestamp).toISOString()
            }]
        };
    }
    
    formatEmailContent(notification) {
        const { eventType, data, timestamp } = notification;
        
        const subject = `A2A Platform: ${this.formatEventName(eventType)} - ${data.appName}`;
        
        const html = `
            <html>
                <body style="font-family: Arial, sans-serif; color: #333;">
                    <h2 style="color: #0070f2;">A2A Platform Deployment Notification</h2>
                    
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3>${this.formatEventName(eventType)}</h3>
                        <p><strong>Application:</strong> ${data.appName || 'Unknown'}</p>
                        <p><strong>Environment:</strong> ${data.environment || 'Unknown'}</p>
                        <p><strong>Version:</strong> ${data.version || 'Unknown'}</p>
                        <p><strong>Timestamp:</strong> ${new Date(timestamp).toLocaleString()}</p>
                        ${data.duration ? `<p><strong>Duration:</strong> ${Math.round(data.duration / 60)} minutes</p>` : ''}
                        ${data.error ? `<p><strong>Error:</strong> ${data.error}</p>` : ''}
                    </div>
                    
                    <p style="color: #666; font-size: 12px;">
                        This is an automated message from the A2A Platform deployment system.
                    </p>
                </body>
            </html>
        `;
        
        const text = `
A2A Platform Deployment Notification

${this.formatEventName(eventType)}

Application: ${data.appName || 'Unknown'}
Environment: ${data.environment || 'Unknown'}
Version: ${data.version || 'Unknown'}
Timestamp: ${new Date(timestamp).toLocaleString()}
${data.duration ? `Duration: ${Math.round(data.duration / 60)} minutes` : ''}
${data.error ? `Error: ${data.error}` : ''}

This is an automated message from the A2A Platform deployment system.
        `;
        
        return { subject, html, text };
    }
    
    formatEventName(eventType) {
        const eventNames = {
            deploymentStarted: 'Deployment Started',
            deploymentCompleted: 'Deployment Completed',
            deploymentFailed: 'Deployment Failed',
            deploymentRollback: 'Deployment Rolled Back',
            deploymentApproved: 'Deployment Approved',
            deploymentPaused: 'Deployment Paused'
        };
        
        return eventNames[eventType] || eventType;
    }
    
    getWebhookStats() {
        const stats = Array.from(this.webhooks.values()).map(webhook => ({
            id: webhook.id,
            name: webhook.name,
            type: webhook.type,
            active: webhook.active,
            totalCalls: webhook.totalCalls,
            successfulCalls: webhook.successfulCalls,
            failedCalls: webhook.failedCalls,
            successRate: webhook.totalCalls > 0 ? 
                Math.round((webhook.successfulCalls / webhook.totalCalls) * 100) : 0,
            lastTriggered: webhook.lastTriggered
        }));
        
        return {
            totalWebhooks: this.webhooks.size,
            activeWebhooks: stats.filter(w => w.active).length,
            queueSize: this.notificationQueue.length,
            webhooks: stats
        };
    }
    
    startQueueProcessor() {
        // Process queue every 30 seconds to handle any stuck notifications
        setInterval(() => {
            if (!this.processing && this.notificationQueue.length > 0) {
                this.processQueue();
            }
        }, 30000);
    }
    
    updateWebhook(webhookId, updates) {
        const webhook = this.webhooks.get(webhookId);
        if (!webhook) {
            throw new Error(`Webhook not found: ${webhookId}`);
        }
        
        Object.assign(webhook, updates);
        console.log(`Updated webhook: ${webhook.name}`);
        return webhook;
    }
    
    deleteWebhook(webhookId) {
        const webhook = this.webhooks.get(webhookId);
        if (!webhook) {
            throw new Error(`Webhook not found: ${webhookId}`);
        }
        
        this.webhooks.delete(webhookId);
        console.log(`Deleted webhook: ${webhook.name}`);
        return true;
    }
}

module.exports = WebhookService;