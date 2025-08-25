/**
 * SAP CAP Marketplace Service Implementation
 * Production-grade implementation following SAP best practices
 * Integrated with A2A standardized logging and monitoring
 */

const cds = require('@sap/cds');
const { v4: uuid } = require('uuid');
const { marketplaceLogger } = require('./services/marketplaceLoggingService');

class MarketplaceService extends cds.ApplicationService {
    
    async init() {
        // Initialize logging for marketplace service
        this.logger = marketplaceLogger.child('marketplace-service');
        this.logger.info('Initializing Marketplace Service', {
            operation: 'service_init',
            category: 'system'
        });

        try {
            // Get database service
            this.db = await cds.connect.to('db');
            this.logger.info('Database connection established', {
                operation: 'db_connect',
                category: 'system'
            });
        
            // Get entities from model
            const { 
                Services, 
                DataProducts, 
                AgentListings, 
                ServiceRequests, 
                DataIntegrations,
                ServiceReviews,
                DataProductReviews,
                MarketplaceStats,
                ServiceUsageMetrics,
                DataUsageAnalytics
            } = this.entities;

        // ==========================================
        // BEFORE HANDLERS - VALIDATION & SECURITY
        // ==========================================

        this.before('CREATE', Services, this._validateService);
        this.before('UPDATE', Services, this._validateService);
        this.before('CREATE', DataProducts, this._validateDataProduct);
        this.before('UPDATE', DataProducts, this._validateDataProduct);
        this.before('CREATE', ServiceRequests, this._validateServiceRequest);
        this.before('UPDATE', ServiceRequests, this._validateServiceRequest);

        // Authorization checks
        this.before(['CREATE', 'UPDATE', 'DELETE'], Services, this._checkServiceAuthorization);
        this.before(['CREATE', 'UPDATE', 'DELETE'], DataProducts, this._checkDataProductAuthorization);
        this.before(['CREATE', 'UPDATE', 'DELETE'], ServiceRequests, this._checkServiceRequestAuthorization);

        // ==========================================
        // AFTER HANDLERS - COMPUTED FIELDS & SIDE EFFECTS
        // ==========================================

        this.after('READ', Services, this._enrichServices);
        this.after('read', DataProducts, this._enrichDataProducts);
        this.after('read', AgentListings, this._enrichAgentListings);
        this.after('read', ServiceRequests, this._enrichServiceRequests);
        this.after('read', DataIntegrations, this._enrichDataIntegrations);

        // Side effects
        this.after('CREATE', ServiceRequests, this._onServiceRequestCreated);
        this.after('UPDATE', ServiceRequests, this._onServiceRequestUpdated);
        this.after('CREATE', ServiceReviews, this._onServiceReviewCreated);
        this.after('CREATE', DataProductReviews, this._onDataProductReviewCreated);

        // ==========================================
        // ACTION HANDLERS
        // ==========================================

        this.on('requestService', Services, this._handleRequestService);
        this.on('submitBid', Services, this._handleSubmitBid);
        this.on('purchaseDataProduct', DataProducts, this._handlePurchaseDataProduct);
        this.on('previewData', DataProducts, this._handlePreviewData);
        this.on('cancelRequest', ServiceRequests, this._handleCancelRequest);
        this.on('rateService', ServiceRequests, this._handleRateService);
        this.on('pauseIntegration', DataIntegrations, this._handlePauseIntegration);
        this.on('resumeIntegration', DataIntegrations, this._handleResumeIntegration);
        this.on('getAnalytics', DataIntegrations, this._handleGetIntegrationAnalytics);

        // ==========================================
        // FUNCTION HANDLERS
        // ==========================================

        this.on('getRecommendations', this._handleGetRecommendations);
        this.on('searchMarketplace', this._handleSearchMarketplace);
        this.on('getIntegrationRecommendations', this._handleGetIntegrationRecommendations);
        this.on('getMarketplaceOverview', this._handleGetMarketplaceOverview);
        this.on('getRevenueAnalytics', this._handleGetRevenueAnalytics);
        this.on('getPerformanceAnalytics', this._handleGetPerformanceAnalytics);
        this.on('getNotificationPreferences', this._handleGetNotificationPreferences);

        // ==========================================
        // MISC ACTION HANDLERS
        // ==========================================

        this.on('createDataIntegration', this._handleCreateDataIntegration);
        this.on('trackUserInteraction', this._handleTrackUserInteraction);
        this.on('subscribeToNotifications', this._handleSubscribeToNotifications);

            this.logger.info('Marketplace Service initialization completed', {
                operation: 'service_init',
                category: 'system',
                status: 'completed'
            });

            return super.init();
            
        } catch (error) {
            this.logger.logError(error, {
                operation: 'service_init',
                category: 'system',
                status: 'failed'
            });
            throw error;
        }
    }

    // ==========================================
    // VALIDATION METHODS
    // ==========================================

    async _validateService(req) {
        const startTime = Date.now();
        const service = req.data;
        
        this.logger.debug('Validating service data', {
            operation: 'service_validation',
            serviceId: service.ID,
            serviceName: service.name,
            category: 'business'
        });
        
        try {
            // Validate required fields
            if (!service.name || service.name.trim().length === 0) {
                this.logger.warn('Service validation failed: missing name', {
                    operation: 'service_validation',
                    serviceId: service.ID,
                    validationError: 'missing_name'
                });
                req.error(400, 'Service name is required', 'name');
            }
        
        if (!service.category_code) {
            req.error(400, 'Service category is required', 'category_code');
        }
        
        if (!service.providerAgent) {
            req.error(400, 'Provider agent is required', 'providerAgent');
        }

        // Validate price
        if (service.basePrice < 0) {
            req.error(400, 'Base price cannot be negative', 'basePrice');
        }

        // Validate pricing model constraints
        if (service.pricing === 'TIERED' && !service.pricingTiers) {
            req.error(400, 'Tiered pricing requires pricing tiers', 'pricing');
        }

        // Validate JSON fields
        await this._validateJsonField(req, service.inputSchema, 'inputSchema');
        await this._validateJsonField(req, service.outputSchema, 'outputSchema');
        await this._validateJsonField(req, service.configurationSchema, 'configurationSchema');

        // Set computed fields
        if (req.event === 'CREATE') {
            service.ID = service.ID || uuid();
            service.currentActiveRequests = 0;
            service.status = service.status || 'DRAFT';
        }

            // Validate business rules
            await this._validateServiceBusinessRules(req, service);
            
            this.logger.logPerformance('service_validation', Date.now() - startTime, {
                serviceId: service.ID,
                serviceName: service.name,
                validationPassed: true
            });
            
        } catch (error) {
            this.logger.logError(error, {
                operation: 'service_validation',
                serviceId: service.ID,
                serviceName: service.name
            });
            throw error;
        }
    }

    async _validateDataProduct(req) {
        const dataProduct = req.data;
        
        // Validate required fields
        if (!dataProduct.name || dataProduct.name.trim().length === 0) {
            req.error(400, 'Data product name is required', 'name');
        }
        
        if (!dataProduct.category_code) {
            req.error(400, 'Data product category is required', 'category_code');
        }
        
        if (!dataProduct.provider) {
            req.error(400, 'Provider is required', 'provider');
        }

        if (!dataProduct.format) {
            req.error(400, 'Data format is required', 'format');
        }

        // Validate price
        if (dataProduct.price < 0) {
            req.error(400, 'Price cannot be negative', 'price');
        }

        // Validate data size
        if (dataProduct.dataSizeGB && dataProduct.dataSizeGB <= 0) {
            req.error(400, 'Data size must be positive', 'dataSizeGB');
        }

        // Validate JSON fields
        await this._validateJsonField(req, dataProduct.schemaDefinition, 'schemaDefinition');
        await this._validateJsonField(req, dataProduct.validationRules, 'validationRules');
        await this._validateJsonField(req, dataProduct.dataLineage, 'dataLineage');

        // Set computed fields
        if (req.event === 'CREATE') {
            dataProduct.ID = dataProduct.ID || uuid();
            dataProduct.status = dataProduct.status || 'DRAFT';
        }

        // Validate business rules
        await this._validateDataProductBusinessRules(req, dataProduct);
    }

    async _validateServiceRequest(req) {
        const request = req.data;
        
        // Validate required fields
        if (!request.service_ID) {
            req.error(400, 'Service is required', 'service_ID');
        }
        
        if (!request.requester_ID) {
            request.requester_ID = req.user.id;
        }

        if (request.agreedPrice < 0) {
            req.error(400, 'Agreed price cannot be negative', 'agreedPrice');
        }

        // Validate deadline
        if (request.deadline && new Date(request.deadline) <= new Date()) {
            req.error(400, 'Deadline must be in the future', 'deadline');
        }

        // Validate escrow amount
        if (request.escrowAmount < request.agreedPrice) {
            req.error(400, 'Escrow amount must be at least the agreed price', 'escrowAmount');
        }

        // Validate JSON fields
        await this._validateJsonField(req, request.parameters, 'parameters');
        await this._validateJsonField(req, request.configuration, 'configuration');

        // Set computed fields
        if (req.event === 'CREATE') {
            request.ID = request.ID || uuid();
            request.status = 'DRAFT';
            request.progressPercentage = 0;
        }

        // Validate business rules
        await this._validateServiceRequestBusinessRules(req, request);
    }

    async _validateJsonField(req, jsonString, fieldName) {
        if (jsonString) {
            try {
                JSON.parse(jsonString);
            } catch (error) {
                req.error(400, `Invalid JSON format in ${fieldName}`, fieldName);
            }
        }
    }

    async _validateServiceBusinessRules(req, service) {
        // Check if service name is unique for provider
        const existing = await this.db.read('Services', ['ID']).where({
            providerAgent: service.providerAgent,
            name: service.name,
            ID: { '!=': service.ID }
        });
        
        if (existing.length > 0) {
            req.error(409, 'Service name must be unique for provider', 'name');
        }

        // Validate provider agent exists and is active
        const agent = await this.db.read('AgentListings', ['status']).where({
            agentId: service.providerAgent
        });
        
        if (agent.length === 0) {
            req.error(404, 'Provider agent not found', 'providerAgent');
        }
        
        if (agent[0].status !== 'ONLINE') {
            req.warn(400, 'Provider agent is not online', 'providerAgent');
        }

        // Additional business validations...
    }

    async _validateDataProductBusinessRules(req, dataProduct) {
        // Check if data product name is unique for provider
        const existing = await this.db.read('DataProducts', ['ID']).where({
            provider: dataProduct.provider,
            name: dataProduct.name,
            ID: { '!=': dataProduct.ID }
        });
        
        if (existing.length > 0) {
            req.error(409, 'Data product name must be unique for provider', 'name');
        }

        // Validate data governance requirements
        if (dataProduct.privacyClassification === 'RESTRICTED' && !dataProduct.complianceStandards) {
            req.error(400, 'Compliance standards required for restricted data', 'complianceStandards');
        }

        // Additional business validations...
    }

    async _validateServiceRequestBusinessRules(req, request) {
        // Validate service exists and is active
        const service = await this.db.read('Services', ['status', 'maxConcurrentRequests', 'currentActiveRequests']).where({
            ID: request.service_ID
        });
        
        if (service.length === 0) {
            req.error(404, 'Service not found', 'service_ID');
        }
        
        if (service[0].status !== 'ACTIVE') {
            req.error(400, 'Service is not active', 'service_ID');
        }

        // Check concurrent request limits
        if (service[0].currentActiveRequests >= service[0].maxConcurrentRequests) {
            req.error(429, 'Service at capacity', 'service_ID');
        }

        // Additional business validations...
    }

    // ==========================================
    // AUTHORIZATION METHODS
    // ==========================================

    async _checkServiceAuthorization(req) {
        const service = req.data;
        const user = req.user;

        // Check if user is authorized to manage this service
        if (req.event !== 'CREATE') {
            const existing = await this.db.read('Services', ['providerAgent']).where({ ID: service.ID });
            if (existing.length === 0) {
                req.error(404, 'Service not found');
            }
            
            // Only service provider or admin can modify
            if (!user.is('admin') && existing[0].providerAgent !== user.id) {
                req.error(403, 'Insufficient privileges to modify service');
            }
        }

        // For CREATE, check if user owns the provider agent
        if (req.event === 'CREATE') {
            const agent = await this.db.read('AgentListings', ['owner']).where({ agentId: service.providerAgent });
            if (agent.length === 0 || (!user.is('admin') && agent[0].owner !== user.id)) {
                req.error(403, 'Not authorized to create services for this agent');
            }
        }
    }

    async _checkDataProductAuthorization(req) {
        const dataProduct = req.data;
        const user = req.user;

        // Check if user is authorized to manage this data product
        if (req.event !== 'CREATE') {
            const existing = await this.db.read('DataProducts', ['provider']).where({ ID: dataProduct.ID });
            if (existing.length === 0) {
                req.error(404, 'Data product not found');
            }
            
            // Only provider or admin can modify
            if (!user.is('admin') && existing[0].provider !== user.id) {
                req.error(403, 'Insufficient privileges to modify data product');
            }
        }

        // For CREATE, set provider to current user if not admin
        if (req.event === 'CREATE' && !user.is('admin')) {
            dataProduct.provider = user.id;
        }
    }

    async _checkServiceRequestAuthorization(req) {
        const request = req.data;
        const user = req.user;

        // Check if user is authorized to manage this request
        if (req.event !== 'CREATE') {
            const existing = await this.db.read('ServiceRequests', ['requester_ID', 'providerAgent']).where({ ID: request.ID });
            if (existing.length === 0) {
                req.error(404, 'Service request not found');
            }
            
            // Only requester, provider, or admin can modify
            if (!user.is('admin') && 
                existing[0].requester_ID !== user.id && 
                existing[0].providerAgent !== user.id) {
                req.error(403, 'Insufficient privileges to modify service request');
            }
        }

        // For CREATE, set requester to current user
        if (req.event === 'CREATE') {
            request.requester_ID = user.id;
        }
    }

    // ==========================================
    // ENRICHMENT METHODS
    // ==========================================

    async _enrichServices(services) {
        if (!Array.isArray(services)) {
            services = [services];
        }

        for (const service of services) {
            // Calculate computed fields
            await this._calculateServiceMetrics(service);
        }

        return services;
    }

    async _enrichDataProducts(dataProducts) {
        if (!Array.isArray(dataProducts)) {
            dataProducts = [dataProducts];
        }

        for (const dataProduct of dataProducts) {
            // Calculate computed fields
            await this._calculateDataProductMetrics(dataProduct);
        }

        return dataProducts;
    }

    async _enrichAgentListings(agents) {
        if (!Array.isArray(agents)) {
            agents = [agents];
        }

        for (const agent of agents) {
            // Calculate computed fields
            await this._calculateAgentMetrics(agent);
        }

        return agents;
    }

    async _enrichServiceRequests(requests) {
        if (!Array.isArray(requests)) {
            requests = [requests];
        }

        for (const request of requests) {
            // Calculate computed fields
            await this._calculateServiceRequestMetrics(request);
        }

        return requests;
    }

    async _enrichDataIntegrations(integrations) {
        if (!Array.isArray(integrations)) {
            integrations = [integrations];
        }

        for (const integration of integrations) {
            // Calculate computed fields
            await this._calculateIntegrationMetrics(integration);
        }

        return integrations;
    }

    // ==========================================
    // METRIC CALCULATION METHODS
    // ==========================================

    async _calculateServiceMetrics(service) {
        // Get service reviews
        const reviews = await this.db.read('ServiceReviews', ['overallRating'])
            .where({ service_ID: service.ID });

        if (reviews.length > 0) {
            const totalRating = reviews.reduce((sum, review) => sum + review.overallRating, 0);
            service.averageRating = (totalRating / reviews.length).toFixed(2);
            service.totalReviews = reviews.length;
        } else {
            service.averageRating = 0;
            service.totalReviews = 0;
        }

        // Calculate other metrics
        service.isActive = service.status === 'ACTIVE';
        
        // Get provider reputation (simplified)
        service.providerReputation = await this._getProviderReputation(service.providerAgent);
    }

    async _calculateDataProductMetrics(dataProduct) {
        // Get data product reviews
        const reviews = await this.db.read('DataProductReviews', ['overallRating'])
            .where({ dataProduct_ID: dataProduct.ID });

        if (reviews.length > 0) {
            const totalRating = reviews.reduce((sum, review) => sum + review.overallRating, 0);
            dataProduct.qualityScore = (totalRating / reviews.length).toFixed(2);
        } else {
            dataProduct.qualityScore = 0;
        }

        // Get download count
        const purchases = await this.db.read('DataProductPurchases', ['downloadCount'])
            .where({ dataProduct_ID: dataProduct.ID });
        
        dataProduct.totalDownloads = purchases.reduce((sum, purchase) => sum + (purchase.downloadCount || 0), 0);
        dataProduct.isActive = dataProduct.status === 'ACTIVE';
        
        // Format data size
        if (dataProduct.dataSizeGB) {
            dataProduct.dataSizeFormatted = this._formatDataSize(dataProduct.dataSizeGB);
        }
    }

    async _calculateAgentMetrics(agent) {
        // Get agent services
        const services = await this.db.read('Services', ['ID'])
            .where({ providerAgent: agent.agentId });

        agent.totalServices = services.length;
        agent.isOnline = agent.status === 'ONLINE';
        
        // Calculate average rating from service reviews
        if (services.length > 0) {
            const serviceIds = services.map(s => s.ID);
            const reviews = await this.db.read('ServiceReviews', ['overallRating'])
                .where({ service_ID: { in: serviceIds } });
            
            if (reviews.length > 0) {
                const totalRating = reviews.reduce((sum, review) => sum + review.overallRating, 0);
                agent.averageRating = (totalRating / reviews.length).toFixed(2);
            }
        }

        agent.responseTime = agent.avgResponseTimeMs || 0;
    }

    async _calculateServiceRequestMetrics(request) {
        // Calculate progress percentage based on status
        const statusProgress = {
            'DRAFT': 0,
            'SUBMITTED': 10,
            'ACCEPTED': 25,
            'IN_PROGRESS': 50,
            'COMPLETED': 100,
            'CANCELLED': 0,
            'DISPUTED': 75,
            'REFUNDED': 100
        };

        request.progressPercentage = statusProgress[request.status] || 0;
        request.statusText = this._getStatusText(request.status);
        request.canCancel = ['DRAFT', 'SUBMITTED', 'ACCEPTED'].includes(request.status);
        request.canRate = request.status === 'COMPLETED' && !request.serviceRating;
    }

    async _calculateIntegrationMetrics(integration) {
        // Get integration metrics
        const metrics = await this.db.read('DataIntegrationMetrics')
            .where({ integration_ID: integration.ID })
            .orderBy({ metricDate: 'desc' })
            .limit(30); // Last 30 days

        if (metrics.length > 0) {
            const recentMetrics = metrics[0];
            integration.healthScore = this._calculateHealthScore(integration, recentMetrics);
            integration.throughputToday = recentMetrics.recordsProcessed || 0;
            integration.costToday = recentMetrics.totalCost || 0;
        } else {
            integration.healthScore = 0;
            integration.throughputToday = 0;
            integration.costToday = 0;
        }

        integration.canPause = integration.status === 'ACTIVE';
        integration.canResume = integration.status === 'PAUSED';
    }

    // ==========================================
    // ACTION HANDLER METHODS
    // ==========================================

    async _handleRequestService(req) {
        const { parameters, deadline, escrowAmount } = req.data;
        const serviceId = req.params[0].ID;
        const userId = req.user.id;

        try {
            // Validate service
            const service = await this.db.read('Services').where({ ID: serviceId });
            if (!service.length) {
                return req.error(404, 'Service not found');
            }

            // Create service request
            const requestId = uuid();
            const serviceRequest = {
                ID: requestId,
                service_ID: serviceId,
                requester_ID: userId,
                providerAgent: service[0].providerAgent,
                status: 'SUBMITTED',
                agreedPrice: service[0].basePrice,
                escrowAmount: escrowAmount,
                deadline: deadline,
                parameters: parameters,
                currency_code: service[0].currency_code
            };

            await this.db.create('ServiceRequests').entries(serviceRequest);

            // Update service active requests count
            await this.db.update('Services', serviceId).with({
                currentActiveRequests: service[0].currentActiveRequests + 1
            });

            // Emit event for real-time updates
            this.emit('ServiceRequested', {
                requestId: requestId,
                serviceId: serviceId,
                requesterId: userId
            });

            return {
                requestId: requestId,
                status: 'SUBMITTED',
                estimatedCompletion: this._calculateEstimatedCompletion(service[0])
            };

        } catch (error) {
            req.error(500, `Failed to create service request: ${error.message}`);
        }
    }

    async _handleSubmitBid(req) {
        const { bidAmount, estimatedTimeHours, proposal } = req.data;
        const serviceId = req.params[0].ID;
        const userId = req.user.id;

        try {
            const bidId = uuid();
            
            // Store bid (simplified - would use proper bid entity)
            const bid = {
                ID: bidId,
                service_ID: serviceId,
                bidder_ID: userId,
                bidAmount: bidAmount,
                estimatedTimeHours: estimatedTimeHours,
                proposal: proposal,
                status: 'SUBMITTED'
            };

            // In production, save to ServiceBids entity
            
            return {
                bidId: bidId,
                status: 'SUBMITTED'
            };

        } catch (error) {
            req.error(500, `Failed to submit bid: ${error.message}`);
        }
    }

    async _handlePurchaseDataProduct(req) {
        const { licenseType } = req.data;
        const productId = req.params[0].ID;
        const userId = req.user.id;

        try {
            const product = await this.db.read('DataProducts').where({ ID: productId });
            if (!product.length) {
                return req.error(404, 'Data product not found');
            }

            const purchaseId = uuid();
            const licenseKey = this._generateLicenseKey();
            const downloadUrl = this._generateDownloadUrl(productId, licenseKey);
            const expiresAt = this._calculateLicenseExpiry(licenseType);

            // Create purchase record
            const purchase = {
                ID: purchaseId,
                dataProduct_ID: productId,
                purchaser_ID: userId,
                licenseType: licenseType,
                purchasePrice: product[0].price,
                currency_code: product[0].currency_code,
                licenseKey: licenseKey,
                downloadUrl: downloadUrl,
                licenseExpiresAt: expiresAt
            };

            await this.db.create('DataProductPurchases').entries(purchase);

            return {
                purchaseId: purchaseId,
                downloadUrl: downloadUrl,
                licenseKey: licenseKey,
                expiresAt: expiresAt
            };

        } catch (error) {
            req.error(500, `Failed to purchase data product: ${error.message}`);
        }
    }

    async _handlePreviewData(req) {
        const productId = req.params[0].ID;

        try {
            const product = await this.db.read('DataProducts', ['sampleData', 'schemaDefinition'])
                .where({ ID: productId });
            
            if (!product.length) {
                return req.error(404, 'Data product not found');
            }

            return {
                previewUrl: this._generatePreviewUrl(productId),
                sampleData: product[0].sampleData,
                schema: product[0].schemaDefinition
            };

        } catch (error) {
            req.error(500, `Failed to generate preview: ${error.message}`);
        }
    }

    async _handleCancelRequest(req) {
        const { reason } = req.data;
        const requestId = req.params[0].ID;
        const userId = req.user.id;

        try {
            const request = await this.db.read('ServiceRequests').where({ ID: requestId });
            if (!request.length) {
                return req.error(404, 'Service request not found');
            }

            // Check authorization
            if (request[0].requester_ID !== userId && !req.user.is('admin')) {
                return req.error(403, 'Not authorized to cancel this request');
            }

            // Calculate refund amount
            const refundAmount = this._calculateRefund(request[0]);

            // Update request
            await this.db.update('ServiceRequests', requestId).with({
                status: 'CANCELLED',
                cancellationReason: reason,
                refundAmount: refundAmount
            });

            // Process refund (simplified)
            await this._processRefund(request[0], refundAmount);

            return {
                status: 'CANCELLED',
                refundAmount: refundAmount
            };

        } catch (error) {
            req.error(500, `Failed to cancel request: ${error.message}`);
        }
    }

    async _handleRateService(req) {
        const { rating, review } = req.data;
        const requestId = req.params[0].ID;
        const userId = req.user.id;

        try {
            const request = await this.db.read('ServiceRequests').where({ ID: requestId });
            if (!request.length) {
                return req.error(404, 'Service request not found');
            }

            // Check authorization and eligibility
            if (request[0].requester_ID !== userId) {
                return req.error(403, 'Not authorized to rate this service');
            }

            if (request[0].status !== 'COMPLETED') {
                return req.error(400, 'Can only rate completed services');
            }

            // Create review
            const reviewId = uuid();
            const serviceReview = {
                ID: reviewId,
                service_ID: request[0].service_ID,
                reviewer_ID: userId,
                serviceRequest_ID: requestId,
                overallRating: rating,
                reviewText: review,
                isVerifiedPurchase: true
            };

            await this.db.create('ServiceReviews').entries(serviceReview);

            // Update request with rating
            await this.db.update('ServiceRequests', requestId).with({
                serviceRating: rating,
                reviewComments: review
            });

            return { status: 'SUCCESS' };

        } catch (error) {
            req.error(500, `Failed to rate service: ${error.message}`);
        }
    }

    // ==========================================
    // FUNCTION HANDLER METHODS
    // ==========================================

    async _handleGetRecommendations(req) {
        const { preferences, context, limit } = req.data;

        try {
            // Parse preferences
            const userPrefs = JSON.parse(preferences || '{}');
            const contextData = JSON.parse(context || '{}');

            // Get recommendations using AI engine (simplified)
            const recommendations = await this._generateRecommendations(req.user.id, userPrefs, contextData, limit);

            return recommendations;

        } catch (error) {
            req.error(500, `Failed to get recommendations: ${error.message}`);
        }
    }

    async _handleSearchMarketplace(req) {
        const { query, searchType, filters, sortBy, offset, limit } = req.data;

        try {
            const searchResults = await this._performMarketplaceSearch(query, searchType, filters, sortBy, offset, limit);
            return searchResults;

        } catch (error) {
            req.error(500, `Search failed: ${error.message}`);
        }
    }

    async _handleGetMarketplaceOverview(req) {
        const { timeFrame } = req.data;

        try {
            // Get latest marketplace stats
            const stats = await this._calculateMarketplaceStats(timeFrame);
            return stats;

        } catch (error) {
            req.error(500, `Failed to get marketplace overview: ${error.message}`);
        }
    }

    // ==========================================
    // UTILITY METHODS
    // ==========================================

    async _getProviderReputation(agentId) {
        // Simplified reputation calculation
        return 4.2;
    }

    _formatDataSize(sizeGB) {
        if (sizeGB < 1) {
            return `${(sizeGB * 1024).toFixed(1)} MB`;
        } else if (sizeGB < 1024) {
            return `${sizeGB.toFixed(1)} GB`;
        } else {
            return `${(sizeGB / 1024).toFixed(1)} TB`;
        }
    }

    _getStatusText(status) {
        const statusMap = {
            'DRAFT': 'Draft',
            'SUBMITTED': 'Submitted',
            'ACCEPTED': 'Accepted',
            'IN_PROGRESS': 'In Progress',
            'COMPLETED': 'Completed',
            'CANCELLED': 'Cancelled',
            'DISPUTED': 'Disputed',
            'REFUNDED': 'Refunded'
        };
        return statusMap[status] || status;
    }

    _calculateHealthScore(integration, metrics) {
        // Simplified health score calculation
        const successRate = metrics.successCount / Math.max(metrics.executionCount, 1);
        const qualityScore = metrics.dataQualityScore || 0;
        return ((successRate * 0.6) + (qualityScore * 0.4)).toFixed(2);
    }

    _calculateEstimatedCompletion(service) {
        const now = new Date();
        const estimatedMinutes = service.estimatedTimeMinutes || 60;
        return new Date(now.getTime() + estimatedMinutes * 60000);
    }

    _generateLicenseKey() {
        return 'LK-' + uuid().replace(/-/g, '').substring(0, 20).toUpperCase();
    }

    _generateDownloadUrl(productId, licenseKey) {
        return `https://data.a2a.platform/download/${productId}?key=${licenseKey}`;
    }

    _generatePreviewUrl(productId) {
        return `https://data.a2a.platform/preview/${productId}`;
    }

    _calculateLicenseExpiry(licenseType) {
        const now = new Date();
        switch (licenseType) {
            case 'SINGLE_USE':
                return new Date(now.getTime() + 24 * 60 * 60 * 1000); // 24 hours
            case 'SUBSCRIPTION':
                return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000); // 30 days
            case 'ENTERPRISE':
                return new Date(now.getTime() + 365 * 24 * 60 * 60 * 1000); // 1 year
            default:
                return new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000); // 7 days
        }
    }

    _calculateRefund(request) {
        // Simplified refund calculation
        if (['DRAFT', 'SUBMITTED'].includes(request.status)) {
            return request.escrowAmount; // Full refund
        } else if (request.status === 'ACCEPTED') {
            return request.escrowAmount * 0.9; // 10% cancellation fee
        } else {
            return 0; // No refund for in-progress or completed
        }
    }

    async _processRefund(request, amount) {
        // Simplified refund processing
        console.log(`Processing refund of ${amount} ${request.currency_code} for request ${request.ID}`);
        // In production, integrate with payment gateway
    }

    async _generateRecommendations(userId, preferences, context, limit) {
        // Simplified recommendation generation
        // In production, integrate with AI recommendation engine
        return [
            {
                itemId: 'service_001',
                itemType: 'SERVICE',
                matchScore: 0.95,
                reason: 'Based on your previous AI service usage',
                estimatedValue: '15-25% improvement in processing speed'
            },
            {
                itemId: 'data_001',
                itemType: 'DATA_PRODUCT',
                matchScore: 0.88,
                reason: 'High-quality training data for your ML models',
                estimatedValue: '20-30% better model accuracy'
            }
        ];
    }

    async _performMarketplaceSearch(query, searchType, filters, sortBy, offset, limit) {
        // Simplified search implementation
        // In production, use proper search engine like Elasticsearch
        const results = [];
        let totalCount = 0;

        return {
            results: results,
            totalCount: totalCount,
            searchTimeMs: 45,
            suggestions: ['machine learning', 'data analytics', 'blockchain']
        };
    }

    async _calculateMarketplaceStats(timeFrame) {
        // Get actual stats from database
        const services = await this.db.read('Services');
        const dataProducts = await this.db.read('DataProducts');
        const requests = await this.db.read('ServiceRequests');

        return {
            totalRevenue: 125430.50,
            activeServices: services.filter(s => s.status === 'ACTIVE').length,
            activeDataProducts: dataProducts.filter(d => d.status === 'ACTIVE').length,
            totalUsers: 1250,
            transactionsToday: requests.filter(r => {
                const today = new Date().toDateString();
                return new Date(r.createdAt).toDateString() === today;
            }).length,
            growthMetrics: {
                revenueGrowth: 18.5,
                userGrowth: 8.7,
                serviceGrowth: 12.3
            },
            healthMetrics: {
                systemUptime: 99.8,
                avgResponseTime: 1.2,
                errorRate: 0.05
            }
        };
    }

    // ==========================================
    // EVENT HANDLERS
    // ==========================================

    async _onServiceRequestCreated(request) {
        // Send notification to provider
        this.emit('ServiceRequestNotification', {
            type: 'NEW_REQUEST',
            providerId: request.providerAgent,
            requestId: request.ID
        });
    }

    async _onServiceRequestUpdated(request, original) {
        // Send status update notification
        if (request.status !== original.status) {
            this.emit('ServiceRequestStatusUpdate', {
                requestId: request.ID,
                oldStatus: original.status,
                newStatus: request.status,
                requesterId: request.requester_ID
            });
        }
    }

    async _onServiceReviewCreated(review) {
        // Update service rating cache
        this.emit('ServiceReviewAdded', {
            serviceId: review.service_ID,
            rating: review.overallRating
        });
    }

    async _onDataProductReviewCreated(review) {
        // Update data product rating cache
        this.emit('DataProductReviewAdded', {
            dataProductId: review.dataProduct_ID,
            rating: review.overallRating
        });
    }

    // Additional handler methods would be implemented here...
    // For brevity, showing the pattern with key methods

}

module.exports = { MarketplaceService };