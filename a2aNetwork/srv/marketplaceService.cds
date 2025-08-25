using a2a.marketplace as mp from '../db/marketplaceSchema';
using { Currency, Country, Language, User, cuid, managed } from '@sap/cds/common';

/**
 * SAP CAP Marketplace Service
 * Production-grade service definition following SAP best practices
 */
@impl : './marketplaceService.js'
@requires: ['authenticated-user']
service MarketplaceService @(path : '/api/v1/marketplace') {

    // ======================================
    // ENTITIES - READ/WRITE OPERATIONS
    // ======================================
    
    @odata.draft.enabled
    @Common.Label: 'Services'
    @Capabilities: {
        ReadRestrictions: {Readable: true},
        InsertRestrictions: {Insertable: true},
        UpdateRestrictions: {Updatable: true},
        DeleteRestrictions: {Deletable: true}
    }
    entity Services as projection on mp.Services {
        *,
        // Computed fields
        @Core.Computed : true
        averageRating : Decimal(3,2),
        @Core.Computed : true
        totalReviews : Integer,
        @Core.Computed : true
        isActive : Boolean,
        @Core.Computed : true
        providerReputation : Decimal(4,2)
    } actions {
        @Common.Label: 'Request Service'
        @Core.OperationAvailable: isActive
        action requestService(
            @Common.Label: 'Parameters'
            @Core.Description: 'Service request parameters in JSON format'
            parameters: String,
            @Common.Label: 'Deadline'
            deadline: DateTime,
            @Common.Label: 'Escrow Amount'
            @Measures.ISOCurrency: currency_code
            escrowAmount: Decimal(15,2)
        ) returns {
            @Common.Label: 'Request ID'
            requestId: String;
            @Common.Label: 'Status'
            status: String;
            @Common.Label: 'Estimated Completion'
            estimatedCompletion: DateTime;
        };

        @Common.Label: 'Submit Bid'
        @Core.OperationAvailable: isActive
        action submitBid(
            @Common.Label: 'Bid Amount'
            @Measures.ISOCurrency: currency_code
            bidAmount: Decimal(15,2),
            @Common.Label: 'Estimated Time (hours)'
            estimatedTimeHours: Integer,
            @Common.Label: 'Proposal'
            proposal: String(2000)
        ) returns {
            @Common.Label: 'Bid ID'
            bidId: String;
            @Common.Label: 'Status'
            status: String;
        };
    };

    @odata.draft.enabled
    @Common.Label: 'Data Products'
    @Capabilities: {
        ReadRestrictions: {Readable: true},
        InsertRestrictions: {Insertable: true},
        UpdateRestrictions: {Updatable: true},
        DeleteRestrictions: {Deletable: true}
    }
    entity DataProducts as projection on mp.DataProducts {
        *,
        // Computed fields
        @Core.Computed : true
        qualityScore : Decimal(3,2),
        @Core.Computed : true
        totalDownloads : Integer,
        @Core.Computed : true
        isActive : Boolean,
        @Core.Computed : true
        dataSizeFormatted : String
    } actions {
        @Common.Label: 'Purchase Data Product'
        @Core.OperationAvailable: isActive
        action purchaseDataProduct(
            @Common.Label: 'License Type'
            licenseType: String enum {
                SINGLE_USE = 'SINGLE_USE';
                SUBSCRIPTION = 'SUBSCRIPTION';
                ENTERPRISE = 'ENTERPRISE';
            }
        ) returns {
            @Common.Label: 'Purchase ID'
            purchaseId: String;
            @Common.Label: 'Download URL'
            downloadUrl: String;
            @Common.Label: 'License Key'
            licenseKey: String;
            @Common.Label: 'Expires At'
            expiresAt: DateTime;
        };

        @Common.Label: 'Preview Data'
        action previewData() returns {
            @Common.Label: 'Preview URL'
            previewUrl: String;
            @Common.Label: 'Sample Data'
            sampleData: String;
            @Common.Label: 'Schema'
            schema: String;
        };
    };

    @Common.Label: 'Agent Listings'
    @readonly
    entity AgentListings as projection on mp.AgentListings {
        *,
        // Computed fields
        @Core.Computed : true
        averageRating : Decimal(3,2),
        @Core.Computed : true
        totalServices : Integer,
        @Core.Computed : true
        isOnline : Boolean,
        @Core.Computed : true
        responseTime : Integer
    };

    @Common.Label: 'Service Requests'
    @Capabilities: {
        ReadRestrictions: {Readable: true},
        InsertRestrictions: {Insertable: true},
        UpdateRestrictions: {Updatable: true},
        DeleteRestrictions: {Deletable: false}
    }
    entity ServiceRequests as projection on mp.ServiceRequests {
        *,
        // Computed fields
        @Core.Computed : true
        statusText : String,
        @Core.Computed : true
        progressPercentage : Integer,
        @Core.Computed : true
        canCancel : Boolean,
        @Core.Computed : true
        canRate : Boolean
    } actions {
        @Common.Label: 'Cancel Request'
        @Core.OperationAvailable: canCancel
        action cancelRequest(
            @Common.Label: 'Cancellation Reason'
            reason: String(500)
        ) returns {
            @Common.Label: 'Status'
            status: String;
            @Common.Label: 'Refund Amount'
            @Measures.ISOCurrency: currency_code
            refundAmount: Decimal(15,2);
        };

        @Common.Label: 'Rate Service'
        @Core.OperationAvailable: canRate
        action rateService(
            @Common.Label: 'Rating'
            @assert.range: [1, 5]
            rating: Integer,
            @Common.Label: 'Review'
            review: String(1000)
        ) returns {
            @Common.Label: 'Status'
            status: String;
        };
    };

    @Common.Label: 'Data Integrations'
    @Capabilities: {
        ReadRestrictions: {Readable: true},
        InsertRestrictions: {Insertable: true},
        UpdateRestrictions: {Updatable: true},
        DeleteRestrictions: {Deletable: true}
    }
    entity DataIntegrations as projection on mp.DataIntegrations {
        *,
        // Computed fields
        @Core.Computed : true
        healthScore : Decimal(3,2),
        @Core.Computed : true
        throughputToday : Integer,
        @Core.Computed : true
        costToday : Decimal(10,2),
        @Core.Computed : true
        canPause : Boolean,
        @Core.Computed : true
        canResume : Boolean
    } actions {
        @Common.Label: 'Pause Integration'
        @Core.OperationAvailable: canPause
        action pauseIntegration(
            @Common.Label: 'Reason'
            reason: String(200)
        ) returns {
            @Common.Label: 'Status'
            status: String;
        };

        @Common.Label: 'Resume Integration'
        @Core.OperationAvailable: canResume
        action resumeIntegration() returns {
            @Common.Label: 'Status'
            status: String;
        };

        @Common.Label: 'Get Analytics'
        action getAnalytics(
            @Common.Label: 'Time Range'
            timeRange: String enum {
                LAST_HOUR = 'LAST_HOUR';
                LAST_24_HOURS = 'LAST_24_HOURS';
                LAST_7_DAYS = 'LAST_7_DAYS';
                LAST_30_DAYS = 'LAST_30_DAYS';
            }
        ) returns {
            @Common.Label: 'Performance Metrics'
            performanceMetrics: String;
            @Common.Label: 'Cost Metrics'
            costMetrics: String;
            @Common.Label: 'Quality Metrics'
            qualityMetrics: String;
        };
    };

    // ======================================
    // READ-ONLY VIEWS AND CODE LISTS
    // ======================================

    @readonly
    @Common.Label: 'Service Categories'
    entity ServiceCategories as projection on mp.ServiceCategories;

    @readonly
    @Common.Label: 'Data Product Categories'
    entity DataProductCategories as projection on mp.DataProductCategories;

    @readonly
    @Common.Label: 'Service Reviews'
    entity ServiceReviews as projection on mp.ServiceReviews {
        *,
        // Computed fields
        @Core.Computed : true
        isHelpful : Boolean,
        @Core.Computed : true
        timeAgo : String
    };

    @readonly
    @Common.Label: 'Marketplace Statistics'
    entity MarketplaceStats as projection on mp.MarketplaceStats;

    // ======================================
    // ACTIONS AND FUNCTIONS
    // ======================================

    @Common.Label: 'Get Recommendations'
    function getRecommendations(
        @Common.Label: 'User Preferences'
        preferences: String,
        @Common.Label: 'Context'
        context: String,
        @Common.Label: 'Limit'
        @assert.range: [1, 50]
        limit: Integer
    ) returns array of {
        @Common.Label: 'Item ID'
        itemId: String;
        @Common.Label: 'Item Type'
        itemType: String enum {
            SERVICE = 'SERVICE';
            DATA_PRODUCT = 'DATA_PRODUCT';
            AGENT = 'AGENT';
        };
        @Common.Label: 'Match Score'
        matchScore: Decimal(3,2);
        @Common.Label: 'Reason'
        reason: String(500);
        @Common.Label: 'Estimated Value'
        estimatedValue: String;
    };

    @Common.Label: 'Search Marketplace'
    function searchMarketplace(
        @Common.Label: 'Search Query'
        query: String(200),
        @Common.Label: 'Search Type'
        searchType: String enum {
            ALL = 'ALL';
            SERVICES = 'SERVICES';
            DATA_PRODUCTS = 'DATA_PRODUCTS';
            AGENTS = 'AGENTS';
        },
        @Common.Label: 'Filters'
        filters: String,
        @Common.Label: 'Sort By'
        sortBy: String enum {
            RELEVANCE = 'RELEVANCE';
            RATING = 'RATING';
            PRICE_LOW = 'PRICE_LOW';
            PRICE_HIGH = 'PRICE_HIGH';
            NEWEST = 'NEWEST';
            POPULAR = 'POPULAR';
        },
        @Common.Label: 'Offset'
        offset: Integer,
        @Common.Label: 'Limit'
        @assert.range: [1, 100]
        limit: Integer
    ) returns {
        @Common.Label: 'Results'
        results: array of {
            @Common.Label: 'Item ID'
            itemId: String;
            @Common.Label: 'Item Type'
            itemType: String;
            @Common.Label: 'Title'
            title: String;
            @Common.Label: 'Description'
            description: String;
            @Common.Label: 'Provider'
            provider: String;
            @Common.Label: 'Rating'
            rating: Decimal(3,2);
            @Common.Label: 'Price'
            price: Decimal(15,2);
            @Common.Label: 'Currency'
            currency: String(3);
        };
        @Common.Label: 'Total Count'
        totalCount: Integer;
        @Common.Label: 'Search Time'
        searchTimeMs: Integer;
        @Common.Label: 'Suggestions'
        suggestions: array of String;
    };

    @Common.Label: 'Create Integration'
    action createDataIntegration(
        @Common.Label: 'Agent ID'
        @assert.notNull
        agentId: String,
        @Common.Label: 'Service ID'
        @assert.notNull
        serviceId: String,
        @Common.Label: 'Data Product ID'
        @assert.notNull
        dataProductId: String,
        @Common.Label: 'Integration Type'
        integrationType: String enum {
            DATA_PROCESSING = 'DATA_PROCESSING';
            ANALYTICS_ENHANCEMENT = 'ANALYTICS_ENHANCEMENT';
            AI_TRAINING = 'AI_TRAINING';
            REAL_TIME_FEED = 'REAL_TIME_FEED';
            BATCH_PROCESSING = 'BATCH_PROCESSING';
        },
        @Common.Label: 'Configuration'
        configuration: String,
        @Common.Label: 'Frequency'
        frequency: String enum {
            REAL_TIME = 'REAL_TIME';
            HOURLY = 'HOURLY';
            DAILY = 'DAILY';
            WEEKLY = 'WEEKLY';
            ON_DEMAND = 'ON_DEMAND';
        }
    ) returns {
        @Common.Label: 'Integration ID'
        integrationId: String;
        @Common.Label: 'Status'
        status: String;
        @Common.Label: 'Setup Time Estimate'
        setupTimeEstimate: String;
        @Common.Label: 'Monitoring URL'
        monitoringUrl: String;
    };

    @Common.Label: 'Get Integration Recommendations'
    function getIntegrationRecommendations(
        @Common.Label: 'Agent ID'
        @assert.notNull
        agentId: String
    ) returns array of {
        @Common.Label: 'Data Product ID'
        dataProductId: String;
        @Common.Label: 'Data Product Name'
        dataProductName: String;
        @Common.Label: 'Integration Type'
        integrationType: String;
        @Common.Label: 'Compatibility Score'
        compatibilityScore: Decimal(3,2);
        @Common.Label: 'Estimated Improvement'
        estimatedImprovement: String;
        @Common.Label: 'Setup Complexity'
        setupComplexity: String enum {
            LOW = 'LOW';
            MEDIUM = 'MEDIUM';
            HIGH = 'HIGH';
        };
        @Common.Label: 'Cost Estimate'
        costEstimate: {
            @Common.Label: 'Setup Cost'
            setupCost: Decimal(10,2);
            @Common.Label: 'Monthly Cost'
            monthlyCost: Decimal(10,2);
            @Common.Label: 'Per Record Cost'
            perRecordCost: Decimal(8,6);
        };
        @Common.Label: 'Benefits'
        benefits: array of String;
    };

    @Common.Label: 'Track Interaction'
    action trackUserInteraction(
        @Common.Label: 'Item ID'
        itemId: String,
        @Common.Label: 'Item Type'
        itemType: String,
        @Common.Label: 'Interaction Type'
        interactionType: String enum {
            VIEW = 'VIEW';
            CLICK = 'CLICK';
            PURCHASE = 'PURCHASE';
            RATE = 'RATE';
            SHARE = 'SHARE';
            BOOKMARK = 'BOOKMARK';
        },
        @Common.Label: 'Context'
        context: String,
        @Common.Label: 'Rating'
        rating: Integer
    ) returns {
        @Common.Label: 'Status'
        status: String;
    };

    // ======================================
    // ANALYTICS FUNCTIONS
    // ======================================

    @Common.Label: 'Get Marketplace Overview'
    function getMarketplaceOverview(
        @Common.Label: 'Time Frame'
        timeFrame: String enum {
            LAST_HOUR = 'LAST_HOUR';
            LAST_24_HOURS = 'LAST_24_HOURS';
            LAST_7_DAYS = 'LAST_7_DAYS';
            LAST_30_DAYS = 'LAST_30_DAYS';
            LAST_90_DAYS = 'LAST_90_DAYS';
        }
    ) returns {
        @Common.Label: 'Total Revenue'
        @Measures.ISOCurrency: 'USD'
        totalRevenue: Decimal(15,2);
        @Common.Label: 'Active Services'
        activeServices: Integer;
        @Common.Label: 'Active Data Products'
        activeDataProducts: Integer;
        @Common.Label: 'Total Users'
        totalUsers: Integer;
        @Common.Label: 'Transactions Today'
        transactionsToday: Integer;
        @Common.Label: 'Growth Metrics'
        growthMetrics: {
            @Common.Label: 'Revenue Growth'
            revenueGrowth: Decimal(5,2);
            @Common.Label: 'User Growth'
            userGrowth: Decimal(5,2);
            @Common.Label: 'Service Growth'
            serviceGrowth: Decimal(5,2);
        };
        @Common.Label: 'Health Metrics'
        healthMetrics: {
            @Common.Label: 'System Uptime'
            systemUptime: Decimal(5,2);
            @Common.Label: 'Average Response Time'
            avgResponseTime: Decimal(6,3);
            @Common.Label: 'Error Rate'
            errorRate: Decimal(5,4);
        };
    };

    @Common.Label: 'Get Revenue Analytics'
    function getRevenueAnalytics(
        @Common.Label: 'Time Frame'
        timeFrame: String,
        @Common.Label: 'Breakdown'
        breakdown: String enum {
            HOURLY = 'HOURLY';
            DAILY = 'DAILY';
            WEEKLY = 'WEEKLY';
            MONTHLY = 'MONTHLY';
        }
    ) returns {
        @Common.Label: 'Total Revenue'
        @Measures.ISOCurrency: 'USD'
        totalRevenue: Decimal(15,2);
        @Common.Label: 'Revenue Growth'
        revenueGrowth: Decimal(5,2);
        @Common.Label: 'Average Transaction Value'
        avgTransactionValue: Decimal(10,2);
        @Common.Label: 'Revenue by Category'
        revenueByCategory: array of {
            @Common.Label: 'Category'
            category: String;
            @Common.Label: 'Revenue'
            revenue: Decimal(15,2);
            @Common.Label: 'Percentage'
            percentage: Decimal(5,2);
        };
        @Common.Label: 'Revenue Forecast'
        revenueForecast: {
            @Common.Label: 'Next 30 Days'
            next30Days: Decimal(15,2);
            @Common.Label: 'Confidence'
            confidence: Decimal(3,2);
            @Common.Label: 'Trend'
            trend: String;
        };
        @Common.Label: 'Time Series'
        timeSeries: array of {
            @Common.Label: 'Date'
            date: String;
            @Common.Label: 'Revenue'
            revenue: Decimal(15,2);
            @Common.Label: 'Transactions'
            transactions: Integer;
        };
    };

    @Common.Label: 'Get Performance Analytics'
    function getPerformanceAnalytics(
        @Common.Label: 'Time Frame'
        timeFrame: String
    ) returns {
        @Common.Label: 'Service Metrics'
        serviceMetrics: {
            @Common.Label: 'Total Services'
            totalServices: Integer;
            @Common.Label: 'Active Services'
            activeServices: Integer;
            @Common.Label: 'Average Rating'
            avgRating: Decimal(3,2);
            @Common.Label: 'Service Uptime'
            serviceUptime: Decimal(5,2);
            @Common.Label: 'Average Response Time'
            avgResponseTime: Decimal(6,3);
        };
        @Common.Label: 'Top Performing Services'
        topPerformingServices: array of {
            @Common.Label: 'Service ID'
            serviceId: String;
            @Common.Label: 'Service Name'
            serviceName: String;
            @Common.Label: 'Provider'
            provider: String;
            @Common.Label: 'Revenue'
            revenue: Decimal(15,2);
            @Common.Label: 'Requests'
            requests: Integer;
            @Common.Label: 'Success Rate'
            successRate: Decimal(5,2);
            @Common.Label: 'Rating'
            rating: Decimal(3,2);
        };
        @Common.Label: 'Agent Performance'
        agentPerformance: {
            @Common.Label: 'Total Agents'
            totalAgents: Integer;
            @Common.Label: 'Active Agents'
            activeAgents: Integer;
            @Common.Label: 'Average Utilization'
            avgUtilization: Decimal(5,2);
        };
    };

    // ======================================
    // NOTIFICATIONS AND EVENTS
    // ======================================

    @Common.Label: 'Subscribe to Notifications'
    action subscribeToNotifications(
        @Common.Label: 'Notification Types'
        notificationTypes: array of String,
        @Common.Label: 'Delivery Method'
        deliveryMethod: String enum {
            WEBSOCKET = 'WEBSOCKET';
            EMAIL = 'EMAIL';
            PUSH = 'PUSH';
        }
    ) returns {
        @Common.Label: 'Subscription ID'
        subscriptionId: String;
        @Common.Label: 'Status'
        status: String;
    };

    @Common.Label: 'Get Notification Preferences'
    function getNotificationPreferences() returns {
        @Common.Label: 'Preferences'
        preferences: {
            @Common.Label: 'New Services'
            newServices: Boolean;
            @Common.Label: 'Price Changes'
            priceChanges: Boolean;
            @Common.Label: 'Service Updates'
            serviceUpdates: Boolean;
            @Common.Label: 'Recommendations'
            recommendations: Boolean;
            @Common.Label: 'System Alerts'
            systemAlerts: Boolean;
            @Common.Label: 'Email Enabled'
            emailEnabled: Boolean;
            @Common.Label: 'WebSocket Enabled'
            websocketEnabled: Boolean;
        };
    };
}

// ======================================
// ANNOTATIONS FOR FIORI ELEMENTS
// ======================================

annotate MarketplaceService.Services with @(
    UI.SelectionFields: [
        category_code,
        providerAgent,
        pricing,
        isActive
    ],
    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: name,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: category.name,
            Label: 'Category'
        },
        {
            $Type: 'UI.DataField',
            Value: providerAgent,
            Label: 'Provider'
        },
        {
            $Type: 'UI.DataFieldForAnnotation',
            Target: 'averageRating/@UI.DataPoint',
            Label: 'Rating'
        },
        {
            $Type: 'UI.DataField',
            Value: basePrice,
            Label: 'Price',
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: status,
            Label: 'Status',
            Criticality: status
        }
    ],
    UI.HeaderInfo: {
        TypeName: 'Service',
        TypeNamePlural: 'Services',
        Title: {Value: name},
        Description: {Value: description}
    },
    UI.Facets: [
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'General Information',
            Target: '@UI.FieldGroup#GeneralInfo'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Pricing & Terms',
            Target: '@UI.FieldGroup#Pricing'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Performance Metrics',
            Target: '@UI.FieldGroup#Performance'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Label: 'Reviews',
            Target: 'reviews/@UI.LineItem'
        }
    ]
);

annotate MarketplaceService.DataProducts with @(
    UI.SelectionFields: [
        category_code,
        provider,
        format,
        pricing,
        isActive
    ],
    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: name,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: category.name,
            Label: 'Category'
        },
        {
            $Type: 'UI.DataField',
            Value: provider,
            Label: 'Provider'
        },
        {
            $Type: 'UI.DataField',
            Value: format,
            Label: 'Format'
        },
        {
            $Type: 'UI.DataFieldForAnnotation',
            Target: 'qualityScore/@UI.DataPoint',
            Label: 'Quality'
        },
        {
            $Type: 'UI.DataField',
            Value: price,
            Label: 'Price'
        },
        {
            $Type: 'UI.DataField',
            Value: totalDownloads,
            Label: 'Downloads'
        }
    ]
);