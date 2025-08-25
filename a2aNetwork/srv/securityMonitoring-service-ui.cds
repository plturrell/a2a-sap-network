using SecurityMonitoringService from './securityMonitoring-service';

/**
 * SAP Fiori Elements UI Annotations for Security Monitoring
 * Follows SAP Fiori Design Guidelines and UX patterns
 */

//=============================================================================
// Security Alerts - List Report and Object Page
//=============================================================================

annotate SecurityMonitoringService.SecurityAlerts with @(
    UI.HeaderInfo: {
        TypeName: 'Security Alert',
        TypeNamePlural: 'Security Alerts',
        Title: { Value: title },
        Description: { Value: alertType }
    },
    
    // List Report - Main table view
    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: priority,
            Criticality: priorityCriticality,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField', 
            Value: status,
            Criticality: statusCriticality,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: title,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: alertType,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: severity,
            Criticality: {
                $Type: 'UI.CriticalityType',
                $Path: 'severityLevel'
            },
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: sourceIpAddress,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: detectedAt,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: daysSinceDetection,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: slaBreached,
            Criticality: slaBreached,
            ![@UI.Importance]: #High
        }
    ],

    // Selection Fields for filtering
    UI.SelectionFields: [
        status,
        severity, 
        alertType,
        priority,
        sourceIpAddress
    ],

    // Object Page Header
    UI.HeaderFacets: [
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#Priority',
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.ReferenceFacet', 
            Target: '@UI.DataPoint#Status',
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#RiskScore', 
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#ResolutionTime',
            ![@UI.Importance]: #Medium
        }
    ],

    // Key Performance Indicators
    UI.DataPoint #Priority: {
        Value: priority,
        Criticality: priorityCriticality,
        Title: 'Priority Level'
    },
    
    UI.DataPoint #Status: {
        Value: status,
        Criticality: statusCriticality, 
        Title: 'Alert Status'
    },
    
    UI.DataPoint #RiskScore: {
        Value: riskScore,
        Criticality: #Critical,
        Title: 'Risk Score'
    },

    UI.DataPoint #ResolutionTime: {
        Value: resolutionTimeHours,
        Title: 'Resolution Time (Hours)'
    },

    // Object Page content
    UI.Facets: [
        {
            $Type: 'UI.CollectionFacet',
            Label: 'Alert Details',
            ID: 'AlertDetails',
            Facets: [
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Basic Information',
                    Target: '@UI.FieldGroup#BasicInfo'
                },
                {
                    $Type: 'UI.ReferenceFacet', 
                    Label: 'Detection Information',
                    Target: '@UI.FieldGroup#DetectionInfo'
                }
            ]
        },
        {
            $Type: 'UI.CollectionFacet',
            Label: 'Response & Resolution',
            ID: 'ResponseResolution', 
            Facets: [
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Response Actions',
                    Target: '@UI.FieldGroup#ResponseActions'
                },
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Resolution Details', 
                    Target: '@UI.FieldGroup#ResolutionDetails'
                }
            ]
        }
    ],

    UI.FieldGroup #BasicInfo: {
        Data: [
            { Value: alertType },
            { Value: severity },
            { Value: priority },
            { Value: title },
            { Value: description }
        ]
    },

    UI.FieldGroup #DetectionInfo: {
        Data: [
            { Value: detectedAt },
            { Value: sourceIpAddress },
            { Value: affectedUsers },
            { Value: affectedEndpoints },
            { Value: relatedEvents },
            { Value: riskScore },
            { Value: impactScore },
            { Value: confidenceScore }
        ]
    },

    UI.FieldGroup #ResponseActions: {
        Data: [
            { Value: status },
            { Value: acknowledgedBy },
            { Value: acknowledgedAt },
            { Value: actionsCompleted },
            { Value: recommendedActions }
        ]
    },

    UI.FieldGroup #ResolutionDetails: {
        Data: [
            { Value: resolvedBy },
            { Value: resolvedAt }, 
            { Value: resolution },
            { Value: resolutionTimeHours },
            { Value: ticketId },
            { Value: runbookUrl }
        ]
    }
);

//=============================================================================
// Security Events - Analytical List Page
//=============================================================================

annotate SecurityMonitoringService.SecurityEvents with @(
    UI.HeaderInfo: {
        TypeName: 'Security Event',
        TypeNamePlural: 'Security Events'
    },

    // Analytical List Page configuration
    UI.Chart: {
        Title: 'Security Events Over Time',
        ChartType: #Column,
        Dimensions: [eventType],
        Measures: [threatScore],
        MeasureAttributes: [{
            Measure: threatScore,
            Role: #Axis1
        }]
    },

    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: eventType,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField', 
            Value: severity,
            Criticality: severityLevel,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: description,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: ipAddress,
            ![@UI.Importance]: #Medium  
        },
        {
            $Type: 'UI.DataField',
            Value: userId,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: endpoint,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: threatScore,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: createdAt,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: blocked,
            Criticality: blocked,
            ![@UI.Importance]: #High
        }
    ],

    UI.SelectionFields: [
        eventType,
        severity,
        category,
        ipAddress,
        userId,
        blocked,
        createdAt
    ]
);

//=============================================================================
// Security Metrics - Overview Page  
//=============================================================================

annotate SecurityMonitoringService.SecurityMetrics with @(
    UI.HeaderInfo: {
        TypeName: 'Security Metrics',
        TypeNamePlural: 'Security Metrics Period'
    },

    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: periodType,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: periodStart,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: totalEvents,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: criticalEvents,
            Criticality: #Critical,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: totalAlerts,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: complianceScore,
            Criticality: complianceCriticality,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: avgThreatScore,
            Criticality: threatLevelCriticality,
            ![@UI.Importance]: #High
        }
    ],

    UI.HeaderFacets: [
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#ComplianceScore'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#TotalEvents'
        },
        {
            $Type: 'UI.ReferenceFacet', 
            Target: '@UI.DataPoint#CriticalEvents'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#AvgThreatScore'
        }
    ],

    UI.DataPoint #ComplianceScore: {
        Value: complianceScore,
        Criticality: complianceCriticality,
        Title: 'Compliance Score (%)'
    },

    UI.DataPoint #TotalEvents: {
        Value: totalEvents,
        Title: 'Total Security Events'
    },

    UI.DataPoint #CriticalEvents: {
        Value: criticalEvents,
        Criticality: #Critical,
        Title: 'Critical Events'
    },

    UI.DataPoint #AvgThreatScore: {
        Value: avgThreatScore,
        Criticality: threatLevelCriticality,
        Title: 'Average Threat Score'
    }
);

//=============================================================================
// Blocked IPs - Management View
//=============================================================================

annotate SecurityMonitoringService.BlockedIPs with @(
    UI.HeaderInfo: {
        TypeName: 'Blocked IP Address',
        TypeNamePlural: 'Blocked IP Addresses',
        Title: { Value: ipAddress },
        Description: { Value: reason }
    },

    UI.LineItem: [
        {
            $Type: 'UI.DataField',
            Value: ipAddress,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: blockType,
            Criticality: blockTypeCriticality,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: isActive,
            Criticality: isActive,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: reason,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: country,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField', 
            Value: organization,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: blockedAt,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: expiresAt,
            ![@UI.Importance]: #High
        },
        {
            $Type: 'UI.DataField',
            Value: daysSinceBlocked,
            ![@UI.Importance]: #Medium
        },
        {
            $Type: 'UI.DataField',
            Value: hoursUntilUnblock,
            ![@UI.Importance]: #Medium
        }
    ],

    UI.SelectionFields: [
        blockType,
        isActive,
        country,
        organization,
        reason
    ]
);

//=============================================================================
// Security Dashboard - Executive Summary
//=============================================================================

annotate SecurityMonitoringService.SecurityDashboard with @(
    UI.HeaderInfo: {
        TypeName: 'Security Dashboard',
        TypeNamePlural: 'Security Overview'
    },

    UI.HeaderFacets: [
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#OverallRisk'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#ComplianceScore'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#SystemHealth'
        },
        {
            $Type: 'UI.ReferenceFacet',
            Target: '@UI.DataPoint#ActiveAlerts'
        }
    ],

    UI.DataPoint #OverallRisk: {
        Value: overallRiskScore,
        Criticality: #Critical,
        Title: 'Overall Risk Score',
        TrendCalculation: {
            ReferenceValue: 50.0,
            IsRelativeDifference: true
        }
    },

    UI.DataPoint #ComplianceScore: {
        Value: complianceScore,
        Criticality: #Positive,
        Title: 'Compliance Score (%)'
    },

    UI.DataPoint #SystemHealth: {
        Value: systemHealthScore,
        Criticality: #Information,
        Title: 'System Health Score'
    },

    UI.DataPoint #ActiveAlerts: {
        Value: totalActiveAlerts,
        Criticality: #Negative,
        Title: 'Active Security Alerts'
    },

    UI.Facets: [
        {
            $Type: 'UI.CollectionFacet',
            Label: 'Current Status',
            ID: 'CurrentStatus',
            Facets: [
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Alert Summary',
                    Target: '@UI.FieldGroup#AlertSummary'
                },
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Security Posture',
                    Target: '@UI.FieldGroup#SecurityPosture'
                }
            ]
        },
        {
            $Type: 'UI.CollectionFacet',
            Label: 'Activity & Performance',
            ID: 'ActivityPerformance',
            Facets: [
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'Recent Activity',
                    Target: '@UI.FieldGroup#RecentActivity'
                },
                {
                    $Type: 'UI.ReferenceFacet',
                    Label: 'System Performance',
                    Target: '@UI.FieldGroup#SystemPerformance'
                }
            ]
        }
    ],

    UI.FieldGroup #AlertSummary: {
        Data: [
            { Value: totalActiveAlerts },
            { Value: criticalAlerts },
            { Value: highPriorityAlerts },
            { Value: slaBreachCount }
        ]
    },

    UI.FieldGroup #SecurityPosture: {
        Data: [
            { Value: overallRiskScore },
            { Value: complianceScore },
            { Value: systemHealthScore },
            { Value: blockedIPsCount },
            { Value: quarantinedUsersCount }
        ]
    },

    UI.FieldGroup #RecentActivity: {
        Data: [
            { Value: newEventsToday },
            { Value: blockedAttacksToday },
            { Value: newAlertsToday },
            { Value: resolvedAlertsToday }
        ]
    },

    UI.FieldGroup #SystemPerformance: {
        Data: [
            { Value: avgResponseTime },
            { Value: systemAvailability },
            { Value: successRate }
        ]
    }
);

//=============================================================================
// Common Annotations
//=============================================================================

// Standard criticality colors for all entities
annotate SecurityMonitoringService with @(
    Common.SideEffects.SourceProperties: ['status'],
    Common.SideEffects.TargetProperties: ['statusCriticality', 'priorityCriticality']
);

// Value help for common fields
annotate SecurityMonitoringService.SecurityAlerts with {
    status @(
        Common.ValueList: {
            CollectionPath: 'SecurityEventTypes',
            Parameters: [
                {
                    $Type: 'Common.ValueListParameterInOut',
                    LocalDataProperty: status,
                    ValueListProperty: 'status'
                }
            ]
        }
    );
    
    severity @(
        Common.ValueList: {
            CollectionPath: 'SecurityEventTypes', 
            Parameters: [
                {
                    $Type: 'Common.ValueListParameterInOut',
                    LocalDataProperty: severity,
                    ValueListProperty: 'defaultSeverity'
                }
            ]
        }
    );
};