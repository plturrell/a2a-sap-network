/*!
 * SAP UI5 Custom Control: Agent Status Indicator
 * Fiori 3.0 Compliant Component
 */

sap.ui.define([
    "sap/ui/core/Control",
    "sap/m/Label",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/ObjectStatus",
    "sap/m/ProgressIndicator",
    "sap/suite/ui/microchart/RadialMicroChart",
    "sap/ui/core/Icon"
], function(
    Control,
    Label,
    VBox,
    HBox,
    ObjectStatus,
    ProgressIndicator,
    RadialMicroChart,
    Icon
) {
    "use strict";

    /**
     * Agent Status Indicator Control
     * Displays agent status in compliance with SAP Fiori 3.0 guidelines
     * @namespace com.sap.a2a.portal.control
     */
    return Control.extend("com.sap.a2a.portal.control.AgentStatusIndicator", {
        metadata: {
            properties: {
                /**
                 * Agent ID
                 */
                agentId: { type: "string", defaultValue: "" },
                
                /**
                 * Agent name
                 */
                agentName: { type: "string", defaultValue: "" },
                
                /**
                 * Agent type/role
                 */
                agentType: { type: "string", defaultValue: "" },
                
                /**
                 * Current status
                 */
                status: { 
                    type: "string", 
                    defaultValue: "inactive",
                    bindable: true
                },
                
                /**
                 * Health percentage (0-100)
                 */
                health: { 
                    type: "int", 
                    defaultValue: 0,
                    bindable: true
                },
                
                /**
                 * Performance score (0-100)
                 */
                performance: { 
                    type: "int", 
                    defaultValue: 0,
                    bindable: true
                },
                
                /**
                 * Response time in milliseconds
                 */
                responseTime: { 
                    type: "int", 
                    defaultValue: 0,
                    bindable: true
                },
                
                /**
                 * Messages processed count
                 */
                messagesProcessed: { 
                    type: "int", 
                    defaultValue: 0,
                    bindable: true
                },
                
                /**
                 * Show detailed metrics
                 */
                showDetails: { 
                    type: "boolean", 
                    defaultValue: true
                },
                
                /**
                 * Compact mode for smaller displays
                 */
                compact: { 
                    type: "boolean", 
                    defaultValue: false
                }
            },
            
            aggregations: {
                /**
                 * Internal content aggregation
                 */
                _content: { 
                    type: "sap.ui.core.Control", 
                    multiple: false, 
                    visibility: "hidden" 
                }
            },
            
            events: {
                /**
                 * Fired when the indicator is pressed
                 */
                press: {
                    parameters: {
                        agentId: { type: "string" }
                    }
                }
            }
        },

        init: function() {
            this._createContent();
        },

        _createContent: function() {
            const oContent = new VBox({
                class: "sapUiSmallMargin"
            });

            // Header with icon and status
            const oHeader = new HBox({
                alignItems: "Center",
                justifyContent: "SpaceBetween"
            });

            this._oIcon = new Icon({
                size: "2rem",
                color: "{= ${status} === 'active' ? 'Positive' : 'Neutral' }"
            });

            this._oStatus = new ObjectStatus({
                state: "{= ${status} === 'active' ? 'Success' : ${status} === 'error' ? 'Error' : 'None' }",
                icon: "{= ${status} === 'active' ? 'sap-icon://status-positive' : ${status} === 'error' ? 'sap-icon://status-negative' : 'sap-icon://status-inactive' }"
            });

            oHeader.addItem(this._oIcon);
            oHeader.addItem(this._oStatus);
            oContent.addItem(oHeader);

            // Agent name and type
            this._oNameLabel = new Label({
                design: "Bold"
            });
            oContent.addItem(this._oNameLabel);

            this._oTypeLabel = new Label({
                class: "sapUiTinyMarginBottom"
            });
            oContent.addItem(this._oTypeLabel);

            // Performance indicator
            this._oPerformanceChart = new RadialMicroChart({
                size: "M",
                press: this._onPress.bind(this)
            });
            
            // Health progress
            this._oHealthProgress = new ProgressIndicator({
                displayValue: "{health}%",
                showValue: true,
                state: "{= ${health} > 80 ? 'Success' : ${health} > 50 ? 'Warning' : 'Error' }"
            });

            // Metrics box
            const oMetricsBox = new VBox({
                visible: "{= !${compact} && ${showDetails} }"
            });

            this._oResponseTime = new ObjectStatus({
                text: "{responseTime} ms",
                state: "{= ${responseTime} < 500 ? 'Success' : ${responseTime} < 1000 ? 'Warning' : 'Error' }"
            });

            this._oMessageCount = new ObjectStatus({
                text: "{messagesProcessed}",
                state: "Information"
            });

            oMetricsBox.addItem(new Label({ text: "Response Time:" }));
            oMetricsBox.addItem(this._oResponseTime);
            oMetricsBox.addItem(new Label({ text: "Messages:" }));
            oMetricsBox.addItem(this._oMessageCount);

            oContent.addItem(this._oPerformanceChart);
            oContent.addItem(this._oHealthProgress);
            oContent.addItem(oMetricsBox);

            this.setAggregation("_content", oContent);
        },

        _updateContent: function() {
            // Update icon based on agent type
            const iconMap = {
                "data_product": "sap-icon://database",
                "standardization": "sap-icon://synchronize",
                "ai_preparation": "sap-icon://artificial-intelligence",
                "vector_processing": "sap-icon://network-settings",
                "calculation": "sap-icon://calculate",
                "qa_validation": "sap-icon://validate",
                "agent_manager": "sap-icon://settings",
                "data_manager": "sap-icon://data-configuration",
                "catalog_manager": "sap-icon://folder-full",
                "agent_builder": "sap-icon://build"
            };

            this._oIcon.setSrc(iconMap[this.getAgentType()] || "sap-icon://group");
            this._oIcon.setColor(this._getStatusColor());
            
            this._oStatus.setText(this._getStatusText());
            this._oStatus.setState(this._getStatusState());
            
            this._oNameLabel.setText(this.getAgentName());
            this._oTypeLabel.setText(this._getAgentTypeText());
            
            this._oPerformanceChart.setPercentage(this.getPerformance());
            this._oPerformanceChart.setValueColor(this._getPerformanceColor());
            
            this._oHealthProgress.setPercentValue(this.getHealth());
            
            this._oResponseTime.setText(this.getResponseTime() + " ms");
            this._oMessageCount.setText(this._formatNumber(this.getMessagesProcessed()));
        },

        _getStatusColor: function() {
            switch (this.getStatus()) {
                case "active": return "Positive";
                case "error": return "Negative";
                case "warning": return "Critical";
                default: return "Neutral";
            }
        },

        _getStatusText: function() {
            switch (this.getStatus()) {
                case "active": return "Active";
                case "error": return "Error";
                case "warning": return "Warning";
                case "inactive": return "Inactive";
                default: return "Unknown";
            }
        },

        _getStatusState: function() {
            switch (this.getStatus()) {
                case "active": return "Success";
                case "error": return "Error";
                case "warning": return "Warning";
                default: return "None";
            }
        },

        _getPerformanceColor: function() {
            const perf = this.getPerformance();
            if (perf >= 80) return "Good";
            if (perf >= 50) return "Critical";
            return "Error";
        },

        _getAgentTypeText: function() {
            const typeMap = {
                "data_product": "Data Product Registration",
                "standardization": "Financial Standardization",
                "ai_preparation": "AI Data Preparation",
                "vector_processing": "Vector Processing",
                "calculation": "Calculation & Validation",
                "qa_validation": "Quality Assurance",
                "agent_manager": "Agent Orchestration",
                "data_manager": "Data Management",
                "catalog_manager": "Catalog Management",
                "agent_builder": "Agent Builder"
            };
            return typeMap[this.getAgentType()] || this.getAgentType();
        },

        _formatNumber: function(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + "M";
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + "K";
            }
            return num.toString();
        },

        _onPress: function() {
            this.firePress({
                agentId: this.getAgentId()
            });
        },

        renderer: {
            apiVersion: 2,
            
            render: function(oRm, oControl) {
                oRm.openStart("div", oControl);
                oRm.class("sapUiAgentStatusIndicator");
                
                if (oControl.getCompact()) {
                    oRm.class("sapUiAgentStatusIndicatorCompact");
                }
                
                // Add semantic ARIA attributes
                oRm.attr("role", "article");
                oRm.attr("aria-label", "Agent Status: " + oControl.getAgentName());
                oRm.attr("aria-live", "polite");
                oRm.attr("aria-atomic", "true");
                
                oRm.openEnd();
                
                oRm.renderControl(oControl.getAggregation("_content"));
                
                oRm.close("div");
            }
        },

        // Property setters that trigger content update
        setStatus: function(sValue) {
            this.setProperty("status", sValue, true);
            this._updateContent();
            return this;
        },

        setHealth: function(iValue) {
            this.setProperty("health", iValue, true);
            this._updateContent();
            return this;
        },

        setPerformance: function(iValue) {
            this.setProperty("performance", iValue, true);
            this._updateContent();
            return this;
        },

        setResponseTime: function(iValue) {
            this.setProperty("responseTime", iValue, true);
            this._updateContent();
            return this;
        },

        setMessagesProcessed: function(iValue) {
            this.setProperty("messagesProcessed", iValue, true);
            this._updateContent();
            return this;
        },

        setAgentName: function(sValue) {
            this.setProperty("agentName", sValue, true);
            this._updateContent();
            return this;
        },

        setAgentType: function(sValue) {
            this.setProperty("agentType", sValue, true);
            this._updateContent();
            return this;
        }
    });
});