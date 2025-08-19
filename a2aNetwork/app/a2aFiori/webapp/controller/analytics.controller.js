sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "../model/formatter"
], function(Controller, MessageToast, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Analytics", {
        formatter,

        onInit() {
            this.getRouter().getRoute("analytics").attachPatternMatched(this._onRouteMatched, this);

            // Initialize sample data for charts
            this._initializeChartData();
        },

        _onRouteMatched() {
            this.getView().getModel().refresh();
            this._updateCharts();
        },

        _initializeChartData() {
            // Sample data for trend chart
            const trendData = [];
            const today = new Date();
            for (let i = 30; i >= 0; i--) {
                const date = new Date(today);
                date.setDate(date.getDate() - i);
                trendData.push({
                    date: date.toISOString().split("T")[0],
                    activeAgents: Math.floor(Math.random() * 50) + 50,
                    totalMessages: Math.floor(Math.random() * 1000) + 500
                });
            }

            const oTrendModel = new sap.ui.model.json.JSONModel(trendData);
            this.byId("agentTrendChart").setModel(oTrendModel);

            // Sample data for category chart
            const categoryData = [
                { category: "Computation", count: 25 },
                { category: "Storage", count: 15 },
                { category: "Analysis", count: 30 },
                { category: "Communication", count: 20 },
                { category: "Governance", count: 10 }
            ];

            const oCategoryModel = new sap.ui.model.json.JSONModel(categoryData);
            this.byId("serviceCategoryChart").setModel(oCategoryModel);
        },

        _updateCharts() {
            // Refresh chart data
            const oVizFrame = this.byId("agentTrendChart");
            oVizFrame.getVizProperties().title.text = "Agent Activity - Last 30 Days";

            const oCategoryChart = this.byId("serviceCategoryChart");
            oCategoryChart.getVizProperties().title.text = "Services by Category";
        },

        onExportReport() {
            MessageToast.show("Export functionality coming soon");
        },

        onDateRangeChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            MessageToast.show(`Date range changed: ${ sValue}`);
            this._updateCharts();
        },

        getRouter() {
            return this.getOwnerComponent().getRouter();
        }
    });
});