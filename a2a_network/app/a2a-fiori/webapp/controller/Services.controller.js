sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter"
], function(Controller, MessageToast, MessageBox, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Services", {
        formatter: formatter,

        onInit: function() {
            this.getRouter().getRoute("services").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched: function() {
            this.getView().getModel().refresh();
        },

        onSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            var oGrid = this.byId("servicesGrid");
            var oBinding = oGrid.getBinding("items");
            
            if (sQuery) {
                oBinding.filter([
                    new sap.ui.model.Filter({
                        filters: [
                            new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("description", sap.ui.model.FilterOperator.Contains, sQuery)
                        ],
                        and: false
                    })
                ]);
            } else {
                oBinding.filter([]);
            }
        },

        onCategoryChange: function(oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            var oGrid = this.byId("servicesGrid");
            var oBinding = oGrid.getBinding("items");
            
            if (sKey && sKey !== "all") {
                oBinding.filter([new sap.ui.model.Filter("category", sap.ui.model.FilterOperator.EQ, sKey)]);
            } else {
                oBinding.filter([]);
            }
        },

        onPriceRangeChange: function(oEvent) {
            var aRange = oEvent.getParameter("range");
            var oGrid = this.byId("servicesGrid");
            var oBinding = oGrid.getBinding("items");
            
            oBinding.filter([
                new sap.ui.model.Filter("pricePerCall", sap.ui.model.FilterOperator.GE, aRange[0]),
                new sap.ui.model.Filter("pricePerCall", sap.ui.model.FilterOperator.LE, aRange[1])
            ]);
        },

        onSortChange: function(oEvent) {
            var iIndex = oEvent.getParameter("selectedIndex");
            var oGrid = this.byId("servicesGrid");
            var oBinding = oGrid.getBinding("items");
            var oSorter;
            
            switch(iIndex) {
                case 0: // Most Popular
                    oSorter = new sap.ui.model.Sorter("totalCalls", true);
                    break;
                case 1: // Highest Rated
                    oSorter = new sap.ui.model.Sorter("averageRating", true);
                    break;
                case 2: // Lowest Price
                    oSorter = new sap.ui.model.Sorter("pricePerCall", false);
                    break;
                case 3: // Newest
                    oSorter = new sap.ui.model.Sorter("createdAt", true);
                    break;
            }
            
            if (oSorter) {
                oBinding.sort(oSorter);
            }
        },

        onListService: function() {
            MessageToast.show("List Service - Coming Soon");
        },

        onServicePress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            MessageToast.show("Service details for: " + oContext.getProperty("name"));
        },

        onOrderService: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sServiceName = oContext.getProperty("name");
            
            MessageBox.confirm("Order service: " + sServiceName + "?", {
                actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                onClose: function(sAction) {
                    if (sAction === MessageBox.Action.YES) {
                        MessageToast.show("Service ordered successfully");
                    }
                }
            });
        },

        getRouter: function() {
            return this.getOwnerComponent().getRouter();
        }
    });
});