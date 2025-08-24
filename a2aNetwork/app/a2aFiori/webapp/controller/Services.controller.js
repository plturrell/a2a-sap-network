sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter"
], (Controller, MessageToast, MessageBox, formatter) => {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Services", {
        formatter,

        onInit() {
            this.getRouter().getRoute("services").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched() {
            this.getView().getModel().refresh();
        },

        onSearch(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oGrid = this.byId("servicesGrid");
            const _oBinding = oGrid.getBinding("items");

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

        onCategoryChange(oEvent) {
            const sKey = oEvent.getSource().getSelectedKey();
            const oGrid = this.byId("servicesGrid");
            const _oBinding = oGrid.getBinding("items");

            if (sKey && sKey !== "all") {
                oBinding.filter([new sap.ui.model.Filter("category", sap.ui.model.FilterOperator.EQ, sKey)]);
            } else {
                oBinding.filter([]);
            }
        },

        onPriceRangeChange(oEvent) {
            const aRange = oEvent.getParameter("range");
            const oGrid = this.byId("servicesGrid");
            const _oBinding = oGrid.getBinding("items");

            oBinding.filter([
                new sap.ui.model.Filter("pricePerCall", sap.ui.model.FilterOperator.GE, aRange[0]),
                new sap.ui.model.Filter("pricePerCall", sap.ui.model.FilterOperator.LE, aRange[1])
            ]);
        },

        onSortChange(oEvent) {
            const iIndex = oEvent.getParameter("selectedIndex");
            const oGrid = this.byId("servicesGrid");
            const _oBinding = oGrid.getBinding("items");
            let oSorter;

            switch (iIndex) {
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

        onListService() {
            MessageToast.show("List Service - Coming Soon");
        },

        onServicePress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext();
            MessageToast.show(`Service details for: ${ oContext.getProperty("name")}`);
        },

        onOrderService(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext();
            const sServiceName = oContext.getProperty("name");

            MessageBox.confirm(`Order service: ${ sServiceName }?`, {
                actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                onClose(sAction) {
                    if (sAction === MessageBox.Action.YES) {
                        MessageToast.show("Service ordered successfully");
                    }
                }
            });
        },

        getRouter() {
            return this.getOwnerComponent().getRouter();
        }
    });
});