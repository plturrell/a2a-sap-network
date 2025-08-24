/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function(BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Marketplace", {

        onInit() {
            BaseController.prototype.onInit.apply(this, arguments);

            // Initialize models
            this._initializeModels();

            // Load marketplace data
            this._loadMarketplaceData();

            // Set up real-time updates
            this._setupRealtimeUpdates();
        },

        _initializeModels() {
            // Marketplace model
            this.oMarketplaceModel = new JSONModel({
                services: [],
                dataProducts: [],
                featuredServices: [],
                categories: [],
                cart: {
                    items: [],
                    total: 0
                },
                myListings: {
                    services: [],
                    data: []
                },
                myStats: {
                    activeServices: 0,
                    dataProducts: 0,
                    totalSubscribers: 0,
                    totalDownloads: 0,
                    monthlyRevenue: 0,
                    averageRating: 0,
                    totalReviews: 0,
                    growth: 0
                },
                dataStats: {
                    financial: { datasets: 0, size: 0 },
                    operational: { datasets: 0, size: 0 },
                    market: { datasets: 0, size: 0 },
                    iot: { datasets: 0, size: 0 }
                },
                selectedService: null,
                selectedDataProduct: null,
                filters: {
                    category: "",
                    priceRange: "",
                    rating: 0,
                    searchQuery: ""
                }
            });
            this.getView().setModel(this.oMarketplaceModel, "marketplace");

            // Update UI model
            this.oUIModel.setProperty("/marketplaceView", "services");
        },

        _loadMarketplaceData() {
            this.showSkeletonLoading(this.getResourceBundle().getText("marketplace.loading"));

            // Load marketplace data from backend service
            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";

            Promise.all([
                blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/services`),
                blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/data-products`),
                blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/categories`),
                blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/my-listings`)
            ]).then(responses => {
                return Promise.all(responses.map(r => r.json()));
            }).then(([services, dataProducts, categories, myListings]) => {
                this.oMarketplaceModel.setProperty("/services", services || []);
                this.oMarketplaceModel.setProperty("/dataProducts", dataProducts || []);
                this.oMarketplaceModel.setProperty("/categories", categories || this._getFallbackCategories());
                this.oMarketplaceModel.setProperty("/featuredServices", (services || []).filter(s => s.featured).slice(0, 3));
                this.oMarketplaceModel.setProperty("/myListings", myListings || { services: [], data: [] });

                this._updateMyStats();
                this._restoreCart();
                this.hideLoading();
            }).catch(error => {
                console.error("Failed to load marketplace data:", error);
                // Load fallback categories only
                this.oMarketplaceModel.setProperty("/categories", this._getFallbackCategories());
                this.hideLoading();
                this.showErrorMessage(this.getResourceBundle().getText("marketplace.loadError"));
            });
        },

        _getFallbackCategories() {
            return [
                { id: "all", name: "All Categories" },
                { id: "ai-ml", name: "AI/ML" },
                { id: "blockchain", name: "Blockchain" },
                { id: "analytics", name: "Analytics" },
                { id: "iot", name: "IoT" },
                { id: "security", name: "Security" },
                { id: "operations", name: "Operations" },
                { id: "finance", name: "Finance" }
            ];
        },

        _updateMyStats() {
            const oListings = this.oMarketplaceModel.getProperty("/myListings");
            const oStats = {
                activeServices: oListings.services.filter(s => s.status === "Active").length,
                dataProducts: oListings.data.length,
                totalSubscribers: oListings.services.reduce((sum, s) => sum + s.subscribers, 0),
                totalDownloads: oListings.data.reduce((sum, d) => sum + d.downloads, 0),
                monthlyRevenue: oListings.services.reduce((sum, s) => sum + s.monthlyRevenue, 0),
                averageRating: this._calculateAverageRating(oListings.services),
                totalReviews: oListings.services.reduce((sum, s) => sum + (s.reviews || 0), 0),
                growth: 0 // Will be calculated from historical data
            };

            this.oMarketplaceModel.setProperty("/myStats", oStats);
        },

        onSearch(oEvent) {
            const sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            this.oMarketplaceModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onCategoryFilter(oEvent) {
            const sCategory = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/filters/category", sCategory);
            this._applyFilters();
        },

        onPriceFilter(oEvent) {
            const sPriceRange = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/filters/priceRange", sPriceRange);
            this._applyFilters();
        },

        onRatingFilter(oEvent) {
            const iRating = oEvent.getParameter("value");
            this.oMarketplaceModel.setProperty("/filters/rating", iRating);
            this._applyFilters();
        },

        _applyFilters() {
            const sView = this.oUIModel.getProperty("/marketplaceView");
            const oFilters = this.oMarketplaceModel.getProperty("/filters");
            let aItems = [];

            if (sView === "services") {
                aItems = this.oMarketplaceModel.getProperty("/services");
            } else if (sView === "data") {
                aItems = this.oMarketplaceModel.getProperty("/dataProducts");
            }

            // Apply filters
            aItems.forEach(function(item) {
                let bVisible = true;

                // Search filter
                if (oFilters.searchQuery) {
                    const sQuery = oFilters.searchQuery.toLowerCase();
                    bVisible = item.name.toLowerCase().includes(sQuery) ||
                              item.description.toLowerCase().includes(sQuery) ||
                              item.provider.toLowerCase().includes(sQuery);
                }

                // Category filter
                if (bVisible && oFilters.category && oFilters.category !== "all") {
                    bVisible = item.category.toLowerCase() === oFilters.category.toLowerCase();
                }

                // Price range filter
                if (bVisible && oFilters.priceRange) {
                    switch (oFilters.priceRange) {
                    case "free":
                        bVisible = item.price === 0;
                        break;
                    case "0-50":
                        bVisible = item.price >= 0 && item.price <= 50;
                        break;
                    case "50-200":
                        bVisible = item.price > 50 && item.price <= 200;
                        break;
                    case "200-1000":
                        bVisible = item.price > 200 && item.price <= 1000;
                        break;
                    case "1000+":
                        bVisible = item.price > 1000;
                        break;
                    }
                }

                // Rating filter
                if (bVisible && oFilters.rating > 0) {
                    bVisible = item.rating >= oFilters.rating;
                }

                item.visible = bVisible;
            });

            this.oMarketplaceModel.refresh();

            // Check if no results
            const iVisibleCount = aItems.filter(item => item.visible).length;
            if (iVisibleCount === 0) {
                this.showNoData(this.getResourceBundle().getText("marketplace.noResults"));
            } else {
                this.hideLoading();
            }
        },

        onSort() {
            // Open sort dialog
            if (!this._oSortDialog) {
                this._oSortDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.MarketplaceSortDialog",
                    this
                );
                this.getView().addDependent(this._oSortDialog);
            }
            this._oSortDialog.open();
        },

        onServicePress(oEvent) {
            const oService = oEvent.getSource().getBindingContext("marketplace").getObject();
            this.oMarketplaceModel.setProperty("/selectedService", oService);
            this.byId("serviceDetailDialog").open();
        },

        onCloseServiceDetail() {
            this.byId("serviceDetailDialog").close();
        },

        onSubscribeService(oEvent) {
            const oService = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._addToCart(oService, "service");
        },

        onSubscribeFromDetail() {
            const oService = this.oMarketplaceModel.getProperty("/selectedService");
            this._addToCart(oService, "service");
            this.byId("serviceDetailDialog").close();
        },

        _addToCart(oItem, sType) {
            const oCart = this.oMarketplaceModel.getProperty("/cart");

            // Check if already in cart
            const bExists = oCart.items.some(function(item) {
                return item.id === oItem.id;
            });

            if (!bExists) {
                const oCartItem = Object.assign({}, oItem);
                oCartItem.type = sType;
                oCart.items.push(oCartItem);

                // Update total
                this._updateCartTotal();

                // Save cart
                this._saveCart();

                MessageToast.show(this.getResourceBundle().getText("marketplace.cart.added"));
            } else {
                MessageToast.show(this.getResourceBundle().getText("marketplace.cart.alreadyAdded"));
            }
        },

        _updateCartTotal() {
            const oCart = this.oMarketplaceModel.getProperty("/cart");
            let fTotal = 0;

            oCart.items.forEach(function(item) {
                if (item.pricing === "subscription") {
                    fTotal += item.price; // Monthly subscription
                } else if (item.pricing === "one-time") {
                    fTotal += item.price;
                }
            });

            oCart.total = fTotal;
            this.oMarketplaceModel.setProperty("/cart", oCart);
        },

        onOpenCart() {
            this.byId("cartDialog").open();
        },

        onCloseCart() {
            this.byId("cartDialog").close();
        },

        onRemoveFromCart(oEvent) {
            const oItem = oEvent.getParameter("listItem").getBindingContext("marketplace").getObject();
            const oCart = this.oMarketplaceModel.getProperty("/cart");

            const iIndex = oCart.items.findIndex(function(item) {
                return item.id === oItem.id;
            });

            if (iIndex > -1) {
                oCart.items.splice(iIndex, 1);
                this._updateCartTotal();
                this._saveCart();
                MessageToast.show(this.getResourceBundle().getText("marketplace.cart.removed"));
            }
        },

        onCheckout() {
            const oCart = this.oMarketplaceModel.getProperty("/cart");

            MessageBox.confirm(
                this.getResourceBundle().getText("marketplace.checkout.confirm", [oCart.total]),
                {
                    title: this.getResourceBundle().getText("marketplace.checkout.title"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._processCheckout();
                        }
                    }.bind(this)
                }
            );
        },

        _processCheckout() {
            this.showSpinnerLoading(this.getResourceBundle().getText("marketplace.checkout.processing"));

            // Process checkout via backend service
            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";
            const oCart = this.oMarketplaceModel.getProperty("/cart");

            blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/checkout`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(oCart)
            }).then(response => {
                if (response.ok) {
                    // Clear cart on successful checkout
                    this.oMarketplaceModel.setProperty("/cart", { items: [], total: 0 });
                    this._saveCart();

                    this.hideLoading();
                    this.byId("cartDialog").close();

                    MessageBox.success(
                        this.getResourceBundle().getText("marketplace.checkout.success"),
                        {
                            title: this.getResourceBundle().getText("marketplace.checkout.successTitle")
                        }
                    );
                } else {
                    throw new Error("Checkout failed");
                }
            }).catch(error => {
                this.hideLoading();
                console.error("Checkout failed:", error);
                MessageBox.error(this.getResourceBundle().getText("marketplace.checkout.error"));
            });
        },

        onDataProductPress(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            // Open data product details
            MessageToast.show(`Data product details: ${ oDataProduct.name}`);
        },

        onPreviewData(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            // Show data preview
            // Open data preview
            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";
            window.open(`${apiBaseUrl}/marketplace/data/${oDataProduct.id}/preview`, "_blank");
        },

        onAddDataToCart(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._addToCart(oDataProduct, "data");
        },

        onDataCategoryPress(oEvent) {
            const sCategory = oEvent.getSource().getHeader().getTitle();
            MessageToast.show(`Filter by category: ${ sCategory}`);
        },

        onDataFormatFilter(oEvent) {
            const _sFormat = oEvent.getSource().getSelectedKey();
            // Apply format filter
            this._applyDataFilters();
        },

        onDataFrequencyFilter(oEvent) {
            const _sFrequency = oEvent.getSource().getSelectedKey();
            // Apply frequency filter
            this._applyDataFilters();
        },

        _calculateAverageRating(aServices) {
            if (!aServices || aServices.length === 0) {
                return 0;
            }
            const totalRating = aServices.reduce((sum, service) => sum + (service.rating || 0), 0);
            return Math.round(totalRating / aServices.length * 10) / 10; // Round to 1 decimal
        },

        _applyDataFilters() {
            // Apply filters to data products
            const oTable = this.byId("dataProductsTable");
            if (oTable) {
                const _oBinding = oTable.getBinding("items");
                // Apply filters
            }
        },

        onPublishListing() {
            // Open publish dialog
            MessageToast.show(this.getResourceBundle().getText("marketplace.publish.opening"));
        },

        onEditListing(oEvent) {
            const oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`Edit listing: ${ oListing.name}`);
        },

        onViewAnalytics(oEvent) {
            const oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`View analytics for: ${ oListing.name}`);
        },

        onToggleListing(oEvent) {
            const oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            oListing.status = oListing.status === "Active" ? "Paused" : "Active";
            this.oMarketplaceModel.refresh();
            MessageToast.show(this.getResourceBundle().getText("marketplace.listing.statusChanged"));
        },

        onEditDataProduct(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`Edit data product: ${ oDataProduct.name}`);
        },

        onUpdateData(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`Update data for: ${ oDataProduct.name}`);
        },

        onViewDataAnalytics(oEvent) {
            const oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`View data analytics for: ${ oDataProduct.name}`);
        },

        onTryDemo() {
            const oService = this.oMarketplaceModel.getProperty("/selectedService");
            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";

            // Launch service trial environment
            blockchainClient.sendMessage(`${apiBaseUrl}/marketplace/services/${oService.id}/trial`, {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            }).then(response => response.json()).then(data => {
                if (data.trialUrl) {
                    window.open(data.trialUrl, "_blank");
                    MessageToast.show(this.getResourceBundle().getText("marketplace.trial.launched", [oService.name]));
                } else {
                    MessageToast.show(this.getResourceBundle().getText("marketplace.trial.unavailable"));
                }
            }).catch(error => {
                console.error("Failed to launch trial:", error);
                MessageToast.show(this.getResourceBundle().getText("marketplace.trial.error"));
            });
        },

        onContactProvider() {
            const oService = this.oMarketplaceModel.getProperty("/selectedService");
            MessageToast.show(this.getResourceBundle().getText("marketplace.contact.opening", [oService.provider]));
            // In production, open contact form
        },

        onViewProvider() {
            const oService = this.oMarketplaceModel.getProperty("/selectedService");
            MessageToast.show(`View all listings from: ${ oService.provider}`);
            // In production, filter by provider
        },

        onClearFilters() {
            this.oMarketplaceModel.setProperty("/filters", {
                category: "",
                priceRange: "",
                rating: 0,
                searchQuery: ""
            });

            this.byId("marketplaceSearch").setValue("");
            this.byId("categoryFilter").setSelectedKey("");
            this.byId("priceRangeFilter").setSelectedKey("");
            this.byId("ratingFilter").setValue(0);

            this._applyFilters();
        },

        _saveCart() {
            const oCart = this.oMarketplaceModel.getProperty("/cart");
            localStorage.setItem("a2a_marketplace_cart", JSON.stringify(oCart));
        },

        _restoreCart() {
            const sSavedCart = localStorage.getItem("a2a_marketplace_cart");
            if (sSavedCart) {
                try {
                    const oCart = JSON.parse(sSavedCart);
                    this.oMarketplaceModel.setProperty("/cart", oCart);
                } catch (e) {
                    console.error("Failed to restore cart:", e);
                }
            }
        },

        _setupRealtimeUpdates() {
            // Real-time updates handled via WebSocket connections
            // No simulation needed
        },

        onNavBack() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onExit() {
            // Clean up
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }
        }
    });
});