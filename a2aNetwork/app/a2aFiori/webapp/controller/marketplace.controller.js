sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Marketplace", {

        onInit: function () {
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Initialize models
            this._initializeModels();
            
            // Load marketplace data
            this._loadMarketplaceData();
            
            // Set up real-time updates
            this._setupRealtimeUpdates();
        },

        _initializeModels: function() {
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
                    financial: { datasets: 124, size: 2.4 },
                    operational: { datasets: 89, size: 1.8 },
                    market: { datasets: 156, size: 3.2 },
                    iot: { datasets: 234, size: 5.6 }
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

        _loadMarketplaceData: function() {
            this.showSkeletonLoading(this.getResourceBundle().getText("marketplace.loading"));
            
            // Simulate loading marketplace data - in production, call backend service
            setTimeout(function() {
                var aServices = this._generateServices();
                var aDataProducts = this._generateDataProducts();
                var aCategories = this._generateCategories();
                var aFeatured = aServices.filter(s => s.featured).slice(0, 3);
                var oMyListings = this._generateMyListings();
                
                this.oMarketplaceModel.setProperty("/services", aServices);
                this.oMarketplaceModel.setProperty("/dataProducts", aDataProducts);
                this.oMarketplaceModel.setProperty("/categories", aCategories);
                this.oMarketplaceModel.setProperty("/featuredServices", aFeatured);
                this.oMarketplaceModel.setProperty("/myListings", oMyListings);
                
                this._updateMyStats();
                this._restoreCart();
                this.hideLoading();
            }.bind(this), 1000);
        },

        _generateServices: function() {
            var aServices = [
                {
                    id: "srv_001",
                    name: "Intelligent Document Processor",
                    provider: "SAP AI Services",
                    description: "Extract structured data from unstructured documents using advanced AI",
                    detailedDescription: "Our Intelligent Document Processor uses state-of-the-art machine learning to extract, classify, and validate information from various document types including invoices, contracts, and forms.",
                    icon: "sap-icon://document-text",
                    image: "/images/services/doc-processor.png",
                    category: "AI/ML",
                    capability1: "OCR",
                    capability2: "NLP",
                    price: 199,
                    pricing: "subscription",
                    status: "Active",
                    rating: 4.8,
                    reviewCount: 234,
                    featured: true,
                    features: [
                        "Support for 50+ document types",
                        "99.5% accuracy rate",
                        "Multi-language support",
                        "Custom model training",
                        "REST API integration"
                    ],
                    requirements: [
                        "Minimum 1000 documents/month",
                        "API key required",
                        "HTTPS endpoint"
                    ],
                    reviews: [
                        { user: "john.doe@company.com", rating: 5, comment: "Excellent accuracy and easy integration", date: new Date() },
                        { user: "sarah.smith@corp.com", rating: 4, comment: "Good service, could use more documentation", date: new Date() }
                    ]
                },
                {
                    id: "srv_002",
                    name: "Blockchain Transaction Monitor",
                    provider: "CryptoGuard Solutions",
                    description: "Real-time monitoring and analysis of blockchain transactions",
                    detailedDescription: "Monitor, analyze, and get alerts on blockchain transactions across multiple networks including Ethereum, Bitcoin, and custom chains.",
                    icon: "sap-icon://chain-link",
                    category: "Blockchain",
                    capability1: "Monitoring",
                    capability2: "Analytics",
                    price: 0,
                    pricing: "free",
                    status: "Active",
                    rating: 4.6,
                    reviewCount: 156,
                    featured: true
                },
                {
                    id: "srv_003",
                    name: "Supply Chain Optimizer",
                    provider: "LogiTech AI",
                    description: "AI-powered supply chain optimization and demand forecasting",
                    icon: "sap-icon://shipping-status",
                    category: "Operations",
                    capability1: "Forecasting",
                    capability2: "Optimization",
                    price: 599,
                    pricing: "subscription",
                    status: "Active",
                    rating: 4.9,
                    reviewCount: 89,
                    featured: true
                },
                {
                    id: "srv_004",
                    name: "Customer Sentiment Analyzer",
                    provider: "InsightFlow Analytics",
                    description: "Analyze customer feedback and social media sentiment in real-time",
                    icon: "sap-icon://customer",
                    category: "Analytics",
                    capability1: "Sentiment",
                    capability2: "Real-time",
                    price: 299,
                    pricing: "subscription",
                    status: "Active",
                    rating: 4.5,
                    reviewCount: 178
                },
                {
                    id: "srv_005",
                    name: "Predictive Maintenance Engine",
                    provider: "Industrial AI Corp",
                    description: "Predict equipment failures before they occur using IoT data",
                    icon: "sap-icon://wrench",
                    category: "IoT",
                    capability1: "Predictive",
                    capability2: "IoT",
                    price: 899,
                    pricing: "subscription",
                    status: "Active",
                    rating: 4.7,
                    reviewCount: 67
                },
                {
                    id: "srv_006",
                    name: "Fraud Detection System",
                    provider: "SecureFlow Technologies",
                    description: "Real-time fraud detection using machine learning algorithms",
                    icon: "sap-icon://shield",
                    category: "Security",
                    capability1: "ML",
                    capability2: "Real-time",
                    price: 1299,
                    pricing: "subscription",
                    status: "Active",
                    rating: 4.9,
                    reviewCount: 312
                }
            ];
            
            // Add more properties
            aServices.forEach(function(service) {
                service.providerRating = 4.5 + Math.random() * 0.5;
                service.providerReviews = Math.floor(Math.random() * 500) + 100;
                service.providerDescription = "Leading provider of enterprise " + service.category + " solutions";
                service.visible = true;
                
                if (!service.features) {
                    service.features = [
                        "Enterprise-grade security",
                        "24/7 support",
                        "99.9% uptime SLA",
                        "API access",
                        "Custom integrations"
                    ];
                }
                
                if (!service.requirements) {
                    service.requirements = [
                        "Valid API credentials",
                        "HTTPS endpoint",
                        "Minimum usage commitment"
                    ];
                }
                
                if (!service.reviews) {
                    service.reviews = [
                        { user: "user1@company.com", rating: 5, comment: "Excellent service!", date: new Date() },
                        { user: "user2@corp.com", rating: 4, comment: "Good value for money", date: new Date() }
                    ];
                }
            });
            
            return aServices;
        },

        _generateDataProducts: function() {
            var aDataProducts = [
                {
                    id: "data_001",
                    name: "Global Financial Markets Dataset",
                    provider: "FinData Corp",
                    description: "Real-time and historical financial market data from 50+ exchanges",
                    category: "Financial",
                    format: "api",
                    size: "Streaming",
                    lastUpdated: new Date(),
                    price: 2999,
                    pricing: "subscription",
                    downloads: 1234
                },
                {
                    id: "data_002",
                    name: "IoT Sensor Data - Manufacturing",
                    provider: "Industrial IoT Inc",
                    description: "Aggregated sensor data from 10,000+ manufacturing facilities",
                    category: "IoT",
                    format: "parquet",
                    size: "2.3 TB",
                    lastUpdated: new Date(Date.now() - 86400000),
                    price: 0,
                    pricing: "free",
                    downloads: 567
                },
                {
                    id: "data_003",
                    name: "Consumer Behavior Analytics",
                    provider: "MarketInsights AI",
                    description: "Anonymized consumer behavior data across retail and e-commerce",
                    category: "Market",
                    format: "csv",
                    size: "850 GB",
                    lastUpdated: new Date(Date.now() - 172800000),
                    price: 1499,
                    pricing: "one-time",
                    downloads: 892
                },
                {
                    id: "data_004",
                    name: "Supply Chain Network Data",
                    provider: "LogiData Systems",
                    description: "Global supply chain network data with shipping routes and timings",
                    category: "Operational",
                    format: "json",
                    size: "125 GB",
                    lastUpdated: new Date(Date.now() - 604800000),
                    price: 799,
                    pricing: "subscription",
                    downloads: 445
                },
                {
                    id: "data_005",
                    name: "Weather Pattern Historical Data",
                    provider: "ClimateData Pro",
                    description: "50 years of weather data from 10,000+ weather stations",
                    category: "Environmental",
                    format: "csv",
                    size: "4.5 TB",
                    lastUpdated: new Date(),
                    price: 0,
                    pricing: "free",
                    downloads: 2341
                }
            ];
            
            return aDataProducts;
        },

        _generateCategories: function() {
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

        _generateMyListings: function() {
            return {
                services: [
                    {
                        id: "my_srv_001",
                        name: "Custom Agent Builder",
                        category: "AI/ML",
                        status: "Active",
                        subscribers: 45,
                        monthlyRevenue: 8955,
                        rating: 4.7
                    },
                    {
                        id: "my_srv_002",
                        name: "Workflow Automation Engine",
                        category: "Operations",
                        status: "Active",
                        subscribers: 23,
                        monthlyRevenue: 4577,
                        rating: 4.5
                    }
                ],
                data: [
                    {
                        id: "my_data_001",
                        name: "Agent Performance Metrics",
                        format: "csv",
                        size: "2.1 GB",
                        status: "Active",
                        downloads: 156,
                        totalRevenue: 15600,
                        lastUpdated: new Date()
                    }
                ]
            };
        },

        _updateMyStats: function() {
            var oListings = this.oMarketplaceModel.getProperty("/myListings");
            var oStats = {
                activeServices: oListings.services.filter(s => s.status === "Active").length,
                dataProducts: oListings.data.length,
                totalSubscribers: oListings.services.reduce((sum, s) => sum + s.subscribers, 0),
                totalDownloads: oListings.data.reduce((sum, d) => sum + d.downloads, 0),
                monthlyRevenue: oListings.services.reduce((sum, s) => sum + s.monthlyRevenue, 0),
                averageRating: 4.6,
                totalReviews: 234,
                growth: 12.5
            };
            
            this.oMarketplaceModel.setProperty("/myStats", oStats);
        },

        onSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            this.oMarketplaceModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onCategoryFilter: function(oEvent) {
            var sCategory = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/filters/category", sCategory);
            this._applyFilters();
        },

        onPriceFilter: function(oEvent) {
            var sPriceRange = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/filters/priceRange", sPriceRange);
            this._applyFilters();
        },

        onRatingFilter: function(oEvent) {
            var iRating = oEvent.getParameter("value");
            this.oMarketplaceModel.setProperty("/filters/rating", iRating);
            this._applyFilters();
        },

        _applyFilters: function() {
            var sView = this.oUIModel.getProperty("/marketplaceView");
            var oFilters = this.oMarketplaceModel.getProperty("/filters");
            var aItems = [];
            
            if (sView === "services") {
                aItems = this.oMarketplaceModel.getProperty("/services");
            } else if (sView === "data") {
                aItems = this.oMarketplaceModel.getProperty("/dataProducts");
            }
            
            // Apply filters
            aItems.forEach(function(item) {
                var bVisible = true;
                
                // Search filter
                if (oFilters.searchQuery) {
                    var sQuery = oFilters.searchQuery.toLowerCase();
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
            var iVisibleCount = aItems.filter(item => item.visible).length;
            if (iVisibleCount === 0) {
                this.showNoData(this.getResourceBundle().getText("marketplace.noResults"));
            } else {
                this.hideLoading();
            }
        },

        onSort: function() {
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

        onServicePress: function(oEvent) {
            var oService = oEvent.getSource().getBindingContext("marketplace").getObject();
            this.oMarketplaceModel.setProperty("/selectedService", oService);
            this.byId("serviceDetailDialog").open();
        },

        onCloseServiceDetail: function() {
            this.byId("serviceDetailDialog").close();
        },

        onSubscribeService: function(oEvent) {
            var oService = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._addToCart(oService, "service");
        },

        onSubscribeFromDetail: function() {
            var oService = this.oMarketplaceModel.getProperty("/selectedService");
            this._addToCart(oService, "service");
            this.byId("serviceDetailDialog").close();
        },

        _addToCart: function(oItem, sType) {
            var oCart = this.oMarketplaceModel.getProperty("/cart");
            
            // Check if already in cart
            var bExists = oCart.items.some(function(item) {
                return item.id === oItem.id;
            });
            
            if (!bExists) {
                var oCartItem = Object.assign({}, oItem);
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

        _updateCartTotal: function() {
            var oCart = this.oMarketplaceModel.getProperty("/cart");
            var fTotal = 0;
            
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

        onOpenCart: function() {
            this.byId("cartDialog").open();
        },

        onCloseCart: function() {
            this.byId("cartDialog").close();
        },

        onRemoveFromCart: function(oEvent) {
            var oItem = oEvent.getParameter("listItem").getBindingContext("marketplace").getObject();
            var oCart = this.oMarketplaceModel.getProperty("/cart");
            
            var iIndex = oCart.items.findIndex(function(item) {
                return item.id === oItem.id;
            });
            
            if (iIndex > -1) {
                oCart.items.splice(iIndex, 1);
                this._updateCartTotal();
                this._saveCart();
                MessageToast.show(this.getResourceBundle().getText("marketplace.cart.removed"));
            }
        },

        onCheckout: function() {
            var oCart = this.oMarketplaceModel.getProperty("/cart");
            
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

        _processCheckout: function() {
            this.showSpinnerLoading(this.getResourceBundle().getText("marketplace.checkout.processing"));
            
            // Simulate checkout process
            setTimeout(function() {
                // Clear cart
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
            }.bind(this), 2000);
        },

        onDataProductPress: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            // Open data product details
            MessageToast.show("Data product details: " + oDataProduct.name);
        },

        onPreviewData: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            // Show data preview
            MessageToast.show("Preview not available in demo mode");
        },

        onAddDataToCart: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._addToCart(oDataProduct, "data");
        },

        onDataCategoryPress: function(oEvent) {
            var sCategory = oEvent.getSource().getHeader().getTitle();
            MessageToast.show("Filter by category: " + sCategory);
        },

        onDataFormatFilter: function(oEvent) {
            var sFormat = oEvent.getSource().getSelectedKey();
            // Apply format filter
            this._applyDataFilters();
        },

        onDataFrequencyFilter: function(oEvent) {
            var sFrequency = oEvent.getSource().getSelectedKey();
            // Apply frequency filter
            this._applyDataFilters();
        },

        _applyDataFilters: function() {
            // Apply filters to data products
            var oTable = this.byId("dataProductsTable");
            if (oTable) {
                var oBinding = oTable.getBinding("items");
                // Apply filters
            }
        },

        onPublishListing: function() {
            // Open publish dialog
            MessageToast.show(this.getResourceBundle().getText("marketplace.publish.opening"));
        },

        onEditListing: function(oEvent) {
            var oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show("Edit listing: " + oListing.name);
        },

        onViewAnalytics: function(oEvent) {
            var oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show("View analytics for: " + oListing.name);
        },

        onToggleListing: function(oEvent) {
            var oListing = oEvent.getSource().getBindingContext("marketplace").getObject();
            oListing.status = oListing.status === "Active" ? "Paused" : "Active";
            this.oMarketplaceModel.refresh();
            MessageToast.show(this.getResourceBundle().getText("marketplace.listing.statusChanged"));
        },

        onEditDataProduct: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show("Edit data product: " + oDataProduct.name);
        },

        onUpdateData: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show("Update data for: " + oDataProduct.name);
        },

        onViewDataAnalytics: function(oEvent) {
            var oDataProduct = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show("View data analytics for: " + oDataProduct.name);
        },

        onTryDemo: function() {
            var oService = this.oMarketplaceModel.getProperty("/selectedService");
            MessageToast.show(this.getResourceBundle().getText("marketplace.demo.starting", [oService.name]));
            // In production, launch demo environment
        },

        onContactProvider: function() {
            var oService = this.oMarketplaceModel.getProperty("/selectedService");
            MessageToast.show(this.getResourceBundle().getText("marketplace.contact.opening", [oService.provider]));
            // In production, open contact form
        },

        onViewProvider: function() {
            var oService = this.oMarketplaceModel.getProperty("/selectedService");
            MessageToast.show("View all listings from: " + oService.provider);
            // In production, filter by provider
        },

        onClearFilters: function() {
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

        _saveCart: function() {
            var oCart = this.oMarketplaceModel.getProperty("/cart");
            localStorage.setItem("a2a_marketplace_cart", JSON.stringify(oCart));
        },

        _restoreCart: function() {
            var sSavedCart = localStorage.getItem("a2a_marketplace_cart");
            if (sSavedCart) {
                try {
                    var oCart = JSON.parse(sSavedCart);
                    this.oMarketplaceModel.setProperty("/cart", oCart);
                } catch (e) {
                    console.error("Failed to restore cart:", e);
                }
            }
        },

        _setupRealtimeUpdates: function() {
            // Simulate real-time updates
            this._updateInterval = setInterval(function() {
                // Update random service rating
                var aServices = this.oMarketplaceModel.getProperty("/services");
                if (aServices.length > 0) {
                    var iRandom = Math.floor(Math.random() * aServices.length);
                    aServices[iRandom].reviewCount += Math.floor(Math.random() * 3);
                    aServices[iRandom].rating = Math.min(5, aServices[iRandom].rating + (Math.random() * 0.1 - 0.05));
                    this.oMarketplaceModel.setProperty("/services", aServices);
                }
            }.bind(this), 30000);
        },

        onNavBack: function() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onExit: function() {
            // Clean up
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }
        }
    });
});