sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/model/json/JSONModel',
    'sap/m/MessageBox',
    'sap/m/MessageToast',
    'sap/ui/core/Fragment',
    'sap/viz/ui5/format/ChartFormatter',
    'sap/viz/ui5/api/env/Format',
    '../formatter/formatter'
], (Controller, JSONModel, MessageBox, MessageToast, Fragment, ChartFormatter, Format, formatter) => {
    'use strict';

    return Controller.extend('a2a.controller.agentRatingReview', {
        formatter: formatter,

        onInit: function () {
            // Initialize models
            const oViewModel = new JSONModel({
                selectedAgent: null,
                agents: [],
                reviews: [],
                newReview: {
                    taskId: '',
                    performanceRating: 0,
                    accuracyRating: 0,
                    communicationRating: 0,
                    overallRating: 0,
                    comments: ''
                },
                requiredStakeAmount: '0.01',
                busy: false
            });
            this.getView().setModel(oViewModel);

            // Initialize chart formatting
            Format.numericFormatter(ChartFormatter.getInstance());

            // Load agents
            this._loadAgents();

            // Set up router
            const oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            oRouter.getRoute('agentRatingReview').attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched: function (oEvent) {
            const sAgentId = oEvent.getParameter('arguments').agentId;
            if (sAgentId) {
                this._selectAgent(sAgentId);
            }
        },

        _loadAgents: function () {
            const oModel = this.getView().getModel();
            oModel.setProperty('/busy', true);

            // Call backend to get list of agents
            jQuery.ajax({
                url: '/api/agents',
                type: 'GET',
                success: function (data) {
                    const aAgents = data.agents.map((agent) => {
                        return {
                            agentId: agent.id,
                            name: agent.name,
                            type: agent.type,
                            status: agent.status
                        };
                    });
                    oModel.setProperty('/agents', aAgents);
                    oModel.setProperty('/busy', false);
                }.bind(this),
                error: function () {
                    MessageBox.error('Failed to load agents');
                    oModel.setProperty('/busy', false);
                }
            });
        },

        onAgentSelectionChange: function (oEvent) {
            const oSelectedItem = oEvent.getParameter('selectedItem');
            if (oSelectedItem) {
                const sAgentId = oSelectedItem.getKey();
                this._selectAgent(sAgentId);
            }
        },

        _selectAgent: function (sAgentId) {
            const oModel = this.getView().getModel();
            oModel.setProperty('/busy', true);

            // Load agent details and reviews
            Promise.all([
                this._loadAgentDetails(sAgentId),
                this._loadAgentReviews(sAgentId)
            ]).then((results) => {
                const oAgentDetails = results[0];
                const aReviews = results[1];

                oModel.setProperty('/selectedAgent', oAgentDetails);
                oModel.setProperty('/reviews', aReviews);

                // Update rating distribution chart
                this._updateRatingDistribution(oAgentDetails.ratingDistribution);

                oModel.setProperty('/busy', false);
            }).catch((error) => {
                MessageBox.error('Failed to load agent information');
                oModel.setProperty('/busy', false);
            });
        },

        _loadAgentDetails: function (sAgentId) {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/api/agents/${  sAgentId  }/details`,
                    type: 'GET',
                    success: function (data) {
                        resolve({
                            agentId: data.agentId,
                            name: data.name,
                            overallRating: data.overallRating || 0,
                            totalReviews: data.totalReviews || 0,
                            successRate: data.successRate || 0,
                            avgResponseTime: data.avgResponseTime || 0,
                            status: data.status || 'Unknown',
                            ratingDistribution: data.ratingDistribution || {
                                '5': 0,
                                '4': 0,
                                '3': 0,
                                '2': 0,
                                '1': 0
                            }
                        });
                    },
                    error: reject
                });
            });
        },

        _loadAgentReviews: function (sAgentId) {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/api/agents/${  sAgentId  }/reviews`,
                    type: 'GET',
                    success: function (data) {
                        const aReviews = data.reviews.map((review) => {
                            return {
                                reviewId: review.id,
                                reviewerAddress: review.reviewerAddress,
                                reviewerName: review.reviewerName || 'Anonymous',
                                taskId: review.taskId,
                                overallRating: review.overallRating,
                                performanceRating: review.performanceRating,
                                accuracyRating: review.accuracyRating,
                                communicationRating: review.communicationRating,
                                comments: review.comments,
                                timestamp: new Date(review.timestamp),
                                validationStatus: review.validationStatus || 'Pending',
                                validatorCount: review.validatorCount || 0
                            };
                        });
                        resolve(aReviews);
                    },
                    error: reject
                });
            });
        },

        _updateRatingDistribution: function (oDistribution) {
            let oVizFrame = this._getVizFrame();
            if (!oVizFrame) {
                // Create chart if it doesn't exist
                oVizFrame = new sap.viz.ui5.controls.VizFrame({
                    id: this.createId('ratingDistributionChart'),
                    uiConfig: {
                        applicationSet: 'fiori'
                    },
                    height: '300px',
                    width: '100%',
                    vizType: 'column'
                });

                oVizFrame.setVizProperties({
                    plotArea: {
                        dataLabel: {
                            visible: true
                        },
                        colorPalette: ['#5899DA']
                    },
                    valueAxis: {
                        title: {
                            visible: true,
                            text: 'Number of Reviews'
                        }
                    },
                    categoryAxis: {
                        title: {
                            visible: true,
                            text: 'Rating'
                        }
                    },
                    title: {
                        visible: false
                    }
                });

                const oDataset = new sap.viz.ui5.data.FlattenedDataset({
                    dimensions: [{
                        name: 'Rating',
                        value: '{Rating}'
                    }],
                    measures: [{
                        name: 'Count',
                        value: '{Count}'
                    }],
                    data: '{/ratingChartData}'
                });

                oVizFrame.setDataset(oDataset);

                const oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                    uid: 'valueAxis',
                    type: 'Measure',
                    values: ['Count']
                });
                const oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                    uid: 'categoryAxis',
                    type: 'Dimension',
                    values: ['Rating']
                });
                oVizFrame.addFeed(oFeedValueAxis);
                oVizFrame.addFeed(oFeedCategoryAxis);

                this.byId('ratingDistributionContainer').addContent(oVizFrame);
            }

            // Update chart data
            const aChartData = [];
            for (const rating in oDistribution) {
                aChartData.push({
                    Rating: `${rating  } Stars`,
                    Count: oDistribution[rating]
                });
            }
            this.getView().getModel().setProperty('/ratingChartData', aChartData);
        },

        _getVizFrame: function () {
            return this.byId('ratingDistributionChart');
        },

        onSubmitReview: function () {
            const oModel = this.getView().getModel();
            const oNewReview = oModel.getProperty('/newReview');
            const oSelectedAgent = oModel.getProperty('/selectedAgent');

            if (!oSelectedAgent) {
                MessageBox.error('Please select an agent first');
                return;
            }

            // Validate review
            if (!oNewReview.taskId) {
                MessageBox.error('Please enter a Task ID');
                return;
            }

            if (oNewReview.overallRating === 0) {
                MessageBox.error('Please provide an overall rating');
                return;
            }

            MessageBox.confirm(
                `Submit this review? This will require staking ${  oModel.getProperty('/requiredStakeAmount')  } ETH.`,
                {
                    title: 'Confirm Review Submission',
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._submitReview(oSelectedAgent.agentId, oNewReview);
                        }
                    }.bind(this)
                }
            );
        },

        _submitReview: function (sAgentId, oReview) {
            const oModel = this.getView().getModel();
            oModel.setProperty('/busy', true);

            const oReviewData = {
                agentId: sAgentId,
                taskId: oReview.taskId,
                performanceRating: oReview.performanceRating || oReview.overallRating,
                accuracyRating: oReview.accuracyRating || oReview.overallRating,
                communicationRating: oReview.communicationRating || oReview.overallRating,
                overallRating: oReview.overallRating,
                comments: oReview.comments,
                stakeAmount: oModel.getProperty('/requiredStakeAmount')
            };

            jQuery.ajax({
                url: `/api/agents/${  sAgentId  }/reviews`,
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(oReviewData),
                success: function (data) {
                    MessageToast.show('Review submitted successfully!');

                    // Reset form
                    oModel.setProperty('/newReview', {
                        taskId: '',
                        performanceRating: 0,
                        accuracyRating: 0,
                        communicationRating: 0,
                        overallRating: 0,
                        comments: ''
                    });

                    // Reload reviews
                    this._loadAgentReviews(sAgentId).then((aReviews) => {
                        oModel.setProperty('/reviews', aReviews);
                    });

                    oModel.setProperty('/busy', false);
                }.bind(this),
                error: function (xhr) {
                    let sError = 'Failed to submit review';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        sError = xhr.responseJSON.error;
                    }
                    MessageBox.error(sError);
                    oModel.setProperty('/busy', false);
                }
            });
        },

        onSearchReviews: function (oEvent) {
            const sQuery = oEvent.getParameter('query');
            const oTable = this.byId('reviewHistoryTable');
            const oBinding = oTable.getBinding('items');

            if (oBinding) {
                const aFilters = [];
                if (sQuery) {
                    aFilters.push(new sap.ui.model.Filter({
                        filters: [
                            new sap.ui.model.Filter('reviewerName', sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter('taskId', sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter('comments', sap.ui.model.FilterOperator.Contains, sQuery)
                        ],
                        and: false
                    }));
                }
                oBinding.filter(aFilters);
            }
        },

        onFilterReviews: function () {
            // Open filter dialog
            if (!this._oFilterDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: 'a2a.fragment.ReviewFilterDialog',
                    controller: this
                }).then((oDialog) => {
                    this._oFilterDialog = oDialog;
                    this.getView().addDependent(this._oFilterDialog);
                    this._oFilterDialog.open();
                });
            } else {
                this._oFilterDialog.open();
            }
        },

        onReviewPress: function (oEvent) {
            const oItem = oEvent.getSource();
            const oBindingContext = oItem.getBindingContext();
            const oReview = oBindingContext.getObject();

            // Open review detail dialog
            if (!this._oReviewDetailDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: 'a2a.fragment.ReviewDetailDialog',
                    controller: this
                }).then((oDialog) => {
                    this._oReviewDetailDialog = oDialog;
                    this.getView().addDependent(this._oReviewDetailDialog);
                    this._showReviewDetail(oReview);
                });
            } else {
                this._showReviewDetail(oReview);
            }
        },

        _showReviewDetail: function (oReview) {
            const oDetailModel = new JSONModel(oReview);
            this._oReviewDetailDialog.setModel(oDetailModel, 'reviewDetail');
            this._oReviewDetailDialog.open();
        },

        onRefreshData: function () {
            const oModel = this.getView().getModel();
            const oSelectedAgent = oModel.getProperty('/selectedAgent');

            if (oSelectedAgent) {
                this._selectAgent(oSelectedAgent.agentId);
            } else {
                this._loadAgents();
            }

            MessageToast.show('Data refreshed');
        },

        onNavBack: function () {
            const oHistory = sap.ui.core.routing.History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                const oRouter = sap.ui.core.UIComponent.getRouterFor(this);
                oRouter.navTo('home', {}, true);
            }
        },

        onApplyReviewFilter: function () {
            const oTable = this.byId('reviewHistoryTable');
            const oBinding = oTable.getBinding('items');

            if (oBinding) {
                const aFilters = [];

                // Get filter values
                const iMinRating = this.byId('filterMinRating').getValue();
                const oDateFrom = this.byId('filterDateFrom').getDateValue();
                const oDateTo = this.byId('filterDateTo').getDateValue();
                const sStatus = this.byId('filterValidationStatus').getSelectedKey();

                // Apply rating filter
                if (iMinRating > 0) {
                    aFilters.push(new sap.ui.model.Filter('overallRating', sap.ui.model.FilterOperator.GE, iMinRating));
                }

                // Apply date filters
                if (oDateFrom) {
                    aFilters.push(new sap.ui.model.Filter('timestamp', sap.ui.model.FilterOperator.GE, oDateFrom));
                }
                if (oDateTo) {
                    aFilters.push(new sap.ui.model.Filter('timestamp', sap.ui.model.FilterOperator.LE, oDateTo));
                }

                // Apply status filter
                if (sStatus && sStatus !== 'all') {
                    aFilters.push(new sap.ui.model.Filter('validationStatus', sap.ui.model.FilterOperator.EQ,
                        sStatus.charAt(0).toUpperCase() + sStatus.slice(1)));
                }

                oBinding.filter(aFilters);
            }

            this._oFilterDialog.close();
        },

        onCancelReviewFilter: function () {
            this._oFilterDialog.close();
        },

        onValidateReview: function () {
            const oReview = this._oReviewDetailDialog.getModel('reviewDetail').getData();
            const oModel = this.getView().getModel();
            const oSelectedAgent = oModel.getProperty('/selectedAgent');

            MessageBox.confirm(
                'Are you sure you want to validate this review? This action cannot be undone.',
                {
                    title: 'Confirm Validation',
                    onClose: function (oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._validateReview(oSelectedAgent.agentId, oReview.reviewId);
                        }
                    }.bind(this)
                }
            );
        },

        _validateReview: function (sAgentId, sReviewId) {
            const oModel = this.getView().getModel();
            oModel.setProperty('/busy', true);

            jQuery.ajax({
                url: `/api/agents/${  sAgentId  }/reviews/${  sReviewId  }/validate`,
                type: 'POST',
                success: function (data) {
                    MessageToast.show('Review validated successfully!');

                    // Update review status in dialog
                    this._oReviewDetailDialog.getModel('reviewDetail').setProperty('/validationStatus', 'Validated');
                    this._oReviewDetailDialog.getModel('reviewDetail').setProperty('/validatorCount',
                        this._oReviewDetailDialog.getModel('reviewDetail').getProperty('/validatorCount') + 1);

                    // Reload reviews
                    this._loadAgentReviews(sAgentId).then((aReviews) => {
                        oModel.setProperty('/reviews', aReviews);
                    });

                    oModel.setProperty('/busy', false);
                }.bind(this),
                error: function (xhr) {
                    let sError = 'Failed to validate review';
                    if (xhr.responseJSON && xhr.responseJSON.error) {
                        sError = xhr.responseJSON.error;
                    }
                    MessageBox.error(sError);
                    oModel.setProperty('/busy', false);
                }
            });
        },

        onCloseReviewDetail: function () {
            this._oReviewDetailDialog.close();
        }
    });
});