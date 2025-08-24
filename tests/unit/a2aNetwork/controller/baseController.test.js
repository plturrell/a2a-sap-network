/*
 * SAP A2A Network - Enterprise Unit Test Suite
 * Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved.
 *
 * QUnit tests for BaseController
 * Tests common functionality and enterprise patterns for all controllers
 *
 * @namespace a2a.network.fiori.test.unit.controller
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */

sap.ui.define([
    'a2a/network/fiori/controller/BaseController',
    'sap/ui/core/mvc/View',
    'sap/ui/core/UIComponent',
    'sap/ui/model/json/JSONModel',
    'sap/ui/model/resource/ResourceModel',
    'sap/ui/core/routing/Router',
    'sap/base/i18n/ResourceBundle',
    'sap/m/MessageToast',
    'sap/m/MessageBox'
], (
    BaseController,
    View,
    UIComponent,
    JSONModel,
    ResourceModel,
    Router,
    ResourceBundle,
    MessageToast,
    MessageBox
) => {
    'use strict';

    /**
     * Test module for BaseController functionality
     *
     * Tests core enterprise patterns including:
     * - Loading state management
     * - Error handling
     * - Resource bundle access
     * - Navigation patterns
     * - Model management
     *
     * @public
     * @static
     */
    QUnit.module('a2a.network.fiori.controller.BaseController', {

        /**
         * Set up test environment before each test
         * Creates mock view, component, models, and router
         *
         * @function
         * @memberOf a2a.network.fiori.test.unit.controller.BaseController
         * @private
         */
        beforeEach() {
            // Create mock resource bundle
            this.oResourceBundle = new ResourceBundle({
                url: 'test-resources/a2a/network/fiori/i18n/i18n.properties',
                locale: 'en'
            });

            // Create mock i18n model
            this.oI18nModel = new ResourceModel({
                bundle: this.oResourceBundle
            });

            // Create mock router
            this.oRouter = new Router();
            sinon.stub(this.oRouter, 'navTo');
            sinon.stub(this.oRouter, 'getHashChanger').returns({
                getPreviousHash: sinon.stub().returns('previousHash')
            });

            // Create mock component
            this.oComponent = new UIComponent();
            sinon.stub(this.oComponent, 'getRouter').returns(this.oRouter);
            sinon.stub(this.oComponent, 'getModel').withArgs('i18n').returns(this.oI18nModel);

            // Create mock view
            this.oView = new View();
            sinon.stub(this.oView, 'getModel');
            sinon.stub(this.oView, 'setModel');

            // Create controller instance
            this.oController = new BaseController();
            sinon.stub(this.oController, 'getView').returns(this.oView);
            sinon.stub(this.oController, 'getOwnerComponent').returns(this.oComponent);

            // Stub MessageToast and MessageBox
            sinon.stub(MessageToast, 'show');
            sinon.stub(MessageBox, 'show');
        },

        /**
         * Clean up test environment after each test
         * Restores all stubs and cleans up objects
         *
         * @function
         * @memberOf a2a.network.fiori.test.unit.controller.BaseController
         * @private
         */
        afterEach() {
            this.oController.destroy();
            this.oView.destroy();
            this.oComponent.destroy();
            this.oRouter.destroy();
            this.oI18nModel.destroy();

            sinon.restore();
        }
    });

    /**
     * Test controller initialization
     * Verifies that the UI model is properly initialized with correct structure
     */
    QUnit.test('Should initialize controller with UI model', function(assert) {
        // Arrange
        const oUIModel = new JSONModel();
        this.oView.setModel.withArgs(sinon.match.instanceOf(JSONModel), 'ui');

        // Act
        this.oController.onInit();

        // Assert
        assert.ok(this.oView.setModel.calledWith(sinon.match.instanceOf(JSONModel), 'ui'),
            'UI model should be set on the view');
        assert.ok(this.oController.oUIModel instanceof JSONModel,
            'Controller should have UI model instance');

        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, false, 'Initial skeleton loading state should be false');
        assert.equal(oModelData.isLoadingSpinner, false, 'Initial spinner loading state should be false');
        assert.equal(oModelData.isLoadingProgress, false, 'Initial progress loading state should be false');
        assert.equal(oModelData.hasError, false, 'Initial error state should be false');
        assert.equal(oModelData.hasNoData, false, 'Initial no data state should be false');
    });

    /**
     * Test resource bundle access
     * Verifies that the resource bundle is properly retrieved from i18n model
     */
    QUnit.test('Should return resource bundle from i18n model', function(assert) {
        // Arrange
        sinon.stub(this.oResourceBundle, 'getText').withArgs('test.key').returns('Test Value');
        sinon.stub(this.oI18nModel, 'getResourceBundle').returns(this.oResourceBundle);

        // Act
        const oResourceBundle = this.oController.getResourceBundle();

        // Assert
        assert.equal(oResourceBundle, this.oResourceBundle, 'Should return the correct resource bundle');
        assert.ok(this.oComponent.getModel.calledWith('i18n'), 'Should get i18n model from component');
    });

    /**
     * Test router access
     * Verifies that the router is properly retrieved from component
     */
    QUnit.test('Should return router from component', function(assert) {
        // Act
        const oRouter = this.oController.getRouter();

        // Assert
        assert.equal(oRouter, this.oRouter, 'Should return the correct router instance');
        assert.ok(this.oComponent.getRouter.called, 'Should call getRouter on component');
    });

    /**
     * Test model access
     * Verifies that models are properly retrieved from the view
     */
    QUnit.test('Should get model from view', function(assert) {
        // Arrange
        const oTestModel = new JSONModel();
        this.oView.getModel.withArgs('testModel').returns(oTestModel);

        // Act
        const oModel = this.oController.getModel('testModel');

        // Assert
        assert.equal(oModel, oTestModel, 'Should return the correct model');
        assert.ok(this.oView.getModel.calledWith('testModel'), 'Should call getModel on view with model name');

        // Cleanup
        oTestModel.destroy();
    });

    /**
     * Test model setting
     * Verifies that models are properly set on the view
     */
    QUnit.test('Should set model on view', function(assert) {
        // Arrange
        const oTestModel = new JSONModel();

        // Act
        this.oController.setModel(oTestModel, 'testModel');

        // Assert
        assert.ok(this.oView.setModel.calledWith(oTestModel, 'testModel'),
            'Should call setModel on view with model and name');

        // Cleanup
        oTestModel.destroy();
    });

    /**
     * Test skeleton loading state
     * Verifies that skeleton loading is properly activated and configured
     */
    QUnit.test('Should show skeleton loading state', function(assert) {
        // Arrange
        this.oController.onInit();
        const sTestMessage = 'Loading test data...';

        // Act
        this.oController.showSkeletonLoading(sTestMessage);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, true, 'Skeleton loading should be active');
        assert.equal(oModelData.isLoadingSpinner, false, 'Spinner loading should be inactive');
        assert.equal(oModelData.isLoadingProgress, false, 'Progress loading should be inactive');
        assert.equal(oModelData.hasError, false, 'Error state should be inactive');
        assert.equal(oModelData.loadingMessage, sTestMessage, 'Loading message should be set correctly');
    });

    /**
     * Test spinner loading state
     * Verifies that spinner loading is properly activated and configured
     */
    QUnit.test('Should show spinner loading state', function(assert) {
        // Arrange
        this.oController.onInit();
        const sTestMessage = 'Processing request...';
        const sTestSubMessage = 'Please wait...';

        // Act
        this.oController.showSpinnerLoading(sTestMessage, sTestSubMessage);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, false, 'Skeleton loading should be inactive');
        assert.equal(oModelData.isLoadingSpinner, true, 'Spinner loading should be active');
        assert.equal(oModelData.isLoadingProgress, false, 'Progress loading should be inactive');
        assert.equal(oModelData.loadingMessage, sTestMessage, 'Loading message should be set correctly');
        assert.equal(oModelData.loadingSubMessage, sTestSubMessage, 'Loading sub-message should be set correctly');
    });

    /**
     * Test progress loading state
     * Verifies that progress loading is properly activated and configured
     */
    QUnit.test('Should show progress loading state', function(assert) {
        // Arrange
        this.oController.onInit();
        const oOptions = {
            title: 'Uploading files...',
            message: 'Processing step 2 of 5',
            value: 40,
            state: 'Success',
            description: 'Uploading document.pdf'
        };

        // Act
        this.oController.showProgressLoading(oOptions);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, false, 'Skeleton loading should be inactive');
        assert.equal(oModelData.isLoadingSpinner, false, 'Spinner loading should be inactive');
        assert.equal(oModelData.isLoadingProgress, true, 'Progress loading should be active');
        assert.equal(oModelData.progressTitle, oOptions.title, 'Progress title should be set correctly');
        assert.equal(oModelData.loadingMessage, oOptions.message, 'Progress message should be set correctly');
        assert.equal(oModelData.progressValue, oOptions.value, 'Progress value should be set correctly');
        assert.equal(oModelData.progressText, '40%', 'Progress text should be formatted correctly');
        assert.equal(oModelData.progressState, oOptions.state, 'Progress state should be set correctly');
    });

    /**
     * Test blockchain loading state
     * Verifies that blockchain-specific loading is properly activated and configured
     */
    QUnit.test('Should show blockchain loading state', function(assert) {
        // Arrange
        this.oController.onInit();
        const sTestStep = 'Waiting for transaction confirmation...';

        // Act
        this.oController.showBlockchainLoading(sTestStep);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, false, 'Skeleton loading should be inactive');
        assert.equal(oModelData.isLoadingSpinner, false, 'Spinner loading should be inactive');
        assert.equal(oModelData.isLoadingProgress, false, 'Progress loading should be inactive');
        assert.equal(oModelData.isLoadingBlockchain, true, 'Blockchain loading should be active');
        assert.equal(oModelData.blockchainStep, sTestStep, 'Blockchain step should be set correctly');
    });

    /**
     * Test hiding loading states
     * Verifies that all loading states are properly deactivated
     */
    QUnit.test('Should hide all loading states', function(assert) {
        // Arrange
        this.oController.onInit();
        this.oController.showSpinnerLoading('Test loading');

        // Act
        this.oController.hideLoading();

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.isLoadingSkeleton, false, 'Skeleton loading should be inactive');
        assert.equal(oModelData.isLoadingSpinner, false, 'Spinner loading should be inactive');
        assert.equal(oModelData.isLoadingProgress, false, 'Progress loading should be inactive');
        assert.equal(oModelData.isLoadingBlockchain, false, 'Blockchain loading should be inactive');
    });

    /**
     * Test error state
     * Verifies that error state is properly activated and configured
     */
    QUnit.test('Should show error state', function(assert) {
        // Arrange
        this.oController.onInit();
        const sErrorMessage = 'Failed to load data';
        const sErrorTitle = 'Network Error';

        // Act
        this.oController.showError(sErrorMessage, sErrorTitle);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.hasError, true, 'Error state should be active');
        assert.equal(oModelData.hasNoData, false, 'No data state should be inactive');
        assert.equal(oModelData.errorMessage, sErrorMessage, 'Error message should be set correctly');
        assert.equal(oModelData.errorTitle, sErrorTitle, 'Error title should be set correctly');
        assert.equal(oModelData.isLoadingSpinner, false, 'All loading states should be inactive');
    });

    /**
     * Test no data state
     * Verifies that no data state is properly activated and configured
     */
    QUnit.test('Should show no data state', function(assert) {
        // Arrange
        this.oController.onInit();
        const sNoDataMessage = 'No items found';
        const sNoDataIcon = 'sap-icon://inbox';

        // Act
        this.oController.showNoData(sNoDataMessage, sNoDataIcon);

        // Assert
        const oModelData = this.oController.oUIModel.getData();
        assert.equal(oModelData.hasNoData, true, 'No data state should be active');
        assert.equal(oModelData.hasError, false, 'Error state should be inactive');
        assert.equal(oModelData.noDataMessage, sNoDataMessage, 'No data message should be set correctly');
        assert.equal(oModelData.noDataIcon, sNoDataIcon, 'No data icon should be set correctly');
    });

    /**
     * Test navigation back with history
     * Verifies that back navigation uses browser history when available
     */
    QUnit.test('Should navigate back using browser history', function(assert) {
        // Arrange
        const oHistoryStub = sinon.stub(window.history, 'go');

        // Act
        this.oController.onNavBack();

        // Assert
        assert.ok(oHistoryStub.calledWith(-1), 'Should go back in browser history');
        assert.notOk(this.oRouter.navTo.called, 'Should not use router navigation');

        // Cleanup
        oHistoryStub.restore();
    });

    /**
     * Test navigation back without history
     * Verifies that default route navigation is used when no history exists
     */
    QUnit.test('Should navigate to default route when no history', function(assert) {
        // Arrange
        this.oRouter.getHashChanger.returns({
            getPreviousHash: sinon.stub().returns(undefined)
        });
        const sDefaultRoute = 'dashboard';

        // Act
        this.oController.onNavBack(sDefaultRoute);

        // Assert
        assert.ok(this.oRouter.navTo.calledWith(sDefaultRoute, {}, true),
            'Should navigate to default route with replace history');
    });

    /**
     * Test message toast display
     * Verifies that message toasts are properly displayed with correct options
     */
    QUnit.test('Should show message toast with options', function(assert) {
        // Arrange
        const sMessage = 'Operation completed successfully';
        const oOptions = { duration: 5000 };

        // Act
        this.oController.showMessageToast(sMessage, oOptions);

        // Assert
        assert.ok(MessageToast.show.calledWith(sMessage), 'MessageToast should be called with message');
        const oCallArgs = MessageToast.show.getCall(0).args[1];
        assert.equal(oCallArgs.duration, 5000, 'Custom duration should be applied');
    });

    /**
     * Test message box display
     * Verifies that message boxes are properly displayed with correct options
     */
    QUnit.test('Should show message box with options', function(assert) {
        // Arrange
        const sMessage = 'Are you sure you want to delete this item?';
        const oOptions = {
            icon: MessageBox.Icon.WARNING,
            title: 'Confirm Deletion',
            actions: [MessageBox.Action.YES, MessageBox.Action.NO]
        };

        // Act
        this.oController.showMessageBox(sMessage, oOptions);

        // Assert
        assert.ok(MessageBox.show.calledWith(sMessage), 'MessageBox should be called with message');
        const oCallArgs = MessageBox.show.getCall(0).args[1];
        assert.equal(oCallArgs.icon, MessageBox.Icon.WARNING, 'Custom icon should be applied');
        assert.equal(oCallArgs.title, 'Confirm Deletion', 'Custom title should be applied');
        assert.deepEqual(oCallArgs.actions, [MessageBox.Action.YES, MessageBox.Action.NO],
            'Custom actions should be applied');
    });

    /**
     * Integration test for loading state transitions
     * Verifies that loading states transition correctly without conflicts
     */
    QUnit.test('Should handle loading state transitions correctly', function(assert) {
        // Arrange
        this.oController.onInit();

        // Act & Assert - Test state transitions
        this.oController.showSkeletonLoading('Loading...');
        assert.equal(this.oController.oUIModel.getData().isLoadingSkeleton, true,
            'Skeleton loading should be active');

        this.oController.showSpinnerLoading('Processing...');
        assert.equal(this.oController.oUIModel.getData().isLoadingSkeleton, false,
            'Skeleton loading should be deactivated');
        assert.equal(this.oController.oUIModel.getData().isLoadingSpinner, true,
            'Spinner loading should be active');

        this.oController.showError('Error occurred');
        assert.equal(this.oController.oUIModel.getData().isLoadingSpinner, false,
            'Spinner loading should be deactivated');
        assert.equal(this.oController.oUIModel.getData().hasError, true,
            'Error state should be active');

        this.oController.hideLoading();
        assert.equal(this.oController.oUIModel.getData().hasError, false,
            'Error state should be cleared by hideLoading');
    });
});