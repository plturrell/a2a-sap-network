sap.ui.define([
    'com/sap/a2a/portal/controller/App',
    'sap/ui/base/ManagedObject',
    'sap/ui/core/mvc/View',
    'sap/ui/model/json/JSONModel',
    'sap/m/MessageToast'
], (AppController, ManagedObject, View, JSONModel, MessageToast) => {
    'use strict';

    QUnit.module('App Controller', {
        beforeEach: function () {
            this.oAppController = new AppController();
            this.oViewStub = new ManagedObject();
            this.oModelStub = new JSONModel({});
            
            sinon.stub(this.oAppController, 'getView').returns(this.oViewStub);
            sinon.stub(this.oAppController, 'getOwnerComponent').returns({
                getModel: sinon.stub().returns(this.oModelStub),
                getRouter: sinon.stub().returns({
                    navTo: sinon.stub(),
                    getRoute: sinon.stub().returns({
                        attachMatched: sinon.stub()
                    })
                })
            });
        },

        afterEach: function () {
            this.oAppController.destroy();
            this.oViewStub.destroy();
            this.oModelStub.destroy();
            sinon.restore();
        }
    });

    QUnit.test('Should initialize controller properly', function (assert) {
        // Arrange
        sinon.stub(this.oAppController, 'initPersonalization');
        sinon.stub(this.oAppController, 'initOfflineCapabilities');

        // Act
        this.oAppController.onInit();

        // Assert
        assert.ok(this.oAppController.initPersonalization.calledOnce, 'Personalization should be initialized');
        assert.ok(this.oAppController.initOfflineCapabilities.calledOnce, 'Offline capabilities should be initialized');
    });

    QUnit.test('Should handle navigation item selection', function (assert) {
        // Arrange
        const sNavigationKey = 'projects';
        const oEvent = {
            getParameter: sinon.stub().withArgs('item').returns({
                getKey: sinon.stub().returns(sNavigationKey)
            })
        };
        const oRouter = this.oAppController.getOwnerComponent().getRouter();

        // Act
        this.oAppController.onNavigationItemSelect(oEvent);

        // Assert
        assert.ok(oRouter.navTo.calledWith(sNavigationKey), 'Router should navigate to selected key');
    });

    QUnit.test('Should handle avatar press', function (assert) {
        // Arrange
        const oEvent = {
            getSource: sinon.stub().returns({
                getDomRef: sinon.stub().returns(document.createElement('div'))
            })
        };
        sinon.stub(this.oAppController, 'byId').returns({
            openBy: sinon.stub()
        });

        // Act
        this.oAppController.onAvatarPress(oEvent);

        // Assert
        assert.ok(this.oAppController.byId.calledWith('userActionSheet'), 'User action sheet should be accessed');
    });

    QUnit.test('Should handle notification press', function (assert) {
        // Arrange
        const oEvent = {
            getSource: sinon.stub().returns({
                getDomRef: sinon.stub().returns(document.createElement('div'))
            })
        };
        sinon.stub(this.oAppController, 'byId').returns({
            openBy: sinon.stub()
        });

        // Act
        this.oAppController.onNotificationPress(oEvent);

        // Assert
        assert.ok(this.oAppController.byId.calledWith('notificationPopover'), 'Notification popover should be accessed');
    });

    QUnit.test('Should handle search functionality', function (assert) {
        // Arrange
        const sSearchQuery = 'test query';
        const oEvent = {
            getParameter: sinon.stub().withArgs('query').returns(sSearchQuery)
        };
        sinon.stub(MessageToast, 'show');

        // Act
        this.oAppController.onSearch(oEvent);

        // Assert
        assert.ok(MessageToast.show.calledWith(`Searching for: ${sSearchQuery}`), 'Search message should be shown');
    });

    QUnit.test('Should handle user settings press', function (assert) {
        // Arrange
        const oRouter = this.oAppController.getOwnerComponent().getRouter();

        // Act
        this.oAppController.onUserSettingsPress();

        // Assert
        assert.ok(oRouter.navTo.calledWith('settings'), 'Should navigate to settings');
    });

    QUnit.test('Should handle logout press', function (assert) {
        // Arrange
        sinon.stub(MessageToast, 'show');

        // Act
        this.oAppController.onLogoutPress();

        // Assert
        assert.ok(MessageToast.show.calledWith('Logout functionality not implemented'), 'Logout message should be shown');
    });

    QUnit.test('Should handle side navigation toggle', function (assert) {
        // Arrange
        const oSideNavigation = {
            getExpanded: sinon.stub().returns(true),
            setExpanded: sinon.stub()
        };
        sinon.stub(this.oAppController, 'byId').returns(oSideNavigation);

        // Act
        this.oAppController.onSideNavButtonPress();

        // Assert
        assert.ok(oSideNavigation.setExpanded.calledWith(false), 'Side navigation should be collapsed');
    });

    QUnit.test('Should handle help press', function (assert) {
        // Arrange
        sinon.stub(this.oAppController, 'onOpenPersonalizationDialog');

        // Act
        this.oAppController.onHelpPress();

        // Assert
        assert.ok(this.oAppController.onOpenPersonalizationDialog.calledOnce, 'Personalization dialog should open');
    });

    QUnit.test('Should clean up on exit', function (assert) {
        // Arrange
        sinon.stub(this.oAppController, 'cleanupPersonalization');
        sinon.stub(this.oAppController, 'cleanupOfflineCapabilities');

        // Act
        this.oAppController.onExit();

        // Assert
        assert.ok(this.oAppController.cleanupPersonalization.calledOnce, 'Personalization should be cleaned up');
        assert.ok(this.oAppController.cleanupOfflineCapabilities.calledOnce, 'Offline capabilities should be cleaned up');
    });
});