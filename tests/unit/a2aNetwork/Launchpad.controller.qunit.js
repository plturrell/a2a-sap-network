/* global QUnit, sinon, window, sap */
sap.ui.define([
    'a2a/network/launchpad/controller/Launchpad.controller',
    'sap/ui/model/json/JSONModel',
    'sap/ui/model/resource/ResourceModel'
], (LaunchpadController, JSONModel, ResourceModel) => {
    'use strict';

    QUnit.module('Launchpad Controller', {
        beforeEach: function () {
            this.oController = new LaunchpadController();
            const oView = new sap.ui.core.mvc.XMLView();
            const oComponent = {
                getModel: (sName) => {
                    if (sName === 'i18n') {
                        return new ResourceModel({
                            bundleUrl: '../i18n/i18n.properties'
                        });
                    }
                    return new JSONModel({});
                }
            };
            sinon.stub(this.oController, 'getView').returns(oView);
            sinon.stub(this.oController, 'getOwnerComponent').returns(oComponent);
        },
        afterEach: function () {
            this.oController.destroy();
        }
    });

    QUnit.test('Should initialize models onInit', function (assert) {
        // Act
        this.oController.onInit();

        // Assert
        assert.ok(this.oController.getView().getModel('launchpad'), 'Launchpad model should be created');
        assert.ok(this.oController.getView().getModel('notifications'), 'Notifications model should be created');
        assert.strictEqual(this.oController.getView().getModel('launchpad').getData().tiles.length, 6, 'Launchpad model should have 6 tiles');
    });

    QUnit.test('Should navigate to Analytics view', function (assert) {
        // Arrange
        const oRouter = { navTo: sinon.spy() };
        this.oController.getOwnerComponent().getRouter = () => oRouter;

        // Act
        this.oController.onOpenAnalytics();

        // Assert
        assert.ok(oRouter.navTo.calledOnceWith('Analytics'), 'navTo should be called once with \'Analytics\' route');
    });

    QUnit.test('Should update notification model on open', function (assert) {
        // Arrange
        const oModel = this.oController.getView().getModel('notifications');

        // Act
        this.oController.onOpenNotifications({});

        // Assert
        assert.strictEqual(oModel.getData().items.length, 3, 'Notifications model should be populated with 3 items');
    });

    QUnit.test('Should fetch and update tile data successfully', async function (assert) {
        // Arrange
        const oFakeResponse = { agentCount: 5, services: 10, workflows: 2, performance: 95, notifications: 3, security: 1 };
        const oResponse = new window.Response(JSON.stringify(oFakeResponse), { status: 200, headers: { 'Content-type': 'application/json' } });
        sinon.stub(window, 'fetch').resolves(oResponse);
        this.oController.onInit(); // To initialize the model

        // Act
        await this.oController._fetchAndSetTileData();

        // Assert
        const oModel = this.oController.getView().getModel('launchpad');
        assert.strictEqual(oModel.getProperty('/tiles/0/value'), 5, 'Agent count should be updated');
        assert.strictEqual(oModel.getProperty('/tiles/1/value'), 10, 'Services count should be updated');

        window.fetch.restore();
    });

});
