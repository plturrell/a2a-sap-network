sap.ui.define([
    "com/sap/a2a/controller/mixin/PersonalizationMixin",
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (PersonalizationMixin, Controller, JSONModel, MessageToast) {
    "use strict";

    QUnit.module("PersonalizationMixin", {
        beforeEach: function () {
            this.oController = new Controller();
            Object.assign(this.oController, PersonalizationMixin);
            
            this.oViewStub = {
                getId: sinon.stub().returns("testView"),
                addDependent: sinon.stub(),
                addStyleClass: sinon.stub(),
                removeStyleClass: sinon.stub(),
                byId: sinon.stub(),
                findAggregatedObjects: sinon.stub().returns([])
            };
            
            this.oComponentStub = {
                getModel: sinon.stub().returns(new JSONModel({
                    id: "testUser"
                }))
            };
            
            sinon.stub(this.oController, "getView").returns(this.oViewStub);
            sinon.stub(this.oController, "getOwnerComponent").returns(this.oComponentStub);
            sinon.stub(this.oController, "byId").returns({});
            
            // Mock localStorage
            this.oLocalStorageStub = {
                getItem: sinon.stub(),
                setItem: sinon.stub()
            };
            sinon.stub(window, "localStorage").value(this.oLocalStorageStub);
        },

        afterEach: function () {
            this.oController.destroy();
            sinon.restore();
        }
    });

    QUnit.test("Should initialize personalization", function (assert) {
        // Arrange
        sinon.stub(this.oController, "_loadPersonalizationData").returns(Promise.resolve({}));
        sinon.stub(this.oController, "_registerPersonalizationHandlers");
        sinon.stub(this.oController, "_initializeVariantManagement");

        // Act
        this.oController.initPersonalization();

        // Assert
        assert.ok(this.oController._loadPersonalizationData.calledOnce, "Should load personalization data");
        assert.ok(this.oController._registerPersonalizationHandlers.calledOnce, "Should register handlers");
        assert.ok(this.oController._initializeVariantManagement.calledOnce, "Should initialize variant management");
    });

    QUnit.test("Should load personalization data from localStorage", function (assert) {
        // Arrange
        const oPersonalizationData = {
            theme: "sap_horizon_dark",
            density: "compact"
        };
        this.oLocalStorageStub.getItem.returns(JSON.stringify(oPersonalizationData));
        sinon.stub(this.oController, "_applyPersonalization");

        // Act
        return this.oController._loadPersonalizationData().then(function (oData) {
            // Assert
            assert.deepEqual(oData, oPersonalizationData, "Should return personalization data from localStorage");
            assert.ok(this.oController._applyPersonalization.calledWith(oPersonalizationData), "Should apply personalization");
        }.bind(this));
    });

    QUnit.test("Should return default personalization when none exists", function (assert) {
        // Arrange
        this.oLocalStorageStub.getItem.returns(null);
        sinon.stub(this.oController, "_getDefaultPersonalization").returns({
            theme: "sap_horizon",
            density: "cozy"
        });

        // Act
        return this.oController._loadPersonalizationData().then(function (oData) {
            // Assert
            assert.ok(this.oController._getDefaultPersonalization.calledOnce, "Should get default personalization");
            assert.equal(oData.theme, "sap_horizon", "Should return default theme");
            assert.equal(oData.density, "cozy", "Should return default density");
        }.bind(this));
    });

    QUnit.test("Should apply theme personalization", function (assert) {
        // Arrange
        const oPersonalizationData = { theme: "sap_horizon_dark" };
        sinon.stub(sap.ui.getCore(), "applyTheme");

        // Act
        this.oController._applyPersonalization(oPersonalizationData);

        // Assert
        assert.ok(sap.ui.getCore().applyTheme.calledWith("sap_horizon_dark"), "Should apply theme");
    });

    QUnit.test("Should apply density personalization", function (assert) {
        // Arrange
        const oPersonalizationData = { density: "compact" };

        // Act
        this.oController._applyPersonalization(oPersonalizationData);

        // Assert
        assert.ok(this.oViewStub.addStyleClass.calledWith("sapUiSizecompact"), "Should add density class");
    });

    QUnit.test("Should save personalization to localStorage", function (assert) {
        // Arrange
        const oPersonalizationData = {
            theme: "sap_horizon_dark",
            density: "compact"
        };
        this.oComponentStub.getModel.returns(new JSONModel({
            getData: sinon.stub().returns(oPersonalizationData)
        }));
        sinon.stub(MessageToast, "show");

        // Act
        this.oController.onSavePersonalization();

        // Assert
        const sExpectedKey = "a2a-personalization-testUser-testView";
        assert.ok(this.oLocalStorageStub.setItem.calledWith(sExpectedKey, JSON.stringify(oPersonalizationData)), 
                 "Should save to localStorage");
        assert.ok(MessageToast.show.calledWith("Personalization saved successfully"), "Should show success message");
    });

    QUnit.test("Should handle density change", function (assert) {
        // Arrange
        const oEvent = {
            getParameter: sinon.stub().withArgs("item").returns({
                getKey: sinon.stub().returns("Compact")
            })
        };
        
        const oPersonalizationModel = new JSONModel({});
        this.oComponentStub.getModel.withArgs("personalization").returns(oPersonalizationModel);

        // Act
        this.oController.onDensityChange(oEvent);

        // Assert
        assert.ok(this.oViewStub.removeStyleClass.calledWith("sapUiSizeCozy"), "Should remove old density class");
        assert.ok(this.oViewStub.removeStyleClass.calledWith("sapUiSizeCompact"), "Should remove old density class");
        assert.ok(this.oViewStub.addStyleClass.calledWith("sapUiSizeCompact"), "Should add new density class");
        assert.equal(oPersonalizationModel.getProperty("/density"), "Compact", "Should update model");
    });

    QUnit.test("Should reset personalization to defaults", function (assert) {
        // Arrange
        const oDefaultPersonalization = {
            theme: "sap_horizon",
            density: "cozy"
        };
        sinon.stub(this.oController, "_getDefaultPersonalization").returns(oDefaultPersonalization);
        sinon.stub(this.oController, "_applyPersonalization");
        sinon.stub(this.oController, "onSavePersonalization");
        sinon.stub(sap.m.MessageBox, "confirm").callsArgWith(1, sap.m.MessageBox.Action.OK);

        // Act
        this.oController.onResetPersonalization();

        // Assert
        assert.ok(this.oController._applyPersonalization.calledWith(oDefaultPersonalization), 
                 "Should apply default personalization");
        assert.ok(this.oController.onSavePersonalization.calledOnce, "Should save after reset");
    });

    QUnit.test("Should handle drag and drop events", function (assert) {
        // Arrange
        const oDraggedControl = {
            getId: sinon.stub().returns("draggedTile"),
            getParent: sinon.stub().returns({
                indexOfAggregation: sinon.stub().returns(0)
            })
        };
        const oDragSession = {
            setComplexData: sinon.stub()
        };
        const oEvent = {
            getParameter: sinon.stub()
        };
        
        oEvent.getParameter.withArgs("draggedControl").returns(oDraggedControl);
        oEvent.getParameter.withArgs("dragSession").returns(oDragSession);

        // Act
        this.oController.onDragStart(oEvent);

        // Assert
        assert.ok(oDragSession.setComplexData.calledOnce, "Should set drag session data");
        const oComplexData = oDragSession.setComplexData.getCall(0).args[1];
        assert.equal(oComplexData.id, "draggedTile", "Should set dragged control ID");
        assert.equal(oComplexData.index, 0, "Should set dragged control index");
    });

    QUnit.test("Should enable personalization mode", function (assert) {
        // Arrange
        sinon.stub(this.oController, "_addPersonalizationToolbar");
        sinon.stub(this.oController, "_enableDashboardDragDrop");

        // Act
        this.oController._registerPersonalizationHandlers();

        // Assert
        assert.ok(this.oController._enableDashboardDragDrop.calledOnce, "Should enable drag and drop");
        assert.ok(this.oController._addPersonalizationToolbar.calledOnce, "Should add toolbar");
    });

    QUnit.test("Should save tile order after drag and drop", function (assert) {
        // Arrange
        const oTile1 = { getId: sinon.stub().returns("tile1") };
        const oTile2 = { getId: sinon.stub().returns("tile2") };
        const oDashboard = {
            getAggregation: sinon.stub().withArgs("tiles").returns([oTile1, oTile2])
        };
        
        this.oViewStub.byId = sinon.stub().withArgs("dashboard").returns(oDashboard);
        
        const oPersonalizationModel = new JSONModel({
            dashboardLayout: {},
            preferences: { autoSave: true }
        });
        this.oComponentStub.getModel.withArgs("personalization").returns(oPersonalizationModel);
        sinon.stub(this.oController, "onSavePersonalization");

        // Act
        this.oController._saveTileOrder();

        // Assert
        const aTileOrder = oPersonalizationModel.getProperty("/dashboardLayout/tiles");
        assert.deepEqual(aTileOrder, ["tile1", "tile2"], "Should save tile order");
        assert.ok(this.oController.onSavePersonalization.calledOnce, "Should auto-save when enabled");
    });

    QUnit.test("Should clean up on exit", function (assert) {
        // Arrange
        this.oController._oPersonalizationDialog = {
            destroy: sinon.stub()
        };
        this.oController._oVariantManagement = {
            destroy: sinon.stub()
        };

        // Act
        this.oController.cleanupPersonalization();

        // Assert
        assert.ok(this.oController._oPersonalizationDialog.destroy.calledOnce, "Should destroy dialog");
        assert.ok(this.oController._oVariantManagement.destroy.calledOnce, "Should destroy variant management");
        assert.equal(this.oController._oPersonalizationDialog, null, "Should reset dialog reference");
        assert.equal(this.oController._oVariantManagement, null, "Should reset variant management reference");
    });
});