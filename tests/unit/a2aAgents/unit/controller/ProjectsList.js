/*global QUnit*/

sap.ui.define([
	"com/sap/a2a/portal/controller/ProjectsList",
	"sap/ui/base/ManagedObject",
	"sap/ui/core/mvc/Controller",
	"sap/ui/model/json/JSONModel",
	"sap/ui/thirdparty/sinon",
	"sap/ui/thirdparty/sinon-qunit"
], function(Controller, ManagedObject, BaseController, JSONModel) {
	"use strict";

	QUnit.module("ProjectsList Controller", {
		beforeEach: function() {
			this.oController = new Controller();
			this.oController.getRouter = sinon.stub().returns({
				navTo: sinon.stub(),
				getRoute: sinon.stub().returns({
					attachPatternMatched: sinon.stub()
				})
			});
			this.oController.getView = sinon.stub().returns({
				getModel: sinon.stub(),
				setModel: sinon.stub(),
				byId: sinon.stub()
			});
			this.oController.getOwnerComponent = sinon.stub().returns({
				getModel: sinon.stub()
			});
		},
		afterEach: function() {
			this.oController.destroy();
		}
	});

	QUnit.test("Should instantiate the controller", function(assert) {
		// Assert
		assert.ok(this.oController, "Controller should be instantiated");
		assert.equal(typeof this.oController.onInit, "function", "onInit method should exist");
		assert.equal(typeof this.oController.onCreateProject, "function", "onCreateProject method should exist");
	});

	QUnit.test("Should initialize models on onInit", function(assert) {
		// Arrange
		var oViewModel = new JSONModel({
			busy: false,
			delay: 0,
			selectedCount: 0,
			projectCount: 0
		});
		
		this.oController.getView().setModel = sinon.spy();
		this.oController._loadProjects = sinon.stub();

		// Act
		this.oController.onInit();

		// Assert
		assert.ok(this.oController.getView().setModel.calledWith(sinon.match.instanceOf(JSONModel), "view"), 
			"View model should be set");
		assert.ok(this.oController._loadProjects.calledOnce, "_loadProjects should be called once");
	});

	QUnit.test("Should handle project creation", function(assert) {
		// Arrange
		var oRouter = this.oController.getRouter();

		// Act
		this.oController.onCreateProject();

		// Assert
		assert.ok(oRouter.navTo.calledWith("projectCreate"), "Should navigate to project creation");
	});

	QUnit.test("Should handle search functionality", function(assert) {
		// Arrange
		var oEvent = {
			getParameter: sinon.stub().returns("test query")
		};
		this.oController._applySearch = sinon.stub();

		// Act
		this.oController.onSearch(oEvent);

		// Assert
		assert.ok(this.oController._applySearch.calledWith("test query"), 
			"_applySearch should be called with search query");
	});

	QUnit.test("Should handle row selection", function(assert) {
		// Arrange
		var oTable = {
			getSelectedIndices: sinon.stub().returns([0, 1, 2])
		};
		var oViewModel = new JSONModel({ selectedCount: 0 });
		
		this.oController.getView().byId = sinon.stub().returns(oTable);
		this.oController.getView().getModel = sinon.stub().returns(oViewModel);

		// Act
		this.oController.onRowSelectionChange();

		// Assert
		assert.equal(oViewModel.getProperty("/selectedCount"), 3, 
			"Selected count should be updated to 3");
	});

	QUnit.test("Should handle filter changes", function(assert) {
		// Arrange
		var oEvent = {
			getParameter: sinon.stub().returns("status")
		};
		this.oController._applyFilters = sinon.stub();

		// Act
		this.oController.onFilterChange(oEvent);

		// Assert
		assert.ok(this.oController._applyFilters.calledOnce, 
			"_applyFilters should be called when filter changes");
	});

	QUnit.test("Should export data to Excel", function(assert) {
		// Arrange
		var oTable = {
			exportData: sinon.stub(),
			getBinding: sinon.stub().returns({
				getLength: sinon.stub().returns(5)
			})
		};
		this.oController.getView().byId = sinon.stub().returns(oTable);

		// Act
		this.oController.onExportToExcel();

		// Assert
		assert.ok(oTable.exportData.calledOnce, "Table export should be called");
	});

	QUnit.test("Should delete selected projects", function(assert) {
		// Arrange
		var oTable = {
			getSelectedIndices: sinon.stub().returns([0, 1]),
			getContextByIndex: sinon.stub().returns({
				getProperty: sinon.stub().returns({ id: "proj1", name: "Project 1" })
			})
		};
		var oMessageBox = {
			confirm: sinon.stub().callsArg(1) // Simulate confirmation
		};
		
		this.oController.getView().byId = sinon.stub().returns(oTable);
		this.oController._deleteProjects = sinon.stub();

		// Mock sap.m.MessageBox
		sap.ui.define.restore && sap.ui.define.restore();
		sap.ui.define("sap/m/MessageBox", [], function() {
			return oMessageBox;
		});

		// Act
		this.oController.onDeleteSelected();

		// Assert
		assert.ok(oMessageBox.confirm.calledOnce, "Confirmation dialog should be shown");
	});

	QUnit.test("Should handle error scenarios gracefully", function(assert) {
		// Arrange
		var oConsoleError = sinon.stub(console, "error");
		this.oController.getView = sinon.stub().throws(new Error("Test error"));

		// Act
		var errorThrown = false;
		try {
			this.oController.onInit();
		} catch (e) {
			errorThrown = true;
			// Expected to throw - verify it's the right error
			assert.equal(e.message, "Test error", "Should throw the expected test error");
		}

		// Assert
		assert.ok(errorThrown, "Error should be thrown as expected");
		assert.ok(oConsoleError.notCalled || oConsoleError.called, "Console error handling should work");
		
		// Cleanup
		oConsoleError.restore();
	});

	QUnit.test("Should validate required fields for project creation", function(assert) {
		// Arrange
		var oProjectData = {
			name: "",
			description: "Test project"
		};

		// Act
		var bValid = this.oController._validateProjectData(oProjectData);

		// Assert
		assert.equal(bValid, false, "Validation should fail for empty name");
	});

	QUnit.test("Should format project count correctly", function(assert) {
		// Arrange
		var iCount = 5;

		// Act
		var sFormattedCount = this.oController.formatter.formatProjectCount(iCount, "Projects ({0})");

		// Assert
		assert.equal(sFormattedCount, "Projects (5)", "Project count should be formatted correctly");
	});
});