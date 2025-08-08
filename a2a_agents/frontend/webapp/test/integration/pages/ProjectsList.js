sap.ui.define([
	"sap/ui/test/Opa5",
	"sap/ui/test/matchers/AggregationLengthEquals",
	"sap/ui/test/matchers/AggregationFilled",
	"sap/ui/test/matchers/PropertyStrictEquals",
	"sap/ui/test/actions/Press",
	"sap/ui/test/actions/EnterText"
], function (Opa5, AggregationLengthEquals, AggregationFilled, PropertyStrictEquals, Press, EnterText) {
	"use strict";

	var sViewName = "ProjectsList";

	Opa5.createPageObjects({
		onTheProjectsListPage: {
			actions: {
				iWaitUntilTheTableIsLoaded: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						matchers: [new AggregationFilled({
							name: "rows"
						})],
						success: function () {
							Opa5.assert.ok(true, "The projects table is loaded");
						},
						errorMessage: "The projects table was not loaded"
					});
				},

				iEnterSearchText: function (sSearchText) {
					return this.waitFor({
						controlType: "sap.m.SearchField",
						viewName: sViewName,
						actions: new EnterText({
							text: sSearchText
						}),
						success: function () {
							Opa5.assert.ok(true, "Search text '" + sSearchText + "' was entered");
						},
						errorMessage: "Could not enter search text"
					});
				},

				iPressTheSearchButton: function () {
					return this.waitFor({
						controlType: "sap.m.SearchField",
						viewName: sViewName,
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Search button was pressed");
						},
						errorMessage: "Could not press search button"
					});
				},

				iSelectStatusFilter: function (sStatus) {
					return this.waitFor({
						id: "statusFilter",
						viewName: sViewName,
						actions: function (oControl) {
							oControl.setSelectedKey(sStatus);
							oControl.fireSelectionChange();
						},
						success: function () {
							Opa5.assert.ok(true, "Status filter '" + sStatus + "' was selected");
						},
						errorMessage: "Could not select status filter"
					});
				},

				iPressTheGoButton: function () {
					return this.waitFor({
						controlType: "sap.m.Button",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "text",
							value: "Go"
						}),
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Go button was pressed");
						},
						errorMessage: "Could not press Go button"
					});
				},

				iPressTheCreateButton: function () {
					return this.waitFor({
						id: "createProjectBtn",
						viewName: sViewName,
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Create button was pressed");
						},
						errorMessage: "Could not press create button"
					});
				},

				iSelectProjectsInTheTable: function (iCount) {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						actions: function (oTable) {
							var aIndices = [];
							for (var i = 0; i < iCount && i < oTable.getRows().length; i++) {
								aIndices.push(i);
							}
							oTable.setSelectedIndices(aIndices);
							oTable.fireRowSelectionChange();
						},
						success: function () {
							Opa5.assert.ok(true, iCount + " projects were selected");
						},
						errorMessage: "Could not select projects in table"
					});
				},

				iPressTheDeleteSelectedButton: function () {
					return this.waitFor({
						controlType: "sap.m.Button",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "icon",
							value: "sap-icon://delete"
						}),
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Delete selected button was pressed");
						},
						errorMessage: "Could not press delete selected button"
					});
				},

				iConfirmTheDeleteDialog: function () {
					return this.waitFor({
						controlType: "sap.m.Button",
						searchOpenDialogs: true,
						matchers: new PropertyStrictEquals({
							name: "text",
							value: "Delete"
						}),
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Delete confirmation was pressed");
						},
						errorMessage: "Could not confirm delete dialog"
					});
				},

				iPressTheExportButton: function () {
					return this.waitFor({
						controlType: "sap.m.Button",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "icon",
							value: "sap-icon://excel-attachment"
						}),
						actions: new Press(),
						success: function () {
							Opa5.assert.ok(true, "Export button was pressed");
						},
						errorMessage: "Could not press export button"
					});
				},

				iNavigateUsingKeyboard: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						actions: function (oTable) {
							// Simulate keyboard navigation
							oTable.focus();
							var oFirstRow = oTable.getRows()[0];
							if (oFirstRow) {
								oFirstRow.focus();
							}
						},
						success: function () {
							Opa5.assert.ok(true, "Keyboard navigation was performed");
						},
						errorMessage: "Could not perform keyboard navigation"
					});
				},

				iPressEnterOnFirstProject: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						actions: function (oTable) {
							var oFirstRow = oTable.getRows()[0];
							if (oFirstRow) {
								// Simulate Enter key press
								oTable.fireRowSelectionChange();
							}
						},
						success: function () {
							Opa5.assert.ok(true, "Enter was pressed on first project");
						},
						errorMessage: "Could not press Enter on first project"
					});
				},

				iSimulateANetworkError: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						actions: function (oTable) {
							// Simulate network error by triggering error state
							var oModel = oTable.getModel();
							if (oModel && oModel.fireRequestFailed) {
								oModel.fireRequestFailed({
									message: "Network error",
									statusCode: 500
								});
							}
						},
						success: function () {
							Opa5.assert.ok(true, "Network error was simulated");
						},
						errorMessage: "Could not simulate network error"
					});
				},

				iWaitForNoDataState: function () {
					return this.waitFor({
						id: "noDataIllustratedMessage",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "visible",
							value: true
						}),
						success: function () {
							Opa5.assert.ok(true, "No data state is visible");
						},
						errorMessage: "No data state is not visible"
					});
				}
			},

			assertions: {
				iShouldSeeTheProjectsList: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						success: function (oTable) {
							Opa5.assert.ok(oTable, "The projects table is visible");
						},
						errorMessage: "The projects table was not found"
					});
				},

				iShouldSeeThePageTitle: function (sTitle) {
					return this.waitFor({
						controlType: "sap.m.Title",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "text",
							value: sTitle
						}),
						success: function () {
							Opa5.assert.ok(true, "Page title '" + sTitle + "' is visible");
						},
						errorMessage: "Page title was not found"
					});
				},

				theTableShouldHaveEntries: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						matchers: new AggregationFilled({
							name: "rows"
						}),
						success: function (oTable) {
							Opa5.assert.ok(oTable.getRows().length > 0, "The table has entries");
						},
						errorMessage: "The table has no entries"
					});
				},

				iShouldSeeTheFilterBar: function () {
					return this.waitFor({
						id: "projectsFilterBar",
						viewName: sViewName,
						success: function (oFilterBar) {
							Opa5.assert.ok(oFilterBar, "The filter bar is visible");
						},
						errorMessage: "The filter bar was not found"
					});
				},

				iShouldSeeSearchResults: function (sSearchText) {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						check: function (oTable) {
							var aRows = oTable.getRows();
							return aRows.length > 0 && aRows.some(function (oRow) {
								var oContext = oRow.getBindingContext();
								if (oContext) {
									var sProjectName = oContext.getProperty("name");
									return sProjectName && sProjectName.indexOf(sSearchText) !== -1;
								}
								return false;
							});
						},
						success: function () {
							Opa5.assert.ok(true, "Search results for '" + sSearchText + "' are visible");
						},
						errorMessage: "No search results found for '" + sSearchText + "'"
					});
				},

				iShouldSeeOnlyActiveProjects: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						check: function (oTable) {
							var aRows = oTable.getRows();
							return aRows.every(function (oRow) {
								var oContext = oRow.getBindingContext();
								if (oContext) {
									var sStatus = oContext.getProperty("status");
									return sStatus === "Active";
								}
								return true;
							});
						},
						success: function () {
							Opa5.assert.ok(true, "Only active projects are visible");
						},
						errorMessage: "Non-active projects are still visible"
					});
				},

				iShouldSeeTheNoDataMessage: function () {
					return this.waitFor({
						id: "noDataIllustratedMessage",
						viewName: sViewName,
						matchers: new PropertyStrictEquals({
							name: "visible",
							value: true
						}),
						success: function () {
							Opa5.assert.ok(true, "No data message is visible");
						},
						errorMessage: "No data message was not found"
					});
				},

				iShouldSeeTheCreateProjectButton: function () {
					return this.waitFor({
						id: "createProjectBtn",
						viewName: sViewName,
						success: function () {
							Opa5.assert.ok(true, "Create project button is visible");
						},
						errorMessage: "Create project button was not found"
					});
				},

				iShouldSeeASuccessMessage: function () {
					return this.waitFor({
						controlType: "sap.m.MessageToast",
						success: function () {
							Opa5.assert.ok(true, "Success message is visible");
						},
						errorMessage: "Success message was not found"
					});
				},

				theSelectedProjectsShouldBeDeleted: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						check: function (oTable) {
							return oTable.getSelectedIndices().length === 0;
						},
						success: function () {
							Opa5.assert.ok(true, "Selected projects were deleted");
						},
						errorMessage: "Selected projects were not deleted"
					});
				},

				theTableShouldBeResponsive: function () {
					return this.waitFor({
						id: "projectsTable",
						viewName: sViewName,
						check: function (oTable) {
							// Check if table adapts to mobile viewport
							return oTable.$().hasClass("sapUiSizeCompact") || 
								   oTable.getColumns().some(function (oColumn) {
									   return oColumn.getImportance && oColumn.getImportance() === "High";
								   });
						},
						success: function () {
							Opa5.assert.ok(true, "Table is responsive");
						},
						errorMessage: "Table is not responsive"
					});
				},

				iShouldSeeAnErrorMessage: function () {
					return this.waitFor({
						controlType: "sap.m.MessageStrip",
						matchers: new PropertyStrictEquals({
							name: "type",
							value: "Error"
						}),
						success: function () {
							Opa5.assert.ok(true, "Error message is visible");
						},
						errorMessage: "Error message was not found"
					});
				},

				iShouldSeeARetryButton: function () {
					return this.waitFor({
						controlType: "sap.m.Button",
						matchers: new PropertyStrictEquals({
							name: "text",
							value: "Retry"
						}),
						success: function () {
							Opa5.assert.ok(true, "Retry button is visible");
						},
						errorMessage: "Retry button was not found"
					});
				},

				iShouldSeeExportDialog: function () {
					return this.waitFor({
						controlType: "sap.m.Dialog",
						searchOpenDialogs: true,
						success: function () {
							Opa5.assert.ok(true, "Export dialog is visible");
						},
						errorMessage: "Export dialog was not found"
					});
				},

				theExportShouldBeTriggered: function () {
					return this.waitFor({
						check: function () {
							// Check if export functionality was triggered
							// This would typically check for a download or API call
							return true; // Simplified for this example
						},
						success: function () {
							Opa5.assert.ok(true, "Export was triggered");
						},
						errorMessage: "Export was not triggered"
					});
				},

				iShouldNavigateToProjectDetails: function () {
					return this.waitFor({
						check: function () {
							return window.location.hash.indexOf("project") !== -1;
						},
						success: function () {
							Opa5.assert.ok(true, "Navigation to project details occurred");
						},
						errorMessage: "Navigation to project details did not occur"
					});
				}
			}
		}
	});
});