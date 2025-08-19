/*global QUnit*/

sap.ui.define([
	"sap/ui/test/opaQunit",
	"./pages/ProjectsList",
	"./pages/App"
], function (opaTest) {
	"use strict";

	QUnit.module("Projects List");

	opaTest("Should see the projects list", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop();

		// Actions
		When.onTheAppPage.iLookAtTheScreen();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeTheProjectsList().
			and.iShouldSeeThePageTitle("Projects");

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be able to load projects", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iWaitUntilTheTableIsLoaded();

		// Assertions
		Then.onTheProjectsListPage.theTableShouldHaveEntries().
			and.iShouldSeeTheFilterBar();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be able to search for projects", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iEnterSearchText("Test Project").
			and.iPressTheSearchButton();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeSearchResults("Test Project");

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be able to filter projects by status", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iSelectStatusFilter("Active").
			and.iPressTheGoButton();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeOnlyActiveProjects();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should show no data message when no projects exist", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iWaitForNoDataState();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeTheNoDataMessage().
			and.iShouldSeeTheCreateProjectButton();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be able to create a new project", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iPressTheCreateButton();

		// Assertions
		Then.onTheAppPage.iShouldNavigateToCreateProject();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be able to select and delete projects", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iSelectProjectsInTheTable(2).
			and.iPressTheDeleteSelectedButton().
			and.iConfirmTheDeleteDialog();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeASuccessMessage().
			and.theSelectedProjectsShouldBeDeleted();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should be responsive on mobile devices", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnAPhone({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iLookAtTheScreen();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeTheProjectsList().
			and.theTableShouldBeResponsive();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should handle error scenarios gracefully", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iSimulateANetworkError();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeAnErrorMessage().
			and.iShouldSeeARetryButton();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should export projects to Excel", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iWaitUntilTheTableIsLoaded().
			and.iPressTheExportButton();

		// Assertions
		Then.onTheProjectsListPage.iShouldSeeExportDialog().
			and.theExportShouldBeTriggered();

		// Cleanup
		Then.iTeardownMyApp();
	});

	opaTest("Should support keyboard navigation", function (Given, When, Then) {
		// Arrangements
		Given.iStartMyAppOnADesktop({
			hash: "projects"
		});

		// Actions
		When.onTheProjectsListPage.iNavigateUsingKeyboard().
			and.iPressEnterOnFirstProject();

		// Assertions
		Then.onTheProjectsListPage.iShouldNavigateToProjectDetails();

		// Cleanup
		Then.iTeardownMyApp();
	});
});