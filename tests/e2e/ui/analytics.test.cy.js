/**
 * Analytics.view.xml Component Tests
 * Test Cases: TC-AN-121 to TC-AN-136
 * Coverage: Analytics Dashboard, KPI Tiles, Charts, Export Functionality
 * 
 * Aligned with actual analytics.view.xml structure:
 * - DynamicPage with title and export actions
 * - KPI tiles using GenericTile and NumericContent
 * - Charts using VizFrame (line, donut)
 * - Top performing agents table
 * - Network statistics display
 */

describe('Analytics.view.xml - Analytics Dashboard', () => {
  beforeEach(() => {
    cy.visit('/analytics');
    cy.viewport(1280, 720);
    
    // Wait for analytics page to load
    cy.get('#analyticsPage').should('be.visible');
  });

  describe('TC-AN-121 to TC-AN-125: Dashboard Core Tests', () => {
    it('TC-AN-121: Should verify analytics dashboard loads correctly', () => {
      // Verify DynamicPage structure from analytics.view.xml
      cy.get('#analyticsPage').should('be.visible');
      cy.get('#analyticsPage .f-DynamicPageTitle').should('be.visible');
      cy.get('#analyticsPage .f-DynamicPageTitle .m-Title').should('contain.text', 'Analytics');
      
      // Verify expanded content description
      cy.get('.f-expandedContent .m-Text').should('contain.text', 'Network performance and insights');
      
      // Verify action buttons in header
      cy.get('[press="onExportReport"]').should('be.visible');
      cy.get('[press="onExportReport"]').should('contain.text', 'Export Report');
      
      // Verify date range selection component
      cy.get('.m-DateRangeSelection').should('be.visible');
      
      // Verify main content area
      cy.get('#analyticsPage .f-content .m-VBox').should('be.visible');
    });

    it('TC-AN-122: Should test KPI tiles display network metrics', () => {
      // Verify KPI tiles from analytics.view.xml structure
      cy.get('.m-HBox .m-VBox').should('have.length', 4);
      
      // Test Total Agents tile
      cy.get('.m-GenericTile').first().within(() => {
        cy.get('.m-GenericTileHeader').should('contain.text', 'Total Agents');
        cy.get('.m-Text').should('contain.text', 'Active in network');
        cy.get('.m-NumericContent').should('be.visible');
        cy.get('[indicator="Up"]').should('exist');
        cy.get('[valueColor="Good"]').should('exist');
      });
      
      // Test Active Services tile
      cy.get('.m-GenericTile').eq(1).within(() => {
        cy.get('.m-GenericTileHeader').should('contain.text', 'Active Services');
        cy.get('.m-Text').should('contain.text', 'Available now');
        cy.get('.m-NumericContent').should('be.visible');
        cy.get('[valueColor="Good"]').should('exist');
      });
      
      // Test Network Load tile with RadialMicroChart
      cy.get('.m-GenericTile').eq(2).within(() => {
        cy.get('.m-GenericTileHeader').should('contain.text', 'Network Load');
        cy.get('.m-Text').should('contain.text', 'Current utilization');
        cy.get('.mc-RadialMicroChart').should('be.visible');
      });
      
      // Test Average Reputation tile
      cy.get('.m-GenericTile').eq(3).within(() => {
        cy.get('.m-GenericTileHeader').should('contain.text', 'Avg Reputation');
        cy.get('.m-Text').should('contain.text', 'Network average');
        cy.get('.m-NumericContent').should('be.visible');
        cy.get('[scale="/200"]').should('exist');
      });
    });

    it('TC-AN-123: Should test date range selection functionality', () => {
      // Test actual DateRangeSelection from analytics.view.xml
      cy.get('.m-DateRangeSelection').should('be.visible');
      cy.get('.m-DateRangeSelection input').should('be.visible');
      
      // Test date range selection
      cy.get('.m-DateRangeSelection input').click();
      cy.get('.m-DateRangeSelection .m-Popover').should('be.visible');
      
      // Select a date range (testing calendar functionality)
      cy.get('.m-Calendar').should('be.visible');
      cy.get('.m-CalendarDate').first().click();
      cy.get('.m-CalendarDate').last().click();
      
      // Verify date range change triggers onDateRangeChange
      cy.get('.m-DateRangeSelection input').should('not.have.value', '');
      
      // Verify data refresh after date change
      cy.get('#agentTrendChart').should('be.visible');
    });

    it('TC-AN-124: Should test export report functionality', () => {
      // Test actual Export Report button from analytics.view.xml
      cy.get('[press="onExportReport"]').should('be.visible');
      cy.get('[press="onExportReport"]').should('have.attr', 'icon', 'sap-icon://excel-attachment');
      
      // Click export report button
      cy.get('[press="onExportReport"]').click();
      
      // Verify the controller action is triggered (onExportReport method)
      // Since this likely shows a message toast or dialog, we can verify behavior
      cy.get('.sapMMessageToast').should('be.visible');
    });

    it('TC-AN-125: Should test analytics page responsiveness', () => {
      // Test tablet class from analytics.view.xml
      cy.get('#analyticsPage.a2a-tablet-page').should('be.visible');
      
      // Test responsive behavior at different viewports
      cy.viewport(768, 1024); // Tablet
      cy.get('#analyticsPage').should('be.visible');
      cy.get('.m-HBox').should('be.visible'); // KPI tiles should still be visible
      
      cy.viewport(320, 568); // Mobile
      cy.get('#analyticsPage').should('be.visible');
      cy.get('.m-GenericTile').should('be.visible'); // Tiles should adapt
      
      cy.viewport(1280, 720); // Desktop
      cy.get('#analyticsPage').should('be.visible');
      cy.get('.m-HBox .m-VBox').should('have.length', 4); // All tiles visible
    });
  });

  describe('TC-AN-126 to TC-AN-130: Charts and Visualization Tests', () => {
    it('TC-AN-126: Should verify agent activity trends chart', () => {
      // Test actual VizFrame from analytics.view.xml
      cy.get('#agentTrendChart').should('be.visible');
      cy.get('#agentTrendChart').should('have.attr', 'vizType', 'line');
      
      // Verify chart panel container
      cy.get('.m-Panel').contains('Agent Activity Trends').should('be.visible');
      
      // Verify chart dimensions and structure
      cy.get('#agentTrendChart').should('have.attr', 'height', '400px');
      cy.get('#agentTrendChart').should('have.attr', 'width', '100%');
      
      // Verify VizFrame configuration
      cy.get('#agentTrendChart').should('have.attr', 'uiConfig');
      
      // Test chart data binding - should have FlattenedDataset
      cy.get('#agentTrendChart .viz-FlattenedDataset').should('exist');
      
      // Verify dimension and measure definitions exist
      cy.get('#agentTrendChart .viz-DimensionDefinition[name="Date"]').should('exist');
      cy.get('#agentTrendChart .viz-MeasureDefinition[name="Active Agents"]').should('exist');
      cy.get('#agentTrendChart .viz-MeasureDefinition[name="Total Messages"]').should('exist');
    });

    it('TC-AN-127: Should test service categories donut chart', () => {
      // Test actual service category chart from analytics.view.xml
      cy.get('#serviceCategoryChart').should('be.visible');
      cy.get('#serviceCategoryChart').should('have.attr', 'vizType', 'donut');
      
      // Verify chart panel container
      cy.get('.m-Panel').contains('Service Categories').should('be.visible');
      
      // Verify chart dimensions
      cy.get('#serviceCategoryChart').should('have.attr', 'height', '300px');
      cy.get('#serviceCategoryChart').should('have.attr', 'width', '400px');
      
      // Verify donut chart specific structure
      cy.get('#serviceCategoryChart .viz-FlattenedDataset').should('exist');
      cy.get('#serviceCategoryChart .viz-DimensionDefinition[name="Category"]').should('exist');
      cy.get('#serviceCategoryChart .viz-MeasureDefinition[name="Count"]').should('exist');
      
      // Test chart positioning within HBox layout
      cy.get('.m-HBox .m-Panel').contains('Service Categories').should('be.visible');
      cy.get('.m-Panel .m-Text').should('have.class', 'sapUiSmallMarginEnd');
    });

    it('TC-AN-128: Should test top performing agents table', () => {
      // Test actual top performing agents table from analytics.view.xml
      cy.get('.m-Panel').contains('Top Performing Agents').should('be.visible');
      
      // Verify table structure
      cy.get('.m-Table[items="{/TopAgents}"]').should('be.visible');
      
      // Test table columns from analytics.view.xml
      cy.get('.m-Table .m-Column').should('have.length', 4);
      cy.get('.m-Table .m-Column').eq(0).should('contain.text', 'Agent');
      cy.get('.m-Table .m-Column').eq(1).should('contain.text', 'Reputation');
      cy.get('.m-Table').eq(2).should('contain.text', 'Tasks');
      cy.get('.m-Table .m-Column').eq(3).should('contain.text', 'Avg Response');
      
      // Test table item structure
      cy.get('.m-ColumnListItem').should('exist');
      cy.get('.m-ColumnListItem .m-Text').should('exist'); // Agent name
      cy.get('.m-ColumnListItem .m-ObjectNumber').should('exist'); // Reputation
      cy.get('.m-ColumnListItem .m-ObjectNumber[state="Success"]').should('exist');
      
      // Test table data binding
      cy.get('.m-Table[items="{/TopAgents}"]').within(() => {
        cy.get('[text="{name}"]').should('exist'); // Agent name binding
        cy.get('[number="{reputation}"]').should('exist'); // Reputation binding
        cy.get('[text="{completedTasks}"]').should('exist'); // Tasks binding
        cy.get('[text="{avgResponseTime} ms"]').should('exist'); // Response time binding
      });
    });

    it('TC-AN-129: Should test layout and spacing in analytics dashboard', () => {
      // Test HBox layout for charts section from analytics.view.xml
      cy.get('.m-HBox').should('be.visible');
      
      // Verify the two panels are side by side in HBox
      cy.get('.m-HBox .m-Panel').should('have.length', 2);
      cy.get('.m-HBox .m-Panel').first().should('contain.text', 'Service Categories');
      cy.get('.m-HBox .m-Panel').last().should('contain.text', 'Top Performing Agents');
      
      // Test VBox main container spacing
      cy.get('.f-content .m-VBox').should('have.class', 'sapUiMediumMargin');
      
      // Test KPI tiles HBox with wrap and margin
      cy.get('.m-HBox[wrap="Wrap"]').should('be.visible');
      cy.get('.m-HBox[wrap="Wrap"]').should('have.class', 'sapUiSmallMarginBottom');
      
      // Test individual VBox containers for KPI tiles
      cy.get('.m-HBox .m-VBox.sapUiSmallMarginEnd').should('have.length', 3);
    });

    it('TC-AN-130: Should test chart panel structure and styling', () => {
      // Test Agent Activity Trends panel structure
      cy.get('.m-Panel').contains('Agent Activity Trends').should('be.visible');
      cy.get('.m-Panel').contains('Agent Activity Trends').should('have.class', 'sapUiMediumMarginBottom');
      
      // Test chart within panel
      cy.get('.m-Panel').contains('Agent Activity Trends').within(() => {
        cy.get('#agentTrendChart').should('be.visible');
        cy.get('.m-PanelHeader').should('contain.text', 'Agent Activity Trends');
      });
      
      // Test other panels don't have margin bottom class
      cy.get('.m-Panel').contains('Service Categories').should('not.have.class', 'sapUiMediumMarginBottom');
      cy.get('.m-Panel').contains('Top Performing Agents').should('not.have.class', 'sapUiMediumMarginBottom');
      
      // Verify chart configuration attributes
      cy.get('#agentTrendChart').should('have.attr', 'uiConfig', '{applicationSet:\'fiori\'}');
    });
  });

  describe('TC-AN-131 to TC-AN-136: Data Model and Binding Tests', () => {
    it('TC-AN-131: Should test NetworkStats data binding', () => {
      // Test data binding from analytics.view.xml for KPI tiles
      // Total Agents tile
      cy.get('.m-NumericContent[value="{/NetworkStats/0/totalAgents}"]').should('exist');
      
      // Active Services tile  
      cy.get('.m-NumericContent[value="{/NetworkStats/0/totalServices}"]').should('exist');
      
      // Network Load RadialMicroChart
      cy.get('.mc-RadialMicroChart[percentage="{= ${/NetworkStats/0/networkLoad} * 100 }"]').should('exist');
      
      // Average Reputation tile
      cy.get('.m-NumericContent[value="{/NetworkStats/0/averageReputation}"]').should('exist');
      
      // Verify data model structure expectations
      cy.window().its('sap.ui.getCore().byId("analyticsPage").getModel()').should('exist');
    });

    it('TC-AN-132: Should test chart data binding and feeds', () => {
      // Test VizFrame feeds configuration from analytics.view.xml
      cy.get('#agentTrendChart .viz-FeedItem[uid="valueAxis"]').should('exist');
      cy.get('#agentTrendChart .viz-FeedItem[type="Measure"]').should('exist');
      cy.get('#agentTrendChart .viz-FeedItem[values="Active Agents,Total Messages"]').should('exist');
      
      cy.get('#agentTrendChart .viz-FeedItem[uid="categoryAxis"]').should('exist');
      cy.get('#agentTrendChart .viz-FeedItem[type="Dimension"]').should('exist');
      cy.get('#agentTrendChart .viz-FeedItem[values="Date"]').should('exist');
      
      // Test service category chart dataset binding
      cy.get('#serviceCategoryChart .viz-DimensionDefinition[value="{category}"]').should('exist');
      cy.get('#serviceCategoryChart .viz-MeasureDefinition[value="{count}"]').should('exist');
    });

    it('TC-AN-133: Should test conditional styling and expressions', () => {
      // Test conditional styling from analytics.view.xml
      // Network Load RadialMicroChart color condition
      cy.get('.mc-RadialMicroChart[valueColor="{= ${/NetworkStats/0/networkLoad} > 0.8 ? \'Error\' : \'Good\'}"]').should('exist');
      
      // Test table item status conditional styling
      cy.get('.m-ObjectNumber[state="Success"]').should('exist');
      
      // Test conditional info state for workflow items
      cy.get('[info="{= ${isActive} ? \'Active\' : \'Inactive\' }"]').should('exist');
      cy.get('[infoState="{= ${isActive} ? \'Success\' : \'Error\' }"]').should('exist');
      
      // Verify expression binding for executions count
      cy.get('[text="{= ${executions}.length }"]').should('exist');
    });

    it('TC-AN-134: Should test internationalization support', () => {
      // Test i18n binding from analytics.view.xml
      cy.get('.f-DynamicPageTitle .m-Title[text="{i18n>analytics}"]').should('exist');
      
      // Verify i18n model is available
      cy.window().its('sap.ui.getCore().byId("analyticsPage").getModel("i18n")').should('exist');
      
      // Test resource model configuration from manifest.json
      // This verifies the supportedLocales: ["en", "de", "fr", "es", "zh", "ja"]
      cy.window().then((win) => {
        const component = win.sap.ui.getCore().byId('analyticsPage').getController().getOwnerComponent();
        const i18nModel = component.getModel('i18n');
        expect(i18nModel).to.exist;
        expect(i18nModel.getResourceBundle()).to.exist;
      });
    });

    it('TC-AN-135: Should test print dashboard functionality', () => {
      cy.get('[data-testid="print-dashboard"]').click();
      
      // Verify print preview
      cy.get('[data-testid="print-preview"]').should('be.visible');
      
      // Check print optimization
      cy.get('[data-testid="print-preview"]').within(() => {
        // Verify print-optimized layout
        cy.get('[data-testid="print-header"]').should('be.visible');
        cy.get('[data-testid="print-kpis"]').should('be.visible');
        cy.get('[data-testid="print-charts"]').should('be.visible');
        cy.get('[data-testid="print-footer"]').should('be.visible');
        
        // Check print settings
        cy.get('[data-testid="print-orientation"]').should('exist');
        cy.get('[data-testid="print-scale"]').should('exist');
        cy.get('[data-testid="include-colors"]').should('exist');
      });
      
      // Configure print settings
      cy.get('[data-testid="print-orientation"]').select('landscape');
      cy.get('[data-testid="print-scale"]').select('fit-to-page');
      cy.get('[data-testid="include-colors"]').check();
      
      // Test print button
      cy.window().then((win) => {
        cy.stub(win, 'print').as('printStub');
      });
      
      cy.get('[data-testid="print-button"]').click();
      cy.get('@printStub').should('have.been.called');
      
      // Close print preview
      cy.get('[data-testid="close-print-preview"]').click();
      cy.get('[data-testid="print-preview"]').should('not.exist');
    });

    it('TC-AN-136: Should verify dashboard sharing features', () => {
      cy.get('[data-testid="share-dashboard"]').click();
      
      // Verify sharing options
      cy.get('[data-testid="share-dialog"]').should('be.visible');
      cy.get('[data-testid="share-dialog"]').within(() => {
        cy.get('[data-testid="share-link"]').should('be.visible');
        cy.get('[data-testid="share-email"]').should('be.visible');
        cy.get('[data-testid="share-embed"]').should('be.visible');
        cy.get('[data-testid="share-permissions"]').should('be.visible');
      });
      
      // Test link sharing
      cy.get('[data-testid="generate-share-link"]').click();
      cy.get('[data-testid="share-link-generated"]').should('be.visible');
      cy.get('[data-testid="share-url"]').should('not.be.empty');
      
      // Test copy link
      cy.get('[data-testid="copy-share-link"]').click();
      cy.get('[data-testid="copy-success"]').should('be.visible');
      
      // Test permission settings
      cy.get('[data-testid="share-permissions"]').select('view-only');
      cy.get('[data-testid="expiration-date"]').type('2024-12-31');
      cy.get('[data-testid="require-login"]').check();
      
      // Update share settings
      cy.get('[data-testid="update-share-settings"]').click();
      cy.get('[data-testid="settings-updated"]').should('be.visible');
      
      // Test email sharing
      cy.get('[data-testid="share-via-email"]').click();
      cy.get('[data-testid="email-recipients"]').type('colleague@company.com');
      cy.get('[data-testid="email-message"]').type('Please review the latest analytics dashboard.');
      
      cy.intercept('POST', '/api/analytics/share/email').as('shareEmail');
      cy.get('[data-testid="send-email"]').click();
      
      cy.wait('@shareEmail');
      cy.get('[data-testid="email-sent-success"]').should('be.visible');
      
      // Test embed code generation
      cy.get('[data-testid="generate-embed-code"]').click();
      cy.get('[data-testid="embed-code"]').should('not.be.empty');
      cy.get('[data-testid="embed-code"]').should('contain', '<iframe');
    });
  });
});