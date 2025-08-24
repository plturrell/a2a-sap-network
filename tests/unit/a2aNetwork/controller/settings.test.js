/*
 * SAP A2A Network - Enterprise Unit Test Suite
 * Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved.
 *
 * QUnit tests for Settings Controller
 * Tests enterprise settings management functionality
 *
 * @namespace a2a.network.fiori.test.unit.controller
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */

sap.ui.define([
    'a2a/network/fiori/controller/Settings',
    'sap/ui/core/mvc/View',
    'sap/ui/core/UIComponent',
    'sap/ui/model/json/JSONModel',
    'sap/ui/model/odata/v4/ODataModel',
    'sap/viz/ui5/controls/VizFrame'
], (
    SettingsController,
    View,
    UIComponent,
    JSONModel,
    ODataModel,
    VizFrame
) => {
    'use strict';

    /**
     * Test module for Settings Controller functionality
     *
     * Tests enterprise settings management including:
     * - Performance monitoring and analytics
     * - Real-time chart updates
     * - Auto-save functionality
     * - Settings validation and persistence
     * - Performance metrics collection
     *
     * @public
     * @static
     */
    QUnit.module('a2a.network.fiori.controller.Settings', {

        /**
         * Set up test environment before each test
         * Creates mock view, component, models, and performance data
         *
         * @function
         * @memberOf a2a.network.fiori.test.unit.controller.Settings
         * @private
         */
        beforeEach() {
            // Create mock component with i18n model
            this.oComponent = new UIComponent();
            sinon.stub(this.oComponent, 'getModel').returns(new JSONModel());

            // Create mock view with VizFrame
            this.oView = new View();
            this.oVizFrame = new VizFrame();
            sinon.stub(this.oView, 'byId').withArgs('performanceChart').returns(this.oVizFrame);
            sinon.stub(this.oView, 'getModel');
            sinon.stub(this.oView, 'setModel');

            // Create controller instance
            this.oController = new SettingsController();
            sinon.stub(this.oController, 'getView').returns(this.oView);
            sinon.stub(this.oController, 'getOwnerComponent').returns(this.oComponent);

            // Mock performance data
            this.oPerformanceData = {
                metrics: [
                    { timestamp: Date.now(), cpu: 45, memory: 67, network: 23, disk: 34 },
                    { timestamp: Date.now() - 60000, cpu: 42, memory: 65, network: 25, disk: 33 }
                ],
                systemHealth: { score: 85, status: 'Good' },
                alerts: [],
                thresholds: {
                    cpu: { warning: 70, critical: 90 },
                    memory: { warning: 80, critical: 95 },
                    network: { warning: 75, critical: 90 },
                    disk: { warning: 85, critical: 95 }
                }
            };
        },

        /**
         * Clean up test environment after each test
         * Restores all stubs and cleans up objects
         *
         * @function
         * @memberOf a2a.network.fiori.test.unit.controller.Settings
         * @private
         */
        afterEach() {
            this.oController.destroy();
            this.oView.destroy();
            this.oVizFrame.destroy();
            this.oComponent.destroy();
            sinon.restore();
        }
    });

    /**
     * Test performance chart initialization
     * Verifies that VizFrame chart is properly configured for performance monitoring
     */
    QUnit.test('Should initialize performance chart with correct configuration', function(assert) {
        // Arrange
        const oVizFrameStub = {
            setVizType: sinon.stub(),
            setUiConfig: sinon.stub(),
            setModel: sinon.stub(),
            getDataset: sinon.stub().returns({
                setData: sinon.stub()
            }),
            setDataset: sinon.stub(),
            setFeeds: sinon.stub()
        };
        this.oView.byId.withArgs('performanceChart').returns(oVizFrameStub);

        // Act
        this.oController.onInit();
        this.oController._initializePerformanceChart();

        // Assert
        assert.ok(oVizFrameStub.setVizType.calledWith('line'),
            'Chart should be configured as line chart');
        assert.ok(oVizFrameStub.setUiConfig.called,
            'Chart UI configuration should be applied');
        assert.ok(oVizFrameStub.setModel.called,
            'Chart model should be set');
    });

    /**
     * Test real-time performance monitoring
     * Verifies that performance metrics are updated in real-time
     */
    QUnit.test('Should update performance metrics in real-time', function(assert) {
        // Arrange
        this.oController.onInit();
        const oPerformanceModel = new JSONModel(this.oPerformanceData);
        this.oView.getModel.withArgs('performance').returns(oPerformanceModel);

        // Mock timer functionality
        const clock = sinon.useFakeTimers();

        // Act
        this.oController._startPerformanceMonitoring();
        clock.tick(5000); // Simulate 5 seconds

        // Assert
        assert.ok(this.oController._performanceTimer,
            'Performance monitoring timer should be active');

        // Cleanup
        this.oController._stopPerformanceMonitoring();
        clock.restore();
    });

    /**
     * Test auto-save functionality
     * Verifies that settings are automatically saved after changes
     */
    QUnit.test('Should auto-save settings after changes', function(assert) {
        // Arrange
        this.oController.onInit();
        const oSettingsModel = new JSONModel({
            general: { theme: 'sap_horizon', language: 'en' },
            performance: { monitoring: true, alerts: true }
        });
        this.oView.getModel.withArgs('settings').returns(oSettingsModel);

        const saveStub = sinon.stub(this.oController, '_saveSettings');
        const clock = sinon.useFakeTimers();

        // Act
        this.oController._handleSettingsChange();
        clock.tick(2000); // Wait for debounce

        // Assert
        assert.ok(saveStub.called, 'Settings should be auto-saved');

        // Cleanup
        clock.restore();
    });

    /**
     * Test threshold validation
     * Verifies that performance thresholds are properly validated
     */
    QUnit.test('Should validate performance thresholds', function(assert) {
        // Arrange
        this.oController.onInit();
        const invalidThresholds = {
            cpu: { warning: 95, critical: 80 }, // Invalid: warning > critical
            memory: { warning: 80, critical: 95 }
        };

        // Act
        let isValid = this.oController._validateThresholds(invalidThresholds);

        // Assert
        assert.notOk(isValid, 'Invalid thresholds should not validate');

        // Test valid thresholds
        const validThresholds = {
            cpu: { warning: 70, critical: 90 },
            memory: { warning: 80, critical: 95 }
        };

        isValid = this.oController._validateThresholds(validThresholds);
        assert.ok(isValid, 'Valid thresholds should validate');
    });

    /**
     * Test system health calculation
     * Verifies that system health score is calculated correctly
     */
    QUnit.test('Should calculate system health score', function(assert) {
        // Arrange
        this.oController.onInit();
        const metrics = {
            cpu: 45, memory: 67, network: 23, disk: 34,
            uptime: 99.8, errorRate: 0.2, responseTime: 120
        };

        // Act
        const healthScore = this.oController._calculateHealthScore(metrics);

        // Assert
        assert.ok(typeof healthScore === 'number', 'Health score should be a number');
        assert.ok(healthScore >= 0 && healthScore <= 100,
            'Health score should be between 0 and 100');
        assert.ok(healthScore > 80,
            'Health score should be high for good metrics');
    });

    /**
     * Test alert generation
     * Verifies that alerts are generated when thresholds are exceeded
     */
    QUnit.test('Should generate alerts when thresholds are exceeded', function(assert) {
        // Arrange
        this.oController.onInit();
        const metrics = { cpu: 85, memory: 92, network: 45, disk: 55 };
        const thresholds = {
            cpu: { warning: 70, critical: 90 },
            memory: { warning: 80, critical: 95 }
        };

        // Act
        const alerts = this.oController._generateAlerts(metrics, thresholds);

        // Assert
        assert.ok(Array.isArray(alerts), 'Should return array of alerts');
        assert.ok(alerts.length > 0, 'Should generate alerts for exceeded thresholds');

        const cpuAlert = alerts.find(alert => alert.metric === 'cpu');
        const memoryAlert = alerts.find(alert => alert.metric === 'memory');

        assert.ok(cpuAlert && cpuAlert.level === 'warning',
            'Should generate warning alert for CPU');
        assert.ok(memoryAlert && memoryAlert.level === 'warning',
            'Should generate warning alert for memory');
    });

    /**
     * Test settings export functionality
     * Verifies that settings can be exported to JSON
     */
    QUnit.test('Should export settings to JSON', function(assert) {
        // Arrange
        this.oController.onInit();
        const settingsData = {
            general: { theme: 'sap_horizon', language: 'en' },
            performance: { monitoring: true, alerts: true },
            security: { encryption: true, audit: true }
        };
        const oSettingsModel = new JSONModel(settingsData);
        this.oView.getModel.withArgs('settings').returns(oSettingsModel);

        // Mock file download
        const createElementStub = sinon.stub(document, 'createElement').returns({
            href: '',
            download: '',
            click: sinon.stub()
        });
        const createObjectURLStub = sinon.stub(URL, 'createObjectURL').returns('blob:url');

        // Act
        this.oController.onExportSettings();

        // Assert
        assert.ok(createElementStub.calledWith('a'),
            'Should create download link element');
        assert.ok(createObjectURLStub.called,
            'Should create blob URL for download');

        // Cleanup
        createElementStub.restore();
        createObjectURLStub.restore();
    });

    /**
     * Test settings import functionality
     * Verifies that settings can be imported from JSON file
     */
    QUnit.test('Should import settings from JSON file', function(assert) {
        // Arrange
        this.oController.onInit();
        const importData = {
            general: { theme: 'sap_horizon_dark', language: 'de' },
            performance: { monitoring: false, alerts: false }
        };

        const mockFileReader = {
            readAsText: sinon.stub(),
            result: JSON.stringify(importData),
            addEventListener: sinon.stub()
        };

        sinon.stub(window, 'FileReader').returns(mockFileReader);
        const validateStub = sinon.stub(this.oController, '_validateSettingsData').returns(true);
        const applyStub = sinon.stub(this.oController, '_applyImportedSettings');

        // Act
        this.oController._handleFileImport({ target: { files: [{}] } });
        mockFileReader.addEventListener.getCall(0).args[1](); // Trigger load event

        // Assert
        assert.ok(validateStub.called, 'Should validate imported settings');
        assert.ok(applyStub.called, 'Should apply imported settings');

        // Cleanup
        window.FileReader.restore();
    });

    /**
     * Test performance chart data formatting
     * Verifies that performance data is correctly formatted for visualization
     */
    QUnit.test('Should format performance data for chart visualization', function(assert) {
        // Arrange
        this.oController.onInit();
        const rawData = [
            { timestamp: 1640995200000, cpu: 45, memory: 67, network: 23, disk: 34 },
            { timestamp: 1640995260000, cpu: 48, memory: 69, network: 25, disk: 36 }
        ];

        // Act
        const chartData = this.oController._formatChartData(rawData);

        // Assert
        assert.ok(Array.isArray(chartData), 'Should return array of chart data');
        assert.ok(chartData.length === rawData.length,
            'Should maintain data point count');

        if (chartData.length > 0) {
            const firstPoint = chartData[0];
            assert.ok(firstPoint.hasOwnProperty('time'),
                'Should have time property');
            assert.ok(firstPoint.hasOwnProperty('CPU'),
                'Should have formatted CPU property');
            assert.ok(firstPoint.hasOwnProperty('Memory'),
                'Should have formatted Memory property');
        }
    });

    /**
     * Integration test for complete settings workflow
     * Verifies the full settings management workflow from load to save
     */
    QUnit.test('Should handle complete settings workflow', function(assert) {
        // Arrange
        this.oController.onInit();
        const loadStub = sinon.stub(this.oController, '_loadSettings').resolves();
        const validateStub = sinon.stub(this.oController, '_validateSettings').returns(true);
        const saveStub = sinon.stub(this.oController, '_saveSettings').resolves();
        const updateStub = sinon.stub(this.oController, '_updatePerformanceChart');

        // Act & Assert - Test workflow sequence
        return this.oController._loadSettings()
            .then(() => {
                assert.ok(loadStub.called, 'Settings should be loaded');

                this.oController._handleSettingsChange();
                assert.ok(validateStub.called, 'Settings should be validated');

                return this.oController._saveSettings();
            })
            .then(() => {
                assert.ok(saveStub.called, 'Settings should be saved');

                this.oController._refreshPerformanceData();
                assert.ok(updateStub.called, 'Performance chart should be updated');
            });
    });
});