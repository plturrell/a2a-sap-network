sap.ui.define([
    'com/sap/a2a/control/AgentStatusIndicator',
    'sap/ui/qunit/utils/createAndAppendDiv'
], (AgentStatusIndicator, createAndAppendDiv) => {
    'use strict';

    QUnit.module('AgentStatusIndicator Control', {
        beforeEach() {
            createAndAppendDiv('content');
            this.oStatusIndicator = new AgentStatusIndicator();
        },

        afterEach() {
            this.oStatusIndicator.destroy();
            document.getElementById('content').innerHTML = '';
        }
    });

    QUnit.test('Should instantiate with default properties', function(assert) {
        // Assert
        assert.equal(this.oStatusIndicator.getStatus(), 'inactive', 'Default status should be \'inactive\'');
        assert.equal(this.oStatusIndicator.getSize(), 'Medium', 'Default size should be \'Medium\'');
        assert.equal(this.oStatusIndicator.getAnimated(), true, 'Default animated should be true');
        assert.equal(this.oStatusIndicator.getShowTooltip(), true, 'Default showTooltip should be true');
    });

    QUnit.test('Should set and get status property', function(assert) {
        // Act
        this.oStatusIndicator.setStatus('active');

        // Assert
        assert.equal(this.oStatusIndicator.getStatus(), 'active', 'Status should be set to \'active\'');
    });

    QUnit.test('Should set and get size property', function(assert) {
        // Act
        this.oStatusIndicator.setSize('Large');

        // Assert
        assert.equal(this.oStatusIndicator.getSize(), 'Large', 'Size should be set to \'Large\'');
    });

    QUnit.test('Should render with correct CSS classes', function(assert) {
        // Arrange
        this.oStatusIndicator.setStatus('active');
        this.oStatusIndicator.setSize('Large');

        // Act
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Assert
        const $indicator = this.oStatusIndicator.$();
        assert.ok($indicator.hasClass('agentStatusIndicator'), 'Should have base CSS class');
        assert.ok($indicator.hasClass('status-active'), 'Should have status-specific CSS class');
        assert.ok($indicator.hasClass('size-Large'), 'Should have size-specific CSS class');
    });

    QUnit.test('Should update CSS classes when status changes', function(assert) {
        // Arrange
        this.oStatusIndicator.setStatus('active');
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Act
        this.oStatusIndicator.setStatus('error');
        sap.ui.getCore().applyChanges();

        // Assert
        const $indicator = this.oStatusIndicator.$();
        assert.ok(!$indicator.hasClass('status-active'), 'Should not have old status class');
        assert.ok($indicator.hasClass('status-error'), 'Should have new status class');
    });

    QUnit.test('Should show tooltip when enabled', function(assert) {
        // Arrange
        this.oStatusIndicator.setStatus('active');
        this.oStatusIndicator.setShowTooltip(true);
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Act
        const sTooltip = this.oStatusIndicator.$().attr('title');

        // Assert
        assert.ok(sTooltip, 'Tooltip should be present');
        assert.equal(sTooltip, 'Active', 'Tooltip should show status');
    });

    QUnit.test('Should not show tooltip when disabled', function(assert) {
        // Arrange
        this.oStatusIndicator.setStatus('active');
        this.oStatusIndicator.setShowTooltip(false);
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Act
        const sTooltip = this.oStatusIndicator.$().attr('title');

        // Assert
        assert.notOk(sTooltip, 'Tooltip should not be present');
    });

    QUnit.test('Should fire status change event', function(assert) {
        // Arrange
        let bEventFired = false;
        let sNewStatus = '';

        this.oStatusIndicator.attachStatusChange((oEvent) => {
            bEventFired = true;
            sNewStatus = oEvent.getParameter('newStatus');
        });

        // Act
        this.oStatusIndicator.setStatus('warning');

        // Assert
        assert.ok(bEventFired, 'Status change event should be fired');
        assert.equal(sNewStatus, 'warning', 'Event should contain new status');
    });

    QUnit.test('Should handle animation toggle', function(assert) {
        // Arrange
        this.oStatusIndicator.setAnimated(true);
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Act
        this.oStatusIndicator.setAnimated(false);
        sap.ui.getCore().applyChanges();

        // Assert
        const $indicator = this.oStatusIndicator.$();
        assert.ok(!$indicator.hasClass('animated'), 'Animation class should be removed');
    });

    QUnit.test('Should support all status values', function(assert) {
        // Arrange
        const aValidStatuses = ['active', 'inactive', 'error', 'warning'];

        // Act & Assert
        aValidStatuses.forEach(sStatus => {
            this.oStatusIndicator.setStatus(sStatus);
            assert.equal(this.oStatusIndicator.getStatus(), sStatus, `Status should be set to '${sStatus}'`);
        });
    });

    QUnit.test('Should support all size values', function(assert) {
        // Arrange
        const aValidSizes = ['Small', 'Medium', 'Large'];

        // Act & Assert
        aValidSizes.forEach(sSize => {
            this.oStatusIndicator.setSize(sSize);
            assert.equal(this.oStatusIndicator.getSize(), sSize, `Size should be set to '${sSize}'`);
        });
    });

    QUnit.test('Should have proper accessibility attributes', function(assert) {
        // Arrange
        this.oStatusIndicator.setStatus('active');
        this.oStatusIndicator.placeAt('content');
        sap.ui.getCore().applyChanges();

        // Act
        const $indicator = this.oStatusIndicator.$();

        // Assert
        assert.equal($indicator.attr('role'), 'status', 'Should have status role');
        assert.equal($indicator.attr('aria-label'), 'Agent Status: Active', 'Should have descriptive aria-label');
    });
});