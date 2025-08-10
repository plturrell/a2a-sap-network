/**
 * Unit Tests for CacheService
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */
sap.ui.define([
    "sap/ui/test/Opa5",
    "a2a/network/fiori/service/CacheService"
], function(Opa5, CacheService) {
    "use strict";

    QUnit.module("CacheService", {
        beforeEach: function() {
            CacheService.init();
            CacheService.clear();
        },
        afterEach: function() {
            CacheService.clear();
            CacheService.destroy();
        }
    });

    QUnit.test("Should initialize successfully", function(assert) {
        // Arrange & Act
        CacheService.init();
        
        // Assert
        assert.ok(true, "CacheService initialized without errors");
    });

    QUnit.test("Should store and retrieve data", function(assert) {
        // Arrange
        var testKey = "test-key";
        var testData = { message: "Hello World", timestamp: Date.now() };
        
        // Act
        var setResult = CacheService.set(testKey, testData);
        var retrievedData = CacheService.get(testKey);
        
        // Assert
        assert.ok(setResult, "Data should be stored successfully");
        assert.deepEqual(retrievedData, testData, "Retrieved data should match stored data");
    });

    QUnit.test("Should handle cache expiration", function(assert) {
        var done = assert.async();
        
        // Arrange
        var testKey = "expire-test";
        var testData = "expiring data";
        var shortTTL = 100; // 100ms
        
        // Act
        CacheService.set(testKey, testData, { ttl: shortTTL });
        
        // Check immediately - should exist
        var immediateResult = CacheService.get(testKey);
        assert.equal(immediateResult, testData, "Data should be available immediately");
        
        // Check after expiration
        setTimeout(function() {
            var expiredResult = CacheService.get(testKey);
            assert.equal(expiredResult, null, "Data should be null after expiration");
            done();
        }, shortTTL + 50);
    });

    QUnit.test("Should remove data", function(assert) {
        // Arrange
        var testKey = "remove-test";
        var testData = "data to remove";
        
        // Act
        CacheService.set(testKey, testData);
        var beforeRemoval = CacheService.get(testKey);
        var removeResult = CacheService.remove(testKey);
        var afterRemoval = CacheService.get(testKey);
        
        // Assert
        assert.equal(beforeRemoval, testData, "Data should exist before removal");
        assert.ok(removeResult, "Remove operation should succeed");
        assert.equal(afterRemoval, null, "Data should be null after removal");
    });

    QUnit.test("Should invalidate by tags", function(assert) {
        // Arrange
        var key1 = "tag-test-1";
        var key2 = "tag-test-2";
        var key3 = "tag-test-3";
        var data1 = "data with tag1";
        var data2 = "data with tag2";
        var data3 = "data with both tags";
        
        // Act
        CacheService.set(key1, data1, { tags: ["tag1"] });
        CacheService.set(key2, data2, { tags: ["tag2"] });
        CacheService.set(key3, data3, { tags: ["tag1", "tag2"] });
        
        var invalidated = CacheService.invalidateByTags(["tag1"]);
        
        // Assert
        assert.equal(invalidated, 2, "Should invalidate 2 entries with tag1");
        assert.equal(CacheService.get(key1), null, "Key1 should be invalidated");
        assert.equal(CacheService.get(key2), data2, "Key2 should remain");
        assert.equal(CacheService.get(key3), null, "Key3 should be invalidated");
    });

    QUnit.test("Should provide cache statistics", function(assert) {
        // Arrange
        CacheService.set("stats-test-1", "data1", { tags: ["test"] });
        CacheService.set("stats-test-2", "data2", { tags: ["test"] });
        
        // Act
        var stats = CacheService.getStats();
        
        // Assert
        assert.ok(stats, "Stats should be returned");
        assert.equal(stats.totalEntries, 2, "Should report correct entry count");
        assert.ok(stats.entries.length >= 2, "Should include entry details");
    });

    QUnit.test("Should handle invalid operations gracefully", function(assert) {
        // Test null/undefined key
        var result1 = CacheService.set(null, "data");
        assert.notOk(result1, "Should reject null key");
        
        var result2 = CacheService.get("non-existent-key");
        assert.equal(result2, null, "Should return null for non-existent key");
        
        var result3 = CacheService.remove("non-existent-key");
        assert.ok(result3, "Remove should succeed even for non-existent key");
        
        var result4 = CacheService.invalidateByTags([]);
        assert.equal(result4, 0, "Should return 0 for empty tags array");
    });

    QUnit.test("Should clear all cache", function(assert) {
        // Arrange
        CacheService.set("clear-test-1", "data1");
        CacheService.set("clear-test-2", "data2");
        CacheService.set("clear-test-3", "data3");
        
        // Act
        var clearResult = CacheService.clear();
        var stats = CacheService.getStats();
        
        // Assert
        assert.ok(clearResult, "Clear operation should succeed");
        assert.equal(stats.totalEntries, 0, "Should have no entries after clear");
    });
});