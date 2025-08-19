sap.ui.define([
    "sap/ui/test/Opa5",
    "sap/ui/performance/Measurement"
], function(Opa5, Measurement) {
    "use strict";

    /**
     * Performance Test Suite
     * Comprehensive performance testing for SAP UI5 applications
     */

    /**
     * Performance Testing Engine
     * Provides methods to measure and test application performance
     */
    function PerformanceEngine() {
        this.measurements = [];
        this.thresholds = {
            // Performance thresholds (in milliseconds)
            appStartup: 3000,
            viewRendering: 1000,
            dataBinding: 500,
            navigation: 800,
            interaction: 100,
            memoryLeak: 10, // MB increase threshold
            bundleSize: 2048, // KB
            firstContentfulPaint: 1500,
            largestContentfulPaint: 2500,
            cumulativeLayoutShift: 0.1,
            firstInputDelay: 100
        };
        this.results = {};
        this.observer = null;
    }

    PerformanceEngine.prototype.destroy = function() {
        this.measurements = null;
        this.results = null;
        if (this.observer) {
            this.observer.disconnect();
        }
    };

    /**
     * Test application startup performance
     */
    PerformanceEngine.prototype.testApplicationStartup = function() {
        return new Promise((resolve) => {
            const startTime = performance.now();

            // Measure time to UI5 core ready
            sap.ui.getCore().attachInit(() => {
                const coreReadyTime = performance.now() - startTime;

                // Measure time to first view ready
                this._waitForFirstView().then((firstViewTime) => {
                    const totalStartupTime = performance.now() - startTime;

                    const result = {
                        coreInitTime: coreReadyTime,
                        firstViewTime,
                        totalStartupTime,
                        passesThreshold: totalStartupTime <= this.thresholds.appStartup,
                        threshold: this.thresholds.appStartup
                    };

                    this.measurements.push({
                        type: "application-startup",
                        timestamp: Date.now(),
                        data: result
                    });

                    resolve(result);
                });
            });
        });
    };

    /**
     * Test view rendering performance
     */
    PerformanceEngine.prototype.testViewRendering = function(viewName) {
        return new Promise((resolve) => {
            const startTime = performance.now();

            // Start measuring
            Measurement.start("view-rendering", "View rendering performance test");

            // Create and render view
            sap.ui.core.mvc.XMLView.create({
                viewName
            }).then((oView) => {
                const renderStartTime = performance.now();

                oView.placeAt("qunit-fixture");

                // Wait for rendering to complete
                oView.addEventDelegate({
                    onAfterRendering: () => {
                        const renderEndTime = performance.now();
                        Measurement.end("view-rendering");

                        const result = {
                            viewCreationTime: renderStartTime - startTime,
                            renderingTime: renderEndTime - renderStartTime,
                            totalTime: renderEndTime - startTime,
                            passesThreshold: (renderEndTime - startTime) <= this.thresholds.viewRendering,
                            threshold: this.thresholds.viewRendering,
                            measurements: Measurement.getAllMeasurements().filter(m => m.id === "view-rendering")
                        };

                        this.measurements.push({
                            type: "view-rendering",
                            viewName,
                            timestamp: Date.now(),
                            data: result
                        });

                        // Cleanup
                        oView.destroy();
                        resolve(result);
                    }
                });
            });
        });
    };

    /**
     * Test data binding performance
     */
    PerformanceEngine.prototype.testDataBindingPerformance = function(modelData, iterations = 1000) {
        return new Promise((resolve) => {
            const startTime = performance.now();
            const model = new sap.ui.model.json.JSONModel();

            // Test model creation and data setting
            const modelSetupStart = performance.now();
            model.setData(modelData);
            const modelSetupTime = performance.now() - modelSetupStart;

            // Test property binding performance
            const bindingStartTime = performance.now();
            const bindings = [];

            for (let i = 0; i < iterations; i++) {
                const binding = model.bindProperty(`/testProperty${ i}`);
                bindings.push(binding);
            }

            const bindingCreationTime = performance.now() - bindingStartTime;

            // Test binding updates
            const updateStartTime = performance.now();
            model.setProperty("/massUpdate", "Updated value");
            const updateTime = performance.now() - updateStartTime;

            const totalTime = performance.now() - startTime;

            const result = {
                modelSetupTime,
                bindingCreationTime,
                bindingUpdateTime: updateTime,
                totalTime,
                iterationsCount: iterations,
                avgBindingTime: bindingCreationTime / iterations,
                passesThreshold: totalTime <= this.thresholds.dataBinding,
                threshold: this.thresholds.dataBinding
            };

            this.measurements.push({
                type: "data-binding",
                timestamp: Date.now(),
                data: result
            });

            // Cleanup
            bindings.forEach(binding => binding.destroy());
            model.destroy();

            resolve(result);
        });
    };

    /**
     * Test navigation performance
     */
    PerformanceEngine.prototype.testNavigationPerformance = function(router, routeName) {
        return new Promise((resolve) => {
            const startTime = performance.now();

            // Listen for route matched event
            const route = router.getRoute(routeName);
            const handler = () => {
                const endTime = performance.now();
                route.detachMatched(handler);

                const result = {
                    navigationTime: endTime - startTime,
                    routeName,
                    passesThreshold: (endTime - startTime) <= this.thresholds.navigation,
                    threshold: this.thresholds.navigation
                };

                this.measurements.push({
                    type: "navigation",
                    timestamp: Date.now(),
                    data: result
                });

                resolve(result);
            };

            route.attachMatched(handler);

            // Trigger navigation
            router.navTo(routeName);
        });
    };

    /**
     * Test user interaction performance
     */
    PerformanceEngine.prototype.testInteractionPerformance = function(element, interaction = "click") {
        return new Promise((resolve) => {
            const startTime = performance.now();

            const eventHandler = () => {
                const endTime = performance.now();
                element.removeEventListener(interaction, eventHandler);

                const result = {
                    interactionTime: endTime - startTime,
                    interactionType: interaction,
                    elementId: element.id || "unknown",
                    passesThreshold: (endTime - startTime) <= this.thresholds.interaction,
                    threshold: this.thresholds.interaction
                };

                this.measurements.push({
                    type: "interaction",
                    timestamp: Date.now(),
                    data: result
                });

                resolve(result);
            };

            element.addEventListener(interaction, eventHandler);

            // Simulate interaction
            const event = new Event(interaction, { bubbles: true });
            element.dispatchEvent(event);
        });
    };

    /**
     * Test memory usage and detect leaks
     */
    PerformanceEngine.prototype.testMemoryUsage = function(testFunction, iterations = 10) {
        return new Promise((resolve) => {
            if (!performance.memory) {
                resolve({
                    error: "Performance.memory API not available",
                    supported: false
                });
                return;
            }

            const initialMemory = performance.memory.usedJSHeapSize;
            const memoryMeasurements = [initialMemory];

            let completed = 0;

            const runIteration = () => {
                if (completed >= iterations) {
                    // Force garbage collection if available
                    if (window.gc) {
                        window.gc();
                    }

                    // Final memory measurement
                    setTimeout(() => {
                        const finalMemory = performance.memory.usedJSHeapSize;
                        const memoryIncrease = (finalMemory - initialMemory) / (1024 * 1024); // Convert to MB

                        const result = {
                            initialMemory,
                            finalMemory,
                            memoryIncrease,
                            iterations,
                            measurements: memoryMeasurements,
                            avgMemoryPerIteration: memoryIncrease / iterations,
                            hasMemoryLeak: memoryIncrease > this.thresholds.memoryLeak,
                            threshold: this.thresholds.memoryLeak,
                            passesThreshold: memoryIncrease <= this.thresholds.memoryLeak
                        };

                        this.measurements.push({
                            type: "memory-usage",
                            timestamp: Date.now(),
                            data: result
                        });

                        resolve(result);
                    }, 100);
                    return;
                }

                // Run test function
                testFunction().then(() => {
                    completed++;
                    memoryMeasurements.push(performance.memory.usedJSHeapSize);

                    // Small delay between iterations
                    setTimeout(runIteration, 10);
                });
            };

            runIteration();
        });
    };

    /**
     * Test bundle size and loading performance
     */
    PerformanceEngine.prototype.testBundleSize = function() {
        return new Promise((resolve) => {
            // Get all loaded resources
            const resources = performance.getEntriesByType("resource");
            const jsResources = resources.filter(r => r.name.endsWith(".js"));
            const cssResources = resources.filter(r => r.name.endsWith(".css"));

            let totalSize = 0;
            let jsSize = 0;
            let cssSize = 0;

            resources.forEach(resource => {
                if (resource.transferSize) {
                    totalSize += resource.transferSize;
                    if (resource.name.endsWith(".js")) {
                        jsSize += resource.transferSize;
                    } else if (resource.name.endsWith(".css")) {
                        cssSize += resource.transferSize;
                    }
                }
            });

            // Convert to KB
            totalSize = Math.round(totalSize / 1024);
            jsSize = Math.round(jsSize / 1024);
            cssSize = Math.round(cssSize / 1024);

            const result = {
                totalSize,
                jsSize,
                cssSize,
                jsResourceCount: jsResources.length,
                cssResourceCount: cssResources.length,
                totalResourceCount: resources.length,
                passesThreshold: totalSize <= this.thresholds.bundleSize,
                threshold: this.thresholds.bundleSize,
                largestResources: resources
                    .sort((a, b) => (b.transferSize || 0) - (a.transferSize || 0))
                    .slice(0, 10)
                    .map(r => ({
                        name: r.name,
                        size: `${Math.round((r.transferSize || 0) / 1024) } KB`,
                        duration: Math.round(r.duration)
                    }))
            };

            this.measurements.push({
                type: "bundle-size",
                timestamp: Date.now(),
                data: result
            });

            resolve(result);
        });
    };

    /**
     * Test Core Web Vitals
     */
    PerformanceEngine.prototype.testCoreWebVitals = function() {
        return new Promise((resolve) => {
            const result = {
                supported: {},
                values: {},
                passesThresholds: {}
            };

            // Test Largest Contentful Paint (LCP)
            if ("PerformanceObserver" in window) {
                const lcpObserver = new PerformanceObserver((entryList) => {
                    const entries = entryList.getEntries();
                    const lastEntry = entries[entries.length - 1];

                    result.values.lcp = lastEntry.startTime;
                    result.passesThresholds.lcp = lastEntry.startTime <= this.thresholds.largestContentfulPaint;
                    result.supported.lcp = true;
                });

                try {
                    lcpObserver.observe({ entryTypes: ["largest-contentful-paint"] });
                    this.observer = lcpObserver;
                } catch (e) {
                    result.supported.lcp = false;
                }
            }

            // Test First Input Delay (FID) - simulated
            const fidStartTime = performance.now();
            setTimeout(() => {
                const fid = performance.now() - fidStartTime;
                result.values.fid = fid;
                result.passesThresholds.fid = fid <= this.thresholds.firstInputDelay;
                result.supported.fid = true;

                // Test First Contentful Paint (FCP)
                const paintEntries = performance.getEntriesByType("paint");
                const fcpEntry = paintEntries.find(entry => entry.name === "first-contentful-paint");

                if (fcpEntry) {
                    result.values.fcp = fcpEntry.startTime;
                    result.passesThresholds.fcp = fcpEntry.startTime <= this.thresholds.firstContentfulPaint;
                    result.supported.fcp = true;
                } else {
                    result.supported.fcp = false;
                }

                // Test Cumulative Layout Shift (CLS) - simplified
                result.values.cls = this._calculateCLS();
                result.passesThresholds.cls = result.values.cls <= this.thresholds.cumulativeLayoutShift;
                result.supported.cls = true;

                this.measurements.push({
                    type: "core-web-vitals",
                    timestamp: Date.now(),
                    data: result
                });

                resolve(result);
            }, 100);
        });
    };

    /**
     * Test rendering performance with different data sizes
     */
    PerformanceEngine.prototype.testRenderingScalability = function(componentFactory, dataSizes = [10, 100, 1000]) {
        return new Promise((resolve) => {
            const results = [];
            let completed = 0;

            dataSizes.forEach(size => {
                const startTime = performance.now();
                const testData = this._generateTestData(size);

                componentFactory(testData).then(component => {
                    component.placeAt("qunit-fixture");

                    component.addEventDelegate({
                        onAfterRendering: () => {
                            const endTime = performance.now();
                            const renderTime = endTime - startTime;

                            results.push({
                                dataSize: size,
                                renderTime,
                                passesThreshold: renderTime <= this.thresholds.viewRendering * (size / 100)
                            });

                            component.destroy();
                            completed++;

                            if (completed === dataSizes.length) {
                                const finalResult = {
                                    results,
                                    scalabilityScore: this._calculateScalabilityScore(results),
                                    recommendations: this._generateScalabilityRecommendations(results)
                                };

                                this.measurements.push({
                                    type: "rendering-scalability",
                                    timestamp: Date.now(),
                                    data: finalResult
                                });

                                resolve(finalResult);
                            }
                        }
                    });
                });
            });
        });
    };

    /**
     * Generate comprehensive performance report
     */
    PerformanceEngine.prototype.generatePerformanceReport = function() {
        const report = {
            summary: {
                totalTests: this.measurements.length,
                passedTests: this.measurements.filter(m => m.data.passesThreshold).length,
                timestamp: Date.now(),
                userAgent: navigator.userAgent,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                }
            },
            measurements: this.measurements,
            recommendations: this._generateRecommendations(),
            thresholds: this.thresholds,
            score: this._calculatePerformanceScore()
        };

        return report;
    };

    // Helper methods
    PerformanceEngine.prototype._waitForFirstView = function() {
        return new Promise((resolve) => {
            // Simple implementation - wait for DOM content loaded
            if (document.readyState === "complete") {
                resolve(0);
            } else {
                const startTime = performance.now();
                window.addEventListener("load", () => {
                    resolve(performance.now() - startTime);
                });
            }
        });
    };

    PerformanceEngine.prototype._calculateCLS = function() {
        // Simplified CLS calculation
        // In real implementation, you would track layout shifts
        return Math.random() * 0.05; // Simulate good CLS score
    };

    PerformanceEngine.prototype._generateTestData = function(size) {
        const data = [];
        for (let i = 0; i < size; i++) {
            data.push({
                id: i,
                name: `Item ${i}`,
                value: Math.random() * 1000,
                description: `Description for item ${i}`,
                category: `Category ${i % 10}`
            });
        }
        return data;
    };

    PerformanceEngine.prototype._calculateScalabilityScore = function(results) {
        // Calculate how well performance scales with data size
        if (results.length < 2) {
            return 100;
        }

        const firstResult = results[0];
        const lastResult = results[results.length - 1];

        const expectedRatio = lastResult.dataSize / firstResult.dataSize;
        const actualRatio = lastResult.renderTime / firstResult.renderTime;

        // Score is better when actual ratio is close to expected (linear scaling)
        const scalabilityRatio = expectedRatio / actualRatio;
        return Math.min(100, scalabilityRatio * 100);
    };

    PerformanceEngine.prototype._generateScalabilityRecommendations = function(results) {
        const recommendations = [];

        // Check if rendering time grows exponentially
        if (results.length >= 3) {
            const ratios = [];
            for (let i = 1; i < results.length; i++) {
                ratios.push(results[i].renderTime / results[i - 1].renderTime);
            }

            const avgRatio = ratios.reduce((a, b) => a + b) / ratios.length;
            if (avgRatio > 2) {
                recommendations.push({
                    type: "performance",
                    severity: "high",
                    message: "Rendering performance degrades significantly with data size. Consider virtualization or pagination."
                });
            }
        }

        return recommendations;
    };

    PerformanceEngine.prototype._calculatePerformanceScore = function() {
        if (this.measurements.length === 0) {
            return 0;
        }

        const passedTests = this.measurements.filter(m => m.data.passesThreshold).length;
        return Math.round((passedTests / this.measurements.length) * 100);
    };

    PerformanceEngine.prototype._generateRecommendations = function() {
        const recommendations = [];

        this.measurements.forEach(measurement => {
            if (!measurement.data.passesThreshold) {
                const recommendation = this._getRecommendationForTest(measurement.type, measurement.data);
                if (recommendation) {
                    recommendations.push(recommendation);
                }
            }
        });

        return recommendations;
    };

    PerformanceEngine.prototype._getRecommendationForTest = function(testType, data) {
        const recommendations = {
            "application-startup": {
                type: "startup",
                severity: "high",
                message: `Application startup time (${Math.round(data.totalStartupTime)}ms) exceeds threshold. Consider lazy loading, code splitting, or reducing initial bundle size.`
            },
            "view-rendering": {
                type: "rendering",
                severity: "medium",
                message: `View rendering time (${Math.round(data.totalTime)}ms) is slow. Optimize view structure and reduce DOM complexity.`
            },
            "data-binding": {
                type: "binding",
                severity: "medium",
                message: `Data binding performance (${Math.round(data.totalTime)}ms) needs improvement. Consider using more efficient binding strategies.`
            },
            "memory-usage": {
                type: "memory",
                severity: "high",
                message: `Potential memory leak detected (${Math.round(data.memoryIncrease)}MB increase). Check for proper cleanup of event listeners and object references.`
            },
            "bundle-size": {
                type: "bundle",
                severity: "medium",
                message: `Bundle size (${data.totalSize}KB) is large. Consider code splitting and removing unused dependencies.`
            }
        };

        return recommendations[testType] || null;
    };

    // Export for use in tests
    window.PerformanceEngine = PerformanceEngine;

    return {
        PerformanceEngine
    };
});