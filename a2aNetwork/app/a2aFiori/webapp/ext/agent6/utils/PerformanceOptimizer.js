sap.ui.define([
    "sap/ui/Device",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "./SecurityUtils"
], (Device, MessageToast, JSONModel, SecurityUtils) => {
    "use strict";

    /**
     * @class PerformanceOptimizer
     * @description Utility class providing performance optimization features for Agent 6.
     * Includes lazy loading, virtualization, caching, and responsive utilities.
     */
    return {

        /**
         * @function createVirtualizedTable
         * @description Creates a table with virtualization for large datasets.
         * @param {Object} oConfig - Configuration object
         * @returns {sap.m.Table} Virtualized table
         */
        createVirtualizedTable(oConfig) {
            const oTable = new sap.m.Table({
                growing: true,
                growingThreshold: this._getGrowingThreshold(),
                growingScrollToLoad: true,
                sticky: Device.system.desktop ? ["ColumnHeaders"] : "",
                mode: oConfig.mode || "None",
                updateFinished: this._onTableUpdateFinished.bind(this)
            });

            // Enable lazy loading
            this._enableLazyLoading(oTable, oConfig);

            return oTable;
        },

        /**
         * @function _getGrowingThreshold
         * @description Returns appropriate growing threshold based on device.
         * @returns {number} Growing threshold
         * @private
         */
        _getGrowingThreshold() {
            if (Device.system.phone) {
                return 10;
            } else if (Device.system.tablet) {
                return 20;
            }
            return 30;

        },

        /**
         * @function _onTableUpdateFinished
         * @description Handles table update finished event for optimization.
         * @param {sap.ui.base.Event} oEvent - Update finished event
         * @private
         */
        _onTableUpdateFinished(oEvent) {
            const aItems = oEvent.getSource().getItems();
            const iThreshold = this._getGrowingThreshold();

            // Defer rendering of non-visible items
            aItems.forEach((oItem, iIndex) => {
                if (iIndex > iThreshold) {
                    oItem.addStyleClass("sapUiInvisibleText");
                    setTimeout(() => {
                        oItem.removeStyleClass("sapUiInvisibleText");
                    }, 50 * (iIndex - iThreshold));
                }
            });
        },

        /**
         * @function _enableLazyLoading
         * @description Enables lazy loading for a table.
         * @param {sap.m.Table} oTable - Table to enhance
         * @param {Object} oConfig - Configuration
         * @private
         */
        _enableLazyLoading(oTable, oConfig) {
            const that = this;

            oTable.attachUpdateFinished((oEvent) => {
                const oBinding = oTable.getBinding("items");
                if (!oBinding || !oConfig.loadMoreHandler) {return;}

                const iTotalItems = oBinding.getLength();
                const iLoadedItems = oTable.getItems().length;

                // Check if we need to load more
                if (iLoadedItems < iTotalItems && iLoadedItems % that._getGrowingThreshold() === 0) {
                    const oBundle = oTable.getModel("i18n").getResourceBundle();
                    MessageToast.show(oBundle.getText("lazyload.loadingMore"));

                    // Call the load more handler
                    oConfig.loadMoreHandler(iLoadedItems, that._getGrowingThreshold());
                }
            });
        },

        /**
         * @function createCachedModel
         * @description Creates a JSON model with caching capabilities.
         * @param {Object} oData - Initial data
         * @param {Object} oOptions - Cache options
         * @returns {sap.ui.model.json.JSONModel} Cached model
         */
        createCachedModel(oData, oOptions) {
            const oModel = new JSONModel(oData);
            const oCacheConfig = Object.assign({
                ttl: 300000, // 5 minutes default
                key: `agent6_cache_${ Date.now()}`
            }, oOptions);

            // Add cache metadata
            oModel._cacheConfig = oCacheConfig;
            oModel._cacheTimestamp = Date.now();

            // Override setData to update cache
            const fnOriginalSetData = oModel.setData.bind(oModel);
            oModel.setData = function(oData, bMerge) {
                fnOriginalSetData(oData, bMerge);
                this._cacheTimestamp = Date.now();

                // Store in session storage
                try {
                    sessionStorage.setItem(this._cacheConfig.key, JSON.stringify({
                        data: oData,
                        timestamp: this._cacheTimestamp
                    }));
                } catch (e) {
                    // Storage quota exceeded
                }
            };

            // Add cache validation method
            oModel.isCacheValid = function() {
                return (Date.now() - this._cacheTimestamp) < this._cacheConfig.ttl;
            };

            // Add cache retrieval method
            oModel.loadFromCache = function() {
                try {
                    const sCached = sessionStorage.getItem(this._cacheConfig.key);
                    if (sCached) {
                        const oCached = JSON.parse(sCached);
                        if ((Date.now() - oCached.timestamp) < this._cacheConfig.ttl) {
                            this.setData(oCached.data);
                            return true;
                        }
                    }
                } catch (e) {
                    // Invalid cache
                }
                return false;
            };

            return oModel;
        },

        /**
         * @function debounce
         * @description Creates a debounced function for performance.
         * @param {Function} fn - Function to debounce
         * @param {number} delay - Delay in milliseconds
         * @returns {Function} Debounced function
         */
        debounce(fn, delay) {
            let timeoutId;
            return function() {
                const context = this;
                const args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    fn.apply(context, args);
                }, delay);
            };
        },

        /**
         * @function throttle
         * @description Creates a throttled function for performance.
         * @param {Function} fn - Function to throttle
         * @param {number} limit - Time limit between calls
         * @returns {Function} Throttled function
         */
        throttle(fn, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => {
                        inThrottle = false;
                    }, limit);
                }
            };
        },

        /**
         * @function optimizeChartData
         * @description Optimizes chart data for better performance.
         * @param {Array} aData - Chart data
         * @param {number} iMaxPoints - Maximum data points
         * @returns {Array} Optimized data
         */
        optimizeChartData(aData, iMaxPoints) {
            if (!aData || aData.length <= iMaxPoints) {
                return aData;
            }

            // Sample data points evenly
            const iStep = Math.ceil(aData.length / iMaxPoints);
            const aOptimized = [];

            for (let i = 0; i < aData.length; i += iStep) {
                aOptimized.push(aData[i]);
            }

            // Always include the last point
            if (aOptimized[aOptimized.length - 1] !== aData[aData.length - 1]) {
                aOptimized.push(aData[aData.length - 1]);
            }

            return aOptimized;
        },

        /**
         * @function lazyLoadImages
         * @description Sets up lazy loading for images.
         * @param {sap.ui.core.Control} oContainer - Container with images
         */
        lazyLoadImages(oContainer) {
            if (!window.IntersectionObserver) {return;}

            const imageObserver = new IntersectionObserver(((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        if (img.dataset.src) {
                            img.src = img.dataset.src;
                            img.classList.remove("lazy-image");
                            imageObserver.unobserve(img);
                        }
                    }
                });
            }), {
                rootMargin: "50px"
            });

            // Find all lazy images
            const $lazyImages = oContainer.$().find("img.lazy-image");
            $lazyImages.each(function() {
                imageObserver.observe(this);
            });
        },

        /**
         * @function cleanupMemory
         * @description Cleans up memory by removing unused objects.
         * @param {Object} oContext - Context object to clean
         */
        cleanupMemory(oContext) {
            // Clean up event listeners
            if (oContext._eventListeners) {
                oContext._eventListeners.forEach((listener) => {
                    if (listener.element && listener.event) {
                        listener.element.removeEventListener(listener.event, listener.handler);
                    }
                });
                oContext._eventListeners = [];
            }

            // Clean up timers
            if (oContext._timers) {
                oContext._timers.forEach((timerId) => {
                    clearTimeout(timerId);
                });
                oContext._timers = [];
            }

            // Clean up intervals
            if (oContext._intervals) {
                oContext._intervals.forEach((intervalId) => {
                    clearInterval(intervalId);
                });
                oContext._intervals = [];
            }

            // Force garbage collection hint
            if (window.gc) {
                window.gc();
            }
        },

        /**
         * @function getResponsiveGridSpan
         * @description Returns responsive grid span based on device.
         * @returns {string} Grid span configuration
         */
        getResponsiveGridSpan() {
            if (Device.system.phone) {
                return "L12 M12 S12";
            } else if (Device.system.tablet) {
                return "L6 M12 S12";
            }
            return "L4 M6 S12";

        },

        /**
         * @function preloadFragments
         * @description Preloads fragments for better performance.
         * @param {Array<string>} aFragmentNames - Fragment names to preload
         * @returns {Promise} Promise when all fragments are loaded
         */
        preloadFragments(aFragmentNames) {
            const aPromises = aFragmentNames.map((sFragmentName) => {
                return sap.ui.core.Fragment.load({
                    name: sFragmentName,
                    type: "XML"
                }).then((oFragment) => {
                    // Cache the fragment
                    if (!window._fragmentCache) {
                        window._fragmentCache = {};
                    }
                    window._fragmentCache[sFragmentName] = oFragment;
                    return oFragment;
                });
            });

            return Promise.all(aPromises);
        },

        /**
         * @function measurePerformance
         * @description Measures performance of a function.
         * @param {string} sName - Performance marker name
         * @param {Function} fn - Function to measure
         * @returns {*} Function result
         */
        measurePerformance(sName, fn) {
            if (window.performance && window.performance.mark) {
                const sStartMark = `${sName }_start`;
                const sEndMark = `${sName }_end`;

                performance.mark(sStartMark);
                const result = fn();
                performance.mark(sEndMark);

                performance.measure(sName, sStartMark, sEndMark);

                const measure = performance.getEntriesByName(sName)[0];
                if (measure) {
                    // Performance measure found
                }

                return result;
            }
            return fn();

        }
    };
});