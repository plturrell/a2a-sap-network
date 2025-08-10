/**
 * Batch Operations Service
 * 
 * Provides efficient batch processing for OData operations, reducing network
 * overhead and improving performance for enterprise applications.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */
sap.ui.define([
    "sap/base/Log"
], function(Log) {
    "use strict";

    var BatchService = {

        /* =========================================================== */
        /* Constants                                                   */
        /* =========================================================== */

        BATCH_SIZE_LIMIT: 20,
        BATCH_TIMEOUT: 100, // milliseconds
        MAX_RETRIES: 3,
        RETRY_DELAY: 1000, // milliseconds

        /* =========================================================== */
        /* Lifecycle                                                   */
        /* =========================================================== */

        /**
         * Initialize the batch service
         * @public
         * @since 1.0.0
         */
        init: function() {
            this._batchQueue = [];
            this._batchTimer = null;
            this._requestCounter = 0;
            this._eventBus = sap.ui.getCore().getEventBus();
            
            Log.info("BatchService initialized", { service: "BatchService" });
        },

        /**
         * Destroy the batch service
         * @public
         * @since 1.0.0
         */
        destroy: function() {
            this._flushQueue();
            
            if (this._batchTimer) {
                clearTimeout(this._batchTimer);
            }
            
            Log.info("BatchService destroyed", { service: "BatchService" });
        },

        /* =========================================================== */
        /* Public API                                                  */
        /* =========================================================== */

        /**
         * Submit request for batch processing
         * @public
         * @param {object} request Request configuration
         * @param {string} request.method HTTP method
         * @param {string} request.url Request URL
         * @param {*} request.data Request data
         * @param {object} request.headers Request headers
         * @param {string} [request.groupId] Batch group ID
         * @param {number} [request.priority=5] Request priority (1-10)
         * @param {boolean} [request.changeSet=false] Whether to include in changeset
         * @returns {Promise} Request promise
         * @since 1.0.0
         */
        submit: function(request) {
            var that = this;
            
            return new Promise(function(resolve, reject) {
                var batchRequest = {
                    id: that._generateRequestId(),
                    method: request.method || "GET",
                    url: request.url,
                    data: request.data,
                    headers: request.headers || {},
                    groupId: request.groupId || "default",
                    priority: request.priority || 5,
                    changeSet: request.changeSet || false,
                    timestamp: Date.now(),
                    retryCount: 0,
                    resolve: resolve,
                    reject: reject
                };
                
                that._addToBatch(batchRequest);
                
                Log.debug("Request added to batch queue", {
                    requestId: batchRequest.id,
                    method: batchRequest.method,
                    url: batchRequest.url,
                    groupId: batchRequest.groupId
                });
            });
        },

        /**
         * Submit multiple requests as a group
         * @public
         * @param {Array<object>} requests Array of request configurations
         * @param {object} options Group options
         * @param {string} [options.groupId] Custom group ID
         * @param {boolean} [options.atomic=false] Whether all requests must succeed
         * @param {boolean} [options.immediate=false] Whether to send immediately
         * @returns {Promise} Group promise
         * @since 1.0.0
         */
        submitGroup: function(requests, options) {
            options = options || {};
            var groupId = options.groupId || this._generateGroupId();
            var that = this;
            
            var groupPromises = requests.map(function(request) {
                request.groupId = groupId;
                return that.submit(request);
            });
            
            if (options.atomic) {
                // All requests must succeed
                return Promise.all(groupPromises);
            } else {
                // Return settled results
                return Promise.allSettled(groupPromises);
            }
        },

        /**
         * Force immediate batch processing
         * @public
         * @param {string} [groupId] Specific group ID to flush
         * @returns {Promise} Flush promise
         * @since 1.0.0
         */
        flush: function(groupId) {
            if (groupId) {
                return this._flushGroup(groupId);
            } else {
                return this._flushQueue();
            }
        },

        /**
         * Get batch queue status
         * @public
         * @returns {object} Queue status
         * @since 1.0.0
         */
        getStatus: function() {
            var groupStats = this._batchQueue.reduce(function(stats, request) {
                if (!stats[request.groupId]) {
                    stats[request.groupId] = {
                        count: 0,
                        methods: {},
                        changeSets: 0,
                        oldestRequest: null
                    };
                }
                
                var group = stats[request.groupId];
                group.count++;
                group.methods[request.method] = (group.methods[request.method] || 0) + 1;
                
                if (request.changeSet) {
                    group.changeSets++;
                }
                
                if (!group.oldestRequest || request.timestamp < group.oldestRequest.timestamp) {
                    group.oldestRequest = request;
                }
                
                return stats;
            }, {});
            
            return {
                totalRequests: this._batchQueue.length,
                groups: Object.keys(groupStats).length,
                groupStats: groupStats,
                isTimerActive: !!this._batchTimer
            };
        },

        /**
         * Clear batch queue
         * @public
         * @param {string} [groupId] Specific group to clear
         * @returns {number} Number of requests cleared
         * @since 1.0.0
         */
        clearQueue: function(groupId) {
            var initialLength = this._batchQueue.length;
            
            if (groupId) {
                this._batchQueue = this._batchQueue.filter(function(request) {
                    if (request.groupId === groupId) {
                        request.reject(new Error("Request cancelled"));
                        return false;
                    }
                    return true;
                });
            } else {
                this._batchQueue.forEach(function(request) {
                    request.reject(new Error("Queue cleared"));
                });
                this._batchQueue = [];
            }
            
            var cleared = initialLength - this._batchQueue.length;
            
            Log.info("Batch queue cleared", {
                requestsCleared: cleared,
                groupId: groupId || "all"
            });
            
            return cleared;
        },

        /* =========================================================== */
        /* Private Methods                                             */
        /* =========================================================== */

        /**
         * Add request to batch queue
         * @private
         * @param {object} request Batch request
         * @since 1.0.0
         */
        _addToBatch: function(request) {
            this._batchQueue.push(request);
            
            // Sort by priority (higher priority first) and timestamp
            this._batchQueue.sort(function(a, b) {
                if (a.priority !== b.priority) {
                    return b.priority - a.priority;
                }
                return a.timestamp - b.timestamp;
            });
            
            this._scheduleFlush();
        },

        /**
         * Schedule batch flush
         * @private
         * @since 1.0.0
         */
        _scheduleFlush: function() {
            var that = this;
            
            if (this._batchTimer) {
                clearTimeout(this._batchTimer);
            }
            
            // Flush immediately if queue is full
            if (this._batchQueue.length >= this.BATCH_SIZE_LIMIT) {
                this._flushQueue();
                return;
            }
            
            // Schedule flush after timeout
            this._batchTimer = setTimeout(function() {
                that._flushQueue();
            }, this.BATCH_TIMEOUT);
        },

        /**
         * Flush batch queue
         * @private
         * @returns {Promise} Flush promise
         * @since 1.0.0
         */
        _flushQueue: function() {
            if (this._batchQueue.length === 0) {
                return Promise.resolve();
            }
            
            if (this._batchTimer) {
                clearTimeout(this._batchTimer);
                this._batchTimer = null;
            }
            
            // Group requests by groupId
            var groups = this._groupRequests();
            var flushPromises = [];
            
            Object.keys(groups).forEach(function(groupId) {
                flushPromises.push(this._processBatchGroup(groupId, groups[groupId]));
            }.bind(this));
            
            // Clear the queue
            this._batchQueue = [];
            
            return Promise.allSettled(flushPromises);
        },

        /**
         * Flush specific group
         * @private
         * @param {string} groupId Group ID
         * @returns {Promise} Flush promise
         * @since 1.0.0
         */
        _flushGroup: function(groupId) {
            var groupRequests = this._batchQueue.filter(function(request) {
                return request.groupId === groupId;
            });
            
            if (groupRequests.length === 0) {
                return Promise.resolve();
            }
            
            // Remove group requests from queue
            this._batchQueue = this._batchQueue.filter(function(request) {
                return request.groupId !== groupId;
            });
            
            return this._processBatchGroup(groupId, groupRequests);
        },

        /**
         * Group requests by groupId
         * @private
         * @returns {object} Grouped requests
         * @since 1.0.0
         */
        _groupRequests: function() {
            return this._batchQueue.reduce(function(groups, request) {
                if (!groups[request.groupId]) {
                    groups[request.groupId] = [];
                }
                groups[request.groupId].push(request);
                return groups;
            }, {});
        },

        /**
         * Process batch group
         * @private
         * @param {string} groupId Group ID
         * @param {Array<object>} requests Group requests
         * @returns {Promise} Processing promise
         * @since 1.0.0
         */
        _processBatchGroup: function(groupId, requests) {
            var that = this;
            
            Log.info("Processing batch group", {
                groupId: groupId,
                requestCount: requests.length
            });
            
            this._eventBus.publish("BatchService", "BatchStarted", {
                groupId: groupId,
                requestCount: requests.length
            });
            
            return this._sendBatch(groupId, requests)
                .then(function(results) {
                    that._handleBatchSuccess(groupId, requests, results);
                })
                .catch(function(error) {
                    that._handleBatchError(groupId, requests, error);
                });
        },

        /**
         * Send batch request
         * @private
         * @param {string} groupId Group ID
         * @param {Array<object>} requests Requests to send
         * @returns {Promise} Batch promise
         * @since 1.0.0
         */
        _sendBatch: function(groupId, requests) {
            var that = this;
            
            return new Promise(function(resolve, reject) {
                // Build batch request
                var batchData = that._buildBatchRequest(requests);
                
                jQuery.ajax({
                    url: "/odata/v1/$batch",
                    type: "POST",
                    data: batchData.content,
                    contentType: batchData.contentType,
                    processData: false,
                    timeout: 60000 // 60 seconds timeout
                })
                .done(function(data, status, xhr) {
                    try {
                        var results = that._parseBatchResponse(xhr.responseText, requests);
                        resolve(results);
                    } catch (error) {
                        reject(error);
                    }
                })
                .fail(function(xhr, status, error) {
                    reject(new Error("Batch request failed: " + error));
                });
            });
        },

        /**
         * Build OData batch request
         * @private
         * @param {Array<object>} requests Requests to batch
         * @returns {object} Batch request data
         * @since 1.0.0
         */
        _buildBatchRequest: function(requests) {
            var boundary = "batch_" + this._generateBoundary();
            var changeBoundary = "changeset_" + this._generateBoundary();
            var content = "";
            var hasChanges = false;
            
            // Separate GET requests from change requests
            var getRequests = requests.filter(function(r) { return r.method === "GET"; });
            var changeRequests = requests.filter(function(r) { return r.method !== "GET"; });
            
            // Add GET requests
            getRequests.forEach(function(request) {
                content += "--" + boundary + "\r\n";
                content += "Content-Type: application/http\r\n";
                content += "Content-Transfer-Encoding: binary\r\n";
                content += "\r\n";
                content += request.method + " " + request.url + " HTTP/1.1\r\n";
                
                Object.keys(request.headers).forEach(function(header) {
                    content += header + ": " + request.headers[header] + "\r\n";
                });
                
                content += "\r\n\r\n";
            });
            
            // Add change requests in changeset
            if (changeRequests.length > 0) {
                hasChanges = true;
                content += "--" + boundary + "\r\n";
                content += "Content-Type: multipart/mixed; boundary=" + changeBoundary + "\r\n";
                content += "\r\n";
                
                changeRequests.forEach(function(request) {
                    content += "--" + changeBoundary + "\r\n";
                    content += "Content-Type: application/http\r\n";
                    content += "Content-Transfer-Encoding: binary\r\n";
                    content += "\r\n";
                    content += request.method + " " + request.url + " HTTP/1.1\r\n";
                    
                    Object.keys(request.headers).forEach(function(header) {
                        content += header + ": " + request.headers[header] + "\r\n";
                    });
                    
                    if (request.data) {
                        var requestData = typeof request.data === "string" ? 
                            request.data : JSON.stringify(request.data);
                        content += "Content-Length: " + requestData.length + "\r\n";
                        content += "\r\n";
                        content += requestData;
                    }
                    
                    content += "\r\n";
                });
                
                content += "--" + changeBoundary + "--\r\n";
            }
            
            content += "--" + boundary + "--\r\n";
            
            return {
                content: content,
                contentType: "multipart/mixed; boundary=" + boundary
            };
        },

        /**
         * Parse OData batch response according to OData v4 specification
         * @private
         * @param {string} responseText Batch response text
         * @param {Array<object>} requests Original requests
         * @returns {Array<object>} Parsed results
         * @since 1.0.0
         */
        _parseBatchResponse: function(responseText, requests) {
            try {
                var results = [];
                var requestIndex = 0;
                
                // Parse the multipart response
                var parts = this._parseMultipartResponse(responseText);
                
                for (var i = 0; i < parts.length; i++) {
                    var part = parts[i];
                    
                    if (part.contentType && part.contentType.indexOf('multipart/mixed') !== -1) {
                        // This is a changeset - parse nested responses
                        var changesetResults = this._parseChangesetResponse(part.body, requests, requestIndex);
                        results = results.concat(changesetResults);
                        requestIndex += changesetResults.length;
                    } else if (part.body && part.body.indexOf('HTTP/1.1') !== -1) {
                        // This is a single response
                        var singleResult = this._parseSingleResponse(part.body, requests[requestIndex]);
                        if (singleResult) {
                            results.push(singleResult);
                            requestIndex++;
                        }
                    }
                }
                
                // Ensure we have results for all requests
                while (results.length < requests.length) {
                    results.push({
                        requestId: requests[results.length].id,
                        status: 500,
                        error: "No response received",
                        data: null,
                        headers: {}
                    });
                }
                
                Log.debug("Batch response parsed", {
                    totalParts: parts.length,
                    totalResults: results.length,
                    requestCount: requests.length
                });
                
                return results.slice(0, requests.length);
                
            } catch (error) {
                Log.error("Failed to parse batch response", error);
                
                // Return error results for all requests
                return requests.map(function(request) {
                    return {
                        requestId: request.id,
                        status: 500,
                        error: "Batch parsing failed: " + error.message,
                        data: null,
                        headers: {}
                    };
                });
            }
        },

        /**
         * Handle batch success
         * @private
         * @param {string} groupId Group ID
         * @param {Array<object>} requests Original requests
         * @param {Array<object>} results Batch results
         * @since 1.0.0
         */
        _handleBatchSuccess: function(groupId, requests, results) {
            var successful = 0;
            var failed = 0;
            
            results.forEach(function(result, index) {
                var request = requests[index];
                
                if (result.status >= 200 && result.status < 300) {
                    request.resolve(result.data);
                    successful++;
                } else {
                    request.reject(new Error("Request failed with status " + result.status));
                    failed++;
                }
            });
            
            this._eventBus.publish("BatchService", "BatchCompleted", {
                groupId: groupId,
                successful: successful,
                failed: failed
            });
            
            Log.info("Batch processing completed", {
                groupId: groupId,
                successful: successful,
                failed: failed
            });
        },

        /**
         * Handle batch error
         * @private
         * @param {string} groupId Group ID
         * @param {Array<object>} requests Original requests
         * @param {Error} error Batch error
         * @since 1.0.0
         */
        _handleBatchError: function(groupId, requests, error) {
            var that = this;
            
            // Retry logic
            var canRetry = requests.every(function(request) {
                return request.retryCount < that.MAX_RETRIES;
            });
            
            if (canRetry) {
                Log.warn("Batch failed, retrying", {
                    groupId: groupId,
                    error: error.message,
                    retryCount: requests[0].retryCount + 1
                });
                
                // Increment retry count and re-queue
                requests.forEach(function(request) {
                    request.retryCount++;
                });
                
                setTimeout(function() {
                    that._processBatchGroup(groupId, requests);
                }, that.RETRY_DELAY);
            } else {
                // Max retries reached, reject all requests
                requests.forEach(function(request) {
                    request.reject(error);
                });
                
                this._eventBus.publish("BatchService", "BatchFailed", {
                    groupId: groupId,
                    error: error.message,
                    requestCount: requests.length
                });
                
                Log.error("Batch processing failed after max retries", {
                    groupId: groupId,
                    error: error.message,
                    requestCount: requests.length
                });
            }
        },

        /**
         * Generate unique request ID
         * @private
         * @returns {string} Request ID
         * @since 1.0.0
         */
        _generateRequestId: function() {
            return "req-" + (++this._requestCounter) + "-" + Date.now();
        },

        /**
         * Generate unique group ID
         * @private
         * @returns {string} Group ID
         * @since 1.0.0
         */
        _generateGroupId: function() {
            return "group-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Generate boundary string
         * @private
         * @returns {string} Boundary string
         * @since 1.0.0
         */
        _generateBoundary: function() {
            return Date.now().toString(16) + Math.random().toString(16).substr(2);
        },

        /* =========================================================== */
        /* OData Batch Response Parsing                                */
        /* =========================================================== */

        /**
         * Parse multipart response according to RFC 2046
         * @private
         * @param {string} responseText Response text
         * @returns {Array<object>} Parsed parts
         * @since 1.0.0
         */
        _parseMultipartResponse: function(responseText) {
            var parts = [];
            
            // Extract boundary from content-type
            var boundaryMatch = responseText.match(/boundary=([^\s;]+)/);
            if (!boundaryMatch) {
                // Try to find boundary in response
                var firstBoundary = responseText.match(/--([a-zA-Z0-9_]+)/);
                if (firstBoundary) {
                    boundaryMatch = [null, firstBoundary[1]];
                }
            }
            
            if (!boundaryMatch) {
                throw new Error("No boundary found in multipart response");
            }
            
            var boundary = "--" + boundaryMatch[1];
            var sections = responseText.split(boundary);
            
            // Skip first (empty) and last (closing) sections
            for (var i = 1; i < sections.length - 1; i++) {
                var section = sections[i].trim();
                if (section) {
                    var part = this._parseResponsePart(section);
                    if (part) {
                        parts.push(part);
                    }
                }
            }
            
            return parts;
        },

        /**
         * Parse individual response part
         * @private
         * @param {string} partText Part text
         * @returns {object} Parsed part
         * @since 1.0.0
         */
        _parseResponsePart: function(partText) {
            var headerBodySplit = partText.indexOf('\r\n\r\n');
            if (headerBodySplit === -1) {
                headerBodySplit = partText.indexOf('\n\n');
            }
            
            if (headerBodySplit === -1) {
                return null;
            }
            
            var headerText = partText.substring(0, headerBodySplit);
            var body = partText.substring(headerBodySplit + (partText.charAt(headerBodySplit + 2) === '\r' ? 4 : 2));
            
            var headers = this._parseHeaders(headerText);
            
            return {
                headers: headers,
                contentType: headers['content-type'] || headers['Content-Type'],
                body: body
            };
        },

        /**
         * Parse changeset response
         * @private
         * @param {string} changesetBody Changeset body
         * @param {Array<object>} requests Original requests
         * @param {number} startIndex Starting request index
         * @returns {Array<object>} Changeset results
         * @since 1.0.0
         */
        _parseChangesetResponse: function(changesetBody, requests, startIndex) {
            var results = [];
            var changesetParts = this._parseMultipartResponse("Content-Type: multipart/mixed; boundary=changeset_123\r\n\r\n" + changesetBody);
            
            for (var i = 0; i < changesetParts.length; i++) {
                var requestIndex = startIndex + i;
                if (requestIndex < requests.length) {
                    var result = this._parseSingleResponse(changesetParts[i].body, requests[requestIndex]);
                    if (result) {
                        results.push(result);
                    }
                }
            }
            
            return results;
        },

        /**
         * Parse single HTTP response
         * @private
         * @param {string} responseText Response text
         * @param {object} originalRequest Original request
         * @returns {object} Parsed response
         * @since 1.0.0
         */
        _parseSingleResponse: function(responseText, originalRequest) {
            if (!responseText || !originalRequest) {
                return null;
            }
            
            try {
                // Parse HTTP status line
                var statusMatch = responseText.match(/HTTP\/1\.1\s+(\d+)\s*([^\r\n]*)/);
                var status = statusMatch ? parseInt(statusMatch[1], 10) : 200;
                var statusText = statusMatch ? statusMatch[2].trim() : "OK";
                
                // Split headers and body
                var headerBodySplit = responseText.indexOf('\r\n\r\n');
                if (headerBodySplit === -1) {
                    headerBodySplit = responseText.indexOf('\n\n');
                }
                
                var headers = {};
                var body = "";
                
                if (headerBodySplit !== -1) {
                    var headerText = responseText.substring(responseText.indexOf('\n') + 1, headerBodySplit);
                    body = responseText.substring(headerBodySplit + (responseText.charAt(headerBodySplit + 2) === '\r' ? 4 : 2));
                    headers = this._parseHeaders(headerText);
                }
                
                // Parse response data
                var data = null;
                var error = null;
                
                if (status >= 200 && status < 300) {
                    if (body.trim()) {
                        try {
                            data = JSON.parse(body);
                        } catch (e) {
                            // If not JSON, return as string
                            data = body;
                        }
                    } else {
                        data = { success: true };
                    }
                } else {
                    try {
                        var errorData = JSON.parse(body);
                        error = errorData.error || errorData.message || statusText;
                    } catch (e) {
                        error = body || statusText;
                    }
                }
                
                return {
                    requestId: originalRequest.id,
                    status: status,
                    statusText: statusText,
                    data: data,
                    error: error,
                    headers: headers
                };
                
            } catch (parseError) {
                Log.warn("Failed to parse single response", {
                    error: parseError.message,
                    requestId: originalRequest.id
                });
                
                return {
                    requestId: originalRequest.id,
                    status: 500,
                    error: "Response parsing failed: " + parseError.message,
                    data: null,
                    headers: {}
                };
            }
        },

        /**
         * Parse HTTP headers
         * @private
         * @param {string} headerText Header text
         * @returns {object} Parsed headers
         * @since 1.0.0
         */
        _parseHeaders: function(headerText) {
            var headers = {};
            var lines = headerText.split(/\r?\n/);
            
            for (var i = 0; i < lines.length; i++) {
                var line = lines[i].trim();
                if (line) {
                    var colonIndex = line.indexOf(':');
                    if (colonIndex !== -1) {
                        var name = line.substring(0, colonIndex).trim();
                        var value = line.substring(colonIndex + 1).trim();
                        headers[name] = value;
                    }
                }
            }
            
            return headers;
        }
    };

    return BatchService;
});