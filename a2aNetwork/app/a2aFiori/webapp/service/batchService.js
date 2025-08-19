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

    const BatchService = {

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
        init() {
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
        destroy() {
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
        submit(request) {
            const that = this;

            return new Promise(function(resolve, reject) {
                const batchRequest = {
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
                    resolve,
                    reject
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
        submitGroup(requests, options) {
            options = options || {};
            const groupId = options.groupId || this._generateGroupId();
            const that = this;

            const groupPromises = requests.map(function(request) {
                request.groupId = groupId;
                return that.submit(request);
            });

            if (options.atomic) {
                // All requests must succeed
                return Promise.all(groupPromises);
            }
            // Return settled results
            return Promise.allSettled(groupPromises);

        },

        /**
         * Force immediate batch processing
         * @public
         * @param {string} [groupId] Specific group ID to flush
         * @returns {Promise} Flush promise
         * @since 1.0.0
         */
        flush(groupId) {
            if (groupId) {
                return this._flushGroup(groupId);
            }
            return this._flushQueue();

        },

        /**
         * Get batch queue status
         * @public
         * @returns {object} Queue status
         * @since 1.0.0
         */
        getStatus() {
            const groupStats = this._batchQueue.reduce(function(stats, request) {
                if (!stats[request.groupId]) {
                    stats[request.groupId] = {
                        count: 0,
                        methods: {},
                        changeSets: 0,
                        oldestRequest: null
                    };
                }

                const group = stats[request.groupId];
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
                groupStats,
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
        clearQueue(groupId) {
            const initialLength = this._batchQueue.length;

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

            const cleared = initialLength - this._batchQueue.length;

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
        _addToBatch(request) {
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
        _scheduleFlush() {
            const that = this;

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
        _flushQueue() {
            if (this._batchQueue.length === 0) {
                return Promise.resolve();
            }

            if (this._batchTimer) {
                clearTimeout(this._batchTimer);
                this._batchTimer = null;
            }

            // Group requests by groupId
            const groups = this._groupRequests();
            const flushPromises = [];

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
        _flushGroup(groupId) {
            const groupRequests = this._batchQueue.filter(function(request) {
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
        _groupRequests() {
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
        _processBatchGroup(groupId, requests) {
            const that = this;

            Log.info("Processing batch group", {
                groupId,
                requestCount: requests.length
            });

            this._eventBus.publish("BatchService", "BatchStarted", {
                groupId,
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
        _sendBatch(groupId, requests) {
            const that = this;

            return new Promise(function(resolve, reject) {
                // Build batch request
                const batchData = that._buildBatchRequest(requests);

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
                            const results = that._parseBatchResponse(xhr.responseText, requests);
                            resolve(results);
                        } catch (error) {
                            reject(error);
                        }
                    })
                    .fail(function(xhr, status, error) {
                        reject(new Error(`Batch request failed: ${ error}`));
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
        _buildBatchRequest(requests) {
            const boundary = `batch_${ this._generateBoundary()}`;
            const changeBoundary = `changeset_${ this._generateBoundary()}`;
            let content = "";
            let _hasChanges = false;

            // Separate GET requests from change requests
            const getRequests = requests.filter(function(r) {
                return r.method === "GET";
            });
            const changeRequests = requests.filter(function(r) {
                return r.method !== "GET";
            });

            // Add GET requests
            getRequests.forEach(function(request) {
                content += `--${ boundary }\r\n`;
                content += "Content-Type: application/http\r\n";
                content += "Content-Transfer-Encoding: binary\r\n";
                content += "\r\n";
                content += `${request.method } ${ request.url } HTTP/1.1\r\n`;

                Object.keys(request.headers).forEach(function(header) {
                    content += `${header }: ${ request.headers[header] }\r\n`;
                });

                content += "\r\n\r\n";
            });

            // Add change requests in changeset
            if (changeRequests.length > 0) {
                _hasChanges = true;
                content += `--${ boundary }\r\n`;
                content += `Content-Type: multipart/mixed; boundary=${ changeBoundary }\r\n`;
                content += "\r\n";

                changeRequests.forEach(function(request) {
                    content += `--${ changeBoundary }\r\n`;
                    content += "Content-Type: application/http\r\n";
                    content += "Content-Transfer-Encoding: binary\r\n";
                    content += "\r\n";
                    content += `${request.method } ${ request.url } HTTP/1.1\r\n`;

                    Object.keys(request.headers).forEach(function(header) {
                        content += `${header }: ${ request.headers[header] }\r\n`;
                    });

                    if (request.data) {
                        const requestData = typeof request.data === "string" ?
                            request.data : JSON.stringify(request.data);
                        content += `Content-Length: ${ requestData.length }\r\n`;
                        content += "\r\n";
                        content += requestData;
                    }

                    content += "\r\n";
                });

                content += `--${ changeBoundary }--\r\n`;
            }

            content += `--${ boundary }--\r\n`;

            return {
                content,
                contentType: `multipart/mixed; boundary=${ boundary}`
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
        _parseBatchResponse(responseText, requests) {
            try {
                let results = [];
                let requestIndex = 0;

                // Parse the multipart response
                const parts = this._parseMultipartResponse(responseText);

                for (let i = 0; i < parts.length; i++) {
                    const part = parts[i];

                    if (part.contentType && part.contentType.indexOf("multipart/mixed") !== -1) {
                        // This is a changeset - parse nested responses
                        const changesetResults = this._parseChangesetResponse(part.body, requests, requestIndex);
                        results = results.concat(changesetResults);
                        requestIndex += changesetResults.length;
                    } else if (part.body && part.body.indexOf("HTTP/1.1") !== -1) {
                        // This is a single response
                        const singleResult = this._parseSingleResponse(part.body, requests[requestIndex]);
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
                        error: `Batch parsing failed: ${ error.message}`,
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
        _handleBatchSuccess(groupId, requests, results) {
            let successful = 0;
            let failed = 0;

            results.forEach(function(result, index) {
                const request = requests[index];

                if (result.status >= 200 && result.status < 300) {
                    request.resolve(result.data);
                    successful++;
                } else {
                    request.reject(new Error(`Request failed with status ${ result.status}`));
                    failed++;
                }
            });

            this._eventBus.publish("BatchService", "BatchCompleted", {
                groupId,
                successful,
                failed
            });

            Log.info("Batch processing completed", {
                groupId,
                successful,
                failed
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
        _handleBatchError(groupId, requests, error) {
            const that = this;

            // Retry logic
            const canRetry = requests.every(function(request) {
                return request.retryCount < that.MAX_RETRIES;
            });

            if (canRetry) {
                Log.warn("Batch failed, retrying", {
                    groupId,
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
                    groupId,
                    error: error.message,
                    requestCount: requests.length
                });

                Log.error("Batch processing failed after max retries", {
                    groupId,
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
        _generateRequestId() {
            return `req-${ ++this._requestCounter }-${ Date.now()}`;
        },

        /**
         * Generate unique group ID
         * @private
         * @returns {string} Group ID
         * @since 1.0.0
         */
        _generateGroupId() {
            return `group-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Generate boundary string
         * @private
         * @returns {string} Boundary string
         * @since 1.0.0
         */
        _generateBoundary() {
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
        _parseMultipartResponse(responseText) {
            const parts = [];

            // Extract boundary from content-type
            let boundaryMatch = responseText.match(/boundary=([^\s;]+)/);
            if (!boundaryMatch) {
                // Try to find boundary in response
                const firstBoundary = responseText.match(/--([a-zA-Z0-9_]+)/);
                if (firstBoundary) {
                    boundaryMatch = [null, firstBoundary[1]];
                }
            }

            if (!boundaryMatch) {
                throw new Error("No boundary found in multipart response");
            }

            const boundary = `--${ boundaryMatch[1]}`;
            const sections = responseText.split(boundary);

            // Skip first (empty) and last (closing) sections
            for (let i = 1; i < sections.length - 1; i++) {
                const section = sections[i].trim();
                if (section) {
                    const part = this._parseResponsePart(section);
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
        _parseResponsePart(partText) {
            let headerBodySplit = partText.indexOf("\r\n\r\n");
            if (headerBodySplit === -1) {
                headerBodySplit = partText.indexOf("\n\n");
            }

            if (headerBodySplit === -1) {
                return null;
            }

            const headerText = partText.substring(0, headerBodySplit);
            const body = partText.substring(headerBodySplit + (partText.charAt(headerBodySplit + 2) === "\r" ? 4 : 2));

            const headers = this._parseHeaders(headerText);

            return {
                headers,
                contentType: headers["content-type"] || headers["Content-Type"],
                body
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
        _parseChangesetResponse(changesetBody, requests, startIndex) {
            const results = [];
            const changesetParts = this._parseMultipartResponse(`Content-Type: multipart/mixed; boundary=changeset_123\r\n\r\n${ changesetBody}`);

            for (let i = 0; i < changesetParts.length; i++) {
                const requestIndex = startIndex + i;
                if (requestIndex < requests.length) {
                    const result = this._parseSingleResponse(changesetParts[i].body, requests[requestIndex]);
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
        _parseSingleResponse(responseText, originalRequest) {
            if (!responseText || !originalRequest) {
                return null;
            }

            try {
                // Parse HTTP status line
                const statusMatch = responseText.match(/HTTP\/1\.1\s+(\d+)\s*([^\r\n]*)/);
                const status = statusMatch ? parseInt(statusMatch[1], 10) : 200;
                const statusText = statusMatch ? statusMatch[2].trim() : "OK";

                // Split headers and body
                let headerBodySplit = responseText.indexOf("\r\n\r\n");
                if (headerBodySplit === -1) {
                    headerBodySplit = responseText.indexOf("\n\n");
                }

                let headers = {};
                let body = "";

                if (headerBodySplit !== -1) {
                    const headerText = responseText.substring(responseText.indexOf("\n") + 1, headerBodySplit);
                    body = responseText.substring(headerBodySplit + (responseText.charAt(headerBodySplit + 2) === "\r" ? 4 : 2));
                    headers = this._parseHeaders(headerText);
                }

                // Parse response data
                let data = null;
                let error = null;

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
                        const errorData = JSON.parse(body);
                        error = errorData.error || errorData.message || statusText;
                    } catch (e) {
                        error = body || statusText;
                    }
                }

                return {
                    requestId: originalRequest.id,
                    status,
                    statusText,
                    data,
                    error,
                    headers
                };

            } catch (parseError) {
                Log.warn("Failed to parse single response", {
                    error: parseError.message,
                    requestId: originalRequest.id
                });

                return {
                    requestId: originalRequest.id,
                    status: 500,
                    error: `Response parsing failed: ${ parseError.message}`,
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
        _parseHeaders(headerText) {
            const headers = {};
            const lines = headerText.split(/\r?\n/);

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                if (line) {
                    const colonIndex = line.indexOf(":");
                    if (colonIndex !== -1) {
                        const name = line.substring(0, colonIndex).trim();
                        const value = line.substring(colonIndex + 1).trim();
                        headers[name] = value;
                    }
                }
            }

            return headers;
        }
    };

    return BatchService;
});