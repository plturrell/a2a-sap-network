/**
 * SAP Cloud Logging Exporter for OpenTelemetry
 * Exports traces to SAP Cloud Logging service
 */

const { ExportResultCode } = require('@opentelemetry/core');
const axios = require('axios');

class SAPCloudLoggingExporter {
    constructor(config) {
        this.serviceName = config.serviceName;
        this.endpoint = config.endpoint;
        this.apiKey = config.apiKey;
        this.spaceGuid = config.spaceGuid;
        this.orgGuid = config.orgGuid;
        this.batchSize = config.batchSize || 100;
        this.timeout = config.timeout || 10000;
        
        // Configure axios instance
        this.client = axios.create({
            baseURL: this.endpoint,
            timeout: this.timeout,
            headers: {
                'Authorization': `Bearer ${this.apiKey}`,
                'Content-Type': 'application/json',
                'X-SAP-Space-Guid': this.spaceGuid,
                'X-SAP-Org-Guid': this.orgGuid
            }
        });
    }

    /**
     * Export spans to SAP Cloud Logging
     */
    async export(spans, resultCallback) {
        try {
            // Convert spans to SAP Cloud Logging format
            const logs = spans.map(span => this._convertSpanToLog(span));
            
            // Batch logs if necessary
            const batches = this._createBatches(logs, this.batchSize);
            
            // Send batches
            const promises = batches.map(batch => this._sendBatch(batch));
            await Promise.all(promises);
            
            resultCallback({ code: ExportResultCode.SUCCESS });
        } catch (error) {
            console.error('Failed to export spans to SAP Cloud Logging:', error);
            resultCallback({ 
                code: ExportResultCode.FAILED,
                error: error
            });
        }
    }

    /**
     * Convert OpenTelemetry span to SAP Cloud Logging format
     */
    _convertSpanToLog(span) {
        const spanContext = span.spanContext();
        const attributes = span.attributes || {};
        
        return {
            timestamp: span.startTime[0] * 1000 + span.startTime[1] / 1000000,
            level: this._getLogLevel(span.status),
            component: this.serviceName,
            msg: span.name,
            trace_id: spanContext.traceId,
            span_id: spanContext.spanId,
            parent_span_id: span.parentSpanId,
            duration_ms: this._calculateDuration(span.startTime, span.endTime),
            custom_fields: {
                ...attributes,
                'span.kind': span.kind,
                'span.status.code': span.status?.code,
                'span.status.message': span.status?.message,
                'span.events': span.events?.map(event => ({
                    name: event.name,
                    timestamp: event.time,
                    attributes: event.attributes
                })),
                'service.name': this.serviceName,
                'telemetry.sdk.name': '@opentelemetry/node',
                'telemetry.sdk.language': 'nodejs',
                'telemetry.sdk.version': '1.0.0'
            },
            categories: ['tracing', 'distributed-trace', 'opentelemetry'],
            correlation_id: attributes['sap.correlation_id'] || spanContext.traceId,
            tenant_id: attributes['sap.tenant_id'] || 'default'
        };
    }

    /**
     * Determine log level based on span status
     */
    _getLogLevel(status) {
        if (!status) return 'INFO';
        
        switch (status.code) {
            case 0: // UNSET
                return 'INFO';
            case 1: // OK
                return 'INFO';
            case 2: // ERROR
                return 'ERROR';
            default:
                return 'INFO';
        }
    }

    /**
     * Calculate span duration in milliseconds
     */
    _calculateDuration(startTime, endTime) {
        if (!endTime) return 0;
        
        const startMs = startTime[0] * 1000 + startTime[1] / 1000000;
        const endMs = endTime[0] * 1000 + endTime[1] / 1000000;
        return endMs - startMs;
    }

    /**
     * Create batches of logs
     */
    _createBatches(logs, batchSize) {
        const batches = [];
        for (let i = 0; i < logs.length; i += batchSize) {
            batches.push(logs.slice(i, i + batchSize));
        }
        return batches;
    }

    /**
     * Send a batch of logs to SAP Cloud Logging
     */
    async _sendBatch(batch) {
        try {
            const response = await this.client.post('/v1/logs', {
                logs: batch
            });
            
            if (response.status !== 200 && response.status !== 202) {
                throw new Error(`Unexpected status code: ${response.status}`);
            }
            
            return response.data;
        } catch (error) {
            // Retry logic for transient failures
            if (this._isRetryable(error)) {
                await this._sleep(1000);
                return this._sendBatch(batch);
            }
            throw error;
        }
    }

    /**
     * Check if error is retryable
     */
    _isRetryable(error) {
        if (error.response) {
            const status = error.response.status;
            return status === 429 || status >= 500;
        }
        return error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT';
    }

    /**
     * Sleep helper
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Shutdown the exporter
     */
    async shutdown() {
        // Clean up any resources
        return Promise.resolve();
    }
}

module.exports = { SAPCloudLoggingExporter };