/**
 * Performance Tracking Module for A2A Platform
 * Tracks and analyzes performance metrics across services
 */

const { EventEmitter } = require('events');

/**
 * Performance metric types
 */
const MetricType = {
  COUNTER: 'counter',
  GAUGE: 'gauge',
  HISTOGRAM: 'histogram',
  SUMMARY: 'summary'
};

/**
 * Performance categories
 */
const PerformanceCategory = {
  DATABASE: 'database',
  API: 'api',
  SERVICE: 'service',
  AGENT: 'agent',
  WORKFLOW: 'workflow',
  BLOCKCHAIN: 'blockchain',
  CACHE: 'cache',
  QUEUE: 'queue'
};

/**
 * Performance Tracker class
 */
class PerformanceTracker extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      enableMetrics: config.enableMetrics ?? true,
      enableTracing: config.enableTracing ?? true,
      sampleRate: config.sampleRate || 1.0,
      flushInterval: config.flushInterval || 60000, // 1 minute
      retentionPeriod: config.retentionPeriod || 3600000, // 1 hour
      thresholds: config.thresholds || this.getDefaultThresholds()
    };
    
    this.metrics = new Map();
    this.traces = new Map();
    this.activeOperations = new Map();
    
    if (this.config.enableMetrics) {
      this.startMetricsCollection();
    }
  }

  /**
   * Get default performance thresholds
   */
  getDefaultThresholds() {
    return {
      [PerformanceCategory.DATABASE]: {
        warning: 100,  // ms
        critical: 500  // ms
      },
      [PerformanceCategory.API]: {
        warning: 500,  // ms
        critical: 2000 // ms
      },
      [PerformanceCategory.SERVICE]: {
        warning: 200,  // ms
        critical: 1000 // ms
      },
      [PerformanceCategory.AGENT]: {
        warning: 1000, // ms
        critical: 5000 // ms
      },
      [PerformanceCategory.WORKFLOW]: {
        warning: 5000,  // ms
        critical: 30000 // ms
      },
      [PerformanceCategory.BLOCKCHAIN]: {
        warning: 2000,  // ms
        critical: 10000 // ms
      }
    };
  }

  /**
   * Start performance tracking for an operation
   */
  startOperation(operationId, metadata = {}) {
    const operation = {
      id: operationId,
      startTime: Date.now(),
      startHrTime: process.hrtime.bigint(),
      metadata,
      checkpoints: []
    };
    
    this.activeOperations.set(operationId, operation);
    
    return {
      checkpoint: function(name, data = {}) {
        this.addCheckpoint(operationId, name, data);
      }.bind(this),
      end: function(resultMetadata = {}) {
        return this.endOperation(operationId, resultMetadata);
      }.bind(this)
    };
  }

  /**
   * Add checkpoint to an operation
   */
  addCheckpoint(operationId, name, data = {}) {
    const operation = this.activeOperations.get(operationId);
    if (!operation) return;
    
    const checkpoint = {
      name,
      timestamp: Date.now(),
      duration: Date.now() - operation.startTime,
      data
    };
    
    operation.checkpoints.push(checkpoint);
  }

  /**
   * End performance tracking for an operation
   */
  endOperation(operationId, resultMetadata = {}) {
    const operation = this.activeOperations.get(operationId);
    if (!operation) return null;
    
    const endTime = Date.now();
    const endHrTime = process.hrtime.bigint();
    const duration = Number(endHrTime - operation.startHrTime) / 1000000; // Convert to ms
    
    const performanceData = {
      ...operation,
      endTime,
      duration: Math.round(duration * 100) / 100, // Round to 2 decimal places
      resultMetadata,
      category: operation.metadata.category || PerformanceCategory.SERVICE,
      status: this.evaluatePerformance(operation.metadata.category, duration)
    };
    
    // Clean up
    this.activeOperations.delete(operationId);
    
    // Record metrics
    this.recordMetric(performanceData);
    
    // Store trace if enabled
    if (this.config.enableTracing && this.shouldSample()) {
      this.storeTrace(performanceData);
    }
    
    // Emit event for monitoring
    this.emit('operation-completed', performanceData);
    
    return performanceData;
  }

  /**
   * Evaluate performance based on thresholds
   */
  evaluatePerformance(category, duration) {
    const thresholds = this.config.thresholds[category] || 
                      this.config.thresholds[PerformanceCategory.SERVICE];
    
    if (duration >= thresholds.critical) {
      return 'critical';
    } else if (duration >= thresholds.warning) {
      return 'warning';
    }
    return 'normal';
  }

  /**
   * Record performance metric
   */
  recordMetric(performanceData) {
    const key = `${performanceData.category}:${performanceData.metadata.operation || 'unknown'}`;
    
    if (!this.metrics.has(key)) {
      this.metrics.set(key, {
        count: 0,
        totalDuration: 0,
        minDuration: Infinity,
        maxDuration: 0,
        avgDuration: 0,
        p50: 0,
        p95: 0,
        p99: 0,
        durations: [],
        errors: 0,
        lastUpdated: Date.now()
      });
    }
    
    const metric = this.metrics.get(key);
    
    // Update basic metrics
    metric.count++;
    metric.totalDuration += performanceData.duration;
    metric.minDuration = Math.min(metric.minDuration, performanceData.duration);
    metric.maxDuration = Math.max(metric.maxDuration, performanceData.duration);
    metric.avgDuration = metric.totalDuration / metric.count;
    metric.lastUpdated = Date.now();
    
    // Track errors
    if (performanceData.resultMetadata.error) {
      metric.errors++;
    }
    
    // Store duration for percentile calculation
    metric.durations.push(performanceData.duration);
    
    // Limit stored durations to prevent memory issues
    if (metric.durations.length > 1000) {
      metric.durations = metric.durations.slice(-1000);
    }
    
    // Calculate percentiles
    this.calculatePercentiles(metric);
  }

  /**
   * Calculate percentiles for a metric
   */
  calculatePercentiles(metric) {
    const sorted = [...metric.durations].sort((a, b) => { return a - b; });
    const len = sorted.length;
    
    metric.p50 = sorted[Math.floor(len * 0.5)];
    metric.p95 = sorted[Math.floor(len * 0.95)];
    metric.p99 = sorted[Math.floor(len * 0.99)];
  }

  /**
   * Store trace data
   */
  storeTrace(performanceData) {
    const traceKey = `${performanceData.category}:${Date.now()}`;
    this.traces.set(traceKey, performanceData);
    
    // Clean up old traces
    this.cleanupOldTraces();
  }

  /**
   * Check if operation should be sampled
   */
  shouldSample() {
    return Math.random() < this.config.sampleRate;
  }

  /**
   * Get current metrics
   */
  getMetrics(category = null) {
    const result = {};
    
    for (const [key, metric] of this.metrics.entries()) {
      if (!category || key.startsWith(`${category}:`)) {
        result[key] = {
          count: metric.count,
          avgDuration: Math.round(metric.avgDuration * 100) / 100,
          minDuration: Math.round(metric.minDuration * 100) / 100,
          maxDuration: Math.round(metric.maxDuration * 100) / 100,
          p50: Math.round(metric.p50 * 100) / 100,
          p95: Math.round(metric.p95 * 100) / 100,
          p99: Math.round(metric.p99 * 100) / 100,
          errorRate: metric.count > 0 ? (metric.errors / metric.count) : 0,
          lastUpdated: new Date(metric.lastUpdated).toISOString()
        };
      }
    }
    
    return result;
  }

  /**
   * Get performance report
   */
  getPerformanceReport() {
    const metrics = this.getMetrics();
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        totalOperations: 0,
        averageDuration: 0,
        errorRate: 0,
        slowOperations: 0,
        criticalOperations: 0
      },
      categories: {},
      topSlowest: [],
      recommendations: []
    };
    
    // Calculate summary
    let totalDuration = 0;
    let totalErrors = 0;
    
    for (const [key, metric] of Object.entries(metrics)) {
      const [category] = key.split(':');
      
      report.summary.totalOperations += metric.count;
      totalDuration += metric.avgDuration * metric.count;
      totalErrors += metric.errorRate * metric.count;
      
      // Count slow operations
      const threshold = this.config.thresholds[category];
      if (threshold) {
        if (metric.p95 > threshold.warning) report.summary.slowOperations++;
        if (metric.p95 > threshold.critical) report.summary.criticalOperations++;
      }
      
      // Group by category
      if (!report.categories[category]) {
        report.categories[category] = [];
      }
      report.categories[category].push({ operation: key, ...metric });
    }
    
    // Calculate averages
    if (report.summary.totalOperations > 0) {
      report.summary.averageDuration = Math.round(
        (totalDuration / report.summary.totalOperations) * 100
      ) / 100;
      report.summary.errorRate = totalErrors / report.summary.totalOperations;
    }
    
    // Find top slowest operations
    const allOperations = Object.entries(metrics)
      .map(([key, metric]) => { return { operation: key, ...metric }; })
      .sort((a, b) => { return b.p95 - a.p95; })
      .slice(0, 10);
    
    report.topSlowest = allOperations;
    
    // Generate recommendations
    report.recommendations = this.generateRecommendations(report);
    
    return report;
  }

  /**
   * Generate performance recommendations
   */
  generateRecommendations(report) {
    const recommendations = [];
    
    // Check overall performance
    if (report.summary.averageDuration > 1000) {
      recommendations.push({
        severity: 'high',
        category: 'general',
        message: 'Overall average response time is high. Consider optimizing slow operations.'
      });
    }
    
    // Check error rate
    if (report.summary.errorRate > 0.05) {
      recommendations.push({
        severity: 'high',
        category: 'reliability',
        message: `Error rate is ${(report.summary.errorRate * 100).toFixed(2)}%. Investigate and fix errors.`
      });
    }
    
    // Check for slow database queries
    const dbOps = report.categories[PerformanceCategory.DATABASE] || [];
    const slowDbOps = dbOps.filter((op) => { return op.p95 > 500; });
    if (slowDbOps.length > 0) {
      recommendations.push({
        severity: 'medium',
        category: 'database',
        message: `${slowDbOps.length} database operations are slow. Consider adding indexes or optimizing queries.`
      });
    }
    
    // Check for slow API calls
    const apiOps = report.categories[PerformanceCategory.API] || [];
    const slowApiOps = apiOps.filter((op) => { return op.p95 > 2000; });
    if (slowApiOps.length > 0) {
      recommendations.push({
        severity: 'medium',
        category: 'api',
        message: `${slowApiOps.length} API operations are slow. Consider implementing caching or pagination.`
      });
    }
    
    return recommendations;
  }

  /**
   * Start metrics collection
   */
  startMetricsCollection() {
    // Periodic flush
    this.flushInterval = setInterval(() => {
      this.emit('metrics-flush', this.getMetrics());
      this.cleanupOldMetrics();
    }, this.config.flushInterval);
    
    // Handle process termination
    process.on('SIGINT', () => { this.shutdown(); });
    process.on('SIGTERM', () => { this.shutdown(); });
  }

  /**
   * Clean up old traces
   */
  cleanupOldTraces() {
    const cutoff = Date.now() - this.config.retentionPeriod;
    
    for (const [key, trace] of this.traces.entries()) {
      if (trace.endTime < cutoff) {
        this.traces.delete(key);
      }
    }
  }

  /**
   * Clean up old metrics
   */
  cleanupOldMetrics() {
    const cutoff = Date.now() - this.config.retentionPeriod;
    
    for (const [key, metric] of this.metrics.entries()) {
      if (metric.lastUpdated < cutoff) {
        this.metrics.delete(key);
      }
    }
  }

  /**
   * Shutdown tracker
   */
  shutdown() {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    
    // Final metrics flush
    this.emit('metrics-flush', this.getMetrics());
    this.emit('shutdown');
  }
}

/**
 * Global performance tracker instance
 */
let globalTracker = null;

/**
 * Get or create global performance tracker
 */
function getGlobalTracker(config) {
  if (!globalTracker) {
    globalTracker = new PerformanceTracker(config);
  }
  return globalTracker;
}

module.exports = {
  PerformanceTracker,
  MetricType,
  PerformanceCategory,
  getGlobalTracker
};