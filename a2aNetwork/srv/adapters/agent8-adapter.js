/**
 * Agent 8 Adapter - Data Management Agent
 * Converts between REST API and OData formats for data management, storage, cache, versioning, and backup operations
 */

const { BlockchainClient } = require('../core/blockchain-client') = const { BlockchainClient } = require('../core/blockchain-client');
const { v4: uuidv4 } = require('uuid');

class Agent8Adapter {
    constructor() {
        this.baseUrl = process.env.AGENT8_BASE_URL || 'http://localhost:8007';
        this.apiVersion = 'v1';
        this.timeout = 30000;
    }

    // Data Tasks
    async getDataTasks(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'DataTask');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createDataTask(data) {
        try {
            const restData = this._convertODataTaskToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async updateDataTask(id, data) {
        try {
            const restData = this._convertODataTaskToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${id}`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTTaskToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async deleteDataTask(id) {
        try {
            await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${id}`, {
                timeout: this.timeout
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Data Task Operations
    async executeTask(taskId, parameters) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/execute`, {
                parameters
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                taskType: response.data.task_type,
                executionId: response.data.execution_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async pauseTask(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/pause`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async resumeTask(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/resume`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async cancelTask(taskId, reason) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/cancel`, {
                reason
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateTask(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/validate`, {}, {
                timeout: this.timeout
            });
            
            return {
                status: response.data.status?.toUpperCase(),
                message: response.data.message,
                errors: response.data.errors || [],
                warnings: response.data.warnings || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getTaskProgress(taskId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/progress`, {
                timeout: this.timeout
            });
            
            return {
                percentage: response.data.percentage,
                processedRecords: response.data.processed_records,
                totalRecords: response.data.total_records,
                estimatedCompletion: response.data.estimated_completion,
                currentPhase: response.data.current_phase
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getTaskLogs(taskId, logLevel, startTime, endTime) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/${taskId}/logs`, {
                params: {
                    log_level: logLevel,
                    start_time: startTime,
                    end_time: endTime
                },
                timeout: this.timeout
            });
            
            return response.data.logs.map(log => ({
                timestamp: log.timestamp,
                level: log.level?.toUpperCase(),
                message: log.message,
                context: log.context
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Storage Backends
    async getStorageBackends(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-backends`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'StorageBackend');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createStorageBackend(data) {
        try {
            const restData = this._convertODataBackendToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-backends`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTBackendToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async performBackendHealthCheck(backendId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-backends/${backendId}/health-check`, {}, {
                timeout: this.timeout
            });
            
            return {
                status: response.data.status?.toUpperCase(),
                responseTime: response.data.response_time,
                errorRate: response.data.error_rate,
                availableCapacity: response.data.available_capacity,
                errorDetails: response.data.error_details,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeStorage(backendId, optimizationType) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-backends/${backendId}/optimize`, {
                optimization_type: optimizationType
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                optimizationResults: response.data.optimization_results,
                performanceImprovement: response.data.performance_improvement
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Cache Management
    async getCacheConfigurations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/configurations`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'CacheConfiguration');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createCacheConfiguration(data) {
        try {
            const restData = this._convertODataCacheToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/configurations`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTCacheToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async clearCache(cacheId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/${cacheId}/clear`, {}, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                details: {
                    clearedEntries: response.data.cleared_entries,
                    freedMemory: response.data.freed_memory
                }
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async warmupCache(cacheId, dataKeys) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/${cacheId}/warmup`, {
                data_keys: dataKeys
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                details: {
                    warmedKeys: response.data.warmed_keys,
                    cacheHitRate: response.data.cache_hit_rate
                }
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getCacheStats(cacheId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/${cacheId}/stats`, {
                timeout: this.timeout
            });
            
            return {
                hitRate: response.data.hit_rate,
                missRate: response.data.miss_rate,
                totalOperations: response.data.total_operations,
                memoryUsage: response.data.memory_usage,
                entryCount: response.data.entry_count,
                averageResponseTime: response.data.average_response_time
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Data Versioning
    async getDataVersions(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-versions`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'DataVersion');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createDataVersion(data) {
        try {
            const restData = this._convertODataVersionToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-versions`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTVersionToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async restoreVersion(versionId, targetDataset, backupCurrent) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-versions/${versionId}/restore`, {
                target_dataset: targetDataset,
                backup_current: backupCurrent
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                restoredRecords: response.data.restored_records,
                backupId: response.data.backup_id
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async compareVersions(versionId, compareVersionId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-versions/${versionId}/compare/${compareVersionId}`, {
                timeout: this.timeout
            });
            
            return {
                summary: response.data.summary,
                differences: response.data.differences.map(diff => ({
                    type: diff.type?.toUpperCase(),
                    field: diff.field,
                    oldValue: diff.old_value,
                    newValue: diff.new_value,
                    recordId: diff.record_id
                })),
                statistics: response.data.statistics
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Data Backups
    async getDataBackups(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-backups`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'DataBackup');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async createDataBackup(data) {
        try {
            const restData = this._convertODataBackupToREST(data);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-backups`, restData, {
                timeout: this.timeout
            });
            
            return this._convertRESTBackupToOData(response.data);
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async restoreBackup(backupId, targetLocation, verifyIntegrity) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-backups/${backupId}/restore`, {
                target_location: targetLocation,
                verify_integrity: verifyIntegrity
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                restoredFiles: response.data.restored_files,
                restoredSize: response.data.restored_size
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async verifyBackup(backupId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-backups/${backupId}/verify`, {}, {
                timeout: this.timeout
            });
            
            return {
                status: response.data.status?.toUpperCase(),
                integrityScore: response.data.integrity_score,
                verificationResults: response.data.verification_results,
                issues: response.data.issues || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Function implementations
    async getStorageMetrics() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/metrics/storage`, {
                timeout: this.timeout
            });
            
            return {
                totalCapacity: response.data.total_capacity,
                usedCapacity: response.data.used_capacity,
                availableCapacity: response.data.available_capacity,
                utilizationPercent: response.data.utilization_percent,
                backends: response.data.backends.map(backend => ({
                    name: backend.name,
                    type: backend.type?.toUpperCase(),
                    capacity: backend.capacity,
                    used: backend.used,
                    health: backend.health?.toUpperCase()
                }))
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getCachePerformance(timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/performance/cache`, {
                params: { time_range: timeRange },
                timeout: this.timeout
            });
            
            return {
                overallHitRate: response.data.overall_hit_rate,
                averageResponseTime: response.data.average_response_time,
                operationsPerSecond: response.data.operations_per_second,
                memoryUsage: response.data.memory_usage,
                trends: response.data.trends.map(trend => ({
                    timestamp: trend.timestamp,
                    hitRate: trend.hit_rate,
                    responseTime: trend.response_time
                }))
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDataTasksStatus(status, timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/data-tasks/status`, {
                params: { 
                    status: status?.toLowerCase(),
                    time_range: timeRange 
                },
                timeout: this.timeout
            });
            
            return response.data.tasks.map(task => ({
                id: task.id,
                name: task.name,
                status: task.status?.toUpperCase(),
                progress: task.progress,
                startTime: task.start_time,
                estimatedCompletion: task.estimated_completion
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getStorageUtilization(backendId) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-backends/${backendId}/utilization`, {
                timeout: this.timeout
            });
            
            return {
                backendId: response.data.backend_id,
                totalCapacity: response.data.total_capacity,
                usedCapacity: response.data.used_capacity,
                utilizationPercent: response.data.utilization_percent,
                growthRate: response.data.growth_rate,
                projectedFull: response.data.projected_full
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getBackupSchedules() {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/backup-schedules`, {
                timeout: this.timeout
            });
            
            return response.data.schedules.map(schedule => ({
                id: schedule.id,
                name: schedule.name,
                frequency: schedule.frequency?.toUpperCase(),
                nextRun: schedule.next_run,
                lastRun: schedule.last_run,
                status: schedule.status?.toUpperCase()
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getVersionHistory(datasetId, limit) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/datasets/${datasetId}/versions`, {
                params: { limit },
                timeout: this.timeout
            });
            
            return response.data.versions.map(version => ({
                id: version.id,
                versionNumber: version.version_number,
                createdAt: version.created_at,
                author: version.author,
                description: version.description,
                size: version.size,
                checksum: version.checksum
            }));
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async analyzePerformanceTrends(metricType, timeRange) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/analytics/performance-trends`, {
                params: {
                    metric_type: metricType,
                    time_range: timeRange
                },
                timeout: this.timeout
            });
            
            return {
                metricType: response.data.metric_type,
                trend: response.data.trend?.toUpperCase(),
                dataPoints: response.data.data_points,
                analysis: response.data.analysis,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async optimizeDataLayout(datasetId, strategy) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/datasets/${datasetId}/optimize-layout`, {
                strategy: strategy?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                success: response.data.success,
                message: response.data.message,
                optimizationResults: response.data.optimization_results,
                performanceGain: response.data.performance_gain
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async validateDataIntegrity(datasetId, validationType) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/datasets/${datasetId}/validate-integrity`, {
                validation_type: validationType?.toLowerCase()
            }, {
                timeout: this.timeout
            });
            
            return {
                status: response.data.status?.toUpperCase(),
                integrityScore: response.data.integrity_score,
                issues: response.data.issues || [],
                recommendations: response.data.recommendations || []
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async estimateStorageNeeds(dataSize, retentionPeriod, compressionRatio) {
        try {
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/estimate-storage`, {
                data_size: dataSize,
                retention_period: retentionPeriod,
                compression_ratio: compressionRatio
            }, {
                timeout: this.timeout
            });
            
            return {
                estimatedSize: response.data.estimated_size,
                recommendedBackends: response.data.recommended_backends,
                costEstimate: response.data.cost_estimate,
                recommendations: response.data.recommendations
            };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Additional entity methods
    async getStorageUtilizations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/storage-utilizations`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'StorageUtilization');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getCacheOperations(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/cache/operations`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'CacheOperation');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getDataPerformanceMetrics(query = {}) {
        try {
            const params = this._convertODataToREST(query);
            const response = await blockchainClient.sendMessage(`${this.baseUrl}/api/${this.apiVersion}/performance/metrics`, {
                params,
                timeout: this.timeout
            });
            
            return this._convertRESTToOData(response.data, 'DataPerformanceMetric');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // Utility methods
    _convertODataToREST(query) {
        const params = {};
        
        if (query.$top) params.limit = query.$top;
        if (query.$skip) params.offset = query.$skip;
        if (query.$orderby) params.sort = query.$orderby.replace(/ desc/gi, '-').replace(/ asc/gi, '');
        if (query.$filter) params.filter = this._parseODataFilter(query.$filter);
        if (query.$select) params.fields = query.$select;
        
        return params;
    }

    _parseODataFilter(filter) {
        // Convert OData filter to REST query parameters
        return filter
            .replace(/ eq /g, '=')
            .replace(/ ne /g, '!=')
            .replace(/ gt /g, '>')
            .replace(/ ge /g, '>=')
            .replace(/ lt /g, '<')
            .replace(/ le /g, '<=')
            .replace(/ and /g, '&')
            .replace(/ or /g, '|');
    }

    _convertRESTToOData(data, entityType) {
        if (Array.isArray(data)) {
            return data.map(item => this._convertRESTItemToOData(item, entityType));
        }
        return this._convertRESTItemToOData(data, entityType);
    }

    _convertRESTItemToOData(item, entityType) {
        switch (entityType) {
            case 'DataTask':
                return this._convertRESTTaskToOData(item);
            case 'StorageBackend':
                return this._convertRESTBackendToOData(item);
            case 'CacheConfiguration':
                return this._convertRESTCacheToOData(item);
            case 'DataVersion':
                return this._convertRESTVersionToOData(item);
            case 'DataBackup':
                return this._convertRESTBackupToOData(item);
            case 'StorageUtilization':
                return this._convertRESTUtilizationToOData(item);
            case 'CacheOperation':
                return this._convertRESTCacheOpToOData(item);
            case 'DataPerformanceMetric':
                return this._convertRESTMetricToOData(item);
            default:
                return item;
        }
    }

    // Entity conversion methods
    _convertODataTaskToREST(data) {
        return {
            task_name: data.taskName,
            description: data.description,
            task_type: data.taskType?.toLowerCase(),
            status: data.status?.toLowerCase(),
            priority: data.priority?.toLowerCase(),
            data_source: data.dataSource,
            target_destination: data.targetDestination,
            configuration: data.configuration,
            metadata: data.metadata
        };
    }

    _convertRESTTaskToOData(item) {
        return {
            ID: item.id,
            taskName: item.task_name,
            description: item.description,
            taskType: item.task_type?.toUpperCase(),
            status: item.status?.toUpperCase(),
            priority: item.priority?.toUpperCase(),
            dataSource: item.data_source,
            targetDestination: item.target_destination,
            dataSize: item.data_size || 0,
            processedSize: item.processed_size || 0,
            progress: item.progress || 0,
            startTime: item.start_time,
            endTime: item.end_time,
            estimatedDuration: item.estimated_duration,
            actualDuration: item.actual_duration,
            errorMessage: item.error_message,
            configuration: item.configuration,
            metadata: item.metadata,
            createdAt: item.created_at,
            createdBy: item.created_by,
            modifiedAt: item.modified_at,
            modifiedBy: item.modified_by
        };
    }

    _convertODataBackendToREST(data) {
        return {
            backend_name: data.backendName,
            backend_type: data.backendType?.toLowerCase(),
            connection_string: data.connectionString,
            configuration: data.configuration,
            credentials: data.credentials
        };
    }

    _convertRESTBackendToOData(item) {
        return {
            ID: item.id,
            backendName: item.backend_name,
            backendType: item.backend_type?.toUpperCase(),
            connectionString: item.connection_string,
            status: item.status?.toUpperCase(),
            healthStatus: item.health_status?.toUpperCase(),
            totalCapacity: item.total_capacity || 0,
            usedCapacity: item.used_capacity || 0,
            availableCapacity: item.available_capacity || 0,
            compressionEnabled: item.compression_enabled !== false,
            encryptionEnabled: item.encryption_enabled !== false,
            lastHealthCheck: item.last_health_check,
            responseTime: item.response_time,
            errorRate: item.error_rate || 0,
            configuration: item.configuration,
            credentials: item.credentials,
            createdAt: item.created_at,
            createdBy: item.created_by,
            modifiedAt: item.modified_at,
            modifiedBy: item.modified_by
        };
    }

    _convertODataCacheToREST(data) {
        return {
            cache_name: data.cacheName,
            cache_type: data.cacheType?.toLowerCase(),
            max_size: data.maxSize,
            ttl: data.ttl,
            eviction_policy: data.evictionPolicy?.toLowerCase(),
            configuration: data.configuration
        };
    }

    _convertRESTCacheToOData(item) {
        return {
            ID: item.id,
            cacheName: item.cache_name,
            cacheType: item.cache_type?.toUpperCase(),
            status: item.status?.toUpperCase(),
            maxSize: item.max_size || 0,
            currentSize: item.current_size || 0,
            ttl: item.ttl || 0,
            evictionPolicy: item.eviction_policy?.toUpperCase(),
            hitRate: item.hit_rate || 0,
            missRate: item.miss_rate || 0,
            operationsPerSecond: item.operations_per_second || 0,
            lastAccessed: item.last_accessed,
            configuration: item.configuration,
            createdAt: item.created_at,
            createdBy: item.created_by,
            modifiedAt: item.modified_at,
            modifiedBy: item.modified_by
        };
    }

    _convertODataVersionToREST(data) {
        return {
            version_name: data.versionName,
            dataset_id: data.datasetId,
            description: data.description,
            version_data: data.versionData,
            metadata: data.metadata
        };
    }

    _convertRESTVersionToOData(item) {
        return {
            ID: item.id,
            versionName: item.version_name,
            datasetId: item.dataset_id,
            versionNumber: item.version_number,
            description: item.description,
            dataSize: item.data_size || 0,
            checksum: item.checksum,
            compressionRatio: item.compression_ratio || 0,
            createdAt: item.created_at,
            createdBy: item.created_by,
            isActive: item.is_active !== false,
            parentVersion: item.parent_version,
            versionData: item.version_data,
            metadata: item.metadata
        };
    }

    _convertODataBackupToREST(data) {
        return {
            backup_name: data.backupName,
            dataset_id: data.datasetId,
            backup_type: data.backupType?.toLowerCase(),
            storage_location: data.storageLocation,
            compression_type: data.compressionType?.toLowerCase(),
            metadata: data.metadata
        };
    }

    _convertRESTBackupToOData(item) {
        return {
            ID: item.id,
            backupName: item.backup_name,
            datasetId: item.dataset_id,
            backupType: item.backup_type?.toUpperCase(),
            status: item.status?.toUpperCase(),
            originalSize: item.original_size || 0,
            compressedSize: item.compressed_size || 0,
            compressionRatio: item.compression_ratio || 0,
            compressionType: item.compression_type?.toUpperCase(),
            storageLocation: item.storage_location,
            verificationStatus: item.verification_status?.toUpperCase(),
            lastVerification: item.last_verification,
            retentionUntil: item.retention_until,
            isEncrypted: item.is_encrypted !== false,
            checksum: item.checksum,
            metadata: item.metadata,
            createdAt: item.created_at,
            createdBy: item.created_by
        };
    }

    _convertRESTUtilizationToOData(item) {
        return {
            ID: item.id,
            backend_ID: item.backend_id,
            utilizationPercent: item.utilization_percent || 0,
            freeSpace: item.free_space || 0,
            usedSpace: item.used_space || 0,
            totalSpace: item.total_space || 0,
            timestamp: item.timestamp,
            alertThreshold: item.alert_threshold || 80,
            isAlertTriggered: item.is_alert_triggered !== false
        };
    }

    _convertRESTCacheOpToOData(item) {
        return {
            ID: item.id,
            cache_ID: item.cache_id,
            operationType: item.operation_type?.toUpperCase(),
            result: item.result?.toUpperCase(),
            executionTime: item.execution_time,
            responseTime: item.response_time || 0,
            dataSize: item.data_size || 0,
            details: item.details
        };
    }

    _convertRESTMetricToOData(item) {
        return {
            ID: item.id,
            metricName: item.metric_name,
            metricType: item.metric_type?.toUpperCase(),
            value: item.value || 0,
            unit: item.unit,
            timestamp: item.timestamp,
            source: item.source,
            context: item.context
        };
    }

    _handleError(error) {
        const errorMessage = error.response?.data?.message || error.message || 'Unknown error occurred';
        const errorCode = error.response?.status || 500;
        
        const customError = new Error(`Agent 8 Data Management service error: ${errorMessage}`);
        customError.statusCode = errorCode;
        customError.originalError = error;
        
        return customError;
    }
}

module.exports = Agent8Adapter;