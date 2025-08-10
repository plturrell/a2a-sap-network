/**
 * @fileoverview Message Transformation Service for A2A Network
 * @description Advanced message transformation engine with format conversion,
 * protocol adaptation, content enrichment, and validation pipelines
 * @module messageTransformation
 * @since 2.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');
const Ajv = require('ajv');
const addFormats = require('ajv-formats');
const xmljs = require('xml-js');
const yaml = require('js-yaml');
const csv = require('csv-parser');
const { Transform } = require('stream');

/**
 * Message Transformation Service
 * Handles format conversion, protocol adaptation, and content transformation
 */
class MessageTransformationService extends cds.Service {
    
    async init() {
        this.log = cds.log('message-transformation');
        
        // Initialize JSON schema validator
        this.ajv = new Ajv({ allErrors: true });
        addFormats(this.ajv);
        
        // Database entities
        const { 
            TransformationRules, 
            TransformationHistory, 
            FormatSchemas,
            TransformationMetrics 
        } = cds.entities('a2a.transformation');
        this.entities = { TransformationRules, TransformationHistory, FormatSchemas, TransformationMetrics };
        
        // Transformation configuration
        this.config = {
            maxMessageSize: 10 * 1024 * 1024,    // 10MB
            supportedFormats: [
                'json', 'xml', 'yaml', 'csv', 'text', 
                'a2a', 'fiori', 'sap-idoc', 'odata'
            ],
            transformationTimeout: 30000,         // 30 seconds
            batchSize: 100,                      // Batch processing size
            cacheTransformationsTTL: 3600,       // Cache for 1 hour
            enableValidation: true,
            enableEnrichment: true
        };
        
        // Built-in transformers
        this.transformers = new Map();
        await this._initializeBuiltInTransformers();
        
        // Load custom transformation rules
        await this._loadTransformationRules();
        
        // Register handlers
        this._registerHandlers();
        
        this.log.info('Message Transformation Service initialized');
        return super.init();
    }

    _registerHandlers() {
        // Core transformation operations
        this.on('transformMessage', this._transformMessage.bind(this));
        this.on('transformBatch', this._transformBatch.bind(this));
        this.on('validateMessage', this._validateMessage.bind(this));
        this.on('enrichMessage', this._enrichMessage.bind(this));
        
        // Format conversion
        this.on('convertFormat', this._convertFormat.bind(this));
        this.on('detectFormat', this._detectFormat.bind(this));
        this.on('getSupportedFormats', this._getSupportedFormats.bind(this));
        
        // Rule management
        this.on('createTransformationRule', this._createTransformationRule.bind(this));
        this.on('updateTransformationRule', this._updateTransformationRule.bind(this));
        this.on('deleteTransformationRule', this._deleteTransformationRule.bind(this));
        this.on('getTransformationRules', this._getTransformationRules.bind(this));
        
        // Schema management
        this.on('registerSchema', this._registerSchema.bind(this));
        this.on('validateAgainstSchema', this._validateAgainstSchema.bind(this));
        
        // Analytics
        this.on('getTransformationMetrics', this._getTransformationMetrics.bind(this));
        this.on('getTransformationHistory', this._getTransformationHistory.bind(this));
    }

    /**
     * Transform a message using specified transformation rules
     */
    async _transformMessage(req) {
        const {
            messageId = uuidv4(),
            content,
            sourceFormat,
            targetFormat,
            transformationRule,
            validationSchema,
            enrichmentOptions = {},
            metadata = {}
        } = req.data;

        const startTime = Date.now();
        const transformationId = uuidv4();

        try {
            // Validate input
            if (!content) {
                throw new Error('Message content is required');
            }

            if (Buffer.byteLength(JSON.stringify(content)) > this.config.maxMessageSize) {
                throw new Error('Message exceeds maximum size limit');
            }

            // Auto-detect source format if not provided
            const detectedSourceFormat = sourceFormat || await this._detectMessageFormat(content);
            
            this.log.debug(`Transforming message ${messageId}`, {
                sourceFormat: detectedSourceFormat,
                targetFormat,
                rule: transformationRule
            });

            // Parse source content
            let parsedContent = await this._parseContent(content, detectedSourceFormat);

            // Apply transformation rule
            let transformedContent = parsedContent;
            if (transformationRule) {
                transformedContent = await this._applyTransformationRule(
                    parsedContent, 
                    transformationRule
                );
            }

            // Enrich content if requested
            if (this.config.enableEnrichment && Object.keys(enrichmentOptions).length > 0) {
                transformedContent = await this._enrichContent(
                    transformedContent, 
                    enrichmentOptions
                );
            }

            // Validate against schema if provided
            if (this.config.enableValidation && validationSchema) {
                const validationResult = await this._validateContent(
                    transformedContent, 
                    validationSchema
                );
                if (!validationResult.valid) {
                    throw new Error(`Validation failed: ${validationResult.errors.join(', ')}`);
                }
            }

            // Convert to target format
            const finalContent = await this._formatContent(transformedContent, targetFormat);

            const duration = Date.now() - startTime;

            // Record transformation history
            await this._recordTransformationHistory({
                transformationId,
                messageId,
                sourceFormat: detectedSourceFormat,
                targetFormat,
                transformationRule,
                duration,
                status: 'success',
                metadata
            });

            // Update metrics
            await this._updateTransformationMetrics(
                detectedSourceFormat, 
                targetFormat, 
                'success', 
                duration
            );

            this.log.info(`Message transformation completed: ${messageId}`, {
                duration,
                sourceFormat: detectedSourceFormat,
                targetFormat
            });

            return {
                success: true,
                transformationId,
                messageId,
                transformedContent: finalContent,
                sourceFormat: detectedSourceFormat,
                targetFormat,
                duration,
                metadata: {
                    ...metadata,
                    transformedAt: new Date().toISOString(),
                    transformationVersion: '2.0.0'
                }
            };

        } catch (error) {
            const duration = Date.now() - startTime;
            
            // Record failed transformation
            await this._recordTransformationHistory({
                transformationId,
                messageId,
                sourceFormat: sourceFormat || 'unknown',
                targetFormat,
                transformationRule,
                duration,
                status: 'failed',
                error: error.message,
                metadata
            });

            // Update error metrics
            await this._updateTransformationMetrics(
                sourceFormat || 'unknown', 
                targetFormat, 
                'failed', 
                duration
            );

            this.log.error(`Message transformation failed: ${messageId}`, error);
            throw new Error(`Transformation failed: ${error.message}`);
        }
    }

    /**
     * Transform multiple messages in batch
     */
    async _transformBatch(req) {
        const { messages, batchOptions = {} } = req.data;
        const batchId = uuidv4();
        const startTime = Date.now();

        try {
            if (!Array.isArray(messages) || messages.length === 0) {
                throw new Error('Messages array is required and must not be empty');
            }

            const batchSize = batchOptions.batchSize || this.config.batchSize;
            const results = [];
            const errors = [];

            // Process in batches to manage memory
            for (let i = 0; i < messages.length; i += batchSize) {
                const batch = messages.slice(i, i + batchSize);
                
                const batchPromises = batch.map(async (message, index) => {
                    try {
                        const result = await this._transformMessage({ data: message });
                        return { index: i + index, success: true, result };
                    } catch (error) {
                        return { 
                            index: i + index, 
                            success: false, 
                            error: error.message,
                            messageId: message.messageId
                        };
                    }
                });

                const batchResults = await Promise.allSettled(batchPromises);
                
                batchResults.forEach(result => {
                    if (result.status === 'fulfilled') {
                        if (result.value.success) {
                            results.push(result.value.result);
                        } else {
                            errors.push(result.value);
                        }
                    } else {
                        errors.push({
                            error: result.reason.message,
                            index: results.length + errors.length
                        });
                    }
                });
            }

            const duration = Date.now() - startTime;
            const successCount = results.length;
            const errorCount = errors.length;
            const totalCount = successCount + errorCount;

            this.log.info(`Batch transformation completed: ${batchId}`, {
                total: totalCount,
                successful: successCount,
                failed: errorCount,
                duration
            });

            return {
                success: true,
                batchId,
                summary: {
                    total: totalCount,
                    successful: successCount,
                    failed: errorCount,
                    successRate: successCount / totalCount,
                    duration
                },
                results,
                errors
            };

        } catch (error) {
            const duration = Date.now() - startTime;
            this.log.error(`Batch transformation failed: ${batchId}`, error);
            
            return {
                success: false,
                batchId,
                error: error.message,
                duration
            };
        }
    }

    /**
     * Auto-detect message format
     */
    async _detectFormat(req) {
        const { content } = req.data;
        
        try {
            const format = await this._detectMessageFormat(content);
            
            return {
                success: true,
                detectedFormat: format,
                confidence: this._calculateFormatConfidence(content, format)
            };
            
        } catch (error) {
            this.log.error('Format detection failed:', error);
            throw new Error(`Format detection failed: ${error.message}`);
        }
    }

    /**
     * Convert between formats without transformation rules
     */
    async _convertFormat(req) {
        const { content, sourceFormat, targetFormat } = req.data;
        
        try {
            // Parse source content
            const parsedContent = await this._parseContent(content, sourceFormat);
            
            // Convert to target format
            const convertedContent = await this._formatContent(parsedContent, targetFormat);
            
            return {
                success: true,
                convertedContent,
                sourceFormat,
                targetFormat
            };
            
        } catch (error) {
            this.log.error('Format conversion failed:', error);
            throw new Error(`Format conversion failed: ${error.message}`);
        }
    }

    // Built-in transformers initialization
    async _initializeBuiltInTransformers() {
        // JSON transformers
        this.transformers.set('json-to-xml', this._jsonToXml.bind(this));
        this.transformers.set('xml-to-json', this._xmlToJson.bind(this));
        this.transformers.set('json-to-yaml', this._jsonToYaml.bind(this));
        this.transformers.set('yaml-to-json', this._yamlToJson.bind(this));
        this.transformers.set('json-to-csv', this._jsonToCsv.bind(this));
        this.transformers.set('csv-to-json', this._csvToJson.bind(this));
        
        // A2A specific transformers
        this.transformers.set('a2a-to-fiori', this._a2aToFiori.bind(this));
        this.transformers.set('fiori-to-a2a', this._fioriToA2a.bind(this));
        this.transformers.set('a2a-to-odata', this._a2aToOdata.bind(this));
        this.transformers.set('odata-to-a2a', this._odataToA2a.bind(this));
        this.transformers.set('sap-idoc-to-a2a', this._sapIdocToA2a.bind(this));
        
        this.log.debug(`Initialized ${this.transformers.size} built-in transformers`);
    }

    // Format detection methods
    async _detectMessageFormat(content) {
        if (typeof content === 'object') {
            // Check for A2A message structure
            if (content.messageId && content.parts && Array.isArray(content.parts)) {
                return 'a2a';
            }
            // Check for Fiori structure
            if (content.d && content.d.results) {
                return 'fiori';
            }
            // Check for OData structure
            if (content.value || (content['odata.metadata'] || content['@odata.context'])) {
                return 'odata';
            }
            return 'json';
        }

        if (typeof content === 'string') {
            const trimmed = content.trim();
            
            // XML detection
            if (trimmed.startsWith('<?xml') || (trimmed.startsWith('<') && trimmed.endsWith('>'))) {
                return 'xml';
            }
            
            // JSON detection
            if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || 
                (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
                try {
                    JSON.parse(trimmed);
                    return 'json';
                } catch (e) {
                    // Not valid JSON
                }
            }
            
            // YAML detection
            if (trimmed.includes('---') || trimmed.includes(': ')) {
                try {
                    yaml.load(trimmed);
                    return 'yaml';
                } catch (e) {
                    // Not valid YAML
                }
            }
            
            // CSV detection
            if (trimmed.includes(',') && trimmed.split('\n').length > 1) {
                return 'csv';
            }
            
            // SAP IDoc detection
            if (trimmed.includes('BEGIN:IDOC') || trimmed.includes('IDOC_TYPE')) {
                return 'sap-idoc';
            }
        }

        return 'text';
    }

    _calculateFormatConfidence(content, format) {
        // Simple confidence calculation based on format characteristics
        let confidence = 0.5;
        
        switch (format) {
            case 'json':
                if (typeof content === 'object') confidence = 0.95;
                else if (typeof content === 'string') {
                    try {
                        JSON.parse(content);
                        confidence = 0.9;
                    } catch (e) {
                        confidence = 0.1;
                    }
                }
                break;
                
            case 'xml':
                if (typeof content === 'string' && content.includes('<?xml')) confidence = 0.95;
                else if (content.startsWith('<') && content.endsWith('>')) confidence = 0.8;
                break;
                
            case 'a2a':
                if (content.messageId && content.parts) confidence = 0.95;
                break;
                
            default:
                confidence = 0.7;
        }
        
        return Math.min(confidence, 1.0);
    }

    // Content parsing methods
    async _parseContent(content, format) {
        switch (format) {
            case 'json':
                return typeof content === 'string' ? JSON.parse(content) : content;
                
            case 'xml':
                return xmljs.xml2js(content, { compact: true, ignoreText: false });
                
            case 'yaml':
                return yaml.load(content);
                
            case 'csv':
                return await this._parseCsv(content);
                
            case 'text':
                return { content: content.toString(), format: 'text' };
                
            case 'a2a':
            case 'fiori':
            case 'odata':
            case 'sap-idoc':
                return typeof content === 'string' ? JSON.parse(content) : content;
                
            default:
                throw new Error(`Unsupported source format: ${format}`);
        }
    }

    // Content formatting methods
    async _formatContent(content, format) {
        switch (format) {
            case 'json':
                return typeof content === 'object' ? content : JSON.parse(content);
                
            case 'xml':
                return xmljs.js2xml(content, { compact: true, ignoreComment: true, spaces: 2 });
                
            case 'yaml':
                return yaml.dump(content);
                
            case 'csv':
                return this._formatAsCsv(content);
                
            case 'text':
                return typeof content === 'string' ? content : JSON.stringify(content);
                
            case 'a2a':
                return this._formatAsA2A(content);
                
            case 'fiori':
                return this._formatAsFiori(content);
                
            case 'odata':
                return this._formatAsOData(content);
                
            default:
                throw new Error(`Unsupported target format: ${format}`);
        }
    }

    // A2A specific formatters
    _formatAsA2A(content) {
        // Convert content to A2A message format
        return {
            messageId: content.messageId || uuidv4(),
            role: content.role || 'agent',
            parts: content.parts || [{
                partType: 'text',
                text: typeof content === 'string' ? content : JSON.stringify(content)
            }],
            taskId: content.taskId || null,
            contextId: content.contextId || null,
            timestamp: new Date().toISOString(),
            metadata: {
                ...content.metadata,
                transformedToA2A: true,
                protocolVersion: '0.2.9'
            }
        };
    }

    _formatAsFiori(content) {
        // Convert to SAP Fiori format
        return {
            d: {
                results: Array.isArray(content) ? content : [content],
                __count: Array.isArray(content) ? content.length : 1
            }
        };
    }

    _formatAsOData(content) {
        // Convert to OData format
        return {
            "@odata.context": "$metadata#Collection(Edm.String)",
            "@odata.count": Array.isArray(content) ? content.length : 1,
            value: Array.isArray(content) ? content : [content]
        };
    }

    // Built-in transformation methods
    _jsonToXml(content) {
        return xmljs.js2xml({ root: content }, { compact: true, spaces: 2 });
    }

    _xmlToJson(content) {
        const result = xmljs.xml2js(content, { compact: true });
        return result.root || result;
    }

    _jsonToYaml(content) {
        return yaml.dump(content);
    }

    _yamlToJson(content) {
        return yaml.load(content);
    }

    _jsonToCsv(content) {
        if (!Array.isArray(content)) {
            content = [content];
        }
        
        if (content.length === 0) return '';
        
        const headers = Object.keys(content[0]);
        const csvHeaders = headers.join(',');
        const csvRows = content.map(row => 
            headers.map(header => `"${String(row[header] || '').replace(/"/g, '""')}"`).join(',')
        );
        
        return [csvHeaders, ...csvRows].join('\n');
    }

    async _csvToJson(content) {
        return new Promise((resolve, reject) => {
            const results = [];
            const parser = csv();
            
            parser.on('data', (data) => results.push(data));
            parser.on('end', () => resolve(results));
            parser.on('error', reject);
            
            // Convert string to stream
            const { Readable } = require('stream');
            Readable.from(content).pipe(parser);
        });
    }

    // A2A protocol transformations
    _a2aToFiori(a2aMessage) {
        const fioriData = {
            MessageId: a2aMessage.messageId,
            Role: a2aMessage.role,
            TaskId: a2aMessage.taskId,
            ContextId: a2aMessage.contextId,
            Timestamp: a2aMessage.timestamp,
            Content: a2aMessage.parts?.map(part => ({
                Type: part.partType,
                Text: part.text,
                FunctionName: part.functionName,
                FunctionArgs: part.functionArgs
            })) || []
        };
        
        return this._formatAsFiori(fioriData);
    }

    _fioriToA2a(fioriMessage) {
        const data = fioriMessage.d?.results?.[0] || fioriMessage;
        
        return this._formatAsA2A({
            messageId: data.MessageId,
            role: data.Role || 'agent',
            taskId: data.TaskId,
            contextId: data.ContextId,
            timestamp: data.Timestamp,
            parts: data.Content?.map(content => ({
                partType: content.Type || 'text',
                text: content.Text,
                functionName: content.FunctionName,
                functionArgs: content.FunctionArgs
            })) || []
        });
    }

    _a2aToOdata(a2aMessage) {
        return this._formatAsOData({
            MessageId: a2aMessage.messageId,
            Role: a2aMessage.role,
            TaskId: a2aMessage.taskId,
            ContextId: a2aMessage.contextId,
            Timestamp: a2aMessage.timestamp,
            PartsCount: a2aMessage.parts?.length || 0,
            Content: JSON.stringify(a2aMessage.parts || [])
        });
    }

    _odataToA2a(odataMessage) {
        const data = odataMessage.value?.[0] || odataMessage;
        
        return this._formatAsA2A({
            messageId: data.MessageId,
            role: data.Role || 'agent',
            taskId: data.TaskId,
            contextId: data.ContextId,
            timestamp: data.Timestamp,
            parts: data.Content ? JSON.parse(data.Content) : []
        });
    }

    _sapIdocToA2a(idocContent) {
        // Simplified SAP IDoc to A2A conversion
        // In production, this would be much more sophisticated
        
        const segments = idocContent.split('\n').filter(line => line.trim());
        const messageData = {};
        
        segments.forEach(segment => {
            if (segment.startsWith('E1')) {
                // Parse segment data
                const parts = segment.split(/\s+/);
                messageData[parts[0]] = parts.slice(1).join(' ');
            }
        });
        
        return this._formatAsA2A({
            messageId: messageData.DOCNUM || uuidv4(),
            role: 'system',
            parts: [{
                partType: 'idoc_data',
                text: JSON.stringify(messageData)
            }],
            metadata: {
                source: 'sap_idoc',
                idocType: messageData.IDOCTYP
            }
        });
    }

    // Utility methods
    async _applyTransformationRule(content, ruleName) {
        // Apply custom transformation rules
        // This would load and execute transformation rules from the database
        const transformer = this.transformers.get(ruleName);
        
        if (!transformer) {
            throw new Error(`Transformation rule not found: ${ruleName}`);
        }
        
        return await transformer(content);
    }

    async _enrichContent(content, enrichmentOptions) {
        // Content enrichment based on options
        let enriched = { ...content };
        
        if (enrichmentOptions.addTimestamp) {
            enriched.timestamp = new Date().toISOString();
        }
        
        if (enrichmentOptions.addCorrelationId) {
            enriched.correlationId = uuidv4();
        }
        
        if (enrichmentOptions.addMetadata) {
            enriched.metadata = {
                ...enriched.metadata,
                enrichedAt: new Date().toISOString(),
                enrichmentVersion: '2.0.0'
            };
        }
        
        return enriched;
    }

    async _validateContent(content, schemaName) {
        // Validate content against registered schema
        const { FormatSchemas } = this.entities;
        const schemas = await SELECT.from(FormatSchemas).where({ name: schemaName });
        
        if (schemas.length === 0) {
            throw new Error(`Schema not found: ${schemaName}`);
        }
        
        const schema = JSON.parse(schemas[0].schema);
        const validate = this.ajv.compile(schema);
        const valid = validate(content);
        
        return {
            valid,
            errors: validate.errors?.map(err => `${err.instancePath} ${err.message}`) || []
        };
    }

    async _loadTransformationRules() {
        // Load custom transformation rules from database
        const { TransformationRules } = this.entities;
        const rules = await SELECT.from(TransformationRules).where({ active: true });
        
        for (const rule of rules) {
            try {
                // In production, you'd have a secure way to load and execute transformation logic
                // For now, we'll just log that the rule is loaded
                this.log.debug(`Loaded transformation rule: ${rule.name}`);
            } catch (error) {
                this.log.error(`Failed to load transformation rule ${rule.name}:`, error);
            }
        }
    }

    async _recordTransformationHistory(historyData) {
        const { TransformationHistory } = this.entities;
        
        await INSERT.into(TransformationHistory).entries({
            ...historyData,
            createdAt: new Date()
        });
    }

    async _updateTransformationMetrics(sourceFormat, targetFormat, status, duration) {
        const { TransformationMetrics } = this.entities;
        const date = new Date().toISOString().split('T')[0];
        
        await UPSERT.into(TransformationMetrics).entries({
            date,
            sourceFormat,
            targetFormat,
            status,
            count: 1,
            totalDuration: duration,
            avgDuration: duration,
            timestamp: new Date()
        });
    }

    async _getSupportedFormats() {
        return {
            success: true,
            formats: this.config.supportedFormats,
            transformers: Array.from(this.transformers.keys())
        };
    }
}

module.exports = MessageTransformationService;