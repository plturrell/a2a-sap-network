/**
 * @fileoverview CAP-specific Fact Transformer for advanced Glean analysis
 * @module capFactTransformer
 * @since 1.0.0
 * 
 * Transforms advanced CAP/CDS parsing results into comprehensive Glean facts
 * with support for all CAP patterns and enterprise features
 */

const crypto = require('crypto');
const path = require('path');

class CAPFactTransformer {
    constructor() {
        this.factIdCounter = 0;
        this.symbolTable = new Map();
        this.crossReferences = new Map();
        this.capPatterns = new Map();
    }

    /**
     * Transform advanced CDS parsing results to comprehensive Glean facts
     */
    transformCAPToGlean(parseResult, filePath, content) {
        const factBatches = {
            // Core CDS facts
            'src.CDSFile': [],
            'src.CDSEntity': [],
            'src.CDSService': [],
            'src.CDSField': [],
            'src.CDSAssociation': [],
            'src.CDSComposition': [],
            'src.CDSType': [],
            'src.CDSAspect': [],
            'src.CDSAnnotation': [],
            'src.CDSView': [],
            'src.CDSAction': [],
            'src.CDSFunction': [],
            'src.CDSEvent': [],
            'src.CDSProjection': [],
            
            // CAP Service Implementation facts
            'src.CAPServiceHandler': [],
            'src.CAPEventHandler': [],
            'src.CAPMiddleware': [],
            
            // Cross-reference facts
            'src.CDSXRef': [],
            'src.CDSDependency': [],
            
            // Analysis facts
            'src.CAPComplexity': [],
            'src.CAPBestPractice': [],
            'src.CAPSecurity': [],
            'src.CAPPerformance': []
        };

        // Create CDS file fact
        this.createCDSFileFact(parseResult, filePath, content, factBatches);
        
        // Process all symbols
        parseResult.symbols.forEach(symbol => {
            this.processAdvancedSymbol(symbol, parseResult, factBatches);
        });

        // Analyze CAP patterns and best practices
        this.analyzeCAPPatterns(parseResult, content, factBatches);
        
        // Generate cross-references
        this.generateCDSCrossReferences(parseResult, factBatches);

        return factBatches;
    }

    createCDSFileFact(parseResult, filePath, content, factBatches) {
        const lines = content.split('\n').length;
        const checksum = crypto.createHash('sha256').update(content).digest('hex');
        
        // Count different symbol types
        const symbolCounts = parseResult.symbols.reduce((counts, symbol) => {
            counts[symbol.type] = (counts[symbol.type] || 0) + 1;
            return counts;
        }, {});

        const fileFact = {
            id: this.generateFactId(),
            key: {
                file: filePath
            },
            value: {
                file: filePath,
                namespace: parseResult.metadata.namespace,
                language: 'cds',
                size: content.length,
                lines: lines,
                checksum: checksum,
                entities: symbolCounts.entity || 0,
                services: symbolCounts.service || 0,
                types: symbolCounts.type || 0,
                aspects: symbolCounts.aspect || 0,
                views: symbolCounts.view || 0,
                projections: symbolCounts.projection || 0,
                annotations: parseResult.metadata.annotations.length,
                complexity: parseResult.metadata.complexity
            }
        };

        factBatches['src.CDSFile'].push(fileFact);
    }

    processAdvancedSymbol(symbol, parseResult, factBatches) {
        switch (symbol.type) {
            case 'entity':
                this.createEntityFact(symbol, factBatches);
                this.createFieldFacts(symbol, factBatches);
                this.createAssociationFacts(symbol, factBatches);
                this.createCompositionFacts(symbol, factBatches);
                break;
            case 'service':
                this.createServiceFact(symbol, factBatches);
                this.createExposedEntityFacts(symbol, factBatches);
                break;
            case 'type':
                this.createTypeFact(symbol, factBatches);
                break;
            case 'aspect':
                this.createAspectFact(symbol, factBatches);
                this.createFieldFacts(symbol, factBatches);
                break;
            case 'view':
                this.createViewFact(symbol, factBatches);
                break;
            case 'action':
                this.createActionFact(symbol, factBatches);
                break;
            case 'function':
                this.createFunctionFact(symbol, factBatches);
                break;
            case 'event':
                this.createEventFact(symbol, factBatches);
                break;
            case 'projection':
                this.createProjectionFact(symbol, factBatches);
                break;
            case 'association':
                this.createStandaloneAssociationFact(symbol, factBatches);
                break;
            case 'composition':
                this.createStandaloneCompositionFact(symbol, factBatches);
                break;
        }

        // Create annotation facts for any symbol
        if (symbol.annotations && symbol.annotations.length > 0) {
            this.createAnnotationFacts(symbol, factBatches);
        }
    }

    createEntityFact(symbol, factBatches) {
        const entityFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                namespace: symbol.namespace,
                line: symbol.line,
                fields: symbol.fields ? symbol.fields.map(f => f.name) : [],
                keys: symbol.keys || [],
                associations: symbol.associations ? symbol.associations.map(a => a.name) : [],
                compositions: symbol.compositions ? symbol.compositions.map(c => c.name) : [],
                aspects: symbol.aspects || [],
                annotations: symbol.annotations ? symbol.annotations.map(a => a.name) : [],
                inheritance: symbol.inheritance,
                isManaged: symbol.inheritance ? symbol.inheritance.managed : false,
                isCUID: symbol.inheritance ? symbol.inheritance.cuid : false,
                fieldCount: symbol.fields ? symbol.fields.length : 0,
                complexity: this.calculateEntityComplexity(symbol)
            }
        };

        factBatches['src.CDSEntity'].push(entityFact);
    }

    createServiceFact(symbol, factBatches) {
        const serviceFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                namespace: symbol.namespace,
                line: symbol.line,
                path: symbol.path,
                exposedEntities: symbol.exposedEntities ? symbol.exposedEntities.map(e => e.name) : [],
                actions: symbol.actions ? symbol.actions.map(a => a.name) : [],
                functions: symbol.functions ? symbol.functions.map(f => f.name) : [],
                annotations: symbol.annotations ? symbol.annotations.map(a => a.name) : [],
                entityCount: symbol.exposedEntities ? symbol.exposedEntities.length : 0,
                actionCount: symbol.actions ? symbol.actions.length : 0,
                functionCount: symbol.functions ? symbol.functions.length : 0,
                hasCustomPath: !!symbol.path,
                isRESTful: symbol.path && symbol.path.startsWith('/api')
            }
        };

        factBatches['src.CDSService'].push(serviceFact);
    }

    createFieldFacts(symbol, factBatches) {
        if (!symbol.fields) return;

        symbol.fields.forEach(field => {
            const fieldFact = {
                id: this.generateFactId(),
                key: {
                    file: symbol.file || 'unknown',
                    entity: symbol.name,
                    name: field.name
                },
                value: {
                    file: symbol.file || 'unknown',
                    entity: symbol.name,
                    name: field.name,
                    type: field.type,
                    isKey: field.isKey,
                    isNullable: field.isNullable,
                    isVirtual: field.isVirtual,
                    isCalculated: field.isCalculated,
                    annotations: field.annotations,
                    hasValidation: this.hasFieldValidation(field),
                    hasDefaultValue: field.type.includes('default'),
                    isLocalized: field.type.includes('localized')
                }
            };

            factBatches['src.CDSField'].push(fieldFact);
        });
    }

    createAssociationFacts(symbol, factBatches) {
        if (!symbol.associations) return;

        symbol.associations.forEach(association => {
            const associationFact = {
                id: this.generateFactId(),
                key: {
                    file: symbol.file || 'unknown',
                    source: symbol.name,
                    name: association.name
                },
                value: {
                    file: symbol.file || 'unknown',
                    source: symbol.name,
                    name: association.name,
                    target: association.targetEntity,
                    cardinality: association.cardinality,
                    onCondition: association.onCondition,
                    line: association.line,
                    isBacklink: association.onCondition && association.onCondition.includes('$self'),
                    isManyToMany: association.cardinality === 'many' && this.isTargetMany(association.targetEntity)
                }
            };

            factBatches['src.CDSAssociation'].push(associationFact);
        });
    }

    createCompositionFacts(symbol, factBatches) {
        if (!symbol.compositions) return;

        symbol.compositions.forEach(composition => {
            const compositionFact = {
                id: this.generateFactId(),
                key: {
                    file: symbol.file || 'unknown',
                    source: symbol.name,
                    name: composition.name
                },
                value: {
                    file: symbol.file || 'unknown',
                    source: symbol.name,
                    name: composition.name,
                    target: composition.targetEntity,
                    cardinality: composition.cardinality,
                    line: composition.line,
                    isCascadeDelete: true, // Compositions always cascade delete
                    isOwnership: true
                }
            };

            factBatches['src.CDSComposition'].push(compositionFact);
        });
    }

    createTypeFact(symbol, factBatches) {
        const typeFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                definition: symbol.definition,
                baseType: symbol.baseType,
                isStructured: symbol.isStructured,
                isCustomType: !this.isBuiltInType(symbol.baseType),
                isEnum: symbol.definition && symbol.definition.includes('enum'),
                complexity: symbol.isStructured ? 3 : 1
            }
        };

        factBatches['src.CDSType'].push(typeFact);
    }

    createAspectFact(symbol, factBatches) {
        const aspectFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                inheritance: symbol.inheritance,
                fields: symbol.fields ? symbol.fields.map(f => f.name) : [],
                annotations: symbol.annotations ? symbol.annotations.map(a => a.name) : [],
                fieldCount: symbol.fields ? symbol.fields.length : 0,
                isReusable: true,
                complexity: symbol.fields ? symbol.fields.length : 1
            }
        };

        factBatches['src.CDSAspect'].push(aspectFact);
    }

    createViewFact(symbol, factBatches) {
        const viewFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                fromEntity: symbol.fromEntity,
                columns: symbol.columns || [],
                selectClause: symbol.selectClause,
                isParameterized: symbol.selectClause && symbol.selectClause.includes('$parameters'),
                columnCount: symbol.columns ? symbol.columns.length : 0,
                complexity: this.calculateViewComplexity(symbol)
            }
        };

        factBatches['src.CDSView'].push(viewFact);
    }

    createActionFact(symbol, factBatches) {
        const actionFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                parameters: symbol.parameters || [],
                returnType: symbol.returnType,
                parameterCount: symbol.parameters ? symbol.parameters.length : 0,
                hasReturnValue: !!symbol.returnType,
                isVoidAction: !symbol.returnType
            }
        };

        factBatches['src.CDSAction'].push(actionFact);
    }

    createFunctionFact(symbol, factBatches) {
        const functionFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                parameters: symbol.parameters || [],
                returnType: symbol.returnType,
                parameterCount: symbol.parameters ? symbol.parameters.length : 0,
                hasReturnValue: !!symbol.returnType,
                isPure: true // CDS functions are typically pure
            }
        };

        factBatches['src.CDSFunction'].push(functionFact);
    }

    createEventFact(symbol, factBatches) {
        const eventFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                eventType: symbol.eventType,
                isCustomEvent: !!symbol.eventType,
                isLifecycleEvent: !symbol.eventType
            }
        };

        factBatches['src.CDSEvent'].push(eventFact);
    }

    createProjectionFact(symbol, factBatches) {
        const projectionFact = {
            id: this.generateFactId(),
            key: {
                file: symbol.file || 'unknown',
                name: symbol.name
            },
            value: {
                file: symbol.file || 'unknown',
                name: symbol.name,
                line: symbol.line,
                sourceEntity: symbol.sourceEntity,
                isReadOnly: true, // Projections are typically read-only
                isView: true
            }
        };

        factBatches['src.CDSProjection'].push(projectionFact);
    }

    createStandaloneAssociationFact(symbol, factBatches) {
        // For standalone association declarations
        this.createAssociationFacts({ associations: [symbol], name: 'standalone', file: symbol.file }, factBatches);
    }

    createStandaloneCompositionFact(symbol, factBatches) {
        // For standalone composition declarations
        this.createCompositionFacts({ compositions: [symbol], name: 'standalone', file: symbol.file }, factBatches);
    }

    createAnnotationFacts(symbol, factBatches) {
        if (!symbol.annotations) return;

        symbol.annotations.forEach(annotation => {
            const annotationFact = {
                id: this.generateFactId(),
                key: {
                    file: symbol.file || 'unknown',
                    target: symbol.name,
                    name: annotation.name
                },
                value: {
                    file: symbol.file || 'unknown',
                    target: symbol.name,
                    targetType: symbol.type,
                    name: annotation.name,
                    value: annotation.value,
                    line: annotation.line || symbol.line,
                    isUIAnnotation: annotation.name.startsWith('UI.'),
                    isValidation: this.isValidationAnnotation(annotation.name),
                    isMetadata: this.isMetadataAnnotation(annotation.name),
                    isBehavior: this.isBehaviorAnnotation(annotation.name)
                }
            };

            factBatches['src.CDSAnnotation'].push(annotationFact);
        });
    }

    createExposedEntityFacts(symbol, factBatches) {
        if (!symbol.exposedEntities) return;

        symbol.exposedEntities.forEach(entity => {
            // Create dependency fact for exposed entities
            const dependencyFact = {
                id: this.generateFactId(),
                key: {
                    source: symbol.name,
                    target: entity.sourceEntity,
                    type: 'exposure'
                },
                value: {
                    sourceService: symbol.name,
                    targetEntity: entity.sourceEntity,
                    exposedAs: entity.name,
                    isProjection: entity.isProjection,
                    dependencyType: 'service_exposure'
                }
            };

            factBatches['src.CDSDependency'].push(dependencyFact);
        });
    }

    analyzeCAPPatterns(parseResult, content, factBatches) {
        // Analyze various CAP patterns and best practices
        this.analyzeComplexityPatterns(parseResult, factBatches);
        this.analyzeBestPractices(parseResult, content, factBatches);
        this.analyzeSecurityPatterns(parseResult, content, factBatches);
        this.analyzePerformancePatterns(parseResult, content, factBatches);
    }

    analyzeComplexityPatterns(parseResult, factBatches) {
        const entities = parseResult.symbols.filter(s => s.type === 'entity');
        const services = parseResult.symbols.filter(s => s.type === 'service');
        
        // High complexity entities
        entities.forEach(entity => {
            const complexity = this.calculateEntityComplexity(entity);
            if (complexity > 15) {
                factBatches['src.CAPComplexity'].push({
                    id: this.generateFactId(),
                    key: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        type: 'high_complexity_entity'
                    },
                    value: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        complexity: complexity,
                        issue: 'Entity has high complexity',
                        suggestion: 'Consider breaking down into smaller entities or using aspects',
                        severity: 'medium'
                    }
                });
            }
        });
    }

    analyzeBestPractices(parseResult, content, factBatches) {
        // Check for naming conventions
        parseResult.symbols.forEach(symbol => {
            if (symbol.type === 'entity' && !this.followsNamingConvention(symbol.name)) {
                factBatches['src.CAPBestPractice'].push({
                    id: this.generateFactId(),
                    key: {
                        file: symbol.file || 'unknown',
                        symbol: symbol.name,
                        practice: 'naming_convention'
                    },
                    value: {
                        file: symbol.file || 'unknown',
                        symbol: symbol.name,
                        practice: 'naming_convention',
                        issue: 'Entity name does not follow CAP naming conventions',
                        suggestion: 'Use PascalCase for entity names',
                        severity: 'low'
                    }
                });
            }
        });

        // Check for missing annotations
        const entities = parseResult.symbols.filter(s => s.type === 'entity');
        entities.forEach(entity => {
            if (!entity.annotations || entity.annotations.length === 0) {
                factBatches['src.CAPBestPractice'].push({
                    id: this.generateFactId(),
                    key: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        practice: 'missing_annotations'
                    },
                    value: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        practice: 'missing_annotations',
                        issue: 'Entity lacks UI annotations',
                        suggestion: 'Add @title and @description annotations',
                        severity: 'low'
                    }
                });
            }
        });
    }

    analyzeSecurityPatterns(parseResult, content, factBatches) {
        // Check for missing authorization annotations
        const services = parseResult.symbols.filter(s => s.type === 'service');
        services.forEach(service => {
            const hasAuth = service.annotations && 
                service.annotations.some(a => a.name.includes('requires') || a.name.includes('restrict'));
            
            if (!hasAuth) {
                factBatches['src.CAPSecurity'].push({
                    id: this.generateFactId(),
                    key: {
                        file: service.file || 'unknown',
                        service: service.name,
                        issue: 'missing_authorization'
                    },
                    value: {
                        file: service.file || 'unknown',
                        service: service.name,
                        issue: 'missing_authorization',
                        description: 'Service lacks authorization annotations',
                        suggestion: 'Add @requires or @restrict annotations',
                        severity: 'high'
                    }
                });
            }
        });
    }

    analyzePerformancePatterns(parseResult, content, factBatches) {
        // Check for potential N+1 query problems
        const entities = parseResult.symbols.filter(s => s.type === 'entity');
        entities.forEach(entity => {
            const manyAssociations = entity.associations ? 
                entity.associations.filter(a => a.cardinality === 'many').length : 0;
            
            if (manyAssociations > 3) {
                factBatches['src.CAPPerformance'].push({
                    id: this.generateFactId(),
                    key: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        issue: 'potential_n_plus_one'
                    },
                    value: {
                        file: entity.file || 'unknown',
                        entity: entity.name,
                        issue: 'potential_n_plus_one',
                        description: 'Entity has many "to many" associations',
                        suggestion: 'Consider using projections or views to limit data retrieval',
                        severity: 'medium',
                        associationCount: manyAssociations
                    }
                });
            }
        });
    }

    generateCDSCrossReferences(parseResult, factBatches) {
        // Generate cross-references between entities, services, and their dependencies
        parseResult.symbols.forEach(symbol => {
            if (symbol.type === 'entity' && symbol.associations) {
                symbol.associations.forEach(association => {
                    factBatches['src.CDSXRef'].push({
                        id: this.generateFactId(),
                        key: {
                            source: symbol.name,
                            target: association.targetEntity,
                            type: 'association'
                        },
                        value: {
                            sourceEntity: symbol.name,
                            targetEntity: association.targetEntity,
                            relationshipType: 'association',
                            cardinality: association.cardinality,
                            fieldName: association.name
                        }
                    });
                });
            }
        });
    }

    // Helper methods
    calculateEntityComplexity(entity) {
        let complexity = 1;
        
        if (entity.fields) complexity += entity.fields.length;
        if (entity.associations) complexity += entity.associations.length * 2;
        if (entity.compositions) complexity += entity.compositions.length * 2;
        if (entity.annotations) complexity += Math.floor(entity.annotations.length / 2);
        if (entity.inheritance && entity.inheritance.customAspects) {
            complexity += entity.inheritance.customAspects.length;
        }
        
        return complexity;
    }

    calculateViewComplexity(view) {
        let complexity = 1;
        
        if (view.columns) complexity += view.columns.length;
        if (view.selectClause) {
            const joinCount = (view.selectClause.match(/join/gi) || []).length;
            const whereCount = (view.selectClause.match(/where/gi) || []).length;
            complexity += joinCount * 2 + whereCount;
        }
        
        return complexity;
    }

    hasFieldValidation(field) {
        return field.annotations && field.annotations.includes('assert');
    }

    isTargetMany(targetEntity) {
        // This would need to check if the target entity is typically accessed as many
        // For now, return false as we don't have full entity graph
        return false;
    }

    isBuiltInType(baseType) {
        const builtInTypes = [
            'String', 'Integer', 'Decimal', 'Boolean', 'Date', 'Time', 'DateTime', 'Timestamp',
            'UUID', 'Binary', 'LargeBinary', 'LargeString'
        ];
        return builtInTypes.includes(baseType);
    }

    isValidationAnnotation(name) {
        return name.startsWith('assert') || name.includes('mandatory') || name.includes('readonly');
    }

    isMetadataAnnotation(name) {
        return ['title', 'description', 'label'].includes(name.toLowerCase());
    }

    isBehaviorAnnotation(name) {
        return name.startsWith('cds.on') || name.includes('insertonly') || name.includes('readonly');
    }

    followsNamingConvention(name) {
        // Check if entity name follows PascalCase
        return /^[A-Z][a-zA-Z0-9]*$/.test(name);
    }

    generateFactId() {
        return `cap_fact_${this.factIdCounter++}_${crypto.randomBytes(8).toString('hex')}`;
    }
}

module.exports = CAPFactTransformer;