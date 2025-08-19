/**
 * @fileoverview Advanced SAP CAP CDS Parser for comprehensive code analysis
 * @module advancedCDSParser
 * @since 1.0.0
 * 
 * Provides deep analysis of CDS files including annotations, views, projections,
 * compositions, associations, and advanced CAP patterns
 */

class AdvancedCDSParser {
    constructor() {
        this.symbolCounter = 0;
        this.annotationTypes = new Set([
            'title', 'description', 'readonly', 'insertonly', 'mandatory',
            'cds.on.insert', 'cds.on.update', 'cds.persistence.exists',
            'UI.Hidden', 'UI.Label', 'UI.HeaderInfo', 'Common.ValueList',
            'odata.Type', 'odata.MaxLength', 'restrict', 'requires',
            'path', 'format', 'timezone', 'assert.format', 'assert.range'
        ]);
    }

    parseAdvancedCDSContent(content, filePath) {
        const result = {
            symbols: [],
            occurrences: [],
            metadata: {
                namespace: null,
                imports: [],
                annotations: [],
                complexity: this.calculateComplexity(content)
            }
        };

        if (!content || content.trim().length === 0) {
            console.warn('Empty or null content provided to CDS parser');
            return result;
        }

        // Parse all CDS constructs
        this.parseNamespace(content, result);
        this.parseImports(content, result);
        this.parseEntities(content, result);
        this.parseServices(content, result);
        this.parseTypes(content, result);
        this.parseAspects(content, result);
        this.parseViews(content, result);
        this.parseActions(content, result);
        this.parseFunctions(content, result);
        this.parseEvents(content, result);
        this.parseAnnotations(content, result);
        this.parseCompositions(content, result);
        this.parseAssociations(content, result);
        this.parseProjections(content, result);

        return result;
    }

    parseNamespace(content, result) {
        const namespaceRegex = /namespace\s+([\w\.]+)\s*;/g;
        let match;
        
        while ((match = namespaceRegex.exec(content)) !== null) {
            const namespaceName = match[1];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `namespace_${this.symbolCounter++}`;
            
            result.metadata.namespace = namespaceName;
            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'Namespace'
                },
                name: namespaceName,
                type: 'namespace',
                line: lineNumber,
                fullPath: namespaceName.split('.')
            });
        }
    }

    parseImports(content, result) {
        // Enhanced import parsing with different patterns
        const patterns = [
            // using { items } from 'module'
            /using\s*\{\s*([^}]+)\s*\}\s+from\s+['"]([^'"]+)['"]/g,
            // using module as alias
            /using\s+([\w\.]+)\s+as\s+([\w]+)/g,
            // using module
            /using\s+([\w\.@\/]+)(?:\s*;|\s+from\s+['"]([^'"]+)['"])/g
        ];

        patterns.forEach((pattern, patternIndex) => {
            let match;
            while ((match = pattern.exec(content)) !== null) {
                const lineNumber = this.getLineNumber(content, match.index);
                const symbolId = `import_${this.symbolCounter++}`;
                
                let importData = {
                    symbol: symbolId,
                    definition: {
                        range: this.createRange(content, match.index, match[0].length),
                        syntax_kind: 'ImportDeclaration'
                    },
                    type: 'import',
                    line: lineNumber
                };

                if (patternIndex === 0) {
                    // Named imports
                    importData.name = match[2];
                    importData.importedNames = match[1].split(',').map(name => name.trim());
                    importData.importType = 'named';
                } else if (patternIndex === 1) {
                    // Aliased import
                    importData.name = match[1];
                    importData.alias = match[2];
                    importData.importType = 'alias';
                } else {
                    // Simple import
                    importData.name = match[1];
                    importData.importType = 'simple';
                }

                result.symbols.push(importData);
                result.metadata.imports.push(importData);
            }
        });
    }

    parseEntities(content, result) {
        // Enhanced entity parsing with inheritance and aspects
        const entityRegex = /entity\s+([\w]+)(?:\s*:\s*([^{]+))?\s*\{([^}]+)\}/gs;
        let match;

        while ((match = entityRegex.exec(content)) !== null) {
            const entityName = match[1];
            const inheritance = match[2] ? match[2].trim() : null;
            const entityBody = match[3];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `entity_${this.symbolCounter++}`;

            const entityData = {
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'EntityDeclaration'
                },
                name: entityName,
                type: 'entity',
                line: lineNumber,
                inheritance: this.parseInheritance(inheritance),
                fields: this.parseEntityFields(entityBody),
                keys: this.parseEntityKeys(entityBody),
                associations: this.parseEntityAssociations(entityBody),
                compositions: this.parseEntityCompositions(entityBody),
                annotations: this.parseInlineAnnotations(entityBody),
                aspects: this.parseEntityAspects(inheritance)
            };

            result.symbols.push(entityData);
        }
    }

    parseServices(content, result) {
        // Enhanced service parsing with actions and functions
        const serviceRegex = /service\s+([\w]+)(?:\s*@\([^)]+\))?\s*\{([^}]+)\}/gs;
        let match;

        while ((match = serviceRegex.exec(content)) !== null) {
            const serviceName = match[1];
            const serviceBody = match[2];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `service_${this.symbolCounter++}`;

            const serviceData = {
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'ServiceDeclaration'
                },
                name: serviceName,
                type: 'service',
                line: lineNumber,
                exposedEntities: this.parseExposedEntities(serviceBody),
                actions: this.parseServiceActions(serviceBody),
                functions: this.parseServiceFunctions(serviceBody),
                annotations: this.parseServiceAnnotations(match[0]),
                path: this.extractServicePath(match[0])
            };

            result.symbols.push(serviceData);
        }
    }

    parseTypes(content, result) {
        const typeRegex = /type\s+([\w]+)(?:\s*:\s*([^;]+))?/g;
        let match;

        while ((match = typeRegex.exec(content)) !== null) {
            const typeName = match[1];
            const typeDefinition = match[2] ? match[2].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `type_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'TypeDeclaration'
                },
                name: typeName,
                type: 'type',
                line: lineNumber,
                definition: typeDefinition,
                isStructured: typeDefinition && typeDefinition.includes('{'),
                baseType: this.extractBaseType(typeDefinition)
            });
        }
    }

    parseAspects(content, result) {
        const aspectRegex = /aspect\s+([\w]+)(?:\s*:\s*([^{]+))?\s*\{([^}]+)\}/gs;
        let match;

        while ((match = aspectRegex.exec(content)) !== null) {
            const aspectName = match[1];
            const inheritance = match[2] ? match[2].trim() : null;
            const aspectBody = match[3];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `aspect_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'AspectDeclaration'
                },
                name: aspectName,
                type: 'aspect',
                line: lineNumber,
                inheritance: inheritance,
                fields: this.parseEntityFields(aspectBody),
                annotations: this.parseInlineAnnotations(aspectBody)
            });
        }
    }

    parseViews(content, result) {
        const viewRegex = /view\s+([\w]+)(?:\s*@\([^)]+\))?\s*as\s+select\s+([^;]+);/gs;
        let match;

        while ((match = viewRegex.exec(content)) !== null) {
            const viewName = match[1];
            const selectClause = match[2];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `view_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'ViewDeclaration'
                },
                name: viewName,
                type: 'view',
                line: lineNumber,
                selectClause: selectClause.trim(),
                fromEntity: this.extractFromEntity(selectClause),
                columns: this.extractViewColumns(selectClause)
            });
        }
    }

    parseActions(content, result) {
        const actionRegex = /action\s+([\w]+)\s*\(([^)]*)\)(?:\s*returns\s+([^;]+))?/g;
        let match;

        while ((match = actionRegex.exec(content)) !== null) {
            const actionName = match[1];
            const parameters = match[2];
            const returnType = match[3] ? match[3].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `action_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'ActionDeclaration'
                },
                name: actionName,
                type: 'action',
                line: lineNumber,
                parameters: this.parseActionParameters(parameters),
                returnType: returnType,
                isFunction: false
            });
        }
    }

    parseFunctions(content, result) {
        const functionRegex = /function\s+([\w]+)\s*\(([^)]*)\)(?:\s*returns\s+([^;]+))?/g;
        let match;

        while ((match = functionRegex.exec(content)) !== null) {
            const functionName = match[1];
            const parameters = match[2];
            const returnType = match[3] ? match[3].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `function_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'FunctionDeclaration'
                },
                name: functionName,
                type: 'function',
                line: lineNumber,
                parameters: this.parseActionParameters(parameters),
                returnType: returnType,
                isFunction: true
            });
        }
    }

    parseEvents(content, result) {
        const eventRegex = /event\s+([\w]+)(?:\s*:\s*([^;]+))?/g;
        let match;

        while ((match = eventRegex.exec(content)) !== null) {
            const eventName = match[1];
            const eventType = match[2] ? match[2].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `event_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'EventDeclaration'
                },
                name: eventName,
                type: 'event',
                line: lineNumber,
                eventType: eventType
            });
        }
    }

    parseAnnotations(content, result) {
        // Parse standalone annotations and annotation values
        const annotationRegex = /@([\w\.]+)(?:\s*:\s*([^;@]+))?/g;
        let match;

        while ((match = annotationRegex.exec(content)) !== null) {
            const annotationName = match[1];
            const annotationValue = match[2] ? match[2].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);

            const annotation = {
                name: annotationName,
                value: annotationValue,
                line: lineNumber,
                isKnownAnnotation: this.annotationTypes.has(annotationName)
            };

            result.metadata.annotations.push(annotation);
        }
    }

    parseCompositions(content, result) {
        const compositionRegex = /([\w]+)\s*:\s*Composition\s+of\s+(many\s+)?([\w\.]+)/g;
        let match;

        while ((match = compositionRegex.exec(content)) !== null) {
            const fieldName = match[1];
            const isMany = !!match[2];
            const targetEntity = match[3];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `composition_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'CompositionDeclaration'
                },
                name: fieldName,
                type: 'composition',
                line: lineNumber,
                targetEntity: targetEntity,
                cardinality: isMany ? 'many' : 'one'
            });
        }
    }

    parseAssociations(content, result) {
        const associationRegex = /([\w]+)\s*:\s*Association\s+to\s+(many\s+)?([\w\.]+)(?:\s+on\s+([^;]+))?/g;
        let match;

        while ((match = associationRegex.exec(content)) !== null) {
            const fieldName = match[1];
            const isMany = !!match[2];
            const targetEntity = match[3];
            const onCondition = match[4] ? match[4].trim() : null;
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `association_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'AssociationDeclaration'
                },
                name: fieldName,
                type: 'association',
                line: lineNumber,
                targetEntity: targetEntity,
                cardinality: isMany ? 'many' : 'one',
                onCondition: onCondition
            });
        }
    }

    parseProjections(content, result) {
        const projectionRegex = /entity\s+([\w]+)\s+as\s+projection\s+on\s+([\w\.]+)/g;
        let match;

        while ((match = projectionRegex.exec(content)) !== null) {
            const projectionName = match[1];
            const sourceEntity = match[2];
            const lineNumber = this.getLineNumber(content, match.index);
            const symbolId = `projection_${this.symbolCounter++}`;

            result.symbols.push({
                symbol: symbolId,
                definition: {
                    range: this.createRange(content, match.index, match[0].length),
                    syntax_kind: 'ProjectionDeclaration'
                },
                name: projectionName,
                type: 'projection',
                line: lineNumber,
                sourceEntity: sourceEntity
            });
        }
    }

    // Helper methods
    parseInheritance(inheritance) {
        if (!inheritance) return null;
        
        const aspects = [];
        const managed = inheritance.includes('managed');
        const cuid = inheritance.includes('cuid');
        
        if (managed) aspects.push('managed');
        if (cuid) aspects.push('cuid');
        
        // Parse custom aspects
        const customAspects = inheritance.replace(/managed|cuid/g, '').split(',')
            .map(aspect => aspect.trim()).filter(aspect => aspect);
        
        return {
            managed,
            cuid,
            customAspects,
            allAspects: [...aspects, ...customAspects]
        };
    }

    parseEntityFields(entityBody) {
        const fieldRegex = /([\w]+)\s*:\s*([^;@]+)(?:@([^;]+))?/g;
        const fields = [];
        let match;

        while ((match = fieldRegex.exec(entityBody)) !== null) {
            const fieldName = match[1];
            const fieldType = match[2].trim();
            const annotations = match[3] ? match[3].trim() : null;

            fields.push({
                name: fieldName,
                type: fieldType,
                annotations: annotations,
                isKey: fieldType.includes('key') || fieldName === 'ID',
                isNullable: !fieldType.includes('not null'),
                isVirtual: fieldType.includes('virtual'),
                isCalculated: fieldType.includes('=')
            });
        }

        return fields;
    }

    parseEntityKeys(entityBody) {
        const keyFields = [];
        const keyRegex = /key\s+([\w]+)/g;
        let match;

        while ((match = keyRegex.exec(entityBody)) !== null) {
            keyFields.push(match[1]);
        }

        return keyFields;
    }

    parseEntityAssociations(entityBody) {
        const associationRegex = /([\w]+)\s*:\s*Association\s+to\s+(many\s+)?([\w\.]+)(?:\s+on\s+([^;]+))?/g;
        const associations = [];
        let match;

        while ((match = associationRegex.exec(entityBody)) !== null) {
            const fieldName = match[1];
            const isMany = !!match[2];
            const targetEntity = match[3];
            const onCondition = match[4] ? match[4].trim() : null;

            associations.push({
                name: fieldName,
                targetEntity: targetEntity,
                cardinality: isMany ? 'many' : 'one',
                onCondition: onCondition
            });
        }

        return associations;
    }

    parseEntityCompositions(entityBody) {
        const compositionRegex = /([\w]+)\s*:\s*Composition\s+of\s+(many\s+)?([\w\.]+)/g;
        const compositions = [];
        let match;

        while ((match = compositionRegex.exec(entityBody)) !== null) {
            const fieldName = match[1];
            const isMany = !!match[2];
            const targetEntity = match[3];

            compositions.push({
                name: fieldName,
                targetEntity: targetEntity,
                cardinality: isMany ? 'many' : 'one'
            });
        }

        return compositions;
    }

    parseInlineAnnotations(body) {
        const annotations = [];
        const annotationRegex = /@([\w\.]+)(?:\s*:\s*([^;@]+))?/g;
        let match;

        while ((match = annotationRegex.exec(body)) !== null) {
            annotations.push({
                name: match[1],
                value: match[2] ? match[2].trim() : null
            });
        }

        return annotations;
    }

    parseEntityAspects(inheritance) {
        if (!inheritance) return [];
        
        const aspectNames = inheritance.split(',')
            .map(aspect => aspect.trim())
            .filter(aspect => aspect && !['managed', 'cuid'].includes(aspect));
        
        return aspectNames;
    }

    parseExposedEntities(serviceBody) {
        const entities = [];
        const entityRegex = /entity\s+([\w]+)(?:\s+as\s+projection\s+on\s+([\w\.]+))?/g;
        let match;

        while ((match = entityRegex.exec(serviceBody)) !== null) {
            entities.push({
                name: match[1],
                sourceEntity: match[2] || match[1],
                isProjection: !!match[2]
            });
        }

        return entities;
    }

    parseServiceActions(serviceBody) {
        const actionRegex = /action\s+([\w]+)\s*\(([^)]*)\)(?:\s*returns\s+([^;]+))?/g;
        const actions = [];
        let match;

        while ((match = actionRegex.exec(serviceBody)) !== null) {
            const actionName = match[1];
            const parameters = match[2];
            const returnType = match[3] ? match[3].trim() : null;

            actions.push({
                name: actionName,
                parameters: this.parseActionParameters(parameters),
                returnType: returnType,
                isFunction: false
            });
        }

        return actions;
    }

    parseServiceFunctions(serviceBody) {
        const functionRegex = /function\s+([\w]+)\s*\(([^)]*)\)(?:\s*returns\s+([^;]+))?/g;
        const functions = [];
        let match;

        while ((match = functionRegex.exec(serviceBody)) !== null) {
            const functionName = match[1];
            const parameters = match[2];
            const returnType = match[3] ? match[3].trim() : null;

            functions.push({
                name: functionName,
                parameters: this.parseActionParameters(parameters),
                returnType: returnType,
                isFunction: true
            });
        }

        return functions;
    }

    parseServiceAnnotations(serviceDeclaration) {
        const annotations = [];
        const annotationRegex = /@\(([^)]+)\)/g;
        let match;

        while ((match = annotationRegex.exec(serviceDeclaration)) !== null) {
            const annotationContent = match[1];
            const pathMatch = annotationContent.match(/path\s*:\s*['"]([^'"]+)['"]/);
            
            if (pathMatch) {
                annotations.push({
                    name: 'path',
                    value: pathMatch[1]
                });
            }
        }

        return annotations;
    }

    extractServicePath(serviceDeclaration) {
        const pathMatch = serviceDeclaration.match(/@\([^)]*path\s*:\s*['"]([^'"]+)['"]/);
        return pathMatch ? pathMatch[1] : null;
    }

    extractFromEntity(selectClause) {
        const fromMatch = selectClause.match(/from\s+([\w\.]+)/i);
        return fromMatch ? fromMatch[1] : null;
    }

    extractViewColumns(selectClause) {
        const selectPart = selectClause.split(/\s+from\s+/i)[0];
        return selectPart.split(',').map(col => col.trim());
    }

    parseActionParameters(parameters) {
        if (!parameters.trim()) return [];
        
        return parameters.split(',').map(param => {
            const parts = param.trim().split(':');
            return {
                name: parts[0].trim(),
                type: parts[1] ? parts[1].trim() : 'String'
            };
        });
    }

    extractBaseType(typeDefinition) {
        if (!typeDefinition) return null;
        
        const baseTypeMatch = typeDefinition.match(/^([\w]+)/);
        return baseTypeMatch ? baseTypeMatch[1] : null;
    }

    calculateComplexity(content) {
        // Calculate CDS complexity based on various factors
        const entityCount = (content.match(/entity\s+[\w]+/g) || []).length;
        const serviceCount = (content.match(/service\s+[\w]+/g) || []).length;
        const associationCount = (content.match(/Association\s+to/g) || []).length;
        const compositionCount = (content.match(/Composition\s+of/g) || []).length;
        const annotationCount = (content.match(/@[\w\.]+/g) || []).length;
        
        return entityCount * 2 + serviceCount * 3 + associationCount + compositionCount + Math.floor(annotationCount / 5);
    }

    getLineNumber(content, index) {
        return content.substring(0, index).split('\n').length - 1;
    }

    createRange(content, index, length) {
        const startLine = this.getLineNumber(content, index);
        const endLine = this.getLineNumber(content, index + length);
        const lines = content.substring(0, index).split('\n');
        const startColumn = lines[lines.length - 1].length;
        const endLines = content.substring(0, index + length).split('\n');
        const endColumn = endLines[endLines.length - 1].length;

        return {
            start: { line: startLine, character: startColumn },
            end: { line: endLine, character: endColumn }
        };
    }
}

module.exports = AdvancedCDSParser;