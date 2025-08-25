/**
 * @fileoverview CAP-specific Query Patterns for Glean Angle queries
 * @module capQueryPatterns
 * @since 1.0.0
 *
 * Provides pre-built query patterns for common CAP analysis scenarios
 */

class CAPQueryPatterns {
    constructor() {
        this.patterns = new Map();
        this.initializePatterns();
    }

    initializePatterns() {
        // Entity Analysis Patterns
        this.patterns.set('complex_entities', {
            name: 'Find Complex Entities',
            query: `query FindComplexEntities(threshold: nat) : [CDSEntity] =
                src.CDSEntity { file, name, fieldCount, complexity }
                where complexity > threshold`,
            description: 'Find entities with high complexity (many fields, associations, compositions)',
            category: 'complexity'
        });

        this.patterns.set('entities_without_keys', {
            name: 'Entities Without Primary Keys',
            query: `query EntitiesWithoutKeys() : [CDSEntity] =
                src.CDSEntity { file, name, keys }
                where keys = []`,
            description: 'Find entities that lack proper primary key definitions',
            category: 'validation'
        });

        this.patterns.set('managed_entities', {
            name: 'Managed Entities',
            query: `query ManagedEntities() : [CDSEntity] =
                src.CDSEntity { file, name, isManaged }
                where isManaged = true`,
            description: 'Find all entities that inherit from managed aspect',
            category: 'architecture'
        });

        // Service Analysis Patterns
        this.patterns.set('services_without_auth', {
            name: 'Services Without Authorization',
            query: `query ServicesWithoutAuth() : [CDSService] =
                src.CDSService { file, name, annotations }
                where !(annotations contains "requires" || annotations contains "restrict")`,
            description: 'Find services that lack authorization annotations',
            category: 'security'
        });

        this.patterns.set('restful_services', {
            name: 'RESTful Services',
            query: `query RESTfulServices() : [CDSService] =
                src.CDSService { file, name, path, isRESTful }
                where isRESTful = true`,
            description: 'Find services with REST API endpoints',
            category: 'architecture'
        });

        this.patterns.set('large_services', {
            name: 'Large Services',
            query: `query LargeServices(threshold: nat) : [CDSService] =
                src.CDSService { file, name, entityCount, actionCount, functionCount }
                where entityCount + actionCount + functionCount > threshold`,
            description: 'Find services with many exposed entities, actions, and functions',
            category: 'complexity'
        });

        // Association and Composition Patterns
        this.patterns.set('circular_associations', {
            name: 'Circular Associations',
            query: `query CircularAssociations() : [(string, string)] =
                (entity1, entity2) where
                    src.CDSAssociation { source = entity1, target = entity2 } &&
                    src.CDSAssociation { source = entity2, target = entity1 }`,
            description: 'Find entities with circular association relationships',
            category: 'design'
        });

        this.patterns.set('many_to_many_associations', {
            name: 'Many-to-Many Associations',
            query: `query ManyToManyAssociations() : [CDSAssociation] =
                src.CDSAssociation { source, target, cardinality, isManyToMany }
                where isManyToMany = true`,
            description: 'Find many-to-many association relationships',
            category: 'performance'
        });

        this.patterns.set('orphaned_compositions', {
            name: 'Orphaned Compositions',
            query: `query OrphanedCompositions() : [CDSComposition] =
                src.CDSComposition { source, target, name }
                where !(src.CDSEntity { name = target })`,
            description: 'Find compositions pointing to non-existent entities',
            category: 'validation'
        });

        // Annotation Patterns
        this.patterns.set('missing_ui_annotations', {
            name: 'Entities Missing UI Annotations',
            query: `query EntitiesMissingUIAnnotations() : [CDSEntity] =
                src.CDSEntity { file, name, annotations }
                where !(annotations contains "title" || annotations contains "label")`,
            description: 'Find entities without UI metadata annotations',
            category: 'usability'
        });

        this.patterns.set('validation_annotations', {
            name: 'Fields with Validation',
            query: `query FieldsWithValidation() : [CDSField] =
                src.CDSField { entity, name, annotations }
                where annotations contains "assert"`,
            description: 'Find fields with validation annotations',
            category: 'validation'
        });

        this.patterns.set('localized_fields', {
            name: 'Localized Fields',
            query: `query LocalizedFields() : [CDSField] =
                src.CDSField { entity, name, isLocalized }
                where isLocalized = true`,
            description: 'Find fields that support localization',
            category: 'i18n'
        });

        // Performance Patterns
        this.patterns.set('potential_n_plus_one', {
            name: 'Potential N+1 Query Issues',
            query: `query PotentialNPlusOne() : [CAPPerformance] =
                src.CAPPerformance { entity, issue, associationCount }
                where issue = "potential_n_plus_one"`,
            description: 'Find entities with potential N+1 query problems',
            category: 'performance'
        });

        this.patterns.set('complex_views', {
            name: 'Complex Views',
            query: `query ComplexViews(threshold: nat) : [CDSView] =
                src.CDSView { file, name, complexity, columnCount }
                where complexity > threshold`,
            description: 'Find views with high complexity (joins, conditions)',
            category: 'performance'
        });

        // Security Patterns
        this.patterns.set('hardcoded_auth', {
            name: 'Hardcoded Authorization',
            query: `query HardcodedAuth() : [CDSAnnotation] =
                src.CDSAnnotation { file, target, name, value }
                where name = "requires" && value contains "hardcoded"`,
            description: 'Find hardcoded authorization configurations',
            category: 'security'
        });

        this.patterns.set('missing_field_auth', {
            name: 'Fields Without Authorization',
            query: `query FieldsWithoutAuth() : [CDSField] =
                src.CDSField { entity, name, annotations }
                where !(annotations contains "readonly" || annotations contains "insertonly")`,
            description: 'Find fields without access control annotations',
            category: 'security'
        });

        // Best Practices Patterns
        this.patterns.set('naming_violations', {
            name: 'Naming Convention Violations',
            query: `query NamingViolations() : [CAPBestPractice] =
                src.CAPBestPractice { symbol, practice, issue }
                where practice = "naming_convention"`,
            description: 'Find entities and fields not following naming conventions',
            category: 'best_practices'
        });

        this.patterns.set('missing_annotations', {
            name: 'Missing Required Annotations',
            query: `query MissingAnnotations() : [CAPBestPractice] =
                src.CAPBestPractice { entity, practice, issue }
                where practice = "missing_annotations"`,
            description: 'Find entities without required annotations',
            category: 'best_practices'
        });

        // Cross-Reference Patterns
        this.patterns.set('entity_relationships', {
            name: 'Entity Relationship Graph',
            query: `query EntityRelationships(entityName: string) : [CDSXRef] =
                src.CDSXRef { sourceEntity, targetEntity, relationshipType }
                where sourceEntity = entityName || targetEntity = entityName`,
            description: 'Find all relationships for a specific entity',
            category: 'architecture'
        });

        this.patterns.set('service_dependencies', {
            name: 'Service Dependencies',
            query: `query ServiceDependencies(serviceName: string) : [CDSDependency] =
                src.CDSDependency { sourceService, targetEntity, dependencyType }
                where sourceService = serviceName`,
            description: 'Find all entities exposed by a service',
            category: 'architecture'
        });

        // Type System Patterns
        this.patterns.set('custom_types', {
            name: 'Custom Type Definitions',
            query: `query CustomTypes() : [CDSType] =
                src.CDSType { file, name, isCustomType }
                where isCustomType = true`,
            description: 'Find custom type definitions',
            category: 'architecture'
        });

        this.patterns.set('structured_types', {
            name: 'Structured Types',
            query: `query StructuredTypes() : [CDSType] =
                src.CDSType { file, name, isStructured, complexity }
                where isStructured = true`,
            description: 'Find complex structured type definitions',
            category: 'complexity'
        });

        // Aspect Patterns
        this.patterns.set('reusable_aspects', {
            name: 'Reusable Aspects',
            query: `query ReusableAspects() : [CDSAspect] =
                src.CDSAspect { file, name, fieldCount, isReusable }
                where isReusable = true`,
            description: 'Find aspect definitions for code reuse',
            category: 'architecture'
        });

        this.patterns.set('unused_aspects', {
            name: 'Unused Aspects',
            query: `query UnusedAspects() : [CDSAspect] =
                src.CDSAspect { file, name }
                where !(src.CDSEntity { aspects contains name })`,
            description: 'Find aspects that are not used by any entity',
            category: 'cleanup'
        });

        // Action and Function Patterns
        this.patterns.set('complex_actions', {
            name: 'Complex Actions',
            query: `query ComplexActions(threshold: nat) : [CDSAction] =
                src.CDSAction { file, name, parameterCount }
                where parameterCount > threshold`,
            description: 'Find actions with many parameters',
            category: 'complexity'
        });

        this.patterns.set('void_actions', {
            name: 'Void Actions',
            query: `query VoidActions() : [CDSAction] =
                src.CDSAction { file, name, isVoidAction }
                where isVoidAction = true`,
            description: 'Find actions that do not return values',
            category: 'design'
        });

        // Projection Patterns
        this.patterns.set('all_projections', {
            name: 'All Projections',
            query: `query AllProjections() : [CDSProjection] =
                src.CDSProjection { file, name, sourceEntity }`,
            description: 'Find all projection definitions',
            category: 'architecture'
        });

        this.patterns.set('projection_chains', {
            name: 'Projection Chains',
            query: `query ProjectionChains() : [(string, string)] =
                (proj1, proj2) where
                    src.CDSProjection { name = proj1, sourceEntity = proj2 } &&
                    src.CDSProjection { name = proj2 }`,
            description: 'Find projections that are based on other projections',
            category: 'architecture'
        });

        // File Organization Patterns
        this.patterns.set('large_files', {
            name: 'Large CDS Files',
            query: `query LargeCDSFiles(threshold: nat) : [CDSFile] =
                src.CDSFile { file, entities, services, types, complexity }
                where entities + services + types > threshold`,
            description: 'Find CDS files with many definitions',
            category: 'organization'
        });

        this.patterns.set('namespace_usage', {
            name: 'Namespace Usage',
            query: `query NamespaceUsage(namespace: string) : [CDSFile] =
                src.CDSFile { file, namespace as ns }
                where ns = namespace`,
            description: 'Find files using a specific namespace',
            category: 'organization'
        });
    }

    getPattern(patternName) {
        return this.patterns.get(patternName);
    }

    getAllPatterns() {
        return Array.from(this.patterns.values());
    }

    getPatternsByCategory(category) {
        return Array.from(this.patterns.values()).filter(pattern => pattern.category === category);
    }

    getCategories() {
        const categories = new Set();
        this.patterns.forEach(pattern => categories.add(pattern.category));
        return Array.from(categories).sort();
    }

    executePattern(patternName, executor, parameters = {}) {
        const pattern = this.patterns.get(patternName);
        if (!pattern) {
            throw new Error(`Pattern '${patternName}' not found`);
        }

        // Replace parameters in query if needed
        let query = pattern.query;
        Object.entries(parameters).forEach(([key, value]) => {
            query = query.replace(new RegExp(`\\b${key}\\b`, 'g'), value);
        });

        return executor.executeQuery(query);
    }

    suggestPatterns(keywords) {
        const suggestions = [];
        const searchTerms = keywords.toLowerCase().split(' ');

        this.patterns.forEach((pattern, name) => {
            const searchText = `${pattern.name} ${pattern.description} ${pattern.category}`.toLowerCase();

            const score = searchTerms.reduce((acc, term) => {
                return acc + (searchText.includes(term) ? 1 : 0);
            }, 0);

            if (score > 0) {
                suggestions.push({
                    name,
                    pattern,
                    relevance: score / searchTerms.length
                });
            }
        });

        return suggestions.sort((a, b) => b.relevance - a.relevance);
    }

    generateComplexQuery(entities, relationships, constraints) {
        // Generate complex queries based on entity relationships
        let query = 'query ComplexAnalysis() : [Result] = ';

        // Add entity joins
        const entityClauses = entities.map(entity =>
            `src.CDSEntity { name = "${entity}", file, complexity }`
        );

        // Add relationship constraints
        const relationshipClauses = relationships.map(rel =>
            `src.CDSAssociation { source = "${rel.from}", target = "${rel.to}" }`
        );

        // Combine clauses
        const allClauses = [...entityClauses, ...relationshipClauses];
        query += allClauses.join(' && ');

        // Add constraints
        if (constraints && constraints.length > 0) {
            query += ` where ${  constraints.join(' && ')}`;
        }

        return query;
    }
}

module.exports = CAPQueryPatterns;