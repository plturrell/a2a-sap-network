using { a2a.network as db } from '../db/schema';

@path: '/api/v1/glean'
@requires: 'authenticated-user'
service GleanService {
    
    // Index management
    action indexCodebase(paths: array of String, languages: array of String) returns {
        id: UUID;
        status: String;
        startTime: DateTime;
        paths: array of String;
        languages: array of String;
        filesIndexed: Integer;
        errors: array of String;
    };
    
    // Code queries
    action queryCode(query: String, language: String, limit: Integer) returns array of {
        file: String;
        line: Integer;
        match: String;
        context: String;
    };
    
    // Component analysis
    action analyzeComponents(componentPath: String) returns {
        path: String;
        type: String enum { service; ui; database; blockchain; library };
        dependencies: array of {
            module: String;
            type: String;
            path: String;
        };
        exports: array of {
            name: String;
            type: String;
        };
        complexity: {
            linesOfCode: Integer;
            cyclomaticComplexity: Integer;
            nestingDepth: Integer;
            functionCount: Integer;
        };
        security: array of {
            issue: String;
            line: Integer;
            severity: String enum { low; medium; high; critical };
        };
        performance: array of {
            issue: String;
            line: Integer;
            severity: String enum { low; medium; high };
        };
    };
    
    // Dependency analysis
    action findDependencies(sourcePath: String, depth: Integer) returns {
        sourcePath: String;
        depth: Integer;
        dependencies: array of {
            type: String;
            module: String;
            path: String;
            depth: Integer;
        };
        graph: {
            nodes: array of String;
            edges: array of {
                ![from]: String;
                to: String;
            };
        };
    };
    
    // Security scanning
    action detectSecurityIssues(paths: array of String, severity: String) returns {
        totalIssues: Integer;
        bySeverity: {
            critical: Integer;
            high: Integer;
            medium: Integer;
            low: Integer;
        };
        issues: array of {
            file: String;
            type: String;
            severity: String;
            line: Integer;
            description: String;
        };
    };
    
    // Performance analysis
    action analyzePerformance(paths: array of String) returns {
        slowQueries: array of {
            file: String;
            issue: String;
            suggestion: String;
        };
        memoryLeaks: array of {
            file: String;
            issue: String;
            suggestion: String;
        };
        inefficientLoops: array of {
            file: String;
            issue: String;
            line: Integer;
        };
        blockingOperations: array of {
            file: String;
            issue: String;
            line: Integer;
        };
    };
    
    // Code health metrics
    action getCodeHealth(paths: array of String) returns {
        totalFiles: Integer;
        totalLines: Integer;
        averageComplexity: Decimal(10,2);
        testCoverage: Decimal(5,2);
        documentationCoverage: Decimal(5,2);
        technicalDebt: array of {
            file: String;
            issue: String;
            coverage: Integer;
        };
        codeSmells: array of {
            file: String;
            issue: String;
            complexity: Integer;
        };
    };
    
    // Integration with diagnostic tools
    function getDiagnosticSummary() returns {
        timestamp: DateTime;
        codeHealth: {
            score: Integer;
            status: String;
            topIssues: array of {
                type: String;
                severity: String;
                file: String;
                message: String;
            };
        };
        dependencies: {
            totalDependencies: Integer;
            circularDependencies: array of {
                component: String;
                cycle: String;
            };
        };
        security: {
            totalIssues: Integer;
            critical: Integer;
            high: Integer;
            medium: Integer;
            low: Integer;
        };
        recommendations: array of {
            category: String;
            priority: String;
            action: String;
            impact: String;
        };
    };
    
    // Events for real-time updates
    event IndexingStarted : {
        indexId: UUID;
        paths: array of String;
        timestamp: DateTime;
    }
    
    event IndexingCompleted : {
        indexId: UUID;
        filesIndexed: Integer;
        duration: Integer;
        timestamp: DateTime;
    }
    
    event SecurityIssueDetected : {
        file: String;
        type: String;
        severity: String;
        description: String;
        timestamp: DateTime;
    }
    
    event CodeHealthChanged : {
        previousScore: Integer;
        currentScore: Integer;
        trend: String enum { improving; degrading; stable };
        timestamp: DateTime;
    }
}