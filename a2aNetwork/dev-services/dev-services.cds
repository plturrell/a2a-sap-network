namespace com.a2a.devservices;

service DevServices {
    // Mock Registry
    entity Agents {
        key ID : UUID;
        name : String;
        capability : String;
        status : String;
        metadata : LargeString;
    }
    
    action registerAgent(agentData : String) returns String;
    function discoverAgents(capability : String) returns array of Agents;
    
    // Test Orchestration
    entity TestSuites {
        key ID : UUID;
        name : String;
        status : String;
        results : LargeString;
        createdAt : DateTime;
    }
    
    action runTestSuite(testData : String) returns TestSuites;
    action testAgent(agentId : UUID, testConfig : String) returns String;
    function getTestResults(testId : UUID) returns TestSuites;
    
    // Agent Simulation
    entity Scenarios {
        key ID : UUID;
        name : String;
        description : String;
        config : LargeString;
    }
    
    action runScenario(scenarioData : String) returns String;
    function getAvailableScenarios() returns array of Scenarios;
    action runLoadTest(loadTestConfig : String) returns String;
    
    // Service Mocking
    entity MockServices {
        key ID : UUID;
        serviceName : String;
        mockConfig : LargeString;
        active : Boolean;
    }
    
    action createMock(mockData : String) returns MockServices;
    action updateMock(mockId : UUID, mockData : String) returns MockServices;
    action deleteMock(mockId : UUID) returns Boolean;
    
    // Development Monitoring
    entity Metrics {
        key ID : UUID;
        timestamp : DateTime;
        metricType : String;
        value : Decimal;
        metadata : LargeString;
    }
    
    entity Logs {
        key ID : UUID;
        timestamp : DateTime;
        level : String;
        message : String;
        source : String;
    }
    
    function getMetrics() returns array of Metrics;
    function getLogs(filters : String) returns array of Logs;
    function getHealthStatus() returns String;
}