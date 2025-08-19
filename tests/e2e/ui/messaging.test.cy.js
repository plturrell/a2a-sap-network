/**
 * Messaging.view.xml Component Tests
 * Test Cases: TC-AN-185 to TC-AN-200
 * Coverage: Message Queue Management, Event Streaming, Communication Protocols, Message Routing
 */

describe('Messaging.view.xml - Messaging System', () => {
  beforeEach(() => {
    cy.visit('/messaging');
    cy.viewport(1280, 720);
    
    // Wait for messaging dashboard to load
    cy.get('[data-testid="messaging-dashboard"]').should('be.visible');
  });

  describe('TC-AN-185 to TC-AN-189: Message Queue Management Tests', () => {
    it('TC-AN-185: Should verify message queue dashboard loads', () => {
      // Verify main messaging dashboard components
      cy.get('[data-testid="queue-overview"]').should('be.visible');
      cy.get('[data-testid="queue-metrics"]').should('be.visible');
      cy.get('[data-testid="active-queues"]').should('be.visible');
      cy.get('[data-testid="message-stats"]').should('be.visible');
      
      // Check queue status indicators
      cy.get('[data-testid="messaging-status"]').should('be.visible');
      cy.get('[data-testid="status-indicator"]').should('have.class', 'status-healthy');
      cy.get('[data-testid="broker-connection"]').should('contain', 'Connected');
      
      // Verify queue metrics
      cy.get('[data-testid="queue-metrics"]').within(() => {
        cy.get('[data-testid="total-queues"]').should('be.visible');
        cy.get('[data-testid="active-consumers"]').should('be.visible');
        cy.get('[data-testid="pending-messages"]').should('be.visible');
        cy.get('[data-testid="throughput-rate"]').should('be.visible');
      });
      
      // Check message statistics
      cy.get('[data-testid="message-stats"]').within(() => {
        cy.get('[data-testid="messages-sent"]').should('be.visible');
        cy.get('[data-testid="messages-received"]').should('be.visible');
        cy.get('[data-testid="messages-failed"]').should('be.visible');
        cy.get('[data-testid="avg-processing-time"]').should('be.visible');
      });
      
      // Verify real-time updates
      cy.get('[data-testid="real-time-indicator"]').should('be.visible');
      cy.get('[data-testid="last-updated"]').should('be.visible');
    });

    it('TC-AN-186: Should test queue creation and configuration', () => {
      // Open queue creation dialog
      cy.get('[data-testid="create-queue"]').click();
      
      // Verify queue creation form
      cy.get('[data-testid="queue-creation-form"]').should('be.visible');
      cy.get('[data-testid="queue-creation-form"]').within(() => {
        cy.get('[data-testid="queue-name"]').should('be.visible');
        cy.get('[data-testid="queue-type"]').should('be.visible');
        cy.get('[data-testid="durability"]').should('be.visible');
        cy.get('[data-testid="max-messages"]').should('be.visible');
        cy.get('[data-testid="message-ttl"]').should('be.visible');
        cy.get('[data-testid="dead-letter-queue"]').should('be.visible');
      });
      
      // Configure new queue
      cy.get('[data-testid="queue-name"]').type('customer-events');
      cy.get('[data-testid="queue-type"]').select('Direct');
      cy.get('[data-testid="durability"]').select('Persistent');
      cy.get('[data-testid="max-messages"]').clear().type('10000');
      cy.get('[data-testid="message-ttl"]').clear().type('3600');
      cy.get('[data-testid="enable-dlq"]').check();
      cy.get('[data-testid="dlq-name"]').type('customer-events-dlq');
      
      // Set routing configuration
      cy.get('[data-testid="routing-key"]').type('customer.#');
      cy.get('[data-testid="auto-delete"]').uncheck();
      cy.get('[data-testid="exclusive"]').uncheck();
      
      // Create queue
      cy.intercept('POST', '/api/messaging/queues').as('createQueue');
      cy.get('[data-testid="create-queue-submit"]').click();
      
      cy.wait('@createQueue');
      
      // Verify queue created
      cy.get('[data-testid="queue-created-success"]').should('be.visible');
      cy.get('[data-testid="active-queues"]').should('contain', 'customer-events');
      
      // Test queue configuration update
      cy.get('[data-testid="queue-item"]').contains('customer-events').click();
      cy.get('[data-testid="edit-queue-config"]').click();
      
      cy.get('[data-testid="max-messages"]').clear().type('20000');
      cy.get('[data-testid="save-queue-config"]').click();
      
      cy.get('[data-testid="queue-config-updated"]').should('be.visible');
    });

    it('TC-AN-187: Should verify message publishing interface', () => {
      // Navigate to message publishing
      cy.get('[data-testid="tab-publisher"]').click();
      
      // Verify publisher interface
      cy.get('[data-testid="message-publisher"]').should('be.visible');
      cy.get('[data-testid="publisher-form"]').should('be.visible');
      
      // Check publisher form fields
      cy.get('[data-testid="publisher-form"]').within(() => {
        cy.get('[data-testid="target-queue"]').should('be.visible');
        cy.get('[data-testid="routing-key"]').should('be.visible');
        cy.get('[data-testid="message-content"]').should('be.visible');
        cy.get('[data-testid="content-type"]').should('be.visible');
        cy.get('[data-testid="message-headers"]').should('be.visible');
        cy.get('[data-testid="delivery-mode"]').should('be.visible');
        cy.get('[data-testid="priority"]').should('be.visible');
      });
      
      // Compose and send message
      cy.get('[data-testid="target-queue"]').select('customer-events');
      cy.get('[data-testid="routing-key"]').type('customer.registration');
      cy.get('[data-testid="content-type"]').select('application/json');
      
      // Add message content
      const messageContent = {
        eventType: 'customer.registered',
        customerId: 'CUST-12345',
        timestamp: new Date().toISOString(),
        data: {
          email: 'test@example.com',
          name: 'Test Customer'
        }
      };
      
      cy.get('[data-testid="message-content"]').type(JSON.stringify(messageContent, null, 2));
      
      // Set delivery options
      cy.get('[data-testid="delivery-mode"]').select('Persistent');
      cy.get('[data-testid="priority"]').select('Normal');
      
      // Add custom headers
      cy.get('[data-testid="add-header"]').click();
      cy.get('[data-testid="header-key"]').type('source');
      cy.get('[data-testid="header-value"]').type('registration-service');
      cy.get('[data-testid="save-header"]').click();
      
      // Publish message
      cy.intercept('POST', '/api/messaging/publish').as('publishMessage');
      cy.get('[data-testid="publish-message"]').click();
      
      cy.wait('@publishMessage');
      
      // Verify message published
      cy.get('[data-testid="message-published"]').should('be.visible');
      cy.get('[data-testid="message-id"]').should('not.be.empty');
      
      // Test batch message publishing
      cy.get('[data-testid="batch-publish"]').click();
      cy.get('[data-testid="upload-messages"]').selectFile('cypress/fixtures/batch-messages.json');
      
      cy.get('[data-testid="batch-preview"]').should('be.visible');
      cy.get('[data-testid="message-count"]').should('contain', 'messages');
      
      cy.get('[data-testid="start-batch-publish"]').click();
      cy.get('[data-testid="batch-progress"]').should('be.visible');
      cy.get('[data-testid="batch-complete"]').should('be.visible');
    });

    it('TC-AN-188: Should test message consumption monitoring', () => {
      // Navigate to consumers tab
      cy.get('[data-testid="tab-consumers"]').click();
      
      // Verify consumers interface
      cy.get('[data-testid="consumers-dashboard"]').should('be.visible');
      cy.get('[data-testid="active-consumers"]').should('be.visible');
      
      // Check consumers table
      cy.get('[data-testid="consumers-table"]').within(() => {
        cy.get('[data-testid="header-consumer-id"]').should('contain', 'Consumer ID');
        cy.get('[data-testid="header-queue"]').should('contain', 'Queue');
        cy.get('[data-testid="header-status"]').should('contain', 'Status');
        cy.get('[data-testid="header-messages-consumed"]').should('contain', 'Consumed');
        cy.get('[data-testid="header-last-activity"]').should('contain', 'Last Activity');
      });
      
      // Verify consumer entries
      cy.get('[data-testid="consumer-row"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="consumer-row"]').first().within(() => {
        cy.get('[data-testid="consumer-id"]').should('not.be.empty');
        cy.get('[data-testid="consumer-queue"]').should('be.visible');
        cy.get('[data-testid="consumer-status"]').should('be.visible');
        cy.get('[data-testid="messages-consumed"]').should('match', /\d+/);
        cy.get('[data-testid="last-activity"]').should('be.visible');
      });
      
      // Test consumer details
      cy.get('[data-testid="consumer-row"]').first().click();
      cy.get('[data-testid="consumer-details"]').should('be.visible');
      
      cy.get('[data-testid="consumer-details"]').within(() => {
        cy.get('[data-testid="consumer-config"]').should('be.visible');
        cy.get('[data-testid="consumption-rate"]').should('be.visible');
        cy.get('[data-testid="error-rate"]').should('be.visible');
        cy.get('[data-testid="retry-count"]').should('be.visible');
        cy.get('[data-testid="processing-time"]').should('be.visible');
      });
      
      // Test consumer management
      cy.get('[data-testid="consumer-actions"]').should('be.visible');
      cy.get('[data-testid="pause-consumer"]').click();
      
      cy.get('[data-testid="pause-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-pause"]').click();
      
      cy.get('[data-testid="consumer-paused"]').should('be.visible');
      cy.get('[data-testid="consumer-status"]').should('contain', 'Paused');
      
      // Resume consumer
      cy.get('[data-testid="resume-consumer"]').click();
      cy.get('[data-testid="consumer-resumed"]').should('be.visible');
      
      // Test consumer scaling
      cy.get('[data-testid="scale-consumer"]').click();
      cy.get('[data-testid="instance-count"]').clear().type('3');
      cy.get('[data-testid="apply-scaling"]').click();
      
      cy.get('[data-testid="scaling-applied"]').should('be.visible');
    });

    it('TC-AN-189: Should verify dead letter queue handling', () => {
      // Navigate to dead letter queues
      cy.get('[data-testid="tab-dlq"]').click();
      
      // Verify DLQ interface
      cy.get('[data-testid="dlq-dashboard"]').should('be.visible');
      cy.get('[data-testid="dlq-list"]').should('be.visible');
      
      // Check DLQ metrics
      cy.get('[data-testid="dlq-metrics"]').within(() => {
        cy.get('[data-testid="total-dlq-messages"]').should('be.visible');
        cy.get('[data-testid="dlq-growth-rate"]').should('be.visible');
        cy.get('[data-testid="oldest-message"]').should('be.visible');
        cy.get('[data-testid="dlq-utilization"]').should('be.visible');
      });
      
      // Examine DLQ messages
      cy.get('[data-testid="dlq-item"]').first().click();
      cy.get('[data-testid="dlq-messages"]').should('be.visible');
      
      cy.get('[data-testid="dlq-message-row"]').first().within(() => {
        cy.get('[data-testid="message-id"]').should('be.visible');
        cy.get('[data-testid="original-queue"]').should('be.visible');
        cy.get('[data-testid="failure-reason"]').should('be.visible');
        cy.get('[data-testid="retry-count"]').should('be.visible');
        cy.get('[data-testid="dlq-timestamp"]').should('be.visible');
      });
      
      // Test message details
      cy.get('[data-testid="dlq-message-row"]').first().click();
      cy.get('[data-testid="message-details-modal"]').should('be.visible');
      
      cy.get('[data-testid="message-details-modal"]').within(() => {
        cy.get('[data-testid="message-content"]').should('be.visible');
        cy.get('[data-testid="message-headers"]').should('be.visible');
        cy.get('[data-testid="error-details"]').should('be.visible');
        cy.get('[data-testid="retry-history"]').should('be.visible');
      });
      
      // Test message reprocessing
      cy.get('[data-testid="reprocess-message"]').click();
      cy.get('[data-testid="reprocess-options"]').within(() => {
        cy.get('[data-testid="target-queue"]').select('customer-events');
        cy.get('[data-testid="modify-content"]').check();
        cy.get('[data-testid="reset-retry-count"]').check();
      });
      
      cy.get('[data-testid="confirm-reprocess"]').click();
      cy.get('[data-testid="message-reprocessed"]').should('be.visible');
      
      // Test bulk DLQ operations
      cy.get('[data-testid="dlq-message-row"]').each(($row, index) => {
        if (index < 3) {
          cy.wrap($row).find('[data-testid="select-message"]').check();
        }
      });
      
      cy.get('[data-testid="bulk-dlq-actions"]').select('Purge Selected');
      cy.get('[data-testid="execute-bulk-dlq"]').click();
      
      cy.get('[data-testid="bulk-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-bulk-dlq"]').click();
      cy.get('[data-testid="bulk-dlq-complete"]').should('be.visible');
    });
  });

  describe('TC-AN-190 to TC-AN-194: Event Streaming Tests', () => {
    it('TC-AN-190: Should test event stream configuration', () => {
      // Navigate to event streams
      cy.get('[data-testid="tab-streams"]').click();
      
      // Verify event streaming interface
      cy.get('[data-testid="event-streams"]').should('be.visible');
      cy.get('[data-testid="stream-topology"]').should('be.visible');
      
      // Check existing streams
      cy.get('[data-testid="active-streams"]').should('be.visible');
      cy.get('[data-testid="stream-metrics"]').should('be.visible');
      
      // Create new event stream
      cy.get('[data-testid="create-stream"]').click();
      
      cy.get('[data-testid="stream-config-form"]').within(() => {
        cy.get('[data-testid="stream-name"]').type('user-activity-stream');
        cy.get('[data-testid="stream-type"]').select('Kafka');
        cy.get('[data-testid="partition-count"]').clear().type('6');
        cy.get('[data-testid="replication-factor"]').clear().type('3');
        cy.get('[data-testid="retention-period"]').clear().type('168'); // 7 days in hours
        cy.get('[data-testid="compression-type"]').select('gzip');
      });
      
      // Configure stream schema
      cy.get('[data-testid="schema-registry"]').check();
      cy.get('[data-testid="schema-format"]').select('Avro');
      
      const avroSchema = {
        type: 'record',
        name: 'UserActivity',
        fields: [
          { name: 'userId', type: 'string' },
          { name: 'action', type: 'string' },
          { name: 'timestamp', type: 'long' }
        ]
      };
      
      cy.get('[data-testid="schema-definition"]').type(JSON.stringify(avroSchema, null, 2));
      
      // Set stream policies
      cy.get('[data-testid="stream-policies"]').within(() => {
        cy.get('[data-testid="cleanup-policy"]').select('delete');
        cy.get('[data-testid="min-insync-replicas"]').clear().type('2');
        cy.get('[data-testid="max-message-size"]').clear().type('1048576'); // 1MB
      });
      
      // Create stream
      cy.intercept('POST', '/api/messaging/streams').as('createStream');
      cy.get('[data-testid="create-stream-submit"]').click();
      
      cy.wait('@createStream');
      
      // Verify stream created
      cy.get('[data-testid="stream-created"]').should('be.visible');
      cy.get('[data-testid="active-streams"]').should('contain', 'user-activity-stream');
      
      // Test stream monitoring
      cy.get('[data-testid="stream-item"]').contains('user-activity-stream').click();
      cy.get('[data-testid="stream-details"]').should('be.visible');
      
      cy.get('[data-testid="stream-details"]').within(() => {
        cy.get('[data-testid="partition-distribution"]').should('be.visible');
        cy.get('[data-testid="throughput-metrics"]').should('be.visible');
        cy.get('[data-testid="consumer-lag"]').should('be.visible');
        cy.get('[data-testid="offset-status"]').should('be.visible');
      });
    });

    it('TC-AN-191: Should verify real-time event monitoring', () => {
      // Navigate to event monitoring
      cy.get('[data-testid="tab-event-monitor"]').click();
      
      // Verify real-time monitoring interface
      cy.get('[data-testid="real-time-monitor"]').should('be.visible');
      cy.get('[data-testid="event-stream-display"]').should('be.visible');
      
      // Check monitoring controls
      cy.get('[data-testid="monitor-controls"]').within(() => {
        cy.get('[data-testid="stream-selector"]').should('be.visible');
        cy.get('[data-testid="filter-controls"]').should('be.visible');
        cy.get('[data-testid="playback-controls"]').should('be.visible');
        cy.get('[data-testid="display-options"]').should('be.visible');
      });
      
      // Select stream to monitor
      cy.get('[data-testid="stream-selector"]').select('user-activity-stream');
      
      // Configure event filters
      cy.get('[data-testid="add-filter"]').click();
      cy.get('[data-testid="filter-field"]').select('action');
      cy.get('[data-testid="filter-operator"]').select('equals');
      cy.get('[data-testid="filter-value"]').type('login');
      cy.get('[data-testid="apply-filter"]').click();
      
      // Start real-time monitoring
      cy.get('[data-testid="start-monitoring"]').click();
      cy.get('[data-testid="monitoring-active"]').should('be.visible');
      
      // Verify event display
      cy.get('[data-testid="event-list"]').should('be.visible');
      cy.get('[data-testid="event-item"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="event-item"]').first().within(() => {
        cy.get('[data-testid="event-timestamp"]').should('be.visible');
        cy.get('[data-testid="event-partition"]').should('be.visible');
        cy.get('[data-testid="event-offset"]').should('be.visible');
        cy.get('[data-testid="event-key"]').should('be.visible');
        cy.get('[data-testid="event-payload"]').should('be.visible');
      });
      
      // Test event details
      cy.get('[data-testid="event-item"]').first().click();
      cy.get('[data-testid="event-details-panel"]').should('be.visible');
      
      cy.get('[data-testid="event-details-panel"]').within(() => {
        cy.get('[data-testid="raw-payload"]').should('be.visible');
        cy.get('[data-testid="formatted-payload"]').should('be.visible');
        cy.get('[data-testid="event-headers"]').should('be.visible');
        cy.get('[data-testid="schema-validation"]').should('be.visible');
      });
      
      // Test event search
      cy.get('[data-testid="event-search"]').type('userId:12345');
      cy.get('[data-testid="execute-search"]').click();
      
      cy.get('[data-testid="search-results"]').should('be.visible');
      cy.get('[data-testid="result-count"]').should('be.visible');
      
      // Test playback controls
      cy.get('[data-testid="pause-monitoring"]').click();
      cy.get('[data-testid="monitoring-paused"]').should('be.visible');
      
      cy.get('[data-testid="resume-monitoring"]').click();
      cy.get('[data-testid="monitoring-active"]').should('be.visible');
    });

    it('TC-AN-192: Should test stream processing pipelines', () => {
      // Navigate to stream processing
      cy.get('[data-testid="tab-processing"]').click();
      
      // Verify stream processing interface
      cy.get('[data-testid="stream-processing"]').should('be.visible');
      cy.get('[data-testid="processing-pipelines"]').should('be.visible');
      
      // Create new processing pipeline
      cy.get('[data-testid="create-pipeline"]').click();
      
      cy.get('[data-testid="pipeline-builder"]').should('be.visible');
      cy.get('[data-testid="pipeline-canvas"]').should('be.visible');
      cy.get('[data-testid="processor-palette"]').should('be.visible');
      
      // Configure pipeline source
      cy.get('[data-testid="add-source"]').click();
      cy.get('[data-testid="source-type"]').select('Stream');
      cy.get('[data-testid="source-stream"]').select('user-activity-stream');
      cy.get('[data-testid="add-source-confirm"]').click();
      
      // Add transformation processor
      cy.get('[data-testid="processor-transform"]').drag('[data-testid="pipeline-canvas"]');
      
      cy.get('[data-testid="transform-config"]').within(() => {
        cy.get('[data-testid="transform-type"]').select('Map');
        cy.get('[data-testid="transform-expression"]').type('$.userId + ":" + $.action');
        cy.get('[data-testid="output-field"]').type('userAction');
      });
      
      cy.get('[data-testid="save-transform"]').click();
      
      // Add filter processor
      cy.get('[data-testid="processor-filter"]').drag('[data-testid="pipeline-canvas"]');
      
      cy.get('[data-testid="filter-config"]').within(() => {
        cy.get('[data-testid="filter-condition"]').type('$.action == "purchase"');
        cy.get('[data-testid="filter-description"]').type('Filter for purchase events');
      });
      
      cy.get('[data-testid="save-filter"]').click();
      
      // Add output sink
      cy.get('[data-testid="add-sink"]').click();
      cy.get('[data-testid="sink-type"]').select('Database');
      cy.get('[data-testid="sink-table"]').type('user_purchases');
      cy.get('[data-testid="add-sink-confirm"]').click();
      
      // Connect pipeline components
      cy.get('[data-testid="source-output"]').drag('[data-testid="transform-input"]');
      cy.get('[data-testid="transform-output"]').drag('[data-testid="filter-input"]');
      cy.get('[data-testid="filter-output"]').drag('[data-testid="sink-input"]');
      
      // Configure pipeline settings
      cy.get('[data-testid="pipeline-settings"]').click();
      cy.get('[data-testid="pipeline-name"]').type('purchase-events-pipeline');
      cy.get('[data-testid="parallelism"]').clear().type('4');
      cy.get('[data-testid="checkpoint-interval"]').clear().type('60000');
      
      // Deploy pipeline
      cy.intercept('POST', '/api/messaging/pipelines/deploy').as('deployPipeline');
      cy.get('[data-testid="deploy-pipeline"]').click();
      
      cy.wait('@deployPipeline');
      
      // Verify pipeline deployed
      cy.get('[data-testid="pipeline-deployed"]').should('be.visible');
      cy.get('[data-testid="pipeline-status"]').should('contain', 'Running');
      
      // Monitor pipeline execution
      cy.get('[data-testid="pipeline-metrics"]').should('be.visible');
      cy.get('[data-testid="throughput-chart"]').should('be.visible');
      cy.get('[data-testid="latency-metrics"]').should('be.visible');
    });

    it('TC-AN-193: Should verify event replay functionality', () => {
      // Navigate to event replay
      cy.get('[data-testid="tab-replay"]').click();
      
      // Verify replay interface
      cy.get('[data-testid="event-replay"]').should('be.visible');
      cy.get('[data-testid="replay-configuration"]').should('be.visible');
      
      // Configure replay parameters
      cy.get('[data-testid="replay-configuration"]').within(() => {
        cy.get('[data-testid="source-stream"]').select('user-activity-stream');
        cy.get('[data-testid="target-stream"]').select('user-activity-replay');
        cy.get('[data-testid="start-timestamp"]').type('2024-01-01T00:00:00Z');
        cy.get('[data-testid="end-timestamp"]').type('2024-01-02T00:00:00Z');
        cy.get('[data-testid="replay-speed"]').select('2x');
      });
      
      // Set replay filters
      cy.get('[data-testid="replay-filters"]').within(() => {
        cy.get('[data-testid="add-replay-filter"]').click();
        cy.get('[data-testid="filter-field"]').select('userId');
        cy.get('[data-testid="filter-operator"]').select('in');
        cy.get('[data-testid="filter-values"]').type('12345,67890,11111');
        cy.get('[data-testid="save-replay-filter"]').click();
      });
      
      // Configure replay options
      cy.get('[data-testid="replay-options"]').within(() => {
        cy.get('[data-testid="preserve-order"]').check();
        cy.get('[data-testid="skip-duplicates"]').check();
        cy.get('[data-testid="batch-size"]').clear().type('1000');
        cy.get('[data-testid="max-parallelism"]').clear().type('8');
      });
      
      // Start event replay
      cy.intercept('POST', '/api/messaging/replay/start').as('startReplay');
      cy.get('[data-testid="start-replay"]').click();
      
      cy.wait('@startReplay');
      
      // Monitor replay progress
      cy.get('[data-testid="replay-progress"]').should('be.visible');
      cy.get('[data-testid="progress-bar"]').should('be.visible');
      cy.get('[data-testid="events-replayed"]').should('be.visible');
      cy.get('[data-testid="replay-rate"]').should('be.visible');
      
      // Test replay controls
      cy.get('[data-testid="pause-replay"]').click();
      cy.get('[data-testid="replay-paused"]').should('be.visible');
      
      cy.get('[data-testid="resume-replay"]').click();
      cy.get('[data-testid="replay-resumed"]').should('be.visible');
      
      // Verify replay completion
      cy.get('[data-testid="replay-complete"]', { timeout: 60000 }).should('be.visible');
      cy.get('[data-testid="replay-summary"]').should('be.visible');
      
      cy.get('[data-testid="replay-summary"]').within(() => {
        cy.get('[data-testid="total-events-replayed"]').should('be.visible');
        cy.get('[data-testid="replay-duration"]').should('be.visible');
        cy.get('[data-testid="average-replay-rate"]').should('be.visible');
        cy.get('[data-testid="errors-encountered"]').should('be.visible');
      });
    });

    it('TC-AN-194: Should test stream analytics and metrics', () => {
      // Navigate to stream analytics
      cy.get('[data-testid="tab-analytics"]').click();
      
      // Verify analytics dashboard
      cy.get('[data-testid="stream-analytics"]').should('be.visible');
      cy.get('[data-testid="analytics-overview"]').should('be.visible');
      
      // Check stream performance metrics
      cy.get('[data-testid="performance-metrics"]').within(() => {
        cy.get('[data-testid="messages-per-second"]').should('be.visible');
        cy.get('[data-testid="bytes-per-second"]').should('be.visible');
        cy.get('[data-testid="average-latency"]').should('be.visible');
        cy.get('[data-testid="error-rate"]').should('be.visible');
      });
      
      // Verify throughput charts
      cy.get('[data-testid="throughput-charts"]').should('be.visible');
      cy.get('[data-testid="messages-throughput"]').should('be.visible');
      cy.get('[data-testid="bytes-throughput"]').should('be.visible');
      
      // Test chart interactions
      cy.get('[data-testid="messages-throughput"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
      
      // Check consumer analytics
      cy.get('[data-testid="consumer-analytics"]').should('be.visible');
      cy.get('[data-testid="consumer-lag-chart"]').should('be.visible');
      cy.get('[data-testid="consumer-throughput"]').should('be.visible');
      
      // Test time range selection
      cy.get('[data-testid="analytics-timerange"]').select('Last 24 hours');
      cy.get('[data-testid="analytics-loading"]').should('be.visible');
      cy.get('[data-testid="analytics-loading"]').should('not.exist');
      
      // Check partition analytics
      cy.get('[data-testid="partition-analytics"]').click();
      
      cy.get('[data-testid="partition-distribution"]').should('be.visible');
      cy.get('[data-testid="partition-metrics"]').should('be.visible');
      
      cy.get('[data-testid="partition-item"]').first().within(() => {
        cy.get('[data-testid="partition-id"]').should('be.visible');
        cy.get('[data-testid="partition-size"]').should('be.visible');
        cy.get('[data-testid="partition-messages"]').should('be.visible');
        cy.get('[data-testid="partition-leaders"]').should('be.visible');
      });
      
      // Test analytics export
      cy.get('[data-testid="export-analytics"]').click();
      
      cy.get('[data-testid="export-options"]').within(() => {
        cy.get('[data-testid="export-format-csv"]').click();
        cy.get('[data-testid="include-metrics"]').check();
        cy.get('[data-testid="include-charts"]').check();
      });
      
      cy.get('[data-testid="generate-analytics-export"]').click();
      cy.get('[data-testid="export-complete"]').should('be.visible');
      
      // Test alerting configuration
      cy.get('[data-testid="configure-alerts"]').click();
      
      cy.get('[data-testid="alert-rules"]').within(() => {
        cy.get('[data-testid="add-alert-rule"]').click();
        cy.get('[data-testid="metric-type"]').select('Consumer Lag');
        cy.get('[data-testid="threshold-value"]').type('1000');
        cy.get('[data-testid="threshold-operator"]').select('greater_than');
        cy.get('[data-testid="notification-channel"]').select('email');
        cy.get('[data-testid="save-alert-rule"]').click();
      });
      
      cy.get('[data-testid="alert-rule-saved"]').should('be.visible');
    });
  });

  describe('TC-AN-195 to TC-AN-200: Protocol and Integration Tests', () => {
    it('TC-AN-195: Should test messaging protocol configuration', () => {
      // Navigate to protocol settings
      cy.get('[data-testid="tab-protocols"]').click();
      
      // Verify protocol configuration interface
      cy.get('[data-testid="protocol-settings"]').should('be.visible');
      cy.get('[data-testid="supported-protocols"]').should('be.visible');
      
      // Check supported protocols
      const protocols = ['AMQP', 'MQTT', 'STOMP', 'WebSocket', 'HTTP/REST'];
      protocols.forEach(protocol => {
        cy.get('[data-testid="supported-protocols"]').should('contain', protocol);
      });
      
      // Configure AMQP protocol
      cy.get('[data-testid="protocol-amqp"]').click();
      
      cy.get('[data-testid="amqp-config"]').within(() => {
        cy.get('[data-testid="amqp-port"]').clear().type('5672');
        cy.get('[data-testid="amqp-ssl-port"]').clear().type('5671');
        cy.get('[data-testid="enable-ssl"]').check();
        cy.get('[data-testid="ssl-cert-path"]').type('/etc/ssl/certs/amqp.pem');
        cy.get('[data-testid="max-connections"]').clear().type('1000');
        cy.get('[data-testid="heartbeat-interval"]').clear().type('60');
      });
      
      cy.get('[data-testid="save-amqp-config"]').click();
      cy.get('[data-testid="amqp-config-saved"]').should('be.visible');
      
      // Configure MQTT protocol
      cy.get('[data-testid="protocol-mqtt"]').click();
      
      cy.get('[data-testid="mqtt-config"]').within(() => {
        cy.get('[data-testid="mqtt-port"]').clear().type('1883');
        cy.get('[data-testid="mqtt-ssl-port"]').clear().type('8883');
        cy.get('[data-testid="qos-levels"]').should('be.visible');
        cy.get('[data-testid="qos-0"]').check();
        cy.get('[data-testid="qos-1"]').check();
        cy.get('[data-testid="qos-2"]').check();
        cy.get('[data-testid="retain-messages"]').check();
        cy.get('[data-testid="clean-session"]').check();
      });
      
      cy.get('[data-testid="save-mqtt-config"]').click();
      cy.get('[data-testid="mqtt-config-saved"]').should('be.visible');
      
      // Test protocol status
      cy.get('[data-testid="protocol-status"]').should('be.visible');
      cy.get('[data-testid="protocol-item"]').each(($item) => {
        cy.wrap($item).within(() => {
          cy.get('[data-testid="protocol-name"]').should('be.visible');
          cy.get('[data-testid="protocol-status"]').should('be.visible');
          cy.get('[data-testid="active-connections"]').should('be.visible');
          cy.get('[data-testid="protocol-version"]').should('be.visible');
        });
      });
      
      // Test protocol enable/disable
      cy.get('[data-testid="protocol-item"]').first().find('[data-testid="toggle-protocol"]').click();
      cy.get('[data-testid="protocol-toggle-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-toggle"]').click();
      
      cy.get('[data-testid="protocol-toggled"]').should('be.visible');
    });

    it('TC-AN-196: Should verify external system integration', () => {
      // Navigate to integrations
      cy.get('[data-testid="tab-integrations"]').click();
      
      // Verify integrations interface
      cy.get('[data-testid="system-integrations"]').should('be.visible');
      cy.get('[data-testid="available-integrations"]').should('be.visible');
      
      // Check available integration types
      const integrationTypes = ['SAP', 'Salesforce', 'AWS SQS', 'Apache Kafka', 'Redis'];
      integrationTypes.forEach(type => {
        cy.get('[data-testid="available-integrations"]').should('contain', type);
      });
      
      // Configure SAP integration
      cy.get('[data-testid="integration-sap"]').click();
      cy.get('[data-testid="configure-integration"]').click();
      
      cy.get('[data-testid="sap-integration-form"]').within(() => {
        cy.get('[data-testid="sap-host"]').type('sap-system.company.com');
        cy.get('[data-testid="sap-system-number"]').type('00');
        cy.get('[data-testid="sap-client"]').type('100');
        cy.get('[data-testid="sap-username"]').type('integration-user');
        cy.get('[data-testid="sap-password"]').type('secure-password');
        cy.get('[data-testid="connection-pool-size"]').clear().type('10');
      });
      
      // Test connection
      cy.get('[data-testid="test-sap-connection"]').click();
      cy.get('[data-testid="connection-testing"]').should('be.visible');
      cy.get('[data-testid="connection-success"]').should('be.visible');
      
      // Configure message mapping
      cy.get('[data-testid="message-mapping"]').click();
      
      cy.get('[data-testid="mapping-config"]').within(() => {
        cy.get('[data-testid="source-format"]').select('IDoc');
        cy.get('[data-testid="target-format"]').select('JSON');
        cy.get('[data-testid="transformation-template"]').type('{\n  "customer": "$.CUSTOMER_ID",\n  "amount": "$.AMOUNT"\n}');
      });
      
      cy.get('[data-testid="save-mapping"]').click();
      
      // Save integration configuration
      cy.intercept('POST', '/api/messaging/integrations/sap').as('saveSapIntegration');
      cy.get('[data-testid="save-integration"]').click();
      
      cy.wait('@saveSapIntegration');
      cy.get('[data-testid="integration-saved"]').should('be.visible');
      
      // Configure AWS SQS integration
      cy.get('[data-testid="integration-aws-sqs"]').click();
      cy.get('[data-testid="configure-integration"]').click();
      
      cy.get('[data-testid="aws-sqs-form"]').within(() => {
        cy.get('[data-testid="aws-region"]').select('us-east-1');
        cy.get('[data-testid="aws-access-key"]').type('AKIA...');
        cy.get('[data-testid="aws-secret-key"]').type('secret-key');
        cy.get('[data-testid="queue-url"]').type('https://sqs.us-east-1.amazonaws.com/123456789012/my-queue');
        cy.get('[data-testid="polling-interval"]').clear().type('30');
      });
      
      cy.get('[data-testid="test-aws-connection"]').click();
      cy.get('[data-testid="aws-connection-success"]').should('be.visible');
      
      cy.get('[data-testid="save-integration"]').click();
      cy.get('[data-testid="integration-saved"]').should('be.visible');
      
      // Monitor integration status
      cy.get('[data-testid="integration-monitoring"]').should('be.visible');
      cy.get('[data-testid="active-integrations"]').should('contain', 'SAP');
      cy.get('[data-testid="active-integrations"]').should('contain', 'AWS SQS');
    });

    it('TC-AN-197: Should test message routing configuration', () => {
      // Navigate to routing
      cy.get('[data-testid="tab-routing"]').click();
      
      // Verify routing interface
      cy.get('[data-testid="message-routing"]').should('be.visible');
      cy.get('[data-testid="routing-rules"]').should('be.visible');
      
      // Create new routing rule
      cy.get('[data-testid="create-routing-rule"]').click();
      
      cy.get('[data-testid="routing-rule-form"]').within(() => {
        cy.get('[data-testid="rule-name"]').type('Customer Events Routing');
        cy.get('[data-testid="rule-priority"]').clear().type('100');
        cy.get('[data-testid="rule-description"]').type('Route customer events based on event type');
      });
      
      // Configure routing conditions
      cy.get('[data-testid="routing-conditions"]').within(() => {
        cy.get('[data-testid="add-condition"]').click();
        cy.get('[data-testid="condition-field"]').select('eventType');
        cy.get('[data-testid="condition-operator"]').select('equals');
        cy.get('[data-testid="condition-value"]').type('customer.registration');
        cy.get('[data-testid="save-condition"]').click();
      });
      
      // Configure routing actions
      cy.get('[data-testid="routing-actions"]').within(() => {
        cy.get('[data-testid="add-action"]').click();
        cy.get('[data-testid="action-type"]').select('Route to Queue');
        cy.get('[data-testid="target-queue"]').select('customer-registration-queue');
        cy.get('[data-testid="routing-key"]').type('customer.new');
        cy.get('[data-testid="save-action"]').click();
      });
      
      // Add transformation action
      cy.get('[data-testid="add-action"]').click();
      cy.get('[data-testid="action-type"]').select('Transform Message');
      cy.get('[data-testid="transformation-script"]').type('message.timestamp = Date.now(); return message;');
      cy.get('[data-testid="save-action"]').click();
      
      // Save routing rule
      cy.intercept('POST', '/api/messaging/routing/rules').as('saveRoutingRule');
      cy.get('[data-testid="save-routing-rule"]').click();
      
      cy.wait('@saveRoutingRule');
      cy.get('[data-testid="routing-rule-saved"]').should('be.visible');
      
      // Test routing rule
      cy.get('[data-testid="test-routing-rule"]').click();
      
      cy.get('[data-testid="test-message"]').type(JSON.stringify({
        eventType: 'customer.registration',
        customerId: 'CUST-12345',
        data: { email: 'test@example.com' }
      }));
      
      cy.get('[data-testid="execute-routing-test"]').click();
      
      // Verify routing test results
      cy.get('[data-testid="routing-test-results"]').should('be.visible');
      cy.get('[data-testid="matched-rules"]').should('contain', 'Customer Events Routing');
      cy.get('[data-testid="applied-actions"]').should('contain', 'Route to Queue');
      cy.get('[data-testid="target-destination"]').should('contain', 'customer-registration-queue');
      
      // Create conditional routing rule
      cy.get('[data-testid="create-routing-rule"]').click();
      
      cy.get('[data-testid="rule-name"]').type('Priority Message Routing');
      cy.get('[data-testid="rule-priority"]').clear().type('200');
      
      // Add multiple conditions
      cy.get('[data-testid="add-condition"]').click();
      cy.get('[data-testid="condition-field"]').select('priority');
      cy.get('[data-testid="condition-operator"]').select('greater_than');
      cy.get('[data-testid="condition-value"]').type('5');
      cy.get('[data-testid="save-condition"]').click();
      
      cy.get('[data-testid="add-condition"]').click();
      cy.get('[data-testid="condition-field"]').select('eventType');
      cy.get('[data-testid="condition-operator"]').select('contains');
      cy.get('[data-testid="condition-value"]').type('urgent');
      cy.get('[data-testid="condition-logic"]').select('AND');
      cy.get('[data-testid="save-condition"]').click();
      
      // Configure priority routing
      cy.get('[data-testid="add-action"]').click();
      cy.get('[data-testid="action-type"]').select('Route to Queue');
      cy.get('[data-testid="target-queue"]').select('high-priority-queue');
      cy.get('[data-testid="save-action"]').click();
      
      cy.get('[data-testid="save-routing-rule"]').click();
      cy.get('[data-testid="routing-rule-saved"]').should('be.visible');
    });

    it('TC-AN-198: Should verify message transformation features', () => {
      // Navigate to transformations
      cy.get('[data-testid="tab-transformations"]').click();
      
      // Verify transformation interface
      cy.get('[data-testid="message-transformations"]').should('be.visible');
      cy.get('[data-testid="transformation-templates"]').should('be.visible');
      
      // Create new transformation template
      cy.get('[data-testid="create-transformation"]').click();
      
      cy.get('[data-testid="transformation-form"]').within(() => {
        cy.get('[data-testid="template-name"]').type('Customer Data Normalization');
        cy.get('[data-testid="template-description"]').type('Normalize customer data format');
        cy.get('[data-testid="transformation-type"]').select('JavaScript');
      });
      
      // Define transformation script
      const transformationScript = `
        function transform(message) {
          return {
            id: message.customerId,
            name: {
              first: message.firstName,
              last: message.lastName
            },
            contact: {
              email: message.email.toLowerCase(),
              phone: message.phone.replace(/\\D/g, '')
            },
            timestamp: new Date().toISOString()
          };
        }
      `;
      
      cy.get('[data-testid="transformation-script"]').type(transformationScript);
      
      // Test transformation
      cy.get('[data-testid="test-transformation"]').click();
      
      const testInput = {
        customerId: 'CUST-12345',
        firstName: 'John',
        lastName: 'Doe',
        email: 'JOHN.DOE@EXAMPLE.COM',
        phone: '(555) 123-4567'
      };
      
      cy.get('[data-testid="test-input"]').type(JSON.stringify(testInput, null, 2));
      cy.get('[data-testid="execute-transformation"]').click();
      
      // Verify transformation output
      cy.get('[data-testid="transformation-output"]').should('be.visible');
      cy.get('[data-testid="transformed-message"]').should('contain', 'john.doe@example.com');
      cy.get('[data-testid="transformed-message"]').should('contain', '5551234567');
      
      // Save transformation template
      cy.intercept('POST', '/api/messaging/transformations').as('saveTransformation');
      cy.get('[data-testid="save-transformation"]').click();
      
      cy.wait('@saveTransformation');
      cy.get('[data-testid="transformation-saved"]').should('be.visible');
      
      // Create JSON-to-XML transformation
      cy.get('[data-testid="create-transformation"]').click();
      
      cy.get('[data-testid="template-name"]').type('JSON to XML Converter');
      cy.get('[data-testid="transformation-type"]').select('XSLT');
      
      const xsltTemplate = `
        <xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
          <xsl:template match="/">
            <customer>
              <id><xsl:value-of select="customerId"/></id>
              <name><xsl:value-of select="firstName"/> <xsl:value-of select="lastName"/></name>
              <email><xsl:value-of select="email"/></email>
            </customer>
          </xsl:template>
        </xsl:stylesheet>
      `;
      
      cy.get('[data-testid="transformation-script"]').type(xsltTemplate);
      cy.get('[data-testid="save-transformation"]').click();
      cy.get('[data-testid="transformation-saved"]').should('be.visible');
      
      // Test batch transformation
      cy.get('[data-testid="batch-transformation"]').click();
      
      cy.get('[data-testid="batch-config"]').within(() => {
        cy.get('[data-testid="source-queue"]').select('raw-customer-data');
        cy.get('[data-testid="target-queue"]').select('normalized-customer-data');
        cy.get('[data-testid="transformation-template"]').select('Customer Data Normalization');
        cy.get('[data-testid="batch-size"]').clear().type('100');
        cy.get('[data-testid="error-handling"]').select('Skip and Log');
      });
      
      cy.get('[data-testid="start-batch-transformation"]').click();
      cy.get('[data-testid="batch-transformation-started"]').should('be.visible');
      
      // Monitor batch progress
      cy.get('[data-testid="batch-progress"]').should('be.visible');
      cy.get('[data-testid="messages-processed"]').should('be.visible');
      cy.get('[data-testid="success-rate"]').should('be.visible');
    });

    it('TC-AN-199: Should test message broker clustering', () => {
      // Navigate to clustering
      cy.get('[data-testid="tab-clustering"]').click();
      
      // Verify clustering interface
      cy.get('[data-testid="broker-clustering"]').should('be.visible');
      cy.get('[data-testid="cluster-topology"]').should('be.visible');
      
      // Check cluster status
      cy.get('[data-testid="cluster-status"]').within(() => {
        cy.get('[data-testid="cluster-health"]').should('be.visible');
        cy.get('[data-testid="node-count"]').should('be.visible');
        cy.get('[data-testid="leader-node"]').should('be.visible');
        cy.get('[data-testid="partition-distribution"]').should('be.visible');
      });
      
      // Verify cluster nodes
      cy.get('[data-testid="cluster-nodes"]').should('be.visible');
      cy.get('[data-testid="node-item"]').should('have.length.greaterThan', 1);
      
      cy.get('[data-testid="node-item"]').first().within(() => {
        cy.get('[data-testid="node-id"]').should('be.visible');
        cy.get('[data-testid="node-status"]').should('contain', 'Active');
        cy.get('[data-testid="node-role"]').should('be.visible');
        cy.get('[data-testid="node-load"]').should('be.visible');
        cy.get('[data-testid="node-partitions"]').should('be.visible');
      });
      
      // Test node management
      cy.get('[data-testid="node-item"]').first().click();
      cy.get('[data-testid="node-details"]').should('be.visible');
      
      cy.get('[data-testid="node-details"]').within(() => {
        cy.get('[data-testid="node-metrics"]').should('be.visible');
        cy.get('[data-testid="hosted-partitions"]').should('be.visible');
        cy.get('[data-testid="replication-status"]').should('be.visible');
        cy.get('[data-testid="network-connections"]').should('be.visible');
      });
      
      // Test cluster rebalancing
      cy.get('[data-testid="rebalance-cluster"]').click();
      
      cy.get('[data-testid="rebalance-options"]').within(() => {
        cy.get('[data-testid="rebalance-strategy"]').select('Round Robin');
        cy.get('[data-testid="exclude-brokers"]').should('be.visible');
        cy.get('[data-testid="throttle-rate"]').clear().type('10000000'); // 10MB/s
        cy.get('[data-testid="concurrent-movements"]').clear().type('5');
      });
      
      cy.intercept('POST', '/api/messaging/cluster/rebalance').as('startRebalance');
      cy.get('[data-testid="start-rebalance"]').click();
      
      cy.wait('@startRebalance');
      
      // Monitor rebalancing progress
      cy.get('[data-testid="rebalance-progress"]').should('be.visible');
      cy.get('[data-testid="partition-movements"]').should('be.visible');
      cy.get('[data-testid="completion-estimate"]').should('be.visible');
      
      // Test cluster configuration
      cy.get('[data-testid="cluster-config"]').click();
      
      cy.get('[data-testid="cluster-settings"]').within(() => {
        cy.get('[data-testid="replication-factor"]').clear().type('3');
        cy.get('[data-testid="min-insync-replicas"]').clear().type('2');
        cy.get('[data-testid="leader-election-timeout"]').clear().type('30000');
        cy.get('[data-testid="heartbeat-interval"]').clear().type('3000');
      });
      
      cy.get('[data-testid="save-cluster-config"]').click();
      cy.get('[data-testid="cluster-config-saved"]').should('be.visible');
      
      // Test failover simulation
      cy.get('[data-testid="simulate-failover"]').click();
      cy.get('[data-testid="select-node-to-fail"]').select('broker-2');
      cy.get('[data-testid="failover-type"]').select('Graceful Shutdown');
      
      cy.get('[data-testid="start-failover-simulation"]').click();
      cy.get('[data-testid="failover-in-progress"]').should('be.visible');
      
      // Verify failover handling
      cy.get('[data-testid="failover-results"]', { timeout: 30000 }).should('be.visible');
      cy.get('[data-testid="leader-reelection"]').should('be.visible');
      cy.get('[data-testid="partition-reassignment"]').should('be.visible');
      cy.get('[data-testid="recovery-time"]').should('be.visible');
    });

    it('TC-AN-200: Should verify messaging security features', () => {
      // Navigate to messaging security
      cy.get('[data-testid="tab-security"]').click();
      
      // Verify security interface
      cy.get('[data-testid="messaging-security"]').should('be.visible');
      cy.get('[data-testid="security-overview"]').should('be.visible');
      
      // Check security status
      cy.get('[data-testid="security-status"]').within(() => {
        cy.get('[data-testid="ssl-enabled"]').should('be.visible');
        cy.get('[data-testid="authentication-enabled"]').should('be.visible');
        cy.get('[data-testid="authorization-enabled"]').should('be.visible');
        cy.get('[data-testid="encryption-status"]').should('be.visible');
      });
      
      // Configure SSL/TLS
      cy.get('[data-testid="configure-ssl"]').click();
      
      cy.get('[data-testid="ssl-configuration"]').within(() => {
        cy.get('[data-testid="ssl-enabled"]').check();
        cy.get('[data-testid="ssl-certificate"]').type('/etc/ssl/certs/messaging.crt');
        cy.get('[data-testid="ssl-private-key"]').type('/etc/ssl/private/messaging.key');
        cy.get('[data-testid="ssl-ca-certificate"]').type('/etc/ssl/certs/ca.crt');
        cy.get('[data-testid="ssl-protocols"]').select(['TLSv1.2', 'TLSv1.3']);
        cy.get('[data-testid="cipher-suites"]').select('Strong Ciphers Only');
      });
      
      cy.get('[data-testid="save-ssl-config"]').click();
      cy.get('[data-testid="ssl-config-saved"]').should('be.visible');
      
      // Configure authentication
      cy.get('[data-testid="configure-authentication"]').click();
      
      cy.get('[data-testid="auth-methods"]').within(() => {
        cy.get('[data-testid="username-password"]').check();
        cy.get('[data-testid="client-certificates"]').check();
        cy.get('[data-testid="oauth2"]').check();
        cy.get('[data-testid="ldap-integration"]').check();
      });
      
      // Configure user management
      cy.get('[data-testid="user-management"]').click();
      cy.get('[data-testid="create-messaging-user"]').click();
      
      cy.get('[data-testid="user-form"]').within(() => {
        cy.get('[data-testid="username"]').type('app-service-1');
        cy.get('[data-testid="password"]').type('secure-password-123');
        cy.get('[data-testid="user-role"]').select('Publisher');
        cy.get('[data-testid="allowed-queues"]').select(['customer-events', 'order-processing']);
      });
      
      cy.get('[data-testid="save-messaging-user"]').click();
      cy.get('[data-testid="user-created"]').should('be.visible');
      
      // Configure access control
      cy.get('[data-testid="configure-access-control"]').click();
      
      cy.get('[data-testid="access-control-rules"]').within(() => {
        cy.get('[data-testid="add-acl-rule"]').click();
        cy.get('[data-testid="resource-type"]').select('Queue');
        cy.get('[data-testid="resource-pattern"]').type('customer.*');
        cy.get('[data-testid="principal"]').select('app-service-1');
        cy.get('[data-testid="operations"]').select(['Read', 'Write']);
        cy.get('[data-testid="permission-type"]').select('Allow');
        cy.get('[data-testid="save-acl-rule"]').click();
      });
      
      cy.get('[data-testid="acl-rule-saved"]').should('be.visible');
      
      // Test message encryption
      cy.get('[data-testid="configure-encryption"]').click();
      
      cy.get('[data-testid="encryption-settings"]').within(() => {
        cy.get('[data-testid="enable-message-encryption"]').check();
        cy.get('[data-testid="encryption-algorithm"]').select('AES-256-GCM');
        cy.get('[data-testid="key-management"]').select('External KMS');
        cy.get('[data-testid="kms-endpoint"]').type('https://kms.company.com');
        cy.get('[data-testid="master-key-id"]').type('arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012');
      });
      
      cy.get('[data-testid="save-encryption-config"]').click();
      cy.get('[data-testid="encryption-config-saved"]').should('be.visible');
      
      // Test security audit
      cy.get('[data-testid="security-audit"]').click();
      
      cy.get('[data-testid="audit-results"]').should('be.visible');
      cy.get('[data-testid="security-findings"]').should('be.visible');
      
      cy.get('[data-testid="finding-item"]').first().within(() => {
        cy.get('[data-testid="finding-severity"]').should('be.visible');
        cy.get('[data-testid="finding-description"]').should('be.visible');
        cy.get('[data-testid="remediation-advice"]').should('be.visible');
      });
      
      // Generate security report
      cy.get('[data-testid="generate-security-report"]').click();
      
      cy.get('[data-testid="report-options"]').within(() => {
        cy.get('[data-testid="include-vulnerabilities"]').check();
        cy.get('[data-testid="include-compliance"]').check();
        cy.get('[data-testid="include-recommendations"]').check();
        cy.get('[data-testid="report-format"]').select('PDF');
      });
      
      cy.get('[data-testid="generate-report"]').click();
      cy.get('[data-testid="security-report-generated"]').should('be.visible');
      
      // Test connection security validation
      cy.get('[data-testid="validate-connections"]').click();
      cy.get('[data-testid="connection-validation-progress"]').should('be.visible');
      cy.get('[data-testid="validation-complete"]').should('be.visible');
      
      cy.get('[data-testid="connection-results"]').within(() => {
        cy.get('[data-testid="secure-connections"]').should('be.visible');
        cy.get('[data-testid="insecure-connections"]').should('be.visible');
        cy.get('[data-testid="authentication-failures"]').should('be.visible');
      });
    });
  });
});