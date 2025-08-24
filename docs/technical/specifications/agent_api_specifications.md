# Agent API Specifications

## Complete OpenAPI 3.0 Specifications for A2A Agent Services

This document provides detailed API specifications for all agent services in the A2A platform, including request/response schemas, error codes, and example payloads.

---

## Agent 0 - Data Product Registration API

```yaml
openapi: 3.0.0
info:
  title: Agent 0 - Data Product Registration API
  description: Handles initial data product registration, validation, and catalog entry
  version: 1.0.0
  contact:
    name: A2A Platform Team
    email: a2a-platform@company.com

servers:
  - url: https://api.a2a-platform.com/v1
    description: Production server
  - url: https://api-staging.a2a-platform.com/v1
    description: Staging server

paths:
  /agents/agent0/register:
    post:
      summary: Register new data product
      operationId: registerDataProduct
      tags:
        - Data Product Registration
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProductRegistration'
            examples:
              financialDataProduct:
                value:
                  product_name: "Treasury Rates Dataset"
                  product_type: "financial_time_series"
                  source_system: "BLOOMBERG"
                  data_format: "JSON"
                  schema_version: "2.0"
                  metadata:
                    asset_class: "RATES"
                    frequency: "DAILY"
                    start_date: "2020-01-01"
                    end_date: "2024-12-31"
                  validation_rules:
                    - rule_type: "COMPLETENESS"
                      threshold: 0.95
                    - rule_type: "ACCURACY"
                      threshold: 0.99
      responses:
        '201':
          description: Data product successfully registered
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RegistrationResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'
        '500':
          $ref: '#/components/responses/InternalError'

  /agents/agent0/validate:
    post:
      summary: Validate data product without registration
      operationId: validateDataProduct
      tags:
        - Data Product Validation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProductValidation'
      responses:
        '200':
          description: Validation results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationResult'

  /agents/agent0/products/{productId}:
    get:
      summary: Get data product details
      operationId: getDataProduct
      tags:
        - Data Product Management
      parameters:
        - name: productId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Data product details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DataProduct'
        '404':
          $ref: '#/components/responses/NotFound'

components:
  schemas:
    DataProductRegistration:
      type: object
      required:
        - product_name
        - product_type
        - source_system
        - data_format
        - metadata
      properties:
        product_name:
          type: string
          minLength: 3
          maxLength: 255
        product_type:
          type: string
          enum: [financial_time_series, reference_data, transaction_data, market_data]
        source_system:
          type: string
          enum: [BLOOMBERG, REUTERS, INTERNAL, SAP_S4, CUSTOM]
        data_format:
          type: string
          enum: [JSON, XML, CSV, PARQUET, AVRO]
        schema_version:
          type: string
          pattern: '^\d+\.\d+$'
        metadata:
          type: object
          additionalProperties: true
        validation_rules:
          type: array
          items:
            $ref: '#/components/schemas/ValidationRule'
    
    ValidationRule:
      type: object
      required:
        - rule_type
        - threshold
      properties:
        rule_type:
          type: string
          enum: [COMPLETENESS, ACCURACY, TIMELINESS, CONSISTENCY, UNIQUENESS]
        threshold:
          type: number
          minimum: 0
          maximum: 1
        parameters:
          type: object
          additionalProperties: true
    
    RegistrationResponse:
      type: object
      properties:
        product_id:
          type: string
          format: uuid
        registration_timestamp:
          type: string
          format: date-time
        catalog_entry_id:
          type: string
        validation_score:
          type: number
          minimum: 0
          maximum: 1
        status:
          type: string
          enum: [REGISTERED, PENDING_APPROVAL, REJECTED]
        next_agent:
          type: string
          description: Next agent in the processing pipeline
```

---

## Agent 1 - Data Standardization API

```yaml
openapi: 3.0.0
info:
  title: Agent 1 - Financial Data Standardization API
  description: Standardizes financial data across different formats and conventions
  version: 1.0.0

paths:
  /agents/agent1/standardize:
    post:
      summary: Standardize financial data
      operationId: standardizeData
      tags:
        - Data Standardization
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/StandardizationRequest'
            examples:
              currencyStandardization:
                value:
                  product_id: "550e8400-e29b-41d4-a716-446655440000"
                  data_format: "CUSTOM"
                  source_data:
                    transactions:
                      - amount: 1000000
                        currency: "EUR"
                        date: "2024-01-15"
                      - amount: 2500000
                        currency: "GBP"
                        date: "2024-01-15"
                  target_format: "ISO_20022"
                  standardization_options:
                    target_currency: "USD"
                    date_format: "ISO8601"
                    decimal_places: 2
      responses:
        '200':
          description: Standardization successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StandardizationResponse'
        '400':
          $ref: '#/components/responses/BadRequest'

  /agents/agent1/mappings:
    get:
      summary: Get available standardization mappings
      operationId: getStandardizationMappings
      tags:
        - Standardization Configuration
      parameters:
        - name: source_format
          in: query
          schema:
            type: string
        - name: target_format
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Available mappings
          content:
            application/json:
              schema:
                type: object
                properties:
                  mappings:
                    type: array
                    items:
                      $ref: '#/components/schemas/StandardizationMapping'

components:
  schemas:
    StandardizationRequest:
      type: object
      required:
        - product_id
        - data_format
        - source_data
        - target_format
      properties:
        product_id:
          type: string
          format: uuid
        data_format:
          type: string
        source_data:
          type: object
          description: Raw data to be standardized
        target_format:
          type: string
          enum: [ISO_20022, FIX_5_0, SWIFT_MT, FPML, CUSTOM]
        standardization_options:
          $ref: '#/components/schemas/StandardizationOptions'
    
    StandardizationOptions:
      type: object
      properties:
        target_currency:
          type: string
          pattern: '^[A-Z]{3}$'
        date_format:
          type: string
          enum: [ISO8601, YYYYMMDD, DD/MM/YYYY, MM/DD/YYYY]
        decimal_places:
          type: integer
          minimum: 0
          maximum: 10
        timezone:
          type: string
          default: "UTC"
        entity_resolution:
          type: boolean
          default: true
    
    StandardizationResponse:
      type: object
      properties:
        standardized_data:
          type: object
          description: Data in standardized format
        transformation_report:
          $ref: '#/components/schemas/TransformationReport'
        quality_metrics:
          $ref: '#/components/schemas/QualityMetrics'
        next_agent:
          type: string
    
    TransformationReport:
      type: object
      properties:
        transformations_applied:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              original_value:
                type: string
              transformed_value:
                type: string
              transformation_type:
                type: string
        warnings:
          type: array
          items:
            type: string
        errors:
          type: array
          items:
            type: string
```

---

## Agent 2 - AI Data Preparation API

```yaml
openapi: 3.0.0
info:
  title: Agent 2 - AI Data Preparation API
  description: Prepares financial data for AI/ML processing with feature engineering
  version: 1.0.0

paths:
  /agents/agent2/prepare:
    post:
      summary: Prepare data for AI processing
      operationId: prepareDataForAI
      tags:
        - AI Data Preparation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AIPreparationRequest'
      responses:
        '200':
          description: Data prepared successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AIPreparationResponse'

  /agents/agent2/features/catalog:
    get:
      summary: Get available feature engineering templates
      operationId: getFeatureCatalog
      tags:
        - Feature Engineering
      parameters:
        - name: data_type
          in: query
          schema:
            type: string
            enum: [time_series, transaction, reference]
        - name: use_case
          in: query
          schema:
            type: string
            enum: [risk_analysis, fraud_detection, portfolio_optimization, credit_scoring]
      responses:
        '200':
          description: Feature catalog
          content:
            application/json:
              schema:
                type: object
                properties:
                  features:
                    type: array
                    items:
                      $ref: '#/components/schemas/FeatureTemplate'

components:
  schemas:
    AIPreparationRequest:
      type: object
      required:
        - product_id
        - standardized_data
        - target_use_case
      properties:
        product_id:
          type: string
          format: uuid
        standardized_data:
          type: object
        target_use_case:
          type: string
          enum: [risk_analysis, fraud_detection, portfolio_optimization, credit_scoring]
        feature_engineering:
          $ref: '#/components/schemas/FeatureEngineeringConfig'
        privacy_settings:
          $ref: '#/components/schemas/PrivacySettings'
    
    FeatureEngineeringConfig:
      type: object
      properties:
        auto_features:
          type: boolean
          default: true
        custom_features:
          type: array
          items:
            $ref: '#/components/schemas/CustomFeature'
        time_windows:
          type: array
          items:
            type: string
            enum: [1D, 7D, 30D, 90D, 1Y]
        aggregations:
          type: array
          items:
            type: string
            enum: [mean, std, min, max, sum, count]
    
    PrivacySettings:
      type: object
      properties:
        anonymization_level:
          type: string
          enum: [none, basic, strict]
        differential_privacy:
          type: boolean
          default: false
        epsilon:
          type: number
          minimum: 0.1
          maximum: 10.0
        remove_pii:
          type: boolean
          default: true
    
    AIPreparationResponse:
      type: object
      properties:
        prepared_data:
          type: object
          properties:
            features:
              type: array
              items:
                $ref: '#/components/schemas/Feature'
            metadata:
              type: object
        feature_importance:
          type: array
          items:
            type: object
            properties:
              feature_name:
                type: string
              importance_score:
                type: number
        data_quality_report:
          $ref: '#/components/schemas/DataQualityReport'
        privacy_report:
          $ref: '#/components/schemas/PrivacyReport'
        next_agent:
          type: string
```

---

## Agent 3 - Vector Processing API

```yaml
openapi: 3.0.0
info:
  title: Agent 3 - Vector Processing API
  description: Converts prepared data into vector embeddings for similarity search and ML
  version: 1.0.0

paths:
  /agents/agent3/vectorize:
    post:
      summary: Generate vector embeddings
      operationId: generateVectors
      tags:
        - Vector Processing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/VectorizationRequest'
      responses:
        '200':
          description: Vectorization successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VectorizationResponse'

  /agents/agent3/similarity/search:
    post:
      summary: Search similar vectors
      operationId: similaritySearch
      tags:
        - Similarity Search
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SimilaritySearchRequest'
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SimilaritySearchResponse'

components:
  schemas:
    VectorizationRequest:
      type: object
      required:
        - product_id
        - prepared_data
        - embedding_model
      properties:
        product_id:
          type: string
          format: uuid
        prepared_data:
          type: object
        embedding_model:
          type: string
          enum: [financial_bert, sentence_transformer, custom_financial]
        vector_config:
          $ref: '#/components/schemas/VectorConfig'
    
    VectorConfig:
      type: object
      properties:
        dimension:
          type: integer
          enum: [128, 256, 512, 768, 1024]
          default: 768
        normalization:
          type: boolean
          default: true
        compression:
          type: string
          enum: [none, pca, autoencoder]
          default: none
        batch_size:
          type: integer
          default: 32
    
    VectorizationResponse:
      type: object
      properties:
        vectors:
          type: array
          items:
            $ref: '#/components/schemas/VectorEmbedding'
        index_id:
          type: string
          description: ID of the created vector index
        metadata:
          type: object
          properties:
            total_vectors:
              type: integer
            dimension:
              type: integer
            model_version:
              type: string
        next_agent:
          type: string
    
    VectorEmbedding:
      type: object
      properties:
        id:
          type: string
        vector:
          type: array
          items:
            type: number
        metadata:
          type: object
        timestamp:
          type: string
          format: date-time
```

---

## Agent 4 - Calculation & Validation API

```yaml
openapi: 3.0.0
info:
  title: Agent 4 - Financial Calculation & Validation API
  description: Performs complex financial calculations and validation checks
  version: 1.0.0

paths:
  /agents/agent4/calculate/risk:
    post:
      summary: Calculate risk metrics
      operationId: calculateRiskMetrics
      tags:
        - Risk Calculations
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RiskCalculationRequest'
      responses:
        '200':
          description: Risk calculations complete
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RiskCalculationResponse'

  /agents/agent4/validate/compliance:
    post:
      summary: Validate compliance rules
      operationId: validateCompliance
      tags:
        - Compliance Validation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ComplianceValidationRequest'
      responses:
        '200':
          description: Compliance validation results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ComplianceValidationResponse'

components:
  schemas:
    RiskCalculationRequest:
      type: object
      required:
        - product_id
        - portfolio_data
        - calculation_type
      properties:
        product_id:
          type: string
          format: uuid
        portfolio_data:
          type: object
          properties:
            positions:
              type: array
              items:
                $ref: '#/components/schemas/Position'
            market_data:
              type: object
        calculation_type:
          type: array
          items:
            type: string
            enum: [var, cvar, stress_test, sensitivity, expected_shortfall]
        parameters:
          $ref: '#/components/schemas/RiskParameters'
    
    RiskParameters:
      type: object
      properties:
        confidence_level:
          type: number
          enum: [0.95, 0.99, 0.999]
          default: 0.95
        time_horizon:
          type: integer
          description: Time horizon in days
          default: 1
        historical_days:
          type: integer
          default: 252
        simulation_count:
          type: integer
          default: 10000
        methodology:
          type: string
          enum: [historical, parametric, monte_carlo]
    
    RiskCalculationResponse:
      type: object
      properties:
        risk_metrics:
          type: object
          properties:
            var:
              $ref: '#/components/schemas/VaRResult'
            cvar:
              $ref: '#/components/schemas/CVaRResult'
            stress_test:
              $ref: '#/components/schemas/StressTestResult'
        calculation_metadata:
          type: object
          properties:
            calculation_timestamp:
              type: string
              format: date-time
            methodology_used:
              type: string
            data_quality_score:
              type: number
        warnings:
          type: array
          items:
            type: string
        next_agent:
          type: string
    
    VaRResult:
      type: object
      properties:
        value:
          type: number
        confidence_level:
          type: number
        time_horizon:
          type: integer
        currency:
          type: string
        breakdown:
          type: array
          items:
            type: object
            properties:
              component:
                type: string
              contribution:
                type: number
```

---

## Agent 5 - Quality Assurance API

```yaml
openapi: 3.0.0
info:
  title: Agent 5 - Quality Assurance & Validation API
  description: Final quality checks and report generation
  version: 1.0.0

paths:
  /agents/agent5/qa/comprehensive:
    post:
      summary: Perform comprehensive QA check
      operationId: performQACheck
      tags:
        - Quality Assurance
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QACheckRequest'
      responses:
        '200':
          description: QA check complete
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QACheckResponse'

  /agents/agent5/reports/generate:
    post:
      summary: Generate compliance report
      operationId: generateReport
      tags:
        - Report Generation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ReportGenerationRequest'
      responses:
        '200':
          description: Report generated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReportGenerationResponse'

components:
  schemas:
    QACheckRequest:
      type: object
      required:
        - product_id
        - processing_history
        - qa_rules
      properties:
        product_id:
          type: string
          format: uuid
        processing_history:
          type: array
          items:
            $ref: '#/components/schemas/ProcessingStep'
        qa_rules:
          type: array
          items:
            $ref: '#/components/schemas/QARule'
        approval_workflow:
          $ref: '#/components/schemas/ApprovalWorkflow'
    
    ProcessingStep:
      type: object
      properties:
        agent_id:
          type: string
        timestamp:
          type: string
          format: date-time
        input_hash:
          type: string
        output_hash:
          type: string
        transformations:
          type: array
          items:
            type: string
        metrics:
          type: object
    
    QARule:
      type: object
      properties:
        rule_id:
          type: string
        rule_type:
          type: string
          enum: [data_quality, business_logic, regulatory, performance]
        severity:
          type: string
          enum: [critical, high, medium, low]
        threshold:
          type: object
    
    QACheckResponse:
      type: object
      properties:
        qa_status:
          type: string
          enum: [passed, failed, conditional_pass]
        quality_score:
          type: number
          minimum: 0
          maximum: 1
        issues:
          type: array
          items:
            $ref: '#/components/schemas/QAIssue'
        recommendations:
          type: array
          items:
            type: string
        audit_trail:
          $ref: '#/components/schemas/AuditTrail'
        approval_status:
          type: string
          enum: [approved, pending_approval, rejected]
    
    QAIssue:
      type: object
      properties:
        issue_id:
          type: string
        severity:
          type: string
        category:
          type: string
        description:
          type: string
        affected_fields:
          type: array
          items:
            type: string
        resolution:
          type: string
```

---

## Common Components

```yaml
components:
  responses:
    BadRequest:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error_code: "INVALID_REQUEST"
            message: "The request body is invalid"
            details:
              - field: "product_name"
                issue: "Required field missing"
    
    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error_code: "RESOURCE_NOT_FOUND"
            message: "The requested resource was not found"
    
    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error_code: "RESOURCE_CONFLICT"
            message: "A resource with the same identifier already exists"
    
    InternalError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error_code: "INTERNAL_ERROR"
            message: "An unexpected error occurred"
  
  schemas:
    Error:
      type: object
      required:
        - error_code
        - message
      properties:
        error_code:
          type: string
        message:
          type: string
        details:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              issue:
                type: string
        correlation_id:
          type: string
          format: uuid
        timestamp:
          type: string
          format: date-time
    
    PaginationParams:
      type: object
      properties:
        page:
          type: integer
          minimum: 1
          default: 1
        page_size:
          type: integer
          minimum: 1
          maximum: 100
          default: 20
        sort_by:
          type: string
        sort_order:
          type: string
          enum: [asc, desc]
          default: asc
    
    PaginatedResponse:
      type: object
      properties:
        items:
          type: array
          items: {}
        pagination:
          type: object
          properties:
            current_page:
              type: integer
            page_size:
              type: integer
            total_items:
              type: integer
            total_pages:
              type: integer
            has_next:
              type: boolean
            has_previous:
              type: boolean
  
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
    
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.a2a-platform.com/oauth/authorize
          tokenUrl: https://auth.a2a-platform.com/oauth/token
          scopes:
            read: Read access to data
            write: Write access to data
            admin: Administrative access
  
security:
  - BearerAuth: []
  - ApiKeyAuth: []
  - OAuth2: [read, write]
```

---

## Error Code Reference

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| INVALID_REQUEST | 400 | Request validation failed |
| AUTHENTICATION_FAILED | 401 | Authentication credentials invalid |
| INSUFFICIENT_PERMISSIONS | 403 | User lacks required permissions |
| RESOURCE_NOT_FOUND | 404 | Requested resource doesn't exist |
| RESOURCE_CONFLICT | 409 | Resource already exists |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error occurred |
| SERVICE_UNAVAILABLE | 503 | Service temporarily unavailable |

---

## Rate Limiting

All APIs implement rate limiting with the following default limits:

| Endpoint Type | Rate Limit | Window |
|--------------|------------|---------|
| Data Registration | 100 requests | 1 minute |
| Data Processing | 50 requests | 1 minute |
| Calculations | 20 requests | 1 minute |
| Search/Query | 1000 requests | 1 minute |

Rate limit headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: UTC timestamp when limit resets

---

## Webhook Events

Agents can send webhook notifications for important events:

```yaml
webhook_event:
  type: object
  properties:
    event_id:
      type: string
      format: uuid
    event_type:
      type: string
      enum: [
        data_product.registered,
        data_product.validated,
        processing.completed,
        processing.failed,
        approval.required,
        approval.completed
      ]
    timestamp:
      type: string
      format: date-time
    agent_id:
      type: string
    payload:
      type: object
    signature:
      type: string
      description: HMAC-SHA256 signature of the payload
```

---

*Last Updated: December 2024*
*Version: 1.0.0*