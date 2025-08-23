# Agent 1: Data Standardization

## Overview

The Data Standardization Agent (Agent 1) is responsible for transforming registered data products into a canonical format that can be used consistently across the A2A Network. Its key functions include:

-   **Schema Mapping**: Maps data from its source schema to the A2A standard schema.
-   **Data Transformation**: Converts data types, formats, and conventions (e.g., currency, date formats) to a unified standard.
-   **Enrichment**: Augments the data with additional information where necessary, such as resolving entities or adding standard identifiers.

This agent ensures that data is clean, consistent, and ready for downstream processing by other agents.

## API Specification

This agent exposes a RESTful API for standardizing data products.

```yaml
openapi: 3.0.0
info:
  title: Agent 1 - Financial Data Standardization API
  description: Standardizes financial data across different formats and conventions
  version: 1.0.0

servers:
  - url: http://localhost:8002
    description: Local development server

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
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent 1 are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent1_sdk.py
```

The agent will be available at `http://localhost:8002`.
