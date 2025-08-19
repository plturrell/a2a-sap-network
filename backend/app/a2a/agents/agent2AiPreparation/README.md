# Agent 2: AI Data Preparation

## Overview

The AI Data Preparation Agent (Agent 2) is responsible for preparing standardized data for use in AI and machine learning models. Its main tasks include:

-   **Feature Engineering**: Creates new features from the existing data to improve model performance. This can include automated feature generation or the application of predefined feature templates.
-   **Data Transformation**: Applies transformations required by ML models, such as normalization, scaling, and encoding of categorical variables.
-   **Privacy Preservation**: Implements privacy-enhancing techniques like anonymization or differential privacy to protect sensitive information.

This agent ensures that data is in the optimal format and shape for effective AI/ML model training and inference.

## API Specification

This agent exposes a RESTful API for preparing data for AI use cases.

```yaml
openapi: 3.0.0
info:
  title: Agent 2 - AI Data Preparation API
  description: Prepares financial data for AI/ML processing with feature engineering
  version: 1.0.0

servers:
  - url: http://localhost:8003
    description: Local development server

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

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent 2 are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent2_sdk.py
```

The agent will be available at `http://localhost:8003`.
