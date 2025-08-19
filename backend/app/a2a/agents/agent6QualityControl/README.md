# Agent 6: Quality Control

## Overview

The Quality Control Agent (Agent 6) is responsible for ongoing monitoring and quality management of data products within the A2A Network. Unlike Agent 5, which performs a final validation, Agent 6 focuses on continuous quality control throughout the data product lifecycle. Its key functions include:

-   **Continuous Monitoring**: Actively monitors data products for degradation in quality, such as data drift, schema changes, or stale information.
-   **Automated Remediation**: Can trigger automated workflows to remediate quality issues, such as reprocessing data or notifying data owners.
-   **Quality Reporting**: Provides dashboards and reports on the overall health and quality of the data ecosystem.

This agent helps maintain the long-term health and reliability of the data products in the network.

## API Specification

This agent exposes a RESTful API for managing and monitoring data quality.

```yaml
openapi: 3.0.0
info:
  title: Agent 6 - Quality Control API
  description: Continuously monitors data products for quality and triggers remediation.
  version: 1.0.0

servers:
  - url: http://localhost:8007
    description: Local development server

paths:
  /agents/agent6/monitors:
    post:
      summary: Create a new quality monitor
      operationId: createQualityMonitor
      tags:
        - Quality Monitoring
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/QualityMonitorRequest'
      responses:
        '201':
          description: Monitor created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QualityMonitorResponse'

  /agents/agent6/monitors/{monitor_id}:
    get:
      summary: Get quality monitor status
      operationId: getQualityMonitor
      tags:
        - Quality Monitoring
      parameters:
        - name: monitor_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Monitor status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QualityMonitorResponse'

  /agents/agent6/reports/{product_id}:
    get:
      summary: Get quality report for a data product
      operationId: getQualityReport
      tags:
        - Reporting
      parameters:
        - name: product_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Quality report
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/QualityReport'

  /agents/agent6/remediate:
    post:
      summary: Trigger a remediation action
      operationId: triggerRemediation
      tags:
        - Remediation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RemediationRequest'
      responses:
        '202':
          description: Remediation action accepted

components:
  schemas:
    QualityMonitorRequest:
      type: object
      required:
        - product_id
        - monitor_type
        - check_frequency
      properties:
        product_id:
          type: string
          format: uuid
        monitor_type:
          type: string
          enum: [data_drift, schema_change, freshness, anomaly_detection]
        check_frequency:
          type: string
          description: Cron-style frequency string (e.g., '0 * * * *')
        thresholds:
          type: object
          description: Key-value pairs for quality thresholds.

    QualityMonitorResponse:
      type: object
      properties:
        monitor_id:
          type: string
        status:
          type: string
          enum: [active, paused, failed]
        last_check:
          type: string
          format: date-time
        last_result:
          type: string
          enum: [pass, fail, warning]

    QualityReport:
      type: object
      properties:
        product_id:
          type: string
          format: uuid
        overall_quality_score:
          type: number
        historical_quality:
          type: array
          items:
            type: object
            properties:
              timestamp:
                type: string
                format: date-time
              score:
                type: number
        active_issues:
          type: array
          items:
            type: object
            properties:
              issue_type:
                type: string
              severity:
                type: string
              description:
                type: string

    RemediationRequest:
      type: object
      required:
        - product_id
        - issue_type
        - action
      properties:
        product_id:
          type: string
          format: uuid
        issue_type:
          type: string
        action:
          type: string
          enum: [reprocess, notify_owner, quarantine]
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent 6 are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
# Launch script not yet defined in agentOrganizationGuide.md
# Example: python launch_agent6_sdk.py
```

The agent will be available at `http://localhost:8007` (port inferred from agent guide for Agent Manager, assuming sequential assignment).
