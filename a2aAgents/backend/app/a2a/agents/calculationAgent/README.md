# Calculation Agent

## Overview

The Calculation Agent is a general-purpose service for executing on-demand financial calculations. Unlike specialized validation agents, this agent provides a flexible framework for running a wide variety of quantitative models and algorithms. Its primary functions are:

-   **On-Demand Calculations**: Exposes an API to run financial calculations, such as pricing derivatives, computing portfolio statistics, or running simulations.
-   **Model Library**: Maintains a library of pre-built calculation models that can be invoked via the API.
-   **Custom Logic**: Supports the execution of custom calculation logic provided in a request.

This agent serves as a core computational engine for the A2A Network, available for any workflow requiring quantitative analysis.

## API Specification

This agent exposes a RESTful API for performing financial calculations.

```yaml
openapi: 3.0.0
info:
  title: Calculation Agent API
  description: API for performing complex, asynchronous calculations on data products.
  version: 1.0.0

servers:
  - url: http://localhost:8010
    description: Local development server

paths:
  /agents/calculationAgent/jobs:
    post:
      summary: Submit a new calculation job
      operationId: submitCalculationJob
      tags:
        - Calculation Jobs
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CalculationRequest'
      responses:
        '202':
          description: Calculation job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatusResponse'

  /agents/calculationAgent/jobs/{job_id}:
    get:
      summary: Get the status of a calculation job
      operationId: getJobStatus
      tags:
        - Calculation Jobs
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatusResponse'

  /agents/calculationAgent/jobs/{job_id}/results:
    get:
      summary: Get the results of a completed calculation job
      operationId: getJobResults
      tags:
        - Calculation Jobs
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Calculation results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CalculationResult'
        '404':
          description: Job not found or results not ready

components:
  schemas:
    CalculationRequest:
      type: object
      required:
        - data_product_id
        - calculation_logic
      properties:
        data_product_id:
          type: string
          description: The ID of the data product to perform calculations on.
        calculation_logic:
          type: object
          description: The logic or formula for the calculation.
          properties:
            type:
              type: string
              enum: [formula, script, model_inference]
            content:
              type: string
              description: The actual calculation script, formula, or model ID.

    JobStatusResponse:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [pending, running, completed, failed]
        message:
          type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    CalculationResult:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [completed]
        results:
          type: object
          description: The output of the calculation.
          additionalProperties: true
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Calculation Agent are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_calculation_agent_sdk.py
```

The agent will be available at `http://localhost:8010`.
