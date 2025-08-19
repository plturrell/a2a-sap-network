# Embedding Fine-Tuner

## Overview

The Embedding Fine-Tuner agent is a specialized service for adapting pre-trained language and embedding models to specific financial domains. Fine-tuning improves model performance on downstream tasks by continuing the training process on a smaller, domain-specific dataset. Its key functions are:

-   **Fine-Tuning Workflows**: Manages the process of fine-tuning embedding models, including data preparation, training, and validation.
-   **Model Evaluation**: Evaluates the performance of fine-tuned models on benchmark tasks to ensure they provide an improvement over the base models.
-   **Model Versioning**: Tracks different versions of fine-tuned models, allowing agents to select the best model for their specific use case.

This agent is critical for achieving state-of-the-art performance on NLP and other AI tasks within the financial domain.

## API Specification

This agent exposes a RESTful API for fine-tuning embedding models.

```yaml
openapi: 3.0.0
info:
  title: Embedding Fine-Tuner API
  description: API for fine-tuning embedding models on specific datasets.
  version: 1.0.0

servers:
  - url: http://localhost:8013
    description: Local development server

paths:
  /agents/embeddingFineTuner/jobs:
    post:
      summary: Create a new fine-tuning job
      operationId: createFineTuningJob
      tags:
        - Fine-Tuning
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/FineTuningJobRequest'
      responses:
        '202':
          description: Fine-tuning job created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FineTuningJob'

  /agents/embeddingFineTuner/jobs/{job_id}:
    get:
      summary: Get fine-tuning job status
      operationId: getFineTuningJob
      tags:
        - Fine-Tuning
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Fine-tuning job status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FineTuningJob'

  /agents/embeddingFineTuner/models:
    get:
      summary: List available fine-tuned models
      operationId: listFineTunedModels
      tags:
        - Models
      responses:
        '200':
          description: A list of fine-tuned models
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/FineTunedModel'

components:
  schemas:
    FineTuningJobRequest:
      type: object
      required:
        - base_model
        - training_data_product_id
      properties:
        base_model:
          type: string
          description: The base embedding model to fine-tune.
          example: 'text-embedding-ada-002'
        training_data_product_id:
          type: string
          description: The ID of the data product containing the training data.
        hyperparameters:
          type: object
          properties:
            epochs:
              type: integer
              default: 3
            learning_rate:
              type: number
              format: float
              default: 0.001

    FineTuningJob:
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
        finished_at:
          type: string
          format: date-time
        fine_tuned_model_id:
          type: string

    FineTunedModel:
      type: object
      properties:
        model_id:
          type: string
        base_model:
          type: string
        status:
          type: string
          enum: [available, creating, failed]
        created_at:
          type: string
          format: date-time
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Embedding Fine-Tuner are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_embedding_fine_tuner_sdk.py
```

The agent will be available at `http://localhost:8013`.
