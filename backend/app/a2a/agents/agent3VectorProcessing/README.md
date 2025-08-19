# Agent 3: Vector Processing

## Overview

The Vector Processing Agent (Agent 3) is responsible for converting prepared data into numerical vector embeddings. These embeddings are crucial for various AI/ML tasks, especially those involving similarity search, clustering, and as input to deep learning models. Its primary functions are:

-   **Vectorization**: Takes prepared data and uses a specified embedding model (e.g., financial_bert, sentence_transformer) to generate high-dimensional vector representations.
-   **Indexing**: Stores the generated vectors in an efficient index for fast retrieval and similarity search.
-   **Similarity Search**: Provides an endpoint to find vectors (and their corresponding data items) that are most similar to a given query vector.

This agent bridges the gap between structured/unstructured data and the mathematical representations required by advanced AI models.

## API Specification

This agent exposes a RESTful API for vectorizing data and performing similarity searches.

```yaml
openapi: 3.0.0
info:
  title: Agent 3 - Vector Processing API
  description: Converts prepared data into vector embeddings for similarity search and ML
  version: 1.0.0

servers:
  - url: http://localhost:8004
    description: Local development server

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

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent 3 are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent3_sdk.py
```

The agent will be available at `http://localhost:8004`.
