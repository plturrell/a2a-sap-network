# Catalog Manager

## Overview

The Catalog Manager is responsible for maintaining a comprehensive, searchable catalog of all data products, services, and agents available within the A2A Network. It serves as a central discovery mechanism for all network participants. Its key functions include:

-   **Metadata Management**: Stores and manages metadata for all registered assets, including schemas, ownership, quality scores, and usage policies.
-   **Search and Discovery**: Provides a powerful search API to allow users and agents to find relevant data products and services based on various criteria.
-   **Data Lineage**: Tracks the lineage of data products, showing how they were created and transformed by different agents in the network.

This agent is crucial for making the A2A Network's resources discoverable and understandable.

## API Specification

This agent exposes a RESTful API for managing and searching the catalog.

```yaml
openapi: 3.0.0
info:
  title: Catalog Manager API
  description: API for managing the metadata, schema, and lineage of data products in the A2A Network.
  version: 1.0.0

servers:
  - url: http://localhost:8009
    description: Local development server

paths:
  /agents/catalogManager/data-products:
    post:
      summary: Register a new data product
      operationId: registerDataProduct
      tags:
        - Data Products
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProduct'
      responses:
        '201':
          description: Data product registered successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DataProduct'
    get:
      summary: List or search for data products
      operationId: listDataProducts
      tags:
        - Data Products
      parameters:
        - name: query
          in: query
          schema:
            type: string
          description: Search query to filter products by name, description, or tags.
      responses:
        '200':
          description: A list of data products
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/DataProduct'

  /agents/catalogManager/data-products/{product_id}:
    get:
      summary: Get data product details
      operationId: getDataProduct
      tags:
        - Data Products
      parameters:
        - name: product_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Data product details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DataProduct'
    put:
      summary: Update data product metadata
      operationId: updateDataProduct
      tags:
        - Data Products
      parameters:
        - name: product_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DataProductUpdate'
      responses:
        '200':
          description: Data product updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DataProduct'

  /agents/catalogManager/data-products/{product_id}/lineage:
    get:
      summary: Get data product lineage
      operationId: getDataProductLineage
      tags:
        - Data Lineage
      parameters:
        - name: product_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Data product lineage graph
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LineageGraph'

components:
  schemas:
    DataProduct:
      type: object
      required:
        - name
        - schema
      properties:
        id:
          type: string
          format: uuid
          readOnly: true
        name:
          type: string
        description:
          type: string
        owner:
          type: string
        tags:
          type: array
          items:
            type: string
        schema:
          type: object
          description: The schema of the data product (e.g., JSON Schema, Avro).
          additionalProperties: true
        created_at:
          type: string
          format: date-time
          readOnly: true

    DataProductUpdate:
      type: object
      properties:
        description:
          type: string
        owner:
          type: string
        tags:
          type: array
          items:
            type: string

    LineageGraph:
      type: object
      properties:
        nodes:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              type:
                type: string
                enum: [data_product, agent, process]
              label:
                type: string
        edges:
          type: array
          items:
            type: object
            properties:
              source:
                type: string
              target:
                type: string
              label:
                type: string
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Catalog Manager are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_catalog_manager_sdk.py
```

The agent will be available at `http://localhost:8009`.
