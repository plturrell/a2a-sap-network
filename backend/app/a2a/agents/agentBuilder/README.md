# Agent Builder

## Overview

The Agent Builder is a meta-agent responsible for creating, configuring, and deploying new agents within the A2A Network. It provides a high-level interface to abstract away the complexities of agent development and management. Its key functions include:

-   **Agent Scaffolding**: Generates the boilerplate code and directory structure for a new agent based on a predefined template.
-   **Configuration Management**: Manages the configuration files and environment variables for new and existing agents.
-   **Deployment**: Automates the process of deploying an agent to the network, including containerization and service registration.

This agent is essential for scaling the A2A Network by simplifying the creation of new, specialized agents.

## API Specification

This agent exposes a RESTful API for building and managing agents.

```yaml
openapi: 3.0.0
info:
  title: Agent Builder API
  description: API for creating, configuring, and deploying new A2A agents.
  version: 1.0.0

servers:
  - url: http://localhost:8012
    description: Local development server

paths:
  /agents/agentBuilder/scaffold:
    post:
      summary: Scaffold a new agent
      operationId: scaffoldAgent
      tags:
        - Agent Lifecycle
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ScaffoldRequest'
      responses:
        '201':
          description: Agent scaffold created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentStatusResponse'

  /agents/agentBuilder/configure/{agent_id}:
    post:
      summary: Configure an existing agent
      operationId: configureAgent
      tags:
        - Agent Lifecycle
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ConfigurationRequest'
      responses:
        '200':
          description: Agent configured successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentStatusResponse'

  /agents/agentBuilder/deploy/{agent_id}:
    post:
      summary: Deploy an agent
      operationId: deployAgent
      tags:
        - Agent Lifecycle
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DeploymentRequest'
      responses:
        '202':
          description: Agent deployment initiated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentStatusResponse'

  /agents/agentBuilder/status/{agent_id}:
    get:
      summary: Get agent build/deployment status
      operationId: getAgentStatus
      tags:
        - Agent Status
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Agent status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentStatusResponse'

components:
  schemas:
    ScaffoldRequest:
      type: object
      required:
        - agent_name
        - agent_template
      properties:
        agent_name:
          type: string
          description: A unique name for the new agent.
        agent_template:
          type: string
          enum: [basic, data_processing, nlp_service, validation]
          description: The template to use for scaffolding.

    ConfigurationRequest:
      type: object
      properties:
        environment_variables:
          type: object
          additionalProperties:
            type: string
        dependencies:
          type: array
          items:
            type: string

    DeploymentRequest:
      type: object
      required:
        - target_environment
      properties:
        target_environment:
          type: string
          enum: [development, staging, production]
        docker_options:
          type: object
          properties:
            build:
              type: boolean
              default: true
            push_to_registry:
              type: boolean
              default: false

    AgentStatusResponse:
      type: object
      properties:
        agent_id:
          type: string
        agent_name:
          type: string
        status:
          type: string
          enum: [scaffolded, configured, deploying, running, failed]
        message:
          type: string
        last_updated:
          type: string
          format: date-time
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent Builder are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent_builder_sdk.py
```

The agent will be available at `http://localhost:8012`.
