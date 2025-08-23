# Agent Manager

## Overview

The Agent Manager is a core component of the A2A Network, responsible for the lifecycle management of all other agents. It acts as a central registry and orchestrator. Its key functions include:

-   **Agent Registration**: Keeps a directory of all active agents in the network, including their capabilities, status, and endpoint information.
-   **Lifecycle Management**: Handles the starting, stopping, and monitoring of agents.
-   **Health Checks**: Continuously monitors the health of all registered agents to ensure the network is functioning correctly.
-   **Service Discovery**: Allows agents and external clients to discover and communicate with other agents in the network.

This agent is fundamental to the dynamic and robust operation of the A2A Network.

## API Specification

This agent exposes a RESTful API for managing agents.

```yaml
openapi: 3.0.0
info:
  title: Agent Manager API
  description: API for managing the lifecycle and registration of agents in the A2A Network.
  version: 1.0.0

servers:
  - url: http://localhost:8008
    description: Local development server

paths:
  /agents/agentManager/register:
    post:
      summary: Register a new agent
      operationId: registerAgent
      tags:
        - Agent Lifecycle
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AgentRegistrationRequest'
      responses:
        '201':
          description: Agent registered successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'

  /agents/agentManager/agents:
    get:
      summary: List all registered agents
      operationId: listAgents
      tags:
        - Service Discovery
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [running, stopped, unhealthy]
        - name: capability
          in: query
          schema:
            type: string
      responses:
        '200':
          description: A list of agents
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Agent'

  /agents/agentManager/agents/{agent_id}:
    get:
      summary: Get agent details
      operationId: getAgent
      tags:
        - Service Discovery
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Agent details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'

  /agents/agentManager/agents/{agent_id}/heartbeat:
    post:
      summary: Agent heartbeat
      operationId: agentHeartbeat
      tags:
        - Agent Lifecycle
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: Heartbeat acknowledged

components:
  schemas:
    AgentRegistrationRequest:
      type: object
      required:
        - agent_name
        - endpoint
        - capabilities
      properties:
        agent_name:
          type: string
        endpoint:
          type: string
          format: uri
        capabilities:
          type: array
          items:
            type: string

    Agent:
      type: object
      properties:
        agent_id:
          type: string
          format: uuid
        agent_name:
          type: string
        endpoint:
          type: string
          format: uri
        status:
          type: string
          enum: [running, stopped, unhealthy]
        registered_at:
          type: string
          format: date-time
        last_heartbeat:
          type: string
          format: date-time
        capabilities:
          type: array
          items:
            type: string
```

## Configuration

This agent is configured through environment variables.

```bash
# No specific environment variables for Agent Manager are defined yet.
# It uses the common A2A platform configurations.
```

## Usage

To run this agent using its SDK implementation:

```bash
python launch_agent_manager_sdk.py
```

The agent will be available at `http://localhost:8008`.
