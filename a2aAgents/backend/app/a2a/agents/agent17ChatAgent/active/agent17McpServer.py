#!/usr/bin/env python3
"""
MCP Server for Agent 17 Chat Agent
Provides MCP interface for the A2A-compliant chat agent
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from mcp import Server
from mcp.types import (
    Tool, Resource, Prompt,
    TextContent, ImageContent, EmbeddedResource,
    GetPromptResult, ListPromptsResult,
    ListResourcesResult, ListToolsResult, ReadResourceResult,
    CallToolResult, ErrorData
)

from agent17ChatAgentSdk import create_agent17_chat_agent, Agent17ChatAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent17MCPServer:
    """MCP Server implementation for Agent 17 Chat Agent"""
    
    def __init__(self):
        self.server = Server("agent17-chat")
        self.agent: Optional[Agent17ChatAgent] = None
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available MCP tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="chat_process_message",
                        description="Process a chat message and route to appropriate agents",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "User message to process"},
                                "user_id": {"type": "string", "description": "User identifier"},
                                "conversation_id": {"type": "string", "description": "Conversation ID"}
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="chat_analyze_intent",
                        description="Analyze user intent to determine best agent routing",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Message to analyze"}
                            },
                            "required": ["prompt"]
                        }
                    ),
                    Tool(
                        name="chat_multi_agent_query",
                        description="Coordinate query across multiple agents",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Query to process"},
                                "target_agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of agent IDs"
                                },
                                "coordination_type": {
                                    "type": "string",
                                    "enum": ["parallel", "sequential"],
                                    "description": "How to coordinate agents"
                                }
                            },
                            "required": ["query", "target_agents"]
                        }
                    ),
                    Tool(
                        name="chat_get_statistics",
                        description="Get chat agent statistics",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    ),
                    Tool(
                        name="chat_list_agents",
                        description="List available agents in the network",
                        inputSchema={
                            "type": "object",
                            "properties": {}
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Execute MCP tool"""
            try:
                if not self.agent:
                    raise RuntimeError("Agent not initialized")
                
                if name == "chat_process_message":
                    prompt = arguments.get("prompt", "")
                    user_id = arguments.get("user_id", "anonymous")
                    conversation_id = arguments.get("conversation_id", "default")
                    
                    # Process through agent
                    result = await self.agent._analyze_and_route(prompt, conversation_id)
                    
                    # Track conversation
                    await self.agent._track_conversation(conversation_id, user_id)
                    
                    return CallToolResult(
                        content=[TextContent(
                            text=json.dumps({
                                "status": "success",
                                "routing_result": result,
                                "conversation_id": conversation_id
                            }, indent=2)
                        )]
                    )
                
                elif name == "chat_analyze_intent":
                    prompt = arguments.get("prompt", "")
                    
                    result = await self.agent._analyze_intent(prompt)
                    
                    return CallToolResult(
                        content=[TextContent(
                            text=json.dumps({
                                "status": "success",
                                "intent_analysis": result
                            }, indent=2)
                        )]
                    )
                
                elif name == "chat_multi_agent_query":
                    query = arguments.get("query", "")
                    target_agents = arguments.get("target_agents", [])
                    coordination_type = arguments.get("coordination_type", "parallel")
                    
                    result = await self.agent._coordinate_agents(
                        query, target_agents, coordination_type, "mcp_context"
                    )
                    
                    return CallToolResult(
                        content=[TextContent(
                            text=json.dumps({
                                "status": "success",
                                "coordination_result": result
                            }, indent=2)
                        )]
                    )
                
                elif name == "chat_get_statistics":
                    stats = await self.agent.get_statistics()
                    
                    return CallToolResult(
                        content=[TextContent(
                            text=json.dumps(stats, indent=2)
                        )]
                    )
                
                elif name == "chat_list_agents":
                    agents = {
                        agent_id: {
                            "id": info["id"],
                            "type": info["type"],
                            "capabilities": info["capabilities"]
                        }
                        for agent_id, info in self.agent.agent_registry.items()
                    }
                    
                    return CallToolResult(
                        content=[TextContent(
                            text=json.dumps({
                                "status": "success",
                                "agents": agents,
                                "count": len(agents)
                            }, indent=2)
                        )]
                    )
                
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return CallToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    isError=True
                )
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available MCP resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="agent17://statistics",
                        name="Chat Agent Statistics",
                        description="Real-time statistics for Agent 17",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="agent17://conversations",
                        name="Active Conversations",
                        description="Currently active chat conversations",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="agent17://agent_registry",
                        name="Agent Registry",
                        description="Discovered agents in the network",
                        mimeType="application/json"
                    )
                ]
            )
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read MCP resource"""
            try:
                if not self.agent:
                    raise RuntimeError("Agent not initialized")
                
                if uri == "agent17://statistics":
                    stats = await self.agent.get_statistics()
                    return ReadResourceResult(
                        contents=[TextContent(
                            text=json.dumps(stats, indent=2),
                            mimeType="application/json"
                        )]
                    )
                
                elif uri == "agent17://conversations":
                    conversations = {
                        conv_id: {
                            "user_id": data["user_id"],
                            "started_at": data["started_at"],
                            "message_count": data["message_count"],
                            "last_activity": data.get("last_activity")
                        }
                        for conv_id, data in self.agent.active_conversations.items()
                    }
                    
                    return ReadResourceResult(
                        contents=[TextContent(
                            text=json.dumps({
                                "conversations": conversations,
                                "count": len(conversations)
                            }, indent=2),
                            mimeType="application/json"
                        )]
                    )
                
                elif uri == "agent17://agent_registry":
                    return ReadResourceResult(
                        contents=[TextContent(
                            text=json.dumps({
                                "agents": self.agent.agent_registry,
                                "count": len(self.agent.agent_registry),
                                "last_discovery": "on_startup"
                            }, indent=2),
                            mimeType="application/json"
                        )]
                    )
                
                else:
                    raise ValueError(f"Unknown resource: {uri}")
                    
            except Exception as e:
                logger.error(f"Resource read error: {e}")
                return ReadResourceResult(
                    contents=[TextContent(text=f"Error: {str(e)}")],
                    isError=True
                )
        
        @self.server.list_prompts()
        async def list_prompts() -> ListPromptsResult:
            """List available MCP prompts"""
            return ListPromptsResult(
                prompts=[
                    Prompt(
                        name="analyze_request",
                        description="Analyze a user request and suggest agent routing",
                        arguments=[
                            {
                                "name": "request",
                                "description": "The user's request",
                                "required": True
                            }
                        ]
                    ),
                    Prompt(
                        name="coordinate_agents",
                        description="Coordinate multiple agents for a complex task",
                        arguments=[
                            {
                                "name": "task",
                                "description": "The task description",
                                "required": True
                            },
                            {
                                "name": "agents",
                                "description": "Comma-separated list of agent IDs",
                                "required": True
                            }
                        ]
                    )
                ]
            )
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Dict[str, str]) -> GetPromptResult:
            """Get MCP prompt"""
            try:
                if name == "analyze_request":
                    request = arguments.get("request", "")
                    
                    return GetPromptResult(
                        messages=[{
                            "role": "user",
                            "content": TextContent(
                                text=f"Analyze this request and determine which A2A agents should handle it:\n\n{request}\n\nProvide your analysis including:\n1. Intent type\n2. Recommended agents\n3. Routing strategy"
                            )
                        }]
                    )
                
                elif name == "coordinate_agents":
                    task = arguments.get("task", "")
                    agents = arguments.get("agents", "").split(",")
                    
                    return GetPromptResult(
                        messages=[{
                            "role": "user", 
                            "content": TextContent(
                                text=f"Coordinate these agents for the following task:\n\nTask: {task}\n\nAgents: {', '.join(agents)}\n\nDetermine:\n1. Coordination type (parallel/sequential)\n2. Order of operations\n3. Data flow between agents"
                            )
                        }]
                    )
                
                else:
                    raise ValueError(f"Unknown prompt: {name}")
                    
            except Exception as e:
                logger.error(f"Prompt error: {e}")
                return GetPromptResult(
                    messages=[{
                        "role": "system",
                        "content": TextContent(text=f"Error: {str(e)}")
                    }]
                )
    
    async def initialize_agent(self):
        """Initialize the Agent 17 instance"""
        try:
            # Get blockchain config from environment
            blockchain_config = {
                "private_key": os.getenv("AGENT17_PRIVATE_KEY"),
                "contract_address": os.getenv("A2A_CONTRACT_ADDRESS"),
                "rpc_url": os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
            }
            
            # Create agent instance
            self.agent = create_agent17_chat_agent(
                base_url="http://localhost:8017",
                blockchain_config=blockchain_config
            )
            
            # Initialize agent
            await self.agent.initialize()
            
            logger.info("Agent 17 MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def run(self):
        """Run the MCP server"""
        try:
            # Initialize agent
            await self.initialize_agent()
            
            # Start MCP server
            async with self.server:
                logger.info("Agent 17 MCP Server running on stdio")
                # Keep server running
                await asyncio.Event().wait()
                
        except Exception as e:
            logger.error(f"MCP server error: {e}")
            raise
        finally:
            if self.agent:
                await self.agent.shutdown()


async def main():
    """Main entry point"""
    server = Agent17MCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)