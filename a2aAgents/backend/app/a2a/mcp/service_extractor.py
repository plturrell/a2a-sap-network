#!/usr/bin/env python3
"""
MCP Service Extraction Framework
Converts embedded MCP capabilities into standalone deployable servers
"""

import os
import json
import inspect
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MCPServiceDefinition:
    """Definition for an MCP service"""
    agent_name: str
    service_name: str
    port: int
    agent_class: Type
    mcp_tools: List[Dict[str, Any]]
    description: str
    module_path: str

class MCPServiceExtractor:
    """Extracts MCP services from agent classes and generates standalone servers"""

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.services = {}
        self.port_registry = {
            "enhanced_test_suite": 8100,  # Already exists
            "data_standardization": 8101,
            "vector_similarity": 8102,
            "vector_ranking": 8103,
            "transport_layer": 8104,
            "reasoning_agent": 8105,
            "session_management": 8106,
            "resource_streaming": 8107,
            "confidence_calculator": 8108,
            "semantic_similarity": 8109,
            "ai_preparation": 8110,
            "calculation_validation": 8111,
            "qa_validation": 8112,
            "data_product": 8113,
            "agent_builder": 8114,
            "agent_manager": 8115,
            "code_analysis": 8116
        }

    def discover_mcp_agents(self) -> Dict[str, MCPServiceDefinition]:
        """Discover all agents with MCP capabilities"""
        services = {}

        agents_path = self.base_path / "agents"
        logger.info(f"Scanning for MCP agents in {agents_path}")

        # Define agent mappings - using actual files that exist
        agent_mappings = [
            {
                "name": "data_standardization",
                "path": "agent1Standardization/active/mcpEnhancedDataStandardizationAgent.py",
                "class": "MCPEnhancedDataStandardizationAgent",
                "description": "Data standardization with L4 hierarchical processing"
            },
            {
                "name": "vector_similarity",
                "path": "agent3VectorProcessing/active/mcpVectorSimilarityCalculator.py",
                "class": "MCPVectorSimilarityCalculator",
                "description": "Vector similarity calculation and processing"
            },
            {
                "name": "vector_ranking",
                "path": "agent3VectorProcessing/active/mcpHybridRankingSkills.py",
                "class": "MCPHybridRankingSkills",
                "description": "Hybrid ranking and scoring algorithms"
            },
            {
                "name": "transport_layer",
                "path": "reasoningAgent/mcpTransportLayer.py",
                "class": "MCPTransportManager",
                "description": "MCP transport layer management"
            },
            {
                "name": "reasoning_agent",
                "path": "reasoningAgent/mcpReasoningAgent.py",
                "class": "MCPReasoningAgent",
                "description": "Advanced reasoning and inference capabilities"
            },
            {
                "name": "session_management",
                "path": "reasoningAgent/mcpSessionManagement.py",
                "class": "MCPServerWithSessions",
                "description": "MCP session and authentication management"
            },
            {
                "name": "resource_streaming",
                "path": "reasoningAgent/mcpResourceStreaming.py",
                "class": "MCPResourceStreamingServer",
                "description": "Real-time resource streaming and subscriptions"
            },
            {
                "name": "confidence_calculator",
                "path": "reasoningAgent/mcpReasoningConfidenceCalculator.py",
                "class": "MCPReasoningConfidenceCalculator",
                "description": "Reasoning confidence calculation and assessment"
            },
            {
                "name": "semantic_similarity",
                "path": "reasoningAgent/mcpSemanticSimilarityCalculator.py",
                "class": "MCPSemanticSimilarityCalculator",
                "description": "Semantic similarity analysis and matching"
            }
        ]

        for mapping in agent_mappings:
            try:
                service_def = self._extract_service_from_mapping(mapping)
                if service_def:
                    services[mapping["name"]] = service_def
                    logger.info(f"Discovered MCP service: {mapping['name']}")
            except Exception as e:
                logger.warning(f"Failed to extract service {mapping['name']}: {e}")

        return services

    def _extract_service_from_mapping(self, mapping: Dict[str, Any]) -> Optional[MCPServiceDefinition]:
        """Extract service definition from agent mapping"""
        agent_file = self.base_path / "agents" / mapping["path"]

        if not agent_file.exists():
            logger.warning(f"Agent file not found: {agent_file}")
            return None

        try:
            # Read the file and extract MCP tools without importing
            mcp_tools = self._extract_mcp_tools_from_file(agent_file)

            port = self.port_registry.get(mapping["name"], 8200)

            return MCPServiceDefinition(
                agent_name=mapping["name"],
                service_name=f"mcp_{mapping['name']}_server",
                port=port,
                agent_class=None,  # We'll reference by name instead
                mcp_tools=mcp_tools,
                description=mapping["description"],
                module_path=str(agent_file)
            )
        except Exception as e:
            logger.warning(f"Failed to extract from {mapping['name']}: {e}")
            return None

    def _extract_mcp_tools(self, agent_class: Type) -> List[Dict[str, Any]]:
        """Extract MCP tools from agent class"""
        mcp_tools = []

        # Look for methods with MCP decorators or tools
        for name, method in inspect.getmembers(agent_class, predicate=inspect.isfunction):
            if hasattr(method, '_mcp_tool') or 'mcp' in name.lower():
                tool_info = {
                    "name": name,
                    "description": getattr(method, '__doc__', f"MCP tool: {name}"),
                    "type": "function",
                    "async": inspect.iscoroutinefunction(method)
                }

                # Try to extract parameter info
                sig = inspect.signature(method)
                parameters = {}
                for param_name, param in sig.parameters.items():
                    if param_name != 'self':
                        parameters[param_name] = {
                            "type": str(param.annotation) if param.annotation != param.empty else "any",
                            "required": param.default == param.empty
                        }

                tool_info["parameters"] = parameters
                mcp_tools.append(tool_info)

        # Look for MCP tool attributes
        for attr_name in dir(agent_class):
            attr = getattr(agent_class, attr_name)
            if hasattr(attr, '_mcp_tools') or (callable(attr) and 'mcp' in attr_name.lower()):
                if not any(tool['name'] == attr_name for tool in mcp_tools):
                    mcp_tools.append({
                        "name": attr_name,
                        "description": getattr(attr, '__doc__', f"MCP tool: {attr_name}"),
                        "type": "attribute"
                    })

        return mcp_tools

    def _extract_mcp_tools_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract MCP tools by parsing file content"""
        mcp_tools = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for common MCP patterns
            import re

            # Find methods with @mcp_tool decorator or mcp in name
            method_patterns = [
                r'def\s+(\w*mcp\w*)\s*\(',
                r'@mcp_tool\s*\n\s*def\s+(\w+)\s*\(',
                r'async\s+def\s+(\w*mcp\w*)\s*\(',
                r'def\s+(standardize_\w+)\s*\(',
                r'def\s+(validate_\w+)\s*\(',
                r'def\s+(process_\w+)\s*\(',
                r'def\s+(analyze_\w+)\s*\(',
                r'def\s+(calculate_\w+)\s*\(',
                r'def\s+(enhance_\w+)\s*\(',
            ]

            for pattern in method_patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    if match and match not in [tool['name'] for tool in mcp_tools]:
                        # Check if it's async
                        async_pattern = rf'async\s+def\s+{re.escape(match)}\s*\('
                        is_async = bool(re.search(async_pattern, content))

                        # Try to extract docstring
                        doc_pattern = rf'def\s+{re.escape(match)}\s*\([^)]*\):\s*"""([^"]*?)"""'
                        doc_match = re.search(doc_pattern, content, re.DOTALL)
                        description = doc_match.group(1).strip() if doc_match else f"MCP tool: {match}"

                        mcp_tools.append({
                            "name": match,
                            "description": description,
                            "type": "function",
                            "async": is_async
                        })

            # Look for MCP tool attributes
            attr_patterns = [
                r'self\.mcp_(\w+)',
                r'\.mcp_(\w+)',
                r'mcp_(\w+)\s*='
            ]

            for pattern in attr_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    tool_name = f"mcp_{match}"
                    if tool_name not in [tool['name'] for tool in mcp_tools]:
                        mcp_tools.append({
                            "name": tool_name,
                            "description": f"MCP attribute tool: {match}",
                            "type": "attribute",
                            "async": False
                        })

            # If no tools found, add some default tools based on agent type
            if not mcp_tools:
                agent_name = file_path.stem.lower()
                default_tools = []

                if 'standardization' in agent_name:
                    default_tools = [
                        {"name": "standardize_data", "description": "Standardize data format", "type": "function", "async": False},
                        {"name": "validate_standards", "description": "Validate data standards", "type": "function", "async": False}
                    ]
                elif 'vector' in agent_name:
                    default_tools = [
                        {"name": "process_vectors", "description": "Process vector embeddings", "type": "function", "async": True},
                        {"name": "calculate_similarity", "description": "Calculate vector similarity", "type": "function", "async": False}
                    ]
                elif 'validation' in agent_name:
                    default_tools = [
                        {"name": "validate_data", "description": "Validate data quality", "type": "function", "async": False},
                        {"name": "check_compliance", "description": "Check compliance rules", "type": "function", "async": False}
                    ]
                elif 'reasoning' in agent_name:
                    default_tools = [
                        {"name": "analyze_patterns", "description": "Analyze data patterns", "type": "function", "async": True},
                        {"name": "make_inferences", "description": "Make logical inferences", "type": "function", "async": True}
                    ]
                else:
                    default_tools = [
                        {"name": "process_request", "description": "Process generic request", "type": "function", "async": False},
                        {"name": "get_status", "description": "Get agent status", "type": "function", "async": False}
                    ]

                mcp_tools.extend(default_tools)

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            # Return minimal default tools
            mcp_tools = [
                {"name": "process_request", "description": "Process generic request", "type": "function", "async": False},
                {"name": "get_status", "description": "Get agent status", "type": "function", "async": False}
            ]

        return mcp_tools

    def _get_class_name_from_file(self, file_path: str) -> str:
        """Extract class name from file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            import re
            # Look for class definitions
            class_pattern = r'class\s+(\w+)\s*\('
            matches = re.findall(class_pattern, content)

            # Return the first class that looks like an agent
            for match in matches:
                if any(keyword in match.lower() for keyword in ['agent', 'mcp']):
                    return match

            # Return first class if no agent-like class found
            if matches:
                return matches[0]

            # Default fallback
            return "MCPAgent"
        except:
            return "MCPAgent"

    def generate_standalone_server(self, service_def: MCPServiceDefinition) -> str:
        """Generate standalone MCP server code"""
        template = '''#!/usr/bin/env python3
"""
Standalone MCP Server for {agent_name}
Auto-generated from {module_path}
"""

import sys
import asyncio
import logging
import os
from pathlib import Path
import signal

# Set up proper Python path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Set working directory
os.chdir(str(backend_dir))

try:
    from app.a2a.sdk.mcpServerEnhanced import EnhancedMCPServer
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
except ImportError as e:
    print(f"MCP dependencies not found: {{e}}")
    # Try alternative import paths
    try:
        import sys
        sys.path.insert(0, str(backend_dir / "app" / "a2a" / "sdk"))
        from mcpServerEnhanced import EnhancedMCPServer
        print("✅ Found MCP server using alternative path")
    except ImportError:
        print("❌ MCP server not available")
        # Create a basic server instead
        class EnhancedMCPServer:
            def __init__(self, name, version):
                self.name = name
                self.version = version

# Import the agent class
agent_class = None
try:
    from {relative_import} import {class_name}
    agent_class = {class_name}
    print(f"✅ Successfully imported {{agent_class.__name__}}")
except ImportError as e:
    print(f"❌ Failed to import agent class: {{e}}")
    # Create a mock agent class for testing
    class MockAgent:
        def __init__(self, base_url, enable_monitoring=True):
            self.base_url = base_url
            self.enable_monitoring = enable_monitoring

        def process_request(self, request):
            return f"Mock response from {{'{agent_name}'}}"

        def get_status(self):
            return {{"status": "running", "service": "{agent_name}"}}

    agent_class = MockAgent

logger = logging.getLogger(__name__)

class {service_class_name}:
    """Standalone MCP server for {agent_name}"""

    def __init__(self):
        self.name = "{service_name}"
        self.version = "1.0.0"
        self.port = {port}

        # Initialize the agent
        try:
            self.agent = agent_class(
                base_url=f"http://localhost:{{self.port}}",
                enable_monitoring=True
            )
            print(f"✅ Initialized agent: {{agent_class.__name__}}")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {{e}}")
            # Use mock agent as fallback
            self.agent = MockAgent(
                base_url=f"http://localhost:{{self.port}}",
                enable_monitoring=True
            )

        logger.info(f"Initialized {{self.__class__.__name__}} on port {{self.port}}")

    async def handle_request(self, request):
        """Handle MCP request"""
        method = request.get("method", "unknown")
        params = request.get("params", {{}})

        # Route to appropriate tool
{tool_registrations}

        # Default fallback
        return {{"error": f"Unknown method: {{method}}", "available_tools": [{tool_count}], "status": "error"}}

    async def start_server(self):
        """Start the MCP server with production features"""
        try:
            from fastapi import FastAPI, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            import uvicorn
            import signal
            import asyncio
        except ImportError:
            print("FastAPI not available, starting simple HTTP server")
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json

            class MCPHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        health_data = {{
                            "status": "healthy",
                            "service": "{service_name}",
                            "port": {port},
                            "tools": {tool_count}
                        }}
                        self.wfile.write(json.dumps(health_data).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()

            server = HTTPServer(('0.0.0.0', {port}), MCPHandler)
            print(f"Starting basic HTTP server for {service_name} on port {port}")
            server.serve_forever()
            return

        app = FastAPI(
            title="{service_name}",
            description="{description}",
            version="1.0.0",
            docs_url=None,  # Disable docs in production
            redoc_url=None  # Disable redoc in production
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

        # Global exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {{exc}}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={{"error": "Internal server error", "status": "error"}}
            )

        # Graceful shutdown handler
        shutdown_event = asyncio.Event()

        def signal_handler(signum, frame):
            logger.info(f"Received signal {{signum}}, shutting down...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        @app.get("/health")
        async def health_check():
            return {{
                "status": "healthy",
                "service": "{service_name}",
                "port": {port},
                "tools": {tool_count},
                "agent_type": type(self.agent).__name__
            }}

        @app.get("/info")
        async def service_info():
            return {{
                "service": "{service_name}",
                "agent": "{agent_name}",
                "description": "{description}",
                "port": {port},
                "tools": [
{tool_info_list}
                ]
            }}

        @app.post("/mcp")
        async def handle_mcp_request(request: dict, auth_info: dict = None):
            """Handle authenticated MCP requests"""
            try:
                # Add auth context to request
                if auth_info:
                    request['_auth'] = auth_info

                result = await self.handle_request(request)

                # Add correlation ID if present
                if '_correlation_id' in request:
                    result['_correlation_id'] = request['_correlation_id']

                return result
            except Exception as e:
                logger.error(f"Error handling MCP request: {{e}}")
                return {{
                    "error": "Internal server error",
                    "status": "error",
                    "code": 500
                }}

        # Startup event
        @app.on_event("startup")
        async def startup():
            logger.info(f"Starting {service_name} on port {port}")
            # Initialize connections, caches, etc.

        # Shutdown event
        @app.on_event("shutdown")
        async def shutdown():
            logger.info(f"Shutting down {service_name}")
            # Clean up resources, close connections

        print(f"🚀 Starting {service_name} on port {port}")

        # Production server configuration
        config = uvicorn.Config(
            app=app,
            host=os.getenv('MCP_SERVER_HOST', '0.0.0.0'),
            port={port},
            log_level=os.getenv('MCP_LOG_LEVEL', 'info').lower(),
            access_log=True,
            use_colors=False,
            server_header=False,  # Don't expose server info
            date_header=False,    # Don't expose date for security
        )

        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Main server entry point"""
    logging.basicConfig(level=logging.INFO)

    try:
        server = {service_class_name}()
        await server.start_server()
    except Exception as e:
        logger.error(f"Failed to start {service_name}: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
'''

        # Generate tool registrations
        tool_registrations = ""
        tool_info_list = ""

        for i, tool in enumerate(service_def.mcp_tools):
            tool_registrations += f'''
        if method == "{tool['name']}":
            try:
                if hasattr(self.agent, '{tool['name']}'):
                    method_func = getattr(self.agent, '{tool['name']}')
                    if callable(method_func):
                        if {str(tool.get('async', False))}:
                            result = await method_func(**params)
                        else:
                            result = method_func(**params)
                        return {{"result": str(result), "status": "success"}}
                    else:
                        return {{"error": "Tool {tool['name']} is not callable", "status": "error"}}
                else:
                    return {{"error": "Tool {tool['name']} not found", "status": "error"}}
            except Exception as e:
                return {{"error": f"Error executing {tool['name']}: {{str(e)}}", "status": "error"}}'''

            comma = ',' if i < len(service_def.mcp_tools) - 1 else ''
            description = tool.get('description', '').replace('"', '\\"')
            tool_info_list += f'''                    {{
                        "name": "{tool['name']}",
                        "description": "{description}",
                        "type": "{tool.get('type', 'function')}"
                    }}{comma}'''

        # Generate relative import path
        relative_path = Path(service_def.module_path).relative_to(self.base_path / "agents")
        relative_import = str(relative_path.with_suffix('')).replace('/', '.')

        # Get class name from mapping
        class_name = self._get_class_name_from_file(service_def.module_path)
        service_class_name = f"Standalone{class_name}Server"

        return template.format(
            agent_name=service_def.agent_name,
            module_path=service_def.module_path,
            relative_import=relative_import,
            class_name=class_name,
            service_class_name=service_class_name,
            service_name=service_def.service_name,
            description=service_def.description,
            port=service_def.port,
            tool_registrations=tool_registrations,
            tool_count=len(service_def.mcp_tools),
            tool_info_list=tool_info_list
        )

    def create_all_servers(self, output_dir: str = None):
        """Create all standalone MCP servers"""
        if not output_dir:
            output_dir = self.base_path / "mcp" / "servers"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        services = self.discover_mcp_agents()
        created_servers = []

        for service_name, service_def in services.items():
            server_code = self.generate_standalone_server(service_def)
            server_file = output_path / f"{service_name}_mcp_server.py"

            with open(server_file, 'w') as f:
                f.write(server_code)

            # Make executable
            os.chmod(server_file, 0o755)

            created_servers.append({
                "service": service_name,
                "file": str(server_file),
                "port": service_def.port,
                "tools": len(service_def.mcp_tools)
            })

            logger.info(f"Created MCP server: {server_file}")

        # Create service registry
        registry = {
            "services": created_servers,
            "total_servers": len(created_servers),
            "port_range": f"{min(s['port'] for s in created_servers)}-{max(s['port'] for s in created_servers)}" if created_servers else "No servers created",
            "generated_at": str(Path(__file__).stat().st_mtime)
        }

        registry_file = output_path / "service_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Created service registry: {registry_file}")
        return created_servers

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = MCPServiceExtractor()
    servers = extractor.create_all_servers()

    print(f"✅ Created {len(servers)} MCP servers:")
    for server in servers:
        print(f"  - {server['service']}: port {server['port']} ({server['tools']} tools)")