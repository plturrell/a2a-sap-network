"""
Router to A2A Migration Utility
Automated tool to convert REST routers to A2A blockchain handlers
"""

import os
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class RouterToA2AMigrator:
    """Utility to migrate REST routers to A2A handlers"""

    # Common REST endpoint patterns to A2A operations mapping
    ENDPOINT_MAPPINGS = {
        # Standard endpoints
        r"get_agent_card": "get_agent_card",
        r"json_rpc_handler": "json_rpc",
        r"rest_message_handler": "process_message",
        r"get_task_status": "get_task_status",
        r"get_queue_status": "get_queue_status",
        r"cancel_message": "cancel_message",
        r"health_check": "health_check",
        r"process_.*": "process_{operation}",
        r"handle_.*": "{operation}",
        r"execute.*": "execute_{operation}",
        r".*_status": "get_{operation}_status",
    }

    # Template for A2A handler
    A2A_HANDLER_TEMPLATE = '''"""
A2A-Compliant Message Handler for {agent_name}
Replaces REST endpoints with blockchain-based messaging

A2A PROTOCOL COMPLIANCE:
This handler ensures all agent communication goes through the A2A blockchain
messaging system. No direct HTTP endpoints are exposed.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....core.secure_agent_base import SecureA2AAgent, SecureAgentConfig
from ....sdk.a2aNetworkClient import A2ANetworkClient
from .{sdk_import} import {sdk_class}

logger = logging.getLogger(__name__)


class {class_name}(SecureA2AAgent):
    """
    A2A-compliant handler for {agent_name}
    All communication through blockchain messaging only
    """

    def __init__(self, agent_sdk: {sdk_class}):
        """Initialize A2A handler with agent SDK"""
        # Configure secure agent
        config = SecureAgentConfig(
            agent_id="{agent_id}",
            agent_name="{agent_name}",
            agent_version="{version}",
            allowed_operations={{
{operations}
            }},
            enable_authentication=True,
            enable_rate_limiting=True,
            enable_input_validation=True,
            rate_limit_requests=100,
            rate_limit_window=60
        )

        super().__init__(config)

        self.agent_sdk = agent_sdk

        # Initialize A2A blockchain client
        self.a2a_client = A2ANetworkClient(
            agent_id=config.agent_id,
            private_key=os.getenv('A2A_PRIVATE_KEY'),
            rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
        )

        # Register message handlers
        self._register_handlers()

        logger.info(f"A2A-compliant handler initialized for {{config.agent_name}}")

    def _register_handlers(self):
        """Register A2A message handlers"""
{handlers}

    async def process_a2a_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Main entry point for A2A messages
        Routes messages to appropriate handlers based on operation
        """
        try:
            # Extract operation from message
            operation = None
            data = {{}}

            if message.parts and len(message.parts) > 0:
                part = message.parts[0]
                if part.data:
                    operation = part.data.get("operation")
                    data = part.data.get("data", {{}})

            if not operation:
                return self.create_secure_response(
                    "No operation specified in message",
                    status="error"
                )

            # Get handler for operation
            handler = self.handlers.get(operation)
            if not handler:
                return self.create_secure_response(
                    f"Unknown operation: {{operation}}",
                    status="error"
                )

            # Create context ID
            context_id = f"{{message.sender_id}}:{{operation}}:{{datetime.utcnow().timestamp()}}"

            # Process through handler
            return await handler(message, context_id, data)

        except Exception as e:
            logger.error(f"Failed to process A2A message: {{e}}")
            return self.create_secure_response(str(e), status="error")

    async def _log_blockchain_transaction(self, operation: str, data_hash: str, result_hash: str, context_id: str):
        """Log transaction to blockchain for audit trail"""
        try:
            transaction_data = {{
                "agent_id": self.config.agent_id,
                "operation": operation,
                "data_hash": data_hash,
                "result_hash": result_hash,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat()
            }}

            # Send to blockchain through A2A client
            await self.a2a_client.log_transaction(transaction_data)

        except Exception as e:
            logger.error(f"Failed to log blockchain transaction: {{e}}")

    def _hash_data(self, data: Any) -> str:
        """Create hash of data for blockchain logging"""
        import hashlib
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def _check_blockchain_connection(self) -> bool:
        """Check if blockchain connection is active"""
        try:
            return await self.a2a_client.is_connected()
        except Exception:
            return False

    async def start(self):
        """Start the A2A handler"""
        logger.info(f"Starting A2A handler for {{self.config.agent_name}}")

        # Connect to blockchain
        await self.a2a_client.connect()

        # Register agent on blockchain
        await self.a2a_client.register_agent({{
            "agent_id": self.config.agent_id,
            "agent_name": self.config.agent_name,
            "capabilities": list(self.config.allowed_operations),
            "version": self.config.agent_version
        }})

        logger.info(f"A2A handler started and registered on blockchain")

    async def stop(self):
        """Stop the A2A handler"""
        logger.info(f"Stopping A2A handler for {{self.config.agent_name}}")

        # Unregister from blockchain
        await self.a2a_client.unregister_agent(self.config.agent_id)

        # Disconnect
        await self.a2a_client.disconnect()

        # Parent cleanup
        await self.shutdown()

        logger.info(f"A2A handler stopped")


# Factory function to create A2A handler
def create_{agent_id}_a2a_handler(agent_sdk: {sdk_class}) -> {class_name}:
    """Create A2A-compliant handler for {agent_name}"""
    return {class_name}(agent_sdk)


# Example usage for migration
"""
To migrate from REST endpoints to A2A messaging:

1. Replace router initialization:
   # OLD: router = APIRouter(...)
   # NEW:
   handler = create_{agent_id}_a2a_handler({agent_id}_sdk)

2. Replace FastAPI app with A2A listener:
   # OLD: app.include_router(router)
   # NEW:
   await handler.start()

3. Process messages through A2A:
   # Messages arrive through blockchain
   result = await handler.process_a2a_message(a2a_message)
"""'''

    HANDLER_TEMPLATE = '''
        @self.secure_handler("{operation}")
        async def handle_{operation}(self, message: A2AMessage, context_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """{doc_string}"""
            try:
{implementation}

                # Log blockchain transaction
                await self._log_blockchain_transaction(
                    operation="{operation}",
                    data_hash=self._hash_data(data),
                    result_hash=self._hash_data(result),
                    context_id=context_id
                )

                return self.create_secure_response(result)

            except Exception as e:
                logger.error(f"Failed to {operation}: {{e}}")
                return self.create_secure_response(str(e), status="error")'''

    def __init__(self):
        self.migrations = []

    def extract_router_info(self, router_path: str) -> Dict[str, Any]:
        """Extract information from router file"""
        with open(router_path, 'r') as f:
            content = f.read()

        info = {
            'path': router_path,
            'agent_id': None,
            'agent_name': None,
            'prefix': None,
            'sdk_import': None,
            'sdk_class': None,
            'version': '1.0.0',
            'endpoints': []
        }

        # Extract agent ID from path
        path_match = re.search(r'/agent(\d+\w*)', router_path)
        if path_match:
            info['agent_id'] = f"agent{path_match.group(1)}"
        elif 'agentManager' in router_path:
            info['agent_id'] = 'agent_manager'
        elif 'calculationAgent' in router_path:
            info['agent_id'] = 'calculation_agent'
        elif 'catalogManager' in router_path:
            info['agent_id'] = 'catalog_manager'
        elif 'reasoningAgent' in router_path:
            info['agent_id'] = 'reasoning_agent'

        # Extract router prefix
        prefix_match = re.search(r'prefix="([^"]+)"', content)
        if prefix_match:
            info['prefix'] = prefix_match.group(1)

        # Extract tags for agent name
        tags_match = re.search(r'tags=\["([^"]+)"\]', content)
        if tags_match:
            info['agent_name'] = tags_match.group(1)

        # Extract SDK import
        sdk_match = re.search(r'from \.([\w]+) import ([\w]+)', content)
        if sdk_match:
            info['sdk_import'] = sdk_match.group(1)
            info['sdk_class'] = sdk_match.group(2)

        # Extract endpoints
        endpoint_pattern = re.compile(r'@router\.(get|post|put|delete|patch)\("([^"]+)"\).*?async def (\w+)', re.DOTALL)
        for match in endpoint_pattern.finditer(content):
            method, path, func_name = match.groups()
            info['endpoints'].append({
                'method': method,
                'path': path,
                'func_name': func_name,
                'operation': self._map_to_operation(func_name)
            })

        return info

    def _map_to_operation(self, func_name: str) -> str:
        """Map function name to A2A operation"""
        # Check exact mappings first
        for pattern, operation in self.ENDPOINT_MAPPINGS.items():
            if re.match(pattern, func_name):
                # Handle dynamic operation names
                if '{operation}' in operation:
                    # Extract operation part from function name
                    op_match = re.search(r'(?:process_|handle_|execute_|get_)(.+?)(?:_status)?$', func_name)
                    if op_match:
                        return operation.format(operation=op_match.group(1))
                return operation

        # Default: use function name as operation
        return func_name

    def generate_handler(self, router_info: Dict[str, Any]) -> str:
        """Generate A2A handler code from router info"""
        # Generate operations list
        operations = []
        for endpoint in router_info['endpoints']:
            operations.append(f'                "{endpoint["operation"]}"')
        operations_str = ',\n'.join(operations)

        # Generate handler methods
        handlers = []
        for endpoint in router_info['endpoints']:
            handler = self._generate_handler_method(endpoint, router_info)
            handlers.append(handler)
        handlers_str = '\n'.join(handlers)

        # Generate class name
        class_name = f"{router_info['agent_id'].title().replace('_', '')}A2AHandler"

        # Fill template
        handler_code = self.A2A_HANDLER_TEMPLATE.format(
            agent_name=router_info['agent_name'],
            agent_id=router_info['agent_id'],
            class_name=class_name,
            sdk_import=router_info['sdk_import'],
            sdk_class=router_info['sdk_class'],
            version=router_info['version'],
            operations=operations_str,
            handlers=handlers_str
        )

        return handler_code

    def _generate_handler_method(self, endpoint: Dict[str, Any], router_info: Dict[str, Any]) -> str:
        """Generate handler method for endpoint"""
        operation = endpoint['operation']

        # Generate appropriate implementation based on operation
        if operation == "get_agent_card":
            implementation = '''                agent_card = await self.agent_sdk.get_agent_card()
                result = agent_card'''
            doc_string = "Get agent card information"

        elif operation == "health_check":
            implementation = '''                health_status = {
                    "status": "healthy",
                    "agent": self.config.agent_name,
                    "version": self.config.agent_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "a2a_compliant": True,
                    "blockchain_connected": await self._check_blockchain_connection()
                }
                result = health_status'''
            doc_string = "Health check for agent"

        elif operation == "get_task_status":
            implementation = '''                task_id = data.get("task_id")
                if not task_id:
                    raise ValueError("task_id is required")

                status = await self.agent_sdk.get_task_status(task_id)
                result = status'''
            doc_string = "Get status of a specific task"

        elif operation == "process_message":
            implementation = '''                # Process message through agent SDK
                result = await self.agent_sdk.process_message(message, context_id)'''
            doc_string = "Process incoming message"

        else:
            # Generic implementation
            implementation = f'''                # TODO: Implement {operation} logic
                # Example: result = await self.agent_sdk.{endpoint['func_name']}(data)
                result = {{"status": "success", "operation": "{operation}"}}'''
            doc_string = f"Handle {operation} operation"

        return self.HANDLER_TEMPLATE.format(
            operation=operation,
            doc_string=doc_string,
            implementation=implementation
        )

    def migrate_router(self, router_path: str) -> Tuple[str, str]:
        """Migrate a single router to A2A handler"""
        # Extract router information
        router_info = self.extract_router_info(router_path)

        # Generate A2A handler code
        handler_code = self.generate_handler(router_info)

        # Generate output path
        router_dir = os.path.dirname(router_path)
        handler_filename = f"{router_info['agent_id']}A2AHandler.py"
        handler_path = os.path.join(router_dir, handler_filename)

        return handler_path, handler_code

    def migrate_all_routers(self, router_paths: List[str]) -> List[Dict[str, Any]]:
        """Migrate all routers to A2A handlers"""
        results = []

        for router_path in router_paths:
            try:
                handler_path, handler_code = self.migrate_router(router_path)

                # Write handler file
                with open(handler_path, 'w') as f:
                    f.write(handler_code)

                results.append({
                    'router_path': router_path,
                    'handler_path': handler_path,
                    'status': 'success'
                })

            except Exception as e:
                results.append({
                    'router_path': router_path,
                    'status': 'error',
                    'error': str(e)
                })

        return results


def main():
    """Run migration for all routers"""
    router_files = [
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent0DataProduct/active/agent0Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent1Standardization/active/agent1Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent2AiPreparation/active/agent2Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent3VectorProcessing/active/agent3Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent4CalcValidation/active/agent4Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent5QaValidation/active/agent5Router.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agentManager/active/agentManagerRouter.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/calculationAgent/active/calculationRouter.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/catalogManager/active/catalogManagerRouter.py",
        "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/reasoningAgent/active/agent9Router.py"
    ]

    migrator = RouterToA2AMigrator()
    results = migrator.migrate_all_routers(router_files)

    print("Migration Results:")
    for result in results:
        if result['status'] == 'success':
            print(f"✅ {result['router_path']} -> {result['handler_path']}")
        else:
            print(f"❌ {result['router_path']}: {result['error']}")


if __name__ == "__main__":
    main()
