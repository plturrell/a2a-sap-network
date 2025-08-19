"""
Mock implementations for missing dependencies
Used only for integration testing when actual dependencies are not available
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MockEtcd3Client:
    """Mock etcd3 client for testing"""

    def __init__(self):
        self.data = {}
        logger.info("Using mock etcd3 client for testing")

    async def put(self, key: str, value: str) -> None:
        """Mock put operation"""
        self.data[key] = value

    async def get(self, key: str) -> Optional[str]:
        """Mock get operation"""
        return self.data.get(key)

    async def delete(self, key: str) -> None:
        """Mock delete operation"""
        if key in self.data:
            del self.data[key]

    async def get_prefix(self, prefix: str) -> List[tuple]:
        """Mock get prefix operation"""
        return [(k, v) for k, v in self.data.items() if k.startswith(prefix)]


class MockDistributedStorageClient:
    """Mock distributed storage client"""

    def __init__(self):
        self.storage = {}
        logger.info("Using mock distributed storage for testing")

    async def store(self, key: str, data: Any) -> bool:
        """Store data"""
        try:
            self.storage[key] = json.dumps(data) if not isinstance(data, str) else data
            return True
        except Exception as e:
            logger.error(f"Mock storage error: {e}")
            return False

    async def retrieve(self, key: str) -> Any:
        """Retrieve data"""
        if key in self.storage:
            data = self.storage[key]
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data
        return None


class MockNetworkConnector:
    """Mock network connector"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.connected = True
        logger.info(f"Mock network connector initialized for {base_url}")

    async def send_message(self, message: Any) -> Dict[str, Any]:
        """Mock send message"""
        return {
            "success": True,
            "message_id": f"mock_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def check_connection(self) -> bool:
        """Check connection status"""
        return self.connected


class MockAgentRegistrar:
    """Mock agent registrar"""

    def __init__(self):
        self.registered_agents = {}
        logger.info("Mock agent registrar initialized")

    async def register_agent(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register an agent"""
        agent_id = agent_info.get("agent_id")
        if agent_id:
            self.registered_agents[agent_id] = agent_info
            return {
                "success": True,
                "agent_id": agent_id,
                "registration_time": datetime.utcnow().isoformat(),
            }
        return {"success": False, "error": "Missing agent_id"}

    async def is_agent_registered(self, agent_id: str) -> bool:
        """Check if agent is registered"""
        return agent_id in self.registered_agents


class MockServiceDiscovery:
    """Mock service discovery"""

    def __init__(self):
        self.services = {
            "data-processing": [
                {
                    "agent_id": "agent0",
                    "name": "Data Product Agent",
                    "endpoint": "http://localhost:8001",
                },
                {
                    "agent_id": "agent1",
                    "name": "Data Standardization Agent",
                    "endpoint": "http://localhost:8002",
                },
            ],
            "ai-processing": [
                {
                    "agent_id": "agent2",
                    "name": "AI Preparation Agent",
                    "endpoint": "http://localhost:8003",
                },
                {
                    "agent_id": "agent3",
                    "name": "Vector Processing Agent",
                    "endpoint": "http://localhost:8004",
                },
            ],
        }
        logger.info("Mock service discovery initialized")

    async def discover_agents(self, capabilities: List[str] = None) -> List[Dict[str, Any]]:
        """Discover agents by capabilities"""
        if not capabilities:
            # Return all agents
            all_agents = []
            for agents in self.services.values():
                all_agents.extend(agents)
            return all_agents

        discovered = []
        for capability in capabilities:
            if capability in self.services:
                discovered.extend(self.services[capability])
        return discovered


class MockMessageBroker:
    """Mock message broker"""

    def __init__(self):
        self.messages = {}
        logger.info("Mock message broker initialized")

    async def send_message(self, from_agent: str, to_agent: str, message: Any) -> Dict[str, Any]:
        """Send message between agents"""
        message_id = f"{from_agent}_{to_agent}_{datetime.utcnow().timestamp()}"
        self.messages[message_id] = {
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return {"success": True, "message_id": message_id}


class MockRequestSigner:
    """Mock request signer"""

    def __init__(self):
        logger.info("Mock request signer initialized")

    async def sign_request(self, request: Dict[str, Any]) -> str:
        """Sign a request"""
        # Simple mock signature
        return f"mock_signature_{hash(json.dumps(request, sort_keys=True))}"

    async def verify_signature(self, request: Dict[str, Any], signature: str) -> bool:
        """Verify a signature"""
        expected = f"mock_signature_{hash(json.dumps(request, sort_keys=True))}"
        return signature == expected

    async def sign_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Sign a message"""
        signature = await self.sign_request(message)
        return {**message, "signature": signature}

    async def verify_message(self, signed_message: Dict[str, Any]) -> bool:
        """Verify a signed message"""
        signature = signed_message.get("signature")
        if not signature:
            return False

        message = {k: v for k, v in signed_message.items() if k != "signature"}
        return await self.verify_signature(message, signature)
