
# A2A Protocol Compliant Client
from app.a2a.core.network_client import A2ANetworkClient

class A2AHttpReplacement:
    def __init__(self, agent_id):
        self.a2a_client = A2ANetworkClient(agent_id)
    
    async def send_request(self, target_agent, message_type, data):
        """Replace HTTP calls with A2A messaging"""
        return await self.a2a_client.send_a2a_message(
            to_agent=target_agent,
            message=data,
            message_type=message_type
        )
