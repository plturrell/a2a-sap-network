"""
A2A Registry Handler - Blockchain-based agent registry
Replaces HTTP-based registry with A2A protocol compliant implementation
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from app.a2a.sdk.agentBase import A2AAgentBase
from app.a2a.sdk import a2a_handler, a2a_skill
from app.a2a.sdk.types import A2AMessage, MessageRole, AgentConfig
from app.a2a.sdk.utils import create_agent_id, create_error_response, create_success_response
from app.a2a.core.security_base import SecureA2AAgent

# Import models from existing registry
from .models import (
    AgentCard, AgentRegistrationRequest, AgentRegistrationResponse,
    AgentSearchRequest, AgentSearchResponse, AgentDetails,
    AgentHealthResponse, AgentMetricsResponse, SystemHealthResponse,
    WorkflowMatchRequest, WorkflowMatchResponse,
    HealthStatus, AgentType
)

logger = logging.getLogger(__name__)


class A2ARegistryAgent(SecureA2AAgent):
    """
    A2A Registry Agent - Manages agent registration and discovery via blockchain
    """
    
    def __init__(self):
        config = AgentConfig(
            agent_id=create_agent_id("a2a-registry"),
            name="A2A Registry Agent",
            description="Blockchain-based agent registry for registration, discovery, and orchestration",
            version="2.0.0"
        )
        
        super().__init__(config)
        
        # Initialize security features
        self._init_security_features()
        self._init_rate_limiting()
        self._init_input_validation()
        
        # Registry storage (blockchain will be source of truth)
        self.local_cache = {}  # Local cache for performance
        self.agent_registry = {}  # agent_id -> AgentRegistrationRecord
        self.health_history = {}  # agent_id -> List[health_records]
        self.workflows = {}  # workflow_id -> workflow_data
        
        # Blockchain storage keys
        self.REGISTRY_KEY = "a2a:registry:agents"
        self.HEALTH_KEY = "a2a:registry:health"
        self.WORKFLOW_KEY = "a2a:registry:workflows"
        
        logger.info(f"Initialized A2A Registry Agent: {self.agent_id}")
    
    @a2a_handler("REGISTER_AGENT")
    async def handle_register_agent(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent registration requests"""
        try:
            # Extract registration data
            registration_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    registration_data = part.data
                    break
            
            if not registration_data:
                return create_error_response("No registration data provided")
            
            # Validate required fields
            required_fields = ["name", "description", "version", "endpoint", "capabilities"]
            for field in required_fields:
                if field not in registration_data:
                    return create_error_response(f"Missing required field: {field}")
            
            # Security validation
            is_valid, error_msg = self.validate_input(registration_data)
            if not is_valid:
                return create_error_response(f"Invalid input: {error_msg}")
            
            # Rate limiting
            client_id = registration_data.get('agent_id', message.metadata.get('sender_id', 'unknown'))
            if not self.check_rate_limit(client_id, 'registration'):
                return create_error_response("Rate limit exceeded for registration")
            
            # Create agent card
            agent_card = AgentCard(
                agent_id=registration_data.get('agent_id', create_agent_id(registration_data['name'])),
                name=registration_data['name'],
                description=registration_data['description'],
                version=registration_data['version'],
                endpoint=registration_data['endpoint'],
                type=registration_data.get('type', AgentType.SPECIALIZED),
                capabilities=registration_data['capabilities'],
                skills=registration_data.get('skills', []),
                tags=registration_data.get('tags', []),
                inputModes=registration_data.get('inputModes', []),
                outputModes=registration_data.get('outputModes', []),
                maxConcurrentRequests=registration_data.get('maxConcurrentRequests', 10),
                timeout=registration_data.get('timeout', 30),
                rateLimits=registration_data.get('rateLimits', {})
            )
            
            # Store in local cache
            self.agent_registry[agent_card.agent_id] = {
                'card': agent_card,
                'status': HealthStatus.HEALTHY,
                'registered_at': datetime.utcnow(),
                'last_heartbeat': datetime.utcnow(),
                'metadata': registration_data.get('metadata', {})
            }
            
            # Store on blockchain
            await self._store_on_blockchain(
                f"{self.REGISTRY_KEY}:{agent_card.agent_id}",
                agent_card.dict()
            )
            
            # Audit log
            self._audit_log('agent_registration', {
                'agent_id': agent_card.agent_id,
                'name': agent_card.name,
                'capabilities': agent_card.capabilities
            })
            
            logger.info(f"Registered agent: {agent_card.agent_id} ({agent_card.name})")
            
            return create_success_response({
                'agent_id': agent_card.agent_id,
                'status': 'registered',
                'registration_time': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return create_error_response(f"Registration failed: {str(e)}")
    
    @a2a_handler("SEARCH_AGENTS")
    async def handle_search_agents(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent search/discovery requests"""
        try:
            # Extract search criteria
            search_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    search_data = part.data
                    break
            
            if not search_data:
                search_data = {}  # Empty search returns all agents
            
            # Build search criteria
            skills_filter = search_data.get('skills', [])
            tags_filter = search_data.get('tags', [])
            type_filter = search_data.get('agent_type')
            status_filter = search_data.get('status')
            capabilities_filter = search_data.get('capabilities', [])
            
            # Search through registry
            matching_agents = []
            
            for agent_id, record in self.agent_registry.items():
                agent_card = record['card']
                
                # Apply filters
                if skills_filter and not any(skill in agent_card.skills for skill in skills_filter):
                    continue
                
                if tags_filter and not any(tag in agent_card.tags for tag in tags_filter):
                    continue
                
                if type_filter and agent_card.type != type_filter:
                    continue
                
                if status_filter and record['status'] != status_filter:
                    continue
                
                if capabilities_filter:
                    agent_capabilities = []
                    for cap in agent_card.capabilities:
                        if isinstance(cap, dict):
                            agent_capabilities.append(cap.get('name', ''))
                        else:
                            agent_capabilities.append(str(cap))
                    
                    if not any(cap in agent_capabilities for cap in capabilities_filter):
                        continue
                
                # Add to results
                matching_agents.append({
                    'agent_id': agent_card.agent_id,
                    'name': agent_card.name,
                    'description': agent_card.description,
                    'type': agent_card.type,
                    'capabilities': agent_card.capabilities,
                    'skills': agent_card.skills,
                    'tags': agent_card.tags,
                    'status': record['status'],
                    'endpoint': agent_card.endpoint
                })
            
            logger.info(f"Search returned {len(matching_agents)} agents")
            
            return create_success_response({
                'agents': matching_agents,
                'total_count': len(matching_agents),
                'search_criteria': search_data
            })
            
        except Exception as e:
            logger.error(f"Agent search failed: {e}")
            return create_error_response(f"Search failed: {str(e)}")
    
    @a2a_handler("GET_AGENT_DETAILS")
    async def handle_get_agent_details(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle requests for detailed agent information"""
        try:
            # Extract agent ID
            request_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    request_data = part.data
                    break
            
            if not request_data or 'agent_id' not in request_data:
                return create_error_response("No agent ID provided")
            
            agent_id = request_data['agent_id']
            
            # Check local cache first
            if agent_id in self.agent_registry:
                record = self.agent_registry[agent_id]
                agent_card = record['card']
                
                # Get health information
                health_info = self._get_agent_health(agent_id)
                
                details = {
                    'agent_id': agent_card.agent_id,
                    'name': agent_card.name,
                    'description': agent_card.description,
                    'version': agent_card.version,
                    'type': agent_card.type,
                    'endpoint': agent_card.endpoint,
                    'capabilities': agent_card.capabilities,
                    'skills': agent_card.skills,
                    'tags': agent_card.tags,
                    'inputModes': agent_card.inputModes,
                    'outputModes': agent_card.outputModes,
                    'maxConcurrentRequests': agent_card.maxConcurrentRequests,
                    'timeout': agent_card.timeout,
                    'rateLimits': agent_card.rateLimits,
                    'status': record['status'],
                    'health': health_info,
                    'registered_at': record['registered_at'].isoformat(),
                    'last_heartbeat': record['last_heartbeat'].isoformat()
                }
                
                return create_success_response(details)
            
            # Try to fetch from blockchain if not in cache
            blockchain_data = await self._fetch_from_blockchain(f"{self.REGISTRY_KEY}:{agent_id}")
            if blockchain_data:
                return create_success_response(blockchain_data)
            
            return create_error_response(f"Agent {agent_id} not found")
            
        except Exception as e:
            logger.error(f"Get agent details failed: {e}")
            return create_error_response(f"Failed to get agent details: {str(e)}")
    
    @a2a_handler("UPDATE_AGENT")
    async def handle_update_agent(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent update requests"""
        try:
            # Extract update data
            update_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    update_data = part.data
                    break
            
            if not update_data or 'agent_id' not in update_data:
                return create_error_response("No agent ID provided for update")
            
            agent_id = update_data['agent_id']
            
            # Check if agent exists
            if agent_id not in self.agent_registry:
                return create_error_response(f"Agent {agent_id} not found")
            
            # Security validation
            is_valid, error_msg = self.validate_input(update_data)
            if not is_valid:
                return create_error_response(f"Invalid input: {error_msg}")
            
            # Update agent record
            record = self.agent_registry[agent_id]
            agent_card = record['card']
            
            # Update allowed fields
            updateable_fields = [
                'description', 'version', 'endpoint', 'capabilities',
                'skills', 'tags', 'inputModes', 'outputModes',
                'maxConcurrentRequests', 'timeout', 'rateLimits'
            ]
            
            for field in updateable_fields:
                if field in update_data:
                    setattr(agent_card, field, update_data[field])
            
            # Update timestamp
            record['last_heartbeat'] = datetime.utcnow()
            
            # Update on blockchain
            await self._store_on_blockchain(
                f"{self.REGISTRY_KEY}:{agent_id}",
                agent_card.dict()
            )
            
            # Audit log
            self._audit_log('agent_update', {
                'agent_id': agent_id,
                'updated_fields': list(update_data.keys())
            })
            
            logger.info(f"Updated agent: {agent_id}")
            
            return create_success_response({
                'agent_id': agent_id,
                'status': 'updated',
                'update_time': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Agent update failed: {e}")
            return create_error_response(f"Update failed: {str(e)}")
    
    @a2a_handler("DEREGISTER_AGENT")
    async def handle_deregister_agent(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent deregistration requests"""
        try:
            # Extract agent ID
            request_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    request_data = part.data
                    break
            
            if not request_data or 'agent_id' not in request_data:
                return create_error_response("No agent ID provided")
            
            agent_id = request_data['agent_id']
            
            # Check if agent exists
            if agent_id not in self.agent_registry:
                return create_error_response(f"Agent {agent_id} not found")
            
            # Remove from local cache
            del self.agent_registry[agent_id]
            
            # Remove from blockchain
            await self._remove_from_blockchain(f"{self.REGISTRY_KEY}:{agent_id}")
            
            # Audit log
            self._audit_log('agent_deregistration', {
                'agent_id': agent_id,
                'deregistered_at': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Deregistered agent: {agent_id}")
            
            return create_success_response({
                'agent_id': agent_id,
                'status': 'deregistered',
                'deregistration_time': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Agent deregistration failed: {e}")
            return create_error_response(f"Deregistration failed: {str(e)}")
    
    @a2a_handler("AGENT_HEALTH_CHECK")
    async def handle_agent_health_check(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle agent health check requests"""
        try:
            # Extract agent ID
            request_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    request_data = part.data
                    break
            
            if not request_data or 'agent_id' not in request_data:
                return create_error_response("No agent ID provided")
            
            agent_id = request_data['agent_id']
            
            # Check if agent exists
            if agent_id not in self.agent_registry:
                return create_error_response(f"Agent {agent_id} not found")
            
            # Get health information
            health_info = self._get_agent_health(agent_id)
            
            # Update last heartbeat
            self.agent_registry[agent_id]['last_heartbeat'] = datetime.utcnow()
            
            return create_success_response({
                'agent_id': agent_id,
                'health': health_info,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return create_error_response(f"Health check failed: {str(e)}")
    
    @a2a_handler("WORKFLOW_MATCH")
    async def handle_workflow_match(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Handle workflow matching requests"""
        try:
            # Extract workflow requirements
            workflow_data = None
            for part in message.parts:
                if part.kind == "data" and part.data:
                    workflow_data = part.data
                    break
            
            if not workflow_data or 'stages' not in workflow_data:
                return create_error_response("No workflow stages provided")
            
            stages = workflow_data['stages']
            matched_workflow = []
            
            # Match agents for each stage
            for stage in stages:
                stage_name = stage.get('name', 'Unknown')
                required_capabilities = stage.get('required_capabilities', [])
                required_skills = stage.get('required_skills', [])
                
                # Find matching agents
                matching_agents = []
                for agent_id, record in self.agent_registry.items():
                    agent_card = record['card']
                    
                    # Check if agent is healthy
                    if record['status'] != HealthStatus.HEALTHY:
                        continue
                    
                    # Check capabilities
                    agent_capabilities = []
                    for cap in agent_card.capabilities:
                        if isinstance(cap, dict):
                            agent_capabilities.append(cap.get('name', ''))
                        else:
                            agent_capabilities.append(str(cap))
                    
                    has_required_caps = all(
                        cap in agent_capabilities for cap in required_capabilities
                    )
                    
                    # Check skills
                    has_required_skills = all(
                        skill in agent_card.skills for skill in required_skills
                    )
                    
                    if has_required_caps and has_required_skills:
                        matching_agents.append({
                            'agent_id': agent_card.agent_id,
                            'name': agent_card.name,
                            'match_score': 1.0  # Simple scoring for now
                        })
                
                matched_workflow.append({
                    'stage': stage_name,
                    'matching_agents': matching_agents[:5],  # Top 5 matches
                    'total_matches': len(matching_agents)
                })
            
            return create_success_response({
                'workflow_id': workflow_data.get('workflow_id', str(datetime.utcnow().timestamp())),
                'matched_stages': matched_workflow,
                'total_stages': len(stages)
            })
            
        except Exception as e:
            logger.error(f"Workflow matching failed: {e}")
            return create_error_response(f"Workflow matching failed: {str(e)}")
    
    @a2a_skill("registry_statistics")
    async def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            total_agents = len(self.agent_registry)
            healthy_agents = sum(
                1 for record in self.agent_registry.values()
                if record['status'] == HealthStatus.HEALTHY
            )
            
            # Capability distribution
            capability_counts = {}
            skill_counts = {}
            
            for record in self.agent_registry.values():
                agent_card = record['card']
                
                # Count capabilities
                for cap in agent_card.capabilities:
                    cap_name = cap.get('name', str(cap)) if isinstance(cap, dict) else str(cap)
                    capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
                
                # Count skills
                for skill in agent_card.skills:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            return {
                'total_agents': total_agents,
                'healthy_agents': healthy_agents,
                'unhealthy_agents': total_agents - healthy_agents,
                'health_percentage': (healthy_agents / max(total_agents, 1)) * 100,
                'top_capabilities': sorted(
                    capability_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                'top_skills': sorted(
                    skill_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                'registry_uptime': (datetime.utcnow() - self.start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Failed to get registry statistics: {e}")
            return {'error': str(e)}
    
    def _get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get agent health information"""
        if agent_id not in self.agent_registry:
            return {'status': 'unknown'}
        
        record = self.agent_registry[agent_id]
        last_heartbeat = record['last_heartbeat']
        time_since_heartbeat = (datetime.utcnow() - last_heartbeat).total_seconds()
        
        # Determine health status based on heartbeat
        if time_since_heartbeat < 300:  # 5 minutes
            status = HealthStatus.HEALTHY
        elif time_since_heartbeat < 900:  # 15 minutes
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        # Update status
        record['status'] = status
        
        return {
            'status': status,
            'last_heartbeat': last_heartbeat.isoformat(),
            'time_since_heartbeat': time_since_heartbeat,
            'checks_passed': time_since_heartbeat < 300
        }
    
    async def _store_on_blockchain(self, key: str, data: Any):
        """Store data on blockchain (placeholder)"""
        # In production, this would use actual blockchain storage
        logger.debug(f"Storing on blockchain: {key}")
        self.local_cache[key] = data
    
    async def _fetch_from_blockchain(self, key: str) -> Optional[Any]:
        """Fetch data from blockchain (placeholder)"""
        # In production, this would fetch from actual blockchain
        return self.local_cache.get(key)
    
    async def _remove_from_blockchain(self, key: str):
        """Remove data from blockchain (placeholder)"""
        # In production, this would remove from actual blockchain
        if key in self.local_cache:
            del self.local_cache[key]
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.utcnow()


# Create singleton instance
registry_agent = A2ARegistryAgent()


def get_registry_agent() -> A2ARegistryAgent:
    """Get the singleton registry agent instance"""
    return registry_agent