"""
Standardized Agent Lifecycle Framework
Provides consistent lifecycle management, validation, and compliance for all A2A agents
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable, Type
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from uuid import uuid4
import inspect

# A2A imports
from ..a2a.sdk.agentBase import A2AAgentBase
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class LifecyclePhase(str, Enum):
    """Agent lifecycle phases"""
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class LifecycleValidationLevel(str, Enum):
    """Lifecycle validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class ResourceType(str, Enum):
    """Types of resources managed during lifecycle"""
    HTTP_CLIENT = "http_client"
    DATABASE_CONNECTION = "database_connection"
    REDIS_CONNECTION = "redis_connection"
    WEBSOCKET_CONNECTION = "websocket_connection"
    FILE_HANDLE = "file_handle"
    THREAD_POOL = "thread_pool"
    SUBPROCESS = "subprocess"
    NETWORK_SOCKET = "network_socket"
    CACHE = "cache"
    MEMORY_BUFFER = "memory_buffer"


@dataclass
class LifecycleResource:
    """Resource managed during agent lifecycle"""
    resource_id: str
    resource_type: ResourceType
    resource_instance: Any
    cleanup_method: Optional[str] = None
    cleanup_function: Optional[Callable] = None
    critical: bool = False
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LifecycleCheckpoint:
    """Lifecycle checkpoint for validation and recovery"""
    phase: LifecyclePhase
    timestamp: datetime
    success: bool
    duration_seconds: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleValidationResult:
    """Result of lifecycle validation"""
    valid: bool
    compliance_level: LifecycleValidationLevel
    score: float  # 0.0 to 1.0
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class LifecycleValidationRule(ABC):
    """Abstract base class for lifecycle validation rules"""

    @abstractmethod
    def get_name(self) -> str:
        """Get the rule name"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the rule description"""
        pass

    @abstractmethod
    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        """Validate the rule and return list of violations"""
        pass

    @abstractmethod
    def get_compliance_level(self) -> LifecycleValidationLevel:
        """Get the minimum compliance level for this rule"""
        pass


class InitializeMethodRule(LifecycleValidationRule):
    """Rule: Agent must implement initialize() method"""

    def get_name(self) -> str:
        return "initialize_method_required"

    def get_description(self) -> str:
        return "Agent must implement async initialize() method"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        if not hasattr(agent, 'initialize'):
            violations.append("Agent does not implement initialize() method")
        elif not asyncio.iscoroutinefunction(agent.initialize):
            violations.append("Agent initialize() method must be async")
        elif inspect.signature(agent.initialize).parameters:
            violations.append("Agent initialize() method should not require parameters")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.BASIC


class ShutdownMethodRule(LifecycleValidationRule):
    """Rule: Agent must implement shutdown() method"""

    def get_name(self) -> str:
        return "shutdown_method_required"

    def get_description(self) -> str:
        return "Agent must implement async shutdown() method"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        if not hasattr(agent, 'shutdown'):
            violations.append("Agent does not implement shutdown() method")
        elif not asyncio.iscoroutinefunction(agent.shutdown):
            violations.append("Agent shutdown() method must be async")
        elif inspect.signature(agent.shutdown).parameters:
            violations.append("Agent shutdown() method should not require parameters")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.BASIC


class ResourceManagementRule(LifecycleValidationRule):
    """Rule: Agent must properly manage resources"""

    def get_name(self) -> str:
        return "resource_management_required"

    def get_description(self) -> str:
        return "Agent must register and properly cleanup resources"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        # Check if agent has registered any resources
        agent_resources = lifecycle_manager.get_agent_resources(agent.agent_id)

        # Check for common resource patterns
        has_http_client = hasattr(agent, 'http_client')
        has_redis_client = hasattr(agent, 'redis_client') or hasattr(agent, 'redis')
        has_db_connection = hasattr(agent, 'db') or hasattr(agent, 'database')

        if has_http_client and not any(r.resource_type == ResourceType.HTTP_CLIENT for r in agent_resources):
            violations.append("HTTP client detected but not registered for lifecycle management")

        if has_redis_client and not any(r.resource_type == ResourceType.REDIS_CONNECTION for r in agent_resources):
            violations.append("Redis client detected but not registered for lifecycle management")

        if has_db_connection and not any(r.resource_type == ResourceType.DATABASE_CONNECTION for r in agent_resources):
            violations.append("Database connection detected but not registered for lifecycle management")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.STANDARD


class ErrorHandlingRule(LifecycleValidationRule):
    """Rule: Agent must implement proper error handling"""

    def get_name(self) -> str:
        return "error_handling_required"

    def get_description(self) -> str:
        return "Agent lifecycle methods must implement proper error handling"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        # Check initialize method error handling
        if hasattr(agent, 'initialize'):
            source = inspect.getsource(agent.initialize)
            if 'try:' not in source and 'except:' not in source:
                violations.append("initialize() method lacks error handling (try/except)")

        # Check shutdown method error handling
        if hasattr(agent, 'shutdown'):
            source = inspect.getsource(agent.shutdown)
            if 'try:' not in source and 'except:' not in source:
                violations.append("shutdown() method lacks error handling (try/except)")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.STANDARD


class TimeoutComplianceRule(LifecycleValidationRule):
    """Rule: Lifecycle methods must complete within reasonable timeouts"""

    def get_name(self) -> str:
        return "timeout_compliance_required"

    def get_description(self) -> str:
        return "Lifecycle methods must complete within specified timeout limits"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        # Check lifecycle history for timeout violations
        checkpoints = lifecycle_manager.get_agent_checkpoints(agent.agent_id)

        for checkpoint in checkpoints:
            if checkpoint.phase == LifecyclePhase.INITIALIZING and checkpoint.duration_seconds > 60:
                violations.append(f"Initialization took {checkpoint.duration_seconds:.1f}s (max: 60s)")
            elif checkpoint.phase == LifecyclePhase.STOPPING and checkpoint.duration_seconds > 30:
                violations.append(f"Shutdown took {checkpoint.duration_seconds:.1f}s (max: 30s)")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.STRICT


class TelemetryIntegrationRule(LifecycleValidationRule):
    """Rule: Agent must integrate with telemetry system"""

    def get_name(self) -> str:
        return "telemetry_integration_required"

    def get_description(self) -> str:
        return "Agent must integrate with A2A telemetry system"

    async def validate(self, agent: A2AAgentBase, lifecycle_manager: 'StandardizedLifecycleManager') -> List[str]:
        violations = []

        # Check if agent has telemetry enabled
        if not getattr(agent, 'enable_telemetry', False):
            violations.append("Agent does not have telemetry enabled")

        # Check for telemetry decorators on key methods
        key_methods = ['initialize', 'shutdown', 'process_message', 'execute_skill']

        for method_name in key_methods:
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                if not hasattr(method, '_traced'):
                    violations.append(f"Method {method_name} lacks telemetry tracing")

        return violations

    def get_compliance_level(self) -> LifecycleValidationLevel:
        return LifecycleValidationLevel.ENTERPRISE


class StandardizedLifecycleManager:
    """Manages standardized lifecycle for A2A agents"""

    def __init__(self, redis_config: RedisConfig = None):
        self.redis_client = RedisClient(redis_config or RedisConfig())

        # Agent tracking
        self.managed_agents: Dict[str, A2AAgentBase] = {}
        self.agent_phases: Dict[str, LifecyclePhase] = {}
        self.agent_resources: Dict[str, List[LifecycleResource]] = {}
        self.agent_checkpoints: Dict[str, List[LifecycleCheckpoint]] = {}

        # Validation rules
        self.validation_rules: List[LifecycleValidationRule] = [
            InitializeMethodRule(),
            ShutdownMethodRule(),
            ResourceManagementRule(),
            ErrorHandlingRule(),
            TimeoutComplianceRule(),
            TelemetryIntegrationRule()
        ]

        # Lifecycle hooks
        self.phase_hooks: Dict[LifecyclePhase, List[Callable]] = {
            phase: [] for phase in LifecyclePhase
        }

        # Configuration
        self.default_init_timeout = 60
        self.default_shutdown_timeout = 30
        self.resource_cleanup_timeout = 10

    async def initialize(self):
        """Initialize the lifecycle manager"""
        await self.redis_client.initialize()
        logger.info("Standardized lifecycle manager initialized")

    async def shutdown(self):
        """Shutdown the lifecycle manager"""
        # Shutdown all managed agents
        for agent_id in list(self.managed_agents.keys()):
            await self.shutdown_agent(agent_id)

        await self.redis_client.close()
        logger.info("Standardized lifecycle manager shut down")

    def register_agent(self, agent: A2AAgentBase):
        """Register an agent for lifecycle management"""
        agent_id = agent.agent_id

        self.managed_agents[agent_id] = agent
        self.agent_phases[agent_id] = LifecyclePhase.INITIALIZING
        self.agent_resources[agent_id] = []
        self.agent_checkpoints[agent_id] = []

        logger.info(f"Registered agent {agent_id} for lifecycle management")

    def register_phase_hook(self, phase: LifecyclePhase, hook: Callable):
        """Register a hook to be called during a lifecycle phase"""
        self.phase_hooks[phase].append(hook)
        logger.info(f"Registered hook for phase: {phase}")

    def register_resource(
        self,
        agent_id: str,
        resource_type: ResourceType,
        resource_instance: Any,
        cleanup_method: Optional[str] = None,
        cleanup_function: Optional[Callable] = None,
        critical: bool = False,
        timeout_seconds: int = 30
    ) -> str:
        """Register a resource for lifecycle management"""

        if agent_id not in self.agent_resources:
            self.agent_resources[agent_id] = []

        resource = LifecycleResource(
            resource_id=str(uuid4()),
            resource_type=resource_type,
            resource_instance=resource_instance,
            cleanup_method=cleanup_method,
            cleanup_function=cleanup_function,
            critical=critical,
            timeout_seconds=timeout_seconds
        )

        self.agent_resources[agent_id].append(resource)

        logger.info(f"Registered {resource_type} resource for agent {agent_id}")
        return resource.resource_id

    @trace_async("agent_initialize")
    async def initialize_agent(
        self,
        agent_id: str,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """Initialize an agent with standardized lifecycle"""

        if agent_id not in self.managed_agents:
            raise ValueError(f"Agent {agent_id} not registered")

        agent = self.managed_agents[agent_id]
        timeout = timeout_seconds or self.default_init_timeout

        add_span_attributes({
            "agent.id": agent_id,
            "lifecycle.phase": "initialize",
            "timeout.seconds": timeout
        })

        start_time = time.time()

        try:
            # Update phase
            self.agent_phases[agent_id] = LifecyclePhase.INITIALIZING
            await self._call_phase_hooks(LifecyclePhase.INITIALIZING, agent)

            # Run initialization with timeout
            await asyncio.wait_for(agent.initialize(), timeout=timeout)

            # Update phase to initialized
            self.agent_phases[agent_id] = LifecyclePhase.INITIALIZED
            await self._call_phase_hooks(LifecyclePhase.INITIALIZED, agent)

            # Record successful checkpoint
            duration = time.time() - start_time
            checkpoint = LifecycleCheckpoint(
                phase=LifecyclePhase.INITIALIZING,
                timestamp=datetime.utcnow(),
                success=True,
                duration_seconds=duration
            )
            self.agent_checkpoints[agent_id].append(checkpoint)

            # Store checkpoint in Redis
            await self._store_checkpoint(agent_id, checkpoint)

            logger.info(f"Agent {agent_id} initialized successfully in {duration:.2f}s")
            return True

        except asyncio.TimeoutError:
            error = f"Agent initialization timeout after {timeout}s"
            logger.error(f"Agent {agent_id}: {error}")

            self.agent_phases[agent_id] = LifecyclePhase.FAILED
            await self._record_failed_checkpoint(agent_id, LifecyclePhase.INITIALIZING, start_time, error)

            return False

        except Exception as e:
            error = f"Agent initialization failed: {str(e)}"
            logger.error(f"Agent {agent_id}: {error}")

            self.agent_phases[agent_id] = LifecyclePhase.FAILED
            await self._record_failed_checkpoint(agent_id, LifecyclePhase.INITIALIZING, start_time, error)

            return False

    @trace_async("agent_start")
    async def start_agent(self, agent_id: str) -> bool:
        """Start an agent"""

        if agent_id not in self.managed_agents:
            raise ValueError(f"Agent {agent_id} not registered")

        agent = self.managed_agents[agent_id]

        if self.agent_phases[agent_id] != LifecyclePhase.INITIALIZED:
            raise ValueError(f"Agent {agent_id} must be initialized before starting")

        try:
            self.agent_phases[agent_id] = LifecyclePhase.STARTING
            await self._call_phase_hooks(LifecyclePhase.STARTING, agent)

            # Start agent (if it has a start method)
            if hasattr(agent, 'start'):
                await agent.start()

            self.agent_phases[agent_id] = LifecyclePhase.RUNNING
            await self._call_phase_hooks(LifecyclePhase.RUNNING, agent)

            logger.info(f"Agent {agent_id} started successfully")
            return True

        except Exception as e:
            error = f"Agent start failed: {str(e)}"
            logger.error(f"Agent {agent_id}: {error}")

            self.agent_phases[agent_id] = LifecyclePhase.FAILED
            return False

    @trace_async("agent_shutdown")
    async def shutdown_agent(
        self,
        agent_id: str,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """Shutdown an agent with standardized lifecycle"""

        if agent_id not in self.managed_agents:
            logger.warning(f"Agent {agent_id} not registered for shutdown")
            return False

        agent = self.managed_agents[agent_id]
        timeout = timeout_seconds or self.default_shutdown_timeout

        add_span_attributes({
            "agent.id": agent_id,
            "lifecycle.phase": "shutdown",
            "timeout.seconds": timeout
        })

        start_time = time.time()

        try:
            # Update phase
            self.agent_phases[agent_id] = LifecyclePhase.STOPPING
            await self._call_phase_hooks(LifecyclePhase.STOPPING, agent)

            # Shutdown agent with timeout
            await asyncio.wait_for(agent.shutdown(), timeout=timeout)

            # Cleanup resources
            await self._cleanup_agent_resources(agent_id)

            # Update phase
            self.agent_phases[agent_id] = LifecyclePhase.STOPPED
            await self._call_phase_hooks(LifecyclePhase.STOPPED, agent)

            # Record successful checkpoint
            duration = time.time() - start_time
            checkpoint = LifecycleCheckpoint(
                phase=LifecyclePhase.STOPPING,
                timestamp=datetime.utcnow(),
                success=True,
                duration_seconds=duration
            )
            self.agent_checkpoints[agent_id].append(checkpoint)

            # Store checkpoint in Redis
            await self._store_checkpoint(agent_id, checkpoint)

            # Remove from managed agents
            del self.managed_agents[agent_id]

            logger.info(f"Agent {agent_id} shut down successfully in {duration:.2f}s")
            return True

        except asyncio.TimeoutError:
            error = f"Agent shutdown timeout after {timeout}s"
            logger.error(f"Agent {agent_id}: {error}")

            # Force cleanup resources
            await self._cleanup_agent_resources(agent_id, force=True)

            self.agent_phases[agent_id] = LifecyclePhase.FAILED
            await self._record_failed_checkpoint(agent_id, LifecyclePhase.STOPPING, start_time, error)

            return False

        except Exception as e:
            error = f"Agent shutdown failed: {str(e)}"
            logger.error(f"Agent {agent_id}: {error}")

            # Attempt resource cleanup anyway
            await self._cleanup_agent_resources(agent_id, force=True)

            self.agent_phases[agent_id] = LifecyclePhase.FAILED
            await self._record_failed_checkpoint(agent_id, LifecyclePhase.STOPPING, start_time, error)

            return False

    async def validate_agent_lifecycle(
        self,
        agent_id: str,
        compliance_level: LifecycleValidationLevel = LifecycleValidationLevel.STANDARD
    ) -> LifecycleValidationResult:
        """Validate agent lifecycle compliance"""

        if agent_id not in self.managed_agents:
            return LifecycleValidationResult(
                valid=False,
                compliance_level=compliance_level,
                score=0.0,
                violations=["Agent not registered for lifecycle management"]
            )

        agent = self.managed_agents[agent_id]
        violations = []
        warnings = []
        recommendations = []

        # Run applicable validation rules
        applicable_rules = [
            rule for rule in self.validation_rules
            if self._is_rule_applicable(rule, compliance_level)
        ]

        for rule in applicable_rules:
            try:
                rule_violations = await rule.validate(agent, self)
                violations.extend(rule_violations)

            except Exception as e:
                warnings.append(f"Validation rule {rule.get_name()} failed: {str(e)}")

        # Calculate compliance score
        total_rules = len(applicable_rules)
        failed_rules = len([v for v in violations if any(rule.get_name() in v for rule in applicable_rules)])
        score = max(0.0, (total_rules - failed_rules) / total_rules) if total_rules > 0 else 1.0

        # Generate recommendations
        if score < 0.8:
            recommendations.append("Consider implementing missing lifecycle methods")
        if score < 0.6:
            recommendations.append("Review and improve resource management practices")
        if score < 0.4:
            recommendations.append("Implement comprehensive error handling")

        return LifecycleValidationResult(
            valid=len(violations) == 0,
            compliance_level=compliance_level,
            score=score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations
        )

    def _is_rule_applicable(self, rule: LifecycleValidationRule, target_level: LifecycleValidationLevel) -> bool:
        """Check if a validation rule applies to the target compliance level"""
        level_hierarchy = {
            LifecycleValidationLevel.BASIC: 1,
            LifecycleValidationLevel.STANDARD: 2,
            LifecycleValidationLevel.STRICT: 3,
            LifecycleValidationLevel.ENTERPRISE: 4
        }

        return level_hierarchy[rule.get_compliance_level()] <= level_hierarchy[target_level]

    async def _cleanup_agent_resources(self, agent_id: str, force: bool = False):
        """Cleanup all resources for an agent"""
        if agent_id not in self.agent_resources:
            return

        resources = self.agent_resources[agent_id]

        # Sort by criticality (non-critical first)
        resources.sort(key=lambda r: r.critical)

        for resource in resources:
            try:
                timeout = resource.timeout_seconds if not force else 5

                if resource.cleanup_function:
                    # Use custom cleanup function
                    if asyncio.iscoroutinefunction(resource.cleanup_function):
                        await asyncio.wait_for(
                            resource.cleanup_function(resource.resource_instance),
                            timeout=timeout
                        )
                    else:
                        resource.cleanup_function(resource.resource_instance)

                elif resource.cleanup_method:
                    # Use method on the resource instance
                    method = getattr(resource.resource_instance, resource.cleanup_method)
                    if asyncio.iscoroutinefunction(method):
                        await asyncio.wait_for(method(), timeout=timeout)
                    else:
                        method()

                else:
                    # Try common cleanup methods
                    common_methods = ['close', 'shutdown', 'cleanup', 'aclose']

                    for method_name in common_methods:
                        if hasattr(resource.resource_instance, method_name):
                            method = getattr(resource.resource_instance, method_name)
                            try:
                                if asyncio.iscoroutinefunction(method):
                                    await asyncio.wait_for(method(), timeout=timeout)
                                else:
                                    method()
                                break
                            except Exception:
                                continue

                logger.debug(f"Cleaned up {resource.resource_type} resource for agent {agent_id}")

            except asyncio.TimeoutError:
                logger.warning(f"Timeout cleaning up {resource.resource_type} for agent {agent_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup {resource.resource_type} for agent {agent_id}: {e}")

        # Clear resources list
        self.agent_resources[agent_id] = []

    async def _call_phase_hooks(self, phase: LifecyclePhase, agent: A2AAgentBase):
        """Call registered hooks for a lifecycle phase"""
        hooks = self.phase_hooks.get(phase, [])

        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(agent, phase)
                else:
                    hook(agent, phase)
            except Exception as e:
                logger.error(f"Lifecycle hook failed for phase {phase}: {e}")

    async def _record_failed_checkpoint(self, agent_id: str, phase: LifecyclePhase, start_time: float, error: str):
        """Record a failed lifecycle checkpoint"""
        duration = time.time() - start_time
        checkpoint = LifecycleCheckpoint(
            phase=phase,
            timestamp=datetime.utcnow(),
            success=False,
            duration_seconds=duration,
            error=error
        )

        self.agent_checkpoints[agent_id].append(checkpoint)
        await self._store_checkpoint(agent_id, checkpoint)

    async def _store_checkpoint(self, agent_id: str, checkpoint: LifecycleCheckpoint):
        """Store checkpoint in Redis"""
        try:
            key = f"lifecycle_checkpoints:{agent_id}"
            data = {
                "phase": checkpoint.phase.value,
                "timestamp": checkpoint.timestamp.isoformat(),
                "success": checkpoint.success,
                "duration_seconds": checkpoint.duration_seconds,
                "error": checkpoint.error,
                "metadata": checkpoint.metadata
            }

            # Add to list with expiration
            await self.redis_client.lpush(key, json.dumps(data))
            await self.redis_client.ltrim(key, 0, 99)  # Keep last 100 checkpoints
            await self.redis_client.expire(key, timedelta(days=7))

        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")

    def get_agent_phase(self, agent_id: str) -> Optional[LifecyclePhase]:
        """Get current phase of an agent"""
        return self.agent_phases.get(agent_id)

    def get_agent_resources(self, agent_id: str) -> List[LifecycleResource]:
        """Get resources for an agent"""
        return self.agent_resources.get(agent_id, [])

    def get_agent_checkpoints(self, agent_id: str) -> List[LifecycleCheckpoint]:
        """Get checkpoints for an agent"""
        return self.agent_checkpoints.get(agent_id, [])

    async def get_lifecycle_status(self) -> Dict[str, Any]:
        """Get overall lifecycle status"""
        status = {
            "managed_agents": len(self.managed_agents),
            "phases": {},
            "total_resources": 0,
            "validation_rules": len(self.validation_rules)
        }

        # Count agents by phase
        for phase in LifecyclePhase:
            count = sum(1 for p in self.agent_phases.values() if p == phase)
            status["phases"][phase.value] = count

        # Count total resources
        for resources in self.agent_resources.values():
            status["total_resources"] += len(resources)

        return status


# Global lifecycle manager
_lifecycle_manager = None


async def initialize_standardized_lifecycle(redis_config: RedisConfig = None) -> StandardizedLifecycleManager:
    """Initialize global standardized lifecycle manager"""
    global _lifecycle_manager

    if _lifecycle_manager is None:
        _lifecycle_manager = StandardizedLifecycleManager(redis_config)
        await _lifecycle_manager.initialize()

    return _lifecycle_manager


async def get_lifecycle_manager() -> Optional[StandardizedLifecycleManager]:
    """Get the global lifecycle manager"""
    return _lifecycle_manager


async def shutdown_standardized_lifecycle():
    """Shutdown global standardized lifecycle manager"""
    global _lifecycle_manager

    if _lifecycle_manager:
        await _lifecycle_manager.shutdown()
        _lifecycle_manager = None


# Convenience decorators for agent lifecycle
def managed_resource(
    resource_type: ResourceType,
    cleanup_method: Optional[str] = None,
    critical: bool = False,
    timeout_seconds: int = 30
):
    """Decorator to automatically register a resource for lifecycle management"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            resource = await func(self, *args, **kwargs)

            if _lifecycle_manager and hasattr(self, 'agent_id'):
                _lifecycle_manager.register_resource(
                    agent_id=self.agent_id,
                    resource_type=resource_type,
                    resource_instance=resource,
                    cleanup_method=cleanup_method,
                    critical=critical,
                    timeout_seconds=timeout_seconds
                )

            return resource
        return wrapper
    return decorator


def lifecycle_compliant(compliance_level: LifecycleValidationLevel = LifecycleValidationLevel.STANDARD):
    """Decorator to mark an agent as lifecycle compliant"""
    def decorator(agent_class: Type[A2AAgentBase]):
        agent_class._lifecycle_compliance_level = compliance_level
        return agent_class
    return decorator
