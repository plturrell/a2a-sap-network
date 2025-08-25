"""
Comprehensive Disaster Recovery System for A2A Agents
Provides backup strategies, failover procedures, and automated recovery
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import gzip
import os
import hashlib
import statistics
from abc import ABC, abstractmethod

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..a2a.sdk.agentBase import A2AAgentBase
from ..clients.redisClient import RedisClient, RedisConfig

logger = logging.getLogger(__name__)


class RecoveryStrategy(str, Enum):
    """Disaster recovery strategies"""
    BACKUP_RESTORE = "backup_restore"
    HOT_STANDBY = "hot_standby"
    COLD_STANDBY = "cold_standby"
    ACTIVE_ACTIVE = "active_active"
    SNAPSHOT_ROLLBACK = "snapshot_rollback"


class BackupType(str, Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    TRANSACTION_LOG = "transaction_log"


class RecoveryStatus(str, Enum):
    """Recovery operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailoverTrigger(str, Enum):
    """Failover trigger types"""
    MANUAL = "manual"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"


@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_type: BackupType
    schedule_cron: str  # Cron expression for scheduling
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    verify_integrity: bool = True
    storage_path: str = "./backups"
    max_parallel_backups: int = 3
    backup_timeout_minutes: int = 60
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    backup_type: BackupType
    created_at: datetime
    size_bytes: int
    checksum: str
    file_count: int
    agent_id: str
    version: str
    recovery_point_objective_minutes: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPoint:
    """Recovery point definition"""
    recovery_point_id: str
    timestamp: datetime
    backup_ids: List[str]
    agent_state: Dict[str, Any]
    database_state: Dict[str, Any]
    configuration_snapshot: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class FailoverPlan:
    """Failover execution plan"""
    plan_id: str
    name: str
    description: str
    trigger: FailoverTrigger
    primary_agent_id: str
    standby_agent_ids: List[str]
    failover_steps: List[Dict[str, Any]]
    rollback_steps: List[Dict[str, Any]]
    max_failover_time_minutes: int = 10
    auto_failback: bool = False
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class DisasterEvent:
    """Disaster event record"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: str
    affected_agents: List[str]
    description: str
    impact_assessment: Dict[str, Any]
    recovery_actions: List[str]
    resolution_time_minutes: Optional[int] = None
    lessons_learned: str = ""


class BackupStorageProvider(ABC):
    """Abstract backup storage provider"""

    @abstractmethod
    async def store_backup(self, backup_id: str, data: bytes, metadata: BackupMetadata) -> str:
        """Store backup data"""
        pass

    @abstractmethod
    async def retrieve_backup(self, backup_id: str) -> bytes:
        """Retrieve backup data"""
        pass

    @abstractmethod
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup data"""
        pass

    @abstractmethod
    async def list_backups(self, agent_id: str = None) -> List[BackupMetadata]:
        """List available backups"""
        pass


class LocalFileStorageProvider(BackupStorageProvider):
    """Local file system storage provider"""

    def __init__(self, storage_path: str = "./backups"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.storage_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)

    async def store_backup(self, backup_id: str, data: bytes, metadata: BackupMetadata) -> str:
        """Store backup to local filesystem"""
        backup_file = self.storage_path / f"{backup_id}.backup"
        metadata_file = self.metadata_path / f"{backup_id}.json"

        # Store backup data
        async with asyncio.Lock():
            with open(backup_file, 'wb') as f:
                f.write(data)

            # Store metadata
            with open(metadata_file, 'w') as f:
                json.dump({
                    "backup_id": metadata.backup_id,
                    "backup_type": metadata.backup_type.value,
                    "created_at": metadata.created_at.isoformat(),
                    "size_bytes": metadata.size_bytes,
                    "checksum": metadata.checksum,
                    "file_count": metadata.file_count,
                    "agent_id": metadata.agent_id,
                    "version": metadata.version,
                    "recovery_point_objective_minutes": metadata.recovery_point_objective_minutes,
                    "metadata": metadata.metadata
                }, f, indent=2)

        return str(backup_file)

    async def retrieve_backup(self, backup_id: str) -> bytes:
        """Retrieve backup from local filesystem"""
        backup_file = self.storage_path / f"{backup_id}.backup"

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} not found")

        with open(backup_file, 'rb') as f:
            return f.read()

    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup from local filesystem"""
        backup_file = self.storage_path / f"{backup_id}.backup"
        metadata_file = self.metadata_path / f"{backup_id}.json"

        success = True

        if backup_file.exists():
            backup_file.unlink()
        else:
            success = False

        if metadata_file.exists():
            metadata_file.unlink()

        return success

    async def list_backups(self, agent_id: str = None) -> List[BackupMetadata]:
        """List available backups"""
        backups = []

        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)

                # Filter by agent_id if specified
                if agent_id and data.get("agent_id") != agent_id:
                    continue

                backup_metadata = BackupMetadata(
                    backup_id=data["backup_id"],
                    backup_type=BackupType(data["backup_type"]),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    size_bytes=data["size_bytes"],
                    checksum=data["checksum"],
                    file_count=data["file_count"],
                    agent_id=data["agent_id"],
                    version=data["version"],
                    recovery_point_objective_minutes=data.get("recovery_point_objective_minutes", 60),
                    metadata=data.get("metadata", {})
                )

                backups.append(backup_metadata)

            except Exception as e:
                logger.error(f"Failed to load backup metadata {metadata_file}: {e}")

        return sorted(backups, key=lambda b: b.created_at, reverse=True)


class BackupManager:
    """Manages backup operations"""

    def __init__(self, storage_provider: BackupStorageProvider):
        self.storage_provider = storage_provider
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.active_backups: Dict[str, asyncio.Task] = {}
        self.backup_scheduler = None
        self.running = False

    def register_backup_config(self, agent_id: str, config: BackupConfig):
        """Register backup configuration for an agent"""
        self.backup_configs[agent_id] = config
        logger.info(f"Registered backup config for agent {agent_id}")

    async def start_scheduler(self):
        """Start backup scheduler"""
        if not self.running:
            self.running = True
            self.backup_scheduler = asyncio.create_task(self._backup_scheduler_loop())

    async def stop_scheduler(self):
        """Stop backup scheduler"""
        self.running = False

        if self.backup_scheduler:
            self.backup_scheduler.cancel()
            try:
                await self.backup_scheduler
            except asyncio.CancelledError:
                pass

        # Cancel active backups
        for backup_task in self.active_backups.values():
            backup_task.cancel()

        if self.active_backups:
            await asyncio.gather(*self.active_backups.values(), return_exceptions=True)

    @trace_async("create_backup")
    async def create_backup(
        self,
        agent_id: str,
        backup_type: BackupType = BackupType.FULL,
        custom_data: Dict[str, Any] = None
    ) -> str:
        """Create a backup for an agent"""

        backup_id = f"{agent_id}_{backup_type.value}_{int(time.time())}"

        add_span_attributes({
            "backup.id": backup_id,
            "backup.type": backup_type.value,
            "agent.id": agent_id
        })

        if agent_id not in self.backup_configs:
            raise ValueError(f"No backup configuration found for agent {agent_id}")

        config = self.backup_configs[agent_id]

        # Prevent too many parallel backups
        if len(self.active_backups) >= config.max_parallel_backups:
            raise Exception(f"Maximum parallel backups ({config.max_parallel_backups}) exceeded")

        # Start backup task
        backup_task = asyncio.create_task(
            self._execute_backup(backup_id, agent_id, backup_type, config, custom_data)
        )

        self.active_backups[backup_id] = backup_task

        try:
            result = await asyncio.wait_for(backup_task, timeout=config.backup_timeout_minutes * 60)
            return result

        except asyncio.TimeoutError:
            backup_task.cancel()
            raise Exception(f"Backup {backup_id} timed out after {config.backup_timeout_minutes} minutes")

        finally:
            if backup_id in self.active_backups:
                del self.active_backups[backup_id]

    async def _execute_backup(
        self,
        backup_id: str,
        agent_id: str,
        backup_type: BackupType,
        config: BackupConfig,
        custom_data: Dict[str, Any] = None
    ) -> str:
        """Execute backup operation"""

        start_time = time.time()
        logger.info(f"Starting backup {backup_id} for agent {agent_id}")

        try:
            # Collect backup data
            backup_data = await self._collect_backup_data(
                agent_id, backup_type, config, custom_data
            )

            # Create archive
            archive_data = await self._create_backup_archive(backup_data, config)

            # Calculate checksum
            checksum = hashlib.sha256(archive_data).hexdigest()

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.utcnow(),
                size_bytes=len(archive_data),
                checksum=checksum,
                file_count=len(backup_data.get("files", [])),
                agent_id=agent_id,
                version="1.0.0",
                recovery_point_objective_minutes=config.retention_days * 24 * 60,
                metadata={
                    "backup_duration_seconds": time.time() - start_time,
                    "compression_enabled": config.compression,
                    "encryption_enabled": config.encryption
                }
            )

            # Store backup
            storage_location = await self.storage_provider.store_backup(
                backup_id, archive_data, metadata
            )

            # Verify integrity if requested
            if config.verify_integrity:
                await self._verify_backup_integrity(backup_id, checksum)

            duration = time.time() - start_time
            logger.info(f"Backup {backup_id} completed successfully in {duration:.2f}s")

            return backup_id

        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {str(e)}")
            raise e

    async def _collect_backup_data(
        self,
        agent_id: str,
        backup_type: BackupType,
        config: BackupConfig,
        custom_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Collect data for backup"""

        backup_data = {
            "agent_id": agent_id,
            "backup_type": backup_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "files": [],
            "database": {},
            "configuration": {},
            "state": {},
            "custom": custom_data or {}
        }

        # Collect file data
        if config.include_patterns:
            for pattern in config.include_patterns:
                files = await self._collect_files_by_pattern(pattern, config.exclude_patterns)
                backup_data["files"].extend(files)

        # Collect database state (if applicable)
        backup_data["database"] = await self._collect_database_state(agent_id)

        # Collect configuration
        backup_data["configuration"] = await self._collect_configuration(agent_id)

        # Collect runtime state
        backup_data["state"] = await self._collect_runtime_state(agent_id)

        return backup_data

    async def _collect_files_by_pattern(
        self,
        pattern: str,
        exclude_patterns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Collect files matching pattern"""
        files = []

        try:
            from pathlib import Path
            import glob

            for file_path in glob.glob(pattern, recursive=True):
                path_obj = Path(file_path)

                if not path_obj.is_file():
                    continue

                # Check exclude patterns
                if exclude_patterns:
                    excluded = False
                    for exclude_pattern in exclude_patterns:
                        if path_obj.match(exclude_pattern):
                            excluded = True
                            break
                    if excluded:
                        continue

                # Read file content
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()

                    files.append({
                        "path": str(path_obj),
                        "size": len(content),
                        "content": content.hex(),  # Store as hex string
                        "modified_at": datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat()
                    })

                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Failed to collect files for pattern {pattern}: {e}")

        return files

    async def _collect_database_state(self, agent_id: str) -> Dict[str, Any]:
        """Collect database state for backup"""
        # This would integrate with actual database systems
        return {
            "tables": [],
            "indexes": [],
            "constraints": [],
            "data_snapshot": {}
        }

    async def _collect_configuration(self, agent_id: str) -> Dict[str, Any]:
        """Collect configuration for backup"""
        # This would collect actual agent configuration
        return {
            "agent_config": {},
            "environment_variables": {},
            "feature_flags": {}
        }

    async def _collect_runtime_state(self, agent_id: str) -> Dict[str, Any]:
        """Collect runtime state for backup"""
        # This would collect actual agent runtime state
        return {
            "active_tasks": [],
            "connections": [],
            "cache_state": {},
            "metrics": {}
        }

    async def _create_backup_archive(
        self,
        backup_data: Dict[str, Any],
        config: BackupConfig
    ) -> bytes:
        """Create compressed backup archive"""

        # Serialize data to JSON
        json_data = json.dumps(backup_data, default=str).encode('utf-8')

        if config.compression:
            # Compress with gzip
            json_data = gzip.compress(json_data)

        if config.encryption:
            # In production, use proper encryption
            # For demo, we'll just add a simple XOR
            key = b"backup_encryption_key_placeholder"
            encrypted_data = bytes(a ^ b for a, b in zip(json_data, key * (len(json_data) // len(key) + 1)))
            json_data = encrypted_data

        return json_data

    async def _verify_backup_integrity(self, backup_id: str, expected_checksum: str):
        """Verify backup integrity"""
        try:
            # Retrieve backup and verify checksum
            backup_data = await self.storage_provider.retrieve_backup(backup_id)
            actual_checksum = hashlib.sha256(backup_data).hexdigest()

            if actual_checksum != expected_checksum:
                raise Exception(f"Backup integrity check failed: {actual_checksum} != {expected_checksum}")

            logger.debug(f"Backup {backup_id} integrity verified")

        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            raise e

    async def _backup_scheduler_loop(self):
        """Background backup scheduler loop"""
        while self.running:
            try:
                # Check if any backups are due
                for agent_id, config in self.backup_configs.items():
                    if await self._is_backup_due(agent_id, config):
                        # Schedule backup
                        asyncio.create_task(self._schedule_backup(agent_id, config))

                # Check every minute
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(60)

    async def _is_backup_due(self, agent_id: str, config: BackupConfig) -> bool:
        """Check if backup is due based on schedule"""
        # Simplified cron evaluation - in production use proper cron library
        # For now, we'll just check if we haven't backed up in the last hour

        backups = await self.storage_provider.list_backups(agent_id)
        if not backups:
            return True

        latest_backup = max(backups, key=lambda b: b.created_at)
        time_since_backup = datetime.utcnow() - latest_backup.created_at

        return time_since_backup > timedelta(hours=1)

    async def _schedule_backup(self, agent_id: str, config: BackupConfig):
        """Schedule a backup for an agent"""
        try:
            backup_id = await self.create_backup(agent_id, BackupType.INCREMENTAL)
            logger.info(f"Scheduled backup completed: {backup_id}")
        except Exception as e:
            logger.error(f"Scheduled backup failed for agent {agent_id}: {e}")


class RecoveryManager:
    """Manages disaster recovery operations"""

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.recovery_plans: Dict[str, FailoverPlan] = {}
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self.disaster_events: List[DisasterEvent] = []
        self.active_recoveries: Dict[str, asyncio.Task] = {}

    def register_failover_plan(self, plan: FailoverPlan):
        """Register a failover plan"""
        self.recovery_plans[plan.plan_id] = plan
        logger.info(f"Registered failover plan: {plan.name}")

    def create_recovery_point(
        self,
        agent_id: str,
        backup_ids: List[str],
        agent_state: Dict[str, Any] = None,
        database_state: Dict[str, Any] = None,
        config_snapshot: Dict[str, Any] = None
    ) -> str:
        """Create a recovery point"""

        recovery_point_id = f"rp_{agent_id}_{int(time.time())}"

        recovery_point = RecoveryPoint(
            recovery_point_id=recovery_point_id,
            timestamp=datetime.utcnow(),
            backup_ids=backup_ids,
            agent_state=agent_state or {},
            database_state=database_state or {},
            configuration_snapshot=config_snapshot or {}
        )

        self.recovery_points[recovery_point_id] = recovery_point
        logger.info(f"Created recovery point: {recovery_point_id}")

        return recovery_point_id

    @trace_async("disaster_recovery")
    async def execute_disaster_recovery(
        self,
        recovery_point_id: str,
        target_agent_id: Optional[str] = None,
        recovery_options: Dict[str, Any] = None
    ) -> str:
        """Execute disaster recovery"""

        if recovery_point_id not in self.recovery_points:
            raise ValueError(f"Recovery point {recovery_point_id} not found")

        recovery_point = self.recovery_points[recovery_point_id]
        recovery_id = f"recovery_{int(time.time())}"

        add_span_attributes({
            "recovery.id": recovery_id,
            "recovery_point.id": recovery_point_id,
            "target_agent.id": target_agent_id or "unknown"
        })

        logger.info(f"Starting disaster recovery {recovery_id} from point {recovery_point_id}")

        try:
            # Create disaster event record
            disaster_event = DisasterEvent(
                event_id=recovery_id,
                timestamp=datetime.utcnow(),
                event_type="disaster_recovery_started",
                severity="high",
                affected_agents=[target_agent_id] if target_agent_id else [],
                description=f"Disaster recovery initiated from recovery point {recovery_point_id}",
                impact_assessment={"recovery_time_estimate_minutes": 30},
                recovery_actions=[]
            )

            self.disaster_events.append(disaster_event)

            # Execute recovery steps
            recovery_steps = [
                "validate_recovery_point",
                "restore_backups",
                "restore_database_state",
                "restore_configuration",
                "restore_agent_state",
                "verify_recovery",
                "resume_operations"
            ]

            for step in recovery_steps:
                logger.info(f"Executing recovery step: {step}")
                await self._execute_recovery_step(step, recovery_point, target_agent_id, recovery_options)
                disaster_event.recovery_actions.append(f"Completed: {step}")

            # Mark recovery as completed
            disaster_event.resolution_time_minutes = (datetime.utcnow() - disaster_event.timestamp).total_seconds() / 60

            logger.info(f"Disaster recovery {recovery_id} completed successfully")
            return recovery_id

        except Exception as e:
            logger.error(f"Disaster recovery {recovery_id} failed: {str(e)}")

            # Update disaster event
            disaster_event.resolution_time_minutes = (datetime.utcnow() - disaster_event.timestamp).total_seconds() / 60
            disaster_event.recovery_actions.append(f"Failed at step: {str(e)}")

            raise e

    async def _execute_recovery_step(
        self,
        step: str,
        recovery_point: RecoveryPoint,
        target_agent_id: Optional[str],
        options: Dict[str, Any] = None
    ):
        """Execute a single recovery step"""

        if step == "validate_recovery_point":
            await self._validate_recovery_point(recovery_point)

        elif step == "restore_backups":
            await self._restore_backups(recovery_point.backup_ids, target_agent_id)

        elif step == "restore_database_state":
            await self._restore_database_state(recovery_point.database_state, target_agent_id)

        elif step == "restore_configuration":
            await self._restore_configuration(recovery_point.configuration_snapshot, target_agent_id)

        elif step == "restore_agent_state":
            await self._restore_agent_state(recovery_point.agent_state, target_agent_id)

        elif step == "verify_recovery":
            await self._verify_recovery(target_agent_id)

        elif step == "resume_operations":
            await self._resume_operations(target_agent_id)

        else:
            raise ValueError(f"Unknown recovery step: {step}")

    async def _validate_recovery_point(self, recovery_point: RecoveryPoint):
        """Validate recovery point integrity"""
        logger.info(f"Validating recovery point {recovery_point.recovery_point_id}")

        # Check that all referenced backups exist
        for backup_id in recovery_point.backup_ids:
            try:
                backups = await self.backup_manager.storage_provider.list_backups()
                backup_exists = any(b.backup_id == backup_id for b in backups)

                if not backup_exists:
                    raise Exception(f"Backup {backup_id} not found")

            except Exception as e:
                raise Exception(f"Failed to validate backup {backup_id}: {str(e)}")

    async def _restore_backups(self, backup_ids: List[str], target_agent_id: Optional[str]):
        """Restore data from backups"""
        logger.info(f"Restoring {len(backup_ids)} backups")

        for backup_id in backup_ids:
            try:
                # Retrieve backup data
                backup_data = await self.backup_manager.storage_provider.retrieve_backup(backup_id)

                # Decompress and decrypt if needed
                restored_data = await self._process_backup_data(backup_data)

                # Restore files and data
                await self._restore_backup_contents(restored_data, target_agent_id)

                logger.info(f"Successfully restored backup {backup_id}")

            except Exception as e:
                raise Exception(f"Failed to restore backup {backup_id}: {str(e)}")

    async def _process_backup_data(self, backup_data: bytes) -> Dict[str, Any]:
        """Process backup data (decrypt, decompress)"""

        # Reverse the encryption (in production, use proper decryption)
        key = b"backup_encryption_key_placeholder"
        decrypted_data = bytes(a ^ b for a, b in zip(backup_data, key * (len(backup_data) // len(key) + 1)))

        # Decompress
        try:
            decompressed_data = gzip.decompress(decrypted_data)
        except:
            decompressed_data = decrypted_data

        # Parse JSON
        return json.loads(decompressed_data.decode('utf-8'))

    async def _restore_backup_contents(self, backup_data: Dict[str, Any], target_agent_id: Optional[str]):
        """Restore contents from backup data"""

        # Restore files
        for file_info in backup_data.get("files", []):
            try:
                file_path = Path(file_info["path"])
                file_content = bytes.fromhex(file_info["content"])

                # Create directory if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                with open(file_path, 'wb') as f:
                    f.write(file_content)

                logger.debug(f"Restored file: {file_path}")

            except Exception as e:
                logger.warning(f"Failed to restore file {file_info['path']}: {e}")

    async def _restore_database_state(self, database_state: Dict[str, Any], target_agent_id: Optional[str]):
        """Restore database state"""
        logger.info("Restoring database state")
        # Implementation would restore actual database state

    async def _restore_configuration(self, config_snapshot: Dict[str, Any], target_agent_id: Optional[str]):
        """Restore configuration"""
        logger.info("Restoring configuration")
        # Implementation would restore actual configuration

    async def _restore_agent_state(self, agent_state: Dict[str, Any], target_agent_id: Optional[str]):
        """Restore agent runtime state"""
        logger.info("Restoring agent state")
        # Implementation would restore actual agent state

    async def _verify_recovery(self, target_agent_id: Optional[str]):
        """Verify recovery was successful"""
        logger.info("Verifying recovery")

        # Run health checks
        health_checks = [
            self._check_agent_health,
            self._check_database_connectivity,
            self._check_configuration_validity,
            self._check_critical_services
        ]

        for check in health_checks:
            try:
                await check(target_agent_id)
            except Exception as e:
                raise Exception(f"Health check failed: {str(e)}")

    async def _resume_operations(self, target_agent_id: Optional[str]):
        """Resume normal operations"""
        logger.info("Resuming normal operations")
        # Implementation would restart services, clear maintenance mode, etc.

    async def _check_agent_health(self, agent_id: Optional[str]):
        """Check agent health"""
        # Implementation would check actual agent health
        pass

    async def _check_database_connectivity(self, agent_id: Optional[str]):
        """Check database connectivity"""
        # Implementation would check actual database connectivity
        pass

    async def _check_configuration_validity(self, agent_id: Optional[str]):
        """Check configuration validity"""
        # Implementation would validate configuration
        pass

    async def _check_critical_services(self, agent_id: Optional[str]):
        """Check critical services"""
        # Implementation would check critical service status
        pass

    def get_disaster_event_history(self, hours: int = 24) -> List[DisasterEvent]:
        """Get recent disaster events"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [event for event in self.disaster_events if event.timestamp > cutoff]

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        recent_events = self.get_disaster_event_history(hours=24)

        return {
            "total_recovery_points": len(self.recovery_points),
            "recent_disasters": len(recent_events),
            "average_recovery_time_minutes": statistics.mean([
                e.resolution_time_minutes for e in recent_events
                if e.resolution_time_minutes is not None
            ]) if recent_events else 0,
            "recovery_success_rate": len([e for e in recent_events if e.resolution_time_minutes is not None]) / len(recent_events) if recent_events else 1.0
        }


class DisasterRecoverySystem:
    """Main disaster recovery system"""

    def __init__(self, storage_provider: BackupStorageProvider = None):
        self.storage_provider = storage_provider or LocalFileStorageProvider()
        self.backup_manager = BackupManager(self.storage_provider)
        self.recovery_manager = RecoveryManager(self.backup_manager)

        self.system_health_checks: List[Callable] = []
        self.health_check_task = None
        self.running = False

    async def initialize(self):
        """Initialize disaster recovery system"""
        await self.backup_manager.start_scheduler()

        # Start health monitoring
        self.running = True
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())

        logger.info("Disaster recovery system initialized")

    async def shutdown(self):
        """Shutdown disaster recovery system"""
        self.running = False

        await self.backup_manager.stop_scheduler()

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Disaster recovery system shut down")

    def register_agent_for_backup(self, agent: A2AAgentBase, config: BackupConfig):
        """Register an agent for backup"""
        self.backup_manager.register_backup_config(agent.agent_id, config)

        # Create initial recovery point
        recovery_point_id = self.recovery_manager.create_recovery_point(
            agent_id=agent.agent_id,
            backup_ids=[],
            agent_state={"status": "registered"},
            database_state={},
            config_snapshot={}
        )

        logger.info(f"Registered agent {agent.agent_id} for disaster recovery")
        return recovery_point_id

    def register_health_check(self, health_check: Callable):
        """Register a system health check"""
        self.system_health_checks.append(health_check)
        logger.info("Registered system health check")

    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                # Run health checks
                for health_check in self.system_health_checks:
                    try:
                        result = await health_check() if asyncio.iscoroutinefunction(health_check) else health_check()

                        # If health check fails, consider triggering recovery
                        if not result:
                            logger.warning("System health check failed")
                            # Implement automatic failover logic here

                    except Exception as e:
                        logger.error(f"Health check failed: {e}")

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall disaster recovery system status"""
        return {
            "backup_manager": {
                "active_backups": len(self.backup_manager.active_backups),
                "configured_agents": len(self.backup_manager.backup_configs)
            },
            "recovery_manager": {
                "recovery_points": len(self.recovery_manager.recovery_points),
                "failover_plans": len(self.recovery_manager.recovery_plans),
                "recent_disasters": len(self.recovery_manager.get_disaster_event_history())
            },
            "health_monitoring": {
                "registered_checks": len(self.system_health_checks),
                "monitoring_active": self.running
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Global disaster recovery system
_disaster_recovery_system = None


async def initialize_disaster_recovery(
    storage_provider: BackupStorageProvider = None
) -> DisasterRecoverySystem:
    """Initialize global disaster recovery system"""
    global _disaster_recovery_system

    if _disaster_recovery_system is None:
        _disaster_recovery_system = DisasterRecoverySystem(storage_provider)
        await _disaster_recovery_system.initialize()

    return _disaster_recovery_system


async def get_disaster_recovery_system() -> Optional[DisasterRecoverySystem]:
    """Get the global disaster recovery system"""
    return _disaster_recovery_system


async def shutdown_disaster_recovery():
    """Shutdown global disaster recovery system"""
    global _disaster_recovery_system

    if _disaster_recovery_system:
        await _disaster_recovery_system.shutdown()
        _disaster_recovery_system = None


# Utility functions
async def create_agent_backup(agent: A2AAgentBase) -> str:
    """Create immediate backup for an agent"""
    dr_system = await get_disaster_recovery_system()

    if not dr_system:
        raise Exception("Disaster recovery system not initialized")

    return await dr_system.backup_manager.create_backup(agent.agent_id)


async def restore_agent_from_backup(agent_id: str, backup_id: str) -> str:
    """Restore agent from specific backup"""
    dr_system = await get_disaster_recovery_system()

    if not dr_system:
        raise Exception("Disaster recovery system not initialized")

    # Create recovery point from backup
    recovery_point_id = dr_system.recovery_manager.create_recovery_point(
        agent_id=agent_id,
        backup_ids=[backup_id],
        agent_state={},
        database_state={},
        config_snapshot={}
    )

    # Execute recovery
    return await dr_system.recovery_manager.execute_disaster_recovery(
        recovery_point_id, agent_id
    )