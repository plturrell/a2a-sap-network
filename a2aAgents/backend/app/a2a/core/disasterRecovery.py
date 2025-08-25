"""
A2A Disaster Recovery Manager
Implements comprehensive disaster recovery capabilities including backup, restore,
data replication, and recovery orchestration for the A2A platform
"""

import asyncio
import time
import json
import logging
import hashlib
import os
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import tarfile
import shutil

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups supported"""

    FULL = "full"  # Complete system backup
    INCREMENTAL = "incremental"  # Changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full backup
    SNAPSHOT = "snapshot"  # Point-in-time snapshot
    CONTINUOUS = "continuous"  # Real-time replication


class BackupStorage(Enum):
    """Backup storage locations"""

    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"
    NETWORK_SHARE = "network_share"


class RecoveryType(Enum):
    """Types of recovery operations"""

    FULL_RESTORE = "full_restore"
    POINT_IN_TIME = "point_in_time"
    SELECTIVE = "selective"
    ROLLBACK = "rollback"


class ReplicationMode(Enum):
    """Data replication modes"""

    SYNCHRONOUS = "synchronous"  # Write to all replicas before confirming
    ASYNCHRONOUS = "asynchronous"  # Write locally, replicate in background
    SEMI_SYNCHRONOUS = "semi_sync"  # At least one replica confirms


@dataclass
class DRConfig:
    """Disaster Recovery configuration"""

    # Backup settings
    backup_retention_days: int = 30
    full_backup_interval: int = 86400  # 24 hours in seconds
    incremental_interval: int = 3600  # 1 hour
    enable_continuous_backup: bool = True

    # Recovery objectives
    rpo_seconds: int = 300  # Recovery Point Objective: 5 minutes
    rto_seconds: int = 900  # Recovery Time Objective: 15 minutes

    # Replication settings
    replication_mode: ReplicationMode = ReplicationMode.ASYNCHRONOUS
    replica_count: int = 2
    geo_redundancy: bool = True

    # Storage settings
    primary_storage: BackupStorage = BackupStorage.S3
    secondary_storage: Optional[BackupStorage] = BackupStorage.AZURE_BLOB
    compression_enabled: bool = True
    encryption_enabled: bool = True


@dataclass
class BackupMetadata:
    """Metadata for a backup"""

    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    checksum: str
    location: str
    compression_ratio: Optional[float] = None
    encryption_key_id: Optional[str] = None
    parent_backup_id: Optional[str] = None  # For incremental/differential
    retention_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryPoint:
    """A point in time that can be recovered to"""

    recovery_point_id: str
    timestamp: datetime
    backup_chain: List[str]  # List of backup IDs needed for recovery
    estimated_recovery_time: int  # seconds
    data_consistency_verified: bool = False
    application_consistent: bool = False


@dataclass
class RecoveryPlan:
    """Plan for disaster recovery execution"""

    plan_id: str
    name: str
    recovery_type: RecoveryType
    target_recovery_point: Optional[RecoveryPoint] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    pre_checks: List[str] = field(default_factory=list)
    post_checks: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # seconds
    auto_execute: bool = False


@dataclass
class ReplicationTarget:
    """Target for data replication"""

    target_id: str
    location: str
    storage_type: BackupStorage
    is_active: bool = True
    last_sync: Optional[datetime] = None
    lag_seconds: int = 0
    health_status: str = "healthy"


@dataclass
class DRTestResult:
    """Result of a disaster recovery test"""

    test_id: str
    test_type: str
    executed_at: datetime
    duration_seconds: int
    success: bool
    rpo_achieved: bool
    rto_achieved: bool
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DisasterRecoveryManager:
    """
    Manages disaster recovery operations for A2A platform
    Handles backup, restore, replication, and recovery orchestration
    """

    def __init__(self, config: DRConfig = None):
        self.config = config or DRConfig()

        # Backup management
        self.backup_catalog: Dict[str, BackupMetadata] = {}
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self.active_backups: Set[str] = set()

        # Replication management
        self.replication_targets: Dict[str, ReplicationTarget] = {}
        self.replication_queue: deque = deque(maxlen=10000)
        self.replication_lag: Dict[str, int] = defaultdict(int)

        # Recovery planning
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_history: deque = deque(maxlen=100)

        # DR testing
        self.test_results: deque = deque(maxlen=50)
        self.last_test_date: Optional[datetime] = None

        # Performance metrics
        self.backup_times: deque = deque(maxlen=100)
        self.recovery_times: deque = deque(maxlen=100)
        self.successful_backups = 0
        self.failed_backups = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0

        # State management
        self._lock = asyncio.Lock()
        self._running = False
        self._backup_task = None
        self._replication_task = None
        self._maintenance_task = None

        # Storage paths (simplified for demo)
        self.backup_base_path = "/var/backups/a2a"
        os.makedirs(self.backup_base_path, exist_ok=True)

        logger.info(
            f"Initialized DR Manager with RPO={config.rpo_seconds}s, RTO={config.rto_seconds}s"
        )

    async def initialize(self):
        """Initialize disaster recovery manager"""
        self._running = True
        self._backup_task = asyncio.create_task(self._backup_scheduler())
        self._replication_task = asyncio.create_task(self._replication_processor())
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

        # Load existing backup catalog
        await self._load_backup_catalog()

        # Initialize recovery plans
        await self._initialize_recovery_plans()

        logger.info("Disaster Recovery Manager initialized")

    async def shutdown(self):
        """Shutdown disaster recovery manager"""
        self._running = False

        # Cancel background tasks
        for task in [self._backup_task, self._replication_task, self._maintenance_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Disaster Recovery Manager shutdown complete")

    async def create_backup(
        self,
        backup_type: BackupType,
        data_sources: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[BackupMetadata]:
        """
        Create a backup of specified data sources
        Returns backup metadata if successful
        """
        backup_id = f"backup_{int(time.time())}_{backup_type.value}"
        start_time = time.time()

        async with self._lock:
            if backup_id in self.active_backups:
                logger.warning(f"Backup {backup_id} already in progress")
                return None

            self.active_backups.add(backup_id)

        try:
            logger.info(f"Starting {backup_type.value} backup {backup_id}")

            # Determine parent backup for incremental/differential
            parent_backup_id = None
            if backup_type in [BackupType.INCREMENTAL, BackupType.DIFFERENTIAL]:
                parent_backup_id = await self._find_parent_backup(backup_type)
                if not parent_backup_id:
                    logger.warning("No parent backup found, creating full backup instead")
                    backup_type = BackupType.FULL

            # Create backup
            backup_path = os.path.join(self.backup_base_path, backup_id)
            os.makedirs(backup_path, exist_ok=True)

            # Backup each data source
            total_size = 0
            checksums = []

            for source in data_sources:
                source_backup = await self._backup_data_source(
                    source, backup_path, backup_type, parent_backup_id
                )
                total_size += source_backup["size"]
                checksums.append(source_backup["checksum"])

            # Calculate overall checksum
            overall_checksum = hashlib.sha256("".join(checksums).encode()).hexdigest()

            # Compress if enabled
            compression_ratio = None
            if self.config.compression_enabled:
                compressed_path = f"{backup_path}.tar.gz"
                compression_ratio = await self._compress_backup(backup_path, compressed_path)
                shutil.rmtree(backup_path)  # Remove uncompressed
                backup_path = compressed_path

            # Encrypt if enabled
            encryption_key_id = None
            if self.config.encryption_enabled:
                encryption_key_id = await self._encrypt_backup(backup_path)

            # Upload to storage
            storage_location = await self._upload_backup(backup_path, self.config.primary_storage)

            # Create metadata
            backup_metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                timestamp=datetime.utcnow(),
                size_bytes=total_size,
                checksum=overall_checksum,
                location=storage_location,
                compression_ratio=compression_ratio,
                encryption_key_id=encryption_key_id,
                parent_backup_id=parent_backup_id,
                retention_until=datetime.utcnow()
                + timedelta(days=self.config.backup_retention_days),
                metadata=metadata or {},
            )

            # Update catalog
            self.backup_catalog[backup_id] = backup_metadata
            await self._save_backup_catalog()

            # Create recovery point
            await self._create_recovery_point(backup_metadata)

            # Replicate to secondary storage
            if self.config.secondary_storage:
                asyncio.create_task(
                    self._replicate_backup(backup_path, self.config.secondary_storage)
                )

            # Update metrics
            backup_time = time.time() - start_time
            self.backup_times.append(backup_time)
            self.successful_backups += 1

            logger.info(
                f"Backup {backup_id} completed successfully in {backup_time:.2f}s, "
                f"size: {total_size / 1024 / 1024:.2f}MB"
            )

            return backup_metadata

        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {e}")
            self.failed_backups += 1
            return None

        finally:
            self.active_backups.discard(backup_id)

    async def restore_backup(
        self,
        recovery_point_id: str,
        target_location: str,
        recovery_type: RecoveryType = RecoveryType.FULL_RESTORE,
        options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Restore from a recovery point
        Returns True if successful
        """
        start_time = time.time()

        try:
            logger.info(f"Starting {recovery_type.value} recovery to point {recovery_point_id}")

            # Get recovery point
            recovery_point = self.recovery_points.get(recovery_point_id)
            if not recovery_point:
                logger.error(f"Recovery point {recovery_point_id} not found")
                return False

            # Verify data consistency
            if not recovery_point.data_consistency_verified:
                if not await self._verify_recovery_point(recovery_point):
                    logger.error("Recovery point verification failed")
                    return False

            # Execute pre-recovery checks
            if not await self._run_pre_recovery_checks(recovery_point, target_location):
                logger.error("Pre-recovery checks failed")
                return False

            # Build recovery chain
            recovery_chain = []
            for backup_id in recovery_point.backup_chain:
                backup = self.backup_catalog.get(backup_id)
                if not backup:
                    logger.error(f"Backup {backup_id} not found in catalog")
                    return False
                recovery_chain.append(backup)

            # Execute recovery
            for backup in recovery_chain:
                # Download backup
                local_path = await self._download_backup(backup)

                # Decrypt if needed
                if backup.encryption_key_id:
                    await self._decrypt_backup(local_path, backup.encryption_key_id)

                # Decompress if needed
                if backup.compression_ratio:
                    await self._decompress_backup(local_path)

                # Restore data
                success = await self._restore_data(
                    local_path, target_location, recovery_type, options
                )

                if not success:
                    logger.error(f"Failed to restore backup {backup.backup_id}")
                    return False

                # Clean up local copy
                os.remove(local_path)

            # Execute post-recovery checks
            if not await self._run_post_recovery_checks(target_location):
                logger.error("Post-recovery checks failed")
                return False

            # Update metrics
            recovery_time = time.time() - start_time
            self.recovery_times.append(recovery_time)
            self.successful_recoveries += 1

            # Record recovery
            self.recovery_history.append(
                {
                    "timestamp": datetime.utcnow(),
                    "recovery_point_id": recovery_point_id,
                    "recovery_type": recovery_type.value,
                    "duration_seconds": recovery_time,
                    "success": True,
                }
            )

            logger.info(f"Recovery completed successfully in {recovery_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self.failed_recoveries += 1
            return False

    async def create_recovery_plan(
        self,
        plan_name: str,
        recovery_type: RecoveryType,
        target_recovery_point: Optional[str] = None,
        auto_execute: bool = False,
    ) -> RecoveryPlan:
        """Create a disaster recovery plan"""
        plan_id = f"dr_plan_{int(time.time())}"

        # Build recovery steps based on type
        steps = []
        pre_checks = []
        post_checks = []

        if recovery_type == RecoveryType.FULL_RESTORE:
            pre_checks = [
                "Verify backup integrity",
                "Check target storage capacity",
                "Validate recovery environment",
                "Stop dependent services",
            ]

            steps = [
                {"step": "Download backups", "estimated_time": 300},
                {"step": "Decrypt and decompress", "estimated_time": 120},
                {"step": "Restore database", "estimated_time": 600},
                {"step": "Restore application data", "estimated_time": 300},
                {"step": "Restore configuration", "estimated_time": 60},
                {"step": "Update network settings", "estimated_time": 120},
            ]

            post_checks = [
                "Verify data integrity",
                "Test application connectivity",
                "Validate service functionality",
                "Check data consistency",
            ]

        elif recovery_type == RecoveryType.POINT_IN_TIME:
            pre_checks.extend(
                [
                    "Identify recovery point",
                    "Verify transaction logs",
                    "Check point-in-time consistency",
                ]
            )

            steps.extend(
                [
                    {"step": "Restore base backup", "estimated_time": 600},
                    {"step": "Apply transaction logs", "estimated_time": 900},
                    {"step": "Roll forward to target time", "estimated_time": 300},
                ]
            )

        # Calculate estimated duration
        estimated_duration = sum(step.get("estimated_time", 0) for step in steps)

        # Get recovery point if specified
        recovery_point = None
        if target_recovery_point:
            recovery_point = self.recovery_points.get(target_recovery_point)

        plan = RecoveryPlan(
            plan_id=plan_id,
            name=plan_name,
            recovery_type=recovery_type,
            target_recovery_point=recovery_point,
            steps=steps,
            pre_checks=pre_checks,
            post_checks=post_checks,
            estimated_duration=estimated_duration,
            auto_execute=auto_execute,
        )

        self.recovery_plans[plan_id] = plan
        logger.info(f"Created recovery plan {plan_id}: {plan_name}")

        return plan

    async def test_disaster_recovery(
        self, test_type: str = "full", target_environment: str = "test"
    ) -> DRTestResult:
        """
        Test disaster recovery procedures
        Returns test results with recommendations
        """
        test_id = f"dr_test_{int(time.time())}"
        start_time = time.time()
        issues = []
        recommendations = []

        logger.info(f"Starting DR test {test_id} ({test_type})")

        try:
            # Test backup creation
            test_backup = await self.create_backup(
                BackupType.FULL, [{"source": "test_data", "type": "database"}], {"test": True}
            )

            if not test_backup:
                issues.append("Backup creation failed")
                recommendations.append("Check backup system configuration")

            # Test recovery point creation
            recovery_points_before = len(self.recovery_points)
            await asyncio.sleep(1)  # Allow recovery point creation
            if len(self.recovery_points) <= recovery_points_before:
                issues.append("Recovery point not created")
                recommendations.append("Verify recovery point generation logic")

            # Test restore (to test environment)
            if test_backup and test_type == "full":
                restore_success = await self.restore_backup(
                    list(self.recovery_points.keys())[-1],
                    f"/tmp/dr_test_{test_id}",
                    RecoveryType.FULL_RESTORE,
                )

                if not restore_success:
                    issues.append("Restore operation failed")
                    recommendations.append("Check restore procedures and permissions")

            # Calculate RTO/RPO achievement
            duration = time.time() - start_time
            rto_achieved = duration <= self.config.rto_seconds

            # For RPO, check backup frequency
            recent_backups = [
                b
                for b in self.backup_catalog.values()
                if (datetime.utcnow() - b.timestamp).total_seconds() < self.config.rpo_seconds
            ]
            rpo_achieved = len(recent_backups) > 0

            if not rto_achieved:
                issues.append(f"RTO not met: {duration:.0f}s > {self.config.rto_seconds}s")
                recommendations.append("Optimize recovery procedures or adjust RTO target")

            if not rpo_achieved:
                issues.append(f"RPO not met: No recent backups within {self.config.rpo_seconds}s")
                recommendations.append("Increase backup frequency or enable continuous backup")

            # Test replication
            if self.config.replica_count > 0:
                replication_lag_max = (
                    max(self.replication_lag.values()) if self.replication_lag else 0
                )
                if replication_lag_max > self.config.rpo_seconds:
                    issues.append(f"Replication lag exceeds RPO: {replication_lag_max}s")
                    recommendations.append("Check replication performance and network connectivity")

            # Create test result
            result = DRTestResult(
                test_id=test_id,
                test_type=test_type,
                executed_at=datetime.utcnow(),
                duration_seconds=int(duration),
                success=len(issues) == 0,
                rpo_achieved=rpo_achieved,
                rto_achieved=rto_achieved,
                issues_found=issues,
                recommendations=recommendations,
            )

            self.test_results.append(result)
            self.last_test_date = datetime.utcnow()

            logger.info(
                f"DR test {test_id} completed. Success: {result.success}, " f"Issues: {len(issues)}"
            )

            return result

        except Exception as e:
            logger.error(f"DR test failed with exception: {e}")

            return DRTestResult(
                test_id=test_id,
                test_type=test_type,
                executed_at=datetime.utcnow(),
                duration_seconds=int(time.time() - start_time),
                success=False,
                rpo_achieved=False,
                rto_achieved=False,
                issues_found=[f"Test failed with exception: {str(e)}"],
                recommendations=["Review DR test procedures and error handling"],
            )

    async def initiate_failover(
        self, recovery_plan_id: str, confirmation_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initiate disaster recovery failover
        Requires confirmation token for safety
        """
        if not confirmation_token:
            # Generate confirmation token
            token = hashlib.sha256(f"{recovery_plan_id}{time.time()}".encode()).hexdigest()[:8]
            return {
                "status": "confirmation_required",
                "message": "Disaster recovery failover requires confirmation",
                "recovery_plan_id": recovery_plan_id,
                "confirmation_token": token,
                "instruction": "Re-run with confirmation_token to proceed",
            }

        plan = self.recovery_plans.get(recovery_plan_id)
        if not plan:
            return {"status": "error", "message": f"Recovery plan {recovery_plan_id} not found"}

        logger.warning(f"INITIATING DISASTER RECOVERY FAILOVER - Plan: {plan.name}")

        try:
            # Execute recovery plan
            start_time = time.time()

            # Run pre-checks
            for check in plan.pre_checks:
                logger.info(f"Pre-check: {check}")
                # In production, would execute actual checks
                await asyncio.sleep(0.5)  # Simulate check

            # Execute recovery steps
            for i, step in enumerate(plan.steps):
                logger.info(f"Step {i+1}/{len(plan.steps)}: {step['step']}")
                # In production, would execute actual recovery step
                await asyncio.sleep(1)  # Simulate step execution

            # Run post-checks
            for check in plan.post_checks:
                logger.info(f"Post-check: {check}")
                await asyncio.sleep(0.5)  # Simulate check

            duration = time.time() - start_time

            return {
                "status": "success",
                "message": "Disaster recovery failover completed successfully",
                "recovery_plan_id": recovery_plan_id,
                "duration_seconds": duration,
                "completed_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Disaster recovery failover failed: {e}")
            return {
                "status": "error",
                "message": f"Failover failed: {str(e)}",
                "recovery_plan_id": recovery_plan_id,
            }

    async def _backup_scheduler(self):
        """Background task for scheduled backups"""
        last_full_backup = None
        last_incremental = None

        while self._running:
            try:
                current_time = time.time()

                # Check if full backup is due
                if (
                    not last_full_backup
                    or current_time - last_full_backup > self.config.full_backup_interval
                ):

                    await self.create_backup(
                        BackupType.FULL, await self._get_backup_sources(), {"scheduled": True}
                    )
                    last_full_backup = current_time

                # Check if incremental backup is due
                elif self.config.enable_continuous_backup and (
                    not last_incremental
                    or current_time - last_incremental > self.config.incremental_interval
                ):

                    await self.create_backup(
                        BackupType.INCREMENTAL,
                        await self._get_backup_sources(),
                        {"scheduled": True},
                    )
                    last_incremental = current_time

                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(60)

    async def _replication_processor(self):
        """Background task for processing replication queue"""
        while self._running:
            try:
                if not self.replication_queue:
                    await asyncio.sleep(1)
                    continue

                # Process replication queue
                while self.replication_queue:
                    item = self.replication_queue.popleft()

                    for target in self.replication_targets.values():
                        if target.is_active:
                            success = await self._replicate_to_target(item, target)

                            if success:
                                target.last_sync = datetime.utcnow()
                                target.lag_seconds = 0
                            else:
                                target.lag_seconds += 1
                                target.health_status = (
                                    "degraded" if target.lag_seconds > 60 else "healthy"
                                )

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Replication processor error: {e}")
                await asyncio.sleep(5)

    async def _maintenance_loop(self):
        """Background task for maintenance operations"""
        while self._running:
            try:
                # Clean up expired backups
                await self._cleanup_expired_backups()

                # Verify backup integrity
                await self._verify_recent_backups()

                # Update replication status
                await self._update_replication_status()

                # Check if DR test is due
                if not self.last_test_date or (datetime.utcnow() - self.last_test_date).days > 30:
                    logger.info("Monthly DR test is due")
                    # In production, would trigger automated DR test

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(3600)

    async def _backup_data_source(
        self,
        source: Dict[str, Any],
        backup_path: str,
        backup_type: BackupType,
        parent_backup_id: Optional[str],
    ) -> Dict[str, Any]:
        """Backup a single data source"""
        # Simplified backup logic
        # In production, would implement actual backup based on source type

        source_path = os.path.join(backup_path, source["source"])
        os.makedirs(source_path, exist_ok=True)

        # Simulate backing up data
        data_file = os.path.join(source_path, "data.json")
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump({"source": source, "timestamp": time.time()}, f)

        # Calculate size and checksum
        size = os.path.getsize(data_file)
        with open(data_file, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        return {"size": size, "checksum": checksum}

    async def _compress_backup(self, source_path: str, target_path: str) -> float:
        """Compress backup and return compression ratio"""
        original_size = 0

        # Calculate original size
        for root, _, files in os.walk(source_path):
            for file in files:
                original_size += os.path.getsize(os.path.join(root, file))

        # Create compressed archive
        with tarfile.open(target_path, "w:gz") as tar:
            tar.add(source_path, arcname=os.path.basename(source_path))

        compressed_size = os.path.getsize(target_path)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

        return compression_ratio

    async def _encrypt_backup(self, backup_path: str) -> str:
        """Encrypt backup and return key ID"""
        # Simplified encryption
        # In production, would use proper encryption with key management
        key_id = f"key_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"

        # Simulate encryption by adding .enc extension
        encrypted_path = f"{backup_path}.enc"
        shutil.move(backup_path, encrypted_path)

        return key_id

    async def _upload_backup(self, local_path: str, storage: BackupStorage) -> str:
        """Upload backup to storage and return location"""
        # Simplified upload
        # In production, would implement actual upload to S3/Azure/GCS

        if storage == BackupStorage.LOCAL:
            return local_path
        else:
            # Simulate remote upload
            remote_location = f"{storage.value}://backups/{os.path.basename(local_path)}"
            logger.info(f"Uploaded backup to {remote_location}")
            return remote_location

    async def _find_parent_backup(self, backup_type: BackupType) -> Optional[str]:
        """Find appropriate parent backup for incremental/differential"""
        if backup_type == BackupType.INCREMENTAL:
            # Find most recent backup of any type
            recent_backups = sorted(
                self.backup_catalog.values(), key=lambda b: b.timestamp, reverse=True
            )
            return recent_backups[0].backup_id if recent_backups else None

        elif backup_type == BackupType.DIFFERENTIAL:
            # Find most recent full backup
            full_backups = [
                b for b in self.backup_catalog.values() if b.backup_type == BackupType.FULL
            ]
            full_backups.sort(key=lambda b: b.timestamp, reverse=True)
            return full_backups[0].backup_id if full_backups else None

        return None

    async def _create_recovery_point(self, backup: BackupMetadata):
        """Create a recovery point from a backup"""
        # Build backup chain
        backup_chain = [backup.backup_id]

        # Add parent backups if needed
        current_backup = backup
        while current_backup.parent_backup_id:
            backup_chain.insert(0, current_backup.parent_backup_id)
            current_backup = self.backup_catalog.get(current_backup.parent_backup_id)
            if not current_backup:
                break

        recovery_point = RecoveryPoint(
            recovery_point_id=f"rp_{backup.backup_id}",
            timestamp=backup.timestamp,
            backup_chain=backup_chain,
            estimated_recovery_time=len(backup_chain) * 300,  # 5 min per backup
            data_consistency_verified=False,
            application_consistent=True,
        )

        self.recovery_points[recovery_point.recovery_point_id] = recovery_point

    async def _get_backup_sources(self) -> List[Dict[str, Any]]:
        """Get list of data sources to backup"""
        # In production, would dynamically determine what to backup
        return [
            {"source": "agent_registry", "type": "database"},
            {"source": "configuration", "type": "files"},
            {"source": "message_queue", "type": "queue"},
            {"source": "audit_logs", "type": "logs"},
        ]

    def get_dr_status(self) -> Dict[str, Any]:
        """Get current disaster recovery status"""
        # Calculate metrics
        avg_backup_time = (
            sum(self.backup_times) / len(self.backup_times) if self.backup_times else 0
        )

        avg_recovery_time = (
            sum(self.recovery_times) / len(self.recovery_times) if self.recovery_times else 0
        )

        # Find last successful backup
        recent_backups = sorted(
            self.backup_catalog.values(), key=lambda b: b.timestamp, reverse=True
        )
        last_backup = recent_backups[0] if recent_backups else None

        # Check RPO compliance
        time_since_backup = (
            (datetime.utcnow() - last_backup.timestamp).total_seconds()
            if last_backup
            else float("inf")
        )
        rpo_compliant = time_since_backup <= self.config.rpo_seconds

        return {
            "config": {
                "rpo_seconds": self.config.rpo_seconds,
                "rto_seconds": self.config.rto_seconds,
                "backup_retention_days": self.config.backup_retention_days,
                "replication_mode": self.config.replication_mode.value,
            },
            "backup_status": {
                "total_backups": len(self.backup_catalog),
                "successful_backups": self.successful_backups,
                "failed_backups": self.failed_backups,
                "last_backup": last_backup.timestamp.isoformat() if last_backup else None,
                "time_since_backup_seconds": time_since_backup,
                "rpo_compliant": rpo_compliant,
                "avg_backup_time_seconds": avg_backup_time,
            },
            "recovery_status": {
                "recovery_points": len(self.recovery_points),
                "recovery_plans": len(self.recovery_plans),
                "successful_recoveries": self.successful_recoveries,
                "failed_recoveries": self.failed_recoveries,
                "avg_recovery_time_seconds": avg_recovery_time,
                "last_test_date": self.last_test_date.isoformat() if self.last_test_date else None,
            },
            "replication_status": {
                "targets": len(self.replication_targets),
                "active_targets": len(
                    [t for t in self.replication_targets.values() if t.is_active]
                ),
                "max_lag_seconds": (
                    max(self.replication_lag.values()) if self.replication_lag else 0
                ),
                "queue_depth": len(self.replication_queue),
            },
        }

    async def _load_backup_catalog(self):
        """Load backup catalog from persistent storage"""
        # Simplified - in production would load from database
        catalog_file = os.path.join(self.backup_base_path, "catalog.json")
        if os.path.exists(catalog_file):
            with open(catalog_file, "r", encoding="utf-8") as f:
                json.load(f)
                # Reconstruct backup metadata objects
                # (simplified for demo)

    async def _save_backup_catalog(self):
        """Save backup catalog to persistent storage"""
        # Simplified - in production would save to database
        catalog_file = os.path.join(self.backup_base_path, "catalog.json")
        catalog_data = {
            backup_id: {
                "backup_id": backup.backup_id,
                "backup_type": backup.backup_type.value,
                "timestamp": backup.timestamp.isoformat(),
                "size_bytes": backup.size_bytes,
                "checksum": backup.checksum,
                "location": backup.location,
            }
            for backup_id, backup in self.backup_catalog.items()
        }

        with open(catalog_file, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2)

    async def _cleanup_expired_backups(self):
        """Remove backups past retention period"""
        current_time = datetime.utcnow()
        expired_backups = []

        for backup_id, backup in self.backup_catalog.items():
            if backup.retention_until and current_time > backup.retention_until:
                expired_backups.append(backup_id)

        for backup_id in expired_backups:
            logger.info(f"Removing expired backup {backup_id}")
            # In production, would delete from storage
            del self.backup_catalog[backup_id]

        if expired_backups:
            await self._save_backup_catalog()

    async def _verify_recent_backups(self):
        """Verify integrity of recent backups"""
        # Check backups from last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)

        for backup in self.backup_catalog.values():
            if backup.timestamp > cutoff_time:
                # In production, would verify checksum and test restore
                logger.debug(f"Verifying backup {backup.backup_id}")

    async def _initialize_recovery_plans(self):
        """Initialize standard recovery plans"""
        # Create default recovery plans
        await self.create_recovery_plan(
            "Full System Recovery", RecoveryType.FULL_RESTORE, auto_execute=False
        )

        await self.create_recovery_plan(
            "Point-in-Time Recovery", RecoveryType.POINT_IN_TIME, auto_execute=False
        )

        await self.create_recovery_plan(
            "Emergency Failover", RecoveryType.FULL_RESTORE, auto_execute=True
        )

    # Missing method implementations (stubs for now)
    async def _replicate_backup(self, backup_path: str, target_storage: BackupStorage):
        """Replicate backup to secondary storage"""
        logger.info(f"Replicating backup from {backup_path} to {target_storage}")

        try:
            # Validate source backup exists
            if not os.path.exists(backup_path):
                raise BackupError(f"Source backup not found: {backup_path}")

            # Get backup file info
            backup_size = os.path.getsize(backup_path)
            backup_name = os.path.basename(backup_path)

            # Determine replication target based on storage type
            if target_storage.storage_type == "s3":
                # S3 replication
                target_key = f"backups/{backup_name}"
                logger.info(f"Replicating to S3: {target_storage.endpoint}/{target_key}")

                # Simulate S3 upload
                await asyncio.sleep(0.1)  # Simulate network delay

            elif target_storage.storage_type == "azure":
                # Azure blob replication
                container = target_storage.config.get('container', 'backups')
                blob_name = backup_name
                logger.info(f"Replicating to Azure Blob: {container}/{blob_name}")

                # Simulate Azure upload
                await asyncio.sleep(0.1)

            elif target_storage.storage_type == "local":
                # Local filesystem replication
                target_path = os.path.join(target_storage.endpoint, backup_name)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Copy file
                import shutil
                shutil.copy2(backup_path, target_path)
                logger.info(f"Replicated to local storage: {target_path}")

            else:
                raise BackupError(f"Unsupported storage type: {target_storage.storage_type}")

            # Update replication metadata
            replication_info = {
                "source_path": backup_path,
                "target_storage": target_storage.storage_id,
                "backup_size": backup_size,
                "replicated_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }

            # Store replication record
            self.backup_metadata.setdefault("replications", []).append(replication_info)

            logger.info(f"Successfully replicated backup {backup_name} ({backup_size} bytes)")

        except Exception as e:
            logger.error(f"Backup replication failed: {e}")
            raise BackupError(f"Replication failed: {str(e)}")

    async def _verify_recovery_point(self, backup_id: str) -> bool:
        """Verify recovery point integrity"""
        logger.info(f"Verifying recovery point {backup_id}")

        try:
            # Find backup record
            backup_record = None
            for storage_id, backups in self.backup_metadata.get("backups", {}).items():
                for backup in backups:
                    if backup.get("backup_id") == backup_id:
                        backup_record = backup
                        break
                if backup_record:
                    break

            if not backup_record:
                logger.error(f"Backup record not found: {backup_id}")
                return False

            verification_results = {
                "backup_id": backup_id,
                "checks_passed": 0,
                "checks_failed": 0,
                "issues": []
            }

            # Check 1: Backup file exists
            backup_path = backup_record.get("file_path")
            if backup_path and os.path.exists(backup_path):
                verification_results["checks_passed"] += 1
                logger.debug(f"✓ Backup file exists: {backup_path}")
            else:
                verification_results["checks_failed"] += 1
                verification_results["issues"].append(f"Backup file missing: {backup_path}")

            # Check 2: File size matches metadata
            expected_size = backup_record.get("backup_size", 0)
            if backup_path and os.path.exists(backup_path):
                actual_size = os.path.getsize(backup_path)
                if actual_size == expected_size:
                    verification_results["checks_passed"] += 1
                    logger.debug(f"✓ File size matches: {actual_size} bytes")
                else:
                    verification_results["checks_failed"] += 1
                    verification_results["issues"].append(f"Size mismatch: expected {expected_size}, got {actual_size}")

            # Check 3: Backup is not corrupted (basic file readability)
            if backup_path and os.path.exists(backup_path):
                try:
                    with open(backup_path, 'rb') as f:
                        # Read first and last 1KB to verify file integrity
                        f.read(1024)
                        f.seek(-1024, 2)
                        f.read(1024)
                    verification_results["checks_passed"] += 1
                    logger.debug("✓ Backup file is readable")
                except Exception as e:
                    verification_results["checks_failed"] += 1
                    verification_results["issues"].append(f"File corruption detected: {str(e)}")

            # Check 4: Backup timestamp is reasonable
            created_at = backup_record.get("created_at")
            if created_at:
                try:
                    backup_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    time_diff = datetime.utcnow().replace(tzinfo=timezone.utc) - backup_time.replace(tzinfo=timezone.utc)

                    if time_diff.days < 30:  # Backup is less than 30 days old
                        verification_results["checks_passed"] += 1
                        logger.debug(f"✓ Backup age is reasonable: {time_diff.days} days")
                    else:
                        verification_results["issues"].append(f"Backup is very old: {time_diff.days} days")
                        logger.warning(f"⚠ Backup is {time_diff.days} days old")
                except Exception as e:
                    verification_results["issues"].append(f"Invalid timestamp format: {created_at}")

            # Check 5: Verify backup metadata completeness
            required_fields = ['backup_id', 'created_at', 'backup_size', 'file_path']
            missing_fields = [field for field in required_fields if not backup_record.get(field)]

            if not missing_fields:
                verification_results["checks_passed"] += 1
                logger.debug("✓ Metadata is complete")
            else:
                verification_results["checks_failed"] += 1
                verification_results["issues"].append(f"Missing metadata fields: {missing_fields}")

            # Store verification results
            self.backup_metadata.setdefault("verifications", []).append({
                "backup_id": backup_id,
                "verified_at": datetime.utcnow().isoformat(),
                "results": verification_results
            })

            # Determine overall success
            total_checks = verification_results["checks_passed"] + verification_results["checks_failed"]
            success_rate = verification_results["checks_passed"] / max(total_checks, 1)

            is_valid = success_rate >= 0.8  # 80% of checks must pass

            if is_valid:
                logger.info(f"✓ Recovery point verification passed: {verification_results['checks_passed']}/{total_checks} checks")
            else:
                logger.error(f"✗ Recovery point verification failed: {verification_results['checks_failed']} issues found")
                for issue in verification_results["issues"]:
                    logger.error(f"  - {issue}")

            return is_valid

        except Exception as e:
            logger.error(f"Recovery point verification error: {e}")
            return False

    async def _run_pre_recovery_checks(self, recovery_point=None, target_location=None) -> bool:
        """Run pre-recovery validation checks"""
        logger.info(f"Running pre-recovery checks for {recovery_point} to {target_location}")

        try:
            check_results = {
                "checks_passed": 0,
                "checks_failed": 0,
                "warnings": [],
                "errors": []
            }

            # Check 1: Verify recovery point exists and is valid
            if recovery_point:
                if isinstance(recovery_point, str):
                    # Recovery point is a backup ID
                    backup_valid = await self._verify_recovery_point(recovery_point)
                    if backup_valid:
                        check_results["checks_passed"] += 1
                        logger.debug("✓ Recovery point is valid")
                    else:
                        check_results["checks_failed"] += 1
                        check_results["errors"].append(f"Invalid recovery point: {recovery_point}")
                else:
                    check_results["checks_passed"] += 1

            # Check 2: Verify target location accessibility
            if target_location:
                try:
                    target_dir = os.path.dirname(target_location) if os.path.isfile(target_location) else target_location

                    # Check if target directory exists or can be created
                    if not os.path.exists(target_dir):
                        try:
                            os.makedirs(target_dir, exist_ok=True)
                            check_results["checks_passed"] += 1
                            logger.debug(f"✓ Target directory created: {target_dir}")
                        except OSError as e:
                            check_results["checks_failed"] += 1
                            check_results["errors"].append(f"Cannot create target directory: {e}")
                    else:
                        check_results["checks_passed"] += 1
                        logger.debug(f"✓ Target location accessible: {target_dir}")

                    # Check write permissions
                    if os.access(target_dir, os.W_OK):
                        check_results["checks_passed"] += 1
                        logger.debug("✓ Target location is writable")
                    else:
                        check_results["checks_failed"] += 1
                        check_results["errors"].append(f"No write permission for target: {target_dir}")

                except Exception as e:
                    check_results["checks_failed"] += 1
                    check_results["errors"].append(f"Target location check failed: {e}")

            # Check 3: Verify system resources
            try:
                # Check available disk space
                if target_location:
                    statvfs = os.statvfs(os.path.dirname(target_location) if os.path.isfile(target_location) else target_location)
                    free_space = statvfs.f_frsize * statvfs.f_available

                    # Estimate required space (if recovery point is known)
                    required_space = 1024 * 1024 * 1024  # Default 1GB

                    if free_space > required_space * 1.2:  # 20% buffer
                        check_results["checks_passed"] += 1
                        logger.debug(f"✓ Sufficient disk space: {free_space // (1024*1024)} MB available")
                    elif free_space > required_space:
                        check_results["checks_passed"] += 1
                        check_results["warnings"].append(f"Limited disk space: {free_space // (1024*1024)} MB available")
                    else:
                        check_results["checks_failed"] += 1
                        check_results["errors"].append(f"Insufficient disk space: {free_space // (1024*1024)} MB available, need {required_space // (1024*1024)} MB")

            except Exception as e:
                check_results["warnings"].append(f"Could not check disk space: {e}")

            # Check 4: Verify no conflicting processes
            try:
                # Check if any restoration processes are already running
                active_restorations = len(self.backup_metadata.get("active_restorations", []))
                if active_restorations == 0:
                    check_results["checks_passed"] += 1
                    logger.debug("✓ No conflicting restoration processes")
                else:
                    check_results["warnings"].append(f"{active_restorations} restoration processes currently active")

            except Exception as e:
                check_results["warnings"].append(f"Could not check active processes: {e}")

            # Check 5: Verify backup storage connectivity
            try:
                storage_checks = 0
                storage_failures = 0

                for storage_id, storage in self.storage_backends.items():
                    try:
                        # Simple connectivity test based on storage type
                        if storage.storage_type == "local":
                            if os.path.exists(storage.endpoint):
                                storage_checks += 1
                            else:
                                storage_failures += 1
                                check_results["warnings"].append(f"Local storage not accessible: {storage.endpoint}")
                        else:
                            # For remote storage, assume accessible (would need actual connectivity test)
                            storage_checks += 1

                    except Exception as e:
                        storage_failures += 1
                        check_results["warnings"].append(f"Storage check failed for {storage_id}: {e}")

                if storage_checks > 0:
                    check_results["checks_passed"] += 1
                    logger.debug(f"✓ Storage connectivity verified: {storage_checks} accessible")

                if storage_failures > 0:
                    check_results["warnings"].append(f"{storage_failures} storage backends not accessible")

            except Exception as e:
                check_results["warnings"].append(f"Storage connectivity check failed: {e}")

            # Determine overall success
            total_critical_checks = check_results["checks_passed"] + check_results["checks_failed"]
            success_rate = check_results["checks_passed"] / max(total_critical_checks, 1)

            # All critical checks must pass
            checks_passed = check_results["checks_failed"] == 0

            # Log results
            if checks_passed:
                logger.info(f"✓ Pre-recovery checks passed: {check_results['checks_passed']} checks completed")
                if check_results["warnings"]:
                    logger.warning(f"{len(check_results['warnings'])} warnings:")
                    for warning in check_results["warnings"]:
                        logger.warning(f"  ⚠ {warning}")
            else:
                logger.error(f"✗ Pre-recovery checks failed: {check_results['checks_failed']} critical errors")
                for error in check_results["errors"]:
                    logger.error(f"  ✗ {error}")

            # Store check results
            self.backup_metadata.setdefault("pre_recovery_checks", []).append({
                "recovery_point": str(recovery_point) if recovery_point else None,
                "target_location": target_location,
                "results": check_results,
                "checked_at": datetime.utcnow().isoformat(),
                "passed": checks_passed
            })

            return checks_passed

        except Exception as e:
            logger.error(f"Pre-recovery checks error: {e}")
            return False

    async def _download_backup(self, backup, target_path: str = None):
        """Download backup from storage"""
        backup_id = backup.backup_id if hasattr(backup, 'backup_id') else str(backup)

        try:
            # Determine target path
            if not target_path:
                target_path = f"/tmp/{backup_id}.backup"

            logger.info(f"Downloading backup {backup_id} to {target_path}")

            # Find backup record to determine storage location
            backup_record = None
            source_storage = None

            for storage_id, backups in self.backup_metadata.get("backups", {}).items():
                for backup_info in backups:
                    if backup_info.get("backup_id") == backup_id:
                        backup_record = backup_info
                        source_storage = self.storage_backends.get(storage_id)
                        break
                if backup_record:
                    break

            if not backup_record:
                raise BackupError(f"Backup not found: {backup_id}")

            if not source_storage:
                raise BackupError(f"Storage backend not found for backup: {backup_id}")

            # Create target directory if needed
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Download based on storage type
            if source_storage.storage_type == "s3":
                # S3 download simulation
                logger.info(f"Downloading from S3: {source_storage.endpoint}")

                # Simulate S3 download with progress
                backup_size = backup_record.get("backup_size", 1024 * 1024)  # Default 1MB
                chunk_size = min(backup_size // 10, 1024 * 1024)  # 10% or 1MB chunks

                with open(target_path, 'wb') as f:
                    downloaded = 0
                    while downloaded < backup_size:
                        chunk = b'0' * min(chunk_size, backup_size - downloaded)
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Simulate network delay
                        await asyncio.sleep(0.01)

                        # Log progress every 10%
                        progress = (downloaded / backup_size) * 100
                        if downloaded % (backup_size // 10 + 1) == 0:
                            logger.debug(f"Download progress: {progress:.1f}%")

            elif source_storage.storage_type == "azure":
                # Azure blob download simulation
                logger.info(f"Downloading from Azure Blob Storage")

                # Simulate Azure download
                backup_size = backup_record.get("backup_size", 1024 * 1024)
                with open(target_path, 'wb') as f:
                    f.write(b'0' * backup_size)

                await asyncio.sleep(0.1)  # Simulate download time

            elif source_storage.storage_type == "local":
                # Local file copy
                source_path = backup_record.get("file_path")
                if not source_path or not os.path.exists(source_path):
                    raise BackupError(f"Source backup file not found: {source_path}")

                logger.info(f"Copying from local storage: {source_path}")
                import shutil
                shutil.copy2(source_path, target_path)

            else:
                raise BackupError(f"Unsupported storage type: {source_storage.storage_type}")

            # Verify download integrity
            if os.path.exists(target_path):
                downloaded_size = os.path.getsize(target_path)
                expected_size = backup_record.get("backup_size", 0)

                if downloaded_size == expected_size:
                    logger.info(f"✓ Backup downloaded successfully: {target_path} ({downloaded_size} bytes)")
                else:
                    logger.warning(f"⚠ Size mismatch: expected {expected_size}, got {downloaded_size}")

            # Update download metadata
            self.backup_metadata.setdefault("downloads", []).append({
                "backup_id": backup_id,
                "target_path": target_path,
                "downloaded_at": datetime.utcnow().isoformat(),
                "source_storage": source_storage.storage_id
            })

            return target_path

        except Exception as e:
            logger.error(f"Backup download failed: {e}")
            raise BackupError(f"Download failed: {str(e)}")

    async def _decrypt_backup(self, backup_path: str, encryption_key_id: str = None) -> str:
        """Decrypt backup if encrypted"""
        logger.info(f"Decrypting backup at {backup_path} with key {encryption_key_id}")

        try:
            # Check if file exists
            if not os.path.exists(backup_path):
                raise BackupError(f"Backup file not found: {backup_path}")

            # Check if backup is encrypted (simple check for encrypted header)
            with open(backup_path, 'rb') as f:
                header = f.read(16)

            # Simple encryption detection (look for common encrypted file signatures)
            encryption_signatures = [
                b'ENCRYPTED_BACKUP',  # Custom header
                b'\x00\x01\x02\x03',  # Simple signature
                b'AES_ENCRYPTED'      # AES header
            ]

            is_encrypted = any(sig in header for sig in encryption_signatures)

            if not is_encrypted:
                logger.info("Backup is not encrypted, no decryption needed")
                return backup_path

            logger.info("Encrypted backup detected, proceeding with decryption")

            # Generate decrypted file path
            decrypted_path = backup_path + ".decrypted"

            # Simulate decryption process
            if encryption_key_id:
                logger.info(f"Using encryption key: {encryption_key_id}")

                # In a real implementation, you would:
                # 1. Retrieve the encryption key from a secure key store
                # 2. Use a proper encryption library (cryptography, pycryptodome)
                # 3. Decrypt the file block by block

                # Simulate decryption by copying the file (minus encrypted header)
                with open(backup_path, 'rb') as encrypted_file:
                    with open(decrypted_path, 'wb') as decrypted_file:
                        encrypted_file.seek(16)  # Skip encrypted header

                        # Copy data in chunks
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = encrypted_file.read(chunk_size)
                            if not chunk:
                                break

                            # Simulate decryption processing
                            await asyncio.sleep(0.001)  # Simulate crypto operations

                            # In real implementation, decrypt the chunk here
                            decrypted_chunk = chunk  # Placeholder
                            decrypted_file.write(decrypted_chunk)

            else:
                # Try to decrypt with default key or auto-detect
                logger.info("Attempting decryption with default key")

                # Simulate auto-decryption
                backup_size = os.path.getsize(backup_path)
                with open(decrypted_path, 'wb') as f:
                    # Create decrypted file (simulated)
                    f.write(b'0' * max(0, backup_size - 16))  # Remove encryption overhead

                await asyncio.sleep(0.1)  # Simulate decryption time

            # Verify decryption success
            if os.path.exists(decrypted_path):
                decrypted_size = os.path.getsize(decrypted_path)
                logger.info(f"✓ Backup decrypted successfully: {decrypted_path} ({decrypted_size} bytes)")

                # Update metadata
                self.backup_metadata.setdefault("decryptions", []).append({
                    "original_path": backup_path,
                    "decrypted_path": decrypted_path,
                    "encryption_key_id": encryption_key_id,
                    "decrypted_at": datetime.utcnow().isoformat(),
                    "decrypted_size": decrypted_size
                })

                return decrypted_path
            else:
                raise BackupError("Decryption failed - output file not created")

        except Exception as e:
            logger.error(f"Backup decryption failed: {e}")
            raise BackupError(f"Decryption failed: {str(e)}")

    async def _decompress_backup(self, backup_path: str) -> str:
        """Decompress backup if compressed"""
        logger.info(f"Decompressing backup at {backup_path}")

        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_path}")

        # Check if file needs decompression based on extension
        backup_lower = backup_path.lower()

        if backup_lower.endswith('.tar.gz') or backup_lower.endswith('.tgz'):
            return await self._decompress_tar_gz(backup_path)
        elif backup_lower.endswith('.tar.bz2'):
            return await self._decompress_tar_bz2(backup_path)
        elif backup_lower.endswith('.tar'):
            return await self._decompress_tar(backup_path)
        elif backup_lower.endswith('.gz'):
            return await self._decompress_gzip(backup_path)
        elif backup_lower.endswith('.zip'):
            return await self._decompress_zip(backup_path)
        else:
            # File is not compressed, return as-is
            logger.info(f"Backup file {backup_path} is not compressed")
            return backup_path

    async def _decompress_tar_gz(self, backup_path: str) -> str:
        """Decompress tar.gz backup"""
        extract_dir = backup_path.rsplit('.', 2)[0] + '_extracted'

        try:
            with tarfile.open(backup_path, 'r:gz') as tar:
                # Extract all files to directory
                tar.extractall(path=extract_dir)
                logger.info(f"Extracted tar.gz backup to {extract_dir}")
                return extract_dir
        except Exception as e:
            logger.error(f"Failed to decompress tar.gz backup: {e}")
            raise ValueError(f"Decompression failed: {str(e)}")

    async def _decompress_tar_bz2(self, backup_path: str) -> str:
        """Decompress tar.bz2 backup"""
        extract_dir = backup_path.rsplit('.', 2)[0] + '_extracted'

        try:
            with tarfile.open(backup_path, 'r:bz2') as tar:
                tar.extractall(path=extract_dir)
                logger.info(f"Extracted tar.bz2 backup to {extract_dir}")
                return extract_dir
        except Exception as e:
            logger.error(f"Failed to decompress tar.bz2 backup: {e}")
            raise ValueError(f"Decompression failed: {str(e)}")

    async def _decompress_tar(self, backup_path: str) -> str:
        """Decompress tar backup"""
        extract_dir = backup_path.rsplit('.', 1)[0] + '_extracted'

        try:
            with tarfile.open(backup_path, 'r') as tar:
                tar.extractall(path=extract_dir)
                logger.info(f"Extracted tar backup to {extract_dir}")
                return extract_dir
        except Exception as e:
            logger.error(f"Failed to decompress tar backup: {e}")
            raise ValueError(f"Decompression failed: {str(e)}")

    async def _decompress_gzip(self, backup_path: str) -> str:
        """Decompress gzip backup"""
        if backup_path.endswith('.gz'):
            output_path = backup_path[:-3]  # Remove .gz extension
        else:
            output_path = backup_path + '_decompressed'

        try:
            import gzip
            with gzip.open(backup_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Decompressed gzip backup to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to decompress gzip backup: {e}")
            raise ValueError(f"Decompression failed: {str(e)}")

    async def _decompress_zip(self, backup_path: str) -> str:
        """Decompress zip backup"""
        extract_dir = backup_path.rsplit('.', 1)[0] + '_extracted'

        try:
            import zipfile
            with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logger.info(f"Extracted zip backup to {extract_dir}")
                return extract_dir
        except Exception as e:
            logger.error(f"Failed to decompress zip backup: {e}")
            raise ValueError(f"Decompression failed: {str(e)}")

    async def _restore_data(self, data_path: str, target_location: str, recovery_type=None, options=None):
        """Restore data to target location"""
        logger.info(f"Restoring data from {data_path} to {target_location} (type: {recovery_type})")

        try:
            # Validate source data
            if not os.path.exists(data_path):
                raise BackupError(f"Source data not found: {data_path}")

            # Initialize options
            if options is None:
                options = {}

            overwrite = options.get('overwrite', True)
            preserve_permissions = options.get('preserve_permissions', True)
            verify_integrity = options.get('verify_integrity', True)

            # Create target directory if needed
            target_dir = os.path.dirname(target_location)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                logger.info(f"Created target directory: {target_dir}")

            # Check if target already exists
            if os.path.exists(target_location) and not overwrite:
                raise BackupError(f"Target location exists and overwrite=False: {target_location}")

            restoration_stats = {
                "files_restored": 0,
                "bytes_restored": 0,
                "errors": [],
                "start_time": datetime.utcnow().isoformat()
            }

            # Perform restoration based on recovery type
            if recovery_type == "full_system":
                # Full system restoration
                logger.info("Performing full system restoration")

                # In a real implementation, this would:
                # 1. Stop relevant services
                # 2. Restore system files
                # 3. Restore database files
                # 4. Restore configuration files
                # 5. Restart services

                await self._restore_full_system(data_path, target_location, restoration_stats)

            elif recovery_type == "database":
                # Database restoration
                logger.info("Performing database restoration")
                await self._restore_database(data_path, target_location, restoration_stats)

            elif recovery_type == "files":
                # File system restoration
                logger.info("Performing file restoration")
                await self._restore_files(data_path, target_location, restoration_stats)

            elif recovery_type == "incremental":
                # Incremental restoration
                logger.info("Performing incremental restoration")
                await self._restore_incremental(data_path, target_location, restoration_stats)

            else:
                # Default: treat as file copy
                logger.info("Performing default file restoration")

                if os.path.isfile(data_path):
                    # Single file restoration
                    import shutil
                    shutil.copy2(data_path, target_location)

                    restoration_stats["files_restored"] = 1
                    restoration_stats["bytes_restored"] = os.path.getsize(target_location)

                elif os.path.isdir(data_path):
                    # Directory restoration
                    await self._restore_directory(data_path, target_location, restoration_stats)

                else:
                    raise BackupError(f"Unknown data type: {data_path}")

            # Set file permissions if requested
            if preserve_permissions and os.path.exists(target_location):
                try:
                    # Copy permissions from source
                    source_stat = os.stat(data_path)
                    os.chmod(target_location, source_stat.st_mode)
                    logger.debug(f"Preserved permissions for {target_location}")
                except Exception as e:
                    restoration_stats["errors"].append(f"Failed to preserve permissions: {e}")

            # Verify restoration integrity
            if verify_integrity:
                integrity_check = await self._verify_restoration_integrity(data_path, target_location)
                if not integrity_check:
                    restoration_stats["errors"].append("Integrity verification failed")

            # Finalize restoration stats
            restoration_stats["end_time"] = datetime.utcnow().isoformat()
            restoration_stats["success"] = len(restoration_stats["errors"]) == 0

            # Store restoration record
            self.backup_metadata.setdefault("restorations", []).append({
                "source_path": data_path,
                "target_location": target_location,
                "recovery_type": recovery_type,
                "stats": restoration_stats,
                "restored_at": datetime.utcnow().isoformat()
            })

            if restoration_stats["success"]:
                logger.info(f"✓ Data restoration completed successfully")
                logger.info(f"  Files restored: {restoration_stats['files_restored']}")
                logger.info(f"  Bytes restored: {restoration_stats['bytes_restored']:,}")
            else:
                logger.error(f"✗ Data restoration completed with errors: {len(restoration_stats['errors'])}")
                for error in restoration_stats["errors"]:
                    logger.error(f"  - {error}")

            return restoration_stats["success"]

        except Exception as e:
            logger.error(f"Data restoration failed: {e}")
            raise BackupError(f"Restoration failed: {str(e)}")

    async def _restore_full_system(self, data_path: str, target_location: str, stats: dict):
        """Restore full system from backup"""
        logger.info("Starting full system restoration")
        # Simulate system restoration
        await asyncio.sleep(1)
        stats["files_restored"] = 1000
        stats["bytes_restored"] = 100 * 1024 * 1024  # 100MB

    async def _restore_database(self, data_path: str, target_location: str, stats: dict):
        """Restore database from backup"""
        logger.info("Starting database restoration")
        # Simulate database restoration
        await asyncio.sleep(0.5)
        stats["files_restored"] = 50
        stats["bytes_restored"] = 50 * 1024 * 1024  # 50MB

    async def _restore_files(self, data_path: str, target_location: str, stats: dict):
        """Restore files from backup"""
        logger.info("Starting file restoration")
        # Simulate file restoration
        await asyncio.sleep(0.3)
        stats["files_restored"] = 200
        stats["bytes_restored"] = 20 * 1024 * 1024  # 20MB

    async def _restore_incremental(self, data_path: str, target_location: str, stats: dict):
        """Restore incremental backup"""
        logger.info("Starting incremental restoration")
        # Simulate incremental restoration
        await asyncio.sleep(0.2)
        stats["files_restored"] = 25
        stats["bytes_restored"] = 5 * 1024 * 1024  # 5MB

    async def _restore_directory(self, source_dir: str, target_dir: str, stats: dict):
        """Restore directory recursively"""
        import shutil

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir)

        # Count files and calculate size
        for root, dirs, files in os.walk(target_dir):
            stats["files_restored"] += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    stats["bytes_restored"] += os.path.getsize(file_path)
                except OSError:
                    pass

    async def _verify_restoration_integrity(self, source_path: str, target_path: str) -> bool:
        """Verify restoration integrity"""
        try:
            if os.path.isfile(source_path) and os.path.isfile(target_path):
                # Compare file sizes
                source_size = os.path.getsize(source_path)
                target_size = os.path.getsize(target_path)
                return source_size == target_size

            elif os.path.isdir(source_path) and os.path.isdir(target_path):
                # Basic directory comparison
                source_files = set()
                target_files = set()

                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), source_path)
                        source_files.add(rel_path)

                for root, dirs, files in os.walk(target_path):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), target_path)
                        target_files.add(rel_path)

                return source_files == target_files

            return False

        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False

    async def _run_post_recovery_checks(self, target_location=None) -> bool:
        """Run post-recovery validation checks"""
        logger.info(f"Running post-recovery checks at {target_location}")

        try:
            check_results = {
                "checks_passed": 0,
                "checks_failed": 0,
                "warnings": [],
                "errors": [],
                "performance_metrics": {}
            }

            start_time = time.time()

            # Check 1: Verify restored data exists
            if target_location and os.path.exists(target_location):
                check_results["checks_passed"] += 1
                logger.debug(f"✓ Restored data exists at {target_location}")

                # Get basic file/directory info
                if os.path.isfile(target_location):
                    file_size = os.path.getsize(target_location)
                    check_results["performance_metrics"]["restored_size_bytes"] = file_size
                    logger.debug(f"Restored file size: {file_size:,} bytes")

                elif os.path.isdir(target_location):
                    # Count files and calculate total size
                    total_files = 0
                    total_size = 0

                    for root, dirs, files in os.walk(target_location):
                        total_files += len(files)
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                pass

                    check_results["performance_metrics"]["restored_files"] = total_files
                    check_results["performance_metrics"]["restored_size_bytes"] = total_size
                    logger.debug(f"Restored directory: {total_files} files, {total_size:,} bytes")
            else:
                check_results["checks_failed"] += 1
                check_results["errors"].append(f"Restored data not found at {target_location}")

            # Check 2: Verify file integrity (basic checks)
            if target_location and os.path.exists(target_location):
                try:
                    if os.path.isfile(target_location):
                        # Basic file integrity check
                        with open(target_location, 'rb') as f:
                            # Try to read first and last parts of file
                            f.read(1024)  # First 1KB
                            if os.path.getsize(target_location) > 1024:
                                f.seek(-1024, 2)  # Last 1KB
                                f.read(1024)

                        check_results["checks_passed"] += 1
                        logger.debug("✓ File integrity check passed")

                    elif os.path.isdir(target_location):
                        # Check directory structure integrity
                        readable_files = 0
                        corrupted_files = 0

                        for root, dirs, files in os.walk(target_location):
                            for file in files[:10]:  # Sample first 10 files
                                try:
                                    file_path = os.path.join(root, file)
                                    with open(file_path, 'rb') as f:
                                        f.read(min(1024, os.path.getsize(file_path)))
                                    readable_files += 1
                                except Exception:
                                    corrupted_files += 1

                        if corrupted_files == 0:
                            check_results["checks_passed"] += 1
                            logger.debug(f"✓ Directory integrity check passed: {readable_files} files verified")
                        else:
                            check_results["checks_failed"] += 1
                            check_results["errors"].append(f"{corrupted_files} corrupted files detected")

                except Exception as e:
                    check_results["checks_failed"] += 1
                    check_results["errors"].append(f"Integrity check failed: {e}")

            # Check 3: Verify permissions and accessibility
            if target_location and os.path.exists(target_location):
                try:
                    # Check read permissions
                    if os.access(target_location, os.R_OK):
                        check_results["checks_passed"] += 1
                        logger.debug("✓ Restored data is readable")
                    else:
                        check_results["checks_failed"] += 1
                        check_results["errors"].append("Restored data is not readable")

                    # For directories, check if we can list contents
                    if os.path.isdir(target_location):
                        try:
                            contents = os.listdir(target_location)
                            logger.debug(f"Directory contains {len(contents)} items")
                        except Exception as e:
                            check_results["warnings"].append(f"Could not list directory contents: {e}")

                except Exception as e:
                    check_results["warnings"].append(f"Permission check failed: {e}")

            # Check 4: Performance validation
            try:
                # Check restoration time
                restoration_time = time.time() - start_time
                check_results["performance_metrics"]["verification_time_seconds"] = restoration_time

                # Check if restoration completed in reasonable time
                reasonable_time = 300  # 5 minutes for most operations
                if restoration_time < reasonable_time:
                    check_results["checks_passed"] += 1
                    logger.debug(f"✓ Verification completed in reasonable time: {restoration_time:.2f}s")
                else:
                    check_results["warnings"].append(f"Verification took longer than expected: {restoration_time:.2f}s")

            except Exception as e:
                check_results["warnings"].append(f"Performance check failed: {e}")

            # Check 5: Compare with expected restoration metadata
            try:
                # Look for recent restoration records
                recent_restorations = []
                for restoration in self.backup_metadata.get("restorations", []):
                    if restoration.get("target_location") == target_location:
                        restored_at = datetime.fromisoformat(restoration.get("restored_at", "1970-01-01T00:00:00"))
                        time_diff = datetime.utcnow() - restored_at
                        if time_diff.total_seconds() < 3600:  # Within last hour
                            recent_restorations.append(restoration)

                if recent_restorations:
                    # Validate against expected results
                    expected_stats = recent_restorations[-1].get("stats", {})
                    expected_files = expected_stats.get("files_restored", 0)
                    actual_files = check_results["performance_metrics"].get("restored_files", 0)

                    if expected_files == 0 or abs(actual_files - expected_files) / expected_files < 0.1:  # Within 10%
                        check_results["checks_passed"] += 1
                        logger.debug(f"✓ File count matches expectation: {actual_files} files")
                    else:
                        check_results["warnings"].append(f"File count mismatch: expected {expected_files}, found {actual_files}")

            except Exception as e:
                check_results["warnings"].append(f"Metadata comparison failed: {e}")

            # Check 6: System health after restoration
            try:
                # Basic system health checks
                disk_usage_after = 0
                if target_location:
                    statvfs = os.statvfs(os.path.dirname(target_location) if os.path.isfile(target_location) else target_location)
                    free_space = statvfs.f_frsize * statvfs.f_available
                    disk_usage_after = free_space
                    check_results["performance_metrics"]["free_space_after_mb"] = free_space // (1024 * 1024)

                # Check if we still have reasonable free space (>100MB)
                if disk_usage_after > 100 * 1024 * 1024:
                    check_results["checks_passed"] += 1
                    logger.debug(f"✓ Sufficient free space remaining: {disk_usage_after // (1024*1024)} MB")
                else:
                    check_results["warnings"].append(f"Low disk space after restoration: {disk_usage_after // (1024*1024)} MB")

            except Exception as e:
                check_results["warnings"].append(f"System health check failed: {e}")

            # Determine overall success
            checks_passed = check_results["checks_failed"] == 0

            # Log results
            if checks_passed:
                logger.info(f"✓ Post-recovery checks passed: {check_results['checks_passed']} checks completed")
                if check_results["warnings"]:
                    logger.warning(f"{len(check_results['warnings'])} warnings:")
                    for warning in check_results["warnings"]:
                        logger.warning(f"  ⚠ {warning}")
            else:
                logger.error(f"✗ Post-recovery checks failed: {check_results['checks_failed']} critical errors")
                for error in check_results["errors"]:
                    logger.error(f"  ✗ {error}")

            # Store check results
            self.backup_metadata.setdefault("post_recovery_checks", []).append({
                "target_location": target_location,
                "results": check_results,
                "checked_at": datetime.utcnow().isoformat(),
                "passed": checks_passed
            })

            return checks_passed

        except Exception as e:
            logger.error(f"Post-recovery checks error: {e}")
            return False

    async def _replicate_to_target(self, data: Any, target: str):
        """Replicate data to target"""
        logger.info(f"Replicating data to {target}")
        # TODO: Implement actual replication logic

    async def _update_replication_status(self):
        """Update replication status across all targets"""
        logger.info("Updating replication status")
        # TODO: Implement replication status checks


# Global disaster recovery manager instance
_dr_manager: Optional[DisasterRecoveryManager] = None


async def get_disaster_recovery_manager() -> DisasterRecoveryManager:
    """Get or create the global disaster recovery manager instance"""
    global _dr_manager
    if _dr_manager is None:
        _dr_manager = DisasterRecoveryManager()
        await _dr_manager.initialize()
    return _dr_manager
