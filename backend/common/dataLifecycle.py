"""
Data lifecycle management for A2A agents.
Provides comprehensive data management, retention, and cleanup capabilities.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import gzip
import shutil

logger = logging.getLogger(__name__)


class DataStage(Enum):
    """Data lifecycle stages."""
    CREATED = "created"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    PURGED = "purged"


class DataType(Enum):
    """Types of data managed."""
    USER_DATA = "user_data"
    SYSTEM_DATA = "system_data"
    CACHE_DATA = "cache_data"
    LOG_DATA = "log_data"
    TEMP_DATA = "temp_data"
    BACKUP_DATA = "backup_data"


class RetentionPolicy(Enum):
    """Data retention policies."""
    SHORT_TERM = "short_term"      # 7 days
    MEDIUM_TERM = "medium_term"    # 30 days
    LONG_TERM = "long_term"        # 365 days
    PERMANENT = "permanent"        # Never delete
    CUSTOM = "custom"              # Custom retention period


@dataclass
class DataRecord:
    """Data record with lifecycle metadata."""
    data_id: str
    data_type: DataType
    stage: DataStage
    created_at: datetime
    last_accessed: datetime
    size_bytes: int
    location: str
    retention_policy: RetentionPolicy
    retention_days: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    checksum: Optional[str] = None


@dataclass
class LifecycleRule:
    """Data lifecycle rule definition."""
    rule_id: str
    name: str
    data_type: DataType
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool = True
    priority: int = 0


class DataLifecycleManager:
    """
    Comprehensive data lifecycle management system.
    """
    
    def __init__(
        self,
        base_storage_path: str,
        archive_storage_path: Optional[str] = None,
        enable_compression: bool = True,
        enable_encryption: bool = False
    ):
        """
        Initialize data lifecycle manager.
        
        Args:
            base_storage_path: Base path for active data storage
            archive_storage_path: Path for archived data (defaults to base_path/archive)
            enable_compression: Enable data compression for archives
            enable_encryption: Enable data encryption (requires additional setup)
        """
        self.base_storage_path = Path(base_storage_path)
        self.archive_storage_path = Path(
            archive_storage_path or self.base_storage_path / "archive"
        )
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Data registry
        self.data_registry = {}  # data_id -> DataRecord
        self.lifecycle_rules = {}  # rule_id -> LifecycleRule
        
        # Retention policies (in days)
        self.retention_policies = {
            RetentionPolicy.SHORT_TERM: 7,
            RetentionPolicy.MEDIUM_TERM: 30,
            RetentionPolicy.LONG_TERM: 365,
            RetentionPolicy.PERMANENT: None
        }
        
        # Statistics
        self.lifecycle_stats = {
            'total_records': 0,
            'total_size_bytes': 0,
            'records_by_stage': {stage.value: 0 for stage in DataStage},
            'records_by_type': {dtype.value: 0 for dtype in DataType},
            'last_cleanup': None,
            'last_archive': None
        }
        
        # Ensure directories exist
        self._initialize_storage()
        
        # Load existing registry
        self._load_registry()
        
        # Setup default lifecycle rules
        self._setup_default_rules()
    
    def _initialize_storage(self):
        """Initialize storage directories."""
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        self.archive_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        for data_type in DataType:
            (self.base_storage_path / data_type.value).mkdir(exist_ok=True)
            (self.archive_storage_path / data_type.value).mkdir(exist_ok=True)
    
    def _load_registry(self):
        """Load data registry from storage."""
        registry_file = self.base_storage_path / "data_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for data_id, record_data in registry_data.items():
                    # Convert datetime strings back to datetime objects
                    record_data['created_at'] = datetime.fromisoformat(record_data['created_at'])
                    record_data['last_accessed'] = datetime.fromisoformat(record_data['last_accessed'])
                    record_data['data_type'] = DataType(record_data['data_type'])
                    record_data['stage'] = DataStage(record_data['stage'])
                    record_data['retention_policy'] = RetentionPolicy(record_data['retention_policy'])
                    
                    self.data_registry[data_id] = DataRecord(**record_data)
                
                logger.info(f"Loaded {len(self.data_registry)} data records from registry")
                
            except Exception as e:
                logger.error(f"Failed to load data registry: {e}")
    
    def _save_registry(self):
        """Save data registry to storage."""
        registry_file = self.base_storage_path / "data_registry.json"
        
        try:
            registry_data = {}
            for data_id, record in self.data_registry.items():
                registry_data[data_id] = {
                    'data_id': record.data_id,
                    'data_type': record.data_type.value,
                    'stage': record.stage.value,
                    'created_at': record.created_at.isoformat(),
                    'last_accessed': record.last_accessed.isoformat(),
                    'size_bytes': record.size_bytes,
                    'location': record.location,
                    'retention_policy': record.retention_policy.value,
                    'retention_days': record.retention_days,
                    'metadata': record.metadata,
                    'tags': record.tags,
                    'access_count': record.access_count,
                    'checksum': record.checksum
                }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save data registry: {e}")
    
    def _setup_default_rules(self):
        """Setup default lifecycle rules."""
        default_rules = [
            LifecycleRule(
                rule_id="temp_data_cleanup",
                name="Cleanup temporary data",
                data_type=DataType.TEMP_DATA,
                conditions={"age_days": 1},
                actions=["purge"],
                priority=1
            ),
            LifecycleRule(
                rule_id="cache_data_archive",
                name="Archive old cache data",
                data_type=DataType.CACHE_DATA,
                conditions={"age_days": 7, "not_accessed_days": 3},
                actions=["archive"],
                priority=2
            ),
            LifecycleRule(
                rule_id="log_data_archive",
                name="Archive log data",
                data_type=DataType.LOG_DATA,
                conditions={"age_days": 30},
                actions=["archive"],
                priority=3
            ),
            LifecycleRule(
                rule_id="user_data_long_term",
                name="Long-term retention for user data",
                data_type=DataType.USER_DATA,
                conditions={"age_days": 365},
                actions=["archive"],
                priority=4
            )
        ]
        
        for rule in default_rules:
            self.lifecycle_rules[rule.rule_id] = rule
    
    async def register_data(
        self,
        data_id: str,
        data_type: DataType,
        file_path: str,
        retention_policy: RetentionPolicy = RetentionPolicy.MEDIUM_TERM,
        retention_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> DataRecord:
        """
        Register new data in the lifecycle management system.
        
        Args:
            data_id: Unique identifier for the data
            data_type: Type of data
            file_path: Path to the data file
            retention_policy: Retention policy
            retention_days: Custom retention period (for CUSTOM policy)
            metadata: Additional metadata
            tags: Data tags
            
        Returns:
            Created data record
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Calculate file size and checksum
        size_bytes = file_path.stat().st_size
        checksum = await self._calculate_checksum(file_path)
        
        # Create data record
        record = DataRecord(
            data_id=data_id,
            data_type=data_type,
            stage=DataStage.CREATED,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=size_bytes,
            location=str(file_path),
            retention_policy=retention_policy,
            retention_days=retention_days,
            metadata=metadata or {},
            tags=tags or [],
            checksum=checksum
        )
        
        # Register in system
        self.data_registry[data_id] = record
        self._update_statistics()
        self._save_registry()
        
        logger.info(f"Registered data: {data_id} ({data_type.value}, {size_bytes} bytes)")
        
        return record
    
    async def access_data(self, data_id: str) -> Optional[str]:
        """
        Access data and update access metrics.
        
        Args:
            data_id: Data identifier
            
        Returns:
            Path to data file if exists
        """
        if data_id not in self.data_registry:
            return None
        
        record = self.data_registry[data_id]
        
        # Update access metrics
        record.last_accessed = datetime.now()
        record.access_count += 1
        
        # Transition to active if not already
        if record.stage == DataStage.CREATED:
            record.stage = DataStage.ACTIVE
        
        self._save_registry()
        
        # Return path if file exists
        if Path(record.location).exists():
            return record.location
        else:
            logger.warning(f"Data file missing: {record.location}")
            return None
    
    async def delete_data(self, data_id: str, force: bool = False) -> bool:
        """
        Delete data from the system.
        
        Args:
            data_id: Data identifier
            force: Force deletion even if retention policy prevents it
            
        Returns:
            True if deletion successful
        """
        if data_id not in self.data_registry:
            return False
        
        record = self.data_registry[data_id]
        
        # Check if deletion is allowed
        if not force and not self._can_delete(record):
            logger.warning(f"Cannot delete {data_id}: retention policy prevents deletion")
            return False
        
        try:
            # Delete physical file
            file_path = Path(record.location)
            if file_path.exists():
                file_path.unlink()
            
            # Update record
            record.stage = DataStage.PURGED
            
            # Remove from registry
            del self.data_registry[data_id]
            
            self._update_statistics()
            self._save_registry()
            
            logger.info(f"Deleted data: {data_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete data {data_id}: {e}")
            return False
    
    async def archive_data(self, data_id: str) -> bool:
        """
        Archive data to long-term storage.
        
        Args:
            data_id: Data identifier
            
        Returns:
            True if archival successful
        """
        if data_id not in self.data_registry:
            return False
        
        record = self.data_registry[data_id]
        
        if record.stage == DataStage.ARCHIVED:
            return True  # Already archived
        
        try:
            source_path = Path(record.location)
            if not source_path.exists():
                logger.error(f"Source file not found for archival: {source_path}")
                return False
            
            # Create archive path
            archive_path = self.archive_storage_path / record.data_type.value / f"{data_id}"
            
            if self.enable_compression:
                archive_path = archive_path.with_suffix('.gz')
                
                # Compress and copy
                with open(source_path, 'rb') as f_in:
                    with gzip.open(archive_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Simple copy
                shutil.copy2(source_path, archive_path)
            
            # Delete original file
            source_path.unlink()
            
            # Update record
            record.location = str(archive_path)
            record.stage = DataStage.ARCHIVED
            
            self._save_registry()
            
            logger.info(f"Archived data: {data_id} -> {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive data {data_id}: {e}")
            return False
    
    async def run_lifecycle_cleanup(self) -> Dict[str, Any]:
        """
        Run lifecycle cleanup based on defined rules.
        
        Returns:
            Cleanup summary
        """
        logger.info("Starting data lifecycle cleanup")
        start_time = time.time()
        
        cleanup_summary = {
            'start_time': datetime.now().isoformat(),
            'records_processed': 0,
            'records_archived': 0,
            'records_purged': 0,
            'bytes_cleaned': 0,
            'errors': []
        }
        
        # Process each lifecycle rule
        for rule in sorted(self.lifecycle_rules.values(), key=lambda r: r.priority):
            if not rule.enabled:
                continue
            
            logger.info(f"Processing rule: {rule.name}")
            
            # Find matching records
            matching_records = self._find_matching_records(rule)
            
            for record in matching_records:
                cleanup_summary['records_processed'] += 1
                
                try:
                    for action in rule.actions:
                        if action == "archive":
                            if await self.archive_data(record.data_id):
                                cleanup_summary['records_archived'] += 1
                                cleanup_summary['bytes_cleaned'] += record.size_bytes
                        elif action == "purge":
                            if await self.delete_data(record.data_id, force=True):
                                cleanup_summary['records_purged'] += 1
                                cleanup_summary['bytes_cleaned'] += record.size_bytes
                                
                except Exception as e:
                    error_msg = f"Error processing {record.data_id}: {e}"
                    cleanup_summary['errors'].append(error_msg)
                    logger.error(error_msg)
        
        # Update statistics
        self.lifecycle_stats['last_cleanup'] = datetime.now()
        self._update_statistics()
        
        cleanup_summary['duration'] = time.time() - start_time
        cleanup_summary['end_time'] = datetime.now().isoformat()
        
        logger.info(
            f"Lifecycle cleanup completed: "
            f"{cleanup_summary['records_archived']} archived, "
            f"{cleanup_summary['records_purged']} purged, "
            f"{cleanup_summary['bytes_cleaned']} bytes cleaned"
        )
        
        return cleanup_summary
    
    def _find_matching_records(self, rule: LifecycleRule) -> List[DataRecord]:
        """Find records matching lifecycle rule conditions."""
        matching_records = []
        
        for record in self.data_registry.values():
            if record.data_type != rule.data_type:
                continue
            
            if record.stage in [DataStage.ARCHIVED, DataStage.PURGED]:
                continue
            
            # Check conditions
            if self._record_matches_conditions(record, rule.conditions):
                matching_records.append(record)
        
        return matching_records
    
    def _record_matches_conditions(
        self,
        record: DataRecord,
        conditions: Dict[str, Any]
    ) -> bool:
        """Check if record matches rule conditions."""
        now = datetime.now()
        
        # Age conditions
        if 'age_days' in conditions:
            age_days = (now - record.created_at).days
            if age_days < conditions['age_days']:
                return False
        
        # Last accessed conditions
        if 'not_accessed_days' in conditions:
            not_accessed_days = (now - record.last_accessed).days
            if not_accessed_days < conditions['not_accessed_days']:
                return False
        
        # Size conditions
        if 'min_size_bytes' in conditions:
            if record.size_bytes < conditions['min_size_bytes']:
                return False
        
        if 'max_size_bytes' in conditions:
            if record.size_bytes > conditions['max_size_bytes']:
                return False
        
        # Access count conditions
        if 'max_access_count' in conditions:
            if record.access_count > conditions['max_access_count']:
                return False
        
        # Tag conditions
        if 'required_tags' in conditions:
            if not all(tag in record.tags for tag in conditions['required_tags']):
                return False
        
        return True
    
    def _can_delete(self, record: DataRecord) -> bool:
        """Check if record can be deleted based on retention policy."""
        if record.retention_policy == RetentionPolicy.PERMANENT:
            return False
        
        now = datetime.now()
        retention_days = record.retention_days
        
        if retention_days is None:
            retention_days = self.retention_policies.get(record.retention_policy)
        
        if retention_days is None:
            return False  # Cannot determine retention period
        
        age_days = (now - record.created_at).days
        return age_days >= retention_days
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _update_statistics(self):
        """Update lifecycle statistics."""
        self.lifecycle_stats['total_records'] = len(self.data_registry)
        self.lifecycle_stats['total_size_bytes'] = sum(
            record.size_bytes for record in self.data_registry.values()
        )
        
        # Reset counters
        for stage in DataStage:
            self.lifecycle_stats['records_by_stage'][stage.value] = 0
        for dtype in DataType:
            self.lifecycle_stats['records_by_type'][dtype.value] = 0
        
        # Count by stage and type
        for record in self.data_registry.values():
            self.lifecycle_stats['records_by_stage'][record.stage.value] += 1
            self.lifecycle_stats['records_by_type'][record.data_type.value] += 1
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle management statistics."""
        self._update_statistics()
        return self.lifecycle_stats.copy()
    
    def add_lifecycle_rule(self, rule: LifecycleRule):
        """Add a new lifecycle rule."""
        self.lifecycle_rules[rule.rule_id] = rule
        logger.info(f"Added lifecycle rule: {rule.name}")
    
    def remove_lifecycle_rule(self, rule_id: str) -> bool:
        """Remove a lifecycle rule."""
        if rule_id in self.lifecycle_rules:
            del self.lifecycle_rules[rule_id]
            logger.info(f"Removed lifecycle rule: {rule_id}")
            return True
        return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all managed data."""
        return {
            'total_records': len(self.data_registry),
            'total_size_gb': sum(r.size_bytes for r in self.data_registry.values()) / (1024**3),
            'by_stage': {
                stage.value: len([r for r in self.data_registry.values() if r.stage == stage])
                for stage in DataStage
            },
            'by_type': {
                dtype.value: len([r for r in self.data_registry.values() if r.data_type == dtype])
                for dtype in DataType
            },
            'oldest_data': min(
                (r.created_at for r in self.data_registry.values()),
                default=None
            ),
            'most_accessed': max(
                (r.access_count for r in self.data_registry.values()),
                default=0
            )
        }