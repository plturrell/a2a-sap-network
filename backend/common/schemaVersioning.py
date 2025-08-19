"""
Schema versioning system for A2A Catalog Manager Agent.
Provides comprehensive schema evolution, migration, and compatibility management.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import semver

from config.agentConfig import config
from common.errorHandling import with_circuit_breaker, with_retry

logger = logging.getLogger(__name__)


class SchemaFormat(Enum):
    """Supported schema formats."""
    JSON_SCHEMA = "json_schema"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    OAS = "openapi"
    GRAPHQL = "graphql"


class VersioningStrategy(Enum):
    """Schema versioning strategies."""
    SEMANTIC = "semantic"      # Major.Minor.Patch
    SEQUENTIAL = "sequential"  # v1, v2, v3
    TIMESTAMP = "timestamp"    # Based on creation time
    HASH = "hash"             # Content-based hash


class CompatibilityLevel(Enum):
    """Schema compatibility levels."""
    NONE = "none"                    # No compatibility checking
    BACKWARD = "backward"            # New schema can read old data
    FORWARD = "forward"              # Old schema can read new data
    FULL = "full"                   # Both backward and forward compatible
    TRANSITIVE = "transitive"       # Compatibility across all versions


@dataclass
class SchemaVersion:
    """Schema version metadata."""
    schema_id: str
    version: str
    format: SchemaFormat
    schema_content: Dict[str, Any]
    created_at: datetime
    created_by: str
    compatibility_level: CompatibilityLevel
    parent_versions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    checksum: Optional[str] = None


@dataclass
class CompatibilityCheck:
    """Schema compatibility check result."""
    is_compatible: bool
    compatibility_type: CompatibilityLevel
    source_version: str
    target_version: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    migration_required: bool = False


@dataclass
class MigrationRule:
    """Schema migration rule."""
    rule_id: str
    from_version: str
    to_version: str
    transformation: Dict[str, Any]
    is_reversible: bool = False
    validation_rules: List[str] = field(default_factory=list)


class SchemaVersioningManager:
    """
    Comprehensive schema versioning and evolution management system.
    """
    
    def __init__(
        self,
        storage_path: str,
        default_strategy: VersioningStrategy = VersioningStrategy.SEMANTIC,
        default_compatibility: CompatibilityLevel = CompatibilityLevel.BACKWARD
    ):
        """
        Initialize schema versioning manager.
        
        Args:
            storage_path: Path for schema storage
            default_strategy: Default versioning strategy
            default_compatibility: Default compatibility level
        """
        self.storage_path = Path(storage_path)
        self.default_strategy = default_strategy
        self.default_compatibility = default_compatibility
        
        # Schema registry
        self.schemas = {}  # schema_id -> Dict[version -> SchemaVersion]
        self.migration_rules = {}  # rule_id -> MigrationRule
        
        # Version tracking
        self.latest_versions = {}  # schema_id -> latest_version
        self.version_history = {}  # schema_id -> List[version]
        
        # Cache for compatibility checks
        self._compatibility_cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Initialize storage
        self._initialize_storage()
        self._load_existing_schemas()
    
    def _initialize_storage(self):
        """Initialize schema storage structure."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "schemas").mkdir(exist_ok=True)
        (self.storage_path / "migrations").mkdir(exist_ok=True)
        (self.storage_path / "compatibility").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)
    
    def _load_existing_schemas(self):
        """Load existing schemas from storage."""
        schemas_dir = self.storage_path / "schemas"
        
        if not schemas_dir.exists():
            return
        
        try:
            for schema_file in schemas_dir.glob("*.json"):
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                
                # Reconstruct schema versions
                schema_id = schema_data.get('schema_id')
                if schema_id:
                    self.schemas[schema_id] = {}
                    self.version_history[schema_id] = []
                    
                    for version_data in schema_data.get('versions', []):
                        version = SchemaVersion(
                            schema_id=version_data['schema_id'],
                            version=version_data['version'],
                            format=SchemaFormat(version_data['format']),
                            schema_content=version_data['schema_content'],
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            created_by=version_data['created_by'],
                            compatibility_level=CompatibilityLevel(version_data['compatibility_level']),
                            parent_versions=version_data.get('parent_versions', []),
                            tags=version_data.get('tags', []),
                            metadata=version_data.get('metadata', {}),
                            is_active=version_data.get('is_active', True),
                            checksum=version_data.get('checksum')
                        )
                        
                        self.schemas[schema_id][version.version] = version
                        self.version_history[schema_id].append(version.version)
                    
                    # Determine latest version
                    if self.version_history[schema_id]:
                        self.latest_versions[schema_id] = self._get_latest_version(schema_id)
            
            logger.info(f"Loaded {len(self.schemas)} schema definitions from storage")
            
        except Exception as e:
            logger.error(f"Failed to load existing schemas: {e}")
    
    def _save_schema(self, schema_id: str):
        """Save schema to storage."""
        if schema_id not in self.schemas:
            return
        
        try:
            schema_file = self.storage_path / "schemas" / f"{schema_id}.json"
            
            schema_data = {
                'schema_id': schema_id,
                'versions': []
            }
            
            for version, schema_version in self.schemas[schema_id].items():
                version_data = {
                    'schema_id': schema_version.schema_id,
                    'version': schema_version.version,
                    'format': schema_version.format.value,
                    'schema_content': schema_version.schema_content,
                    'created_at': schema_version.created_at.isoformat(),
                    'created_by': schema_version.created_by,
                    'compatibility_level': schema_version.compatibility_level.value,
                    'parent_versions': schema_version.parent_versions,
                    'tags': schema_version.tags,
                    'metadata': schema_version.metadata,
                    'is_active': schema_version.is_active,
                    'checksum': schema_version.checksum
                }
                schema_data['versions'].append(version_data)
            
            with open(schema_file, 'w') as f:
                json.dump(schema_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save schema {schema_id}: {e}")
    
    async def register_schema(
        self,
        schema_id: str,
        schema_content: Dict[str, Any],
        format: SchemaFormat,
        version: Optional[str] = None,
        created_by: str = "system",
        compatibility_level: Optional[CompatibilityLevel] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SchemaVersion:
        """
        Register a new schema version.
        
        Args:
            schema_id: Unique schema identifier
            schema_content: Schema definition
            format: Schema format
            version: Explicit version (auto-generated if None)
            created_by: Creator identifier
            compatibility_level: Compatibility level
            tags: Schema tags
            metadata: Additional metadata
            
        Returns:
            Created schema version
        """
        # Initialize schema if new
        if schema_id not in self.schemas:
            self.schemas[schema_id] = {}
            self.version_history[schema_id] = []
        
        # Generate version if not provided
        if version is None:
            version = self._generate_next_version(schema_id)
        
        # Check if version already exists
        if version in self.schemas[schema_id]:
            raise ValueError(f"Schema version {schema_id}:{version} already exists")
        
        # Calculate checksum
        checksum = self._calculate_checksum(schema_content)
        
        # Determine parent versions
        parent_versions = []
        if self.version_history[schema_id]:
            parent_versions = [self.latest_versions.get(schema_id)]
        
        # Create schema version
        schema_version = SchemaVersion(
            schema_id=schema_id,
            version=version,
            format=format,
            schema_content=schema_content,
            created_at=datetime.now(),
            created_by=created_by,
            compatibility_level=compatibility_level or self.default_compatibility,
            parent_versions=parent_versions,
            tags=tags or [],
            metadata=metadata or {},
            checksum=checksum
        )
        
        # Validate compatibility if there are existing versions
        if self.version_history[schema_id]:
            latest_version = self.latest_versions[schema_id]
            compatibility_check = await self.check_compatibility(
                schema_id,
                latest_version,
                version,
                schema_version.schema_content
            )
            
            if not compatibility_check.is_compatible:
                logger.warning(
                    f"Schema {schema_id}:{version} is not compatible with {latest_version}. "
                    f"Issues: {compatibility_check.issues}"
                )
        
        # Store schema version
        self.schemas[schema_id][version] = schema_version
        self.version_history[schema_id].append(version)
        self.latest_versions[schema_id] = version
        
        # Save to storage
        self._save_schema(schema_id)
        
        logger.info(f"Registered schema version: {schema_id}:{version}")
        
        return schema_version
    
    def _generate_next_version(self, schema_id: str) -> str:
        """Generate next version based on strategy."""
        if self.default_strategy == VersioningStrategy.SEMANTIC:
            if schema_id not in self.latest_versions:
                return "1.0.0"
            else:
                latest = self.latest_versions[schema_id]
                return semver.bump_minor(latest)
        
        elif self.default_strategy == VersioningStrategy.SEQUENTIAL:
            count = len(self.version_history.get(schema_id, []))
            return f"v{count + 1}"
        
        elif self.default_strategy == VersioningStrategy.TIMESTAMP:
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        
        elif self.default_strategy == VersioningStrategy.HASH:
            import uuid
            return str(uuid.uuid4())[:8]
        
        else:
            return "1.0.0"
    
    def _calculate_checksum(self, schema_content: Dict[str, Any]) -> str:
        """Calculate schema content checksum."""
        content_str = json.dumps(schema_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _get_latest_version(self, schema_id: str) -> Optional[str]:
        """Get latest version for schema."""
        if schema_id not in self.version_history:
            return None
        
        versions = self.version_history[schema_id]
        if not versions:
            return None
        
        if self.default_strategy == VersioningStrategy.SEMANTIC:
            # Sort semantic versions
            try:
                semantic_versions = [v for v in versions if self._is_semantic_version(v)]
                if semantic_versions:
                    return max(semantic_versions, key=lambda v: semver.VersionInfo.parse(v))
            except:
                pass
        
        # Fall back to latest by creation time
        latest_version = None
        latest_time = None
        
        for version in versions:
            schema_version = self.schemas[schema_id][version]
            if latest_time is None or schema_version.created_at > latest_time:
                latest_time = schema_version.created_at
                latest_version = version
        
        return latest_version
    
    def _is_semantic_version(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        try:
            semver.VersionInfo.parse(version)
            return True
        except:
            return False
    
    async def check_compatibility(
        self,
        schema_id: str,
        from_version: str,
        to_version: str,
        to_schema_content: Optional[Dict[str, Any]] = None
    ) -> CompatibilityCheck:
        """
        Check compatibility between schema versions.
        
        Args:
            schema_id: Schema identifier
            from_version: Source version
            to_version: Target version
            to_schema_content: Target schema content (if not stored)
            
        Returns:
            Compatibility check result
        """
        # Check cache first
        cache_key = f"{schema_id}:{from_version}:{to_version}"
        if cache_key in self._compatibility_cache:
            cached_result, cached_time = self._compatibility_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_ttl):
                return cached_result
        
        # Get schema versions
        if schema_id not in self.schemas:
            return CompatibilityCheck(
                is_compatible=False,
                compatibility_type=CompatibilityLevel.NONE,
                source_version=from_version,
                target_version=to_version,
                issues=["Schema not found"]
            )
        
        from_schema = self.schemas[schema_id].get(from_version)
        if not from_schema:
            return CompatibilityCheck(
                is_compatible=False,
                compatibility_type=CompatibilityLevel.NONE,
                source_version=from_version,
                target_version=to_version,
                issues=[f"Source version {from_version} not found"]
            )
        
        # Get target schema
        if to_schema_content:
            to_schema_content_dict = to_schema_content
        else:
            to_schema = self.schemas[schema_id].get(to_version)
            if not to_schema:
                return CompatibilityCheck(
                    is_compatible=False,
                    compatibility_type=CompatibilityLevel.NONE,
                    source_version=from_version,
                    target_version=to_version,
                    issues=[f"Target version {to_version} not found"]
                )
            to_schema_content_dict = to_schema.schema_content
        
        # Perform compatibility check based on format
        compatibility_check = await self._perform_compatibility_check(
            from_schema.schema_content,
            to_schema_content_dict,
            from_schema.format,
            from_schema.compatibility_level
        )
        
        # Cache result
        self._compatibility_cache[cache_key] = (compatibility_check, datetime.now())
        
        return compatibility_check
    
    async def _perform_compatibility_check(
        self,
        from_schema: Dict[str, Any],
        to_schema: Dict[str, Any],
        format: SchemaFormat,
        compatibility_level: CompatibilityLevel
    ) -> CompatibilityCheck:
        """Perform format-specific compatibility check."""
        issues = []
        warnings = []
        migration_required = False
        
        if format == SchemaFormat.JSON_SCHEMA:
            issues, warnings, migration_required = await self._check_json_schema_compatibility(
                from_schema, to_schema, compatibility_level
            )
        elif format == SchemaFormat.AVRO:
            issues, warnings, migration_required = await self._check_avro_compatibility(
                from_schema, to_schema, compatibility_level
            )
        else:
            # Generic compatibility check
            issues, warnings, migration_required = await self._check_generic_compatibility(
                from_schema, to_schema, compatibility_level
            )
        
        is_compatible = len(issues) == 0
        
        return CompatibilityCheck(
            is_compatible=is_compatible,
            compatibility_type=compatibility_level,
            source_version="",
            target_version="",
            issues=issues,
            warnings=warnings,
            migration_required=migration_required
        )
    
    async def _check_json_schema_compatibility(
        self,
        from_schema: Dict[str, Any],
        to_schema: Dict[str, Any],
        compatibility_level: CompatibilityLevel
    ) -> Tuple[List[str], List[str], bool]:
        """Check JSON Schema compatibility."""
        issues = []
        warnings = []
        migration_required = False
        
        # Check required fields
        from_required = set(from_schema.get('required', []))
        to_required = set(to_schema.get('required', []))
        
        if compatibility_level in [CompatibilityLevel.BACKWARD, CompatibilityLevel.FULL]:
            # Backward compatibility: new schema can't add required fields
            new_required = to_required - from_required
            if new_required:
                issues.append(f"New required fields added: {new_required}")
                migration_required = True
        
        if compatibility_level in [CompatibilityLevel.FORWARD, CompatibilityLevel.FULL]:
            # Forward compatibility: old schema can't have required fields removed
            removed_required = from_required - to_required
            if removed_required:
                warnings.append(f"Required fields removed: {removed_required}")
        
        # Check property types
        from_props = from_schema.get('properties', {})
        to_props = to_schema.get('properties', {})
        
        for prop_name in from_props:
            if prop_name in to_props:
                from_type = from_props[prop_name].get('type')
                to_type = to_props[prop_name].get('type')
                
                if from_type != to_type:
                    issues.append(f"Property '{prop_name}' type changed: {from_type} -> {to_type}")
                    migration_required = True
        
        # Check for removed properties
        removed_props = set(from_props.keys()) - set(to_props.keys())
        if removed_props:
            warnings.append(f"Properties removed: {removed_props}")
        
        return issues, warnings, migration_required
    
    async def _check_avro_compatibility(
        self,
        from_schema: Dict[str, Any],
        to_schema: Dict[str, Any],
        compatibility_level: CompatibilityLevel
    ) -> Tuple[List[str], List[str], bool]:
        """Check Avro schema compatibility."""
        issues = []
        warnings = []
        migration_required = False
        
        # Avro-specific compatibility rules
        from_fields = {f['name']: f for f in from_schema.get('fields', [])}
        to_fields = {f['name']: f for f in to_schema.get('fields', [])}
        
        # Check field additions/removals
        if compatibility_level in [CompatibilityLevel.BACKWARD, CompatibilityLevel.FULL]:
            for field_name, field_def in to_fields.items():
                if field_name not in from_fields:
                    if 'default' not in field_def:
                        issues.append(f"New field '{field_name}' without default value")
                        migration_required = True
        
        # Check type changes
        for field_name in from_fields:
            if field_name in to_fields:
                from_type = from_fields[field_name]['type']
                to_type = to_fields[field_name]['type']
                
                if not self._is_avro_type_compatible(from_type, to_type):
                    issues.append(f"Incompatible type change for '{field_name}': {from_type} -> {to_type}")
                    migration_required = True
        
        return issues, warnings, migration_required
    
    def _is_avro_type_compatible(self, from_type: Any, to_type: Any) -> bool:
        """Check if Avro types are compatible."""
        # Simplified Avro type compatibility rules
        if from_type == to_type:
            return True
        
        # Some basic promotion rules
        promotions = {
            'int': ['long', 'float', 'double'],
            'long': ['float', 'double'],
            'float': ['double']
        }
        
        if isinstance(from_type, str) and isinstance(to_type, str):
            return to_type in promotions.get(from_type, [])
        
        return False
    
    async def _check_generic_compatibility(
        self,
        from_schema: Dict[str, Any],
        to_schema: Dict[str, Any],
        compatibility_level: CompatibilityLevel
    ) -> Tuple[List[str], List[str], bool]:
        """Generic compatibility check."""
        issues = []
        warnings = []
        migration_required = False
        
        # Basic structure comparison
        if json.dumps(from_schema, sort_keys=True) != json.dumps(to_schema, sort_keys=True):
            warnings.append("Schema structure has changed")
        
        return issues, warnings, migration_required
    
    def get_schema_versions(self, schema_id: str) -> List[str]:
        """Get all versions for a schema."""
        return self.version_history.get(schema_id, [])
    
    def get_schema_version(self, schema_id: str, version: str) -> Optional[SchemaVersion]:
        """Get specific schema version."""
        return self.schemas.get(schema_id, {}).get(version)
    
    def get_latest_schema(self, schema_id: str) -> Optional[SchemaVersion]:
        """Get latest schema version."""
        latest_version = self.latest_versions.get(schema_id)
        if latest_version:
            return self.get_schema_version(schema_id, latest_version)
        return None
    
    def deprecate_schema_version(self, schema_id: str, version: str) -> bool:
        """Deprecate a schema version."""
        schema_version = self.get_schema_version(schema_id, version)
        if schema_version:
            schema_version.is_active = False
            self._save_schema(schema_id)
            logger.info(f"Deprecated schema version: {schema_id}:{version}")
            return True
        return False
    
    def get_schema_evolution_history(self, schema_id: str) -> List[Dict[str, Any]]:
        """Get evolution history for a schema."""
        if schema_id not in self.schemas:
            return []
        
        history = []
        for version in self.version_history[schema_id]:
            schema_version = self.schemas[schema_id][version]
            history.append({
                'version': version,
                'created_at': schema_version.created_at.isoformat(),
                'created_by': schema_version.created_by,
                'compatibility_level': schema_version.compatibility_level.value,
                'is_active': schema_version.is_active,
                'tags': schema_version.tags,
                'checksum': schema_version.checksum
            })
        
        return sorted(history, key=lambda x: x['created_at'])
    
    def get_compatibility_matrix(self, schema_id: str) -> Dict[str, Dict[str, bool]]:
        """Get compatibility matrix for all versions of a schema."""
        versions = self.get_schema_versions(schema_id)
        matrix = {}
        
        for from_version in versions:
            matrix[from_version] = {}
            for to_version in versions:
                if from_version == to_version:
                    matrix[from_version][to_version] = True
                else:
                    # This would require async, so simplified for now
                    matrix[from_version][to_version] = False
        
        return matrix
    
    def get_versioning_stats(self) -> Dict[str, Any]:
        """Get versioning system statistics."""
        total_schemas = len(self.schemas)
        total_versions = sum(len(versions) for versions in self.schemas.values())
        
        format_counts = {}
        compatibility_counts = {}
        
        for schema_versions in self.schemas.values():
            for version in schema_versions.values():
                format_counts[version.format.value] = format_counts.get(version.format.value, 0) + 1
                compat_level = version.compatibility_level.value
                compatibility_counts[compat_level] = compatibility_counts.get(compat_level, 0) + 1
        
        return {
            'total_schemas': total_schemas,
            'total_versions': total_versions,
            'average_versions_per_schema': total_versions / max(total_schemas, 1),
            'format_distribution': format_counts,
            'compatibility_distribution': compatibility_counts,
            'cache_size': len(self._compatibility_cache),
            'storage_path': str(self.storage_path)
        }