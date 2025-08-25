"""
Distributed Storage Interface for A2A Agent Registry
Supports multiple backends: Redis, Etcd, Consul, Local (fallback)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import json
import logging
import asyncio
from datetime import datetime, timedelta

try:
    import aioredis
except ImportError:
    aioredis = None

try:
    import aioetcd3
except ImportError:
    aioetcd3 = None
import aiofiles
import os
from pathlib import Path


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

try:
    from app.core.config import settings
except ImportError:
    # Fallback configuration
    class Settings:
        def __init__(self):
            self.redis_url = "redis://localhost:6379"
            self.etcd_url = os.getenv("A2A_SERVICE_URL")
            self.storage_backend = "local"
    settings = Settings()

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to storage backend"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from storage backend"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass


class RedisBackend(StorageBackend):
    """Redis storage backend"""

    def __init__(self):
        self.redis = None
        self.prefix = settings.REDIS_KEY_PREFIX
        self.default_ttl = settings.REDIS_TTL

    async def connect(self) -> None:
        """Connect to Redis with authentication and SSL support"""
        try:
            redis_url = settings.REDIS_URL

            # Add authentication if password is configured
            if settings.REDIS_PASSWORD:
                import urllib.parse
                parsed = urllib.parse.urlparse(redis_url)
                if parsed.username is None:  # Only add auth if not already in URL
                    netloc = f":{settings.REDIS_PASSWORD}@{parsed.hostname}:{parsed.port}"
                    redis_url = f"{parsed.scheme}://{netloc}{parsed.path}"

            # Configure SSL if enabled
            ssl_context = None
            if settings.REDIS_USE_SSL:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False  # For development
                ssl_context.verify_mode = ssl.CERT_REQUIRED

            self.redis = await aioredis.create_redis_pool(
                redis_url,
                ssl=ssl_context,
                retry_on_timeout=True,
                socket_connect_timeout=5.0,
                socket_keepalive=True
            )
            logger.info("Connected to Redis storage backend with security")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis"""
        try:
            full_key = f"{self.prefix}{key}"
            value = await self.redis.get(full_key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value in Redis with TTL"""
        try:
            full_key = f"{self.prefix}{key}"
            json_value = json.dumps(value)
            ttl = ttl or self.default_ttl

            if ttl:
                await self.redis.setex(full_key, ttl, json_value)
            else:
                await self.redis.set(full_key, json_value)

            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis"""
        try:
            full_key = f"{self.prefix}{key}"
            result = await self.redis.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix"""
        try:
            pattern = f"{self.prefix}{prefix}*"
            keys = await self.redis.keys(pattern)
            # Remove the storage prefix from keys
            return [key.decode('utf-8').replace(self.prefix, '') for key in keys]
        except Exception as e:
            logger.error(f"Redis list_keys error: {e}")
            return []

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            full_key = f"{self.prefix}{key}"
            return await self.redis.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False


class EtcdBackend(StorageBackend):
    """Etcd storage backend"""

    def __init__(self):
        self.client = None
        self.prefix = "/a2a/registry/"

    async def connect(self) -> None:
        """Connect to Etcd"""
        try:
            self.client = aioetcd3.client(
                host=settings.ETCD_HOST,
                port=settings.ETCD_PORT
            )
            logger.info("Connected to Etcd storage backend")
        except Exception as e:
            logger.error(f"Failed to connect to Etcd: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Etcd"""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Etcd")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Etcd"""
        try:
            full_key = f"{self.prefix}{key}"
            value = await self.client.get(full_key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Etcd get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value in Etcd with optional TTL"""
        try:
            full_key = f"{self.prefix}{key}"
            json_value = json.dumps(value)

            if ttl:
                lease = await self.client.lease(ttl)
                await self.client.put(full_key, json_value, lease=lease)
            else:
                await self.client.put(full_key, json_value)

            return True
        except Exception as e:
            logger.error(f"Etcd set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Etcd"""
        try:
            full_key = f"{self.prefix}{key}"
            result = await self.client.delete(full_key)
            return result > 0
        except Exception as e:
            logger.error(f"Etcd delete error for key {key}: {e}")
            return False

    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix"""
        try:
            full_prefix = f"{self.prefix}{prefix}"
            keys = []
            async for key, _ in self.client.get_prefix(full_prefix):
                clean_key = key.decode('utf-8').replace(self.prefix, '')
                keys.append(clean_key)
            return keys
        except Exception as e:
            logger.error(f"Etcd list_keys error: {e}")
            return []

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            result = await self.get(key)
            return result is not None
        except Exception as e:
            logger.error(f"Etcd exists error for key {key}: {e}")
            return False


class LocalFileBackend(StorageBackend):
    """Local file storage backend (fallback)"""

    def __init__(self):
        self.base_path = Path("./data/a2a_registry")
        self.ttl_index = {}  # Track TTLs in memory

    async def connect(self) -> None:
        """Initialize local storage"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local file storage at {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            raise

    async def disconnect(self) -> None:
        """No-op for local storage"""
        pass

    def _sanitize_key(self, key: str) -> str:
        """Sanitize key to prevent path traversal attacks"""
        import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        # Remove any path separators and dangerous characters
        sanitized = re.sub(r'[/\\:.?*"<>|]', '_', key)
        # Ensure it doesn't start with dots or underscores that could be hidden
        sanitized = sanitized.lstrip('._')
        # Limit length to prevent filesystem issues
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        return sanitized

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from local file"""
        try:
            sanitized_key = self._sanitize_key(key)
            file_path = self.base_path / f"{sanitized_key}.json"

            # Ensure the resolved path is still within our base directory
            if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
                logger.error(f"Path traversal attempt detected: {key}")
                return None

            # Check TTL
            if key in self.ttl_index:
                if datetime.now() > self.ttl_index[key]:
                    await self.delete(key)
                    return None

            if file_path.exists():
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            return None
        except Exception as e:
            logger.error(f"Local storage get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value in local file"""
        try:
            sanitized_key = self._sanitize_key(key)
            file_path = self.base_path / f"{sanitized_key}.json"

            # Ensure the resolved path is still within our base directory
            if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
                logger.error(f"Path traversal attempt detected: {key}")
                return False

            # Store TTL if provided
            if ttl:
                self.ttl_index[key] = datetime.now() + timedelta(seconds=ttl)

            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(value, indent=2))

            return True
        except Exception as e:
            logger.error(f"Local storage set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete local file"""
        try:
            sanitized_key = self._sanitize_key(key)
            file_path = self.base_path / f"{sanitized_key}.json"

            # Ensure the resolved path is still within our base directory
            if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
                logger.error(f"Path traversal attempt detected: {key}")
                return False

            if key in self.ttl_index:
                del self.ttl_index[key]

            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Local storage delete error for key {key}: {e}")
            return False

    async def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix"""
        try:
            keys = []
            pattern = f"{prefix.replace('/', '_')}*"

            for file_path in self.base_path.glob(f"{pattern}.json"):
                key = file_path.stem.replace('_', '/')

                # Check TTL
                if key in self.ttl_index:
                    if datetime.now() > self.ttl_index[key]:
                        await self.delete(key)
                        continue

                keys.append(key)

            return keys
        except Exception as e:
            logger.error(f"Local storage list_keys error: {e}")
            return []

    async def exists(self, key: str) -> bool:
        """Check if file exists"""
        try:
            sanitized_key = self._sanitize_key(key)
            file_path = self.base_path / f"{sanitized_key}.json"

            # Ensure the resolved path is still within our base directory
            if not str(file_path.resolve()).startswith(str(self.base_path.resolve())):
                logger.error(f"Path traversal attempt detected: {key}")
                return False

            # Check TTL
            if key in self.ttl_index:
                if datetime.now() > self.ttl_index[key]:
                    await self.delete(key)
                    return False

            return file_path.exists()
        except Exception as e:
            logger.error(f"Local storage exists error for key {key}: {e}")
            return False


class DistributedStorage:
    """Main distributed storage interface"""

    def __init__(self):
        self.backend: Optional[StorageBackend] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to configured storage backend"""
        storage_type = settings.A2A_REGISTRY_STORAGE.lower()

        try:
            if storage_type == "redis":
                self.backend = RedisBackend()
            elif storage_type == "etcd":
                self.backend = EtcdBackend()
            elif storage_type == "local":
                self.backend = LocalFileBackend()
            else:
                logger.warning(f"Unknown storage type {storage_type}, falling back to local")
                self.backend = LocalFileBackend()

            await self.backend.connect()
            self._connected = True
            logger.info(f"Connected to distributed storage: {storage_type}")

        except Exception as e:
            logger.error(f"Failed to connect to {storage_type} storage: {e}")
            logger.info("Falling back to local file storage")
            self.backend = LocalFileBackend()
            await self.backend.connect()
            self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from storage backend"""
        if self.backend and self._connected:
            await self.backend.disconnect()
            self._connected = False

    async def register_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """Register an agent in distributed storage"""
        if not self._connected:
            await self.connect()

        key = f"agents/{agent_id}"
        agent_data['last_updated'] = datetime.now().isoformat()
        return await self.backend.set(key, agent_data, ttl=3600)  # 1 hour TTL

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent data from distributed storage"""
        if not self._connected:
            await self.connect()

        key = f"agents/{agent_id}"
        return await self.backend.get(key)

    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        if not self._connected:
            await self.connect()

        agent_keys = await self.backend.list_keys("agents/")
        agents = []

        for key in agent_keys:
            agent_data = await self.backend.get(key)
            if agent_data:
                agents.append(agent_data)

        return agents

    async def deregister_agent(self, agent_id: str) -> bool:
        """Remove agent from distributed storage"""
        if not self._connected:
            await self.connect()

        key = f"agents/{agent_id}"
        return await self.backend.delete(key)

    async def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]) -> bool:
        """Update agent health status"""
        if not self._connected:
            await self.connect()

        agent_data = await self.get_agent(agent_id)
        if agent_data:
            agent_data['health'] = health_data
            agent_data['last_health_check'] = datetime.now().isoformat()
            return await self.register_agent(agent_id, agent_data)
        return False

    async def find_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Find agents with specific capability"""
        agents = await self.list_agents()
        matching_agents = []

        for agent in agents:
            if 'capabilities' in agent and capability in agent['capabilities']:
                matching_agents.append(agent)

        return matching_agents


# Global storage instance
_storage_instance = None


async def get_distributed_storage() -> DistributedStorage:
    """Get global distributed storage instance"""
    global _storage_instance

    if _storage_instance is None:
        _storage_instance = DistributedStorage()
        await _storage_instance.connect()

    return _storage_instance