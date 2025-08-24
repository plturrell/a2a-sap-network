"""
Authentication Manager for Platform Integrations
Handles OAuth2, token caching, and refresh logic
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import json
import time
from typing import Dict, Optional, Any
from datetime import datetime
import logging
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenCache:
    """In-memory token cache with file persistence"""

    def __init__(self, cache_dir: str = "/tmp/a2a_token_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load tokens from disk"""
        try:
            cache_file = self.cache_dir / "tokens.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.tokens = json.load(f)
                    logger.info(f"Loaded {len(self.tokens)} cached tokens")
        except Exception as e:
            logger.error(f"Failed to load token cache: {e}")
            self.tokens = {}

    def _save_cache(self):
        """Save tokens to disk"""
        try:
            cache_file = self.cache_dir / "tokens.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.tokens, f)
        except Exception as e:
            logger.error(f"Failed to save token cache: {e}")

    def get_token(self, key: str) -> Optional[Dict[str, Any]]:
        """Get token if valid"""
        if key in self.tokens:
            token_data = self.tokens[key]
            expires_at = token_data.get("expires_at", 0)

            # Check if token is still valid (with 5 min buffer)
            if time.time() < (expires_at - 300):
                return token_data
            else:
                logger.info(f"Token expired for {key}")
                del self.tokens[key]

        return None

    def set_token(self, key: str, token: str, expires_in: int, refresh_token: Optional[str] = None):
        """Cache token with expiration"""
        self.tokens[key] = {
            "access_token": token,
            "refresh_token": refresh_token,
            "expires_at": time.time() + expires_in,
            "cached_at": datetime.utcnow().isoformat(),
        }
        self._save_cache()


class OAuth2Client:
    """OAuth2 client with automatic token refresh"""

    def __init__(
        self, client_id: str, client_secret: str, token_url: str, scope: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.token_cache = TokenCache()
        self.cache_key = f"{client_id}_{token_url}"

    async def get_access_token(self) -> str:
        """Get valid access token, refreshing if needed"""
        # Check cache first
        cached = self.token_cache.get_token(self.cache_key)
        if cached:
            return cached["access_token"]

        # Get new token
        token_data = await self._request_token()
        return token_data["access_token"]

    async def _request_token(self, refresh_token: Optional[str] = None) -> Dict[str, Any]:
        """Request new token from OAuth2 server via A2A blockchain messaging"""
        # A2A Protocol Compliance: Use blockchain messaging for OAuth2 token requests
        from .networkClient import A2ANetworkClient
        
        network_client = A2ANetworkClient(agent_id="auth_manager")
        
        token_request = {
            "operation": "oauth2_token_request",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token_url": self.token_url,
            "grant_type": "refresh_token" if refresh_token else "client_credentials",
            "refresh_token": refresh_token,
            "scope": self.scope,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            response = await network_client.send_a2a_message(
                to_agent="oauth_proxy_agent",
                message=token_request,
                message_type="OAUTH2_TOKEN_REQUEST"
            )
            
            if not response or response.get('error'):
                error_msg = f"OAuth2 token request failed via blockchain: {response.get('error', 'Unknown error')}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            token_data = response.get("token_data", {})
            
            if not token_data.get("access_token"):
                raise RuntimeError("No access token received from OAuth2 proxy agent")

            # Cache the token
            self.token_cache.set_token(
                self.cache_key,
                token_data["access_token"],
                token_data.get("expires_in", 3600),
                token_data.get("refresh_token"),
            )

            logger.info(f"✅ Successfully obtained OAuth2 token for {self.cache_key} via blockchain")
            return token_data

        except Exception as e:
            logger.error(f"❌ OAuth2 token request error via blockchain: {e}")
            raise RuntimeError(f"OAuth2 token request failed: {e}")


class BasicAuthClient:
    """Basic authentication client"""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def get_auth_header(self) -> Dict[str, str]:
        """Get basic auth header"""
        import base64

        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}


class BearerTokenClient:
    """Simple bearer token authentication"""

    def __init__(self, token: str):
        self.token = token

    def get_auth_header(self) -> Dict[str, str]:
        """Get bearer token header"""
        return {"Authorization": f"Bearer {self.token}"}


class AuthManager:
    """Centralized authentication manager for all platforms"""

    def __init__(self):
        self.oauth_clients: Dict[str, OAuth2Client] = {}
        self.basic_clients: Dict[str, BasicAuthClient] = {}
        self.bearer_clients: Dict[str, BearerTokenClient] = {}

    def register_oauth2(
        self,
        platform_id: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None,
    ):
        """Register OAuth2 client for platform"""
        self.oauth_clients[platform_id] = OAuth2Client(client_id, client_secret, token_url, scope)
        logger.info(f"Registered OAuth2 client for {platform_id}")

    def register_basic_auth(self, platform_id: str, username: str, password: str):
        """Register basic auth for platform"""
        self.basic_clients[platform_id] = BasicAuthClient(username, password)
        logger.info(f"Registered basic auth for {platform_id}")

    def register_bearer_token(self, platform_id: str, token: str):
        """Register bearer token for platform"""
        self.bearer_clients[platform_id] = BearerTokenClient(token)
        logger.info(f"Registered bearer token for {platform_id}")

    async def get_auth_headers(self, platform_id: str) -> Dict[str, str]:
        """Get authentication headers for platform"""
        # Check OAuth2
        if platform_id in self.oauth_clients:
            token = await self.oauth_clients[platform_id].get_access_token()
            return {"Authorization": f"Bearer {token}"}

        # Check basic auth
        if platform_id in self.basic_clients:
            return self.basic_clients[platform_id].get_auth_header()

        # Check bearer token
        if platform_id in self.bearer_clients:
            return self.bearer_clients[platform_id].get_auth_header()

        logger.warning(f"No authentication configured for {platform_id}")
        return {}

    def configure_from_dict(self, config: Dict[str, Any]):
        """Configure authentication from config dict"""
        for platform_id, auth_config in config.items():
            method = auth_config.get("method", "").lower()

            if method == "oauth2":
                self.register_oauth2(
                    platform_id,
                    auth_config["client_id"],
                    auth_config["client_secret"],
                    auth_config["token_url"],
                    auth_config.get("scope"),
                )
            elif method == "basic":
                self.register_basic_auth(
                    platform_id, auth_config["username"], auth_config["password"]
                )
            elif method in ["bearer", "token"]:
                self.register_bearer_token(platform_id, auth_config["token"])
            else:
                logger.warning(f"Unknown auth method '{method}' for {platform_id}")


# Global auth manager instance
_auth_manager = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
