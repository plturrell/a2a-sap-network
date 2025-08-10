"""
SAP Cloud SDK Integration for A2A Agents
Provides enterprise-grade integration with SAP services
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class SAPServiceConfig:
    """Configuration for SAP Cloud services"""
    service_name: str
    service_url: str
    client_id: str
    client_secret: str
    token_url: str
    scopes: List[str]
    timeout: int = 30


class SAPCloudSDK:
    """
    SAP Cloud SDK wrapper for enterprise integration
    Implements SAP best practices for cloud services
    """
    
    def __init__(self):
        self.services: Dict[str, SAPServiceConfig] = {}
        self.tokens: Dict[str, str] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize SAP service configurations"""
        # Alert Notification Service
        if os.getenv("SAP_ANS_CLIENT_ID"):
            self.services["alert_notification"] = SAPServiceConfig(
                service_name="Alert Notification",
                service_url=os.getenv("SAP_ANS_URL", "https://ans.cfapps.eu10.hana.ondemand.com"),
                client_id=os.getenv("SAP_ANS_CLIENT_ID"),
                client_secret=os.getenv("SAP_ANS_CLIENT_SECRET"),
                token_url=os.getenv("SAP_ANS_TOKEN_URL"),
                scopes=["ans!b1.write", "ans!b1.read"]
            )
        
        # Application Logging Service
        if os.getenv("SAP_ALS_CLIENT_ID"):
            self.services["application_logging"] = SAPServiceConfig(
                service_name="Application Logging",
                service_url=os.getenv("SAP_ALS_URL"),
                client_id=os.getenv("SAP_ALS_CLIENT_ID"),
                client_secret=os.getenv("SAP_ALS_CLIENT_SECRET"),
                token_url=os.getenv("SAP_ALS_TOKEN_URL"),
                scopes=["logs.write", "logs.read"]
            )
        
        # Destination Service
        if os.getenv("SAP_DEST_CLIENT_ID"):
            self.services["destination"] = SAPServiceConfig(
                service_name="Destination",
                service_url=os.getenv("SAP_DEST_URL"),
                client_id=os.getenv("SAP_DEST_CLIENT_ID"),
                client_secret=os.getenv("SAP_DEST_CLIENT_SECRET"),
                token_url=os.getenv("SAP_DEST_TOKEN_URL"),
                scopes=["destination.read"]
            )
        
        # Connectivity Service
        if os.getenv("SAP_CONN_CLIENT_ID"):
            self.services["connectivity"] = SAPServiceConfig(
                service_name="Connectivity",
                service_url=os.getenv("SAP_CONN_URL"),
                client_id=os.getenv("SAP_CONN_CLIENT_ID"),
                client_secret=os.getenv("SAP_CONN_CLIENT_SECRET"),
                token_url=os.getenv("SAP_CONN_TOKEN_URL"),
                scopes=["connectivity!b1.read"]
            )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_access_token(self, service_name: str) -> str:
        """Get OAuth2 access token for SAP service"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not configured")
        
        config = self.services[service_name]
        
        # Check if we have a valid cached token
        if service_name in self.tokens:
            # TODO: Implement token expiry check
            return self.tokens[service_name]
        
        # Request new token
        token_data = {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "scope": " ".join(config.scopes)
        }
        
        response = await self.http_client.post(
            config.token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            token_response = response.json()
            access_token = token_response["access_token"]
            self.tokens[service_name] = access_token
            logger.info(f"Successfully obtained token for {service_name}")
            return access_token
        else:
            logger.error(f"Failed to get token for {service_name}: {response.status_code}")
            raise Exception(f"Token request failed: {response.text}")
    
    async def send_alert(
        self,
        subject: str,
        body: str,
        severity: str = "INFO",
        category: str = "NOTIFICATION",
        resource_name: str = "A2A_AGENTS",
        tags: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send alert using SAP Alert Notification Service
        
        Args:
            subject: Alert subject
            body: Alert body/description
            severity: INFO, WARNING, ERROR, FATAL
            category: ALERT, NOTIFICATION
            resource_name: Resource identifier
            tags: Additional metadata tags
        
        Returns:
            Success status
        """
        if "alert_notification" not in self.services:
            logger.warning("Alert Notification Service not configured")
            return False
        
        try:
            token = await self._get_access_token("alert_notification")
            config = self.services["alert_notification"]
            
            alert_data = {
                "eventType": f"{resource_name}.{category}",
                "severity": severity,
                "category": category,
                "subject": subject,
                "body": body,
                "resource": {
                    "resourceName": resource_name,
                    "resourceType": "application"
                },
                "tags": tags or {}
            }
            
            response = await self.http_client.post(
                f"{config.service_url}/cf/producer/v1/resource-events",
                json=alert_data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Alert sent successfully: {subject}")
                return True
            else:
                logger.error(f"Failed to send alert: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def log_to_sap(
        self,
        level: str,
        message: str,
        component: str = "a2a-agents",
        custom_fields: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send logs to SAP Application Logging Service
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            component: Component name
            custom_fields: Additional fields
        
        Returns:
            Success status
        """
        if "application_logging" not in self.services:
            # Fallback to standard logging
            getattr(logger, level.lower(), logger.info)(message)
            return True
        
        try:
            token = await self._get_access_token("application_logging")
            config = self.services["application_logging"]
            
            log_entry = {
                "level": level,
                "msg": message,
                "component": component,
                "timestamp": os.popen('date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"').read().strip(),
                "custom_fields": custom_fields or {}
            }
            
            response = await self.http_client.post(
                f"{config.service_url}/log/entries",
                json=log_entry,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            logger.error(f"Error sending log to SAP: {e}")
            # Fallback to standard logging
            getattr(logger, level.lower(), logger.info)(message)
            return False
    
    async def get_destination(self, destination_name: str) -> Optional[Dict[str, Any]]:
        """
        Get destination configuration from SAP Destination Service
        
        Args:
            destination_name: Name of the destination
        
        Returns:
            Destination configuration or None
        """
        if "destination" not in self.services:
            logger.warning("Destination Service not configured")
            return None
        
        try:
            token = await self._get_access_token("destination")
            config = self.services["destination"]
            
            response = await self.http_client.get(
                f"{config.service_url}/destination-configuration/v1/destinations/{destination_name}",
                headers={
                    "Authorization": f"Bearer {token}"
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get destination: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting destination: {e}")
            return None
    
    async def check_connectivity(self, virtual_host: str) -> bool:
        """
        Check connectivity via SAP Cloud Connector
        
        Args:
            virtual_host: Virtual host to check
        
        Returns:
            Connectivity status
        """
        if "connectivity" not in self.services:
            logger.warning("Connectivity Service not configured")
            return False
        
        try:
            token = await self._get_access_token("connectivity")
            config = self.services["connectivity"]
            
            response = await self.http_client.get(
                f"{config.service_url}/connectivity/v1/subaccountTunnels",
                headers={
                    "Authorization": f"Bearer {token}"
                },
                params={"virtualHost": virtual_host}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error checking connectivity: {e}")
            return False
    
    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()


# Singleton instance
_sap_cloud_sdk: Optional[SAPCloudSDK] = None


def get_sap_cloud_sdk() -> SAPCloudSDK:
    """Get singleton SAP Cloud SDK instance"""
    global _sap_cloud_sdk
    if _sap_cloud_sdk is None:
        _sap_cloud_sdk = SAPCloudSDK()
    return _sap_cloud_sdk


# Integration with existing logging
class SAPLogHandler(logging.Handler):
    """Custom log handler that sends logs to SAP Application Logging Service"""
    
    def __init__(self, component: str = "a2a-agents"):
        super().__init__()
        self.component = component
        self.sdk = get_sap_cloud_sdk()
    
    def emit(self, record):
        """Send log record to SAP"""
        import asyncio
        
        # Skip if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use asyncio.run
            return
        except RuntimeError:
            # No event loop, safe to create one
            pass
        
        level_map = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "ERROR"
        }
        
        custom_fields = {
            "logger_name": record.name,
            "line_number": record.lineno,
            "function_name": record.funcName,
            "thread_name": record.threadName
        }
        
        if hasattr(record, 'exc_info') and record.exc_info:
            custom_fields["exception"] = self.format(record)
        
        # Run async operation in new event loop
        asyncio.run(
            self.sdk.log_to_sap(
                level=level_map.get(record.levelno, "INFO"),
                message=record.getMessage(),
                component=self.component,
                custom_fields=custom_fields
            )
        )