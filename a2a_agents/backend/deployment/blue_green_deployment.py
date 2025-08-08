"""
Blue/Green Deployment Manager for A2A Network
Implements zero-downtime deployments using blue/green strategy
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import httpx
import yaml

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DeploymentEnvironment(str, Enum):
    BLUE = "blue"
    GREEN = "green"


class DeploymentStatus(str, Enum):
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    TESTING = "testing"
    READY = "ready"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLBACK = "rollback"


class ServiceConfig(BaseModel):
    name: str
    image: str
    port: int
    health_endpoint: str
    environment_variables: Dict[str, str] = {}
    resource_limits: Dict[str, str] = {}


class DeploymentConfig(BaseModel):
    version: str
    services: List[ServiceConfig]
    pre_deployment_checks: List[str] = []
    post_deployment_tests: List[str] = []
    rollback_on_failure: bool = True
    health_check_timeout: int = 300  # seconds
    traffic_switch_delay: int = 30   # seconds


class DeploymentState(BaseModel):
    environment: DeploymentEnvironment
    status: DeploymentStatus
    version: str
    deployed_at: datetime
    services: List[Dict[str, Any]]
    health_checks: Dict[str, bool] = {}
    test_results: Dict[str, bool] = {}


class BlueGreenDeploymentManager:
    """
    Blue/Green Deployment Manager for A2A Network
    Manages zero-downtime deployments using Docker Compose
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.deployment_dir = os.path.dirname(config_path)
        self.state_file = os.path.join(self.deployment_dir, "deployment_state.json")
        self.load_balancer_config = os.path.join(self.deployment_dir, "nginx.conf")
        
        self.current_active: Optional[DeploymentEnvironment] = None
        self.environments = {
            DeploymentEnvironment.BLUE: None,
            DeploymentEnvironment.GREEN: None
        }
        
        # Load existing state
        self._load_deployment_state()
    
    async def deploy(self, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """
        Execute blue/green deployment
        """
        logger.info(f"Starting blue/green deployment for version {deployment_config.version}")
        
        try:
            # Determine target environment
            target_env = self._get_target_environment()
            logger.info(f"Deploying to {target_env} environment")
            
            # Create deployment state
            deployment_state = DeploymentState(
                environment=target_env,
                status=DeploymentStatus.PREPARING,
                version=deployment_config.version,
                deployed_at=datetime.now(),
                services=[]
            )
            
            # Update environment state
            self.environments[target_env] = deployment_state
            self._save_deployment_state()
            
            # Phase 1: Pre-deployment checks
            await self._run_pre_deployment_checks(deployment_config)
            
            # Phase 2: Deploy to target environment
            deployment_state.status = DeploymentStatus.DEPLOYING
            self._save_deployment_state()
            
            await self._deploy_services(target_env, deployment_config)
            
            # Phase 3: Health checks
            deployment_state.status = DeploymentStatus.TESTING
            self._save_deployment_state()
            
            health_check_success = await self._run_health_checks(target_env, deployment_config)
            if not health_check_success:
                if deployment_config.rollback_on_failure:
                    await self._rollback_deployment(target_env)
                    return {"success": False, "error": "Health checks failed, deployment rolled back"}
                else:
                    return {"success": False, "error": "Health checks failed"}
            
            # Phase 4: Post-deployment tests
            test_success = await self._run_post_deployment_tests(target_env, deployment_config)
            if not test_success:
                if deployment_config.rollback_on_failure:
                    await self._rollback_deployment(target_env)
                    return {"success": False, "error": "Post-deployment tests failed, deployment rolled back"}
                else:
                    return {"success": False, "error": "Post-deployment tests failed"}
            
            # Phase 5: Mark as ready
            deployment_state.status = DeploymentStatus.READY
            self._save_deployment_state()
            
            # Phase 6: Switch traffic (optional, can be done separately)
            # await self._switch_traffic(target_env)
            
            logger.info(f"Deployment to {target_env} completed successfully")
            return {
                "success": True,
                "environment": target_env,
                "version": deployment_config.version,
                "message": f"Deployment ready in {target_env} environment. Call switch_traffic() to activate."
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if target_env in self.environments and self.environments[target_env]:
                self.environments[target_env].status = DeploymentStatus.FAILED
                self._save_deployment_state()
            
            return {"success": False, "error": str(e)}
    
    async def switch_traffic(self, target_env: Optional[DeploymentEnvironment] = None) -> Dict[str, Any]:
        """
        Switch traffic to the target environment
        """
        if not target_env:
            # Find the ready environment
            for env, state in self.environments.items():
                if state and state.status == DeploymentStatus.READY:
                    target_env = env
                    break
            
            if not target_env:
                return {"success": False, "error": "No ready environment found"}
        
        logger.info(f"Switching traffic to {target_env} environment")
        
        try:
            # Update load balancer configuration
            await self._update_load_balancer(target_env)
            
            # Wait for traffic switch delay
            await asyncio.sleep(self.environments[target_env].deployment_config.traffic_switch_delay if hasattr(self.environments[target_env], 'deployment_config') else 30)
            
            # Mark new environment as active
            old_active = self.current_active
            self.current_active = target_env
            self.environments[target_env].status = DeploymentStatus.ACTIVE
            
            # Mark old environment as inactive (if exists)
            if old_active and old_active in self.environments and self.environments[old_active]:
                self.environments[old_active].status = DeploymentStatus.READY
            
            self._save_deployment_state()
            
            logger.info(f"Traffic successfully switched to {target_env}")
            return {
                "success": True,
                "new_active": target_env,
                "previous_active": old_active,
                "message": f"Traffic switched to {target_env} environment"
            }
            
        except Exception as e:
            logger.error(f"Traffic switch failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def rollback(self) -> Dict[str, Any]:
        """
        Rollback to the previous active environment
        """
        logger.info("Starting rollback procedure")
        
        try:
            # Find the previous stable environment
            rollback_env = None
            for env, state in self.environments.items():
                if state and env != self.current_active and state.status in [DeploymentStatus.READY, DeploymentStatus.ACTIVE]:
                    rollback_env = env
                    break
            
            if not rollback_env:
                return {"success": False, "error": "No stable environment available for rollback"}
            
            # Switch traffic back
            result = await self.switch_traffic(rollback_env)
            if result["success"]:
                # Mark current failed environment
                if self.current_active and self.current_active in self.environments:
                    self.environments[self.current_active].status = DeploymentStatus.ROLLBACK
                
                logger.info(f"Rollback completed to {rollback_env}")
                return {
                    "success": True,
                    "rollback_environment": rollback_env,
                    "message": f"Successfully rolled back to {rollback_env} environment"
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current deployment status
        """
        return {
            "current_active": self.current_active,
            "environments": {
                env.value: state.dict() if state else None
                for env, state in self.environments.items()
            },
            "last_updated": datetime.now()
        }
    
    def _get_target_environment(self) -> DeploymentEnvironment:
        """
        Determine which environment to deploy to
        """
        if self.current_active == DeploymentEnvironment.BLUE:
            return DeploymentEnvironment.GREEN
        else:
            return DeploymentEnvironment.BLUE
    
    async def _run_pre_deployment_checks(self, config: DeploymentConfig):
        """
        Run pre-deployment checks
        """
        logger.info("Running pre-deployment checks")
        
        for check in config.pre_deployment_checks:
            logger.info(f"Running check: {check}")
            try:
                process = await asyncio.create_subprocess_shell(
                    check,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"Pre-deployment check failed: {check}. Error: {stderr.decode()}")
                
                logger.info(f"Check passed: {check}")
            except Exception as e:
                logger.error(f"Pre-deployment check failed: {e}")
                raise
    
    async def _deploy_services(self, target_env: DeploymentEnvironment, config: DeploymentConfig):
        """
        Deploy services to target environment
        """
        logger.info(f"Deploying services to {target_env}")
        
        # Generate docker-compose file for target environment
        compose_file = self._generate_compose_file(target_env, config)
        compose_path = os.path.join(self.deployment_dir, f"docker-compose.{target_env.value}.yml")
        
        with open(compose_path, 'w') as f:
            yaml.dump(compose_file, f, default_flow_style=False)
        
        # Deploy using docker-compose
        try:
            process = await asyncio.create_subprocess_exec(
                'docker-compose', '-f', compose_path, 'up', '-d',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.deployment_dir
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Docker compose deployment failed: {stderr.decode()}")
            
            logger.info(f"Services deployed successfully to {target_env}")
            
            # Update deployment state with service information
            deployment_state = self.environments[target_env]
            deployment_state.services = [
                {
                    "name": service.name,
                    "image": service.image,
                    "port": service.port,
                    "status": "deployed"
                }
                for service in config.services
            ]
            
        except Exception as e:
            logger.error(f"Service deployment failed: {e}")
            raise
    
    def _generate_compose_file(self, target_env: DeploymentEnvironment, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Generate docker-compose configuration for target environment
        """
        services = {}
        
        for service in config.services:
            # Calculate port offset for environment
            port_offset = 1000 if target_env == DeploymentEnvironment.GREEN else 0
            external_port = service.port + port_offset
            
            service_config = {
                "image": service.image,
                "ports": [f"{external_port}:{service.port}"],
                "environment": {
                    **service.environment_variables,
                    "DEPLOYMENT_ENV": target_env.value,
                    "DEPLOYMENT_VERSION": config.version,
                    "PROMETHEUS_PORT": str(service.port)  # For metrics
                },
                "restart": "unless-stopped",
                "networks": [f"a2a-{target_env.value}"]
            }
            
            # Add resource limits if specified
            if service.resource_limits:
                service_config["deploy"] = {
                    "resources": {
                        "limits": service.resource_limits
                    }
                }
            
            # Add health check
            service_config["healthcheck"] = {
                "test": f"curl -f {service.health_endpoint} || exit 1",
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
            
            services[f"{service.name}-{target_env.value}"] = service_config
        
        return {
            "version": "3.8",
            "services": services,
            "networks": {
                f"a2a-{target_env.value}": {
                    "external": False
                }
            }
        }
    
    async def _run_health_checks(self, target_env: DeploymentEnvironment, config: DeploymentConfig) -> bool:
        """
        Run health checks on deployed services
        """
        logger.info(f"Running health checks for {target_env}")
        
        deployment_state = self.environments[target_env]
        port_offset = 1000 if target_env == DeploymentEnvironment.GREEN else 0
        
        # Wait for services to start
        await asyncio.sleep(30)
        
        all_healthy = True
        
        for service in config.services:
            service_port = service.port + port_offset
            health_url = f"http://localhost:{service_port}{service.health_endpoint}"
            
            logger.info(f"Checking health of {service.name} at {health_url}")
            
            healthy = await self._check_service_health(health_url, config.health_check_timeout)
            deployment_state.health_checks[service.name] = healthy
            
            if not healthy:
                all_healthy = False
                logger.error(f"Health check failed for {service.name}")
            else:
                logger.info(f"Health check passed for {service.name}")
        
        self._save_deployment_state()
        return all_healthy
    
    async def _check_service_health(self, health_url: str, timeout: int) -> bool:
        """
        Check health of a single service
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(10)  # Wait before retry
        
        return False
    
    async def _run_post_deployment_tests(self, target_env: DeploymentEnvironment, config: DeploymentConfig) -> bool:
        """
        Run post-deployment tests
        """
        logger.info(f"Running post-deployment tests for {target_env}")
        
        deployment_state = self.environments[target_env]
        all_tests_passed = True
        
        for test in config.post_deployment_tests:
            logger.info(f"Running test: {test}")
            
            try:
                # Set environment variables for tests
                env = os.environ.copy()
                env["DEPLOYMENT_ENV"] = target_env.value
                env["PORT_OFFSET"] = str(1000 if target_env == DeploymentEnvironment.GREEN else 0)
                
                process = await asyncio.create_subprocess_shell(
                    test,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                stdout, stderr = await process.communicate()
                
                test_passed = process.returncode == 0
                deployment_state.test_results[test] = test_passed
                
                if not test_passed:
                    all_tests_passed = False
                    logger.error(f"Test failed: {test}. Error: {stderr.decode()}")
                else:
                    logger.info(f"Test passed: {test}")
                    
            except Exception as e:
                logger.error(f"Test execution failed: {test}. Error: {e}")
                deployment_state.test_results[test] = False
                all_tests_passed = False
        
        self._save_deployment_state()
        return all_tests_passed
    
    async def _update_load_balancer(self, target_env: DeploymentEnvironment):
        """
        Update load balancer configuration to point to target environment
        """
        logger.info(f"Updating load balancer for {target_env}")
        
        port_offset = 1000 if target_env == DeploymentEnvironment.GREEN else 0
        
        # Generate nginx configuration
        nginx_config = self._generate_nginx_config(target_env, port_offset)
        
        with open(self.load_balancer_config, 'w') as f:
            f.write(nginx_config)
        
        # Reload nginx (if running in container)
        try:
            process = await asyncio.create_subprocess_exec(
                'docker', 'exec', 'a2a-nginx', 'nginx', '-s', 'reload',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode == 0:
                logger.info("Nginx configuration reloaded successfully")
            else:
                logger.warning("Failed to reload nginx, may need manual intervention")
                
        except Exception as e:
            logger.warning(f"Could not reload nginx automatically: {e}")
    
    def _generate_nginx_config(self, target_env: DeploymentEnvironment, port_offset: int) -> str:
        """
        Generate nginx configuration for load balancing
        """
        return f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream a2a_agents {{
        server host.docker.internal:{8001 + port_offset};  # Data Product Agent
        server host.docker.internal:{8002 + port_offset};  # Data Standardization Agent
        server host.docker.internal:{8003 + port_offset};  # AI Preparation Agent
        server host.docker.internal:{8004 + port_offset};  # Vector Processing Agent
        server host.docker.internal:{8005 + port_offset};  # Catalog Manager Agent
        server host.docker.internal:{8006 + port_offset};  # Data Manager Agent
    }}
    
    upstream a2a_gateway {{
        server host.docker.internal:{8080 + port_offset};  # API Gateway
    }}
    
    server {{
        listen 80;
        server_name localhost;
        
        # Health check endpoint
        location /health {{
            return 200 'Active Environment: {target_env.value}\\nPort Offset: {port_offset}\\n';
            add_header Content-Type text/plain;
        }}
        
        # API Gateway
        location /api/ {{
            proxy_pass http://a2a_gateway;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
        
        # Direct agent access
        location /agents/ {{
            proxy_pass http://a2a_agents;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
        
        # Health Dashboard
        location /dashboard/ {{
            proxy_pass http://host.docker.internal:8888/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
        
        # WebSocket support for dashboard
        location /ws {{
            proxy_pass http://host.docker.internal:8888/ws;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
        }}
    }}
}}
"""
    
    async def _rollback_deployment(self, failed_env: DeploymentEnvironment):
        """
        Rollback a failed deployment
        """
        logger.info(f"Rolling back failed deployment in {failed_env}")
        
        try:
            # Stop services in failed environment
            compose_path = os.path.join(self.deployment_dir, f"docker-compose.{failed_env.value}.yml")
            
            if os.path.exists(compose_path):
                process = await asyncio.create_subprocess_exec(
                    'docker-compose', '-f', compose_path, 'down',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.deployment_dir
                )
                await process.communicate()
            
            # Mark environment as rolled back
            if self.environments[failed_env]:
                self.environments[failed_env].status = DeploymentStatus.ROLLBACK
                self._save_deployment_state()
            
            logger.info(f"Rollback completed for {failed_env}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    def _load_deployment_state(self):
        """
        Load deployment state from file
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.current_active = DeploymentEnvironment(state_data.get("current_active")) if state_data.get("current_active") else None
                
                for env_str, env_data in state_data.get("environments", {}).items():
                    if env_data:
                        env = DeploymentEnvironment(env_str)
                        self.environments[env] = DeploymentState(**env_data)
                
                logger.info("Deployment state loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load deployment state: {e}")
    
    def _save_deployment_state(self):
        """
        Save deployment state to file
        """
        try:
            state_data = {
                "current_active": self.current_active.value if self.current_active else None,
                "environments": {
                    env.value: state.dict() if state else None
                    for env, state in self.environments.items()
                }
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, default=str, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save deployment state: {e}")


# Factory function
def create_blue_green_manager(config_path: str) -> BlueGreenDeploymentManager:
    """Create Blue/Green deployment manager instance"""
    return BlueGreenDeploymentManager(config_path)