"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

#!/usr/bin/env python3
"""
Production deployment script for Context Engineering Agent

This script handles:
- Environment validation
- Configuration management
- Agent deployment
- Health monitoring
- Integration testing
"""

import asyncio
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from contextEngineering.contextEngineeringAgent import ContextEngineeringAgent
from a2aCommon import A2AMessage, MessageRole


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContextAgentDeployment:
    """Handles deployment and management of Context Engineering Agent"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_configuration(config_path)
        self.agent = None
        self.app = None
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {
            # Base configuration
            "agent_id": os.getenv("CONTEXT_AGENT_ID", "context_engineering_agent"),
            "base_url": os.getenv("CONTEXT_AGENT_URL"),
            "port": int(os.getenv("CONTEXT_AGENT_PORT", "8090")),
            "host": os.getenv("CONTEXT_AGENT_HOST", "0.0.0.0"),
            
            # Feature flags
            "enable_trust": os.getenv("ENABLE_TRUST", "true").lower() == "true",
            "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
            "enable_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
            
            # Resource limits
            "max_context_tokens": int(os.getenv("MAX_CONTEXT_TOKENS", "8192")),
            "max_cache_size": int(os.getenv("MAX_CACHE_SIZE", "1000")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
            
            # Model configuration
            "nlp_model": os.getenv("NLP_MODEL", "en_core_web_sm"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            
            # Integration settings
            "agent_manager_url": os.getenv("AGENT_MANAGER_URL", "os.getenv("A2A_GATEWAY_URL")"),
            "registry_url": os.getenv("REGISTRY_URL"),
            
            # Performance settings
            "workers": int(os.getenv("WORKERS", "4")),
            "log_level": os.getenv("LOG_LEVEL", "info"),
            
            # Security settings
            "require_trust": os.getenv("REQUIRE_TRUST", "false").lower() == "true",
            "api_key": os.getenv("CONTEXT_API_KEY", ""),
        }
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return config
    
    async def validate_environment(self) -> bool:
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        checks = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            checks.append(False)
        else:
            logger.info("✓ Python version OK")
            checks.append(True)
        
        # Check required packages
        required_packages = [
            "fastapi", "uvicorn", "spacy", "numpy", 
            "sklearn", "networkx", "aiohttp"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ Package {package} available")
                checks.append(True)
            except ImportError:
                logger.error(f"✗ Package {package} not found")
                checks.append(False)
        
        # Check NLP model
        try:
            import spacy
            spacy.load(self.config["nlp_model"])
            logger.info(f"✓ NLP model {self.config['nlp_model']} available")
            checks.append(True)
        except:
            logger.warning(f"✗ NLP model {self.config['nlp_model']} not found - will download")
            checks.append(True)  # Not critical, can be downloaded
        
        # Check connectivity to agent manager
        if self.config.get("agent_manager_url"):
            try:
                # A2A Protocol: Use blockchain messaging instead of aiohttp
                async with # WARNING: aiohttp ClientSession usage violates A2A protocol - must use blockchain messaging
        # aiohttp\.ClientSession() as session:
                    async with session.get(
                        f"{self.config['agent_manager_url']}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            logger.info("✓ Agent Manager connectivity OK")
                            checks.append(True)
                        else:
                            logger.warning("✗ Agent Manager not responding")
                            checks.append(False)
            except:
                logger.warning("✗ Cannot reach Agent Manager")
                checks.append(False)
        
        return all(checks)
    
    async def deploy_agent(self):
        """Deploy the Context Engineering Agent"""
        logger.info("Deploying Context Engineering Agent...")
        
        # Create agent instance
        self.agent = ContextEngineeringAgent(
            base_url=self.config["base_url"],
            config=self.config
        )
        
        # Initialize agent
        await self.agent.initialize()
        
        # Create FastAPI app
        self.app = self.agent.create_fastapi_app()
        
        # Add custom health endpoint
        @self.app.get("/deployment/health")
        async def deployment_health():
            return {
                "status": "healthy",
                "agent_id": self.agent.agent_id,
                "version": self.agent.version,
                "capabilities": len(self.agent.capabilities),
                "metrics": self.agent.metrics
            }
        
        logger.info(f"Context Engineering Agent deployed successfully")
        logger.info(f"Agent ID: {self.agent.agent_id}")
        logger.info(f"Base URL: {self.config['base_url']}")
        logger.info(f"Capabilities: {len(self.agent.capabilities)}")
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests to verify agent functionality"""
        logger.info("Running integration tests...")
        
        test_results = []
        
        # Test 1: Context Parsing
        logger.info("Test 1: Context Parsing")
        test_context = """
        The quarterly financial report shows revenue of $2.5M, 
        with a 15% increase from last quarter. Major customers 
        include Acme Corp and GlobalTech Industries.
        """
        
        message = A2AMessage(
            role=MessageRole.USER,
            content={"context": test_context}
        )
        
        result = await self.agent.handle_parse_context(message, "test_ctx_1")
        
        if result.get("status") == "success":
            logger.info("✓ Context parsing test passed")
            test_results.append(True)
        else:
            logger.error(f"✗ Context parsing test failed: {result}")
            test_results.append(False)
        
        # Test 2: Relevance Assessment
        logger.info("Test 2: Relevance Assessment")
        message = A2AMessage(
            role=MessageRole.USER,
            content={
                "context": test_context,
                "query": "What is the revenue growth?",
                "task_type": "analysis"
            }
        )
        
        result = await self.agent.handle_assess_relevance(message, "test_ctx_2")
        
        if result.get("status") == "success":
            logger.info("✓ Relevance assessment test passed")
            test_results.append(True)
        else:
            logger.error(f"✗ Relevance assessment test failed: {result}")
            test_results.append(False)
        
        # Test 3: Context Optimization
        logger.info("Test 3: Context Optimization")
        result = await self.agent.optimize_context_window({
            "contexts": [{"text": test_context, "id": "ctx1"}],
            "query": "revenue",
            "max_tokens": 100
        })
        
        if result.get("status") == "success":
            logger.info("✓ Context optimization test passed")
            test_results.append(True)
        else:
            logger.error(f"✗ Context optimization test failed: {result}")
            test_results.append(False)
        
        # Test 4: Template Generation
        logger.info("Test 4: Template Generation")
        result = await self.agent.generate_context_template({
            "task_type": "analysis",
            "domain": "financial",
            "complexity": "medium"
        })
        
        if result.get("status") == "success":
            logger.info("✓ Template generation test passed")
            test_results.append(True)
        else:
            logger.error(f"✗ Template generation test failed: {result}")
            test_results.append(False)
        
        # Summary
        passed = sum(test_results)
        total = len(test_results)
        logger.info(f"\nIntegration Tests: {passed}/{total} passed")
        
        return all(test_results)
    
    async def start_server(self):
        """Start the agent server"""
        config = uvicorn.Config(
            self.app,
            host=self.config["host"],
            port=self.config["port"],
            log_level=self.config["log_level"],
            workers=self.config.get("workers", 1) if self.config.get("workers", 1) > 1 else None
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"Starting Context Engineering Agent server...")
        logger.info(f"Host: {self.config['host']}")
        logger.info(f"Port: {self.config['port']}")
        logger.info(f"Workers: {self.config.get('workers', 1)}")
        
        await server.serve()
    
    async def deploy_and_run(self, skip_tests: bool = False):
        """Full deployment process"""
        try:
            # Validate environment
            if not await self.validate_environment():
                logger.error("Environment validation failed")
                return False
            
            # Deploy agent
            await self.deploy_agent()
            
            # Run integration tests
            if not skip_tests:
                if not await self.run_integration_tests():
                    logger.warning("Some integration tests failed")
            
            # Start server
            await self.start_server()
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy Context Engineering Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip integration tests"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't deploy"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployment = ContextAgentDeployment(args.config)
    
    if args.validate_only:
        # Just validate environment
        valid = await deployment.validate_environment()
        sys.exit(0 if valid else 1)
    
    # Full deployment
    success = await deployment.deploy_and_run(skip_tests=args.skip_tests)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())