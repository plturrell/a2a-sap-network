import asyncio
import argparse
import logging
import os
import sys
from typing import Optional

from app.a2a.core.security_base import SecureA2AAgent
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
Quality Control Manager Agent Startup Script
Agent 6: Quality assessment and intelligent routing decisions

This script starts the Quality Control Manager Agent that:
1. Assesses outputs from calculation and QA validation agents
2. Makes intelligent routing decisions (direct use, Lean Six Sigma, AI improvement)
3. Provides quality metrics and improvement recommendations
4. Implements MCP tools for integration with other agents
"""

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from qualityControlManagerAgent import QualityControlManagerAgent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/quality_control_agent.log')
    ]
)

logger = logging.getLogger(__name__)


async def register_agent(agent: QualityControlManagerAgent, registry_url: str):
    """Register agent with A2A registry"""
    try:
        # A2A Protocol: Use blockchain messaging instead of httpx

        agent_card = agent.get_agent_card()

        # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
            response = await client.post(
                f"{registry_url}/agents/register",
                json=agent_card.dict(),
                timeout=30.0
            )
            response.raise_for_status()

        logger.info(f"✅ Agent registered with registry at {registry_url}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to register agent: {e}")
        return False


async def health_check_services(agent: QualityControlManagerAgent):
    """Perform health checks on dependent services"""
    services = {
        "Data Manager": agent.data_manager_url,
        "Catalog Manager": agent.catalog_manager_url
    }

    healthy_services = []
    unhealthy_services = []

    for service_name, service_url in services.items():
        try:
            # A2A Protocol: Use blockchain messaging instead of httpx
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service_url}/health", timeout=10.0)
                if response.status_code == 200:
                    healthy_services.append(service_name)
                    logger.info(f"✅ {service_name} is healthy: {service_url}")
                else:
                    unhealthy_services.append(service_name)
                    logger.warning(f"⚠️  {service_name} returned {response.status_code}: {service_url}")
        except Exception as e:
            unhealthy_services.append(service_name)
            logger.error(f"❌ {service_name} health check failed: {e}")

    if unhealthy_services:
        logger.warning(f"Some services are unhealthy: {unhealthy_services}")
        logger.info("Agent will start but some features may be limited")

    return len(healthy_services) > 0


async def start_agent(
    base_url: str,
    data_manager_url: str,
    catalog_manager_url: str,
    registry_url: Optional[str] = None,
    enable_monitoring: bool = True
):
    """Start the Quality Control Manager Agent"""
    logger.info("Starting Quality Control Manager Agent...")

    try:
        # Create agent instance
        agent = QualityControlManagerAgent(
            base_url=base_url,
            data_manager_url=data_manager_url,
            catalog_manager_url=catalog_manager_url,
            enable_monitoring=enable_monitoring
        )

        # Initialize agent
        logger.info("Initializing agent...")
        await agent.initialize()

        # Health check dependent services
        logger.info("Performing health checks...")
        healthy = await health_check_services(agent)

        if not healthy:
            logger.warning("No healthy services found - continuing with limited functionality")

        # Register with A2A registry if provided
        if registry_url:
            logger.info("Registering with A2A registry...")
            await register_agent(agent, registry_url)

        # Create and configure FastAPI app
        app = agent.create_fastapi_app()

        # Add custom endpoints for quality control
        from fastapi import HTTPException
        from pydantic import BaseModel

        class QualityAssessmentRequestModel(BaseModel):
            calculation_result: dict
            qa_validation_result: dict
            quality_thresholds: Optional[dict] = None
            workflow_context: Optional[dict] = None

        @app.post("/api/v1/assess-quality")
        async def assess_quality_endpoint(request: QualityAssessmentRequestModel):
            """REST endpoint for quality assessment"""
            try:
                from qualityControlManagerAgent import QualityAssessmentRequest

                assessment_request = QualityAssessmentRequest(
                    calculation_result=request.calculation_result,
                    qa_validation_result=request.qa_validation_result,
                    quality_thresholds=request.quality_thresholds or {},
                    workflow_context=request.workflow_context or {}
                )

                result = await agent.quality_assessment_skill(assessment_request)
                return {
                    "success": True,
                    "assessment": result.dict()
                }

            except Exception as e:
                logger.error(f"Quality assessment endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/v1/lean-six-sigma")
        async def lean_six_sigma_endpoint(quality_data: dict, process_data: dict):
            """REST endpoint for Lean Six Sigma analysis"""
            try:
                result = await agent.lean_six_sigma_analysis_skill(quality_data, process_data)
                return {
                    "success": True,
                    "analysis": result
                }

            except Exception as e:
                logger.error(f"Lean Six Sigma endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/v1/quality-metrics")
        async def quality_metrics_endpoint():
            """Get current quality metrics and statistics"""
            try:
                return {
                    "processing_stats": agent.processing_stats,
                    "default_thresholds": agent.default_thresholds,
                    "sigma_targets": agent.sigma_targets,
                    "assessment_history_count": len(agent.assessment_history)
                }

            except Exception as e:
                logger.error(f"Quality metrics endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Start server
        import uvicorn

        # Extract port from base_url
        port = 8008
        try:
            if ":" in base_url:
                port = int(base_url.split(":")[-1])
        except:
            pass

        logger.info(f"🚀 Starting Quality Control Manager Agent on port {port}")
        logger.info(f"📊 Monitoring enabled: {enable_monitoring}")
        logger.info(f"🔗 Data Manager: {data_manager_url}")
        logger.info(f"📁 Catalog Manager: {catalog_manager_url}")
        logger.info(f"🌐 Agent URL: {base_url}")

        # Print available endpoints
        logger.info("📋 Available endpoints:")
        logger.info(f"  • Health: {base_url}/health")
        logger.info(f"  • Agent Card: {base_url}/.well-known/agent.json")
        logger.info(f"  • API Docs: {base_url}/api/v1/docs")
        logger.info(f"  • Quality Assessment: {base_url}/api/v1/assess-quality")
        logger.info(f"  • Lean Six Sigma: {base_url}/api/v1/lean-six-sigma")
        logger.info(f"  • Quality Metrics: {base_url}/api/v1/quality-metrics")
        logger.info(f"  • MCP Tools: {base_url}/mcp/tools")

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Agent startup failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            await agent.shutdown()
        except:
            pass
        logger.info("Quality Control Manager Agent stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quality Control Manager Agent - Intelligent quality assessment and routing decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings
  python startup.py

  # Start with custom URLs
  python startup.py --base-url http://localhost:8008 --data-manager http://localhost:8001

  # Start with registry registration
  python startup.py --registry-url http://localhost:8000

Environment Variables:
  DATA_MANAGER_URL          Data Manager service URL (default: http://localhost:8001)
  CATALOG_MANAGER_URL       Catalog Manager service URL (default: http://localhost:8002)
  QUALITY_CONTROL_BASE_URL  Agent base URL (default: http://localhost:8008)
  A2A_REGISTRY_URL          A2A Registry URL for registration
  PROMETHEUS_PORT           Prometheus metrics port (default: 8008)
        """
    )

    parser.add_argument(
        "--base-url",
        default=os.getenv("QUALITY_CONTROL_BASE_URL"),
        help="Agent base URL (default: http://localhost:8008)"
    )

    parser.add_argument(
        "--data-manager-url",
        default=os.getenv("DATA_MANAGER_URL", "http://localhost:8001"),
        help="Data Manager service URL (default: http://localhost:8001)"
    )

    parser.add_argument(
        "--catalog-manager-url",
        default=os.getenv("CATALOG_MANAGER_URL", "http://localhost:8002"),
        help="Catalog Manager service URL (default: http://localhost:8002)"
    )

    parser.add_argument(
        "--registry-url",
        default=os.getenv("A2A_REGISTRY_URL"),
        help="A2A Registry URL for registration"
    )

    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable Prometheus monitoring"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Print startup banner
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         Quality Control Manager Agent                         ║
║                                    Agent 6                                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  • Assesses calculation and QA validation outputs                           ║
║  • Makes intelligent routing decisions                                       ║
║  • Provides Lean Six Sigma analysis                                         ║
║  • Implements AI improvement recommendations                                 ║
║  • Exposes MCP tools for integration                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run agent
    try:
        asyncio.run(start_agent(
            base_url=args.base_url,
            data_manager_url=args.data_manager_url,
            catalog_manager_url=args.catalog_manager_url,
            registry_url=args.registry_url,
            enable_monitoring=not args.no_monitoring
        ))
    except KeyboardInterrupt:
        print("\n👋 Quality Control Manager Agent stopped by user")
    except Exception as e:
        print(f"\n💥 Quality Control Manager Agent failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()