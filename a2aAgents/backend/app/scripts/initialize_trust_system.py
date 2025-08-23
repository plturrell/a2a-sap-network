#!/usr/bin/env python3
"""
Trust System Initialization Script
Initializes trust identities for all A2A agents in the system
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.trustInitializer import get_trust_initializer, initialize_agent_trust_system
from app.core.trustMiddleware import get_trust_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(Path(__import__('app.a2a.config.storageConfig', fromlist=['get_logs_path']).get_logs_path()) / 'trust_initialization.log'))
    ]
)

logger = logging.getLogger(__name__)


async def initialize_all_system_agents():
    """Initialize trust for all system agents"""
    logger.info("🚀 Starting A2A Trust System Initialization")
    
    # Define all system agents that need trust initialization
    system_agents = [
        # Core Data Agents
        {
            "agent_id": "data_product_registration_agent",
            "agent_type": "DataProductRegistrationAgent",
            "agent_name": "Data Product Registration Agent",
            "description": "Manages data product registration and lifecycle"
        },
        {
            "agent_id": "financial_standardization_agent", 
            "agent_type": "FinancialStandardizationAgent",
            "agent_name": "Financial Standardization Agent",
            "description": "Handles financial data standardization and compliance"
        },
        {
            "agent_id": "ord_registry_agent",
            "agent_type": "ORDRegistryAgent", 
            "agent_name": "ORD Registry Agent",
            "description": "Object Resource Discovery registry management"
        },
        {
            "agent_id": "data_manager_agent",
            "agent_type": "DataManagerAgent",
            "agent_name": "Data Manager Agent", 
            "description": "Central data management and caching"
        },
        
        # Processing and Analysis Agents
        {
            "agent_id": "search_agent",
            "agent_type": "SearchAgent",
            "agent_name": "Search Agent",
            "description": "Enhanced search and discovery capabilities"
        },
        {
            "agent_id": "cache_manager_agent",
            "agent_type": "CacheManagerAgent",
            "agent_name": "Cache Manager Agent",
            "description": "Multi-tier cache management system"
        },
        
        # Governance and Compliance Agents
        {
            "agent_id": "compliance_agent",
            "agent_type": "ComplianceAgent",
            "agent_name": "Compliance Agent",
            "description": "Regulatory compliance and audit support"
        },
        {
            "agent_id": "audit_agent",
            "agent_type": "AuditAgent",
            "agent_name": "Audit Agent",
            "description": "System audit and logging capabilities"
        },
        
        # System Management Agents
        {
            "agent_id": "agent_builder_agent",
            "agent_type": "AgentBuilderAgent", 
            "agent_name": "Agent Builder Agent",
            "description": "Dynamic agent generation and template management"
        },
        {
            "agent_id": "workflow_orchestrator_agent",
            "agent_type": "WorkflowOrchestratorAgent",
            "agent_name": "Workflow Orchestrator Agent", 
            "description": "Cross-agent workflow coordination"
        },
        {
            "agent_id": "security_monitor_agent",
            "agent_type": "SecurityMonitorAgent",
            "agent_name": "Security Monitor Agent",
            "description": "Real-time security monitoring and threat detection"
        },
        
        # Specialized Agents
        {
            "agent_id": "notification_agent",
            "agent_type": "NotificationAgent",
            "agent_name": "Notification Agent",
            "description": "Multi-channel notification delivery"
        },
        {
            "agent_id": "integration_agent",
            "agent_type": "IntegrationAgent", 
            "agent_name": "Integration Agent",
            "description": "External system integration and API management"
        },
        {
            "agent_id": "analytics_agent",
            "agent_type": "AnalyticsAgent",
            "agent_name": "Analytics Agent",
            "description": "Data analytics and insights generation"
        }
    ]
    
    try:
        # Initialize trust system
        trust_initializer = await get_trust_initializer()
        trust_middleware = await get_trust_middleware()
        
        initialization_results = []
        
        # Initialize each agent
        for agent_config in system_agents:
            logger.info(f"Initializing trust for: {agent_config['agent_name']}")
            
            result = await initialize_agent_trust_system(
                agent_config["agent_id"],
                agent_config["agent_type"], 
                agent_config["agent_name"]
            )
            
            result["description"] = agent_config["description"]
            initialization_results.append(result)
            
            if result.get("status") == "initialized":
                logger.info(f"✅ {agent_config['agent_name']} trust initialized successfully")
            else:
                logger.warning(f"⚠️ {agent_config['agent_name']} trust initialization: {result.get('status', 'unknown')}")
        
        # Generate initialization report
        await generate_initialization_report(initialization_results, trust_initializer)
        
        logger.info("✅ A2A Trust System Initialization Complete")
        
        return initialization_results
        
    except Exception as e:
        logger.error(f"❌ Trust system initialization failed: {e}")
        raise


async def generate_initialization_report(results, trust_initializer):
    """Generate a comprehensive initialization report"""
    try:
        # Get trust system status
        trust_status = await trust_initializer.get_trust_status()
        
        # Create report
        report = {
            "initialization_report": {
                "generated_at": datetime.utcnow().isoformat(),
                "trust_system_status": trust_status,
                "agent_initialization_results": results,
                "summary": {
                    "total_agents": len(results),
                    "successfully_initialized": len([r for r in results if r.get("status") == "initialized"]),
                    "already_initialized": len([r for r in results if r.get("status") == "already_initialized"]),
                    "failed_initialization": len([r for r in results if r.get("status") == "failed"]),
                    "skipped": len([r for r in results if r.get("status") == "skipped"])
                }
            }
        }
        
        # Save report
        from app.a2a.config.storageConfig import get_reports_path
        report_file = get_reports_path() / "a2a_trust_initialization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Initialization report saved: {report_file}")
        
        # Print summary
        summary = report["initialization_report"]["summary"]
        logger.info("🔍 Initialization Summary:")
        logger.info(f"   Total Agents: {summary['total_agents']}")
        logger.info(f"   Successfully Initialized: {summary['successfully_initialized']}")
        logger.info(f"   Already Initialized: {summary['already_initialized']}")
        logger.info(f"   Failed: {summary['failed_initialization']}")
        logger.info(f"   Skipped: {summary['skipped']}")
        
    except Exception as e:
        logger.error(f"Failed to generate initialization report: {e}")


async def verify_trust_system():
    """Verify that the trust system is working correctly"""
    logger.info("🔍 Verifying Trust System Functionality")
    
    try:
        trust_initializer = await get_trust_initializer()
        
        # Test message signing and verification
        test_agent_id = "test_verification_agent"
        test_agent_type = "TestAgent"
        
        # Initialize test agent
        await trust_initializer.initialize_agent_trust_identity(
            test_agent_id, 
            test_agent_type,
            "Test Verification Agent"
        )
        
        # Create test message
        test_message = {
            "message_type": "test",
            "content": "Trust system verification test",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Sign message
        signed_message = await trust_initializer.sign_agent_message(test_agent_id, test_message)
        
        # Verify message
        verified, verification_result = await trust_initializer.verify_agent_message(signed_message)
        
        if verified:
            logger.info("✅ Trust system verification successful")
            logger.info(f"   Verification details: {verification_result}")
        else:
            logger.error("❌ Trust system verification failed")
            logger.error(f"   Verification error: {verification_result}")
        
        return verified
        
    except Exception as e:
        logger.error(f"Trust system verification failed: {e}")
        return False


async def cleanup_test_data():
    """Clean up test data created during verification"""
    try:
        # Remove test agent data if needed
        from app.a2a.config.storageConfig import get_trust_storage_path
        trust_storage_path = os.getenv("TRUST_STORAGE_PATH", str(get_trust_storage_path()))
        test_files = [
            "test_verification_agent_trust.json"
        ]
        
        for test_file in test_files:
            file_path = Path(trust_storage_path) / test_file
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up test file: {test_file}")
        
    except Exception as e:
        logger.warning(f"Test cleanup failed: {e}")


async def main():
    """Main execution function"""
    try:
        logger.info("🎯 A2A Trust System Initialization Starting...")
        
        # Initialize all system agents
        results = await initialize_all_system_agents()
        
        # Verify trust system is working
        verification_success = await verify_trust_system()
        
        if not verification_success:
            logger.error("❌ Trust system verification failed - please check configuration")
            return 1
        
        # Clean up test data
        await cleanup_test_data()
        
        logger.info("🎉 Trust System Initialization Completed Successfully!")
        
        # Print final status
        success_count = len([r for r in results if r.get("status") in ["initialized", "already_initialized"]])
        logger.info(f"📊 Final Status: {success_count}/{len(results)} agents have trust identities")
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Trust system initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)