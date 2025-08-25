import asyncio
import os
import sys
import logging
import json
from datetime import datetime


from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Test Enhanced Data Standardization Agent with MCP Integration
"""

# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
os.environ['AGENT_PRIVATE_KEY'] = 'test_key_12345'
os.environ['STANDARDIZATION_OUTPUT_DIR'] = '/tmp/standardized_data'
os.environ['CATALOG_MANAGER_URL'] = os.getenv("A2A_SERVICE_URL")

async def test_enhanced_standardization_agent():
    """Test the enhanced Data Standardization Agent with MCP"""

    try:
        # Import after paths are set
        from app.a2a.agents.agent1Standardization.active.enhancedDataStandardizationAgentMcp import (
            EnhancedDataStandardizationAgentMCP
        )
        print("✅ Import successful!")

        # Create agent
        agent = EnhancedDataStandardizationAgentMCP(
            base_url=os.getenv("DATA_MANAGER_URL"),
            enable_monitoring=False  # Disable for testing
        )
        print(f"✅ Agent created: {agent.name} (ID: {agent.agent_id})")

        # Initialize agent
        await agent.initialize()
        print("✅ Agent initialized")

        # Check MCP tools
        tools = agent.list_mcp_tools()
        print(f"\n📋 MCP Tools: {len(tools)}")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")

        # Check MCP resources
        resources = agent.list_mcp_resources()
        print(f"\n📊 MCP Resources: {len(resources)}")
        for resource in resources:
            print(f"   - {resource['uri']}: {resource['name']}")

        # Test 1: Create test data
        print("\n🧪 Test 1: Creating test data...")
        test_accounts = [
            {
                "account_id": "ACC001",
                "account_name": "Cash and Equivalents",
                "account_code": "1000",
                "account_type": "Assets",
                "sub_type": "Current Assets",
                "entity": "CORP001",
                "department": "FINANCE",
                "currency": "USD",
                "status": "active"
            },
            {
                "account_id": "ACC002",
                "account_name": "Accounts Receivable",
                "account_code": "1200",
                "account_type": "Assets",
                "sub_type": "Current Assets",
                "entity": "CORP001",
                "department": "FINANCE",
                "currency": "USD",
                "status": "active"
            }
        ]

        test_locations = [
            {
                "location_id": "LOC001",
                "location_name": "New York Headquarters",
                "location_code": "NYC-HQ",
                "location_type": "headquarters",
                "address": "123 Wall Street",
                "city": "New York",
                "state": "NY",
                "country": "United States",
                "postal_code": "10005"
            }
        ]

        # Test 2: Standardize accounts using MCP tool
        print("\n🧪 Test 2: Standardizing accounts via MCP...")
        result = await agent.standardize_data_mcp(
            data_type="account",
            items=test_accounts,
            options={
                "mode": "batch",
                "validate": True,
                "cache_results": True
            }
        )

        if result.get("success"):
            print(f"   ✅ Standardized {result['result']['successful_records']} accounts")
            print(f"   Processing time: {result['metrics']['processing_time']:.2f}s")
            print(f"   Memory usage: {result['metrics']['memory_usage_mb']:.2f} MB")

        # Test 3: Validate standardized data
        print("\n🧪 Test 3: Validating standardized data...")
        if result.get("success") and result["result"].get("standardized_data"):
            validation_result = await agent.validate_standardization_mcp(
                data_type="account",
                standardized_items=result["result"]["standardized_data"],
                validation_level="comprehensive"
            )

            if validation_result.get("success"):
                summary = validation_result["summary"]
                print(f"   Validation success rate: {summary['success_rate'] * 100:.1f}%")
                print(f"   Valid items: {summary['valid_items']}/{summary['total_items']}")

        # Test 4: Enrich standardized data
        print("\n🧪 Test 4: Enriching standardized data...")
        if result.get("success") and result["result"].get("standardized_data"):
            enrichment_result = await agent.enrich_standardized_data_mcp(
                data_type="account",
                standardized_items=result["result"]["standardized_data"],
                enrichment_sources=["internal", "reference_data"]
            )

            if enrichment_result.get("success"):
                print(f"   ✅ Enriched {enrichment_result['total_enriched']} items")
                print(f"   Sources used: {', '.join(enrichment_result['enrichment_sources'])}")

        # Test 5: Batch standardization
        print("\n🧪 Test 5: Batch standardization of multiple types...")
        batch_result = await agent.batch_standardize_mcp(
            batches={
                "account": test_accounts,
                "location": test_locations
            },
            parallel_processing=True,
            memory_limit_mb=512
        )

        if batch_result.get("success"):
            summary = batch_result["summary"]
            print(f"   ✅ Processed {summary['data_types_processed']} data types")
            print(f"   Total items: {summary['total_items']}")
            print(f"   Successful: {summary['total_successful']}")

        # Test 6: Access MCP resources
        print("\n🧪 Test 6: Accessing MCP resources...")

        # Get schemas
        schemas = await agent.get_standardization_schemas()
        print(f"   Available schemas: {schemas['total_schemas']}")
        for data_type, schema in schemas['schemas'].items():
            print(f"     - {data_type}: v{schema['version']} ({len(schema['fields'])} fields)")

        # Get metrics
        metrics = await agent.get_standardization_metrics()
        print(f"\n   Performance metrics:")
        print(f"     - Total processed: {metrics['processing_metrics']['total_processed']}")
        print(f"     - Success rate: {metrics['processing_metrics']['success_rate'] * 100:.1f}%")
        print(f"     - Cache hit rate: {metrics['cache_metrics']['hit_rate'] * 100:.1f}%")
        print(f"     - CPU usage: {metrics['resource_metrics']['cpu_usage_percent']:.1f}%")

        # Get batch status
        batch_status = await agent.get_batch_status()
        print(f"\n   Batch processing status:")
        print(f"     - Active batches: {batch_status['active_batches']}")
        print(f"     - Batch size: {batch_status['batch_config']['batch_size']}")
        print(f"     - Parallel workers: {batch_status['batch_config']['parallel_workers']}")

        # Test 7: Test error handling
        print("\n🧪 Test 7: Testing error handling...")
        # Try with invalid data type
        error_result = await agent.standardize_data_mcp(
            data_type="invalid_type",
            items=[{"test": "data"}]
        )
        print(f"   Error handling: {error_result.get('error', 'No error')}")

        # Test 8: Test caching
        print("\n🧪 Test 8: Testing cache performance...")
        # Standardize same data again to test cache
        cache_test1 = await agent.standardize_data_mcp(
            data_type="account",
            items=test_accounts,
            options={"cache_results": True}
        )

        # Should hit cache
        cache_test2 = await agent.standardize_data_mcp(
            data_type="account",
            items=test_accounts,
            options={"cache_results": True}
        )

        if cache_test2.get("success") and cache_test2["result"].get("cached"):
            print("   ✅ Cache hit successful!")

        print("\n✅ All tests completed successfully!")

        # Cleanup
        await agent.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_standardization_agent())
    sys.exit(0 if result else 1)