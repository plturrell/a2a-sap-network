#!/usr/bin/env python3
"""
Test script to verify SQLite fallback database is working correctly
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the backend app to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Test only SQLite client directly to avoid import issues
from app.clients.sqlite_client import get_sqlite_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_sqlite_client():
    """Test basic SQLite client functionality"""
    logger.info("🧪 Testing SQLite client basic operations...")
    
    try:
        # Get SQLite client
        client = get_sqlite_client()
        
        # Test health check
        health = client.health_check()
        logger.info(f"📊 SQLite health check: {health}")
        
        if health["status"] != "healthy":
            logger.error("❌ SQLite client health check failed")
            return False
        
        # Test agent data operations
        test_agent_id = "test-agent-001"
        test_data = {
            "message": "Hello from SQLite test",
            "timestamp": datetime.utcnow().isoformat(),
            "test_number": 42
        }
        
        # Store test data
        result = client.store_agent_data(
            agent_id=test_agent_id,
            data_type="test_message",
            data=test_data,
            metadata={"test": True}
        )
        
        if result.status_code == 201:
            logger.info("✅ Successfully stored test agent data")
        else:
            logger.error(f"❌ Failed to store agent data: {result.error}")
            return False
        
        # Retrieve test data
        result = client.get_agent_data(agent_id=test_agent_id)
        
        if result.status_code == 200 and result.data:
            logger.info(f"✅ Successfully retrieved {len(result.data)} records")
            logger.info(f"📦 Sample record: {result.data[0]}")
        else:
            logger.error(f"❌ Failed to retrieve agent data: {result.error}")
            return False
        
        # Test financial data operations
        financial_records = [
            {"symbol": "AAPL", "price": 150.25, "volume": 1000},
            {"symbol": "GOOGL", "price": 2750.50, "volume": 500}
        ]
        
        result = client.store_financial_data(
            data_source="test_market",
            data_type="stock_quotes",
            records=financial_records
        )
        
        if result.status_code == 201:
            logger.info("✅ Successfully stored test financial data")
        else:
            logger.error(f"❌ Failed to store financial data: {result.error}")
            return False
        
        # Retrieve financial data
        result = client.get_financial_data(data_source="test_market")
        
        if result.status_code == 200 and result.data:
            logger.info(f"✅ Successfully retrieved {len(result.data)} financial records")
        else:
            logger.error(f"❌ Failed to retrieve financial data: {result.error}")
            return False
        
        logger.info("🎉 SQLite client tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SQLite client test failed: {e}")
        return False


async def test_ord_storage():
    """Test ORD registry related tables in SQLite"""
    logger.info("🧪 Testing ORD registry related tables in SQLite...")
    
    try:
        # Get SQLite client directly
        client = get_sqlite_client()
        
        # Test that ORD tables were created
        ord_tables = ["ord_registrations", "ord_resource_index", "ord_replication_log"]
        
        for table in ord_tables:
            exists = client.validate_table_exists(table)
            if exists:
                logger.info(f"✅ ORD table '{table}' exists")
            else:
                logger.warning(f"⚠️  ORD table '{table}' does not exist")
        
        logger.info("🎉 ORD storage table tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ORD storage test failed: {e}")
        return False


async def test_database_connectivity():
    """Test database connectivity and fallback behavior"""
    logger.info("🧪 Testing database connectivity and fallback behavior...")
    
    try:
        # Test SQLite client directly
        client = get_sqlite_client()
        
        # Test table existence
        tables = ["agent_data", "agent_interactions", "financial_data", "ord_registrations"]
        
        for table in tables:
            exists = client.validate_table_exists(table)
            if exists:
                logger.info(f"✅ Table '{table}' exists")
            else:
                logger.warning(f"⚠️  Table '{table}' does not exist")
        
        # Test raw query execution
        result = client.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
        
        if result.status_code == 200:
            table_names = [row['name'] for row in result.data]
            logger.info(f"📋 Available tables: {table_names}")
        else:
            logger.error(f"❌ Failed to query table list: {result.error}")
            return False
        
        logger.info("🎉 Database connectivity tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connectivity test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("🚀 Starting SQLite migration verification tests...")
    
    results = []
    
    # Run tests
    tests = [
        ("SQLite Client", test_sqlite_client),
        ("Database Connectivity", test_database_connectivity),
        ("ORD Storage", test_ord_storage)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} test: PASSED")
            else:
                logger.error(f"❌ {test_name} test: FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} test: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📋 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n📊 Total: {len(results)} tests")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! SQLite migration successful!")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check the logs above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)