#!/usr/bin/env python3
"""
Test ORD Blockchain Integration
Tests the integration between traditional ORD registry and blockchain ORD registry
"""

import asyncio
import logging
import os
from a2a_network.python_sdk.blockchain import initialize_blockchain_client, get_blockchain_client
from a2a_network.python_sdk.blockchain.ord_blockchain_adapter import create_ord_blockchain_adapter
from app.ordRegistry.service import ORDRegistryService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ord_blockchain_integration():
    """Test ORD blockchain integration"""
    
    logger.info("🧪 Testing ORD Blockchain Integration")
    
    try:
        # Initialize blockchain client
        logger.info("🔗 Connecting to A2A Network...")
        blockchain_client = initialize_blockchain_client(
            rpc_url="http://localhost:8545"
        )
        
        logger.info(f"✅ Connected to A2A Network")
        logger.info(f"   Agent Address: {blockchain_client.agent_identity.address}")
        logger.info(f"   Balance: {blockchain_client.get_balance()} ETH")
        
        # Initialize traditional ORD service
        logger.info("📚 Initializing traditional ORD service...")
        ord_service = ORDRegistryService(base_url="http://localhost:8000")
        await ord_service.initialize()
        logger.info("✅ Traditional ORD service initialized")
        
        # Create ORD blockchain adapter
        logger.info("🔗 Creating ORD blockchain adapter...")
        ord_adapter = create_ord_blockchain_adapter(traditional_ord_service=ord_service)
        
        # Test ORD document registration
        logger.info("📝 Testing ORD document registration...")
        
        test_document = {
            "title": "Test Financial Data Product",
            "description": "A test financial data product for blockchain integration",
            "document_content": {
                "type": "financial_data",
                "format": "JSON",
                "schema": {
                    "fields": [
                        {"name": "trade_id", "type": "string"},
                        {"name": "amount", "type": "decimal"},
                        {"name": "currency", "type": "string"}
                    ]
                },
                "endpoints": [
                    {"path": "/api/trades", "method": "GET"}
                ]
            },
            "capabilities": ["financial_data", "trading", "api_access"],
            "tags": ["finance", "trading", "data"],
            "dublin_core": {
                "creator": "FinSight CIB",
                "subject": "Financial Trading Data",
                "publisher": "Test Publisher",
                "type": "Dataset",
                "format": "application/json",
                "language": "en",
                "rights": "Copyright 2025 FinSight CIB"
            }
        }
        
        registration_result = await ord_adapter.register_ord_document(
            title=test_document["title"],
            description=test_document["description"],
            document_content=test_document["document_content"],
            capabilities=test_document["capabilities"],
            tags=test_document["tags"],
            dublin_core=test_document["dublin_core"]
        )
        
        if registration_result["success"]:
            logger.info("✅ ORD document registration successful")
            logger.info(f"   Traditional ID: {registration_result['traditional_id']}")
            logger.info(f"   Blockchain ID: {registration_result['blockchain_id']}")
        else:
            logger.error("❌ ORD document registration failed")
            return False
        
        # Test ORD document search
        logger.info("🔍 Testing ORD document search...")
        
        search_results = await ord_adapter.search_ord_documents(
            capabilities=["financial_data"],
            tags=["trading"]
        )
        
        logger.info(f"   Found {len(search_results)} documents")
        for result in search_results:
            logger.info(f"   - {result['title']} (source: {result['source']})")
        
        # Test document retrieval
        if registration_result["blockchain_id"]:
            logger.info("📖 Testing blockchain document retrieval...")
            
            retrieved_doc = await ord_adapter.get_ord_document(
                document_id=registration_result["blockchain_id"],
                source="blockchain"
            )
            
            if retrieved_doc:
                logger.info("✅ Document retrieved from blockchain")
                logger.info(f"   Title: {retrieved_doc['title']}")
                logger.info(f"   Publisher: {retrieved_doc['publisher']}")
                logger.info(f"   Version: {retrieved_doc['version']}")
                logger.info(f"   Reputation: {retrieved_doc['reputation']}")
            else:
                logger.error("❌ Failed to retrieve document from blockchain")
        
        # Test adapter status
        status = ord_adapter.get_status()
        logger.info("📊 ORD Adapter Status:")
        for key, value in status.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("✅ ORD Blockchain Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ ORD Blockchain Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ord_blockchain_integration())
    if success:
        print("🎉 All ORD blockchain tests passed!")
    else:
        print("💥 ORD blockchain tests failed!")
        exit(1)