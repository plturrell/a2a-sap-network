#!/usr/bin/env python3
"""
Test AI Preparation Agent with Grok Integration
"""

import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../backend'))

from src.agent import AIPreparationAgent


async def test_grok_integration():
    """Test AI Preparation Agent with Grok client"""
    print("🧪 Testing AI Preparation Agent with Grok Integration")
    print("=" * 60)
    
    # Check if Grok API key is set
    if not os.getenv('XAI_API_KEY'):
        print("⚠️  WARNING: XAI_API_KEY not set. Grok integration will use fallback mode.")
    else:
        print("✅ Grok API key detected")
    
    # Initialize agent
    agent = AIPreparationAgent(
        base_url="http://localhost:8082",
        agent_manager_url="http://localhost:8080",
        downstream_agent_url="http://localhost:8083"
    )
    
    # Initialize resources
    await agent.initialize()
    
    # Check Grok client status
    if agent.grok_client:
        print("✅ Grok client initialized successfully")
        health = agent.grok_client.health_check()
        print(f"🏥 Grok health check: {health}")
    else:
        print("⚠️  Grok client not available, using rule-based enrichment")
    
    # Test data
    test_entities = [
        {
            "id": "ACC001",
            "name": "Corporate Reserve Account",
            "type": "reserve",
            "currency": "USD",
            "balance": 5000000,
            "description": "Primary corporate reserve account for regulatory compliance"
        },
        {
            "id": "ACC002", 
            "name": "Trading Operations Account",
            "type": "trading",
            "currency": "EUR",
            "balance": 2500000,
            "description": "Main trading account for European markets"
        }
    ]
    
    print("\n📊 Testing Semantic Enrichment")
    print("-" * 40)
    
    # Test semantic enrichment
    enriched = await agent.enrich_semantically(test_entities, "account")
    
    for entity in enriched:
        print(f"\n🔍 Entity: {entity['name']}")
        if "semantic_enrichment" in entity:
            enrichment = entity["semantic_enrichment"]
            print(f"   📝 Description: {enrichment['semantic_description'][:100]}...")
            print(f"   🏢 Primary Function: {enrichment['business_context']['primary_function']}")
            print(f"   ⚠️  Risk Indicators: {enrichment['business_context']['risk_indicators']}")
            print(f"   📋 Compliance Flags: {enrichment['business_context']['compliance_flags']}")
            print(f"   🔧 Enrichment Method: {enrichment['contextual_metadata']['enrichment_method']}")
    
    print("\n🧮 Testing Embedding Generation")
    print("-" * 40)
    
    # Test embedding generation
    with_embeddings = await agent.generate_embeddings(enriched)
    
    for entity in with_embeddings:
        if "embedding" in entity:
            emb = entity["embedding"]
            print(f"\n🔢 Entity: {entity['name']}")
            print(f"   📏 Dimension: {emb['dimension']}")
            print(f"   🤖 Model: {emb['model']}")
            print(f"   📊 Vector sample: {emb['vector'][:5]}...")
    
    print("\n📈 Processing Stats")
    print("-" * 40)
    print(f"Entities Enriched: {agent.processing_stats['entities_enriched']}")
    print(f"Embeddings Generated: {agent.processing_stats['embeddings_generated']}")
    
    # Cleanup
    await agent.shutdown()
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_grok_integration())