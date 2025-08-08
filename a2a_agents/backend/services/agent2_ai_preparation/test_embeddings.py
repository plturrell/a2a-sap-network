#!/usr/bin/env python3
"""
Test script for Agent 2 embedding functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent import AIPreparationAgent
from a2a_common import A2AMessage, MessageRole


async def test_embeddings():
    """Test the embedding generation functionality"""
    print("🧪 Testing Agent 2 Embedding Functionality\n")
    
    # Create agent instance
    agent = AIPreparationAgent(
        base_url="http://localhost:8003",
        agent_manager_url="http://localhost:8007",
        downstream_agent_url="http://localhost:8004"
    )
    
    # Initialize the agent
    print("📦 Initializing agent...")
    await agent.initialize()
    
    # Test data
    test_entities = [
        {
            "id": "ACC001",
            "name": "Operating Account",
            "currency": "USD",
            "balance": 150000.00,
            "type": "checking"
        },
        {
            "id": "ACC002", 
            "name": "Reserve Account",
            "currency": "USD",
            "balance": 500000.00,
            "type": "savings"
        },
        {
            "id": "ACC003",
            "name": "Euro Trading Account",
            "currency": "EUR",
            "balance": 250000.00,
            "type": "trading"
        }
    ]
    
    # Test semantic enrichment
    print("\n🎯 Testing semantic enrichment...")
    enriched = await agent.enrich_semantically(test_entities, "account")
    print(f"✅ Enriched {len(enriched)} entities")
    print(f"   Sample enrichment: {enriched[0]['semantic_enrichment']['semantic_description']}")
    
    # Test embedding generation
    print("\n🧮 Testing embedding generation...")
    with_embeddings = await agent.generate_embeddings(enriched)
    print(f"✅ Generated embeddings for {len(with_embeddings)} entities")
    
    for i, entity in enumerate(with_embeddings):
        embedding_info = entity['embedding']
        print(f"\n   Entity {i+1} ({entity['name']}):")
        print(f"   - Model: {embedding_info['model']}")
        print(f"   - Dimension: {embedding_info['dimension']}")
        print(f"   - Normalized: {embedding_info['normalized']}")
        print(f"   - Vector sample: [{embedding_info['vector'][0]:.4f}, {embedding_info['vector'][1]:.4f}, ...]")
    
    # Test relationship extraction
    print("\n🔗 Testing relationship extraction...")
    relationships = await agent.extract_relationships(with_embeddings, "account")
    print(f"✅ Extracted {len(relationships)} relationships")
    
    for rel in relationships:
        print(f"   - {rel['source_id']} ↔️ {rel['target_id']}: {rel['relationship_type']}")
    
    # Test processing stats
    print("\n📊 Processing Statistics:")
    print(f"   - Entities enriched: {agent.processing_stats['entities_enriched']}")
    print(f"   - Embeddings generated: {agent.processing_stats['embeddings_generated']}")
    print(f"   - Relationships extracted: {agent.processing_stats['relationships_extracted']}")
    
    print("\n✨ All tests completed successfully!")
    
    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(test_embeddings())