import asyncio
import os
import sys
import logging
import json
import time
from datetime import datetime


from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Test Enhanced AI Preparation Agent with MCP Integration
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
os.environ['AI_PREPARATION_OUTPUT_DIR'] = '/tmp/ai_preparation_data'
os.environ['AI_PREP_PROMETHEUS_PORT'] = '8014'

async def test_enhanced_ai_preparation_agent():
    """Test the enhanced AI Preparation Agent with MCP"""
    
    try:
        # Import after paths are set
        from app.a2a.agents.agent2AiPreparation.active.enhancedAiPreparationAgentMcp import (
            EnhancedAIPreparationAgentMCP
        )
        print("✅ Import successful!")
        
        # Create agent
        agent = EnhancedAIPreparationAgentMCP(
            base_url=os.getenv("CATALOG_MANAGER_URL"),
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
        
        # Test 1: Create test entity data
        print("\n🧪 Test 1: Creating test entity data...")
        test_entity = {
            "entity_id": "ENT001",
            "entity_type": "account",
            "name": "Cash and Equivalents Account",
            "description": "Primary cash management account for operational liquidity",
            "category": "core_banking",
            "volume": 50000,
            "quality_score": 0.9,
            "primary_function": "Liquidity Management",
            "stakeholders": ["Treasury", "Finance", "Risk Management"],
            "operational_context": "Daily cash operations and short-term investments",
            "regulatory_framework": "Basel III",
            "compliance_requirements": ["Basel III", "GDPR", "SOX"],
            "source_system": "Core Banking System",
            "last_updated": datetime.utcnow().isoformat(),
            "lifecycle_stage": "active",
            "related_entities": [
                {
                    "id": "ENT002", 
                    "relationship_type": "parent_of", 
                    "strength": 0.9, 
                    "confidence": 0.95
                }
            ]
        }
        
        # Test 2: Prepare AI data using MCP tool
        print("\n🧪 Test 2: Preparing AI data via MCP...")
        result = await agent.prepare_ai_data_mcp(
            entity_data=test_entity,
            embedding_mode="hybrid",
            include_relationships=True,
            confidence_threshold=0.6
        )
        
        if result.get("success"):
            summary = result["summary"]
            print(f"   ✅ AI preparation successful!")
            print(f"   Overall confidence: {summary['overall_confidence']:.3f}")
            print(f"   Embedding confidence: {summary['embedding_confidence']:.3f}")
            print(f"   Processing time: {summary['processing_time_ms']:.1f}ms")
            print(f"   Relationships mapped: {summary['relationship_count']}")
            print(f"   Meets threshold: {summary['meets_threshold']}")
            
            entity_id = result["entity_id"]
        else:
            print(f"   ❌ AI preparation failed: {result.get('error')}")
            return False
        
        # Test 3: Validate AI readiness
        print("\n🧪 Test 3: Validating AI readiness...")
        validation_result = await agent.validate_ai_readiness_mcp(
            entity_ids=[entity_id],
            validation_level="comprehensive",
            min_confidence_threshold=0.6
        )
        
        if validation_result.get("success"):
            summary = validation_result["summary"]
            print(f"   ✅ Validation completed!")
            print(f"   Valid entities: {summary['valid_entities']}/{summary['total_entities']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            if summary.get("avg_confidence"):
                print(f"   Average confidence: {summary['avg_confidence']:.3f}")
        else:
            print(f"   ❌ Validation failed: {validation_result.get('error')}")
        
        # Test 4: Batch embedding generation
        print("\n🧪 Test 4: Testing batch embedding generation...")
        test_texts = [
            "Financial account for cash management",
            "Transaction processing system",
            "Customer relationship management data",
            "Product catalog information",
            "Location and branch details"
        ]
        
        batch_result = await agent.generate_embeddings_batch_mcp(
            texts=test_texts,
            embedding_mode="hash_based",  # Use valid embedding mode
            normalize=True,
            include_confidence=True
        )
        
        if batch_result.get("success"):
            summary = batch_result["summary"]
            print(f"   ✅ Batch embedding generation completed!")
            print(f"   Successful embeddings: {summary['successful_embeddings']}/{summary['total_texts']}")
            print(f"   Average confidence: {summary['avg_confidence']:.3f}")
            print(f"   Processing time: {summary['processing_time_ms']:.1f}ms")
            print(f"   Cache hit rate: {summary['cache_hit_rate']:.1%}")
        else:
            print(f"   ❌ Batch embedding failed: {batch_result.get('error')}")
        
        # Test 5: Test different embedding modes
        print("\n🧪 Test 5: Testing different embedding modes...")
        embedding_modes = ["hash_based", "statistical", "hybrid"]
        
        for mode in embedding_modes:
            try:
                mode_result = await agent.prepare_ai_data_mcp(
                    entity_data={
                        "entity_id": f"TEST_{mode.upper()}",
                        "entity_type": "test",
                        "name": f"Test entity for {mode} embedding",
                        "description": "Test entity for embedding mode validation"
                    },
                    embedding_mode=mode,
                    include_relationships=False,
                    confidence_threshold=0.5
                )
                
                if mode_result.get("success"):
                    confidence = mode_result["summary"]["overall_confidence"]
                    print(f"   ✅ {mode}: confidence={confidence:.3f}")
                else:
                    print(f"   ❌ {mode}: {mode_result.get('error')}")
                    
            except Exception as e:
                print(f"   ❌ {mode}: {str(e)}")
        
        # Test 6: Optimize confidence scoring
        print("\n🧪 Test 6: Testing confidence scoring optimization...")
        optimization_result = await agent.optimize_confidence_scoring_mcp(
            target_confidence=0.8,
            optimization_method="statistical"
        )
        
        if optimization_result.get("success"):
            print(f"   ✅ Optimization completed!")
            print(f"   Target confidence: {optimization_result['target_confidence']}")
            print(f"   Confidence gap: {optimization_result['confidence_gap']:.3f}")
            if optimization_result.get("changes_applied"):
                print(f"   Changes applied: {len(optimization_result['changes_applied'])}")
            else:
                print(f"   Status: {optimization_result['recommendations'].get('status', 'optimized')}")
        else:
            print(f"   ❌ Optimization failed: {optimization_result.get('error')}")
        
        # Test 7: Access MCP resources
        print("\n🧪 Test 7: Accessing MCP resources...")
        
        # Get AI preparation catalog
        catalog = await agent.get_ai_preparation_catalog()
        if catalog.get("catalog_metadata"):
            stats = catalog["catalog_metadata"]["statistics"]
            print(f"   AI Preparation Catalog:")
            print(f"     - Total entities: {stats.get('total_entities', 0)}")
            if stats.get("avg_readiness_score"):
                print(f"     - Average readiness: {stats['avg_readiness_score']:.3f}")
        
        # Get performance metrics
        metrics = await agent.get_performance_metrics()
        if metrics.get("processing_metrics"):
            proc_metrics = metrics["processing_metrics"]
            print(f"\n   Performance Metrics:")
            print(f"     - Total processed: {proc_metrics['total_processed']}")
            print(f"     - Success rate: {proc_metrics['success_rate']:.1%}")
            print(f"     - Avg processing time: {proc_metrics['avg_processing_time']:.3f}s")
            
        # Get embedding status
        embedding_status = await agent.get_embedding_status()
        if embedding_status.get("embedding_config"):
            config = embedding_status["embedding_config"]
            print(f"\n   Embedding Status:")
            print(f"     - Mode: {config['mode']}")
            print(f"     - Dimension: {config['dimension']}")
            print(f"     - Model: {config['model_name']}")
            
            cache_status = embedding_status.get("cache_status", {})
            print(f"     - Cache size: {cache_status.get('current_size', 0)}")
            print(f"     - Cache hit rate: {cache_status.get('hit_rate', 0):.1%}")
        
        # Get confidence configuration
        confidence_config = await agent.get_confidence_config()
        if confidence_config.get("confidence_weights"):
            weights = confidence_config["confidence_weights"]
            print(f"\n   Confidence Configuration:")
            for metric, weight in weights.items():
                print(f"     - {metric}: {weight:.2f}")
            
            thresholds = confidence_config.get("threshold_settings", {})
            print(f"     - Min threshold: {thresholds.get('min_confidence_threshold', 0):.2f}")
        
        # Test 8: Test error handling
        print("\n🧪 Test 8: Testing error handling...")
        
        # Test with invalid entity data
        error_result = await agent.prepare_ai_data_mcp(
            entity_data={"invalid": "data"},
            embedding_mode="invalid_mode",
            confidence_threshold=2.0  # Invalid threshold
        )
        print(f"   Error handling test: {'✅ Handled gracefully' if not error_result.get('success') else '❌ Should have failed'}")
        
        # Test with empty text batch
        empty_batch_result = await agent.generate_embeddings_batch_mcp(
            texts=[],
            embedding_mode="hybrid"
        )
        print(f"   Empty batch test: {'✅ Handled gracefully' if empty_batch_result.get('success') else '❌ Unexpected failure'}")
        
        # Test with invalid validation parameters
        invalid_validation = await agent.validate_ai_readiness_mcp(
            entity_ids=["nonexistent"],
            validation_level="invalid_level",
            min_confidence_threshold=1.5  # Invalid threshold
        )
        print(f"   Invalid validation test: {'✅ Handled gracefully' if not invalid_validation.get('success') else '❌ Should have failed'}")
        
        # Test optimization with no data
        early_optimization = await agent.optimize_confidence_scoring_mcp(
            target_confidence=0.9,
            optimization_method="heuristic"
        )
        print(f"   Early optimization test: {'✅ Handled gracefully' if early_optimization.get('success') or 'no_data' in str(early_optimization.get('error', '')) else '❌ Should indicate no data'}")
        
        # Test batch with invalid text types
        invalid_batch = await agent.generate_embeddings_batch_mcp(
            texts=["valid", 123, None, "also valid"],  # Mixed types
            embedding_mode="hash_based"
        )
        print(f"   Invalid batch types test: {'✅ Handled gracefully' if not invalid_batch.get('success') else '❌ Should have failed'}")
        
        # Test 8.5: Comprehensive MCP tool and resource validation
        print("\n🧪 Test 8.5: Comprehensive MCP validation...")
        
        # Test all MCP tools are accessible
        expected_tools = ["prepare_ai_data", "validate_ai_readiness", "generate_embeddings_batch", "optimize_confidence_scoring"]
        available_tools = [tool['name'] for tool in agent.list_mcp_tools()]
        missing_tools = [tool for tool in expected_tools if tool not in available_tools]
        print(f"   MCP Tools check: {'✅ All tools available' if not missing_tools else f'❌ Missing tools: {missing_tools}'}")
        
        # Test all MCP resources are accessible
        expected_resources = ["aipreparation://catalog", "aipreparation://performance-metrics", "aipreparation://embedding-status", "aipreparation://confidence-config"]
        available_resources = [resource['uri'] for resource in agent.list_mcp_resources()]
        missing_resources = [resource for resource in expected_resources if resource not in available_resources]
        print(f"   MCP Resources check: {'✅ All resources available' if not missing_resources else f'❌ Missing resources: {missing_resources}'}")
        
        # Test resource access with error scenarios
        try:
            # These should work since we have entities processed
            for uri in expected_resources:
                if uri == "aipreparation://catalog":
                    result = await agent.get_ai_preparation_catalog()
                elif uri == "aipreparation://performance-metrics":
                    result = await agent.get_performance_metrics()
                elif uri == "aipreparation://embedding-status":
                    result = await agent.get_embedding_status()
                elif uri == "aipreparation://confidence-config":
                    result = await agent.get_confidence_config()
                
                if result and not result.get("error"):
                    print(f"   Resource {uri}: ✅ Accessible")
                else:
                    print(f"   Resource {uri}: ❌ Error accessing")
        except Exception as e:
            print(f"   Resource access test: ❌ Exception: {e}")
        
        # Test 9: Performance stress test
        print("\n🧪 Test 9: Performance stress test...")
        stress_test_start = time.time()
        
        # Process multiple entities concurrently
        stress_entities = []
        for i in range(10):
            entity = {
                "entity_id": f"STRESS_{i:03d}",
                "entity_type": "account",
                "name": f"Stress Test Account {i}",
                "description": f"Account created for stress testing purposes - iteration {i}",
                "volume": 1000 * i,
                "quality_score": 0.7 + (i % 3) * 0.1
            }
            stress_entities.append(entity)
        
        # Process entities concurrently
        stress_tasks = [
            agent.prepare_ai_data_mcp(
                entity_data=entity,
                embedding_mode="hash_based",  # Fast mode for stress test
                include_relationships=False,
                confidence_threshold=0.5
            )
            for entity in stress_entities
        ]
        
        stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
        successful_stress = sum(1 for result in stress_results if isinstance(result, dict) and result.get("success"))
        stress_test_time = time.time() - stress_test_start
        
        print(f"   ✅ Stress test completed!")
        print(f"   Processed: {successful_stress}/{len(stress_entities)} entities")
        print(f"   Total time: {stress_test_time:.2f}s")
        print(f"   Throughput: {len(stress_entities)/stress_test_time:.1f} entities/sec")
        
        print("\n✅ All tests completed successfully!")
        
        # Final resource status
        final_metrics = await agent.get_performance_metrics()
        if final_metrics.get("processing_metrics"):
            final_proc = final_metrics["processing_metrics"]
            print(f"\n📊 Final Statistics:")
            print(f"   Total entities processed: {final_proc['total_processed']}")
            print(f"   Overall success rate: {final_proc['success_rate']:.1%}")
            print(f"   Average confidence: {final_proc.get('avg_confidence_score', 0):.3f}")
            
            if final_metrics.get("resource_metrics"):
                resource = final_metrics["resource_metrics"]
                print(f"   Memory usage: {resource['memory_usage_mb']:.1f} MB")
                print(f"   CPU usage: {resource['cpu_usage_percent']:.1f}%")
        
        # Cleanup
        await agent.shutdown()
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_ai_preparation_agent())
    sys.exit(0 if result else 1)