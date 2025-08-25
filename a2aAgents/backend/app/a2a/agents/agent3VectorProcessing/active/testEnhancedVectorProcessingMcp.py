import asyncio
import os
import sys
import logging
import json
import time
import random
from datetime import datetime


from app.a2a.core.security_base import SecureA2AAgent
#!/usr/bin/env python3
"""
Test Enhanced Vector Processing Agent with MCP Integration
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
os.environ['VECTOR_PROCESSING_OUTPUT_DIR'] = '/tmp/vector_processing_data'
os.environ['VECTOR_PROMETHEUS_PORT'] = '8016'

async def test_enhanced_vector_processing_agent():
    """Test the enhanced Vector Processing Agent with MCP"""

    try:
        # Import after paths are set
        from app.a2a.agents.agent3VectorProcessing.active.enhancedVectorProcessingAgentMcp import (
            EnhancedVectorProcessingAgentMCP
        )
        print("✅ Import successful!")

        # Create agent (without HANA for testing)
        agent = EnhancedVectorProcessingAgentMCP(
            base_url=os.getenv("AGENT_MANAGER_URL"),
            hana_config=None,  # No HANA for testing
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

        # Test 1: Create test vector data
        print("\n🧪 Test 1: Creating test vector data...")

        # Generate test vectors with different characteristics
        def create_test_vector(dim=384, vector_type="normal"):
            if vector_type == "normal":
                return [random.uniform(-1, 1) for _ in range(dim)]
            elif vector_type == "zero":
                return [0.0] * dim
            elif vector_type == "extreme":
                return [random.uniform(-100, 100) for _ in range(dim)]
            elif vector_type == "corrupted":
                vec = [random.uniform(-1, 1) for _ in range(dim)]
                # Add some NaN values
                vec[0] = float('nan')
                vec[1] = float('inf')
                return vec
            else:
                return [random.uniform(-1, 1) for _ in range(dim)]

        test_vectors = []
        test_metadata = []

        # Normal vectors
        for i in range(50):
            test_vectors.append(create_test_vector("normal"))
            test_metadata.append({
                "vector_id": f"normal_{i}",
                "entity_type": "account",
                "source": "test_data",
                "category": "normal"
            })

        # Add some edge cases
        test_vectors.append(create_test_vector("zero"))
        test_metadata.append({"vector_id": "zero_1", "entity_type": "test", "category": "zero"})

        test_vectors.append(create_test_vector("extreme"))
        test_metadata.append({"vector_id": "extreme_1", "entity_type": "test", "category": "extreme"})

        test_vectors.append(create_test_vector("corrupted"))
        test_metadata.append({"vector_id": "corrupted_1", "entity_type": "test", "category": "corrupted"})

        print(f"   Created {len(test_vectors)} test vectors")

        # Test 2: Process vector data with corruption detection
        print("\n🧪 Test 2: Processing vectors with corruption detection...")
        result = await agent.process_vector_data_mcp(
            vectors=test_vectors,
            metadata=test_metadata,
            processing_mode="memory_mapped",
            enable_corruption_detection=True,
            compression_method="gzip"
        )

        if result.get("success"):
            print(f"   ✅ Processed {result['processed_count']} vectors")
            print(f"   Storage strategy: {result['storage_strategy']}")
            print(f"   Memory estimated: {result['memory_analysis']['estimated_memory_mb']:.2f} MB")

            corruption = result.get("corruption_analysis", {})
            if corruption:
                print(f"   Corruption detected: {corruption.get('corrupted', False)}")
                print(f"   Corruption confidence: {corruption.get('confidence', 0):.3f}")
                if corruption.get("issues"):
                    print(f"   Issues found: {len(corruption['issues'])}")
        else:
            print(f"   ❌ Processing failed: {result.get('error')}")
            return False

        # Test 3: Search vectors
        print("\n🧪 Test 3: Testing vector search...")

        # Create a query vector
        query_vector = create_test_vector("normal")

        search_result = await agent.search_vectors_mcp(
            query_vector=query_vector,
            top_k=10,
            filters={"entity_type": "account"},
            search_mode="hybrid",
            similarity_threshold=0.0
        )

        if search_result.get("success"):
            results = search_result["results"]
            print(f"   ✅ Found {len(results)} similar vectors")
            print(f"   Search time: {search_result['search_metadata']['search_time_ms']:.1f}ms")
            print(f"   Search strategies: {search_result['search_metadata']['search_strategies']}")

            if results:
                print(f"   Top result similarity: {results[0].get('similarity', 0):.3f}")
        else:
            print(f"   ❌ Search failed: {search_result.get('error')}")

        # Test 4: Knowledge graph operations
        print("\n🧪 Test 4: Testing knowledge graph operations...")

        # Add nodes
        node_result = await agent.manage_knowledge_graph_mcp(
            operation="add_node",
            node_data={
                "node_id": "entity_1",
                "entity_type": "account",
                "name": "Test Account",
                "properties": {"balance": 1000, "currency": "USD"}
            }
        )

        if node_result.get("success"):
            print(f"   ✅ Added node: {node_result['node_id']}")

        # Add another node
        await agent.manage_knowledge_graph_mcp(
            operation="add_node",
            node_data={
                "node_id": "entity_2",
                "entity_type": "transaction",
                "name": "Test Transaction",
                "properties": {"amount": 500, "type": "transfer"}
            }
        )

        # Add edge
        edge_result = await agent.manage_knowledge_graph_mcp(
            operation="add_edge",
            edge_data={
                "source_node": "entity_1",
                "target_node": "entity_2",
                "relationship_type": "has_transaction",
                "confidence": 0.9,
                "weight": 1.0
            }
        )

        if edge_result.get("success"):
            print(f"   ✅ Added edge: {edge_result['source_node']} -> {edge_result['target_node']}")

        # Find path
        path_result = await agent.manage_knowledge_graph_mcp(
            operation="find_path",
            path_query={
                "source": "entity_1",
                "target": "entity_2",
                "weight": "weight"
            }
        )

        if path_result.get("success"):
            print(f"   ✅ Found path: {' -> '.join(path_result['path'])}")
            print(f"   Path length: {path_result['path_length']}")

        # Centrality analysis
        centrality_result = await agent.manage_knowledge_graph_mcp(
            operation="centrality_analysis",
            centrality_types=["degree", "betweenness"]
        )

        if centrality_result.get("success"):
            print(f"   ✅ Centrality analysis completed")
            for ctype, data in centrality_result["results"].items():
                if "error" not in data:
                    print(f"     - {ctype}: {len(data.get('values', {}))} nodes analyzed")

        # Connected components
        components_result = await agent.manage_knowledge_graph_mcp(
            operation="connected_components"
        )

        if components_result.get("success"):
            print(f"   ✅ Found {components_result['total_components']} connected components")
            if components_result.get("component_analysis"):
                largest = components_result["component_analysis"][0]
                print(f"     - Largest component: {largest['size']} nodes")

        # Test 5: Memory optimization
        print("\n🧪 Test 5: Testing memory optimization...")

        # Create more vectors to test memory management
        large_vectors = [create_test_vector("normal") for _ in range(1000)]
        large_metadata = [{"vector_id": f"large_{i}", "entity_type": "test"} for i in range(1000)]

        # Process large batch
        large_result = await agent.process_vector_data_mcp(
            vectors=large_vectors,
            metadata=large_metadata,
            processing_mode="streaming",
            compression_method="quantization"
        )

        if large_result.get("success"):
            print(f"   ✅ Processed large batch: {large_result['processed_count']} vectors")
            print(f"   Storage strategy: {large_result['storage_strategy']}")

        # Test memory optimization
        memory_result = await agent.optimize_memory_usage_mcp(
            optimization_strategy="compress",
            target_memory_mb=1024,
            force_cleanup=False
        )

        if memory_result.get("success"):
            memory_analysis = memory_result["memory_analysis"]
            print(f"   ✅ Memory optimization completed")
            print(f"   Memory before: {memory_analysis['memory_before_mb']:.1f} MB")
            print(f"   Memory after: {memory_analysis['memory_after_mb']:.1f} MB")
            print(f"   Memory saved: {memory_analysis['memory_saved_mb']:.1f} MB")
            print(f"   Optimizations: {len(memory_result['optimizations_applied'])}")

        # Test 6: Stress test with corrupted data
        print("\n🧪 Test 6: Stress testing with corrupted data...")

        # Create vectors with various corruption patterns
        corrupted_vectors = []
        corrupted_metadata = []

        for i in range(100):
            if i % 10 == 0:
                # Every 10th vector is corrupted
                vec_type = secrets.choice(["zero", "extreme", "corrupted"])
                corrupted_vectors.append(create_test_vector(vector_type=vec_type))
            else:
                corrupted_vectors.append(create_test_vector("normal"))

            corrupted_metadata.append({
                "vector_id": f"stress_{i}",
                "entity_type": "stress_test",
                "corruption_expected": i % 10 == 0
            })

        stress_result = await agent.process_vector_data_mcp(
            vectors=corrupted_vectors,
            metadata=corrupted_metadata,
            processing_mode="memory_mapped",
            enable_corruption_detection=True
        )

        if stress_result.get("success"):
            corruption = stress_result.get("corruption_analysis", {})
            print(f"   ✅ Stress test completed: {stress_result['processed_count']} vectors")
            print(f"   Corruption detected: {corruption.get('corrupted', False)}")
            if corruption.get("issues"):
                print(f"   Corruption issues: {len(corruption['issues'])}")
                for issue in corruption["issues"][:3]:  # Show first 3 issues
                    print(f"     - {issue.get('check', 'unknown')}: {issue.get('passed', 'unknown')}")

        # Test 7: Access MCP resources
        print("\n🧪 Test 7: Accessing MCP resources...")

        # Get processing metrics
        metrics = await agent.get_vector_processing_metrics()
        if metrics.get("processing_metrics"):
            proc_metrics = metrics["processing_metrics"]
            print(f"   Processing Metrics:")
            print(f"     - Total vectors: {proc_metrics['total_vectors']}")
            print(f"     - Processed vectors: {proc_metrics['processed_vectors']}")
            print(f"     - Corrupted vectors: {proc_metrics['corrupted_vectors']}")
            print(f"     - Cache hits: {proc_metrics['cache_hits']}")

            memory_metrics = metrics.get("memory_metrics", {})
            print(f"   Memory Metrics:")
            print(f"     - Current usage: {memory_metrics.get('current_usage_mb', 0):.1f} MB")
            print(f"     - Usage percent: {memory_metrics.get('usage_percent', 0):.1f}%")

        # Get HANA status
        hana_status = await agent.get_hana_status()
        print(f"\n   HANA Status:")
        print(f"     - Status: {hana_status.get('status', 'unknown')}")
        print(f"     - HANA available: {hana_status.get('hana_available', False)}")
        print(f"     - Connection active: {hana_status.get('connection_active', False)}")

        # Get knowledge graph status
        kg_status = await agent.get_knowledge_graph_status()
        if kg_status.get("graph_statistics"):
            stats = kg_status["graph_statistics"]
            print(f"\n   Knowledge Graph Status:")
            print(f"     - Nodes: {stats['nodes']}")
            print(f"     - Edges: {stats['edges']}")
            print(f"     - Density: {stats['density']:.4f}")
            print(f"     - NetworkX available: {kg_status.get('networkx_available', False)}")

        # Get corruption analysis
        corruption_analysis = await agent.get_corruption_analysis()
        if corruption_analysis.get("corruption_detection"):
            detection = corruption_analysis["corruption_detection"]
            print(f"\n   Corruption Detection:")
            print(f"     - Enabled: {detection['enabled']}")
            print(f"     - Patterns: {len(detection.get('detection_patterns', []))}")

            recent_metrics = corruption_analysis.get("recent_metrics", {})
            if recent_metrics.get("total_vectors_processed", 0) > 0:
                corruption_rate = recent_metrics.get("corruption_rate", 0)
                print(f"     - Corruption rate: {corruption_rate:.1%}")

        # Test 8: Error handling
        print("\n🧪 Test 8: Testing error handling...")

        # Test with invalid vectors
        error_result = await agent.process_vector_data_mcp(
            vectors=[],  # Empty vectors
            processing_mode="invalid_mode"
        )
        print(f"   Empty vectors test: {'✅ Handled' if not error_result.get('success') else '❌ Should have failed'}")

        # Test with invalid search
        search_error = await agent.search_vectors_mcp(
            query_vector=[],  # Empty query vector
            top_k=10
        )
        print(f"   Invalid search test: {'✅ Handled' if not search_error.get('success') else '❌ Should have failed'}")

        # Test with invalid graph operation
        graph_error = await agent.manage_knowledge_graph_mcp(
            operation="invalid_operation"
        )
        print(f"   Invalid graph operation: {'✅ Handled' if not graph_error.get('success') else '❌ Should have failed'}")

        # Test 9: Performance benchmarking
        print("\n🧪 Test 9: Performance benchmarking...")

        # Benchmark vector processing
        benchmark_vectors = [create_test_vector("normal") for _ in range(500)]
        benchmark_start = time.time()

        benchmark_result = await agent.process_vector_data_mcp(
            vectors=benchmark_vectors,
            processing_mode="memory_mapped",
            compression_method="gzip"
        )

        benchmark_time = time.time() - benchmark_start

        if benchmark_result.get("success"):
            vectors_per_sec = len(benchmark_vectors) / benchmark_time
            print(f"   ✅ Performance benchmark:")
            print(f"     - Processed {len(benchmark_vectors)} vectors in {benchmark_time:.2f}s")
            print(f"     - Throughput: {vectors_per_sec:.1f} vectors/sec")
            print(f"     - Processing time per vector: {benchmark_time/len(benchmark_vectors)*1000:.2f}ms")

        # Benchmark search performance
        search_benchmark_start = time.time()

        for _ in range(10):  # 10 search queries
            query_vec = create_test_vector("normal")
            await agent.search_vectors_mcp(
                query_vector=query_vec,
                top_k=5,
                search_mode="memory"
            )

        search_benchmark_time = time.time() - search_benchmark_start
        print(f"     - Search benchmark: 10 queries in {search_benchmark_time:.2f}s")
        print(f"     - Search throughput: {10/search_benchmark_time:.1f} queries/sec")

        print("\n✅ All tests completed successfully!")

        # Final statistics
        final_metrics = await agent.get_vector_processing_metrics()
        if final_metrics.get("processing_metrics"):
            final_proc = final_metrics["processing_metrics"]
            print(f"\n📊 Final Statistics:")
            print(f"   Total vectors processed: {final_proc['total_vectors']}")
            print(f"   Corrupted vectors detected: {final_proc['corrupted_vectors']}")
            print(f"   Cache performance: {final_proc['cache_hits']} hits, {final_proc['cache_misses']} misses")

            if final_metrics.get("memory_metrics"):
                memory = final_metrics["memory_metrics"]
                print(f"   Memory usage: {memory['current_usage_mb']:.1f} MB ({memory['usage_percent']:.1f}%)")

            if final_metrics.get("storage_metrics"):
                storage = final_metrics["storage_metrics"]
                print(f"   Storage: {storage['in_memory_vectors']} in-memory, {storage['chunk_files']} chunks")

        # Cleanup
        await agent.shutdown()
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_enhanced_vector_processing_agent())
    sys.exit(0 if result else 1)
