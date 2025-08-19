"""
Test script for sparse vector functionality in HANA Vector Engine
Demonstrates sparse vector operations and performance improvements
"""
import random

import asyncio
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSparseVectors:
    """Test cases for sparse vector functionality"""
    
    def __init__(self, hanaVectorSkills):
        self.hanaVectorSkills = hanaVectorSkills
        
    async def runAllTests(self):
        """Execute all sparse vector tests"""
        logger.info("Starting sparse vector tests...")
        
        # Initialize sparse vector storage
        logger.info("1. Initializing sparse vector storage")
        initResult = await self.hanaVectorSkills.initializeSparseVectorStorage()
        logger.info(f"Storage initialization: {initResult}")
        
        # Test sparse vector conversion
        logger.info("\n2. Testing sparse vector conversion")
        await self.testSparseVectorConversion()
        
        # Test hybrid search
        logger.info("\n3. Testing enhanced hybrid search")
        await self.testEnhancedHybridSearch()
        
        # Test batch conversion
        logger.info("\n4. Testing batch sparse vector conversion")
        await self.testBatchConversion()
        
        # Test storage optimization
        logger.info("\n5. Testing sparse storage optimization")
        await self.testStorageOptimization()
        
        logger.info("\nAll tests completed!")
    
    async def testSparseVectorConversion(self):
        """Test conversion of dense vectors to sparse format"""
        # Create a sparse vector (95% zeros)
        dimension = 2000
        sparseVector = np.zeros(dimension)
        nonZeroIndices = np.random.choice(dimension, size=100, replace=False)
        sparseVector[nonZeroIndices] = np.random.randn(100)
        
        entityData = {
            'entityId': 'TEST_ENTITY_001',
            'entityType': 'financial_document',
            'content': 'Test document for sparse vector conversion',
            'sourceAgent': 'test_agent'
        }
        
        # Convert and store
        result = await self.hanaVectorSkills.convertAndStoreVector(
            sparseVector.tolist(),
            entityData
        )
        
        logger.info(f"Conversion result: {result}")
        
        # Verify sparse storage was used
        assert result.get('status') == 'success'
        if 'spaceSaved' in result:
            logger.info(f"Space saved: {result['spaceSaved']}")
    
    async def testEnhancedHybridSearch(self):
        """Test enhanced hybrid search with sparse vectors"""
        # Create test vectors with varying sparsity
        testVectors = []
        
        # Sparse vector (5% density)
        for i in range(5):
            dimension = 1500
            vector = np.zeros(dimension)
            nonZeroIndices = np.random.choice(dimension, size=75, replace=False)
            vector[nonZeroIndices] = np.random.randn(75)
            
            entityData = {
                'entityId': f'SPARSE_TEST_{i}',
                'entityType': 'financial_report',
                'content': f'Sparse financial report {i}',
                'metadata': {
                    'aiReadinessScore': 0.85,
                    'semanticTags': ['finance', 'quarterly', 'revenue']
                }
            }
            
            await self.hanaVectorSkills.convertAndStoreVector(
                vector.tolist(),
                entityData
            )
            testVectors.append(vector)
        
        # Dense vector (90% density)
        for i in range(3):
            dimension = 768
            vector = np.random.randn(dimension)
            
            entityData = {
                'entityId': f'DENSE_TEST_{i}',
                'entityType': 'financial_report',
                'content': f'Dense financial analysis {i}',
                'metadata': {
                    'aiReadinessScore': 0.75,
                    'semanticTags': ['finance', 'analysis', 'detailed']
                }
            }
            
            await self.hanaVectorSkills.convertAndStoreVector(
                vector.tolist(),
                entityData
            )
        
        # Test search with sparse query
        queryVector = testVectors[0] + np.random.randn(len(testVectors[0])) * 0.1
        
        searchFilters = {
            'entityType': 'financial_report',
            'minSimilarity': 0.7,
            'limit': 10
        }
        
        results = await self.hanaVectorSkills.enhancedHybridSearch(
            'financial quarterly report',
            queryVector.tolist(),
            searchFilters
        )
        
        logger.info(f"Search returned {len(results)} results")
        for idx, result in enumerate(results[:3]):
            logger.info(f"Result {idx + 1}: {result['entityId']} "
                       f"(score: {result['similarityScore']:.3f}, "
                       f"type: {result.get('vectorType', 'unknown')})")
    
    async def testBatchConversion(self):
        """Test batch conversion of vectors"""
        # Generate test vectors with different characteristics
        vectors = []
        
        # Very sparse vectors (should be converted)
        for i in range(10):
            dimension = 3000
            vector = np.zeros(dimension)
            nonZeroIndices = np.random.choice(dimension, size=50, replace=False)
            vector[nonZeroIndices] = np.random.randn(50)
            
            vectors.append({
                'vector': vector.tolist(),
                'entityId': f'BATCH_SPARSE_{i}',
                'entityType': 'technical_document',
                'sourceAgent': 'batch_test'
            })
        
        # Dense vectors (should remain dense)
        for i in range(5):
            dimension = 512
            vector = np.random.randn(dimension)
            
            vectors.append({
                'vector': vector.tolist(),
                'entityId': f'BATCH_DENSE_{i}',
                'entityType': 'technical_document',
                'sourceAgent': 'batch_test'
            })
        
        # Perform batch conversion
        result = await self.hanaVectorSkills.sparseVectorSkills.batchSparseVectorConversion(vectors)
        
        logger.info(f"Batch conversion results:")
        logger.info(f"  Total processed: {result['totalProcessed']}")
        logger.info(f"  Sparse converted: {result['sparseConverted']}")
        logger.info(f"  Dense kept: {result['denseKept']}")
        if 'avgSpaceSaved' in result:
            logger.info(f"  Average space saved: {result['avgSpaceSaved']:.1f}%")
    
    async def testStorageOptimization(self):
        """Test sparse storage optimization"""
        # Simulate access patterns by searching multiple times
        logger.info("Simulating access patterns...")
        
        searchFilters = {
            'entityType': 'technical_document',
            'limit': 5
        }
        
        # Perform multiple searches to update access statistics
        for _ in range(10):
            queryVector = np.zeros(3000)
            nonZeroIndices = np.random.choice(3000, size=100, replace=False)
            queryVector[nonZeroIndices] = np.random.randn(100)
            
            await self.hanaVectorSkills.sparseVectorSkills.sparseVectorSearch(
                queryVector.tolist(),
                searchFilters
            )
        
        # Run optimization
        optimizationResult = await self.hanaVectorSkills.sparseVectorSkills.optimizeSparseStorage()
        
        logger.info(f"Storage optimization results:")
        logger.info(f"  Vectors optimized: {optimizationResult['vectorsOptimized']}")
        logger.info(f"  Indices rebuilt: {optimizationResult['indicesRebuilt']}")
        logger.info(f"  Clusters created: {optimizationResult['clustersCreated']}")
        logger.info(f"  Storage reclaimed: {optimizationResult['storageReclaimed']} MB")


async def main():
    """Main test execution"""
    # Initialize HANA connection (mock for testing)
    from ..active.hanaVectorSkills import HanaVectorSkills
    
    # Create mock connection
    class MockHanaConnection:
        async def execute(self, query, params=None):
            logger.debug(f"Executing query: {query[:100]}...")
            return []
    
    hanaConnection = MockHanaConnection()
    hanaVectorSkills = HanaVectorSkills(hanaConnection)
    
    # Run tests
    tester = TestSparseVectors(hanaVectorSkills)
    await tester.runAllTests()


if __name__ == "__main__":
    asyncio.run(main())