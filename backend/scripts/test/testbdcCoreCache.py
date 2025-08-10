#!/usr/bin/env python3
"""
Test script for BDC Core (Data Manager Agent) 3-tier caching implementation
"""

import asyncio
import time
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cache_manager():
    """Test the CacheManager implementation"""
    from app.services.cacheManager import CacheManager, CacheConfig
    
    # Initialize cache manager
    config = CacheConfig(
        redis_url="redis://localhost:6379",
        l1_max_size=1000,
        l1_default_ttl=60,
        l2_default_ttl=300
    )
    
    cache_manager = CacheManager(config)
    
    try:
        await cache_manager.initialize()
        logger.info("✅ Cache manager initialized successfully")
        
        # Test data
        test_data = {
            "financial_entity": "APPLE INC",
            "ticker": "AAPL",
            "data": [{"date": "2024-01-01", "price": 150.0}],
            "timestamp": datetime.now().isoformat()
        }
        
        # Test L1 cache
        logger.info("🧪 Testing L1 cache...")
        start_time = time.time()
        await cache_manager.set("test", "l1_test", test_data, ttl=60, level=1)
        set_time = time.time() - start_time
        
        start_time = time.time()
        result = await cache_manager.get("test", "l1_test", max_level=1)
        get_time = time.time() - start_time
        
        assert result == test_data, "L1 cache data mismatch"
        logger.info(f"✅ L1 cache test passed (set: {set_time:.4f}s, get: {get_time:.4f}s)")
        
        # Test L2 cache (Redis)
        logger.info("🧪 Testing L2 cache (Redis)...")
        start_time = time.time()
        await cache_manager.set("test", "l2_test", test_data, ttl=300, level=2)
        set_time = time.time() - start_time
        
        start_time = time.time()
        result = await cache_manager.get("test", "l2_test", max_level=2)
        get_time = time.time() - start_time
        
        assert result == test_data, "L2 cache data mismatch"
        logger.info(f"✅ L2 cache test passed (set: {set_time:.4f}s, get: {get_time:.4f}s)")
        
        # Test cache fallback (L1 miss, L2 hit)
        logger.info("🧪 Testing cache fallback...")
        # Clear L1 for this key
        cache_manager._l1_cache.pop("test:l2_test:hash", None)
        
        start_time = time.time()
        result = await cache_manager.get("test", "l2_test", max_level=2)
        get_time = time.time() - start_time
        
        assert result == test_data, "Cache fallback failed"
        logger.info(f"✅ Cache fallback test passed (L1 miss -> L2 hit: {get_time:.4f}s)")
        
        # Test cache statistics
        logger.info("📊 Getting cache statistics...")
        stats = await cache_manager.get_stats()
        logger.info(f"Cache stats: {json.dumps(stats, indent=2)}")
        
        # Test cache invalidation
        logger.info("🧪 Testing cache invalidation...")
        await cache_manager.invalidate("test", "l1_test")
        result = await cache_manager.get("test", "l1_test", max_level=1)
        assert result is None, "Cache invalidation failed"
        logger.info("✅ Cache invalidation test passed")
        
        logger.info("🎉 All cache tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Cache test failed: {e}")
        raise
    finally:
        await cache_manager.close()

async def test_performance():
    """Test cache performance vs database operations"""
    from app.services.cacheManager import CacheManager, CacheConfig
    
    config = CacheConfig(redis_url="redis://localhost:6379")
    cache_manager = CacheManager(config)
    
    try:
        await cache_manager.initialize()
        
        # Simulate database query result
        large_dataset = {
            "query": "SELECT * FROM financial_data WHERE ticker = 'AAPL'",
            "data": [
                {"date": f"2024-01-{i:02d}", "price": 150.0 + i, "volume": 1000000 + i*1000}
                for i in range(1, 1001)  # 1000 records
            ],
            "metadata": {
                "total_records": 1000,
                "query_time": "0.250s",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Test cache performance
        logger.info("⚡ Performance test: caching large dataset...")
        
        # First write (cache miss)
        start_time = time.time()
        await cache_manager.set("perf", "large_query", large_dataset, ttl=600, level=2)
        cache_write_time = time.time() - start_time
        
        # First read (should be cached)
        start_time = time.time()
        result = await cache_manager.get("perf", "large_query", max_level=2)
        cache_read_time = time.time() - start_time
        
        # Simulate multiple reads
        total_read_time = 0
        num_reads = 10
        for i in range(num_reads):
            start_time = time.time()
            result = await cache_manager.get("perf", "large_query", max_level=1)  # L1 cache hit
            total_read_time += time.time() - start_time
        
        avg_read_time = total_read_time / num_reads
        
        logger.info(f"📈 Performance Results:")
        logger.info(f"   Cache write: {cache_write_time:.4f}s")
        logger.info(f"   Cache read (L2): {cache_read_time:.4f}s")
        logger.info(f"   Cache read (L1 avg): {avg_read_time:.6f}s")
        logger.info(f"   Speed improvement: {cache_read_time/avg_read_time:.1f}x faster (L1 vs L2)")
        
        # Estimate database vs cache savings
        simulated_db_time = 0.250  # 250ms typical DB query
        cache_savings = simulated_db_time - avg_read_time
        savings_percent = (cache_savings / simulated_db_time) * 100
        
        logger.info(f"   Estimated DB query time: {simulated_db_time:.3f}s")
        logger.info(f"   Cache savings: {cache_savings:.3f}s ({savings_percent:.1f}% faster)")
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        raise
    finally:
        await cache_manager.close()

async def main():
    """Run all tests"""
    logger.info("🚀 Starting BDC Core cache tests...")
    
    try:
        await test_cache_manager()
        await test_performance()
        logger.info("✅ All tests completed successfully!")
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))