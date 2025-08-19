#!/usr/bin/env python3
"""
Quick verification test for integration fixes
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_fixes():
    """Test that key integration fixes work"""
    print("🔧 Testing Integration Fixes...")
    
    # Test 1: SQLite schema fix
    try:
        from app.core.task_persistence import TaskManager
        print("✅ SQLite schema fix: TaskManager imports successfully")
    except Exception as e:
        print(f"❌ SQLite schema still broken: {e}")
    
    # Test 2: Distributed storage with fallback
    try:
        from app.a2a.storage.distributedStorage import DistributedStorage, LocalFileBackend
        
        # Test local backend only (should work)
        local_backend = LocalFileBackend()
        await local_backend.connect()
        
        # Test basic operations
        test_data = {"test": "fix_verification"}
        await local_backend.set("test_fix", test_data, ttl=30)
        result = await local_backend.get("test_fix")
        
        if result == test_data:
            print("✅ Local storage backend: Working correctly")
        else:
            print("❌ Local storage backend: Data mismatch")
        
        await local_backend.disconnect()
        
    except Exception as e:
        print(f"❌ Storage backend still broken: {e}")
    
    # Test 3: Request signing RSA (should work)
    try:
        from app.a2a.security.requestSigning import A2ARequestSigner
        
        signer = A2ARequestSigner()
        private_pem, public_pem = signer.generate_key_pair()
        
        key_signer = A2ARequestSigner(private_pem, public_pem)
        headers = key_signer.sign_request(
            agent_id="test1",
            target_agent_id="test2", 
            method="POST",
            path="/test"
        )
        
        is_valid, error = key_signer.verify_request(
            headers=headers,
            method="POST", 
            path="/test"
        )
        
        if is_valid:
            print("✅ RSA request signing: Working correctly")
        else:
            print(f"❌ RSA request signing: {error}")
            
    except Exception as e:
        print(f"❌ Request signing broken: {e}")
    
    # Test 4: Configuration integration
    try:
        from app.core.config import settings
        
        # Test key settings
        if hasattr(settings, 'APP_NAME') and hasattr(settings, 'A2A_REGISTRY_STORAGE'):
            print("✅ Configuration: Settings load correctly")
        else:
            print("❌ Configuration: Missing required settings")
            
    except Exception as e:
        print(f"❌ Configuration broken: {e}")
    
    print("\n🏁 Fix verification complete!")

if __name__ == "__main__":
    asyncio.run(test_fixes())